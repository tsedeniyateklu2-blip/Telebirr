"""
Telebirr Verify  -  DEMO prototype (NOT connected to real Telebirr).

A payment-verification assistant for small merchants: check a transaction
reference (or a receipt screenshot) against authorised records, flag suspicious
patterns with an experimental risk model, and review everything in a dashboard.

Sections
  1. Configuration & database        4. Risk model (synthetic data)
  2. Verification engine             5. Demo data & tests
  3. Receipt OCR & parsing           6. Streamlit app (UI)

Run the app   :  streamlit run app.py
Run the tests :  python app.py test
"""
from __future__ import annotations

import difflib
import html
import io
import json
import re
import sqlite3
import sys
import tempfile
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ==================================================================
# 1. CONFIGURATION & DATABASE
# ==================================================================
APP_NAME = "Telebirr Verify"
APP_VERSION = "1.0"
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = str(BASE_DIR / "telebirr_demo.db")
MODEL_PATH = str(BASE_DIR / "risk_model.joblib")
MODEL_META_PATH = str(BASE_DIR / "risk_model_meta.json")

OCR_MIN_CONF = 60.0          # below this a receipt goes to manual review
RISK_LOW, RISK_HIGH = 0.30, 0.70
SIM_TAG = "(sim)"            # marks simulated rows so they never affect real checks
BATCH_LIMIT = 500


@contextmanager
def db_conn():
    """Open a connection, commit on success, always close."""
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    try:
        yield con
        con.commit()
    finally:
        con.close()


def get_db():
    """Backwards-compatible raw connection (caller must close it)."""
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def query_df(sql: str, params: tuple = ()) -> pd.DataFrame:
    with db_conn() as con:
        return pd.read_sql(sql, con, params=params)


SEED_MERCHANTS = [
    ("SHOP-001", "Demo Shop One"),
    ("SHOP-002", "Demo Shop Two"),
    ("SHOP-003", "Demo Shop Three"),
]
SEED_TRANSACTIONS = [
    ("DEMO-TB-1001", 1500, "SHOP-001", "COMPLETED", "2026-11-05"),
    ("DEMO-TB-1002", 250, "SHOP-001", "PENDING", "2026-11-05"),
    ("DEMO-TB-1003", 800, "SHOP-002", "COMPLETED", "2026-11-05"),
    ("DEMO-TB-1004", 420, "SHOP-001", "FAILED", "2026-11-05"),
    ("DEMO-TB-1005", 2750, "SHOP-003", "COMPLETED", "2026-11-06"),
    ("DEMO-TB-1006", 95, "SHOP-002", "COMPLETED", "2026-11-06"),
    ("DEMO-TB-1007", 1200, "SHOP-003", "COMPLETED", "2026-11-07"),
    ("DEMO-TB-1008", 330, "SHOP-001", "COMPLETED", "2026-11-08"),
    ("DEMO-TB-1009", 5000, "SHOP-002", "PENDING", "2026-11-09"),
    ("DEMO-TB-1010", 640, "SHOP-003", "COMPLETED", "2026-11-10"),
]


def init_db(reset: bool = False) -> None:
    with db_conn() as db:
        if reset:
            db.executescript(
                "DROP TABLE IF EXISTS transactions;"
                "DROP TABLE IF EXISTS verification_history;"
                "DROP TABLE IF EXISTS merchants;"
            )
        db.executescript("""
        CREATE TABLE IF NOT EXISTS merchants(
            merchant_id TEXT PRIMARY KEY, name TEXT);
        CREATE TABLE IF NOT EXISTS transactions(
            reference TEXT PRIMARY KEY, amount REAL,
            merchant_id TEXT, status TEXT, transaction_date TEXT);
        CREATE TABLE IF NOT EXISTS verification_history(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            reference TEXT, expected_amount REAL, merchant_id TEXT,
            method TEXT, result TEXT, reason TEXT,
            risk TEXT, ts DATETIME DEFAULT CURRENT_TIMESTAMP);
        CREATE INDEX IF NOT EXISTS idx_hist_ref ON verification_history(reference);
        CREATE INDEX IF NOT EXISTS idx_hist_ts  ON verification_history(ts);
        """)
        if db.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] == 0:
            db.executemany("INSERT OR IGNORE INTO merchants VALUES(?,?)", SEED_MERCHANTS)
            db.executemany("INSERT INTO transactions VALUES(?,?,?,?,?)", SEED_TRANSACTIONS)


def merchant_options() -> dict:
    with db_conn() as db:
        return dict(db.execute(
            "SELECT merchant_id, COALESCE(name, merchant_id) FROM merchants ORDER BY merchant_id"))


# ==================================================================
# 2. VERIFICATION ENGINE
# ==================================================================
REASONS = {
    "VERIFIED": "Completed, correct amount, correct merchant, first use.",
    "NOT FOUND": "No such reference in the authorized records.",
    "PENDING": "Payment is not completed yet. Do not release goods.",
    "FAILED": "The payment failed.",
    "WRONG RECIPIENT": "Payment went to a different merchant.",
    "AMOUNT MISMATCH": "Recorded amount differs from the expected amount.",
    "DUPLICATE WARNING": "Already verified before. May be a retry; review manually.",
    "MANUAL REVIEW": "Could not read the receipt reliably. Check manually.",
}

ACTIONS = {
    "VERIFIED": "Safe to release the goods or service.",
    "NOT FOUND": "Do not release goods. Ask the customer to re-check the reference.",
    "PENDING": "Wait until the payment shows as completed.",
    "FAILED": "Do not release goods. Ask the customer to pay again.",
    "WRONG RECIPIENT": "Do not release goods. The money did not reach this merchant.",
    "AMOUNT MISMATCH": "Do not release goods until the amount is explained.",
    "DUPLICATE WARNING": "Check whether this is a retry or a reused receipt.",
    "MANUAL REVIEW": "A person should compare the receipt with the records.",
}

# (icon, css-kind, chart colour)
RESULT_STYLE = {
    "VERIFIED": ("✅", "ok", "#14A44D"),
    "DUPLICATE WARNING": ("🔁", "warn", "#F59E0B"),
    "PENDING": ("⏳", "warn", "#FBBF24"),
    "MANUAL REVIEW": ("🔍", "warn", "#8B5CF6"),
    "AMOUNT MISMATCH": ("⚠️", "bad", "#EF4444"),
    "WRONG RECIPIENT": ("🚫", "bad", "#DC2626"),
    "FAILED": ("❌", "bad", "#B91C1C"),
    "NOT FOUND": ("❓", "bad", "#6B7280"),
}
RESULT_COLORS = {k: v[2] for k, v in RESULT_STYLE.items()}
RISK_COLORS = {"Low risk": "#14A44D", "Review recommended": "#F59E0B",
               "High risk": "#DC2626", "Not scored": "#9CA3AF"}


def normalize_ref(ref) -> str:
    """Upper-case and strip spaces/odd dashes so ' demo–tb-1001 ' still matches."""
    ref = (ref or "").strip().upper()
    ref = ref.replace("–", "-").replace("—", "-").replace("\u00a0", " ")
    return re.sub(r"\s+", "", ref)


def verify(ref, amount, merchant, method="ID", ocr_conf=100.0,
           tolerance=0.0, record=True) -> dict:
    """Check one claim, optionally save it to history, return a result dict."""
    ref = normalize_ref(ref)
    merchant = (merchant or "").strip().upper()
    amount = float(amount or 0.0)
    with db_conn() as db:
        row = db.execute(
            "SELECT amount, merchant_id, status FROM transactions WHERE reference=?",
            (ref,)).fetchone()
        repeats = db.execute(
            "SELECT COUNT(*) FROM verification_history WHERE reference=? AND method NOT LIKE ?",
            (ref, f"%{SIM_TAG}")).fetchone()[0]
        used = db.execute(
            "SELECT 1 FROM verification_history WHERE reference=? AND result='VERIFIED' "
            "AND method NOT LIKE ?", (ref, f"%{SIM_TAG}")).fetchone()
        all_refs = [r[0] for r in db.execute("SELECT reference FROM transactions")]

        rec_amount = rec_merchant = status = suggestion = None
        if ocr_conf < OCR_MIN_CONF:
            result = "MANUAL REVIEW"
            if row:
                rec_amount, rec_merchant, status = row
        elif row is None:
            result = "NOT FOUND"
            close = difflib.get_close_matches(ref, all_refs, n=1, cutoff=0.75)
            suggestion = close[0] if close else None
        else:
            rec_amount, rec_merchant, status = row
            if status == "PENDING":
                result = "PENDING"
            elif status != "COMPLETED":
                result = "FAILED"
            elif rec_merchant.upper() != merchant:
                result = "WRONG RECIPIENT"
            elif abs(rec_amount - amount) > tolerance + 1e-9:
                result = "AMOUNT MISMATCH"
            else:
                result = "DUPLICATE WARNING" if used else "VERIFIED"

        features = {
            "amount_diff": abs(amount - rec_amount) if rec_amount is not None else 0.0,
            "repeat_count": repeats,
            "merchant_mismatch": int(rec_merchant is not None
                                     and rec_merchant.upper() != merchant),
            "ocr_conf": float(ocr_conf),
        }
        risk, score = risk_score(features)
        if record:
            db.execute(
                "INSERT INTO verification_history"
                "(reference, expected_amount, merchant_id, method, result, reason, risk) "
                "VALUES(?,?,?,?,?,?,?)",
                (ref, amount, merchant, method, result, REASONS[result], risk))
    return {"reference": ref, "result": result, "reason": REASONS[result],
            "action": ACTIONS[result], "risk": risk, "risk_score": score,
            "recorded_amount": rec_amount, "recorded_merchant": rec_merchant,
            "status": status, "prior_checks": repeats, "suggestion": suggestion,
            "features": features}


def verify_batch(df: pd.DataFrame, tolerance=0.0) -> pd.DataFrame:
    """Verify many rows (columns: reference, amount, merchant). Returns a table."""
    out = []
    for _, r in df.head(BATCH_LIMIT).iterrows():
        try:
            amount = float(r["amount"])
        except (TypeError, ValueError):
            amount = 0.0
        res = verify(r["reference"], amount, str(r["merchant"]),
                     method="BATCH", tolerance=tolerance)
        out.append({"reference": res["reference"], "expected_amount": amount,
                    "merchant": str(r["merchant"]).upper(), "result": res["result"],
                    "risk": res["risk"], "action": res["action"]})
    return pd.DataFrame(out)


# ==================================================================
# 3. RECEIPT OCR & PARSING
# ==================================================================
def preprocess(img):
    from PIL import ImageOps
    img = ImageOps.grayscale(img)
    img = ImageOps.autocontrast(img)
    return img.resize((img.width * 2, img.height * 2))


def ocr_image(img):
    """Returns (text, average confidence 0-100)."""
    import pytesseract
    img = preprocess(img)
    text = pytesseract.image_to_string(img)
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    confs = [float(c) for c, w in zip(data["conf"], data["text"])
             if w.strip() and float(c) >= 0]
    return text, (sum(confs) / len(confs) if confs else 0.0)


def parse_receipt(text):
    """Pull (reference, amount) out of OCR text. None for anything not found."""
    ref = re.search(r"\b(DEMO-TB-\d+|(?=[A-Z0-9]*\d)[A-Z0-9]{10})\b", text.upper())
    amt = (re.search(r"(?<![A-Za-z0-9])([\d][\d,]*\.?\d*)\s*(?:ETB|BIRR)", text, re.I)
           or re.search(r"(?:ETB|BIRR)\s*([\d][\d,]*\.?\d*)", text, re.I)
           or re.search(r"amount\D{0,10}([\d][\d,]*\.?\d*)", text, re.I))
    amount = None
    if amt:
        try:
            amount = float(amt.group(1).replace(",", "").rstrip("."))
        except ValueError:
            pass
    return (ref.group(1) if ref else None), amount


# ==================================================================
# 4. RISK MODEL  (trained on SYNTHETIC data - experimental)
# ==================================================================
FEATURES = ["amount_diff", "repeat_count", "merchant_mismatch", "ocr_conf"]
FEATURE_LABELS = {"amount_diff": "Amount difference", "repeat_count": "Prior checks",
                  "merchant_mismatch": "Merchant mismatch", "ocr_conf": "OCR confidence"}


def make_synthetic(n=3000, seed=42):
    """Simulated scenarios. Label 1 = 'deserves review' in the simulation."""
    rng = np.random.default_rng(seed)
    names = ["normal", "amount_edit", "reused", "wrong_merchant", "blurry_legit"]
    kinds = rng.choice(names, size=n, p=[0.60, 0.12, 0.10, 0.08, 0.10])
    rows = []
    for k in kinds:
        diff, rep, mm, conf = 0.0, 0, 0, rng.normal(92, 4)
        if k == "amount_edit":
            diff = rng.choice([100, 250, 500, 1000, 1500])
        elif k == "reused":
            rep = rng.integers(1, 5)
        elif k == "wrong_merchant":
            mm = 1
        elif k == "blurry_legit":
            conf = rng.normal(55, 10)
        rows.append([diff, rep, mm, float(np.clip(conf, 0, 100)),
                     int(k in ("amount_edit", "reused", "wrong_merchant"))])
    return pd.DataFrame(rows, columns=FEATURES + ["label"])


def train_risk_model() -> pd.DataFrame:
    """Train two models, keep the one with the best recall, save model + metadata."""
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, confusion_matrix,
                                 precision_score, recall_score)
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    df = make_synthetic()
    X_tr, X_te, y_tr, y_te = train_test_split(
        df[FEATURES], df["label"], test_size=0.25, random_state=1, stratify=df["label"])
    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression()),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=1),
    }
    report, best, best_name, best_recall, best_pred = [], None, "", -1.0, None
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        p = m.predict(X_te)
        rec = recall_score(y_te, p)
        report.append({"model": name,
                       "accuracy": round(accuracy_score(y_te, p), 3),
                       "precision": round(precision_score(y_te, p), 3),
                       "recall": round(rec, 3)})
        if rec > best_recall:
            best, best_name, best_recall, best_pred = m, name, rec, p
    joblib.dump(best, MODEL_PATH)

    if hasattr(best, "feature_importances_"):
        imp = np.asarray(best.feature_importances_, dtype=float)
    else:
        imp = np.abs(best[-1].coef_[0])
    imp = imp / imp.sum() if imp.sum() else imp
    meta = {
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "n_samples": int(len(df)), "best_model": best_name,
        "confusion": confusion_matrix(y_te, best_pred).tolist(),
        "importance": {f: round(float(v), 4) for f, v in zip(FEATURES, imp)},
        "report": report,
    }
    Path(MODEL_META_PATH).write_text(json.dumps(meta, indent=2))
    _MODEL_CACHE["mtime"] = None
    return pd.DataFrame(report)


def load_model_meta():
    try:
        return json.loads(Path(MODEL_META_PATH).read_text())
    except Exception:
        return None


_MODEL_CACHE = {"mtime": None, "model": None}


def _load_model():
    p = Path(MODEL_PATH)
    if not p.exists():
        return None
    try:
        mtime = p.stat().st_mtime
        if _MODEL_CACHE["mtime"] != mtime:
            import joblib
            _MODEL_CACHE.update(mtime=mtime, model=joblib.load(p))
        return _MODEL_CACHE["model"]
    except Exception:
        return None


def ensure_model() -> None:
    if not Path(MODEL_PATH).exists():
        try:
            train_risk_model()
        except Exception:
            pass


def risk_score(features):
    """Returns (label, probability or None)."""
    model = _load_model()
    if model is None:
        return "Not scored", None
    try:
        p = float(model.predict_proba(pd.DataFrame([features])[FEATURES])[0][1])
    except Exception:
        return "Not scored", None
    if p < RISK_LOW:
        return "Low risk", p
    if p < RISK_HIGH:
        return "Review recommended", p
    return "High risk", p


def risk_label(features):
    """Low risk / Review recommended / High risk. Never says 'fraud'."""
    return risk_score(features)[0]


# ==================================================================
# 5. DEMO DATA & TESTS
# ==================================================================
def seed_demo_activity(n=60, seed=7) -> int:
    """Insert simulated verification attempts spread over the last 14 days."""
    rng = np.random.default_rng(seed)
    with db_conn() as db:
        refs = db.execute("SELECT reference, amount, merchant_id FROM transactions").fetchall()
        merchants = [r[0] for r in db.execute("SELECT merchant_id FROM merchants")]
    if not refs:
        return 0
    scen = list(REASONS)
    probs = np.array([{"VERIFIED": .45, "NOT FOUND": .07, "PENDING": .08, "FAILED": .06,
                       "WRONG RECIPIENT": .07, "AMOUNT MISMATCH": .10,
                       "DUPLICATE WARNING": .12, "MANUAL REVIEW": .05}[s] for s in scen])
    probs = probs / probs.sum()
    now, rows = datetime.now(), []
    for _ in range(n):
        s = str(rng.choice(scen, p=probs))
        ref, amt, mer = refs[int(rng.integers(len(refs)))]
        is_receipt = rng.random() < 0.4
        method = f"{'RECEIPT' if is_receipt else 'ID'} {SIM_TAG}"
        conf = float(np.clip(rng.normal(92, 4), 60, 100)) if is_receipt else 100.0
        feats = {"amount_diff": 0.0, "repeat_count": 0, "merchant_mismatch": 0, "ocr_conf": conf}
        expected, claimed_merchant = float(amt), mer
        if s == "DUPLICATE WARNING":
            feats["repeat_count"] = int(rng.integers(1, 4))
        elif s == "AMOUNT MISMATCH":
            d = float(rng.choice([100, 250, 500, 1000]))
            feats["amount_diff"], expected = d, float(amt) + d
        elif s == "WRONG RECIPIENT":
            others = [m for m in merchants if m != mer] or [mer]
            claimed_merchant = str(rng.choice(others))
            feats["merchant_mismatch"] = 1
        elif s == "NOT FOUND":
            ref, expected = f"UNK-{int(rng.integers(10000, 99999))}", float(rng.choice([100, 300, 750]))
        elif s == "MANUAL REVIEW":
            feats["ocr_conf"] = float(np.clip(rng.normal(48, 8), 10, 59))
        ts = now - timedelta(days=int(rng.integers(0, 14)), hours=int(rng.integers(0, 12)),
                             minutes=int(rng.integers(0, 60)))
        rows.append((str(ref), expected, str(claimed_merchant), method, s, REASONS[s],
                     risk_label(feats), ts.strftime("%Y-%m-%d %H:%M:%S")))
    with db_conn() as db:
        db.executemany(
            "INSERT INTO verification_history"
            "(reference, expected_amount, merchant_id, method, result, reason, risk, ts) "
            "VALUES(?,?,?,?,?,?,?,?)", rows)
    return len(rows)


def clear_demo_activity() -> None:
    with db_conn() as db:
        db.execute("DELETE FROM verification_history WHERE method LIKE ?", (f"%{SIM_TAG}",))


def run_tests():
    global DB_PATH, MODEL_PATH, MODEL_META_PATH
    tmp = tempfile.mkdtemp()
    DB_PATH = str(Path(tmp) / "test.db")
    MODEL_PATH = str(Path(tmp) / "model.joblib")
    MODEL_META_PATH = str(Path(tmp) / "meta.json")
    init_db(reset=True)
    ensure_model()

    passed = total = 0

    def check(name, cond, detail=""):
        nonlocal passed, total
        total += 1
        passed += bool(cond)
        print(("PASS" if cond else "FAIL"), name, detail)

    print("--- Verification engine")
    cases = [
        (("DEMO-TB-1001", 1500, "SHOP-001"), "VERIFIED"),
        (("DEMO-TB-1002", 250, "SHOP-001"), "PENDING"),
        (("DEMO-TB-1003", 800, "SHOP-001"), "WRONG RECIPIENT"),
        (("DEMO-TB-1004", 420, "SHOP-001"), "FAILED"),
        (("DEMO-TB-1001", 1000, "SHOP-001"), "AMOUNT MISMATCH"),
        (("UNKNOWN-1", 100, "SHOP-001"), "NOT FOUND"),
        (("DEMO-TB-1001", 1500, "SHOP-001"), "DUPLICATE WARNING"),
    ]
    for args, expected in cases:
        got = verify(*args)["result"]
        check(args[0], got == expected, f"{expected} -> {got}")

    print("--- Extra engine behaviour")
    check("lowercase + spaces normalised",
          verify(" demo-tb-1005 ", 2750, "shop-003")["result"] == "VERIFIED")
    check("tolerance accepts small gap",
          verify("DEMO-TB-1006", 100, "SHOP-002", tolerance=10)["result"] == "VERIFIED")
    check("tolerance rejects big gap",
          verify("DEMO-TB-1007", 1000, "SHOP-003", tolerance=10)["result"] == "AMOUNT MISMATCH")
    typo = verify("DEMO-TB-1O08", 330, "SHOP-001")
    check("'did you mean' suggestion", typo["suggestion"] == "DEMO-TB-1008", str(typo["suggestion"]))
    check("low OCR confidence -> MANUAL REVIEW",
          verify("DEMO-TB-1010", 640, "SHOP-003", "RECEIPT", ocr_conf=40)["result"] == "MANUAL REVIEW")
    before = query_df("SELECT COUNT(*) AS n FROM verification_history")["n"][0]
    verify("DEMO-TB-1010", 640, "SHOP-003", record=False)
    after = query_df("SELECT COUNT(*) AS n FROM verification_history")["n"][0]
    check("record=False does not write history", before == after)
    batch = verify_batch(pd.DataFrame(
        [{"reference": "DEMO-TB-1010", "amount": 640, "merchant": "SHOP-003"},
         {"reference": "NOPE", "amount": "abc", "merchant": "SHOP-003"}]))
    check("batch verify returns a row per input", len(batch) == 2 and batch.iloc[1]["result"] == "NOT FOUND")

    print("--- Receipt parser")
    samples = [
        ("Transaction Reference: DEMO-TB-1001\nAmount: 1,500 ETB", ("DEMO-TB-1001", 1500.0)),
        ("Ref CKL1A2B3C4  Paid 250.50 Birr", ("CKL1A2B3C4", 250.5)),
        ("Ref CKL1A2B3C4  ETB 1,200.00", ("CKL1A2B3C4", 1200.0)),
        ("blurry text no numbers", (None, None)),
    ]
    for text, expected in samples:
        got = parse_receipt(text)
        check("parse", got == expected, str(got))

    print("--- Risk model & demo data")
    check("model file created", Path(MODEL_PATH).exists())
    low = risk_score({"amount_diff": 0, "repeat_count": 0, "merchant_mismatch": 0, "ocr_conf": 95})
    high = risk_score({"amount_diff": 1000, "repeat_count": 3, "merchant_mismatch": 1, "ocr_conf": 95})
    check("clean case scores low", low[0] == "Low risk", str(low))
    check("suspicious case scores high", high[0] == "High risk", str(high))
    meta = load_model_meta()
    check("model metadata saved", meta and len(meta["confusion"]) == 2)
    n = seed_demo_activity(30)
    check("sample activity seeded", n == 30)
    check("sample rows tagged", query_df(
        "SELECT COUNT(*) AS n FROM verification_history WHERE method LIKE '%(sim)'")["n"][0] == 30)
    clear_demo_activity()
    check("sample rows cleared", query_df(
        "SELECT COUNT(*) AS n FROM verification_history WHERE method LIKE '%(sim)'")["n"][0] == 0)

    print(f"\nTOTAL: {passed}/{total} passed ({passed / total:.0%})")
    return passed == total


# ==================================================================
# 6. STREAMLIT APP (UI)
# ==================================================================
st = px = go = None   # bound inside main() so the engine can be imported/tested without them

PAGES = ["📊  Dashboard", "🔎  Verify by ID", "🧾  Verify receipt", "📦  Batch verify",
         "💳  Transactions", "🏪  Merchants", "🕘  History", "📈  Analytics & risk", "ℹ️  About"]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root{--g:#0B7A3B;--g2:#14A44D;--gd:#095F2E;--ink:#0F1F17;--muted:#5B6B62;--line:#E3EAE6;}
html, body, [class*="css"], .stApp { font-family:'Inter',system-ui,-apple-system,'Segoe UI',sans-serif; }
.stApp { background:#F4F7F5; }
.block-container { padding-top:1.2rem; max-width:1240px; }
header[data-testid="stHeader"] { background:transparent; }
[data-testid="stSidebar"] { background:#FFFFFF; border-right:1px solid var(--line); }
#MainMenu, footer { visibility:hidden; }

.hero { background:linear-gradient(120deg,#095F2E 0%,#0B7A3B 55%,#14A44D 100%); border-radius:18px;
  padding:1.35rem 1.7rem; display:flex; justify-content:space-between; align-items:center;
  margin-bottom:1.2rem; box-shadow:0 8px 24px rgba(11,122,59,.22); }
.stApp .hero .hero-title { color:#fff; font-size:1.8rem; font-weight:700; line-height:1.2; }
.stApp .hero .hero-sub { color:#DDF5E7; font-size:.97rem; margin-top:.2rem; }
.stApp .hero .pill { background:rgba(255,255,255,.18); color:#fff; border:1px solid rgba(255,255,255,.4);
  padding:.3rem .8rem; border-radius:999px; font-size:.75rem; font-weight:600; letter-spacing:.08em; }

.brand { display:flex; align-items:center; gap:.6rem; margin:.2rem 0 .6rem; }
.brand .logo { width:38px; height:38px; border-radius:11px; background:linear-gradient(135deg,#0B7A3B,#14A44D);
  display:flex; align-items:center; justify-content:center; color:#fff; font-weight:700; font-size:1.1rem; }
.brand .name { font-weight:700; font-size:1.1rem; color:var(--ink); line-height:1.1; }
.brand .tag { font-size:.72rem; color:var(--muted); }

.kpi { background:#fff; border:1px solid var(--line); border-left:5px solid var(--g2); border-radius:14px;
  padding:.9rem 1.1rem; box-shadow:0 1px 2px rgba(16,24,40,.05); }
.kpi.amber { border-left-color:#F59E0B; } .kpi.red { border-left-color:#DC2626; }
.kpi.blue { border-left-color:#2563EB; }  .kpi.grey { border-left-color:#9CA3AF; }
.kpi .lbl { font-size:.74rem; color:var(--muted); text-transform:uppercase; letter-spacing:.06em; font-weight:600; }
.kpi .val { font-size:1.75rem; font-weight:700; color:var(--ink); line-height:1.25; }
.kpi .hint { font-size:.78rem; color:var(--muted); }

.rescard { display:flex; gap:1rem; align-items:center; border-radius:16px; padding:1.2rem 1.4rem;
  border:1px solid; margin-bottom:.8rem; }
.rescard.ok   { background:#ECFDF3; border-color:#A6E9C2; }
.rescard.warn { background:#FFF8E6; border-color:#FCDD8A; }
.rescard.bad  { background:#FEF0EF; border-color:#F7B4B0; }
.rescard .ricon { font-size:2.3rem; }
.rescard .rtitle { font-weight:700; font-size:1.3rem; color:var(--ink); }
.rescard .rtext { color:#334155; font-size:.95rem; }
.rescard .raction { color:var(--ink); font-weight:600; font-size:.92rem; margin-top:.3rem; }

.badge { display:inline-block; padding:.2rem .7rem; border-radius:999px; font-size:.8rem; font-weight:600; color:#fff; }
.section-title { font-weight:700; color:var(--ink); font-size:1.05rem; margin:.4rem 0 .5rem; }
.card { background:#fff; border:1px solid var(--line); border-radius:14px; padding:1rem 1.2rem; }
.demo-note { background:#FFF8E6; border:1px solid #FCDD8A; color:#7A5600; border-radius:12px;
  padding:.6rem .9rem; font-size:.85rem; margin-bottom:1rem; }
div[data-testid="stForm"] { background:#fff; border:1px solid var(--line); border-radius:14px; }
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {
  background:var(--g); border-color:var(--g); border-radius:10px; font-weight:600; }
.stButton > button[kind="primary"]:hover, .stFormSubmitButton > button[kind="primary"]:hover {
  background:var(--gd); border-color:var(--gd); }
</style>
"""


def esc(x) -> str:
    return html.escape(str(x))


def _new_width_api() -> bool:
    try:
        major, minor = (int(p) for p in st.__version__.split(".")[:2])
        return (major, minor) >= (1, 50)
    except Exception:
        return False


def stretch() -> dict:
    """Full-width kwarg that works on old and new Streamlit versions."""
    return {"width": "stretch"} if _new_width_api() else {"use_container_width": True}


def hero(title, subtitle):
    st.markdown(
        f"<div class='hero'><div><div class='hero-title'>{esc(title)}</div>"
        f"<div class='hero-sub'>{esc(subtitle)}</div></div><span class='pill'>DEMO</span></div>",
        unsafe_allow_html=True)


def kpi(col, label, value, hint="", tone=""):
    col.markdown(
        f"<div class='kpi {tone}'><div class='lbl'>{esc(label)}</div>"
        f"<div class='val'>{esc(value)}</div><div class='hint'>{esc(hint)}</div></div>",
        unsafe_allow_html=True)


def badge(text, colour):
    return f"<span class='badge' style='background:{colour}'>{esc(text)}</span>"


def section(title):
    st.markdown(f"<div class='section-title'>{esc(title)}</div>", unsafe_allow_html=True)


def style_fig(fig, height=320):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=45, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, system-ui, sans-serif", size=13),
                      legend=dict(orientation="h", y=-0.15), title_font_size=15)
    return fig


def gauge(p, title="Risk score"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round((p or 0) * 100, 1), number={"suffix": "%"},
        title={"text": title},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#0F1F17", "thickness": 0.25},
               "steps": [{"range": [0, RISK_LOW * 100], "color": "#A6E9C2"},
                         {"range": [RISK_LOW * 100, RISK_HIGH * 100], "color": "#FCDD8A"},
                         {"range": [RISK_HIGH * 100, 100], "color": "#F7B4B0"}]}))
    return style_fig(fig, 230)


def history_df() -> pd.DataFrame:
    h = query_df("SELECT * FROM verification_history ORDER BY ts DESC, id DESC")
    if not h.empty:
        h["ts"] = pd.to_datetime(h["ts"], errors="coerce")
    return h


def with_icon(result: str) -> str:
    return f"{RESULT_STYLE.get(result, ('•',))[0]} {result}"


def empty_state():
    st.info("No verification activity yet. Run a check, or load simulated sample activity "
            "to see the dashboard in action.")
    if st.button("Load sample activity", type="primary"):
        seed_demo_activity(60)
        st.rerun()


def render_result(res):
    icon, kind, _ = RESULT_STYLE[res["result"]]
    st.markdown(
        f"<div class='rescard {kind}'><div class='ricon'>{icon}</div><div>"
        f"<div class='rtitle'>{esc(res['result'])}</div><div class='rtext'>{esc(res['reason'])}</div>"
        f"<div class='raction'>{esc(res['action'])}</div></div></div>", unsafe_allow_html=True)
    st.markdown(
        f"Risk signal: {badge(res['risk'], RISK_COLORS.get(res['risk'], '#9CA3AF'))} "
        f"<span style='color:#5B6B62;font-size:.85rem'>&nbsp;a signal for review, not proof of fraud</span>",
        unsafe_allow_html=True)
    if res.get("suggestion"):
        st.warning(f"Did you mean **{res['suggestion']}**?")
    if res.get("risk_score") is not None:
        st.plotly_chart(gauge(res["risk_score"]), **stretch())
    with st.expander("Details"):
        st.table(pd.DataFrame({
            "Field": ["Reference", "Recorded amount (ETB)", "Recorded merchant",
                      "Recorded status", "Earlier checks of this reference"],
            "Value": [res["reference"],
                      "-" if res["recorded_amount"] is None else f"{res['recorded_amount']:,.2f}",
                      res["recorded_merchant"] or "-", res["status"] or "-", res["prior_checks"]]}))


# ---------------------------------------------------------------- pages
def page_dashboard():
    hero("Dashboard", "Live overview of payment-verification activity")
    h = history_df()
    if h.empty:
        empty_state()
        return
    total = len(h)
    verified = int((h["result"] == "VERIFIED").sum())
    review = int(h["risk"].isin(["Review recommended", "High risk"]).sum())
    value = float(h.loc[h["result"] == "VERIFIED", "expected_amount"].sum())
    blocked = int(h["result"].isin(["AMOUNT MISMATCH", "WRONG RECIPIENT", "NOT FOUND", "FAILED"]).sum())

    c = st.columns(5)
    kpi(c[0], "Checks run", f"{total:,}", "all time", "blue")
    kpi(c[1], "Verified", f"{verified:,}", f"{verified / total:.0%} success rate")
    kpi(c[2], "Value verified", f"{value:,.0f} ETB", "completed & matched")
    kpi(c[3], "Needs review", f"{review:,}", "risk signal raised", "amber")
    kpi(c[4], "Rejected", f"{blocked:,}", "mismatch / failed / not found", "red")
    st.write("")

    left, right = st.columns([1, 1.4])
    counts = h["result"].value_counts().rename_axis("result").reset_index(name="count")
    fig = px.pie(counts, names="result", values="count", hole=0.58, color="result",
                 color_discrete_map=RESULT_COLORS, title="Verification outcomes")
    fig.update_traces(textinfo="percent", sort=False)
    left.plotly_chart(style_fig(fig, 340), **stretch())

    h["day"] = h["ts"].dt.floor("D")
    h["group"] = np.where(h["result"] == "VERIFIED", "Verified", "Not verified")
    trend = h.groupby(["day", "group"]).size().reset_index(name="checks")
    fig = px.bar(trend, x="day", y="checks", color="group", title="Checks per day",
                 color_discrete_map={"Verified": "#14A44D", "Not verified": "#F59E0B"})
    fig.update_xaxes(title=""); fig.update_yaxes(title="")
    right.plotly_chart(style_fig(fig, 340), **stretch())

    left, right = st.columns(2)
    risk = h["risk"].value_counts().rename_axis("risk").reset_index(name="count")
    fig = px.bar(risk, x="risk", y="count", color="risk", color_discrete_map=RISK_COLORS,
                 title="Risk signals")
    fig.update_layout(showlegend=False); fig.update_xaxes(title=""); fig.update_yaxes(title="")
    left.plotly_chart(style_fig(fig, 300), **stretch())

    names = merchant_options()
    h["merchant"] = h["merchant_id"].map(lambda m: names.get(m, m))
    bym = h.groupby(["merchant", "result"]).size().reset_index(name="checks")
    fig = px.bar(bym, y="merchant", x="checks", color="result", orientation="h",
                 color_discrete_map=RESULT_COLORS, title="Checks by merchant")
    fig.update_xaxes(title=""); fig.update_yaxes(title="")
    right.plotly_chart(style_fig(fig, 300), **stretch())

    section("Recent activity")
    recent = h.head(12).copy()
    recent["result"] = recent["result"].map(with_icon)
    recent["ts"] = recent["ts"].dt.strftime("%Y-%m-%d %H:%M")
    st.dataframe(recent[["ts", "reference", "expected_amount", "merchant_id", "method",
                         "result", "risk"]], hide_index=True, **stretch())


def page_verify_id(tol):
    hero("Verify by transaction ID", "Check a customer's payment reference before releasing goods")
    merchants = merchant_options()
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        with st.form("verify_id"):
            ref = st.text_input("Transaction reference", "DEMO-TB-1001", placeholder="e.g. DEMO-TB-1001")
            amount = st.number_input("Expected amount (ETB)", min_value=0.0, value=1500.0,
                                     step=50.0, format="%.2f")
            merchant = st.selectbox("Merchant", list(merchants),
                                    format_func=lambda m: f"{m} · {merchants[m]}")
            go_btn = st.form_submit_button("Verify payment", type="primary", **stretch())
        if tol:
            st.caption(f"Amount tolerance active: ±{tol:g} ETB")
        with st.expander("Demo references to try"):
            ex = query_df("SELECT reference, amount, merchant_id, status FROM transactions "
                          "ORDER BY reference LIMIT 12")
            st.dataframe(ex, hide_index=True, **stretch())
    with right:
        if go_btn:
            if not ref.strip():
                st.warning("Enter a transaction reference first.")
            else:
                st.session_state["last_result"] = verify(ref, amount, merchant, "ID", tolerance=tol)
        if st.session_state.get("last_result"):
            render_result(st.session_state["last_result"])
        else:
            st.markdown("<div class='card'><b>How it works</b><br>Enter the reference the customer "
                        "gives you, the amount you expect and your merchant account. The system checks "
                        "status, amount, recipient and whether the reference was used before.</div>",
                        unsafe_allow_html=True)


def page_verify_receipt(tol):
    hero("Verify a receipt screenshot", "Read the reference and amount from an image, then check them")
    merchants = merchant_options()
    c1, c2 = st.columns(2)
    merchant = c1.selectbox("Merchant", list(merchants),
                            format_func=lambda m: f"{m} · {merchants[m]}")
    expected = c2.number_input("Expected amount (ETB)", min_value=0.0, value=1500.0, step=50.0)
    up = st.file_uploader("Receipt screenshot", type=["png", "jpg", "jpeg"])
    if not up:
        st.info("Upload a clear screenshot. Text is read with OCR; low-quality images are "
                "sent to manual review instead of guessed.")
        return
    from PIL import Image
    img = Image.open(up)
    key = f"{up.name}-{up.size}"
    left, right = st.columns([1, 1.3], gap="large")
    with left:
        st.image(img, caption="Uploaded receipt", width=300)
    with right:
        try:
            text, conf = ocr_image(img)
        except Exception as e:
            st.error(f"OCR is not available: {e}\n\nInstall the Tesseract program and "
                     "`pip install pytesseract`.")
            return
        ref0, amt0 = parse_receipt(text)
        st.progress(min(int(conf), 100), text=f"OCR confidence: {conf:.0f}%")
        if conf < OCR_MIN_CONF:
            st.warning("Low confidence read. Check the values below against the image.")
        section("Extracted values (editable)")
        ref = st.text_input("Reference", ref0 or "", key=f"r_{key}")
        amt = st.number_input("Amount on receipt (ETB)", min_value=0.0,
                              value=float(amt0 or 0.0), key=f"a_{key}")
        confirmed = st.checkbox("I compared these values with the receipt myself", key=f"c_{key}")
        with st.expander("Raw OCR text"):
            st.text(text)
        if st.button("Verify receipt", type="primary", **stretch()):
            if not ref.strip():
                st.warning("MANUAL REVIEW: no reference could be read.")
                return
            if amt and abs(amt - expected) > tol:
                st.warning(f"Receipt shows {amt:,.2f} ETB but you expected {expected:,.2f} ETB.")
            used_conf = 100.0 if confirmed else conf
            res = verify(ref, expected, merchant,
                         "RECEIPT (checked)" if confirmed else "RECEIPT",
                         ocr_conf=used_conf, tolerance=tol)
            render_result(res)


def page_batch(tol):
    hero("Batch verification", "Check many payments at once from a CSV file")
    template = pd.DataFrame({"reference": ["DEMO-TB-1001", "DEMO-TB-1005", "DEMO-TB-1009"],
                             "amount": [1500, 2750, 5000],
                             "merchant": ["SHOP-001", "SHOP-003", "SHOP-002"]})
    st.download_button("Download CSV template", template.to_csv(index=False),
                       "batch_template.csv", "text/csv")
    up = st.file_uploader("CSV with columns: reference, amount, merchant", type=["csv"])
    if not up:
        return
    try:
        df = pd.read_csv(up)
        df.columns = [str(c).strip().lower() for c in df.columns]
    except Exception as e:
        st.error(f"Could not read the file: {e}")
        return
    missing = {"reference", "amount", "merchant"} - set(df.columns)
    if missing:
        st.error(f"Missing column(s): {', '.join(sorted(missing))}")
        return
    st.caption(f"{len(df)} rows found" + (f" (only the first {BATCH_LIMIT} are checked)"
                                          if len(df) > BATCH_LIMIT else ""))
    st.dataframe(df.head(10), hide_index=True, **stretch())
    if st.button("Run batch verification", type="primary"):
        with st.spinner("Verifying…"):
            out = verify_batch(df, tol)
        c = st.columns(3)
        kpi(c[0], "Rows checked", len(out))
        kpi(c[1], "Verified", int((out["result"] == "VERIFIED").sum()))
        kpi(c[2], "Not verified", int((out["result"] != "VERIFIED").sum()), tone="amber")
        out["result"] = out["result"].map(with_icon)
        st.dataframe(out, hide_index=True, **stretch())
        st.download_button("Download results (CSV)", out.to_csv(index=False),
                           "batch_results.csv", "text/csv")


def page_transactions():
    hero("Authorised transactions", "The simulated records every check is compared against")
    t = query_df("""
        SELECT t.reference, t.amount, t.merchant_id, m.name AS merchant, t.status, t.transaction_date,
          (SELECT COUNT(*) FROM verification_history v
             WHERE v.reference = t.reference AND v.method NOT LIKE '%(sim)') AS checks
        FROM transactions t LEFT JOIN merchants m ON m.merchant_id = t.merchant_id
        ORDER BY t.transaction_date DESC, t.reference""")
    f1, f2, f3 = st.columns([1, 1, 1.4])
    status = f1.multiselect("Status", sorted(t["status"].unique()))
    mer = f2.multiselect("Merchant", sorted(t["merchant_id"].unique()))
    q = f3.text_input("Search reference")
    if status:
        t = t[t["status"].isin(status)]
    if mer:
        t = t[t["merchant_id"].isin(mer)]
    if q:
        t = t[t["reference"].str.contains(q.strip(), case=False, na=False)]
    st.dataframe(t, hide_index=True, **stretch(),
                 column_config={"amount": st.column_config.NumberColumn("Amount (ETB)", format="%.2f")})
    with st.expander("➕ Add a demo transaction"):
        merchants = merchant_options()
        with st.form("add_tx", clear_on_submit=True):
            a, b = st.columns(2)
            ref = a.text_input("Reference")
            amt = b.number_input("Amount (ETB)", min_value=0.0, step=50.0)
            c, d, e = st.columns(3)
            mid = c.selectbox("Merchant", list(merchants))
            stat = d.selectbox("Status", ["COMPLETED", "PENDING", "FAILED"])
            dt = e.date_input("Date", date.today())
            if st.form_submit_button("Add transaction", type="primary"):
                ref = normalize_ref(ref)
                if not ref:
                    st.warning("Reference is required.")
                else:
                    try:
                        with db_conn() as db:
                            db.execute("INSERT INTO transactions VALUES(?,?,?,?,?)",
                                       (ref, float(amt), mid, stat, dt.isoformat()))
                        st.success(f"Added {ref}.")
                        st.rerun()
                    except sqlite3.IntegrityError:
                        st.error("That reference already exists.")


def page_merchants():
    hero("Merchants", "Accounts that can receive payments in this demo")
    m = query_df("""
        SELECT m.merchant_id, m.name,
          (SELECT COUNT(*) FROM transactions t WHERE t.merchant_id=m.merchant_id) AS transactions,
          (SELECT COALESCE(SUM(amount),0) FROM transactions t
             WHERE t.merchant_id=m.merchant_id AND t.status='COMPLETED') AS completed_value,
          (SELECT COUNT(*) FROM verification_history v WHERE v.merchant_id=m.merchant_id) AS checks,
          (SELECT COUNT(*) FROM verification_history v
             WHERE v.merchant_id=m.merchant_id AND v.result='VERIFIED') AS verified
        FROM merchants m ORDER BY m.merchant_id""")
    m["verified_rate"] = np.where(m["checks"] > 0, m["verified"] / m["checks"].replace(0, 1), 0.0)
    st.dataframe(m, hide_index=True, **stretch(), column_config={
        "completed_value": st.column_config.NumberColumn("Completed value (ETB)", format="%.0f"),
        "verified_rate": st.column_config.ProgressColumn("Verified rate", min_value=0, max_value=1,
                                                         format="percent")})
    with st.expander("➕ Add a merchant"):
        with st.form("add_merchant", clear_on_submit=True):
            a, b = st.columns(2)
            mid = a.text_input("Merchant ID", placeholder="SHOP-004")
            name = b.text_input("Name")
            if st.form_submit_button("Add merchant", type="primary"):
                mid = mid.strip().upper()
                if not mid:
                    st.warning("Merchant ID is required.")
                else:
                    try:
                        with db_conn() as db:
                            db.execute("INSERT INTO merchants VALUES(?,?)", (mid, name.strip() or mid))
                        st.success(f"Added {mid}.")
                        st.rerun()
                    except sqlite3.IntegrityError:
                        st.error("That merchant ID already exists.")


def page_history():
    hero("Verification history", "Every check, with filters and CSV export")
    h = history_df()
    if h.empty:
        empty_state()
        return
    f1, f2, f3 = st.columns(3)
    res = f1.multiselect("Result", sorted(h["result"].unique()))
    risk = f2.multiselect("Risk", sorted(h["risk"].unique()))
    q = f3.text_input("Search reference")
    lo, hi = h["ts"].min().date(), h["ts"].max().date()
    rng = st.date_input("Date range", (lo, hi), min_value=lo, max_value=hi)
    if res:
        h = h[h["result"].isin(res)]
    if risk:
        h = h[h["risk"].isin(risk)]
    if q:
        h = h[h["reference"].str.contains(q.strip(), case=False, na=False)]
    if isinstance(rng, (tuple, list)) and len(rng) == 2:
        h = h[(h["ts"].dt.date >= rng[0]) & (h["ts"].dt.date <= rng[1])]
    st.caption(f"{len(h):,} records")
    show = h.copy()
    show["result"] = show["result"].map(with_icon)
    st.dataframe(show.drop(columns=["id"]), hide_index=True, **stretch())
    st.download_button("Download CSV", h.to_csv(index=False), "history.csv", "text/csv")


def page_analytics():
    hero("Analytics & risk", "Patterns in the data and the experimental risk model")
    tab1, tab2, tab3 = st.tabs(["Patterns", "Risk model", "Risk simulator"])
    with tab1:
        h = history_df()
        if h.empty:
            empty_state()
        else:
            h["method_group"] = h["method"].str.replace(r"\s*\(sim\)", "", regex=True)
            a, b = st.columns(2)
            by = h.groupby(["method_group", "result"]).size().reset_index(name="checks")
            fig = px.bar(by, x="method_group", y="checks", color="result", color_discrete_map=RESULT_COLORS,
                         title="Outcomes by check method")
            fig.update_xaxes(title=""); fig.update_yaxes(title="")
            a.plotly_chart(style_fig(fig, 340), **stretch())
            h["hour"] = h["ts"].dt.hour
            fig = px.histogram(h, x="hour", nbins=24, title="When checks happen (hour of day)",
                               color_discrete_sequence=["#14A44D"])
            fig.update_xaxes(title=""); fig.update_yaxes(title="")
            b.plotly_chart(style_fig(fig, 340), **stretch())
            section("Repeated references")
            dup = h["reference"].value_counts().reset_index()
            dup.columns = ["reference", "attempts"]
            dup = dup[dup["attempts"] > 1]
            if dup.empty:
                st.caption("No reference has been checked more than once.")
            else:
                st.dataframe(dup, hide_index=True, **stretch())
    with tab2:
        st.caption("Trained on simulated scenarios, so the scores show the method works, "
                   "not how it would perform on real payments.")
        if st.button("Train / retrain risk model", type="primary"):
            with st.spinner("Training two models…"):
                train_risk_model()
            st.toast("Risk model retrained")
        meta = load_model_meta()
        if not meta:
            st.info("No model has been trained yet.")
        else:
            c = st.columns(3)
            kpi(c[0], "Best model", meta["best_model"], "chosen by recall")
            kpi(c[1], "Training rows", f"{meta['n_samples']:,}", "synthetic", "blue")
            kpi(c[2], "Last trained", meta["trained_at"], "", "grey")
            st.write("")
            rep = pd.DataFrame(meta["report"])
            a, b = st.columns(2)
            long = rep.melt(id_vars="model", var_name="metric", value_name="score")
            fig = px.bar(long, x="metric", y="score", color="model", barmode="group",
                         title="Model comparison", color_discrete_sequence=["#0B7A3B", "#F59E0B"])
            fig.update_yaxes(range=[0, 1.05], title=""); fig.update_xaxes(title="")
            a.plotly_chart(style_fig(fig, 320), **stretch())
            cm = np.array(meta["confusion"])
            fig = px.imshow(cm, text_auto=True, color_continuous_scale="Greens",
                            x=["Predicted OK", "Predicted review"],
                            y=["Actually OK", "Actually review"], title="Confusion matrix (test set)")
            fig.update_coloraxes(showscale=False)
            b.plotly_chart(style_fig(fig, 320), **stretch())
            imp = pd.DataFrame({"feature": [FEATURE_LABELS[k] for k in meta["importance"]],
                                "importance": list(meta["importance"].values())}).sort_values("importance")
            fig = px.bar(imp, x="importance", y="feature", orientation="h", title="What drives the score",
                         color_discrete_sequence=["#14A44D"])
            fig.update_xaxes(title=""); fig.update_yaxes(title="")
            st.plotly_chart(style_fig(fig, 280), **stretch())
            st.dataframe(rep, hide_index=True, **stretch())
    with tab3:
        st.caption("Move the sliders to see how each signal changes the risk score.")
        a, b = st.columns([1, 1])
        with a:
            f = {"amount_diff": st.slider("Amount difference (ETB)", 0, 2000, 0, 50),
                 "repeat_count": st.slider("Earlier checks of this reference", 0, 6, 0),
                 "merchant_mismatch": int(st.checkbox("Paid to a different merchant")),
                 "ocr_conf": float(st.slider("OCR confidence (%)", 0, 100, 95))}
        label, p = risk_score(f)
        with b:
            if p is None:
                st.warning("Train the risk model first (Risk model tab).")
            else:
                st.plotly_chart(gauge(p), **stretch())
                st.markdown(badge(label, RISK_COLORS[label]), unsafe_allow_html=True)


def page_about():
    hero("About this prototype", "What it does, how to read the results, and what it is not")
    st.markdown("<div class='demo-note'><b>DEMO:</b> uses simulated transactions, not real Telebirr "
                "payments. Never enter a PIN, OTP or password here.</div>", unsafe_allow_html=True)
    a, b = st.columns([1.2, 1], gap="large")
    with a:
        section("Result guide")
        guide = pd.DataFrame({"Result": [with_icon(k) for k in REASONS],
                              "Meaning": list(REASONS.values()),
                              "What to do": [ACTIONS[k] for k in REASONS]})
        st.dataframe(guide, hide_index=True, **stretch())
    with b:
        section("How a check works")
        st.markdown(
            "1. **Find** the reference in the authorised records  \n"
            "2. **Status** must be completed  \n"
            "3. **Recipient** must be your merchant account  \n"
            "4. **Amount** must match (within the tolerance)  \n"
            "5. **First use** of the reference  \n"
            "6. A **risk model** adds a review signal  \n\n"
            "Receipts are read with OCR. Unclear images go to manual review instead of being guessed.")
        section("Limits")
        st.markdown("- Risk scores come from a model trained on **synthetic** data.  \n"
                    "- A risk signal is a prompt to look closer, never proof of fraud.  \n"
                    "- Not connected to any real payment system.")
    st.caption(f"{APP_NAME} v{APP_VERSION} · Streamlit · SQLite · scikit-learn · Plotly · Tesseract OCR")


# ---------------------------------------------------------------- main
def main():
    global st, px, go
    import plotly.express as _px
    import plotly.graph_objects as _go
    import streamlit as _st
    st, px, go = _st, _px, _go

    st.set_page_config(page_title=f"{APP_NAME} · DEMO", page_icon="💚",
                       layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)
    init_db()
    if not Path(MODEL_PATH).exists():
        with st.spinner("Preparing the risk model (first run only)…"):
            ensure_model()

    with st.sidebar:
        st.markdown("<div class='brand'><div class='logo'>T</div><div><div class='name'>"
                    f"{APP_NAME}</div><div class='tag'>Payment checks for merchants</div></div></div>",
                    unsafe_allow_html=True)
        page = st.radio("Navigate", PAGES, label_visibility="collapsed")
        st.divider()
        tol = st.number_input("Amount tolerance (ETB)", 0.0, 100.0, 0.0, 0.5,
                              help="Accept a recorded amount this close to the expected amount.")
        with st.expander("Demo tools"):
            if st.button("Load sample activity", **stretch()):
                n = seed_demo_activity(60)
                st.toast(f"Added {n} simulated checks")
                st.rerun()
            if st.button("Clear sample activity", **stretch()):
                clear_demo_activity()
                st.rerun()
            sure = st.checkbox("I understand this erases all data")
            if st.button("Reset demo database", disabled=not sure, **stretch()):
                init_db(reset=True)
                st.session_state.pop("last_result", None)
                st.rerun()
        st.caption(f"v{APP_VERSION} · demo data only")

    st.markdown("<div class='demo-note'>DEMO: simulated transactions, not real Telebirr payments. "
                "Never enter a PIN, OTP or password.</div>", unsafe_allow_html=True)

    {
        PAGES[0]: page_dashboard,
        PAGES[1]: lambda: page_verify_id(tol),
        PAGES[2]: lambda: page_verify_receipt(tol),
        PAGES[3]: lambda: page_batch(tol),
        PAGES[4]: page_transactions,
        PAGES[5]: page_merchants,
        PAGES[6]: page_history,
        PAGES[7]: page_analytics,
        PAGES[8]: page_about,
    }[page]()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        sys.exit(0 if run_tests() else 1)
    main()
