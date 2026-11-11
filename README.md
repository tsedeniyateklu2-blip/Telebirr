# Telebirr Verify (DEMO)

Payment-verification assistant for small merchants. **Demo only: it uses simulated
transactions and is not connected to Telebirr.** Never enter a PIN, OTP or password.

## Run
```bash
pip install -r requirements.txt
# OCR also needs the Tesseract program: https://github.com/tesseract-ocr/tesseract
streamlit run app.py        # the app
python app.py test          # 25 automated checks
```
Keep `telebirr_demo.db` next to `app.py` (it is created automatically if missing).

## Pages
| Page | What it does |
|---|---|
| Dashboard | KPIs, outcome donut, daily trend, risk signals, checks by merchant |
| Verify by ID | Check one reference; result card, action advice, risk gauge, "did you mean" |
| Verify receipt | OCR a screenshot, edit the extracted values, then verify |
| Batch verify | Upload a CSV, verify hundreds of rows, download results |
| Transactions / Merchants | Browse, filter and add simulated records |
| History | Filter by result, risk, date; export CSV |
| Analytics & risk | Patterns, model comparison, confusion matrix, feature importance, risk simulator |

## Verification results
VERIFIED, NOT FOUND, PENDING, FAILED, WRONG RECIPIENT, AMOUNT MISMATCH,
DUPLICATE WARNING, MANUAL REVIEW. A risk signal is a prompt to look closer, never proof of fraud.

## Notes
- Risk model is trained on synthetic data; scores show the method, not real-world accuracy.
- "Load sample activity" inserts rows tagged `(sim)`; they never affect real duplicate checks.
