# 🌸 AI Skincare Subscription — India Market Validation

## Assignment Coverage
- **Part 1 (10 marks)** — 2000-row synthetic survey dataset with intentional dirty data
- **Part 2 (10 marks)** — Full data cleaning + 5 derived columns
- **Part 3 (30 marks)** — EDA with 8 charts + insights in Excel AND interactive Streamlit dashboard

## Files Included
| File | Description |
|------|-------------|
| `01_Raw_Survey_Data_2000rows.csv` | Raw synthetic data with ~5% dirty records |
| `02_Clean_Data.csv` | Cleaned + transformed dataset (51 columns) |
| `03_EDA_Workbook.xlsx` | 8-sheet Excel workbook: Cover, Raw, Clean, Cleaning Log, Stats, Correlation, Charts, Insights |
| `app.py` | Streamlit dashboard (7 interactive tabs) |
| `requirements.txt` | Python dependencies |

## Run Dashboard Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud
1. Upload all files to GitHub
2. Go to share.streamlit.io → New App → select app.py
3. Deploy
