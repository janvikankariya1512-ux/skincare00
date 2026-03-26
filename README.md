# 🌸 AI Skincare Subscription — Streamlit Dashboard

## Setup & Run Instructions

### Step 1 — Install Python (if not installed)
Download from https://www.python.org/downloads/ (Python 3.9+)

### Step 2 — Install dependencies
Open terminal / command prompt in this folder and run:

```bash
pip install -r requirements.txt
```

### Step 3 — Run the dashboard

```bash
streamlit run app.py
```

The app will open automatically at: **http://localhost:8501**

---

## What's Inside

| File | Description |
|------|-------------|
| `app.py` | Main Streamlit dashboard — 7 tabs of EDA |
| `requirements.txt` | Python package dependencies |
| `AI_Skincare_Subscription_Assignment.xlsx` | Full Excel workbook (7 sheets) |

## Dashboard Tabs

1. **📦 Data Overview** — Raw dirty data documentation + cleaning log + descriptive stats
2. **📊 Subscription Analysis** — Plan-wise revenue, CLV, churn, satisfaction
3. **🤖 AI Engine** — AI score vs churn/CLV/satisfaction scatter plots + skin type analysis
4. **🏙️ Geography** — City & tier-wise customer distribution and CLV
5. **📢 Channels & Demographics** — Acquisition channel ROI + age/gender breakdown
6. **🔗 Correlation** — Interactive Pearson heatmap + key findings
7. **💡 Insights** — Strategic recommendations with priority tags

## Assignment Coverage

- **Part 1 (10 marks)** — Synthetic data generation with dirty data
- **Part 2 (10 marks)** — Data cleaning & transformation
- **Part 3 (30 marks)** — EDA with 15+ interactive charts and business insights
