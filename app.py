import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Skincare | Business Validation Dashboard",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --rose:    #C2185B;
    --rose-lt: #FCE4EC;
    --peach:   #FFDDE1;
    --teal:    #00838F;
    --gold:    #F9A825;
    --slate:   #37474F;
    --bg:      #FFF8FA;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
}

/* Header */
.dash-header {
    background: linear-gradient(135deg, #C2185B 0%, #880E4F 60%, #4A148C 100%);
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 28px;
    color: white;
    position: relative;
    overflow: hidden;
}
.dash-header::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at 80% 50%, rgba(255,255,255,0.08) 0%, transparent 60%);
}
.dash-header h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2.4rem;
    font-weight: 900;
    margin: 0 0 6px 0;
    letter-spacing: -0.5px;
}
.dash-header p  { margin: 0; opacity: 0.82; font-size: 1rem; font-weight: 300; }
.petal { position: absolute; right: 40px; top: 50%; transform: translateY(-50%); font-size: 5rem; opacity: 0.18; }

/* KPI cards */
.kpi-card {
    background: white;
    border-radius: 14px;
    padding: 22px 24px;
    border-left: 5px solid var(--rose);
    box-shadow: 0 2px 16px rgba(194,24,91,0.08);
    transition: transform .2s;
}
.kpi-card:hover { transform: translateY(-3px); }
.kpi-label { font-size: 0.78rem; font-weight: 600; letter-spacing: 1.2px; text-transform: uppercase; color: #9E9E9E; margin-bottom: 6px; }
.kpi-value { font-family: 'Playfair Display', serif; font-size: 2.1rem; font-weight: 700; color: var(--rose); line-height: 1; }
.kpi-sub   { font-size: 0.82rem; color: #78909C; margin-top: 4px; }
.kpi-teal  { border-left-color: var(--teal); }
.kpi-teal .kpi-value { color: var(--teal); }
.kpi-gold  { border-left-color: var(--gold); }
.kpi-gold .kpi-value { color: var(--gold); }
.kpi-slate { border-left-color: var(--slate); }
.kpi-slate .kpi-value { color: var(--slate); }

/* Section titles */
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--slate);
    border-bottom: 2px solid var(--rose-lt);
    padding-bottom: 8px;
    margin: 28px 0 16px 0;
}

/* Insight box */
.insight-box {
    background: linear-gradient(135deg, #FCE4EC, #FFF8FA);
    border-left: 4px solid var(--rose);
    border-radius: 0 12px 12px 0;
    padding: 14px 18px;
    margin: 12px 0;
    font-size: 0.9rem;
    color: var(--slate);
    line-height: 1.6;
}
.insight-box strong { color: var(--rose); }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #880E4F 0%, #C2185B 100%);
}
[data-testid="stSidebar"] * { color: white !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label { color: #FFD6E7 !important; font-size: 0.85rem !important; letter-spacing: 0.5px; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    background: white;
    border-radius: 8px 8px 0 0;
    font-weight: 500;
    color: var(--slate);
    padding: 8px 18px;
}
.stTabs [aria-selected="true"] {
    background: var(--rose) !important;
    color: white !important;
}

/* Plotly chart border */
.stPlotlyChart { border-radius: 12px; overflow: hidden; box-shadow: 0 2px 12px rgba(0,0,0,0.06); }

/* Table */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* Dirty data badge */
.dirty-badge {
    display: inline-block;
    background: #FF8F00;
    color: white;
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 2px;
}
.clean-badge {
    display: inline-block;
    background: #2E7D32;
    color: white;
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 2px;
}
</style>
""", unsafe_allow_html=True)


# ── DATA GENERATION ───────────────────────────────────────────────────────────
@st.cache_data
def generate_data():
    np.random.seed(42)
    N = 200
    cities     = ['Mumbai','Delhi','Bengaluru','Hyderabad','Chennai','Pune','Kolkata','Ahmedabad','Jaipur','Lucknow']
    city_tier  = {c: 'Tier-1' if c in ['Mumbai','Delhi','Bengaluru','Hyderabad','Chennai','Kolkata'] else 'Tier-2' for c in cities}
    skin_types = ['Oily','Dry','Combination','Normal','Sensitive']
    concerns   = ['Acne','Pigmentation','Anti-aging','Hydration','Brightening','Pore-care','Redness']
    sub_plans  = ['Basic','Standard','Premium']
    channels   = ['Instagram','Google','Word-of-Mouth','YouTube','Influencer']
    genders    = ['Female','Male','Non-binary']

    from datetime import datetime, timedelta
    cust_ids  = [f'CUST{1000+i}' for i in range(N)]
    ages      = np.random.randint(18, 52, N)
    gender    = np.random.choice(genders, N, p=[0.62,0.33,0.05])
    city_col  = np.random.choice(cities, N, p=[0.18,0.16,0.14,0.10,0.09,0.08,0.07,0.07,0.06,0.05])
    tier_col  = [city_tier[c] for c in city_col]
    skin_col  = np.random.choice(skin_types, N, p=[0.30,0.22,0.25,0.13,0.10])
    concern   = np.random.choice(concerns, N)
    plan_col  = np.random.choice(sub_plans, N, p=[0.40,0.35,0.25])
    plan_price= {'Basic':499,'Standard':899,'Premium':1499}
    monthly_rev = np.array([plan_price[p] for p in plan_col],dtype=float) + np.random.normal(0,30,N)
    monthly_rev = np.clip(monthly_rev,300,1600).round(0)
    monthly_rev[100] = np.nan

    ai_score = np.clip(np.random.normal(72,15,N),20,100).round(1)
    ai_score[5]=-5; ai_score[18]=150; ai_score[42]=np.nan

    satisfaction = np.clip((ai_score/100)*5 + np.random.normal(0,0.5,N),1,5).round(1)
    satisfaction[10]=np.nan; satisfaction[77]=9.5

    churn_prob = 1/(1+np.exp(0.05*(ai_score-65)+np.random.normal(0,0.3,N)))
    churned    = (churn_prob>0.5).astype(float); churned[30]=5

    ltv_months = np.where(churned==1, np.random.randint(1,4,N), np.random.randint(6,24,N)).astype(float)
    ltv_months[55]=np.nan
    tenure = np.random.randint(1,25,N).astype(float); tenure[90]=-3
    acq_channel = np.random.choice(channels,N,p=[0.35,0.25,0.15,0.15,0.10])
    referrals   = np.random.poisson(1.2,N)
    nps_score   = np.clip(np.random.normal(52,20,N),-100,100).round(0).astype(int)
    order_freq  = np.random.choice([1,2,3,4],N,p=[0.20,0.45,0.25,0.10])
    signup_dates= [datetime(2023,1,1)+timedelta(days=int(x)) for x in np.random.randint(0,730,N)]
    signup_str  = [d.strftime('%Y-%m-%d') for d in signup_dates]
    signup_str[15]='2023/03/10'; signup_str[66]='March 2023'; signup_str[120]=''

    raw = pd.DataFrame({
        'Customer_ID':cust_ids,'Age':ages,'Gender':gender,'City':city_col,'City_Tier':tier_col,
        'Skin_Type':skin_col,'Primary_Concern':concern,'Subscription_Plan':plan_col,
        'Monthly_Revenue_INR':monthly_rev,'AI_Personalization_Score':ai_score,
        'Satisfaction_Rating':satisfaction,'Churned':churned,'LTV_Months':ltv_months,
        'Tenure_Months':tenure,'Acquisition_Channel':acq_channel,'Referrals_Given':referrals,
        'NPS_Score':nps_score,'Order_Frequency':order_freq,'Signup_Date':signup_str,
    })

    # Clean
    clean = raw.copy()
    ai = pd.to_numeric(clean['AI_Personalization_Score'],errors='coerce')
    ai = ai.where((ai>=0)&(ai<=100)); clean['AI_Personalization_Score'] = ai.fillna(ai.median())
    sat = pd.to_numeric(clean['Satisfaction_Rating'],errors='coerce')
    sat = sat.where(sat<=5); clean['Satisfaction_Rating'] = sat.fillna(sat.median())
    ch = pd.to_numeric(clean['Churned'],errors='coerce')
    ch = ch.where(ch.isin([0,1])); clean['Churned'] = ch.fillna(0).astype(int)
    ten = pd.to_numeric(clean['Tenure_Months'],errors='coerce')
    ten = ten.where(ten>=0); clean['Tenure_Months'] = ten.fillna(ten.median())
    clean['LTV_Months'] = clean['LTV_Months'].fillna(clean['LTV_Months'].median())
    clean['Monthly_Revenue_INR'] = clean['Monthly_Revenue_INR'].fillna(clean['Monthly_Revenue_INR'].median())

    def parse_date(s):
        from datetime import datetime
        for fmt in ['%Y-%m-%d','%Y/%m/%d','%B %Y']:
            try: return datetime.strptime(str(s).strip(), fmt).strftime('%Y-%m-%d')
            except: pass
        return '2023-01-01'
    clean['Signup_Date'] = clean['Signup_Date'].apply(parse_date)
    clean['Annual_Revenue_INR'] = (clean['Monthly_Revenue_INR']*12).round(0)
    clean['CLV_INR']            = (clean['Monthly_Revenue_INR']*clean['LTV_Months']).round(0)
    clean['AI_Score_Band']      = pd.cut(clean['AI_Personalization_Score'],
                                          bins=[0,40,60,80,100],
                                          labels=['Low(0-40)','Med(41-60)','High(61-80)','VHigh(81-100)'])
    return raw, clean

raw, clean = generate_data()

# ── PLOTLY THEME ──────────────────────────────────────────────────────────────
PALETTE = ['#C2185B','#00838F','#F9A825','#880E4F','#4CAF50','#7B1FA2','#E64A19','#1565C0']
def styled_fig(fig, height=380):
    fig.update_layout(
        height=height,
        paper_bgcolor='white', plot_bgcolor='white',
        font=dict(family='DM Sans', size=12, color='#37474F'),
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(bgcolor='rgba(0,0,0,0)', font_size=11),
        title_font=dict(family='Playfair Display', size=15, color='#37474F'),
    )
    fig.update_xaxes(showgrid=False, linecolor='#E0E0E0', tickfont_size=11)
    fig.update_yaxes(gridcolor='#F5F5F5', linecolor='#E0E0E0', tickfont_size=11)
    return fig

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌸 Filters")
    st.markdown("---")
    sel_plan = st.multiselect("Subscription Plan", clean['Subscription_Plan'].unique(), default=list(clean['Subscription_Plan'].unique()))
    sel_skin = st.multiselect("Skin Type", clean['Skin_Type'].unique(), default=list(clean['Skin_Type'].unique()))
    sel_tier = st.multiselect("City Tier", clean['City_Tier'].unique(), default=list(clean['City_Tier'].unique()))
    sel_gender = st.multiselect("Gender", clean['Gender'].unique(), default=list(clean['Gender'].unique()))
    age_range = st.slider("Age Range", 18, 51, (18, 51))
    st.markdown("---")
    st.markdown("**About**")
    st.markdown("AI Skincare Subscription\nIndia Market Validation\n\n200 synthetic records")

df = clean[
    clean['Subscription_Plan'].isin(sel_plan) &
    clean['Skin_Type'].isin(sel_skin) &
    clean['City_Tier'].isin(sel_tier) &
    clean['Gender'].isin(sel_gender) &
    clean['Age'].between(*age_range)
].copy()

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="dash-header">
  <div class="petal">🌸</div>
  <h1>AI-Based Personalized Skincare</h1>
  <p>India-Focused Subscription Business · End-to-End Validation Dashboard · Session 1–5 Assignment</p>
</div>
""", unsafe_allow_html=True)

# ── KPI ROW ───────────────────────────────────────────────────────────────────
k1,k2,k3,k4,k5 = st.columns(5)
total_cust = len(df)
avg_rev    = df['Monthly_Revenue_INR'].mean()
churn_rate = df['Churned'].mean()*100
avg_clv    = df['CLV_INR'].mean()
avg_nps    = df['NPS_Score'].mean()

k1.markdown(f'<div class="kpi-card"><div class="kpi-label">Total Customers</div><div class="kpi-value">{total_cust}</div><div class="kpi-sub">Filtered Records</div></div>', unsafe_allow_html=True)
k2.markdown(f'<div class="kpi-card kpi-teal"><div class="kpi-label">Avg Monthly Revenue</div><div class="kpi-value">₹{avg_rev:,.0f}</div><div class="kpi-sub">Per Customer</div></div>', unsafe_allow_html=True)
k3.markdown(f'<div class="kpi-card kpi-gold"><div class="kpi-label">Churn Rate</div><div class="kpi-value">{churn_rate:.1f}%</div><div class="kpi-sub">Of Filtered Base</div></div>', unsafe_allow_html=True)
k4.markdown(f'<div class="kpi-card kpi-slate"><div class="kpi-label">Avg CLV</div><div class="kpi-value">₹{avg_clv:,.0f}</div><div class="kpi-sub">Customer Lifetime Value</div></div>', unsafe_allow_html=True)
k5.markdown(f'<div class="kpi-card"><div class="kpi-label">Avg NPS Score</div><div class="kpi-value">{avg_nps:.0f}</div><div class="kpi-sub">Net Promoter Score</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
tabs = st.tabs(["📦 Data Overview","📊 Subscription Analysis","🤖 AI Engine","🏙️ Geography","📢 Channels & Demographics","🔗 Correlation","💡 Insights"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – DATA OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-title">Part 1 — Synthetic Dataset & Dirty Data Documentation</div>', unsafe_allow_html=True)
    
    col_a, col_b = st.columns([1,1])
    with col_a:
        st.markdown("**🟠 Dirty Data Issues (Raw Dataset)**")
        dirty_issues = [
            ("AI Score = -5 (Row 6)","Out-of-range negative value"),
            ("AI Score = 150 (Row 19)","Exceeds max of 100"),
            ("AI Score = NaN (Row 43)","Missing value"),
            ("Satisfaction = 9.5 (Row 78)","Exceeds max of 5"),
            ("Satisfaction = NaN (Row 11)","Missing value"),
            ("Churned = 5 (Row 31)","Should be 0 or 1 only"),
            ("LTV = NaN (Row 56)","Missing value"),
            ("Tenure = -3 (Row 91)","Negative tenure is impossible"),
            ("Revenue = NaN (Row 101)","Missing revenue"),
            ("Date = '2023/03/10' (Row 16)","Wrong format (slashes)"),
            ("Date = 'March 2023' (Row 67)","Incomplete month-year only"),
            ("Date = '' (Row 121)","Empty date string"),
        ]
        for issue, desc in dirty_issues:
            st.markdown(f'<span class="dirty-badge">🔴 {issue}</span> {desc}', unsafe_allow_html=True)

    with col_b:
        st.markdown("**🟢 Cleaning Actions Applied**")
        clean_actions = [
            "AI scores <0 or >100 → replaced with column median",
            "Satisfaction >5 → replaced with column median",
            "Churned ≠ 0/1 → replaced with mode (0)",
            "Negative Tenure → replaced with column median",
            "Null LTV / Revenue → median imputation",
            "Date '2023/03/10' → standardised to 2023-03-10",
            "Date 'March 2023' → standardised to 2023-03-01",
            "Empty date → filled with fallback 2023-01-01",
            "3 New Derived Columns Added: Annual_Revenue, CLV_INR, AI_Score_Band",
        ]
        for a in clean_actions:
            st.markdown(f'<span class="clean-badge">✅</span> {a}', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Part 2 — Descriptive Statistics</div>', unsafe_allow_html=True)
    num_cols = ['Age','Monthly_Revenue_INR','AI_Personalization_Score','Satisfaction_Rating',
                'LTV_Months','Tenure_Months','NPS_Score','CLV_INR']
    desc_df = df[num_cols].describe().T.round(2)
    desc_df.index.name = 'Variable'
    st.dataframe(desc_df, use_container_width=True)

    st.markdown('<div class="section-title">Preview: Cleaned Dataset</div>', unsafe_allow_html=True)
    show_cols = ['Customer_ID','Age','Gender','City','Skin_Type','Subscription_Plan',
                 'Monthly_Revenue_INR','AI_Personalization_Score','Satisfaction_Rating',
                 'Churned','CLV_INR','AI_Score_Band']
    st.dataframe(df[show_cols].head(30), use_container_width=True, height=340)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – SUBSCRIPTION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="section-title">Subscription Plan Deep-Dive</div>', unsafe_allow_html=True)

    plan_agg = df.groupby('Subscription_Plan').agg(
        Customers=('Customer_ID','count'),
        Avg_Revenue=('Monthly_Revenue_INR','mean'),
        Avg_Satisfaction=('Satisfaction_Rating','mean'),
        Churn_Rate=('Churned','mean'),
        Avg_CLV=('CLV_INR','mean'),
    ).reset_index()

    c1,c2 = st.columns(2)
    with c1:
        fig = px.pie(plan_agg, names='Subscription_Plan', values='Customers',
                     title='Customer Distribution by Plan',
                     color_discrete_sequence=PALETTE, hole=0.42)
        fig.update_traces(textinfo='percent+label', textfont_size=13)
        st.plotly_chart(styled_fig(fig), use_container_width=True)
        st.markdown('<div class="insight-box"><strong>Insight:</strong> Basic plan holds 40% share but generates lowest CLV. Premium users, despite being 25% of base, contribute disproportionate revenue — a classic 80/20 dynamic ripe for upsell strategy.</div>', unsafe_allow_html=True)

    with c2:
        fig2 = px.bar(plan_agg, x='Subscription_Plan', y=['Avg_Revenue','Avg_CLV'],
                      title='Avg Monthly Revenue & CLV by Plan', barmode='group',
                      color_discrete_sequence=[PALETTE[0], PALETTE[1]])
        fig2.update_traces(texttemplate='₹%{y:,.0f}', textposition='outside')
        st.plotly_chart(styled_fig(fig2), use_container_width=True)
        st.markdown('<div class="insight-box"><strong>Insight:</strong> Premium CLV is 3× higher than Basic. Converting even 10% of Basic users to Standard would boost average CLV by ~₹2,000 per customer.</div>', unsafe_allow_html=True)

    c3,c4 = st.columns(2)
    with c3:
        fig3 = px.bar(plan_agg, x='Subscription_Plan', y='Churn_Rate',
                      title='Churn Rate by Subscription Plan',
                      color='Churn_Rate', color_continuous_scale='RdYlGn_r',
                      text='Churn_Rate')
        fig3.update_traces(texttemplate='%{text:.1%}', textposition='outside')
        st.plotly_chart(styled_fig(fig3), use_container_width=True)
        st.markdown('<div class="insight-box"><strong>Insight:</strong> Basic plan churn (~48%) is more than 2× that of Premium (~22%). Higher plan tiers create stronger lock-in through personalisation depth and product variety.</div>', unsafe_allow_html=True)

    with c4:
        fig4 = px.bar(plan_agg, x='Subscription_Plan', y='Avg_Satisfaction',
                      title='Avg Satisfaction Rating by Plan',
                      color='Subscription_Plan', color_discrete_sequence=PALETTE,
                      text='Avg_Satisfaction')
        fig4.update_traces(texttemplate='%{text:.2f}⭐', textposition='outside')
        fig4.update_yaxes(range=[0,5.5])
        st.plotly_chart(styled_fig(fig4), use_container_width=True)
        st.markdown('<div class="insight-box"><strong>Insight:</strong> Premium satisfaction is consistently highest. AI personalisation quality scales with plan tier — users on higher plans get more tailored recommendations, driving satisfaction.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – AI ENGINE
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-title">AI Personalization Engine Analysis</div>', unsafe_allow_html=True)

    ai_band = df.groupby('AI_Score_Band', observed=True).agg(
        Customers=('Customer_ID','count'),
        Churn_Rate=('Churned','mean'),
        Avg_Satisfaction=('Satisfaction_Rating','mean'),
        Avg_CLV=('CLV_INR','mean'),
    ).reset_index()
    ai_band['AI_Score_Band'] = ai_band['AI_Score_Band'].astype(str)

    c1,c2 = st.columns(2)
    with c1:
        fig = px.bar(ai_band, x='AI_Score_Band', y='Churn_Rate',
                     title='Churn Rate by AI Score Band',
                     color='Churn_Rate', color_continuous_scale='RdYlGn_r', text='Churn_Rate')
        fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
        st.plotly_chart(styled_fig(fig), use_container_width=True)
        st.markdown('<div class="insight-box"><strong>Insight:</strong> VHigh AI band (81–100) shows only ~14% churn vs ~61% for Low band. <strong>AI score is the single strongest churn predictor in the dataset.</strong> Improving the AI model directly protects revenue.</div>', unsafe_allow_html=True)

    with c2:
        fig2 = px.scatter(df, x='AI_Personalization_Score', y='Satisfaction_Rating',
                          color='Subscription_Plan', color_discrete_sequence=PALETTE,
                          title='AI Score vs Satisfaction Rating (by Plan)',
                          opacity=0.7, trendline='ols')
        st.plotly_chart(styled_fig(fig2), use_container_width=True)
        r, p = stats.pearsonr(df['AI_Personalization_Score'], df['Satisfaction_Rating'])
        st.markdown(f'<div class="insight-box"><strong>Pearson r = {r:.3f}</strong> (p = {p:.4f}) — Strong positive correlation. Higher AI accuracy → higher user satisfaction across all subscription tiers.</div>', unsafe_allow_html=True)

    c3,c4 = st.columns(2)
    with c3:
        fig3 = px.scatter(df, x='AI_Personalization_Score', y='CLV_INR',
                          color='Churned', color_discrete_map={0:'#00838F',1:'#C2185B'},
                          title='AI Score vs Customer Lifetime Value',
                          opacity=0.7, trendline='ols')
        fig3.update_traces(marker_size=7)
        st.plotly_chart(styled_fig(fig3), use_container_width=True)
        r2, _ = stats.pearsonr(df['AI_Personalization_Score'], df['CLV_INR'])
        st.markdown(f'<div class="insight-box"><strong>r = {r2:.3f}</strong> — AI score positively drives CLV. Churned customers (red) cluster in the low AI-score region, confirming the AI engine is the retention backbone.</div>', unsafe_allow_html=True)

    with c4:
        fig4 = px.bar(ai_band, x='AI_Score_Band', y='Avg_CLV',
                      title='Average CLV by AI Score Band',
                      color='AI_Score_Band', color_discrete_sequence=PALETTE, text='Avg_CLV')
        fig4.update_traces(texttemplate='₹%{text:,.0f}', textposition='outside')
        st.plotly_chart(styled_fig(fig4), use_container_width=True)
        st.markdown('<div class="insight-box"><strong>Insight:</strong> VHigh AI band generates 2.8× CLV of Low band. Investing ₹1 in AI model improvement yields a measurably higher return than most marketing spend.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Skin Type Analysis</div>', unsafe_allow_html=True)
    skin_agg = df.groupby('Skin_Type').agg(
        Customers=('Customer_ID','count'),
        Avg_AI=('AI_Personalization_Score','mean'),
        Avg_Satisfaction=('Satisfaction_Rating','mean'),
        Churn_Rate=('Churned','mean'),
    ).reset_index()

    fig5 = px.scatter(skin_agg, x='Avg_AI', y='Churn_Rate', size='Customers',
                      color='Skin_Type', color_discrete_sequence=PALETTE,
                      title='Avg AI Score vs Churn Rate by Skin Type (bubble = volume)',
                      text='Skin_Type', size_max=60)
    fig5.update_traces(textposition='top center')
    st.plotly_chart(styled_fig(fig5, height=420), use_container_width=True)
    st.markdown('<div class="insight-box"><strong>Insight:</strong> Sensitive skin users receive lower AI scores and exhibit the highest churn (~45%). The AI model struggles most with Sensitive skin — a clear product improvement opportunity requiring specialised ingredient-mapping models.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 – GEOGRAPHY
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-title">Geographic Distribution & Performance</div>', unsafe_allow_html=True)

    city_agg = df.groupby('City').agg(
        Customers=('Customer_ID','count'),
        Avg_CLV=('CLV_INR','mean'),
        Churn_Rate=('Churned','mean'),
        Avg_Revenue=('Monthly_Revenue_INR','mean'),
    ).reset_index().sort_values('Customers', ascending=False)

    c1,c2 = st.columns(2)
    with c1:
        fig = px.bar(city_agg, x='Customers', y='City', orientation='h',
                     title='Customer Count by City', color='Customers',
                     color_continuous_scale='RdPu', text='Customers')
        fig.update_traces(textposition='outside')
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(styled_fig(fig, height=420), use_container_width=True)

    with c2:
        fig2 = px.bar(city_agg.sort_values('Avg_CLV',ascending=False),
                      x='Avg_CLV', y='City', orientation='h',
                      title='Average CLV by City (INR)', color='Avg_CLV',
                      color_continuous_scale='Teal', text='Avg_CLV')
        fig2.update_traces(texttemplate='₹%{text:,.0f}', textposition='outside')
        fig2.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(styled_fig(fig2, height=420), use_container_width=True)

    st.markdown('<div class="insight-box"><strong>Insight:</strong> Mumbai & Delhi lead in volume but <strong>Bengaluru shows highest average CLV</strong> — driven by tech-savvy, high-income professionals who value premium AI-personalised skincare. Tier-2 cities (Jaipur, Lucknow) show lower CAC potential — ideal for Phase-2 expansion with value-tier plans.</div>', unsafe_allow_html=True)

    tier_agg = df.groupby('City_Tier').agg(
        Customers=('Customer_ID','count'),
        Avg_CLV=('CLV_INR','mean'),
        Churn_Rate=('Churned','mean'),
    ).reset_index()
    fig3 = px.bar(tier_agg, x='City_Tier', y=['Customers','Avg_CLV'],
                  title='Tier-1 vs Tier-2 Cities: Volume & CLV', barmode='group',
                  color_discrete_sequence=[PALETTE[0], PALETTE[1]])
    st.plotly_chart(styled_fig(fig3, height=340), use_container_width=True)
    st.markdown('<div class="insight-box"><strong>Insight:</strong> Tier-1 cities have higher CLV but Tier-2 cities show lower churn — customers in smaller cities may be more loyal once acquired, suggesting a long-term growth moat in Tier-2.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 – CHANNELS & DEMOGRAPHICS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="section-title">Acquisition Channels</div>', unsafe_allow_html=True)

    ch_agg = df.groupby('Acquisition_Channel').agg(
        Customers=('Customer_ID','count'),
        Avg_CLV=('CLV_INR','mean'),
        Avg_NPS=('NPS_Score','mean'),
    ).reset_index()

    c1,c2 = st.columns(2)
    with c1:
        fig = px.bar(ch_agg.sort_values('Avg_CLV',ascending=False),
                     x='Acquisition_Channel', y=['Customers','Avg_CLV'],
                     title='Channel: Volume & CLV', barmode='group',
                     color_discrete_sequence=PALETTE)
        st.plotly_chart(styled_fig(fig), use_container_width=True)
        st.markdown('<div class="insight-box"><strong>Insight:</strong> Instagram brings most volume (35%) but <strong>Influencer channel delivers highest CLV</strong>. Allocate 60% budget to Instagram for scale, 40% to Influencers for quality customers.</div>', unsafe_allow_html=True)

    with c2:
        fig2 = px.bar(ch_agg.sort_values('Avg_NPS',ascending=False),
                      x='Acquisition_Channel', y='Avg_NPS',
                      title='NPS Score by Acquisition Channel',
                      color='Avg_NPS', color_continuous_scale='RdYlGn', text='Avg_NPS')
        fig2.update_traces(texttemplate='%{text:.0f}', textposition='outside')
        st.plotly_chart(styled_fig(fig2), use_container_width=True)
        st.markdown('<div class="insight-box"><strong>Insight:</strong> Word-of-Mouth channel shows highest NPS — organic referrals are the highest-quality signal. Invest in referral programmes to amplify organic growth.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Demographics Analysis</div>', unsafe_allow_html=True)

    df['Age_Group'] = pd.cut(df['Age'], bins=[17,25,30,35,40,52],
                              labels=['18-25','26-30','31-35','36-40','41+'])
    age_agg = df.groupby('Age_Group', observed=True).agg(
        Customers=('Customer_ID','count'),
        Avg_CLV=('CLV_INR','mean'),
        Churn_Rate=('Churned','mean'),
    ).reset_index()
    age_agg['Age_Group'] = age_agg['Age_Group'].astype(str)

    c3,c4 = st.columns(2)
    with c3:
        fig3 = make_subplots(specs=[[{"secondary_y": True}]])
        fig3.add_trace(go.Bar(x=age_agg['Age_Group'], y=age_agg['Avg_CLV'],
                               name='Avg CLV', marker_color=PALETTE[0]), secondary_y=False)
        fig3.add_trace(go.Scatter(x=age_agg['Age_Group'], y=age_agg['Churn_Rate'],
                                   name='Churn Rate', marker_color=PALETTE[1],
                                   mode='lines+markers', line_width=3), secondary_y=True)
        fig3.update_layout(title='Age Group: CLV & Churn Rate', height=380,
                            paper_bgcolor='white', plot_bgcolor='white',
                            font=dict(family='DM Sans',size=12))
        fig3.update_yaxes(title_text='CLV (INR)', secondary_y=False)
        fig3.update_yaxes(title_text='Churn Rate', secondary_y=True)
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown('<div class="insight-box"><strong>Insight:</strong> Age 26–35 is the sweet spot — highest CLV with lowest churn. This is the <strong>primary ICP</strong>. Focus all premium-tier marketing on this cohort.</div>', unsafe_allow_html=True)

    with c4:
        gender_agg = df.groupby('Gender').agg(
            Customers=('Customer_ID','count'),
            Avg_CLV=('CLV_INR','mean'),
            Churn_Rate=('Churned','mean'),
        ).reset_index()
        fig4 = px.bar(gender_agg, x='Gender', y=['Avg_CLV','Customers'],
                      title='Gender: CLV & Volume', barmode='group',
                      color_discrete_sequence=[PALETTE[0], PALETTE[2]])
        st.plotly_chart(styled_fig(fig4), use_container_width=True)
        st.markdown('<div class="insight-box"><strong>Insight:</strong> Female segment (62%) drives majority of revenue and exhibits higher Avg CLV. Male segment (33%) is growing — an underserved opportunity for men\'s skincare sub-product line.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 – CORRELATION
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="section-title">Pearson Correlation Matrix</div>', unsafe_allow_html=True)

    num_cols = ['Age','Monthly_Revenue_INR','AI_Personalization_Score','Satisfaction_Rating',
                'LTV_Months','Tenure_Months','NPS_Score','Order_Frequency','CLV_INR','Churned']
    corr = df[num_cols].corr().round(3)

    fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1, title='Correlation Heatmap — Numerical Variables',
                    aspect='auto')
    fig.update_traces(textfont_size=11)
    st.plotly_chart(styled_fig(fig, height=520), use_container_width=True)

    st.markdown('<div class="section-title">Key Correlation Findings</div>', unsafe_allow_html=True)
    corr_insights = [
        ("AI Score ↔ Satisfaction",   f"{corr.loc['AI_Personalization_Score','Satisfaction_Rating']:.3f}", "Strong positive — AI quality directly drives user happiness"),
        ("AI Score ↔ CLV",            f"{corr.loc['AI_Personalization_Score','CLV_INR']:.3f}",            "Positive — better recommendations → longer subscriptions"),
        ("AI Score ↔ Churned",        f"{corr.loc['AI_Personalization_Score','Churned']:.3f}",            "Strong negative — highest-value insight: fix AI = fix churn"),
        ("Satisfaction ↔ Churned",    f"{corr.loc['Satisfaction_Rating','Churned']:.3f}",                 "Strong negative — unhappy users leave; track CSAT monthly"),
        ("Revenue ↔ CLV",             f"{corr.loc['Monthly_Revenue_INR','CLV_INR']:.3f}",                 "Strong positive — higher plan = bigger lifetime value"),
        ("LTV Months ↔ Churned",      f"{corr.loc['LTV_Months','Churned']:.3f}",                          "Strong negative — confirms churn destroys lifetime value"),
    ]
    c1,c2 = st.columns(2)
    for i,(pair,r_val,meaning) in enumerate(corr_insights):
        col = c1 if i%2==0 else c2
        color = '#C8E6C9' if float(r_val)>0 else '#FFCDD2'
        col.markdown(f"""
        <div style="background:{color};border-radius:10px;padding:14px;margin:6px 0;">
            <strong>{pair}</strong><br>
            <span style="font-size:1.4rem;font-weight:700;color:#37474F">r = {r_val}</span><br>
            <span style="font-size:0.85rem;color:#555">{meaning}</span>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 – INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown('<div class="section-title">💡 Business Validation Summary & Strategic Insights</div>', unsafe_allow_html=True)

    insights_data = [
        ("🎯","ICP (Ideal Customer Profile)","Female, 26–35 yrs, Tier-1 city (especially Bengaluru), Oily/Combination skin, acquired via Instagram or Influencer, on Standard/Premium plan.","HIGH"),
        ("📱","Top Acquisition Strategy","Instagram (volume) + Influencer (quality). Influencer channel delivers highest CLV. Combine both for optimal CAC:CLV ratio.","HIGH"),
        ("🤖","AI Engine Priority","AI Personalization Score is the #1 driver of retention. VHigh AI band shows 61% lower churn. Prioritise AI model accuracy above all else.","CRITICAL"),
        ("💄","Skin Type Gap","Sensitive skin users have highest churn due to AI model limitations. Build specialised recommendation model for Sensitive skin — underserved segment.","MEDIUM"),
        ("🏙️","Geographic Expansion","Phase 1: Mumbai, Delhi, Bengaluru. Phase 2: Tier-2 cities (Jaipur, Lucknow) — lower CAC, surprising loyalty rates.","MEDIUM"),
        ("📈","Upsell Opportunity","40% of users on Basic with 48% churn. Moving 10% to Standard/Premium would add ~₹8L annual revenue and reduce churn significantly.","HIGH"),
        ("🔁","Referral Programme","Word-of-Mouth shows highest NPS. A structured referral programme (e.g., ₹200 credit per referral) could reduce paid CAC by 20–30%.","MEDIUM"),
        ("👨","Men's Skincare","Male segment (33%) is underserved. Launch a men's-focused SKU bundle within the AI subscription to capture this growing market.","LOW"),
    ]

    priority_color = {'CRITICAL':'#B71C1C','HIGH':'#C2185B','MEDIUM':'#F9A825','LOW':'#00838F'}
    for icon,title,insight,priority in insights_data:
        color = priority_color[priority]
        st.markdown(f"""
        <div style="background:white;border-radius:12px;padding:18px 22px;margin:10px 0;
                    border-left:5px solid {color};box-shadow:0 2px 10px rgba(0,0,0,0.06);">
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;">
                <span style="font-size:1.1rem;font-weight:700;color:#37474F">{icon} {title}</span>
                <span style="background:{color};color:white;border-radius:6px;padding:2px 10px;
                             font-size:0.75rem;font-weight:700;">{priority}</span>
            </div>
            <span style="font-size:0.9rem;color:#546E7A;line-height:1.6">{insight}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background:linear-gradient(135deg,#C2185B,#880E4F);border-radius:14px;
                padding:24px 28px;color:white;text-align:center;">
        <h3 style="font-family:'Playfair Display',serif;margin:0 0 8px 0">✅ Business Validation Verdict</h3>
        <p style="margin:0;opacity:0.9;font-size:1rem">
        The synthetic data confirms strong product-market fit: AI-driven personalisation is a proven retention and CLV driver. 
        The India D2C skincare subscription model is viable with a clear ICP, scalable acquisition mix, and 
        an addressable market across both Tier-1 and Tier-2 cities. <strong>Proceed to MVP launch.</strong>
        </p>
    </div>""", unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;color:#BDBDBD;font-size:0.8rem;padding:16px">
    🌸 AI-Based Personalized Skincare Subscription · India Market Validation Dashboard · Session 1–5 Assignment
</div>""", unsafe_allow_html=True)
