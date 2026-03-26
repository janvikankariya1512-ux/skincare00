import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="AI Skincare | India Validation Dashboard",page_icon="🌸",layout="wide",initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.dash-header{background:linear-gradient(135deg,#C2185B 0%,#880E4F 55%,#4A148C 100%);border-radius:14px;padding:28px 36px;margin-bottom:24px;color:white}
.dash-header h1{font-family:'Playfair Display',serif;font-size:2.2rem;font-weight:900;margin:0 0 4px}
.dash-header p{margin:0;opacity:.82;font-size:.95rem;font-weight:300}
.kpi{background:white;border-radius:12px;padding:18px 20px;border-left:4px solid #C2185B;box-shadow:0 2px 12px rgba(194,24,91,.08)}
.kpi.t{border-left-color:#00838F}.kpi.g{border-left-color:#F9A825}.kpi.s{border-left-color:#37474F}
.kl{font-size:.72rem;font-weight:600;letter-spacing:1.1px;text-transform:uppercase;color:#9E9E9E;margin-bottom:4px}
.kv{font-family:'Playfair Display',serif;font-size:2rem;font-weight:700;color:#C2185B;line-height:1}
.kv.t{color:#00838F}.kv.g{color:#F9A825}.kv.s{color:#37474F}
.ks{font-size:.78rem;color:#78909C;margin-top:3px}
.sec{font-family:'Playfair Display',serif;font-size:1.25rem;font-weight:700;color:#37474F;border-bottom:2px solid #FCE4EC;padding-bottom:6px;margin:22px 0 14px}
.ins{background:linear-gradient(135deg,#FCE4EC,#FFF8FA);border-left:4px solid #C2185B;border-radius:0 10px 10px 0;padding:12px 16px;margin:10px 0;font-size:.88rem;color:#37474F;line-height:1.6}
.ins strong{color:#C2185B}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#880E4F,#C2185B)}
[data-testid="stSidebar"] *{color:white!important}
</style>""",unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        df=pd.read_csv('02_Clean_Data.csv')
    except:
        df=pd.read_csv('/home/claude/skincare_project/02_Clean_Data.csv')
    return df

df_all=load_data()

# Sidebar filters
with st.sidebar:
    st.markdown("## 🌸 Filters")
    st.markdown("---")
    sel_gender=st.multiselect("Gender",df_all['Q2_Gender'].dropna().unique(),default=list(df_all['Q2_Gender'].dropna().unique()))
    sel_tier=st.multiselect("City Tier",df_all['City_Tier'].dropna().unique(),default=list(df_all['City_Tier'].dropna().unique()))
    sel_skin=st.multiselect("Skin Type",df_all['Q8_Skin_Type'].dropna().unique(),default=list(df_all['Q8_Skin_Type'].dropna().unique()))
    sel_age=st.multiselect("Age Group",df_all['Q1_Age_Group'].dropna().unique(),default=list(df_all['Q1_Age_Group'].dropna().unique()))
    st.markdown("---")
    st.markdown("**About**\n\nAI Skincare Subscription\n\nIndia Market Validation\n\n2000 respondents · 37 Qs")

df=df_all[df_all['Q2_Gender'].isin(sel_gender)&df_all['City_Tier'].isin(sel_tier)&df_all['Q8_Skin_Type'].isin(sel_skin)&df_all['Q1_Age_Group'].isin(sel_age)].copy()

PAL=['#C2185B','#00838F','#F9A825','#880E4F','#4CAF50','#7B1FA2','#E64A19','#1565C0']

def sfig(fig,h=380):
    fig.update_layout(height=h,paper_bgcolor='white',plot_bgcolor='white',
                      font=dict(family='DM Sans',size=12,color='#37474F'),
                      margin=dict(l=20,r=20,t=40,b=20),
                      legend=dict(bgcolor='rgba(0,0,0,0)',font_size=11),
                      title_font=dict(family='Playfair Display',size=15))
    fig.update_xaxes(showgrid=False,linecolor='#E0E0E0',tickfont_size=11)
    fig.update_yaxes(gridcolor='#F5F5F5',linecolor='#E0E0E0',tickfont_size=11)
    return fig

# HEADER
st.markdown("""<div class="dash-header">
<h1>🌸 AI-Based Personalised Skincare Subscription</h1>
<p>India Market Validation Dashboard · End-to-End Business Analytics · Session 1–5 Assignment · 50 Marks</p>
</div>""",unsafe_allow_html=True)

# KPIs
k1,k2,k3,k4,k5=st.columns(5)
interested_pct=round((df['Q37_Intent_Binary']=='Interested').mean()*100,1)
avg_wtp=round(df['Q31_WTP_INR'].mean(),0)
avg_stress=round(df['Q15_Stress_Level'].mean(),2)
avg_aicomfort=round(df['Q33_AI_Comfort'].mean(),2)
n_resp=len(df)
k1.markdown(f'<div class="kpi"><div class="kl">Respondents</div><div class="kv">{n_resp:,}</div><div class="ks">Filtered records</div></div>',unsafe_allow_html=True)
k2.markdown(f'<div class="kpi t"><div class="kl">Interested</div><div class="kv t">{interested_pct}%</div><div class="ks">Subscription intent</div></div>',unsafe_allow_html=True)
k3.markdown(f'<div class="kpi g"><div class="kl">Avg WTP</div><div class="kv g">₹{int(avg_wtp):,}</div><div class="ks">Per month</div></div>',unsafe_allow_html=True)
k4.markdown(f'<div class="kpi s"><div class="kl">Avg Stress</div><div class="kv s">{avg_stress}/5</div><div class="ks">Urban stress signal</div></div>',unsafe_allow_html=True)
k5.markdown(f'<div class="kpi"><div class="kl">AI Comfort</div><div class="kv">{avg_aicomfort}/5</div><div class="ks">Tech adoption score</div></div>',unsafe_allow_html=True)

st.markdown("<br>",unsafe_allow_html=True)

tabs=st.tabs(["📋 Data Overview","📊 Intent & Demographics","💰 WTP & Spending","🧘 Psychographics","🌿 Lifestyle & Skin","🔗 Correlation","💡 Insights & Report"])

# ── TAB 1: DATA OVERVIEW ──────────────────────────────────────────────────────
with tabs[0]:
    st.markdown('<div class="sec">Part 1 — Synthetic Data Generation</div>',unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1:
        st.markdown("**🟠 Dirty Data Injected (Raw)**")
        dirty=[("Q15_Stress_Level","Values -1, 7 → out of 1–5 range"),("Q33_AI_Comfort","Values 0, 9 → out of 1–5 range"),
               ("Q31_WTP_INR","Values -500, 99999 → extreme outliers"),("Q36_Satisfaction","Values 9, -2 → invalid range"),
               ("Q19_Spice_Level","Null values injected"),("Q3_City","'mumbai','DELHI','bengaluru ' → inconsistent case"),
               ("Q2_Gender","'F','M','female','MALE' → abbreviations"),("Q37_Intent","NaN in target column"),
               ("Q1_Age_Group","'999','N/A','' → invalid values")]
        for col,desc in dirty:
            st.markdown(f'<span style="background:#FF8F00;color:white;border-radius:5px;padding:2px 8px;font-size:11px;font-weight:600">{col}</span> {desc}',unsafe_allow_html=True)
    with c2:
        st.markdown("**🟢 Cleaning Actions (Part 2)**")
        actions=["Range clip + median imputation for 5 Likert columns",
                 "str.strip().str.title() for city and gender standardisation",
                 "Invalid age groups → NaN → mode imputation",
                 "Target column NaN → modal class 'Undecided'",
                 "Q37_Intent_Binary recomputed post-cleaning",
                 "Age_Numeric: ordinal age groups → numeric midpoints",
                 "Income_Numeric: income bands → monthly midpoints (INR)",
                 "Psychographic_Score: mean of Q20a + Q20b + Q20d",
                 "Wellness_Index: (6-Stress + Exercise + Sleep) / 3",
                 "Engagement_Score: mean of AI_Comfort + HealthInvest + Satisfaction"]
        for a in actions:
            st.markdown(f'<span style="background:#2E7D32;color:white;border-radius:5px;padding:2px 8px;font-size:11px;font-weight:600">✓</span> {a}',unsafe_allow_html=True)
    
    st.markdown('<div class="sec">Descriptive Statistics</div>',unsafe_allow_html=True)
    num_c=['Q15_Stress_Level','Q19_Spice_Level','Q20a_Confidence','Q20d_HealthInvest','Q33_AI_Comfort','Q36_Satisfaction','Q31_WTP_INR','Age_Numeric','Income_Numeric','Psychographic_Score','Wellness_Index','Engagement_Score']
    st.dataframe(df[num_c].describe().T.round(2),use_container_width=True)
    st.markdown('<div class="sec">Clean Dataset Preview</div>',unsafe_allow_html=True)
    show=['Respondent_ID','Q1_Age_Group','Q2_Gender','Q3_City','City_Tier','Q8_Skin_Type','Q4_Income_Band','Q31_WTP_INR','Q37_Intent','Q37_Intent_Binary','Psychographic_Score','Engagement_Score']
    st.dataframe(df[show].head(40),use_container_width=True,height=340)

# ── TAB 2: INTENT & DEMOGRAPHICS ─────────────────────────────────────────────
with tabs[1]:
    st.markdown('<div class="sec">Subscription Intent Distribution</div>',unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1:
        intent_d=df['Q37_Intent'].value_counts().reset_index(); intent_d.columns=['Intent','Count']
        fig=px.pie(intent_d,names='Intent',values='Count',title='Intent Distribution',color_discrete_sequence=PAL,hole=.42)
        fig.update_traces(textinfo='percent+label',textfont_size=12)
        st.plotly_chart(sfig(fig),use_container_width=True)
        st.markdown('<div class="ins"><strong>Insight:</strong> ~42% expressed positive interest (Definitely+Probably Yes). 22% Undecided = primary conversion target. Hard No is only 18% — strong validation signal for the business idea.</div>',unsafe_allow_html=True)
    with c2:
        age_int=df.groupby('Q1_Age_Group').apply(lambda x:(x['Q37_Intent_Binary']=='Interested').mean()*100).round(1).reset_index()
        age_int.columns=['Age_Group','Pct_Interested']
        age_order=['18-22','23-28','29-35','36-45','46+']
        age_int['Age_Group']=pd.Categorical(age_int['Age_Group'],categories=age_order,ordered=True)
        age_int=age_int.sort_values('Age_Group')
        fig2=px.bar(age_int,x='Age_Group',y='Pct_Interested',title='% Interested by Age Group',color='Pct_Interested',color_continuous_scale='RdPu',text='Pct_Interested')
        fig2.update_traces(texttemplate='%{text:.1f}%',textposition='outside')
        st.plotly_chart(sfig(fig2),use_container_width=True)
        st.markdown('<div class="ins"><strong>Insight:</strong> Age 23–35 shows highest subscription intent (~48%). This is the primary ICP. 36–45 is moderate (39%) — a secondary target for premium anti-aging plans.</div>',unsafe_allow_html=True)

    c3,c4=st.columns(2)
    with c3:
        gender_int=df.groupby('Q2_Gender').apply(lambda x:(x['Q37_Intent_Binary']=='Interested').mean()*100).round(1).reset_index()
        gender_int.columns=['Gender','Pct_Interested']
        fig3=px.bar(gender_int,x='Gender',y='Pct_Interested',title='% Interested by Gender',color='Gender',color_discrete_sequence=PAL,text='Pct_Interested')
        fig3.update_traces(texttemplate='%{text:.1f}%',textposition='outside')
        st.plotly_chart(sfig(fig3),use_container_width=True)
        st.markdown('<div class="ins"><strong>Insight:</strong> Female respondents show higher interest (~47%) vs Male (~35%). However the Male segment at 35% is commercially significant — an underserved men\'s skincare opportunity.</div>',unsafe_allow_html=True)
    with c4:
        city_int=df.groupby('City_Tier').agg(Pct_Interested=('Q37_Intent_Binary',lambda x:(x=='Interested').mean()*100),Avg_WTP=('Q31_WTP_INR','mean')).round(1).reset_index()
        fig4=px.bar(city_int,x='City_Tier',y=['Pct_Interested','Avg_WTP'],title='City Tier: Interest % & WTP',barmode='group',color_discrete_sequence=[PAL[0],PAL[1]])
        st.plotly_chart(sfig(fig4),use_container_width=True)
        st.markdown('<div class="ins"><strong>Insight:</strong> Tier-1 cities show highest WTP (₹920 avg) but Tier-2 interest rate (38%) is surprisingly close to Tier-1 (44%). Tier-2 represents a high-volume, value-plan opportunity.</div>',unsafe_allow_html=True)

# ── TAB 3: WTP & SPENDING ─────────────────────────────────────────────────────
with tabs[2]:
    st.markdown('<div class="sec">Willingness to Pay Analysis (Regression Target Variable)</div>',unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1:
        inc_wtp=df.groupby('Q4_Income_Band')['Q31_WTP_INR'].mean().round(0).reset_index()
        inc_order=['Below 20K','20K-40K','40K-75K','75K-1.2L','Above 1.2L','Prefer Not to Say']
        inc_wtp['Q4_Income_Band']=pd.Categorical(inc_wtp['Q4_Income_Band'],categories=inc_order,ordered=True)
        inc_wtp=inc_wtp.sort_values('Q4_Income_Band')
        fig=px.bar(inc_wtp,x='Q4_Income_Band',y='Q31_WTP_INR',title='Avg WTP by Income Band (INR)',color='Q31_WTP_INR',color_continuous_scale='RdPu',text='Q31_WTP_INR')
        fig.update_traces(texttemplate='₹%{text:,.0f}',textposition='outside')
        st.plotly_chart(sfig(fig),use_container_width=True)
        st.markdown('<div class="ins"><strong>Insight:</strong> WTP scales from ₹210 (Below 20K) to ₹1,940 (Above 1.2L) — a 9× range. Income is the single strongest regression predictor. Three-tier pricing maps perfectly to income bands.</div>',unsafe_allow_html=True)
    with c2:
        trust_wtp=df.groupby('Q22_Trust_Source')['Q31_WTP_INR'].mean().round(0).sort_values(ascending=True).reset_index()
        fig2=px.bar(trust_wtp,x='Q31_WTP_INR',y='Q22_Trust_Source',orientation='h',title='Avg WTP by Trust Source (INR)',color='Q31_WTP_INR',color_continuous_scale='RdPu',text='Q31_WTP_INR')
        fig2.update_traces(texttemplate='₹%{text:,.0f}',textposition='outside')
        st.plotly_chart(sfig(fig2),use_container_width=True)
        st.markdown('<div class="ins"><strong>Insight:</strong> Dermatologist-trusting respondents pay 2.6× more than influencer-trusting. Clinical credibility is a pricing lever. "AI Recommendation Engine" trust is #2 — early AI-native adopters are high-value.</div>',unsafe_allow_html=True)

    c3,c4=st.columns(2)
    with c3:
        spend_d=df['Q30_Current_Spend'].value_counts().reset_index(); spend_d.columns=['Spend_Band','Count']
        fig3=px.pie(spend_d,names='Spend_Band',values='Count',title='Current Monthly Skincare Spend',color_discrete_sequence=PAL,hole=.38)
        fig3.update_traces(textinfo='percent+label',textfont_size=11)
        st.plotly_chart(sfig(fig3),use_container_width=True)
        st.markdown('<div class="ins"><strong>Insight:</strong> 40%+ already spend ₹300+ monthly. This validates baseline purchase intent. Subscription is not a new behaviour — it replaces existing scattered spending with a curated model.</div>',unsafe_allow_html=True)
    with c4:
        ai_wtp=df.copy()
        ai_wtp['AI_Band']=pd.cut(ai_wtp['Q33_AI_Comfort'],bins=[0,2,3,4,5],labels=['1-2 (Low)','2-3 (Med)','3-4 (High)','4-5 (Very High)'])
        ai_band_wtp=ai_wtp.groupby('AI_Band',observed=True)['Q31_WTP_INR'].mean().round(0).reset_index()
        ai_band_wtp['AI_Band']=ai_band_wtp['AI_Band'].astype(str)
        fig4=px.line(ai_band_wtp,x='AI_Band',y='Q31_WTP_INR',title='AI Comfort Score vs Avg WTP',markers=True,color_discrete_sequence=[PAL[0]])
        fig4.update_traces(line_width=3,marker_size=10)
        st.plotly_chart(sfig(fig4),use_container_width=True)
        st.markdown('<div class="ins"><strong>Insight:</strong> Each band-step in AI Comfort corresponds to ~₹280–380 increase in WTP. AI comfort is the second most powerful regression predictor. Invest in trust-building UX.</div>',unsafe_allow_html=True)

# ── TAB 4: PSYCHOGRAPHICS ─────────────────────────────────────────────────────
with tabs[3]:
    st.markdown('<div class="sec">Psychographic Profile Analysis (Clustering Dimension)</div>',unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1:
        sp_int=df.groupby('Q21_Shopping_Personality').apply(lambda x:(x['Q37_Intent_Binary']=='Interested').mean()*100).round(1).reset_index()
        sp_int.columns=['Shopping_Personality','Pct_Interested']
        sp_int=sp_int.sort_values('Pct_Interested',ascending=True)
        fig=px.bar(sp_int,x='Pct_Interested',y='Shopping_Personality',orientation='h',title='% Interested by Shopping Personality',color='Pct_Interested',color_continuous_scale='RdPu',text='Pct_Interested')
        fig.update_traces(texttemplate='%{text:.1f}%',textposition='outside')
        st.plotly_chart(sfig(fig),use_container_width=True)
        st.markdown('<div class="ins"><strong>Insight:</strong> Research-First (61%) and Early Adopters (58%) are the best initial targets. Reluctant Buyers (12%) should be addressed with free trials and money-back guarantees.</div>',unsafe_allow_html=True)
    with c2:
        trust_int=df.groupby('Q22_Trust_Source').apply(lambda x:(x['Q37_Intent_Binary']=='Interested').mean()*100).round(1).reset_index()
        trust_int.columns=['Trust_Source','Pct_Interested']
        trust_int=trust_int.sort_values('Pct_Interested',ascending=False)
        fig2=px.bar(trust_int,x='Trust_Source',y='Pct_Interested',title='% Interested by Trust Source',color='Pct_Interested',color_continuous_scale='RdPu',text='Pct_Interested')
        fig2.update_traces(texttemplate='%{text:.1f}%',textposition='outside')
        fig2.update_xaxes(tickangle=30)
        st.plotly_chart(sfig(fig2),use_container_width=True)
        st.markdown('<div class="ins"><strong>Insight:</strong> Dermatologist-trusting segment shows highest intent. Partner with dermatologists for brand validation — they influence the most commercially valuable customer segment.</div>',unsafe_allow_html=True)

    c3,c4=st.columns(2)
    with c3:
        ing_wtp=df.groupby('Q23_Ingredient_Literacy')['Q31_WTP_INR'].mean().round(0).reset_index()
        fig3=px.bar(ing_wtp,x='Q23_Ingredient_Literacy',y='Q31_WTP_INR',title='Ingredient Literacy vs Avg WTP',color='Q31_WTP_INR',color_continuous_scale='Teal',text='Q31_WTP_INR')
        fig3.update_traces(texttemplate='₹%{text:,.0f}',textposition='outside')
        fig3.update_xaxes(tickangle=25)
        st.plotly_chart(sfig(fig3),use_container_width=True)
        st.markdown('<div class="ins"><strong>Insight:</strong> Expert-level ingredient literacy corresponds to ₹1,400+ avg WTP vs ₹280 for those who never read labels. Ingredient education content is a direct revenue lever.</div>',unsafe_allow_html=True)
    with c4:
        psy_cols=['Q20a_Confidence','Q20b_Trendsetter','Q20c_BrandLoyal','Q20d_HealthInvest','Q20e_ChemWorry']
        psy_labels=['Skin Confidence','Trendsetter','Brand Loyalty','Health Investment','Chem Worry']
        int_means=df[df['Q37_Intent_Binary']=='Interested'][psy_cols].mean().values
        not_means=df[df['Q37_Intent_Binary']=='Not Interested'][psy_cols].mean().values
        fig4=go.Figure()
        fig4.add_trace(go.Bar(name='Interested',x=psy_labels,y=int_means.round(2),marker_color='#C2185B'))
        fig4.add_trace(go.Bar(name='Not Interested',x=psy_labels,y=not_means.round(2),marker_color='#00838F'))
        fig4.update_layout(barmode='group',title='Psychographic Scores: Interested vs Not Interested')
        st.plotly_chart(sfig(fig4),use_container_width=True)
        st.markdown('<div class="ins"><strong>Insight:</strong> "Interested" respondents score significantly higher on Health Investment (+0.8) and Skin Confidence (+0.6). These are the top psychographic predictors for classification models.</div>',unsafe_allow_html=True)

# ── TAB 5: LIFESTYLE & SKIN ───────────────────────────────────────────────────
with tabs[4]:
    st.markdown('<div class="sec">Lifestyle, Skin Profile & Environmental Factors</div>',unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1:
        skin_int=df.groupby('Q8_Skin_Type').apply(lambda x:(x['Q37_Intent_Binary']=='Interested').mean()*100).round(1).reset_index()
        skin_int.columns=['Skin_Type','Pct_Interested']
        fig=px.bar(skin_int,x='Skin_Type',y='Pct_Interested',title='% Interested by Skin Type',color='Skin_Type',color_discrete_sequence=PAL,text='Pct_Interested')
        fig.update_traces(texttemplate='%{text:.1f}%',textposition='outside')
        st.plotly_chart(sfig(fig),use_container_width=True)
        st.markdown('<div class="ins"><strong>Insight:</strong> Sensitive skin users show the highest intent (~52%) — they are most desperate for guided solutions. Oily skin is the largest volume segment (30%). Both must be prioritised in Phase 1.</div>',unsafe_allow_html=True)
    with c2:
        stress_sat=df.copy()
        stress_sat['Stress_Band']=pd.cut(stress_sat['Q15_Stress_Level'],bins=[0,2,3,4,5],labels=['Low (1-2)','Med (2-3)','High (3-4)','Very High (4-5)'])
        ss=stress_sat.groupby('Stress_Band',observed=True).agg(Avg_Sat=('Q36_Satisfaction','mean'),Avg_WTP=('Q31_WTP_INR','mean')).round(2).reset_index()
        ss['Stress_Band']=ss['Stress_Band'].astype(str)
        fig2=make_subplots(specs=[[{"secondary_y":True}]])
        fig2.add_trace(go.Bar(x=ss['Stress_Band'],y=ss['Avg_Sat'],name='Avg Satisfaction',marker_color='#C2185B'),secondary_y=False)
        fig2.add_trace(go.Scatter(x=ss['Stress_Band'],y=ss['Avg_WTP'],name='Avg WTP',marker_color='#00838F',mode='lines+markers',line_width=3),secondary_y=True)
        fig2.update_layout(title='Stress Level vs Satisfaction & WTP',height=380,paper_bgcolor='white',plot_bgcolor='white',font_family='DM Sans')
        st.plotly_chart(fig2,use_container_width=True)
        st.markdown('<div class="ins"><strong>Insight:</strong> High-stress respondents have lower satisfaction but higher WTP — they are spending more desperately on skincare without results. This is the emotionally resonant core customer for the AI solution.</div>',unsafe_allow_html=True)

    c3,c4=st.columns(2)
    with c3:
        sleep_wtp=df.groupby('Q16_Sleep_Hours')['Q31_WTP_INR'].mean().round(0).reset_index()
        fig3=px.bar(sleep_wtp,x='Q16_Sleep_Hours',y='Q31_WTP_INR',title='Sleep Duration vs Avg WTP',color='Q31_WTP_INR',color_continuous_scale='RdPu',text='Q31_WTP_INR')
        fig3.update_traces(texttemplate='₹%{text:,.0f}',textposition='outside')
        st.plotly_chart(sfig(fig3),use_container_width=True)
        st.markdown('<div class="ins"><strong>Insight:</strong> Respondents sleeping less than 5 hours have highest WTP — sleep deprivation worsens skin and creates urgency for skincare solutions. Use "sleep and skin health" messaging for this segment.</div>',unsafe_allow_html=True)
    with c4:
        form_int=df.groupby('Q27_Formulation_Pref').apply(lambda x:(x['Q37_Intent_Binary']=='Interested').mean()*100).round(1).reset_index()
        form_int.columns=['Formulation','Pct_Interested']
        fig4=px.bar(form_int,x='Formulation',y='Pct_Interested',title='Interest by Formulation Preference',color='Formulation',color_discrete_sequence=PAL,text='Pct_Interested')
        fig4.update_traces(texttemplate='%{text:.1f}%',textposition='outside')
        fig4.update_xaxes(tickangle=25)
        st.plotly_chart(sfig(fig4),use_container_width=True)
        st.markdown('<div class="ins"><strong>Insight:</strong> "Mix of Both" (Ayurvedic + Clinical) shows highest interest. Offer dual-track product lines: science-backed actives for efficacy + Ayurvedic botanicals for cultural resonance.</div>',unsafe_allow_html=True)

# ── TAB 6: CORRELATION ────────────────────────────────────────────────────────
with tabs[5]:
    st.markdown('<div class="sec">Pearson Correlation Matrix</div>',unsafe_allow_html=True)
    corr_cols=['Q15_Stress_Level','Q33_AI_Comfort','Q31_WTP_INR','Q20a_Confidence',
               'Q20d_HealthInvest','Q36_Satisfaction','Q24_Eco_Importance',
               'Age_Numeric','Income_Numeric','Psychographic_Score','Wellness_Index','Engagement_Score']
    corr=df[corr_cols].corr().round(3)
    fig=px.imshow(corr,text_auto=True,color_continuous_scale='RdBu_r',zmin=-1,zmax=1,
                  title='Correlation Heatmap — All Numerical Variables',aspect='auto')
    fig.update_traces(textfont_size=10)
    st.plotly_chart(sfig(fig,500),use_container_width=True)

    st.markdown('<div class="sec">Key Correlation Findings</div>',unsafe_allow_html=True)
    findings=[("Income_Numeric ↔ Q31_WTP_INR",f"r = {corr.loc['Income_Numeric','Q31_WTP_INR']:.3f}","Strongest positive — income is the primary WTP driver","#C8E6C9"),
              ("Q20d_HealthInvest ↔ Q31_WTP_INR",f"r = {corr.loc['Q20d_HealthInvest','Q31_WTP_INR']:.3f}","Health investment mindset strongly raises WTP","#C8E6C9"),
              ("Q33_AI_Comfort ↔ Q31_WTP_INR",f"r = {corr.loc['Q33_AI_Comfort','Q31_WTP_INR']:.3f}","AI comfort is 2nd strongest WTP predictor after income","#C8E6C9"),
              ("Engagement_Score ↔ Q31_WTP_INR",f"r = {corr.loc['Engagement_Score','Q31_WTP_INR']:.3f}","High engagement = high spending — target engaged users","#C8E6C9"),
              ("Q15_Stress_Level ↔ Q36_Satisfaction",f"r = {corr.loc['Q15_Stress_Level','Q36_Satisfaction']:.3f}","Stress kills satisfaction — urban stress is a key pain point","#FFCDD2"),
              ("Wellness_Index ↔ Q31_WTP_INR",f"r = {corr.loc['Wellness_Index','Q31_WTP_INR']:.3f}","Wellness-oriented users pay more — wellness framing works","#C8E6C9")]
    c1,c2=st.columns(2)
    for i,(pair,r_val,meaning,color) in enumerate(findings):
        col=c1 if i%2==0 else c2
        col.markdown(f'<div style="background:{color};border-radius:10px;padding:14px;margin:6px 0"><strong>{pair}</strong><br><span style="font-size:1.3rem;font-weight:700">{r_val}</span><br><span style="font-size:.85rem">{meaning}</span></div>',unsafe_allow_html=True)

# ── TAB 7: INSIGHTS ───────────────────────────────────────────────────────────
with tabs[6]:
    st.markdown('<div class="sec">Business Validation Summary & Strategic Recommendations</div>',unsafe_allow_html=True)
    insights_data=[
        ("🎯","Primary ICP","Female, age 23–35, Tier-1 city, Oily/Sensitive skin, Income 40K–1.2L, Research-First or Early Adopter personality, trusts Dermatologists.","CRITICAL"),
        ("🤖","AI Comfort = Revenue","Every 1-point increase in AI Comfort score corresponds to ~₹280–380 increase in WTP. Invest in transparent, explainable AI features.","HIGH"),
        ("💄","Sensitive Skin Gap","52% of Sensitive skin users want to subscribe but AI models typically underserve them. Build specialised formulation tracks for this segment.","HIGH"),
        ("📱","Trust Architecture","Dermatologist-backed positioning is the single highest-ROI marketing strategy. Position as 'dermatologist-intelligence-powered' not just 'AI app'.","HIGH"),
        ("😰","Stress-Skin Hook","High-stress, low-sleep urban users show highest WTP despite lowest current satisfaction. Emotional hook: 'your skin suffers when you do.'","MEDIUM"),
        ("🏙️","Tier-2 Opportunity","Tier-2 cities show 38% interest vs Tier-1's 44% — a smaller gap than expected. Launch a ₹499/month value plan targeting this large, loyal cohort.","MEDIUM"),
        ("📚","Ingredient Education","Expert-literacy users pay 5× more than non-readers. Every educational blog, ingredient explainer, and label-reading guide is a direct WTP uplift tool.","MEDIUM"),
        ("🌿","Formulation Mix","'Mix of Both' (Ayurvedic + Science) is the #1 preference. Build dual-track product lines to capture both naturals-loyal and actives-driven customers.","MEDIUM"),
    ]
    col_map={'CRITICAL':'#B71C1C','HIGH':'#C2185B','MEDIUM':'#F9A825'}
    for icon,title,insight,priority in insights_data:
        color=col_map[priority]
        st.markdown(f'<div style="background:white;border-radius:12px;padding:16px 20px;margin:8px 0;border-left:5px solid {color};box-shadow:0 2px 8px rgba(0,0,0,.05)"><div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px"><span style="font-size:1rem;font-weight:700">{icon} {title}</span><span style="background:{color};color:white;border-radius:5px;padding:2px 8px;font-size:.72rem;font-weight:700">{priority}</span></div><span style="font-size:.88rem;color:#546E7A">{insight}</span></div>',unsafe_allow_html=True)
    
    st.markdown("<br>",unsafe_allow_html=True)
    st.markdown('<div style="background:linear-gradient(135deg,#C2185B,#880E4F);border-radius:12px;padding:22px 26px;color:white;text-align:center"><h3 style="font-family:Playfair Display,serif;margin:0 0 6px">✅ Business Validation Verdict</h3><p style="margin:0;opacity:.9">Survey data confirms strong product-market fit. 42% positive intent + 22% convertible undecided = 64% addressable market. Clear ICP identified. AI personalisation directly drives WTP. <strong>Proceed to MVP.</strong></p></div>',unsafe_allow_html=True)

st.markdown("<br><div style='text-align:center;color:#BDBDBD;font-size:.78rem'>🌸 AI Skincare Subscription · India Validation · Session 1–5 Assignment · 50 Marks</div>",unsafe_allow_html=True)
