import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

ASSETS = os.path.join(os.path.dirname(__file__), '..', 'assets')
DATA   = os.path.join(os.path.dirname(__file__), '..', 'data', 'loan_data_raw.csv')

def show_chart(path, caption=""):
    if os.path.exists(path):
        img = mpimg.imread(path)
        h, w = img.shape[:2]
        fig_w = 5.5
        fig_h = fig_w * h / w
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        fig.patch.set_facecolor('white')
        ax.imshow(img)
        ax.axis('off')
        plt.tight_layout(pad=0)
        st.pyplot(fig, use_container_width=True)
        plt.close()
        if caption:
            st.markdown(f'<div style="font-size:0.75rem;color:#94a3b8;margin-top:3px;padding-bottom:12px">{caption}</div>', unsafe_allow_html=True)
    else:
        st.warning(f"Chart not found. Run `python utils/eda.py`")

def render():
    st.markdown("""
    <div class="page-header">
        <h1>📚 EDA Explorer</h1>
        <p>Exploratory Data Analysis on 5,000 real loan applicant records</p>
    </div>""", unsafe_allow_html=True)

    try:
        df = pd.read_csv(DATA)
        df['Income'] = df['Income'].clip(lower=0)
    except Exception:
        st.error("Dataset not found."); return

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Records",        f"{len(df):,}")
    c2.metric("Features",       str(df.shape[1]))
    c3.metric("Approval Rate",  f"{df['LoanApproved'].mean()*100:.1f}%")
    c4.metric("Avg CreditScore",f"{df['CreditScore'].mean():.0f}")
    c5.metric("Missing Values", str(df.isnull().sum().sum()))

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    with st.expander("📋 Statistical Summary", expanded=False):
        st.dataframe(df.describe().round(2), use_container_width=True)

    with st.expander("🔍 Missing Values Report", expanded=False):
        miss = df.isnull().sum()
        miss = miss[miss>0].reset_index()
        miss.columns = ['Column','Missing Count']
        miss['Missing %'] = (miss['Missing Count']/len(df)*100).round(2)
        st.dataframe(miss, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title" style="margin-top:4px">📈 EDA Chart Gallery</div>', unsafe_allow_html=True)

    charts = [
        ('eda_loan_status.png',  'Loan Approval Distribution',
         'Class imbalance: 76.98% rejected vs 23.02% approved. Required class_weight=balanced in ML training.'),
        ('eda_credit_score.png', 'Credit Score by Approval Status',
         'Approved applicants average 708 vs 534 for rejected. Credit score is the single strongest predictor.'),
        ('eda_income_scatter.png','Income vs Loan Amount',
         'Approved applicants cluster where income is high relative to loan. High loan-to-income ratios are rejected.'),
        ('eda_employment.png',   'Approval Rate by Employment Type',
         'Salaried (33.3%) and Self-Employed (32.5%) nearly identical. Unemployed: only 3.1% approval — major risk signal.'),
        ('eda_education.png',    'Approval Rate by Education',
         'PhD holders show the highest approval rate (25.1%). Differences are modest compared to credit score.'),
        ('eda_age.png',          'Age Distribution by Approval',
         'Approval rates are relatively uniform across age groups — age alone is not a strong predictor.'),
        ('eda_correlation.png',  'Feature Correlation Matrix',
         'CreditScore has the strongest positive correlation with LoanApproved (r≈0.46). Income has a moderate effect.'),
        ('eda_city.png',         'Approval Rate by City',
         'All four cities show similar rates (~22–24%), confirming city is a weak predictor in this dataset.'),
    ]

    for i in range(0, len(charts), 2):
        col1, col2 = st.columns(2)
        pairs = [(col1, charts[i])]
        if i+1 < len(charts): pairs.append((col2, charts[i+1]))
        for col, (fname, title, interpretation) in pairs:
            with col:
                st.markdown(f'<div style="font-size:0.85rem;font-weight:600;color:#1e293b;margin-bottom:6px">{title}</div>', unsafe_allow_html=True)
                show_chart(os.path.join(ASSETS, fname), interpretation)

    st.markdown('<div class="section-title" style="margin-top:4px">🗃️ Dataset Sample</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns([2,1])
    with col_a:
        n = st.slider("Rows to preview", 5, 100, 20)
    with col_b:
        status_filter = st.selectbox("Filter by approval", ['All','Approved (1)','Rejected (0)'])

    if status_filter == 'Approved (1)':
        view_df = df[df['LoanApproved']==1].head(n)
    elif status_filter == 'Rejected (0)':
        view_df = df[df['LoanApproved']==0].head(n)
    else:
        view_df = df.head(n)

    st.dataframe(view_df, use_container_width=True, hide_index=True)