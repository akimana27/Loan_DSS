import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys, joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.preprocessor import preprocess, NUMERICAL_COLS, CATEGORICAL_COLS
from utils.decision_engine import combined_decision

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
DATA_PATH  = os.path.join(os.path.dirname(__file__), '..', 'data', 'loan_data_raw.csv')

@st.cache_resource
def load_artifacts():
    model    = joblib.load(os.path.join(MODELS_DIR,'loan_model.pkl'))
    encoders = joblib.load(os.path.join(MODELS_DIR,'encoders.pkl'))
    scaler   = joblib.load(os.path.join(MODELS_DIR,'scaler.pkl'))
    return model, encoders, scaler

@st.cache_data
def process_batch(df_raw_json, _model, _encoders, _scaler):
    df = pd.read_json(df_raw_json)
    df['Income']       = df['Income'].clip(lower=0)
    df['LoanToIncome'] = df['LoanAmount'] / df['Income'].replace(0, np.nan)
    df['Education']    = df['Education'].fillna('Unknown')
    results = []
    for _, row in df.iterrows():
        applicant = row.to_dict()
        try:
            X, _, _, _ = preprocess(pd.DataFrame([applicant]), fit=False,
                                    encoders=_encoders, scaler=_scaler)
            approval_prob = _model.predict_proba(X)[0][1]
            reject_prob   = 1 - approval_prob
        except Exception:
            reject_prob = 0.5; approval_prob = 0.5
        res = combined_decision(applicant, reject_prob)
        results.append({
            'Decision':      res['decision'],
            'Risk Level':    res['risk_level'],
            'Blended Score': res['blended_score'],
            'Approval Prob%':round(approval_prob*100,1),
            'Rule Score':    res['rule_score'],
            'Flags':         len(res['flags']),
            'Hard Reject':   res['hard_reject'],
        })
    return pd.DataFrame(results)

def render():
    st.markdown("""
    <div class="page-header">
        <h1>📊 Batch Loan Analysis</h1>
        <p>Upload a CSV of applicants and receive instant decisions for all records</p>
    </div>""", unsafe_allow_html=True)

    # Template download
    template = pd.DataFrame({
        'Age':[32,45,27],'Income':[55000,92000,31000],
        'LoanAmount':[8000,20000,5000],'CreditScore':[660,760,480],
        'YearsExperience':[5,15,1],'Gender':['Male','Female','Male'],
        'Education':['Bachelors','Masters','High School'],
        'City':['Houston','New York','Chicago'],
        'EmploymentType':['Salaried','Salaried','Unemployed']
    })

    col_a, col_b = st.columns([3,1])
    with col_a:
        st.info("📎 Upload a CSV with the same columns as the dataset. No file? The demo runs on all 5,000 records.")
    with col_b:
        st.download_button("⬇️ CSV Template",
                           template.to_csv(index=False).encode(),
                           "loan_template.csv","text/csv",use_container_width=True)

    uploaded = st.file_uploader("Upload Applicants CSV", type=['csv'], label_visibility="collapsed")

    if uploaded is None:
        st.markdown('<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:10px 14px;font-size:0.85rem;color:#64748b;margin-bottom:10px">📁 No file uploaded — running demo on full 5,000-record dataset</div>', unsafe_allow_html=True)
        df_raw = pd.read_csv(DATA_PATH)
    else:
        df_raw = pd.read_csv(uploaded)

    st.markdown(f'<div style="font-size:0.85rem;color:#475569;margin-bottom:8px">✅ <strong>{len(df_raw):,} records</strong> loaded · {df_raw.shape[1]} columns</div>', unsafe_allow_html=True)

    try:
        model, encoders, scaler = load_artifacts()
    except Exception as e:
        st.error(f"Model artefacts missing. Run training first.\n{e}"); return

    with st.spinner("Running decision engine on all records..."):
        results_df = process_batch(df_raw.to_json(), model, encoders, scaler)

    combined_out = pd.concat([df_raw.reset_index(drop=True), results_df], axis=1)
    total    = len(results_df)
    approved = (results_df['Decision']=='APPROVE').sum()
    rejected = (results_df['Decision']=='REJECT').sum()
    reviewed = (results_df['Decision']=='REVIEW').sum()
    hard_rej = results_df['Hard Reject'].sum()

    # KPIs
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Total",        f"{total:,}")
    c2.metric("Approved",     f"{approved:,}", f"{approved/total*100:.1f}%")
    c3.metric("Review",       f"{reviewed:,}", f"{reviewed/total*100:.1f}%")
    c4.metric("Rejected",     f"{rejected:,}", f"{rejected/total*100:.1f}%", delta_color="inverse")
    c5.metric("Hard Rejects", f"{int(hard_rej):,}")

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # Charts
    ch1, ch2 = st.columns(2)
    with ch1:
        fig, ax = plt.subplots(figsize=(4, 3.5))
        fig.patch.set_facecolor('white'); ax.set_facecolor('white')
        labels  = ['APPROVE','REVIEW','REJECT']
        counts  = [approved, reviewed, rejected]
        colors  = ['#86efac','#fcd34d','#fca5a5']
        edge_c  = ['#15803d','#92400e','#b91c1c']
        wedges, texts, autotexts = ax.pie(
            counts, labels=labels, colors=colors,
            autopct='%1.1f%%', startangle=90,
            wedgeprops={'linewidth':2,'edgecolor':'white'})
        for t in texts:     t.set_fontsize(10); t.set_color('#475569')
        for at in autotexts: at.set_fontsize(9); at.set_color('#1e293b'); at.set_fontweight('600')
        ax.set_title('Decision Distribution', fontsize=11, fontweight='600',
                     color='#1e293b', pad=10, fontfamily='DejaVu Sans')
        plt.tight_layout(pad=0.5)
        st.pyplot(fig, use_container_width=True); plt.close()

    with ch2:
        fig, ax = plt.subplots(figsize=(4, 3.5))
        fig.patch.set_facecolor('white'); ax.set_facecolor('white')
        risk_counts = results_df['Risk Level'].value_counts()
        risk_colors_map = {'LOW':'#86efac','MEDIUM':'#93c5fd',
                           'MEDIUM-HIGH':'#fcd34d','HIGH':'#fca5a5','CRITICAL':'#f87171'}
        bar_colors = [risk_colors_map.get(r,'#e2e8f0') for r in risk_counts.index]
        bars = ax.bar(risk_counts.index, risk_counts.values,
                      color=bar_colors, edgecolor='white', linewidth=1.5, width=0.55)
        for bar, val in zip(bars, risk_counts.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+3,
                    str(val), ha='center', fontsize=9, color='#475569', fontweight='500')
        ax.set_title('Risk Level Distribution', fontsize=11, fontweight='600',
                     color='#1e293b', pad=10, fontfamily='DejaVu Sans')
        ax.set_ylabel('Count', fontsize=9, color='#64748b')
        ax.tick_params(colors='#94a3b8', labelsize=8)
        ax.grid(axis='y', alpha=0.4, color='#e2e8f0', linewidth=0.8)
        for spine in ax.spines.values(): spine.set_color('#e2e8f0'); spine.set_linewidth(0.5)
        plt.tight_layout(pad=0.5)
        st.pyplot(fig, use_container_width=True); plt.close()

    # Score histogram
    fig, ax = plt.subplots(figsize=(8, 2.8))
    fig.patch.set_facecolor('white'); ax.set_facecolor('white')
    for dec, color in [('APPROVE','#86efac'),('REVIEW','#fcd34d'),('REJECT','#fca5a5')]:
        sub = results_df[results_df['Decision']==dec]['Blended Score']
        if len(sub): ax.hist(sub, bins=30, alpha=0.75, color=color, label=dec, edgecolor='white')
    ax.axvline(95, color='#1e40af', linestyle='--', linewidth=1.2, alpha=0.7, label='Approve threshold (95)')
    ax.axvline(75, color='#92400e', linestyle='--', linewidth=1.2, alpha=0.7, label='Review threshold (75)')
    ax.set_title('Blended Score Distribution', fontsize=11, fontweight='600', color='#1e293b', pad=8)
    ax.set_xlabel('Blended Score', fontsize=9, color='#64748b')
    ax.set_ylabel('Count', fontsize=9, color='#64748b')
    ax.legend(fontsize=8, framealpha=0.9, edgecolor='#e2e8f0')
    ax.grid(alpha=0.4, color='#e2e8f0', linewidth=0.8)
    ax.tick_params(colors='#94a3b8', labelsize=8)
    for spine in ax.spines.values(): spine.set_color('#e2e8f0'); spine.set_linewidth(0.5)
    plt.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=True); plt.close()

    # Results table
    st.markdown('<div class="section-title" style="margin-top:8px">📋 Decision Results</div>', unsafe_allow_html=True)
    col_f, col_dl = st.columns([3,1])
    with col_f:
        filter_dec = st.multiselect("Filter by Decision",
                                    ['APPROVE','REVIEW','REJECT'],
                                    default=['APPROVE','REVIEW','REJECT'])
    with col_dl:
        st.download_button("⬇️ Download Results",
                           combined_out.to_csv(index=False).encode(),
                           "loan_decisions.csv","text/csv",use_container_width=True)

    display = combined_out[combined_out['Decision'].isin(filter_dec)].copy()
    cols_show = ['Age','Income','LoanAmount','CreditScore','EmploymentType',
                 'Education','Decision','Risk Level','Blended Score','Approval Prob%']
    cols_show = [c for c in cols_show if c in display.columns]

    def color_decision(val):
        c = {'APPROVE':'#dcfce7','REVIEW':'#fef3c7','REJECT':'#fee2e2'}.get(val,'')
        return f'background-color:{c}' if c else ''

    st.dataframe(
        display[cols_show].style.map(color_decision, subset=['Decision']),
        use_container_width=True, height=360, hide_index=True
    )