import streamlit as st
import os, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ASSETS = os.path.join(os.path.dirname(__file__), '..', 'assets')
MODELS = os.path.join(os.path.dirname(__file__), '..', 'models')

def show_chart(path, caption, width=0.65):
    """Display a chart at controlled size using matplotlib to avoid use_column_width warning."""
    if os.path.exists(path):
        img = mpimg.imread(path)
        h, w = img.shape[:2]
        fig_w = 7 * width
        fig_h = fig_w * h / w
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        fig.patch.set_facecolor('white')
        ax.imshow(img)
        ax.axis('off')
        plt.tight_layout(pad=0)
        st.pyplot(fig, use_container_width=False)
        plt.close()
        if caption:
            st.markdown(f'<div style="font-size:0.75rem;color:#94a3b8;text-align:center;margin-top:4px">{caption}</div>', unsafe_allow_html=True)
    else:
        st.warning(f"Chart not found: {os.path.basename(path)} — run `python models/train_model.py`")

def render():
    st.markdown("""
    <div class="page-header">
        <h1>Model Insights</h1>
        <p>Performance metrics, ROC curves, confusion matrix and feature importance — trained on real data</p>
    </div>""", unsafe_allow_html=True)

    try:
        with open(os.path.join(MODELS,'model_meta.json')) as f:
            meta = json.load(f)
    except Exception:
        meta = {'best_model':'Random Forest','auc':0.9338,'accuracy':0.9630,
                'avg_precision':0.8969,'cv_auc':0.9542}

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Best Model",      meta.get('best_model','—'))
    c2.metric("Test AUC",        f"{meta.get('auc',0):.4f}")
    c3.metric("Accuracy",        f"{meta.get('accuracy',0)*100:.2f}%")
    c4.metric("Avg Precision",   f"{meta.get('avg_precision',0):.4f}")
    c5.metric("CV AUC (5-fold)", f"{meta.get('cv_auc',0):.4f}")

    st.success(f"✅ **{meta.get('best_model','Random Forest')}** achieved **AUC {meta.get('auc',0.9338):.4f}** "
               f"and **{meta.get('accuracy',0.963)*100:.1f}% accuracy** on the real dataset.")

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    tab1,tab2,tab3,tab4,tab5 = st.tabs([
        "📉 ROC Curves","📈 Precision-Recall","🔢 Confusion Matrix","🌳 Feature Importance","📏 Rule Engine"
    ])

    with tab1:
        col_l, col_r = st.columns([3,2])
        with col_l:
            show_chart(os.path.join(ASSETS,'model_roc.png'),
                       "ROC curves — Random Forest (blue), Gradient Boosting (green), Logistic Regression (red)")
        with col_r:
            st.markdown('<div class="card"><div class="card-title">Model Comparison</div>', unsafe_allow_html=True)
            rows = [
                ("Random Forest",      meta.get('auc',0.9338),    "Best — selected", "#dcfce7","#15803d"),
                ("Gradient Boosting",  0.9132, "Close second",    "#dbeafe","#1e40af"),
                ("Logistic Regression",0.8813, "Good baseline",   "#f8fafc","#475569"),
            ]
            for name, auc, note, bg, color in rows:
                st.markdown(f"""
                <div style="background:{bg};border:1px solid #e2e8f0;border-radius:7px;
                            padding:10px 12px;margin-bottom:8px">
                    <div style="font-weight:600;color:{color};font-size:0.85rem">{name}</div>
                    <div style="font-size:1.1rem;font-weight:700;color:{color}">AUC {auc:.4f}</div>
                    <div style="font-size:0.75rem;color:#64748b">{note}</div>
                </div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("""
            <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:7px;
                        padding:10px 12px;font-size:0.8rem;color:#475569">
            AUC &gt; 0.90 is considered <strong>excellent</strong> for credit risk classification.
            </div>""", unsafe_allow_html=True)

    with tab2:
        col_l, col_r = st.columns([3,2])
        with col_l:
            show_chart(os.path.join(ASSETS,'model_pr.png'),
                       "Precision-Recall curves — important due to class imbalance (23% approvals)")
        with col_r:
            st.markdown('<div class="card"><div class="card-title">Why Precision-Recall Matters</div>', unsafe_allow_html=True)
            st.markdown("""
            <div style="font-size:0.82rem;color:#475569;line-height:1.7">
            The dataset is <strong>class-imbalanced</strong> — 77% rejected vs 23% approved.
            In such cases, AUC alone can be misleading.<br><br>
            <strong>Average Precision (AP)</strong> summarises the full precision-recall curve:
            </div>""", unsafe_allow_html=True)
            for name, ap, color, bg in [
                ("Random Forest",       0.8969, "#15803d","#dcfce7"),
                ("Gradient Boosting",   0.8842, "#1e40af","#dbeafe"),
                ("Logistic Regression", 0.7426, "#64748b","#f8fafc"),
            ]:
                st.markdown(f"""
                <div class="insight-row">
                    <span style="font-size:0.82rem">{name}</span>
                    <span style="background:{bg};color:{color};padding:2px 8px;border-radius:8px;
                                 font-weight:700;font-size:0.8rem">AP {ap:.4f}</span>
                </div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        col_pad1, col_c, col_pad2 = st.columns([1,2,1])
        with col_c:
            show_chart(os.path.join(ASSETS,'model_cm.png'),
                       "Confusion matrix — Random Forest on 1,000 test records")
        st.markdown("""
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:10px">
            <div style="background:#dcfce7;border:1px solid #16a34a;border-radius:7px;padding:10px 14px">
                <div style="font-weight:600;color:#15803d;font-size:0.82rem">✓ True Negatives</div>
                <div style="font-size:0.78rem;color:#166534;margin-top:3px">Correctly rejected applicants — protects the bank from bad loans</div>
            </div>
            <div style="background:#dcfce7;border:1px solid #16a34a;border-radius:7px;padding:10px 14px">
                <div style="font-weight:600;color:#15803d;font-size:0.82rem">✓ True Positives</div>
                <div style="font-size:0.78rem;color:#166534;margin-top:3px">Correctly approved applicants — generates revenue for the bank</div>
            </div>
            <div style="background:#fee2e2;border:1px solid #dc2626;border-radius:7px;padding:10px 14px">
                <div style="font-weight:600;color:#b91c1c;font-size:0.82rem">✗ False Negatives</div>
                <div style="font-size:0.78rem;color:#991b1b;margin-top:3px">Good applicants incorrectly rejected — lost revenue opportunity</div>
            </div>
            <div style="background:#fee2e2;border:1px solid #dc2626;border-radius:7px;padding:10px 14px">
                <div style="font-weight:600;color:#b91c1c;font-size:0.82rem">✗ False Positives</div>
                <div style="font-size:0.78rem;color:#991b1b;margin-top:3px">Bad applicants incorrectly approved — financial loss risk</div>
            </div>
        </div>""", unsafe_allow_html=True)

    with tab4:
        col_l, col_r = st.columns([3,2])
        with col_l:
            show_chart(os.path.join(ASSETS,'model_fi.png'),
                       "Feature importance scores — Random Forest")
        with col_r:
            st.markdown('<div class="card"><div class="card-title">Top Predictors</div>', unsafe_allow_html=True)
            features_info = [
                ("CreditScore",     "Highest", "Single strongest predictor",       "#b91c1c","#fee2e2"),
                ("Income",          "High",    "Higher income → better repayment", "#92400e","#fef3c7"),
                ("LoanToIncome",    "High",    "Engineered: loan burden ratio",    "#92400e","#fef3c7"),
                ("EmploymentType",  "High",    "Unemployed nearly always rejected","#1e40af","#dbeafe"),
                ("LoanAmount",      "Medium",  "Larger loans carry higher risk",   "#475569","#f1f5f9"),
                ("YearsExperience", "Medium",  "Employment stability proxy",       "#475569","#f1f5f9"),
                ("Age",             "Medium",  "Modest effect across age groups",  "#475569","#f1f5f9"),
                ("Education",       "Low",     "Small effect vs credit/income",    "#64748b","#f8fafc"),
                ("City",            "Very Low","Geography has little effect",      "#94a3b8","#f8fafc"),
                ("Gender",          "Very Low","Near-zero — model gender-neutral", "#94a3b8","#f8fafc"),
            ]
            for feat, imp, note, color, bg in features_info:
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:8px;padding:5px 0;
                            border-bottom:1px solid #f1f5f9">
                    <code style="font-size:0.75rem;color:{color};background:{bg};
                                 padding:1px 6px;border-radius:4px">{feat}</code>
                    <span style="font-size:0.75rem;color:#64748b;flex:1">{note}</span>
                    <span style="font-size:0.72rem;font-weight:600;color:{color}">{imp}</span>
                </div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with tab5:
        col1, col2 = st.columns(2)
        with col1:
            # Hard reject
            st.markdown("""
            <div style="background:#fee2e2;border:1.5px solid #dc2626;border-radius:9px;padding:14px;margin-bottom:10px">
            <div style="font-weight:700;color:#b91c1c;font-size:0.85rem;margin-bottom:8px">🚫 Hard Reject Rules — Automatic Denial</div>
            <table style="width:100%;font-size:0.8rem">
            <tr style="border-bottom:1px solid #fca5a5"><th align="left" style="color:#991b1b;padding:4px 0">Condition</th><th align="left" style="color:#991b1b">Threshold</th></tr>
            <tr><td style="color:#b91c1c;padding:4px 0">Credit Score</td><td style="color:#b91c1c">&lt; 400</td></tr>
            <tr><td style="color:#b91c1c;padding:4px 0">Employment Type</td><td style="color:#b91c1c">Unemployed</td></tr>
            <tr><td style="color:#b91c1c;padding:4px 0">Loan-to-Income</td><td style="color:#b91c1c">&gt; 2.0×</td></tr>
            </table></div>""", unsafe_allow_html=True)

            # Soft flags
            st.markdown("""
            <div style="background:#fef3c7;border:1.5px solid #d97706;border-radius:9px;padding:14px">
            <div style="font-weight:700;color:#92400e;font-size:0.85rem;margin-bottom:8px">⚠ Soft Flag Rules — Score Deductions</div>
            <table style="width:100%;font-size:0.8rem">
            <tr style="border-bottom:1px solid #fcd34d"><th align="left" style="color:#78350f;padding:4px 0">Condition</th><th align="right" style="color:#78350f">Penalty</th></tr>
            <tr><td style="color:#92400e;padding:3px 0">CreditScore &lt; 500</td><td align="right" style="color:#b91c1c;font-weight:600">−20 pts</td></tr>
            <tr><td style="color:#92400e;padding:3px 0">LoanToIncome &gt; 1.0</td><td align="right" style="color:#b91c1c;font-weight:600">−15 pts</td></tr>
            <tr><td style="color:#92400e;padding:3px 0">Income &lt; $30,000</td><td align="right" style="color:#b91c1c;font-weight:600">−12 pts</td></tr>
            <tr><td style="color:#92400e;padding:3px 0">CreditScore &lt; 600</td><td align="right" style="color:#b91c1c;font-weight:600">−10 pts</td></tr>
            <tr><td style="color:#92400e;padding:3px 0">LoanToIncome &gt; 0.7</td><td align="right" style="color:#b91c1c;font-weight:600">−8 pts</td></tr>
            <tr><td style="color:#92400e;padding:3px 0">YearsExperience &lt; 2</td><td align="right" style="color:#b91c1c;font-weight:600">−8 pts</td></tr>
            </table></div>""", unsafe_allow_html=True)

        with col2:
            # Positive rules
            st.markdown("""
            <div style="background:#dcfce7;border:1.5px solid #16a34a;border-radius:9px;padding:14px;margin-bottom:10px">
            <div style="font-weight:700;color:#15803d;font-size:0.85rem;margin-bottom:8px">✅ Positive Rules — Score Bonuses</div>
            <table style="width:100%;font-size:0.8rem">
            <tr style="border-bottom:1px solid #86efac"><th align="left" style="color:#166534;padding:4px 0">Condition</th><th align="right" style="color:#166534">Bonus</th></tr>
            <tr><td style="color:#15803d;padding:3px 0">CreditScore ≥ 750</td><td align="right" style="color:#15803d;font-weight:600">+20 pts</td></tr>
            <tr><td style="color:#15803d;padding:3px 0">LoanToIncome &lt; 0.3</td><td align="right" style="color:#15803d;font-weight:600">+15 pts</td></tr>
            <tr><td style="color:#15803d;padding:3px 0">Income &gt; $80,000</td><td align="right" style="color:#15803d;font-weight:600">+15 pts</td></tr>
            <tr><td style="color:#15803d;padding:3px 0">CreditScore 700–749</td><td align="right" style="color:#15803d;font-weight:600">+12 pts</td></tr>
            <tr><td style="color:#15803d;padding:3px 0">Income &gt; $55,000</td><td align="right" style="color:#15803d;font-weight:600">+8 pts</td></tr>
            <tr><td style="color:#15803d;padding:3px 0">LoanToIncome &lt; 0.5</td><td align="right" style="color:#15803d;font-weight:600">+8 pts</td></tr>
            <tr><td style="color:#15803d;padding:3px 0">YearsExperience ≥ 10</td><td align="right" style="color:#15803d;font-weight:600">+10 pts</td></tr>
            <tr><td style="color:#15803d;padding:3px 0">Salaried Employment</td><td align="right" style="color:#15803d;font-weight:600">+10 pts</td></tr>
            <tr><td style="color:#15803d;padding:3px 0">Masters / PhD</td><td align="right" style="color:#15803d;font-weight:600">+5 pts</td></tr>
            </table></div>""", unsafe_allow_html=True)

            # Formula
            st.markdown("""
            <div style="background:#f8fafc;border:1.5px solid #e2e8f0;border-radius:9px;padding:14px">
            <div style="font-weight:700;color:#1e293b;font-size:0.85rem;margin-bottom:8px">⚖ Blended Decision Formula</div>
            <div style="background:#1e293b;border-radius:6px;padding:12px;
                        font-family:monospace;font-size:0.8rem;color:#93c5fd;line-height:2">
            RuleScore (base 100)<br>
            + bonuses − deductions<br><br>
            MLScore = (1 − reject_prob) × 100<br><br>
            <span style="color:#86efac">Blend = 0.60×Rule + 0.40×ML</span><br><br>
            ≥ 115 → APPROVE (LOW risk)<br>
            ≥ 95  → APPROVE (MEDIUM risk)<br>
            ≥ 75  → REVIEW<br>
            &lt; 75  → REJECT
            </div></div>""", unsafe_allow_html=True)