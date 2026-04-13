import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.preprocessor import preprocess_single
from utils.decision_engine import combined_decision

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

@st.cache_resource
def load_model():
    return joblib.load(os.path.join(MODELS_DIR,'loan_model.pkl'))

def gauge_chart(score, decision):
    color_map = {'APPROVE':'#15803d','REVIEW':'#92400e','REJECT':'#b91c1c'}
    color = color_map.get(decision, '#64748b')
    fig, ax = plt.subplots(figsize=(3.5, 2.2), subplot_kw={'projection':'polar'})
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    theta = np.linspace(np.pi, 0, 300)
    ax.plot(theta, [1]*300, color='#e2e8f0', linewidth=14, solid_capstyle='round')
    frac = max(0, min(200, score)) / 200
    theta_fill = np.linspace(np.pi, np.pi - frac*np.pi, 300)
    ax.plot(theta_fill, [1]*300, color=color, linewidth=14, solid_capstyle='round')
    ax.set_ylim(0, 1.4); ax.set_yticks([]); ax.set_xticks([])
    ax.spines['polar'].set_visible(False)
    ax.text(0, 0.1, str(int(score)), ha='center', va='center',
            fontsize=28, fontweight='700', color=color,
            fontfamily='DejaVu Sans')
    ax.text(0,-0.28,'/ 200', ha='center', va='center',
            fontsize=9, color='#94a3b8', fontfamily='DejaVu Sans')
    plt.tight_layout(pad=0)
    return fig

def prob_chart(approval_prob, decision):
    reject_prob = 1 - approval_prob
    fig, ax = plt.subplots(figsize=(4, 1.4))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.barh([''], [reject_prob], color='#fca5a5', height=0.5, label=f'Reject {reject_prob*100:.1f}%')
    ax.barh([''], [approval_prob], left=[reject_prob], color='#86efac', height=0.5, label=f'Approve {approval_prob*100:.1f}%')
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(['0%','25%','50%','75%','100%'], fontsize=8, color='#64748b')
    ax.set_yticks([])
    ax.set_title('ML Approval Probability', fontsize=9, color='#475569', pad=6, fontfamily='DejaVu Sans')
    ax.legend(loc='lower right', fontsize=7.5, framealpha=0.9,
              edgecolor='#e2e8f0', facecolor='white')
    for spine in ax.spines.values():
        spine.set_color('#e2e8f0'); spine.set_linewidth(0.5)
    ax.tick_params(colors='#94a3b8', width=0.5)
    plt.tight_layout(pad=0.3)
    return fig

def render():
    st.markdown("""
    <div class="page-header">
        <h1>Single Applicant Analysis</h1>
        <p>Enter applicant details to generate an instant loan approval decision with full explanation</p>
    </div>""", unsafe_allow_html=True)

    try:
        model = load_model()
    except Exception as e:
        st.error(f"Model not loaded. Run `python models/train_model.py` first.\n{e}")
        return

    with st.form("applicant_form"):
        st.markdown('<div class="form-section-title">👤 Personal Information</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            age    = st.number_input("Age", min_value=18, max_value=69, value=None,
                                     placeholder="Enter age (18–69)")
            gender = st.selectbox("Gender", ['Male','Female'])
            city   = st.selectbox("City", ['Nairobi','Kilimani','Westeland','Kasarani'])
        with c2:
            income   = st.number_input("Annual Income ($)", min_value=0, max_value=100000,
                                       value=None, placeholder="e.g. 55000")
            education= st.selectbox("Education Level",
                                    ['High School','Bachelors','Masters','PhD'])
        with c3:
            loan_amt  = st.number_input("Loan Amount ($)", min_value=1000, max_value=50000,
                                        value=None, placeholder="e.g. 20000")
            emp_type  = st.selectbox("Employment Type",
                                     ['Salaried','Self-Employed','Unemployed'])

        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
        st.markdown('<div class="form-section-title">💳 Credit Profile</div>', unsafe_allow_html=True)
        c4, c5 = st.columns(2)
        with c4:
            credit_score = st.slider("Credit Score", 300, 849, 650,
                                     help="300 = very poor, 849 = excellent")
        with c5:
            years_exp = st.slider("Years of Experience", 0, 39, 5)

        st.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)
        submitted = st.form_submit_button("⚡  Generate Decision",
                                          use_container_width=True, type="primary")

    if submitted:
        # Validate required fields
        if age is None or income is None or loan_amt is None:
            st.error("⚠️ Please fill in Age, Annual Income, and Loan Amount before generating a decision.")
            return

        loan_to_income = loan_amt / max(income, 1)
        applicant = {
            'Age': age, 'Income': income, 'LoanAmount': loan_amt,
            'CreditScore': credit_score, 'YearsExperience': years_exp,
            'Gender': gender, 'Education': education, 'City': city,
            'EmploymentType': emp_type, 'LoanToIncome': loan_to_income
        }

        try:
            X = preprocess_single(applicant)
            approval_prob = model.predict_proba(X)[0][1]
            reject_prob   = 1 - approval_prob
        except Exception as ex:
            st.warning(f"ML prediction fallback: {ex}")
            approval_prob = 0.5; reject_prob = 0.5

        result = combined_decision(applicant, reject_prob)
        dec  = result['decision']
        risk = result['risk_level']

        # ── Decision Banner ──
        st.markdown("<hr style='border:none;border-top:1px solid #e2e8f0;margin:8px 0 14px'>", unsafe_allow_html=True)

        color_map = {
            'APPROVE': ('decision-approve','decision-label-approve','#15803d'),
            'REVIEW':  ('decision-review', 'decision-label-review', '#92400e'),
            'REJECT':  ('decision-reject', 'decision-label-reject', '#b91c1c'),
        }
        div_class, label_class, text_color = color_map.get(dec, color_map['REVIEW'])
        st.markdown(f"""
        <div class="{div_class}">
            <div class="{label_class}">{dec}</div>
            <div class="decision-sub" style="color:{text_color}">
                Risk Level: <strong>{risk}</strong> &nbsp;|&nbsp;
                Blended Score: <strong>{result['blended_score']:.1f} / 200</strong> &nbsp;|&nbsp;
                ML Approval Probability: <strong>{approval_prob*100:.1f}%</strong>
            </div>
        </div>""", unsafe_allow_html=True)

        # ── Charts row ──
        ch1, ch2 = st.columns(2)
        with ch1:
            st.markdown('<div class="card"><div class="card-title">Blended Risk Score</div>', unsafe_allow_html=True)
            st.pyplot(gauge_chart(result['blended_score'], dec), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with ch2:
            st.markdown('<div class="card"><div class="card-title">Approval Probability</div>', unsafe_allow_html=True)
            st.pyplot(prob_chart(approval_prob, dec), use_container_width=True)
            # Score breakdown
            st.markdown(f"""
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:10px">
                <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:7px;padding:10px;text-align:center">
                    <div style="font-size:1.2rem;font-weight:700;color:#1e40af">{result['rule_score']}</div>
                    <div style="font-size:0.72rem;color:#64748b;margin-top:2px">Rule Score</div>
                </div>
                <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:7px;padding:10px;text-align:center">
                    <div style="font-size:1.2rem;font-weight:700;color:#1e40af">{result['blended_score']:.0f}</div>
                    <div style="font-size:0.72rem;color:#64748b;margin-top:2px">Blended Score</div>
                </div>
            </div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Factors ──
        f1, f2 = st.columns(2)
        with f1:
            st.markdown('<div class="card"><div class="card-title">✅ Positive Factors</div>', unsafe_allow_html=True)
            if result['positives']:
                for p in result['positives']:
                    st.markdown(f'<div class="factor-pos">✓ {p}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="factor-neg">No significant positive factors identified</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with f2:
            st.markdown('<div class="card"><div class="card-title">⚠️ Risk Flags</div>', unsafe_allow_html=True)
            all_flags = result['hard_reasons'] + result['flags']
            if result['hard_reasons']:
                for r in result['hard_reasons']:
                    st.markdown(f'<div class="factor-bad">🚫 {r}</div>', unsafe_allow_html=True)
            if result['flags']:
                for flag in result['flags']:
                    st.markdown(f'<div class="factor-neg">⚠ {flag}</div>', unsafe_allow_html=True)
            if not all_flags:
                st.markdown('<div class="factor-pos">✓ No risk flags raised</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Profile ──
        with st.expander("📋 Full Applicant Profile", expanded=False):
            summary = {
                'Field': ['Age','Annual Income','Loan Amount','Credit Score',
                          'Years Experience','Gender','Education','City',
                          'Employment Type','Loan-to-Income Ratio'],
                'Value': [age, f"${income:,}", f"${loan_amt:,}", credit_score,
                          f"{years_exp} yrs", gender, education, city,
                          emp_type, f"{loan_to_income:.3f}"]
            }
            st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)