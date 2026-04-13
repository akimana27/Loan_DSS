import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json, os

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
DATA_PATH  = os.path.join(os.path.dirname(__file__), '..', 'data', 'loan_data_raw.csv')

def render():
    st.markdown("""
    <div class="page-header">
        <h1> Loan Decision Support System</h1>
        <p>Loan approval analysis &mdash; 5,000 real applicant records &middot; Random Forest &middot; Rule-Based MCDM</p>
    </div>""", unsafe_allow_html=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(DATA_PATH)
        df['Income'] = df['Income'].clip(lower=0)
        total    = len(df)
        approved = int(df['LoanApproved'].sum())
        avg_cs   = df['CreditScore'].mean()
        avg_inc  = df['Income'].mean()
        app_rate = approved / total * 100
    except Exception:
        total, approved, avg_cs, avg_inc, app_rate = 5000, 1151, 575, 49738, 23.0

    try:
        with open(os.path.join(MODELS_DIR,'model_meta.json')) as f:
            meta = json.load(f)
    except Exception:
        meta = {'best_model':'Random Forest','auc':0.9338,'accuracy':0.9630,'cv_auc':0.9542}

    # ── KPI Metrics ───────────────────────────────────────────────────────────
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Total Records",     f"{total:,}")
    c2.metric("Approval Rate",     f"{app_rate:.1f}%", f"{approved:,} approved")
    c3.metric("Avg Credit Score",  f"{avg_cs:.0f}")
    c4.metric("Avg Annual Income", f"${avg_inc:,.0f}")
    c5.metric("Model AUC",         f"{meta.get('auc',0):.4f}", meta.get('best_model',''))

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ── Architecture ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-title" style="font-size:1rem">System Architecture</div>',
                unsafe_allow_html=True)
    cols = st.columns([3,1,3,1,3,1,3,1,3])
    boxes = [
        ("📂", "Raw Dataset",   "5,000 rows · 10 features", "#dbeafe", "#1e40af"),
        ("", "Preprocessing", "Clean · Encode · Scale",   "#e0f2fe", "#0369a1"),
        ("🤖", "Random Forest", "AUC 0.93 · 96% accuracy",  "#dcfce7", "#15803d"),
        ("", "Rule Engine",   "MCDM · 20 Rules",           "#fef3c7", "#92400e"),
        ("✅", "Decision",      "Approve · Review · Reject", "#dcfce7", "#15803d"),
    ]
    for i, (icon, title, sub, bg, color) in enumerate(boxes):
        with cols[i*2]:
            st.markdown(f"""
            <div style="background:{bg};border:1.5px solid {color};border-radius:9px;
                        padding:12px;text-align:center">
                <div style="font-size:1.4rem">{icon}</div>
                <div style="font-size:0.85rem;font-weight:700;color:{color};margin-top:4px">{title}</div>
                <div style="font-size:0.75rem;color:#64748b;margin-top:2px">{sub}</div>
            </div>""", unsafe_allow_html=True)
        if i < 4:
            with cols[i*2+1]:
                st.markdown("<div style='text-align:center;font-size:1.4rem;color:#94a3b8;padding-top:20px'></div>",
                            unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # ROW 1 — Dataset Features (full width)
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px">
        <span style="font-size:1rem;font-weight:700;color:#1e293b">Dataset Features</span>
        <span style="background:#e0f2fe;color:#0369a1;padding:3px 12px;border-radius:20px;
                     font-size:0.78rem;font-weight:600">10 variables</span>
    </div>
    """, unsafe_allow_html=True)

    # Search bar (visual only)
    st.markdown("""
    <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;
                padding:9px 14px;margin-bottom:12px;color:#94a3b8;font-size:0.88rem">
         &nbsp; Search features...
    </div>
    """, unsafe_allow_html=True)

    # Feature definitions with icon, name, description, badge
    features = [
        # (icon, name, description, badge_text, badge_bg, badge_color)
        ("",  "ApplicantID",      "Unique identifier for each applicant",        "ID",  "#fef3c7", "#92400e"),
        ("",  "Age",              f"Applicant age · Range 18–69 yrs",             "Num", "#eff6ff", "#1e40af"),
        ("",  "Income",           f"Annual gross income · Avg $49,740",           "Num", "#eff6ff", "#1e40af"),
        ("",  "LoanAmount",       f"Requested loan amount · Avg $19,871",         "Num", "#eff6ff", "#1e40af"),
        ("",  "CreditScore",      f"Credit score 300–850 · Avg 575",              "Num", "#eff6ff", "#1e40af"),
        ("",  "YearsExperience",  f"Years of employment · Avg 19.6 yrs",          "Num", "#eff6ff", "#1e40af"),
        ("", "EmploymentType",   "Salaried / Self-Employed / Unemployed",        "Cat", "#f0fdf4", "#15803d"),
        ("",  "Education",        "High School / Bachelors / Masters / PhD",      "Cat", "#f0fdf4", "#15803d"),
        ("",  "Gender",           "Male / Female",                                "Cat", "#f0fdf4", "#15803d"),
        ("", "City",             "Houston / New York / Chicago / San Francisco", "Cat", "#f0fdf4", "#15803d"),
    ]

    # Render features in 2 columns side by side
    left_features  = features[:5]
    right_features = features[5:]

    f_col1, f_col2 = st.columns(2)

    def feature_card(icon, name, desc, badge, badge_bg, badge_color):
        return f"""
        <div style="display:flex;align-items:center;justify-content:space-between;
                    background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;
                    padding:10px 14px;margin-bottom:7px">
            <div style="display:flex;align-items:center;gap:10px">
                <span style="font-size:1.15rem;width:24px;text-align:center">{icon}</span>
                <div>
                    <div style="font-size:0.88rem;font-weight:600;color:#1e293b">{name}</div>
                    <div style="font-size:0.75rem;color:#64748b;margin-top:1px">{desc}</div>
                </div>
            </div>
            <span style="background:{badge_bg};color:{badge_color};padding:3px 10px;
                         border-radius:20px;font-size:0.75rem;font-weight:700;
                         flex-shrink:0;margin-left:8px">{badge}</span>
        </div>"""

    with f_col1:
        for icon, name, desc, badge, bbg, bc in left_features:
            st.markdown(feature_card(icon, name, desc, badge, bbg, bc),
                        unsafe_allow_html=True)

    with f_col2:
        for icon, name, desc, badge, bbg, bc in right_features:
            st.markdown(feature_card(icon, name, desc, badge, bbg, bc),
                        unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # ROW 2 — Charts side by side
    # ══════════════════════════════════════════════════════════════════════════
    try:
        df3 = pd.read_csv(DATA_PATH)
        df3['Income'] = df3['Income'].clip(lower=0)
        emp = df3.groupby('EmploymentType')['LoanApproved'].mean() * 100
        edu = df3.groupby('Education')['LoanApproved'].mean() * 100
        edu = edu.reindex(['High School','Bachelors','Masters','PhD'])
    except Exception:
        emp = pd.Series({'Salaried':33.3,'Self-Employed':32.5,'Unemployed':3.1})
        edu = pd.Series({'High School':22.0,'Bachelors':21.6,'Masters':23.3,'PhD':25.1})

    chart_col1, chart_col2 = st.columns(2)

    # ── Chart 1: Employment ───────────────────────────────────────────────────
    with chart_col1:
        st.markdown('<div style="font-size:0.92rem;font-weight:700;color:#1e293b;margin-bottom:8px">📊 Approval Rate by Employment Type</div>',
                    unsafe_allow_html=True)
        fig1, ax1 = plt.subplots(figsize=(5, 3.2))
        fig1.patch.set_facecolor('white')
        ax1.set_facecolor('white')
        emp_colors = ['#16a34a' if v > 15 else '#dc2626' for v in emp.values]
        bars1 = ax1.bar(emp.index, emp.values, color=emp_colors,
                        edgecolor='white', linewidth=1.5, width=0.5)
        for bar, val in zip(bars1, emp.values):
            ax1.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.6,
                     f'{val:.1f}%', ha='center', va='bottom',
                     fontsize=10, fontweight='700', color='#1e293b')
        ax1.set_ylabel('Approval Rate (%)', fontsize=9, color='#64748b')
        ax1.set_ylim(0, max(emp.values) * 1.3)
        ax1.tick_params(axis='x', labelsize=9, colors='#475569')
        ax1.tick_params(axis='y', labelsize=8, colors='#94a3b8')
        ax1.grid(axis='y', alpha=0.35, color='#e2e8f0', linewidth=0.8)
        for spine in ax1.spines.values():
            spine.set_color('#e2e8f0'); spine.set_linewidth(0.5)
        plt.tight_layout(pad=0.5)
        st.pyplot(fig1, use_container_width=True)
        plt.close()

    # ── Chart 2: Education ────────────────────────────────────────────────────
    with chart_col2:
        st.markdown('<div style="font-size:0.92rem;font-weight:700;color:#1e293b;margin-bottom:8px">📊 Approval Rate by Education Level</div>',
                    unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(5, 3.2))
        fig2.patch.set_facecolor('white')
        ax2.set_facecolor('white')
        edu_colors = ['#1d4ed8','#2563eb','#3b82f6','#60a5fa']
        bars2 = ax2.bar(edu.index, edu.values, color=edu_colors,
                        edgecolor='white', linewidth=1.5, width=0.5)
        for bar, val in zip(bars2, edu.values):
            ax2.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.1,
                     f'{val:.1f}%', ha='center', va='bottom',
                     fontsize=10, fontweight='700', color='#1e293b')
        ax2.set_ylabel('Approval Rate (%)', fontsize=9, color='#64748b')
        ax2.set_ylim(0, max(edu.values) * 1.3)
        ax2.tick_params(axis='x', labelsize=9, colors='#475569')
        ax2.tick_params(axis='y', labelsize=8, colors='#94a3b8')
        ax2.grid(axis='y', alpha=0.35, color='#e2e8f0', linewidth=0.8)
        for spine in ax2.spines.values():
            spine.set_color('#e2e8f0'); spine.set_linewidth(0.5)
        plt.tight_layout(pad=0.5)
        st.pyplot(fig2, use_container_width=True)
        plt.close()

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # ROW 3 — Model Performance Summary (full width, card grid)
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown('<div style="font-size:1rem;font-weight:700;color:#1e293b;margin-bottom:10px">🤖 Model Performance Summary</div>',
                unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    perf_items = [
        (m1, "Best Model",     meta.get('best_model','Random Forest'), "🏆", "#dbeafe", "#1e40af"),
        (m2, "AUC Score",      f"{meta.get('auc',0.9338):.4f}",        "📈", "#dcfce7", "#15803d"),
        (m3, "Accuracy",       f"{meta.get('accuracy',0.963)*100:.2f}%","✅", "#dcfce7", "#15803d"),
        (m4, "CV AUC 5-fold",  f"{meta.get('cv_auc',0.954):.4f}",      "🔁", "#fef3c7", "#92400e"),
    ]
    for col, label, val, icon, bg, color in perf_items:
        with col:
            st.markdown(f"""
            <div style="background:{bg};border:1.5px solid {color};border-radius:10px;
                        padding:14px 16px;text-align:center">
                <div style="font-size:1.5rem">{icon}</div>
                <div style="font-size:0.78rem;color:#64748b;margin-top:4px;
                            font-weight:600;text-transform:uppercase;
                            letter-spacing:0.5px">{label}</div>
                <div style="font-size:1.15rem;font-weight:800;color:{color};
                            margin-top:4px">{val}</div>
            </div>""", unsafe_allow_html=True)