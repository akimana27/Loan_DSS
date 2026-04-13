"""
LOAN APPROVAL RISK ANALYSIS DSS
Main entry-point — run with:  streamlit run app.py
"""
import streamlit as st

st.set_page_config(
    page_title="Loan DSS",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Hide Streamlit default nav & branding ── */
[data-testid="stSidebarNav"] { display: none !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.2rem !important; padding-bottom: 1rem !important; }

/* ── Root variables ── */
:root {
    --bg: #f0f4f8;
    --card: #ffffff;
    --card2: #f8fafc;
    --border: #e2e8f0;
    --border2: #cbd5e1;
    --accent: #1e40af;
    --accent-light: #dbeafe;
    --accent2: #0369a1;
    --green: #15803d;
    --green-light: #dcfce7;
    --red: #b91c1c;
    --red-light: #fee2e2;
    --yellow: #92400e;
    --yellow-light: #fef3c7;
    --text: #1e293b;
    --text2: #475569;
    --muted: #94a3b8;
    --sidebar: #1e293b;
    --sidebar-text: #e2e8f0;
    --sidebar-muted: #94a3b8;
    --sidebar-active: #3b82f6;
    --sidebar-active-bg: rgba(59,130,246,0.15);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--sidebar) !important;
    border-right: 1px solid rgba(255,255,255,0.08) !important;
    width: 220px !important;
}
[data-testid="stSidebar"] > div { padding: 0 !important; }
[data-testid="stSidebar"] * { color: var(--sidebar-text) !important; }
[data-testid="stSidebar"] .stRadio label { display: none !important; }
[data-testid="stSidebar"] .stRadio > div {
    display: flex !important;
    flex-direction: column !important;
    gap: 2px !important;
}
[data-testid="stSidebar"] .stRadio > div > label {
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
    padding: 9px 16px !important;
    border-radius: 8px !important;
    cursor: pointer !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: var(--sidebar-muted) !important;
    transition: all 0.15s !important;
    margin: 0 !important;
}
[data-testid="stSidebar"] .stRadio > div > label:hover {
    background: rgba(255,255,255,0.06) !important;
    color: white !important;
}
[data-testid="stSidebar"] .stRadio > div > label[data-checked="true"] {
    background: var(--sidebar-active-bg) !important;
    color: #93c5fd !important;
    border-left: 3px solid var(--sidebar-active) !important;
}
[data-testid="stSidebar"] .stRadio input { display: none !important; }

/* ── Page header ── */
.page-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #1e40af 60%, #1d4ed8 100%);
    border-radius: 12px;
    padding: 20px 28px;
    margin-bottom: 20px;
    border: 1px solid #2d5a9e;
}
.page-header h1 {
    margin: 0;
    font-size: 1.5rem;
    font-weight: 700;
    color: #ffffff !important;
    letter-spacing: -0.3px;
}
.page-header p { margin: 5px 0 0; color: #93c5fd !important; font-size: 0.85rem; }

/* ── Metric cards ── */
div[data-testid="stMetric"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 14px 16px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
}
div[data-testid="stMetric"] label {
    color: var(--text2) !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
}

/* ── Cards ── */
.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.card-title {
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--text2);
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 4px !important;
    gap: 2px !important;
}
.stTabs [data-baseweb="tab"] {
    color: var(--text2) !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    border-radius: 6px !important;
    padding: 6px 14px !important;
}
.stTabs [aria-selected="true"] {
    background: var(--accent) !important;
    color: white !important;
}

/* ── Form elements ── */
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: var(--card) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 7px !important;
    color: var(--text) !important;
    font-size: 13px !important;
}
.stSlider [data-testid="stSlider"] { padding: 0 !important; }

/* ── Buttons ── */
.stButton > button {
    background: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 20px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    letter-spacing: 0.3px !important;
    transition: all 0.15s !important;
}
.stButton > button:hover { background: #1d4ed8 !important; }

/* ── Decision badges ── */
.decision-approve {
    background: var(--green-light);
    border: 2px solid #16a34a;
    border-radius: 12px;
    padding: 18px 24px;
    text-align: center;
    margin-bottom: 16px;
}
.decision-reject {
    background: var(--red-light);
    border: 2px solid #dc2626;
    border-radius: 12px;
    padding: 18px 24px;
    text-align: center;
    margin-bottom: 16px;
}
.decision-review {
    background: var(--yellow-light);
    border: 2px solid #d97706;
    border-radius: 12px;
    padding: 18px 24px;
    text-align: center;
    margin-bottom: 16px;
}
.decision-label-approve { font-size: 2rem; font-weight: 800; color: #15803d; letter-spacing: 3px; }
.decision-label-reject  { font-size: 2rem; font-weight: 800; color: #b91c1c; letter-spacing: 3px; }
.decision-label-review  { font-size: 2rem; font-weight: 800; color: #92400e; letter-spacing: 3px; }
.decision-sub { font-size: 0.85rem; margin-top: 5px; opacity: 0.85; }

/* ── Factor items ── */
.factor-pos {
    background: var(--green-light);
    border-left: 3px solid #16a34a;
    border-radius: 6px;
    padding: 7px 12px;
    margin-bottom: 6px;
    color: #15803d;
    font-size: 0.82rem;
    font-weight: 500;
}
.factor-neg {
    background: var(--yellow-light);
    border-left: 3px solid #d97706;
    border-radius: 6px;
    padding: 7px 12px;
    margin-bottom: 6px;
    color: #92400e;
    font-size: 0.82rem;
    font-weight: 500;
}
.factor-bad {
    background: var(--red-light);
    border-left: 3px solid #dc2626;
    border-radius: 6px;
    padding: 7px 12px;
    margin-bottom: 6px;
    color: #b91c1c;
    font-size: 0.82rem;
    font-weight: 500;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: var(--card2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}

/* ── Dataframe ── */
.stDataFrame { border: 1px solid var(--border) !important; border-radius: 8px !important; }

/* ── Info/warning boxes ── */
.stAlert { border-radius: 8px !important; font-size: 13px !important; }

/* ── Warning about deprecated params — hide it ── */
div[data-testid="stWarningMessage"] { display: none !important; }
.element-container:has(div[class*="stWarning"]) { display: none !important; }

/* ── Spinner ── */
.stSpinner { color: var(--accent) !important; }

/* ── Section headers ── */
.section-title {
    font-size: 1rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Arch boxes ── */
.arch-box {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 14px;
    text-align: center;
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--text);
}
.arch-box span { display: block; font-size: 0.72rem; color: var(--muted); font-weight: 400; margin-top: 2px; }

/* ── Insight rows ── */
.insight-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.85rem;
}
.insight-row:last-child { border: none; }

/* ── Form section card ── */
.form-section {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 14px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.form-section-title {
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--text2);
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-bottom: 14px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:20px 16px 16px;border-bottom:1px solid rgba(255,255,255,0.08)">
        <div style="font-size:1.4rem;margin-bottom:6px">🏦</div>
        <div style="font-size:0.95rem;font-weight:700;color:#f1f5f9;letter-spacing:-0.3px">LoanGuard DSS</div>
        <div style="font-size:0.72rem;color:#64748b;margin-top:2px">Decision Support System</div>
    </div>
    <div style="padding:8px 8px 4px;margin-top:4px">
        <div style="font-size:0.65rem;color:#475569;font-weight:600;letter-spacing:1px;padding:4px 8px;margin-bottom:4px">NAVIGATION</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("nav", [
        " Overview",
        " Single Applicant",
        "Batch Analysis",
        "Model Insights",
        "EDA Explorer"
    ], label_visibility="collapsed")

    st.markdown("""
    <div style="position:absolute;bottom:0;left:0;right:0;padding:12px 16px;
                border-top:1px solid rgba(255,255,255,0.06)">
        <div style="font-size:0.68rem;color:#475569;line-height:1.7">
            v1.0 · DSS Group Project<br>
            Loan Risk Analysis System
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Route pages ───────────────────────────────────────────────────────────────
if "Overview" in page:
    from pages import overview; overview.render()
elif "Single" in page:
    from pages import single_applicant; single_applicant.render()
elif "Batch" in page:
    from pages import batch_analysis; batch_analysis.render()
elif "Model Insights" in page:
    from pages import model_insights; model_insights.render()
elif "EDA" in page:
    from pages import eda_explorer; eda_explorer.render()