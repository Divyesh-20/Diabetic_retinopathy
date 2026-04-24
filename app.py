"""
app.py – Main entry point for the DR Detection Streamlit application
"""

import streamlit as st

# ── Page config MUST be first ─────────────────────────────────────────────────
st.set_page_config(
    page_title="DR Detect – Diabetic Retinopathy Detection",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS theme ──────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ─── Base ─────────────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* ─── Dark background ────────────────────────────────────────────────────── */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1421 50%, #0a1628 100%);
    color: #e0e6f0;
}

/* ─── Sidebar ─────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1421 0%, #111827 100%);
    border-right: 1px solid #1e293b;
}
[data-testid="stSidebar"] * {
    color: #cbd5e1 !important;
}

/* ─── Buttons ─────────────────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #1565C0, #7B1FA2);
    color: white !important;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(21,101,192,0.4);
}

/* ─── Metrics ─────────────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 12px 16px;
}
[data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 0.8rem; }
[data-testid="stMetricValue"] { color: #e2e8f0 !important; font-size: 1.4rem; font-weight: 700; }

/* ─── Cards / expander ────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 10px;
}

/* ─── Tables ──────────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: 8px;
    overflow: hidden;
}

/* ─── File uploader ────────────────────────────────────────────────────────── */
[data-testid="stFileUploaderDropzone"] {
    background: #111827;
    border: 2px dashed #334155;
    border-radius: 10px;
}

/* ─── Divider ─────────────────────────────────────────────────────────────── */
hr { border-color: #1e293b !important; }

/* ─── Slider ──────────────────────────────────────────────────────────────── */
[data-testid="stSlider"] > div > div > div {
    background: linear-gradient(90deg, #1565C0, #7B1FA2) !important;
}

/* ─── Input fields ────────────────────────────────────────────────────────── */
.stTextInput input, .stSelectbox select {
    background: #111827 !important;
    color: #e2e8f0 !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
}

/* ─── Download button ──────────────────────────────────────────────────────── */
.stDownloadButton > button {
    background: linear-gradient(135deg, #0d7377, #14a085);
    color: white !important;
    border: none;
    border-radius: 8px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ── Auth & Routing ────────────────────────────────────────────────────────────
from utils.auth import is_logged_in, get_role, logout

# Initialise page state
if "page" not in st.session_state:
    st.session_state["page"] = "login"

# If not logged in, always show login
if not is_logged_in():
    from pages.login import show_login_page
    show_login_page()
    st.stop()

# ── Sidebar Navigation ────────────────────────────────────────────────────────
role = get_role()

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:10px 0 20px;">
        <div style="font-size:2.5rem;">👁️</div>
        <div style="font-size:1.3rem; font-weight:700;
                    background:linear-gradient(135deg,#42A5F5,#AB47BC);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            DR Detect
        </div>
        <div style="font-size:0.75rem; color:#607D8B; margin-top:2px;">
            Diabetic Retinopathy AI
        </div>
    </div>
    """, unsafe_allow_html=True)

    username = st.session_state.get("username", "")
    role_badge = "🔴 Admin" if role == "admin" else "🟢 User"
    st.info(f"**{role_badge}** signed in as `{username}`")
    st.divider()

    if role == "admin":
        st.markdown("**🛡️ Admin Panel**")
        if st.button("🏠 Dashboard",     use_container_width=True):
            st.session_state["page"] = "admin_dashboard"
            st.rerun()
        if st.button("🏋️ Train Model",   use_container_width=True):
            st.session_state["page"] = "admin_train"
            st.rerun()
        if st.button("📈 Evaluate",       use_container_width=True):
            st.session_state["page"] = "admin_evaluate"
            st.rerun()
        if st.button("⚖️ Compare Models", use_container_width=True):
            st.session_state["page"] = "admin_compare"
            st.rerun()

    else:  # user
        st.markdown("**👤 User Panel**")
        if st.button("🔬 Detect DR",     use_container_width=True):
            st.session_state["page"] = "user_detect"
            st.rerun()

    st.divider()
    if st.button("🚪 Logout", use_container_width=True):
        logout()
        st.rerun()


# ── Route to correct page ─────────────────────────────────────────────────────
page = st.session_state.get("page", "")

# Default page per role
if page == "login" or page == "":
    if role == "admin":
        st.session_state["page"] = "admin_dashboard"
    else:
        st.session_state["page"] = "user_detect"
    st.rerun()

if   page == "admin_dashboard":
    from pages.admin.dashboard import show_dashboard_page
    show_dashboard_page()

elif page == "admin_train":
    from pages.admin.train import show_train_page
    show_train_page()

elif page == "admin_evaluate":
    from pages.admin.evaluate import show_evaluate_page
    show_evaluate_page()

elif page == "admin_compare":
    from pages.admin.compare import show_compare_page
    show_compare_page()

elif page == "user_detect":
    from pages.user.detect import show_detect_page
    show_detect_page()

else:
    st.error(f"Unknown page: {page}")
