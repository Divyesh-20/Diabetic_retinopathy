"""
pages/login.py – Login page for the DR Detection System
"""

import streamlit as st
from utils.auth import login, is_logged_in


def show_login_page():
    """Render the login page."""
    st.markdown("""
    <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; padding-top:40px;">
        <div style="font-size:3rem; margin-bottom:8px;">👁️</div>
        <h1 style="font-size:2rem; background:linear-gradient(135deg,#1565C0,#7B1FA2);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                   margin-bottom:4px;">
            DR Detect
        </h1>
        <p style="color:#90CAF9; font-size:1rem; margin-bottom:40px; text-align:center;">
            Diabetic Retinopathy Detection System<br>
            <span style="font-size:0.85rem; color:#607D8B;">AI-powered fundus image analysis</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form", clear_on_submit=False):
            st.markdown("#### 🔐 Sign In")
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            submitted = st.form_submit_button("Login", use_container_width=True)

            if submitted:
                if not username or not password:
                    st.error("Please enter both username and password.")
                elif login(username, password):
                    st.success(f"Welcome, **{username}**! Redirecting…")
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials. Please try again.")

        st.caption("🩺 Admin: `admin / admin123` &nbsp;|&nbsp; 👤 User: `user / user123`")
