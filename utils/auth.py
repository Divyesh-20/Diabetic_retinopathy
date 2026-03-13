"""
utils/auth.py – Session-state-based authentication
"""

import streamlit as st
from config import CREDENTIALS


def login(username: str, password: str) -> bool:
    """Attempt login. Returns True on success."""
    cred = CREDENTIALS.get(username)
    if cred and cred["password"] == password:
        st.session_state["logged_in"] = True
        st.session_state["username"]  = username
        st.session_state["role"]      = cred["role"]
        return True
    return False


def logout():
    """Clear session state."""
    for key in ["logged_in", "username", "role", "page"]:
        st.session_state.pop(key, None)


def is_logged_in() -> bool:
    return st.session_state.get("logged_in", False)


def get_role() -> str:
    return st.session_state.get("role", "")


def require_role(required_role: str):
    """Redirect to login if role doesn't match."""
    if not is_logged_in() or get_role() != required_role:
        st.session_state["page"] = "login"
        st.warning("Access denied. Please log in with the correct credentials.")
        st.stop()
