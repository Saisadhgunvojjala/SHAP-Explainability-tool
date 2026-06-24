"""
backend/auth.py — Authentication using streamlit-authenticator==0.3.2
"""

import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import os

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")


def load_authenticator():
    with open(CONFIG_PATH) as f:
        config = yaml.load(f, Loader=SafeLoader)

    authenticator = stauth.Authenticate(
        config["credentials"],
        config["cookie"]["name"],
        config["cookie"]["key"],
        config["cookie"]["expiry_days"],
    )
    return authenticator, config


def login_page():
    authenticator, config = load_authenticator()

    # Returns (name, auth_status, username) tuple
    result = authenticator.login(location="main")

    if result is not None:
        name, auth_status, username = result
    else:
        name        = st.session_state.get("name")
        auth_status = st.session_state.get("authentication_status")
        username    = st.session_state.get("username")

    return authenticator, name, auth_status, username


def logout_button(authenticator, location="sidebar"):
    authenticator.logout(location=location)


def generate_password(plain_password: str) -> str:
    import bcrypt
    return bcrypt.hashpw(plain_password.encode(), bcrypt.gensalt()).decode()
