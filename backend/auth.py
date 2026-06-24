"""
backend/auth.py — Authentication using streamlit-authenticator==0.3.2

Loads credentials from (in order of priority):
  1. st.secrets (Streamlit Cloud deployment — set via dashboard Secrets box)
  2. local config.yaml (local development — gitignored, never pushed)

This lets the exact same code run locally and on Streamlit Cloud without
any environment-specific branching elsewhere in the app.
"""

import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import os

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")


def _secrets_to_dict(secrets_obj) -> dict:
    """
    Recursively convert Streamlit's AttrDict/Secrets object into a plain
    Python dict, since streamlit_authenticator expects plain dicts/lists,
    not Streamlit's special mapping type.
    """
    if hasattr(secrets_obj, "to_dict"):
        secrets_obj = secrets_obj.to_dict()
    if isinstance(secrets_obj, dict):
        return {k: _secrets_to_dict(v) for k, v in secrets_obj.items()}
    if isinstance(secrets_obj, list):
        return [_secrets_to_dict(v) for v in secrets_obj]
    return secrets_obj


def load_authenticator():
    """
    Load auth config from Streamlit secrets if available (Cloud deployment),
    otherwise fall back to the local config.yaml file (local development).
    """
    config = None

    # ── Try Streamlit secrets first (works on Streamlit Cloud) ────────────────
    try:
        if "credentials" in st.secrets:
            config = {
                "credentials": _secrets_to_dict(st.secrets["credentials"]),
                "cookie": _secrets_to_dict(st.secrets["cookie"]),
                "preauthorized": _secrets_to_dict(st.secrets.get("preauthorized", {"emails": []})),
            }
    except Exception:
        # st.secrets raises if no secrets.toml / dashboard secrets exist at all —
        # that's expected for local dev, just fall through to the file below.
        config = None

    # ── Fall back to local config.yaml ─────────────────────────────────────────
    if config is None:
        if not os.path.exists(CONFIG_PATH):
            st.error(
                "⚠️ No authentication config found. Locally: create `config.yaml` "
                "in the project root (see `config.yaml.example`). "
                "On Streamlit Cloud: add credentials in the app's Secrets settings."
            )
            st.stop()
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
