"""CardioAI — telemedicine web app (Streamlit).

    streamlit run app.py
"""
from pathlib import Path

import streamlit as st

ASSETS = Path(__file__).parent / "assets"
st.set_page_config(page_title="CardioAI", page_icon=str(ASSETS / "icon.svg"), layout="wide")

from core.i18n import DEFAULT_LANG, LANGS, t  # noqa: E402
from core.ui import html, inject_css, tour  # noqa: E402

if "lang" not in st.session_state:                      # remembered in the URL across reloads
    st.session_state["lang"] = st.query_params.get("lang") if st.query_params.get("lang") in LANGS else DEFAULT_LANG
st.query_params["lang"] = st.session_state["lang"]
if st.query_params.get("guide") == "off":               # the tour was already seen (kept in the URL like the language)
    st.session_state["onboarded"] = True
if st.session_state.get("onboarded"):
    st.query_params["guide"] = "off"
inject_css()
st.logo(str(ASSETS / "logo.svg"), size="large", icon_image=str(ASSETS / "icon.svg"))

pages = {
    "": [st.Page("views/home.py", title=t("nav.home"), icon=":material/home:", default=True)],
    t("nav.sec_patients"): [
        st.Page("views/risk.py", title=t("nav.risk"), icon=":material/stethoscope:"),
        st.Page("views/ecg_analysis.py", title=t("nav.ecg"), icon=":material/ecg_heart:"),
    ],
    t("nav.sec_records"): [
        st.Page("views/history.py", title=t("nav.history"), icon=":material/history:"),
        st.Page("views/batch.py", title=t("nav.batch"), icon=":material/group:"),
    ],
    t("nav.sec_info"): [
        st.Page("views/how_it_works.py", title=t("nav.how"), icon=":material/lightbulb:"),
    ],
}
nav = st.navigation(pages)

with st.sidebar:
    st.segmented_control(t("app.language"), list(LANGS), format_func=LANGS.get, key="lang", required=True)
    if st.button(t("app.tour"), icon=":material/explore:", width="stretch"):
        st.session_state["tour_step"] = 0
        tour()
    html(f'<div class="cv-side-note">{t("app.side_note")}</div>')

if not st.session_state.get("onboarded"):          # first visit of the session: short guided tour
    tour()

nav.run()
