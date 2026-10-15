"""Home: what the app is, two clear actions and a three-step explanation. No charts, no jargon."""
import streamlit as st

from core.i18n import t
from core.model_store import tier_kpis
from core.ui import footer, hero, html, icon, section

clinical, ecg = tier_kpis("clinical"), tier_kpis("ecg")
badges = [("hub", t("home.badge_models")), ("verified", t("home.badge_lang"))]
if clinical and ecg:
    badges.insert(1, ("database", t("home.badge_data", n=clinical["rows"], b=ecg["rows"])))
hero(t("home.title"), t("home.subtitle"), badges, eyebrow=t("home.eyebrow"))

section(t("home.start_here"))
c1, c2 = st.columns(2, gap="medium")
CARDS = ((c1, "risk", "stethoscope", "#EFF4FF", "#2563EB", "views/risk.py", ("schedule", "edit_note")),
         (c2, "ecg", "ecg_heart", "#FEF1F1", "#DC2626", "views/ecg_analysis.py", ("folder_open", "monitor_heart")))
for col, key, ic, bg, fg, page, meta_icons in CARDS:
    with col.container(border=True):
        meta = t(f"home.{key}_meta").split(" · ")
        meta_html = "".join(f"<span>{icon(i)}{m}</span>" for i, m in zip(meta_icons, meta))
        html(f'<div class="cv-card-icon" style="background:{bg};color:{fg}">{icon(ic)}</div>'
             f'<div class="cv-card-title">{t(f"home.{key}_title")}</div>'
             f'<p class="cv-card-text">{t(f"home.{key}_text")}</p><div class="cv-meta">{meta_html}</div>')
        if st.button(t(f"home.{key}_btn"), type="primary", width="stretch", key=f"go_{key}",
                     icon=":material/arrow_forward:", icon_position="right"):
            st.switch_page(page)

st.write("")
st.subheader(t("home.how_title"))
cols = st.columns(3, gap="medium")
for i, (col, ic) in enumerate(zip(cols, ("edit_note", "hub", "task_alt")), start=1):
    with col.container(border=True):
        html(f'<div class="cv-num">{icon(ic)}</div><div class="cv-section">{i}. {t(f"home.how{i}_t")}</div>'
             f'<div class="cv-muted">{t(f"home.how{i}")}</div>')

st.write("")
links = st.columns(3)
links[0].page_link("views/how_it_works.py", label=t("home.trust_link"), icon=":material/lightbulb:")
links[1].page_link("views/history.py", label=t("home.more_history"), icon=":material/history:")
links[2].page_link("views/batch.py", label=t("home.more_batch"), icon=":material/group:")

footer(t("common.disclaimer"))
