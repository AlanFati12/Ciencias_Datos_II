"""Functionality 9: history of saved assessments, readable at a glance."""
import json

import pandas as pd
import streamlit as st

from core import storage
from core.clinical import risk_level
from core.i18n import level_label, num, t
from core.ui import header, html, muted, result_card

header(t("hist.title"), t("hist.subtitle"), t("nav.sec_records"), "history")
df = storage.load_all()

if df.empty:
    with st.container(border=True):
        html(f'<div class="cv-section">{t("hist.empty_title")}</div>')
        muted(t("hist.empty_text"))
        st.write("")
        row = st.container(horizontal=True)
        if row.button(t("nav.risk"), icon=":material/stethoscope:"):
            st.switch_page("views/risk.py")
        if row.button(t("nav.ecg"), icon=":material/ecg_heart:"):
            st.switch_page("views/ecg_analysis.py")
    st.stop()

def level_of(row) -> str:
    if row["kind"] == "clinical":
        return risk_level(row["probability"], bool(row["referred"]))
    burden = row["probability"]
    return "low" if burden < 0.01 else "mid" if burden < 0.10 else "high"


def result_text(row) -> str:
    level = level_of(row)
    if row["kind"] == "clinical":
        return t("hist.res_clinical", level=t(f"risk.level.{level}"), p=row["probability"])
    return t("hist.res_ecg", level=t(f"ecg.level.{level}"), p=row["probability"])


df["level"] = df.apply(level_of, axis=1)
df["when"] = pd.to_datetime(df["created_at"]).dt.strftime("%d/%m %H:%M")

kind = st.segmented_control(t("hist.filter"), ["all", "clinical", "ecg"], default="all", required=True,
                            format_func=lambda k: t("hist.all") if k == "all" else t(f"hist.type_{k}"))
query = st.text_input(t("hist.search"), placeholder=t("common.patient_code_ph"))
view = df if kind == "all" else df[df["kind"] == kind]
if query:
    view = view[view["patient_code"].fillna("").str.contains(query, case=False, regex=False)]

table = pd.DataFrame({
    t("hist.col_date"): view["when"],
    t("hist.col_type"): view["kind"].map(lambda k: t(f"hist.type_{k}")),
    t("hist.col_patient"): view["patient_code"].fillna(t("hist.no_code")),
    t("hist.col_result"): [result_text(r) for _, r in view.iterrows()],
})
muted(t("hist.count", n=len(view)) + " · " + t("hist.select_hint"))
event = st.dataframe(table, hide_index=True, width="stretch", on_select="rerun", selection_mode="multi-row")
selected = view.iloc[event.selection.rows]["id"].tolist() if event.selection.rows else []

if len(selected) == 1:
    row = df[df["id"] == selected[0]].iloc[0]
    st.subheader(t("hist.detail", id=row["id"], date=row["when"]))
    inputs, details = json.loads(row["inputs_json"]), json.loads(row["details_json"])
    level = row["level"]
    if row["kind"] == "clinical":
        result_card(level, t("risk.tag"), num(row["probability"], ".0%"), t(f"risk.level.{level}"), t(f"risk.msg.{level}"))
        friendly = {t("in.age"): f"{inputs['age_years']:.0f}",
                    t("in.sex"): t("in.female" if inputs["gender"] == 1 else "in.male"),
                    t("in.height"): f"{inputs['height']:.0f}", t("in.weight"): num(inputs["weight"], ".1f"),
                    t("risk.sec_bp"): f"{inputs['ap_hi']:.0f}/{inputs['ap_lo']:.0f}",
                    t("in.cholesterol"): level_label(inputs["cholesterol"]), t("in.glucose"): level_label(inputs["gluc"]),
                    t("in.smoke"): t("common.yes") if inputs["smoke"] else t("common.no"),
                    t("in.alcohol"): t("common.yes") if inputs["alco"] else t("common.no"),
                    t("in.active"): t("common.yes") if inputs["active"] else t("common.no")}
    else:
        result_card(level, t("ecg.tag"), num(row["probability"], ".0%"), t(f"ecg.level.{level}"), t(f"ecg.msg.{level}"))
        friendly = {t("ecg.source_q"): inputs.get("source", "-"),
                    t("ecg.stat_beats"): num(details.get("beats", 0)),
                    t("ecg.stat_hr"): t("ecg.stat_hr_v", v=details.get("heart_rate", 0))}
    html(f'<div class="cv-section">{t("hist.inputs")}</div>')
    st.dataframe(pd.DataFrame(friendly.items(), columns=["", " "]), hide_index=True, width="stretch")
    with st.expander(t("hist.raw")):
        st.json({"inputs": inputs, "details": details}, expanded=False)


@st.dialog(t("hist.confirm_title"))
def confirm_delete(ids: list[int]) -> None:
    st.write(t("hist.confirm_text", n=len(ids)))
    a, b = st.columns(2)
    if a.button(t("hist.confirm_yes"), type="primary", width="stretch"):
        storage.delete(ids)
        st.rerun()
    if b.button(t("hist.cancel"), width="stretch"):
        st.rerun()


row = st.container(horizontal=True)
row.download_button(t("hist.export"), view.drop(columns=["level", "when"]).to_csv(index=False).encode(),
                    "historial.csv", "text/csv", icon=":material/download:")
if selected and row.button(t("hist.delete", n=len(selected)), icon=":material/delete:"):
    confirm_delete(selected)
