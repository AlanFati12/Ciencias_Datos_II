"""Assess a patient: step 1 = data form, step 2 = result in plain language.

Functionalities 1-5 + save + PDF. The technical detail (the four models and
the stacking equation) is one click away but never in the way.
"""
import numpy as np
import pandas as pd
import streamlit as st

from core import charts, clinical, report_pdf, storage
from core.clinical import bmi_category, risk_level
from core.i18n import level_label, num, t, validation_message
from core.model_store import require_tier
from core.guides import guide
from core.ui import (callout, chart, chart_header, chip, esc, factors, guide_html, header, html, muted, result_card,
                     section, steps, warn)

tier = require_tier("clinical")
W = tier.ensemble.weights()
DEFAULTS = {"code": "", "age_years": 52.0, "gender": 1, "height": 165.0, "weight": 74.0, "ap_hi": 130.0,
            "ap_lo": 85.0, "cholesterol": 1, "gluc": 1, "smoke": 0, "alco": 0, "active": 1}
st.session_state.setdefault("risk_step", "form")
st.session_state.setdefault("risk_inputs", dict(DEFAULTS))
st.session_state.setdefault("risk_run", 0)

header(t("risk.title"), t("risk.subtitle"), t("nav.sec_patients"), "stethoscope")


def bp_stage(hi: float, lo: float) -> int:
    return int(clinical.ClinicalFeatureEngineer.bp_stage(pd.Series([hi]), pd.Series([lo])).iloc[0])


# =============================================================== step 1: form
def form_step() -> None:
    steps([t("risk.step1"), t("risk.step2")], 0)
    d = st.session_state["risk_inputs"]
    left, right = st.columns(2, gap="medium")

    with left.container(border=True):
        section(t("risk.sec_personal"))
        code = st.text_input(t("common.patient_code"), d["code"], placeholder=t("common.patient_code_ph"),
                             help=t("common.patient_code_help"))
        age = st.number_input(t("in.age"), 18, 100, int(d["age_years"]))
        sex = st.segmented_control(t("in.sex"), [1, 2], default=d["gender"], required=True, width="stretch",
                                   format_func=lambda v: t("in.female" if v == 1 else "in.male"))
    with left.container(border=True):
        section(t("risk.sec_body"))
        a, b = st.columns(2, vertical_alignment="bottom")
        height = a.number_input(t("in.height"), 130, 210, int(d["height"]))
        weight = b.number_input(t("in.weight"), 35.0, 200.0, float(d["weight"]), step=0.5, format="%.1f")
        bmi = weight / (height / 100) ** 2
        cat = bmi_category(bmi)
        html(chip(t("risk.bmi_live", bmi=bmi, cat=t(f"bmi_cat.{cat}")), "low" if cat == 1 else "mid"))

    with right.container(border=True):
        section(t("risk.sec_bp"))
        a, b = st.columns(2, vertical_alignment="bottom")
        ap_hi = a.number_input(t("in.ap_hi"), 80, 250, int(d["ap_hi"]))
        ap_lo = b.number_input(t("in.ap_lo"), 40, 160, int(d["ap_lo"]))
        if ap_hi > ap_lo:
            stage = bp_stage(ap_hi, ap_lo)
            html(chip(t("risk.bp_live", hi=ap_hi, lo=ap_lo, stage=t(f"bp_stage.{stage}")),
                      "low" if stage == 0 else "mid" if stage <= 2 else "high"))
    with right.container(border=True):
        section(t("risk.sec_lab"), t("risk.sec_lab_sub"))
        chol = st.segmented_control(t("in.cholesterol"), [1, 2, 3], default=d["cholesterol"], required=True,
                                    format_func=level_label, width="stretch")
        gluc = st.segmented_control(t("in.glucose"), [1, 2, 3], default=d["gluc"], required=True,
                                    format_func=level_label, width="stretch")
    with right.container(border=True):
        section(t("risk.sec_habits"))
        smoke = st.toggle(t("in.smoke"), bool(d["smoke"]))
        alco = st.toggle(t("in.alcohol"), bool(d["alco"]))
        active = st.toggle(t("in.active"), bool(d["active"]))

    inputs = {"code": code, "age_years": float(age), "gender": sex, "height": float(height), "weight": float(weight),
              "ap_hi": float(ap_hi), "ap_lo": float(ap_lo), "cholesterol": chol, "gluc": gluc,
              "smoke": int(smoke), "alco": int(alco), "active": int(active)}
    errors, warnings = clinical.validate(inputs)
    for e in errors:
        st.error(validation_message(e), icon=":material/error:")
    for w in warnings:
        warn(t("risk.warning." + w))
    if st.button(t("risk.submit"), type="primary", width="stretch", disabled=bool(errors),
                 icon=":material/arrow_forward:", icon_position="right"):
        model_inputs = {k: v for k, v in inputs.items() if k != "code"}
        st.session_state.update(
            risk_inputs=inputs, risk_step="result", risk_run=st.session_state["risk_run"] + 1,
            risk_result=clinical.assess(tier, model_inputs), risk_explain=clinical.explain(tier, model_inputs),
            risk_warnings=warnings)
        st.session_state.pop("risk_saved", None)
        st.rerun()


# ============================================================ step 2: result
def factor_value(key: str, d: dict, res: dict) -> str:
    if key == "age":
        return t("fval.age", v=d["age_years"])
    if key == "sex":
        return t("in.female" if d["gender"] == 1 else "in.male")
    if key == "body":
        return t("risk.bmi_live", bmi=res["derived"]["bmi"], cat=t(f"bmi_cat.{bmi_category(res['derived']['bmi'])}"))
    if key == "blood_pressure":
        return t("risk.bp_live", hi=int(d["ap_hi"]), lo=int(d["ap_lo"]), stage=t(f"bp_stage.{int(res['derived']['bp_stage'])}"))
    if key == "cholesterol":
        return level_label(d["cholesterol"])
    if key == "glucose":
        return level_label(d["gluc"])
    return t(f"fval.{dict(smoking='smoke', alcohol='alco', activity='active')[key]}."
             f"{d[dict(smoking='smoke', alcohol='alco', activity='active')[key]]}")


def result_step() -> None:
    d, res, expl = st.session_state["risk_inputs"], st.session_state["risk_result"], st.session_state["risk_explain"]
    p = res["final_probability"]
    level = risk_level(p, res["referred_to_clinician"])
    model_inputs = {k: v for k, v in d.items() if k != "code"}

    top = st.container(horizontal=True)
    if top.button(t("common.edit"), icon=":material/arrow_back:"):
        st.session_state["risk_step"] = "form"
        st.rerun()
    if top.button(t("common.new"), icon=":material/add:"):
        st.session_state.update(risk_step="form", risk_inputs=dict(DEFAULTS))
        st.rerun()
    steps([t("risk.step1"), t("risk.step2")], 1)
    for w in st.session_state.get("risk_warnings", []):
        warn(t("risk.warning." + w))

    result_card(level, t("risk.tag") + (f" · {esc(d['code'])}" if d["code"] else ""), num(p, ".0%"),
                t(f"risk.level.{level}"), t(f"risk.msg.{level}"), t("risk.big_sub", n=round(p * 100)),
                bar=p, bar_labels=(t("risk.bar_low"), t("risk.bar_mid"), t("risk.bar_high")))
    actions = st.container(horizontal=True)
    with actions.popover(t("guide.result_button"), icon=":material/help:"):
        html(guide_html(guide("result")))
    if level != "low":
        if actions.button(t("risk.cta_ecg"), icon=":material/ecg_heart:"):
            st.switch_page("views/ecg_analysis.py")

    tab_why, tab_sim, tab_tech = st.tabs([t("risk.tab_why"), t("risk.tab_sim"), t("risk.tab_tech")])

    # ---------------------------------------------------- why (functionalities 3-4)
    with tab_why:
        with st.container(border=True):
            chart_header(t("risk.why_title"), t("risk.why_sub"), "factors")
            shown = expl[expl["effect"].abs() >= 0.01].copy()
            shown = shown.reindex(shown["effect"].abs().sort_values(ascending=False).index)
            if shown.empty:
                callout(t("risk.why_none"))
            else:
                top = float(shown["effect"].abs().max())
                factors([{"up": r.effect > 0, "name": t(f"factor.{r.factor}"), "value": factor_value(r.factor, d, res),
                          "points": t("common.points", v=r.effect * 100), "size": abs(r.effect) / top}
                         for r in shown.itertuples()])
                if len(shown) < len(expl):
                    muted(t("risk.why_rest"))
        st.write("")
        with st.container(border=True):
            seg = res["risk_segment"]
            profiles = tier.schema.get("segment_profiles", {})
            if seg and seg in profiles:
                section(t("risk.group_title"))
                html(f'<p style="margin:.3rem 0 0">{t("risk.group_text", level=t("segment." + seg), n=round(profiles[seg]["cvd_prevalence"] * 100))} '
                     f'{chip("K-means", "info")}</p>')

    # --------------------------------------------------------- simulate (5)
    with tab_sim:
        muted(t("risk.sim_intro"))
        st.write("")
        run = st.session_state["risk_run"]
        left, right = st.columns([3, 2], gap="large")
        sim = dict(model_inputs)
        with left:
            a, b = st.columns(2)
            sim["ap_hi"] = float(a.slider(t("in.ap_hi"), 90, 220, int(d["ap_hi"]), key=f"sim_hi_{run}"))
            sim["ap_lo"] = float(b.slider(t("in.ap_lo"), 50, 140, int(d["ap_lo"]), key=f"sim_lo_{run}"))
            sim["weight"] = float(st.slider(t("in.weight"), 40, 160, int(d["weight"]), key=f"sim_w_{run}"))
            sim["cholesterol"] = st.segmented_control(t("in.cholesterol"), [1, 2, 3], default=d["cholesterol"],
                                                      required=True, format_func=level_label, key=f"sim_c_{run}",
                                                      width="stretch")
            sim["gluc"] = st.segmented_control(t("in.glucose"), [1, 2, 3], default=d["gluc"], required=True,
                                               format_func=level_label, key=f"sim_g_{run}", width="stretch")
            sim["smoke"] = int(st.toggle(t("in.smoke"), bool(d["smoke"]), key=f"sim_s_{run}"))
            sim["active"] = int(st.toggle(t("in.active"), bool(d["active"]), key=f"sim_a_{run}"))
        with right:
            if sim["ap_hi"] <= sim["ap_lo"]:
                st.error(t("risk.error.bp_order"))
            else:
                p_sim = clinical.predict_frame(tier, pd.DataFrame([sim]))["probability"].iloc[0]
                with st.container(border=True):
                    st.metric(t("risk.sim_now"), num(p, ".0%"))
                    change = (p_sim - p) * 100
                    st.metric(t("risk.sim_new"), num(p_sim, ".0%"), delta_color="inverse",
                              delta=t("common.points", v=change) if abs(change) >= 0.5 else None)
        if sim["ap_hi"] > sim["ap_lo"]:
            with st.container(border=True):
                chart_header(t("risk.sim_curve"), t("risk.sim_curve_sub"), "sweep")
                curve = clinical.sweep(tier, sim, "ap_hi", np.arange(95, 215, 5))
                chart(charts.sweep_line(curve, "ap_hi", sim["ap_hi"], t("in.ap_hi") + " (mmHg)"))

    # ---------------------------------------------------- technical detail (2)
    with tab_tech:
        html(f'<p>{t("risk.tech_intro")}</p>')
        rows = [(k, t(f"model.{k}"), v) for k, v in res["base_probabilities"].items()] + [("final", t("model.final"), p)]
        with st.container(border=True):
            chart_header(t("ecg.models_title"), t("risk.tech_votes", n=sum(res["base_votes"].values())), "opinions")
            chart(charts.opinions(rows))
        st.write("")
        section(t("risk.tech_equation"))
        terms = " + ".join(f"({W[k]:.2f})({v:.3f})" for k, v in res["base_probabilities"].items())
        st.latex(rf"z = {W['intercept']:.2f} + {terms} = {res['meta_logit']:.3f}")
        st.latex(rf"\hat P = \frac{{1}}{{1+e^{{-z}}}} = {p:.3f}")
        table = pd.DataFrame({t("risk.col_model"): [t(f"model.{k}") for k in res["base_probabilities"]],
                              t("risk.col_prob"): list(res["base_probabilities"].values()),
                              t("risk.col_weight"): [W[k] for k in res["base_probabilities"]],
                              t("risk.col_contrib"): list(res["meta_contributions"].values())})
        st.dataframe(table, hide_index=True, width="stretch",
                     column_config={t("risk.col_prob"): st.column_config.NumberColumn(format="percent"),
                                    t("risk.col_weight"): st.column_config.NumberColumn(format="%.2f"),
                                    t("risk.col_contrib"): st.column_config.NumberColumn(format="%+.3f")})
        muted(t("risk.referral_rule"))

    # ------------------------------------------------------ save + PDF report
    st.write("")
    section(t("risk.save_title"))
    row = st.container(horizontal=True)
    if row.button(t("common.save"), icon=":material/save:", disabled="risk_saved" in st.session_state):
        st.session_state["risk_saved"] = storage.save(
            "clinical", d["code"], p, level, model_inputs, {k: v for k, v in res.items() if k != "weights"},
            res["risk_segment"], res["referred_to_clinician"])
        st.rerun()
    row.download_button(t("common.pdf"), build_pdf(d, res, expl, level), file_name=f"riesgo_{d['code'] or 'paciente'}.pdf",
                        mime="application/pdf", icon=":material/picture_as_pdf:")
    if "risk_saved" in st.session_state:
        st.success(t("common.saved", id=st.session_state["risk_saved"]), icon=":material/check_circle:")


def build_pdf(d: dict, res: dict, expl: pd.DataFrame, level: str) -> bytes:
    p = res["final_probability"]
    yes_no = lambda v: t("common.yes") if v else t("common.no")  # noqa: E731
    patient = [(t("common.patient_code"), d["code"] or "-"), (t("in.age"), f"{d['age_years']:.0f}"),
               (t("in.sex"), t("in.female" if d["gender"] == 1 else "in.male")),
               (t("in.height"), f"{d['height']:.0f}"), (t("in.weight"), num(d["weight"], ".1f")),
               (t("risk.sec_bp"), f"{d['ap_hi']:.0f}/{d['ap_lo']:.0f}"),
               (t("in.cholesterol"), level_label(d["cholesterol"])), (t("in.glucose"), level_label(d["gluc"])),
               (t("in.smoke"), yes_no(d["smoke"])), (t("in.alcohol"), yes_no(d["alco"])),
               (t("in.active"), yes_no(d["active"]))]
    result = [(t("pdf.result"), t(f"risk.level.{level}")), (t("pdf.probability"), num(p, ".0%"))]
    if res["risk_segment"]:
        result.append((t("pdf.group"), t("segment." + res["risk_segment"])))
    factor_rows = [(t(f"factor.{r.factor}"), t("common.points", v=r.effect * 100))
                   for r in expl.itertuples() if abs(r.effect) >= 0.01]
    models = [(t(f"model.{k}"), num(v, ".0%")) for k, v in res["base_probabilities"].items()]
    models.append((t("model.final"), num(p, ".0%")))
    sections = [(t("pdf.patient"), patient), (t("pdf.result"), result),
                (t("pdf.recommendation"), t(f"risk.msg.{level}"))]
    if factor_rows:
        sections.append((t("pdf.factors"), factor_rows))
    sections.append((t("pdf.models"), models))
    return report_pdf.build(t("pdf.clinical_title"), sections, t("common.disclaimer"))


if st.session_state["risk_step"] == "result" and "risk_result" in st.session_state:
    result_step()
else:
    form_step()
