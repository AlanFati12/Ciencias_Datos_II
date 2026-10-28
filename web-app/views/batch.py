"""Functionality 8: assess several patients from one CSV file, in three steps."""
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix

from core import clinical, files
from core.clinical import risk_level
from core.i18n import t, validation_message
from core.model_store import SAMPLE_DATA, require_tier
from core.ui import chip, header, html, muted, section

tier = require_tier("clinical")
header(t("batch.title"), t("batch.subtitle"), t("nav.sec_records"), "group")

a, b = st.columns(2, gap="medium")
with a.container(border=True):
    section("1 · " + t("batch.step1"), t("batch.step1_text"))
    template = pd.read_csv(SAMPLE_DATA / "batch_template.csv")
    st.download_button(t("batch.template"), template.to_csv(index=False).encode(), "plantilla_pacientes.csv",
                       "text/csv", icon=":material/download:")
    with st.expander(t("batch.columns")):
        st.markdown(t("batch.columns_help"))
with b.container(border=True):
    section("2 · " + t("batch.step2"))
    up = st.file_uploader(t("batch.upload"), label_visibility="collapsed")   # type checked below
    use_example = st.toggle(t("batch.example"), help=t("batch.example_help"))

if up is not None:
    if not files.is_table(up.name):
        st.error(t("files.wrong_csv", name=up.name))
        st.stop()
    try:
        raw = files.read_table(up.getvalue())
    except Exception as err:
        st.error(t("files.unreadable", error=err))
        st.stop()
elif use_example:
    raw = pd.read_csv(SAMPLE_DATA / "batch_example_test_patients.csv")
else:
    st.stop()

raw.columns = [str(c).strip().lower() for c in raw.columns]    # " Ap_Hi" and "ap_hi" are the same column
if raw.dropna(how="all").empty:
    st.error(t("batch.empty"))
    st.stop()
if "age_years" not in raw.columns and "age" in raw.columns:      # original Kaggle format (age in days)
    age = pd.to_numeric(raw["age"], errors="coerce")
    raw["age_years"] = age / 365.25 if age.median() > 200 else age
missing = [c for c in clinical.RAW_INPUTS if c not in raw.columns]
if missing:
    st.error(t("batch.missing", cols=", ".join(missing)))
    st.stop()

raw = raw.reset_index(drop=True)
problems = []
for i, row in raw[clinical.RAW_INPUTS].iterrows():
    errors, _ = clinical.validate(row.to_dict())
    if errors:
        problems.append({"#": i + 1, "": " ".join(validation_message(e) for e in errors)})
valid = raw.drop(index=[p["#"] - 1 for p in problems]).reset_index(drop=True)
valid[clinical.RAW_INPUTS] = valid[clinical.RAW_INPUTS].apply(pd.to_numeric)   # "120" -> 120 (already validated)
if valid.empty:
    st.error(t("batch.rejected", n=len(problems)))
    st.stop()

with st.spinner(t("batch.running")):
    res = clinical.predict_frame(tier, valid)
levels = [risk_level(p, r) for p, r in zip(res["probability"], res["referred"])]

# ------------------------------------------------------------------ results
st.write("")
section("3 · " + t("batch.step3"))
counts = pd.Series(levels).value_counts()
html(f'<p>{t("batch.summary", n=len(valid))} '
     + " ".join(chip(f'{t(f"risk.level.{lv}")}: {counts.get(lv, 0)}', lv) for lv in ("low", "mid", "high")) + "</p>")
if problems:
    with st.expander(t("batch.rejected", n=len(problems)), icon=":material/warning:"):
        st.dataframe(pd.DataFrame(problems), hide_index=True, width="stretch")

codes = valid["patient_code"] if "patient_code" in valid.columns else pd.Series(range(1, len(valid) + 1)).astype(str)
out = pd.DataFrame({
    t("batch.col_patient"): codes.values,
    t("batch.col_age"): valid["age_years"].round(0).astype(int).values,
    t("batch.col_bp"): [f"{h:.0f}/{l:.0f}" for h, l in zip(valid["ap_hi"], valid["ap_lo"])],
    t("batch.col_bmi"): res["bmi"].round(1).values,
    t("batch.col_prob"): (res["probability"].values * 100).round(0),
    t("batch.col_result"): [t(f"risk.level.{lv}") for lv in levels],
})
st.dataframe(out, hide_index=True, width="stretch", height=380,
             column_config={t("batch.col_prob"): st.column_config.ProgressColumn(min_value=0, max_value=100,
                                                                                 format="%d %%")})
st.download_button(t("batch.download"), out.to_csv(index=False).encode(), "resultados_pacientes.csv", "text/csv",
                   icon=":material/download:", type="primary")

if "cardio" in valid.columns:
    with st.expander(t("batch.compare"), icon=":material/fact_check:"):
        y = valid["cardio"].astype(int)
        tn, fp, fn, tp = confusion_matrix(y, res["prediction"], labels=[0, 1]).ravel()
        html("<p>" + t("batch.compare_text", acc=(tp + tn) / len(y), sens=tp / max(tp + fn, 1),
                       spec=tn / max(tn + fp, 1)) + "</p>")
        muted(t("how.limits").splitlines()[0].lstrip("- "))
