"""How it works: plain explanation first, then results, figures with their reading guide,
glossary, functionalities and sources.

Functionality 10 (model performance and traceability), read from the pipeline outputs.
"""
import pandas as pd
import streamlit as st

from core import charts
from core.guides import glossary
from core.i18n import lang, num, t
from core.model_store import OUTPUTS, load_manifest, tier_kpis
from core.ui import callout, chart, chart_header, chip, guide_html, header, html, muted, section, stats

header(t("how.title"), t("how.subtitle"), t("nav.sec_info"), "lightbulb")
tab_simple, tab_results, tab_figs, tab_gloss, tab_features, tab_sources = st.tabs(
    [t("how.tab_simple"), t("how.tab_results"), t("how.tab_trace"), t("how.tab_glossary"), t("how.tab_features"),
     t("how.tab_sources")])
K = {"clinical": tier_kpis("clinical"), "ecg": tier_kpis("ecg")}
TIER_LABEL = {"clinical": "how.tier1_short", "ecg": "how.tier2_short"}

# ------------------------------------------------------------- in a nutshell
with tab_simple:
    a, b = st.columns(2, gap="medium")
    for col, tier, n_key, ic in ((a, "clinical", "tier1", "stethoscope"), (b, "ecg", "tier2", "ecg_heart")):
        k = K[tier]
        with col.container(border=True):
            section(t(f"how.{n_key}_title"))
            html(f'<p><b>{t(f"how.{n_key}_q")}</b></p>')
            if k:
                html(f'<p class="cv-muted">{t(f"how.{n_key}_data", n=k["rows"])}</p>')
                acc = t("how.acc_ecg", acc=num(k["accuracy"], ".1%")) if k["accuracy"] > 0.95 else t("how.acc", acc=k["accuracy"])
                html(f'<p>{acc}</p>')
    st.write("")
    with st.container(border=True):
        section(t("how.flow_title"))
        html(f'<p>{t("how.flow_text")}</p>')
        st.graphviz_chart(f"""
        digraph {{
          rankdir=LR; bgcolor="transparent"; nodesep=0.25;
          node [shape=box, style="rounded,filled", fillcolor="#F8FAFC", color="#CBD5E1", fontname="Helvetica", fontsize=11];
          edge [color="#94A3B8", arrowsize=0.7];
          x [label="{t('how.dot_input')}", fillcolor="#EFF6FF", color="#BFDBFE"];
          lr [label="{t('model.LR')}"]; knn [label="{t('model.KNN')}"]; rf [label="{t('model.RF')}"]; svm [label="{t('model.SVM')}"];
          meta [label="{t('how.dot_meta')}", fillcolor="#FEF3C7", color="#FDE68A"];
          out [label="{t('how.dot_answer')}", fillcolor="#DCFCE7", color="#BBF7D0"];
          x -> lr; x -> knn; x -> rf; x -> svm; lr -> meta; knn -> meta; rf -> meta; svm -> meta; meta -> out;
        }}""")
    st.write("")
    with st.container(border=True):
        section(t("how.limits_title"))
        ecg_k = K["ecg"]
        st.markdown(t("how.limits", known=ecg_k.get("accuracy", 0), new=ecg_k.get("new_accuracy", 0),
                      sd=ecg_k.get("new_accuracy_sd", 0)))

# ------------------------------------------------------------------- results
with tab_results:
    tier = st.segmented_control(t("how.pick_tier"), ["clinical", "ecg"], default="clinical", required=True,
                                format_func=lambda x: t(TIER_LABEL[x]))
    m, k = load_manifest(tier), K[tier]
    if not m:
        st.warning(t("common.models_missing", tier=tier))
    else:
        ok = k["accuracy"] > 0.95
        tiles = [(t("how.records"), num(k["rows"]), t("how.records_h"), "database"),
                 (t("how.accuracy"), num(k["accuracy"], ".1%"), t("how.accuracy_h", lo=k["ci_low"], hi=k["ci_high"]),
                  "target")]
        if ok:
            tiles.append((t("how.target"), t("how.target_ok"), t("how.target_ok_h"), "verified"))
        tiles.append(("ROC-AUC", num(k["auc"], ".3f"), t("how.auc_h"), "insights"))
        stats(tiles)
        if not ok:
            callout(t("how.target_no_text"), "info")

        table = pd.read_csv(OUTPUTS / tier / "tables" / "test_metrics.csv").set_index("model")
        keys = {"Logistic Regression": "LR", "K-Nearest Neighbors": "KNN", "Random Forest": "RF",
                "Support Vector Machine": "SVM", "Hard voting": "hard", "Soft voting": "soft",
                "Stacking ensemble": "final"}
        rows = [(keys[n], t(f"model.{keys[n]}") if keys[n] not in ("hard", "soft") else t(f"how.vote_{keys[n]}"),
                 table.loc[n, "accuracy"]) for n in keys if n in table.index]
        with st.container(border=True):
            chart_header(t("how.chart_title"), t("how.chart_sub") + (" " + t("how.chart_sub_ecg") if tier == "ecg" else ""),
                         "accuracy")
            chart(charts.accuracy_bars(rows))
        w = m["params"].get("meta-model weights", {})
        with st.expander(t("how.equation"), icon=":material/function:"):
            st.latex(r"\hat P = \sigma\left(" + f"{w['intercept']:.2f}" +
                     "".join(f" {v:+.2f}\\,p_{{{key}}}" for key, v in w.items() if key != "intercept") + r"\right)")
            muted(t("how.equation_text"))
            muted(t("risk.referral_rule"))
        with st.expander(t("how.table"), icon=":material/table_chart:"):
            cols = ["accuracy", "precision", "recall_sensitivity", "specificity", "f1", "roc_auc", "mcc"]
            st.dataframe(table[cols].style.format("{:.3f}", na_rep="-"), width="stretch")

        if "new_accuracy" in k:     # ECG: extra robustness test on patients never seen (inter-patient protocol)
            with st.expander(t("how.protocols_title"), icon=":material/person_search:"):
                muted(t("how.protocols_sub"))
                a, b = st.columns(2, gap="medium")
                a.markdown(f'<div class="cv-stat"><div class="k">{t("how.known_title")}</div>'
                           f'<div class="v">{num(k["accuracy"], ".1%")}</div><div class="h">{t("how.known_text")}</div></div>',
                           unsafe_allow_html=True)
                b.markdown(f'<div class="cv-stat"><div class="k">{t("how.new_title")}</div>'
                           f'<div class="v">{num(k["new_accuracy"], ".1%")} ± {num(k["new_accuracy_sd"], ".1%")}</div>'
                           f'<div class="h">{t("how.new_text")}</div></div>', unsafe_allow_html=True)
                st.write("")
                if "leak_overlap" in k:
                    callout(t("how.leak_text", overlap=k["leak_overlap"], nn=k["leak_nn"]), "policy")
                folds = OUTPUTS / "ecg" / "tables" / "patient_cv_folds.csv"
                if folds.exists():
                    f = pd.read_csv(folds)
                    f = f[f["model"] == "Stacking ensemble"]
                    chart_header(t("how.folds_title"), guide_key="folds")
                    chart(charts.fold_dots(list(f["accuracy"]), float(f["accuracy"].mean())))
                cls = OUTPUTS / "ecg" / "tables" / "patient_cv_per_class.csv"
                if cls.exists():
                    c = pd.read_csv(cls).set_index("aami_class")
                    muted(t("how.class_recall", **{x: c.loc[x, "recall"] for x in ("N", "S", "V", "F") if x in c.index}))
                if "auto_accuracy" in k:
                    muted(t("how.auto_text", det=k["auto_detection"], bal=k["auto_balanced"]))

# ------------------------------------------------------ figures + traceability
CATEGORIES = {
    "data": ["raw_outliers", "ecg_etl_example", "cleaning_funnel", "beat_counts", "class_balance"],
    "eda": ["distributions", "iqr_boxplots", "split_violins", "categorical_prevalence", "scatter", "pairplot",
            "mean_beats", "rr_features", "correlations", "vif_candidates", "vif_model_features", "vif_rhythm_features",
            "pca_projection"],
    "models": ["metric_comparison", "confusion_matrices", "roc_pr_curves", "calibration", "meta_weights",
               "model_diversity", "probability_distribution", "coverage_curve"],
    "explain": ["permutation_importance", "logistic_coefficients", "plausibility_audit", "kmeans_k_scan",
                "risk_segments_pca", "risk_segments_profile", "segments_vs_ensemble"],
    "validation": ["leakage_check", "patient_cv", "inter_patient_metrics", "automatic_detection", "monitoring_session"],
    "simulation": ["simulation_archetypes", "simulation_test_patients", "simulation_test_beats"],
}


def slug_of(fig: dict) -> str:
    return fig.get("slug") or fig["file"][7:-4]


with tab_figs:
    muted(t("how.figs_intro"))
    tier = st.segmented_control(t("how.pick_tier"), ["clinical", "ecg"], default="clinical", required=True,
                                key="trace_tier", format_func=lambda x: t(TIER_LABEL[x]))
    m = load_manifest(tier)
    if m:
        figs = m["figures"]
        present = [c for c, slugs in CATEGORIES.items() if any(slug_of(f) in slugs for f in figs)]
        cat = st.pills(t("how.category"), present, default=present[0], format_func=lambda c: t(f"how.cat.{c}"),
                       key=f"cat_{tier}") or present[0]
        chosen = [f for f in figs if slug_of(f) in CATEGORIES[cat]]
        title = lambda f: (f.get("guide", {}).get("title", {}).get(lang()) or f["caption"])  # noqa: E731
        pick = st.selectbox(t("how.fig_pick"), range(len(chosen)), key=f"fig_{tier}_{cat}",
                            format_func=lambda i: f"{chosen[i]['number']:02d}. {title(chosen[i])}")
        fig = chosen[pick]
        left, right = st.columns([2, 1], gap="medium")
        with left.container(border=True):
            st.image(str(OUTPUTS / tier / "figures" / fig["file"]), width="stretch")
            muted(f"{fig['file']} · {fig['caption']}")
        with right.container(border=True):
            section(title(fig))
            g = fig.get("guide")
            if g:
                html(guide_html({"what": g["what"][lang()], "read": g["read"][lang()]}))
            else:
                muted(fig["caption"])
        st.write("")
        with st.expander(t("how.lineage"), icon=":material/filter_alt:"):
            lineage = pd.DataFrame(m["lineage"])[["step", "rows_before", "rows_after", "removed", "detail"]]
            st.dataframe(lineage, hide_index=True, width="stretch")
        with st.expander(t("how.trace_full"), icon=":material/description:"):
            st.markdown((OUTPUTS / tier / "TRACEABILITY.md").read_text())
    summary = OUTPUTS / "RESULTS_SUMMARY.md"
    with st.expander(t("how.summary_full"), icon=":material/fact_check:"):
        st.markdown(summary.read_text() if summary.exists() else t("how.no_summary"))

# ------------------------------------------------------------------ glossary
with tab_gloss:
    muted(t("how.glossary_intro"))
    st.write("")
    terms = glossary()
    a, b = st.columns(2, gap="medium")
    for i, (term, text) in enumerate(terms):
        with (a if i % 2 == 0 else b).container(border=True):
            html(f'<div class="cv-section">{term}</div><div class="cv-muted" style="font-size:.95rem">{text}</div>')

# ----------------------------------------------------------- functionalities
with tab_features:
    muted(t("how.features_intro"))
    st.write("")
    pages = {1: "risk", 2: "risk", 3: "risk", 4: "risk", 5: "risk", 6: "ecg_analysis", 7: "ecg_analysis",
             8: "batch", 9: "history", 10: "how_it_works", 11: "risk", 12: "home"}
    a, b = st.columns(2)
    for i, page in pages.items():
        (a if i <= 6 else b).page_link(f"views/{page}.py", label=f"{i}. {t(f'feature.{i}')}")

# ------------------------------------------------------------------- sources
with tab_sources:
    st.markdown("""
- S. Ulianova, *Cardiovascular Disease dataset*, Kaggle, 2019.
- G. B. Moody and R. G. Mark, "The impact of the MIT-BIH Arrhythmia Database," *IEEE Eng. Med. Biol. Mag.*, vol. 20, no. 3, pp. 45–50, 2001. PhysioNet, ODC-By 1.0.
- S. D. Greenwald, *MIT-BIH Supraventricular Arrhythmia Database*, PhysioNet, ODC-By 1.0.
- A. L. Goldberger *et al.*, "PhysioBank, PhysioToolkit, and PhysioNet," *Circulation*, vol. 101, no. 23, 2000.
- ANSI/AAMI EC57, *Testing and reporting performance results of cardiac rhythm and ST segment measurement algorithms*.
- P. de Chazal, M. O'Dwyer and R. B. Reilly, "Automatic classification of heartbeats using ECG morphology and heartbeat interval features," *IEEE Trans. Biomed. Eng.*, vol. 51, no. 7, pp. 1196–1206, 2004.
- E. J. S. Luz *et al.*, "ECG-based heartbeat classification for arrhythmia detection: A survey," *Comput. Methods Programs Biomed.*, vol. 127, pp. 144–164, 2016.
- D. H. Wolpert, "Stacked generalization," *Neural Networks*, vol. 5, no. 2, pp. 241–259, 1992.
""")
    section(t("how.authors"))
    html(chip("Jans Alan David Perea Cerquera", "gray") + " " + chip("Nicolás Daniel Pérez Varón", "gray")
         + '<p class="cv-muted" style="margin-top:.5rem">Universidad Surcolombiana · 2026</p>')
    muted(t("common.disclaimer"))
