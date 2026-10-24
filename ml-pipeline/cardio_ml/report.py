"""Merge the run manifests of both tiers into one results summary.

    python -m cardio_ml.report      ->  outputs/RESULTS_SUMMARY.md
"""
from __future__ import annotations

import json
from datetime import datetime

import pandas as pd

from . import config

KEY_METRICS = ["accuracy", "precision", "recall_sensitivity", "specificity", "f1", "roc_auc", "mcc"]
BASE_MODELS = ["Logistic Regression", "K-Nearest Neighbors", "Random Forest", "Support Vector Machine"]
SYSTEMS = BASE_MODELS + ["Hard voting", "Soft voting", "Stacking ensemble"]


def _load(tier: str) -> dict | None:
    path = config.OUTPUTS / tier / "run_manifest.json"
    return json.loads(path.read_text()) if path.exists() else None


def _metric_table(m: dict, names: list[str], prefix: str = "") -> str:
    rows = {n: m["metrics"][prefix + n] for n in names if prefix + n in m["metrics"]}
    df = pd.DataFrame(rows).T
    cols = [c for c in KEY_METRICS + ["balanced_accuracy"] if c in df.columns]
    return df[cols].astype(float).round(4).to_markdown()


def _modelling_rows(m: dict) -> tuple[int, int]:
    balance = next(s for s in m["lineage"] if s["step"].startswith("Balance classes"))
    split = next(s for s in m["lineage"] if s["step"].startswith("Stratified train/test"))
    return balance["rows_after"], split["rows_after"]


def _cv_table(m: dict) -> str:
    rows = {}
    for name in BASE_MODELS + ["Stacking ensemble"]:
        v = m["metrics"].get(f"patient-wise CV {name}")
        if v:
            rows[name] = {"accuracy (balanced test)": f"{v['accuracy_mean']:.2%} ± {v['accuracy_std']:.2%}",
                          "sensitivity": f"{v['recall_sensitivity_mean']:.2%}",
                          "specificity": f"{v['specificity_mean']:.2%}", "roc_auc": f"{v['roc_auc_mean']:.4f}",
                          "accuracy (natural prevalence)": f"{v['natural_accuracy_mean']:.2%}",
                          "balanced accuracy (natural prevalence)": f"{v['natural_balanced_accuracy_mean']:.2%}"}
    return pd.DataFrame(rows).T.to_markdown()


def build_summary() -> str:
    clinical, ecg = _load("clinical"), _load("ecg")
    out = ["# Results summary — Hito 2", "",
           f"Generated {datetime.now():%Y-%m-%d %H:%M} from `outputs/*/run_manifest.json`. "
           "Every number below can be traced to its tier's `TRACEABILITY.md`.", ""]

    checklist = []
    for name, m in (("Tier 1 — clinical risk screening", clinical), ("Tier 2 — ECG beat diagnosis", ecg)):
        if m is None:
            out += [f"## {name}", "", "_Not run yet._", ""]
            continue
        rows, train = _modelling_rows(m)
        s = m["metrics"]["system summary"]
        out += [f"## {name}", "",
                f"* Modelling dataset: **{rows:,} balanced records** (train {train:,} / test {rows - train:,}).",
                f"* Final system (stacking of 4 models): **accuracy {s['stacking_accuracy']:.2%}** "
                f"(95 % CI {s['accuracy_95ci_low']:.2%} – {s['accuracy_95ci_high']:.2%}).",
                f"* Best single model: {s['best_base_model']}; McNemar p-value stacking vs. best model = "
                f"{s['mcnemar_vs_best_base_p']:.3f}.",
                f"* Referral rule: {s['referral_rate']:.1%} of cases referred to a clinician; accuracy on the "
                f"rest = {s['accuracy_on_non_referred']:.2%}.", "",
                "### Held-out test metrics", "", _metric_table(m, SYSTEMS), "",
                "### Meta-model weights", "", "```json", json.dumps(m["params"].get("meta-model weights"), indent=2),
                "```", ""]
        if "leakage audit (beat-level split)" in m["metrics"]:
            la = m["metrics"]["leakage audit (beat-level split)"]
            out += ["### Additional analysis: leakage audit and generalisation to new patients", "",
                    "* The official accuracy above follows the standard MIT-BIH protocol (stratified split by beat, "
                    "*intra-patient*): the test beats were never used for training, but "
                    f"{la['test_beats_from_training_patients']:.0%} of them come from patients that also gave other "
                    f"beats to training, and for {la['nearest_training_beat_same_patient']:.0%} of them the most "
                    "similar training beat belongs to the same patient.",
                    "* As an additional robustness analysis, patient-wise 5-fold cross-validation (no patient in train "
                    "and test at the same time, *inter-patient*) measures the accuracy on completely new patients. "
                    "As in the inter-patient literature (de Chazal et al., 2004), it is lower; it is reported as a "
                    "limitation of the system:", "", _cv_table(m), ""]
        if "monitoring session (unseen patient)" in m["metrics"]:
            ms = m["metrics"]["monitoring session (unseen patient)"]
            out += [f"* Monitoring session on unseen record `{ms['record']}`: {int(ms['beats']):,} beats, "
                    f"true abnormal burden {ms['true_abnormal_burden']:.1%}, predicted "
                    f"{ms['predicted_abnormal_burden']:.1%}, beat accuracy {ms['beat_accuracy']:.2%}.", ""]
        cv = m["metrics"].get("patient-wise CV Stacking ensemble")
        checklist.append((name, rows, s["stacking_accuracy"], cv))

    out += ["## Rubric checklist (Hito 2)", "", "| Criterion | Status |", "|---|---|"]
    for name, rows, acc, cv in checklist:
        out.append(f"| Dataset balanced and > 35,000 records — {name} | {'✅' if rows > 35000 else '❌'} {rows:,} |")
    out.append("| 4 ML models with theoretical support | ✅ LR, KNN, RF, SVM (`docs/mathematical_model.md` §4) |")
    out.append("| Results simulation with traceability | ✅ `TRACEABILITY.md`, `run_manifest.json`, `*_trace.json` |")
    out.append("| Mathematical model with equations and graphs | ✅ `docs/mathematical_model.md` + numbered figures |")
    for name, rows, acc, cv in checklist:
        line = f"| Accuracy > 95 % — {name} | {'✅' if acc > 0.95 else '❌'} {acc:.2%}"
        if cv:
            line += " (held-out test beats, standard MIT-BIH beat-level protocol)"
        out.append(line + " |")
    app = config.ROOT.parent / "web-app" / "app.py"
    out.append("| Application with 10 functionalities | " + ("✅ `web-app/` (Streamlit, 10 functionalities + PDF "
               "report + bilingual UI)" if app.exists() else "⏳ pending (`web-app/`)") + " |")
    return "\n".join(out) + "\n"


def main() -> None:
    path = config.OUTPUTS / "RESULTS_SUMMARY.md"
    path.write_text(build_summary())
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
