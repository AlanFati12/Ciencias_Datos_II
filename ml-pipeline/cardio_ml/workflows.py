"""End-to-end workflows for the two tiers of the system.

Tier 1 - ClinicalRiskWorkflow: risk screening from a basic medical exam
         (Kaggle cardio_train, 70k patients).
Tier 2 - ECGDiagnosisWorkflow: heartbeat diagnosis from a single-lead ECG
         (PhysioNet MIT-BIH Arrhythmia + Supraventricular Arrhythmia DBs).

Both share the template in BaseWorkflow: ingest -> clean -> explore -> split
-> tune -> train (4 models + stacking) -> evaluate -> explain -> simulate ->
export. Each step can be called on its own from the notebooks.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from . import config, eda, evaluation as ev, explainability as xai, segmentation as seg
from .ecg_etl import AMPLITUDE_FEATURE, PATIENT_FEATURES, RR_FEATURES, BeatExtractor

ECG_TABULAR = RR_FEATURES + [AMPLITUDE_FEATURE] + PATIENT_FEATURES
from .ensemble import StackingEnsemble
from .ingestion import ClinicalCSVIngester, PhysioNetECGIngester
from .models import KNNModel, LogisticRegressionModel, RandomForestModel, SVMModel
from .plausibility import monotonicity_audit, plot_audit
from .plotting import AAMI_COLORS, CLASS_COLORS, CRITICAL, ENSEMBLE_COLOR, RISK_COLORS, VOTE_COLORS, plt, sns
from .preprocessing import (AgeInYears, BloodPressureConsistency, ClassBalancer, ClinicalFeatureEngineer,
                            ConflictingLabelRemover, DropColumns, DuplicateRemover, FiniteFilter, Pipeline,
                            RangeFilter)
from .simulation import plot_simulation, save_case_traces, simulate_cases
from .tracking import RunTracker


class BaseWorkflow(ABC):
    tier: str = ""
    target: str = ""
    labels: dict = {}
    positive: str = "Positive"

    def __init__(self, fast: bool = False, show: bool = False) -> None:
        self.fast = fast
        self.show = show
        self.tracker = RunTracker(self.tier + ("_fast" if fast else ""))
        self.tracker.log_param("mode", "fast (subsample, for testing only)" if fast else "full")
        self.models = self.build_models()
        self.ensemble: StackingEnsemble | None = None
        self.results: dict = {}
        self.groups_train: pd.Series | None = None    # patient of each training row (tiers with several rows per patient)

    # ---------------------------------------------------------- tier specific
    @abstractmethod
    def ingest(self) -> None: ...

    @abstractmethod
    def clean(self) -> None: ...

    @abstractmethod
    def explore(self) -> None: ...

    @abstractmethod
    def build_models(self) -> list: ...

    @abstractmethod
    def importance_groups(self) -> dict[str, list[str]]: ...

    @abstractmethod
    def simulate(self) -> None: ...

    @property
    @abstractmethod
    def features(self) -> list[str]: ...

    # ---------------------------------------------------------------- helpers
    def figure(self, fig, slug: str, caption: str) -> None:
        """Save a figure (traceability) and, in a notebook, show it with its reading guide underneath."""
        self.tracker.save_figure(fig, slug, caption)
        if self.show:
            from IPython.display import Markdown, display
            display(fig)
            entry = self.tracker.figures[-1]
            text = f"**Figure {entry['number']} — {caption}**"
            if "guide" in entry:
                g = entry["guide"]
                text += f"  \n*What it shows:* {g['what']['en']}  \n*How to read it:* {g['read']['en']}"
            display(Markdown("> " + text.replace("\n", "\n> ")))
        plt.close(fig)

    @property
    def colors(self) -> dict:
        c = {m.name: m.color for m in self.models}
        c.update({m.short: m.color for m in self.models})
        c.update(VOTE_COLORS)
        c["Stacking ensemble"] = ENSEMBLE_COLOR
        return c

    # ------------------------------------------------------------ shared flow
    def split(self) -> None:
        X, y = self.df[self.features], self.df[self.target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.SEED)
        self.tracker.log_step("Stratified train/test split", len(X), len(self.X_train),
                              f"train {len(self.X_train):,} / test {len(self.X_test):,} "
                              f"({config.TEST_SIZE:.0%} held out, seed {config.SEED})")
        assignment = pd.DataFrame({"row": X.index, "set": np.where(X.index.isin(self.X_test.index), "test", "train")})
        self.tracker.save_table(assignment, "split_assignment", "Row-level train/test assignment")
        self.tracker.log_param("model features", self.features if len(self.features) < 30
                               else f"{len(self.features)} columns ({self.features[:6]} ...)")

    def tune(self) -> None:
        for m in self.models:
            print(f"Tuning {m.name} ...")
            res = m.tune(self.X_train, self.y_train, self.tracker, groups=self.groups_train)
            self.tracker.save_table(res, f"tuning_{m.short}", f"Randomized search results for {m.name}")

    def train(self) -> None:
        print("Training level-0 models with out-of-fold predictions, then the meta-model ...")
        self.ensemble = StackingEnsemble(self.models).fit(self.X_train, self.y_train, self.tracker,
                                                          groups=self.groups_train)
        oof = self.ensemble.oof_
        rows = [{"model": m.name, **ev.classification_metrics(oof["y_true"], oof[m.short])} for m in self.models]
        oof_table = pd.DataFrame(rows).set_index("model")
        self.tracker.save_table(oof_table.reset_index(), "oof_metrics", "Cross-validated (out-of-fold) metrics")
        self.tracker.save_table(oof.reset_index(names="row"), "oof_predictions",
                                "Out-of-fold probabilities used to train the meta-model")

    def evaluate(self) -> None:
        names = self.ensemble.names
        Z = self.ensemble.base_probabilities(self.X_test)
        p = self.ensemble.combine(Z)
        y = self.y_test.values
        scores = {m.name: Z[m.short].values for m in self.models}
        scores["Hard voting"] = self.ensemble.hard_vote(Z).astype(float)   # 0/1 decisions, no probability
        scores["Soft voting"] = self.ensemble.soft_vote(Z)
        scores["Stacking ensemble"] = p
        table = pd.DataFrame({k: ev.classification_metrics(y, v) for k, v in scores.items()}).T
        table.loc["Hard voting", ["roc_auc", "pr_auc", "brier"]] = np.nan    # needs a probability
        for name, row in table.iterrows():
            self.tracker.log_metrics(name, row.to_dict())
        self.tracker.save_table(table.reset_index(names="model"), "test_metrics", "Held-out test metrics")
        self.results["test_metrics"] = table

        lo, hi = ev.bootstrap_ci(y, p)
        best_base = table.loc[[m.name for m in self.models], "accuracy"].idxmax()
        mc = ev.mcnemar_test(y, (p >= 0.5).astype(int), (scores[best_base] >= 0.5).astype(int))
        referred = self.ensemble.referral_mask(p, Z)
        auto_acc = float(((p >= 0.5).astype(int) == y)[~referred].mean())
        summary = {"stacking_accuracy": table.loc["Stacking ensemble", "accuracy"],
                   "accuracy_95ci_low": lo, "accuracy_95ci_high": hi, "best_base_model": best_base,
                   "mcnemar_vs_best_base_p": mc["p_value"], "referral_rate": float(referred.mean()),
                   "accuracy_on_non_referred": auto_acc, "meets_95_target": bool(table.loc["Stacking ensemble",
                                                                                           "accuracy"] > 0.95)}
        self.tracker.log_metrics("system summary", summary)
        self.results["summary"] = summary
        print(json.dumps(summary, indent=2, default=float))

        short = {m.name: m.short for m in self.models}
        disp = table.rename(index=short)
        self.figure(ev.plot_metric_comparison(disp, colors=self.colors), "metric_comparison",
                    "Test-set metrics of the four base models, voting schemes and the stacking ensemble")
        cms = {m.short: ev.confusion_matrix(y, (Z[m.short] >= 0.5).astype(int)) for m in self.models}
        cms["Stacking"] = ev.confusion_matrix(y, (p >= 0.5).astype(int))
        self.figure(ev.plot_confusion_matrices(cms, labels=tuple(self.labels.values())), "confusion_matrices",
                    "Confusion matrices on the held-out test set")
        curves = {k: v for k, v in scores.items() if k != "Hard voting"}
        self.figure(ev.plot_roc_pr(y, curves, self.colors), "roc_pr_curves", "ROC and precision-recall curves")
        self.figure(ev.plot_calibration(y, curves, self.colors), "calibration",
                    "Reliability of the predicted probabilities")
        self.figure(ev.plot_meta_weights(self.ensemble.weights(), self.colors), "meta_weights",
                    "Coefficients learned by the meta-model for each base model")
        self.figure(ev.plot_model_diversity(self.ensemble.oof_, names), "model_diversity",
                    "Diversity between base models (OOF correlation and disagreement)")
        self.figure(ev.plot_probability_distributions(y, p), "probability_distribution",
                    "Distribution of the final probability per true class and referral band")
        cov = {k: ev.coverage_curve(y, v) for k, v in curves.items()}
        self.figure(ev.plot_coverage(cov, self.colors), "coverage_curve",
                    "Accuracy when the system answers only its most confident cases")

    def explain(self, n: int = 4000) -> None:
        Xs = self.X_test.sample(min(n, len(self.X_test)), random_state=config.SEED)
        ys = self.y_test.loc[Xs.index]
        imp = xai.permutation_importance(self.ensemble.predict, Xs, ys, self.importance_groups())
        self.tracker.save_table(imp, "permutation_importance", "Permutation importance of the stacking ensemble")
        self.figure(xai.plot_importance(imp, "What the combined system relies on (permutation importance)"),
                    "permutation_importance", "Permutation importance of each feature group for the ensemble")
        self.results["importance"] = imp

    def export(self) -> None:
        self.tracker.save_model(self.ensemble, "stacking_ensemble",
                                "StackingEnsemble: 4 tuned base models + logistic meta-model")
        schema = self.schema()
        (self.tracker.root / "models" / "model_schema.json").write_text(json.dumps(schema, indent=2, default=str))

    def schema(self) -> dict:
        return {"tier": self.tier, "target": self.target, "labels": self.labels, "features": self.features,
                "threshold": 0.5, "referral_band": config.REFERRAL_BAND,
                "meta_weights": self.ensemble.weights(), "base_models": {m.short: m.name for m in self.models}}

    def run_all(self) -> RunTracker:
        for step in (self.ingest, self.clean, self.explore, self.split, self.tune, self.train,
                     self.evaluate, self.explain, self.simulate, self.export):
            print(f"\n=== {self.tier}: {step.__name__} ===")
            step()
        self.tracker.finalize()
        return self.tracker


# =============================================================================
class ClinicalRiskWorkflow(BaseWorkflow):
    tier = "clinical"
    target = config.CLINICAL_TARGET
    labels = {0: "No CVD", 1: "CVD"}
    positive = "CVD"

    @property
    def features(self) -> list[str]:
        return config.CLINICAL_FEATURES

    def build_models(self):
        from scipy.stats import loguniform
        # Smooth settings on purpose: with 11 noisy inputs, flexible KNN/SVM fits add wiggles
        # (implausible ups and downs in the what-if curves) but no accuracy. KNN estimates the
        # local CVD prevalence, whose noise is sqrt(p(1-p)/k): k = 2,500 keeps it near 1 point
        # (k = 199 gave ~3 points and wrong-way dips for 99 % of patients when weight rose).
        knn_space = {"clf__n_neighbors": [2500], "clf__weights": ["uniform"], "clf__p": [1, 2]}
        svm_space = {"clf__C": loguniform(0.1, 10), "clf__gamma": loguniform(1e-3, 5e-2)}
        signs = [config.CLINICAL_MONOTONIC.get(f, 0) for f in self.features] if config.CLINICAL_MONOTONIC else None
        free = [f for f in self.features if f not in config.CLINICAL_UNCONSTRAINED_EXCLUDE] \
            if config.CLINICAL_MONOTONIC else None
        return [LogisticRegressionModel(signs=signs), KNNModel(space=knn_space, columns=free),
                RandomForestModel(monotonic_cst=signs), SVMModel(space=svm_space, columns=free)]

    def importance_groups(self):
        return {f: [f] for f in self.features}

    def ingest(self) -> None:
        self.raw = ClinicalCSVIngester(self.tracker).load()

    def clean(self) -> None:
        r = config.CLINICAL_RANGES
        steps = [DropColumns(["id"]), DuplicateRemover(), ConflictingLabelRemover(self.target),
                 RangeFilter({k: r[k] for k in ("ap_hi", "ap_lo")}, name="Filter impossible blood pressure"),
                 BloodPressureConsistency(),
                 RangeFilter({k: r[k] for k in ("height", "weight")}, name="Filter impossible height/weight"),
                 AgeInYears(), ClinicalFeatureEngineer(),
                 RangeFilter({"bmi": r["bmi"]}, name="Filter implausible BMI")]
        self.unbalanced = Pipeline(steps, self.tracker).run(self.raw)
        self.balancer = ClassBalancer(self.target)
        self.df = Pipeline([self.balancer], self.tracker).run(self.unbalanced)
        if self.fast:
            self.df = self.df.sample(8000, random_state=config.SEED).reset_index(drop=True)
            self.tracker.log_step("FAST MODE subsample", len(self.unbalanced), len(self.df), "testing only")
        config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(config.DATA_PROCESSED / "clinical_clean.csv", index=False)

    def explore(self) -> None:
        df, t = self.df, self.target
        num = ["age_years", "bmi", "ap_hi", "ap_lo", "pulse_pressure", "weight"]
        self.figure(eda.plot_raw_outliers(self.raw), "raw_outliers",
                    "Raw measurements: impossible blood pressures, heights and weights")
        self.figure(eda.plot_cleaning_funnel(self.tracker.lineage_frame()), "cleaning_funnel",
                    "Rows remaining after each cleaning step")
        self.figure(eda.plot_class_balance(self.balancer.counts_before, self.balancer.counts_after, self.labels),
                    "class_balance", "Class balance before and after undersampling")
        self.figure(eda.plot_distributions(df, num, t, self.labels), "distributions",
                    "Histograms and KDE of the numeric variables by diagnosis")
        self.figure(eda.plot_iqr_boxplots(df, num, t, self.labels), "iqr_boxplots",
                    "Boxplots with the IQR outlier rule by diagnosis")
        self.figure(eda.plot_split_violins(df, t), "split_violins",
                    "Split violin plots: categorical factors vs. numeric variables")
        self.figure(eda.plot_categorical_rates(df, ["gender", "cholesterol", "gluc", "bp_stage",
                                                    "smoke", "alco", "active"], t),
                    "categorical_prevalence", "CVD prevalence by categorical factor")
        self.figure(eda.plot_scatter_pairs(df, t), "scatter", "Scatter plots of blood pressure and age vs. BMI")
        self.figure(eda.plot_pairplot(df, ["age_years", "bmi", "ap_hi", "ap_lo", "cholesterol"], t), "pairplot",
                    "Pairplot of the main numeric variables")
        self.figure(eda.plot_correlations(df, config.CLINICAL_CANDIDATE_FEATURES + [t]), "correlations",
                    "Pearson vs. Spearman correlation matrices (triangular)")
        vif = eda.vif_table(df, config.CLINICAL_CANDIDATE_FEATURES)
        self.tracker.save_table(vif, "vif_candidates", "VIF of all candidate features")
        self.figure(eda.plot_vif(vif), "vif_candidates", "Multicollinearity of the candidate features (VIF)")
        vif_final = eda.vif_table(df, self.features)
        self.tracker.save_table(vif_final, "vif_model_features", "VIF after dropping collinear features")
        self.figure(eda.plot_vif(vif_final), "vif_model_features",
                    f"VIF after dropping {config.CLINICAL_COLLINEAR_DROP} (exact linear combinations)")
        self.tracker.note(f"Dropped {config.CLINICAL_COLLINEAR_DROP} from the model inputs: map and "
                          "pulse_pressure are linear combinations of ap_hi/ap_lo (VIF -> inf) and weight is "
                          "explained by bmi and height (VIF ~ 100).")
        self.figure(eda.plot_pca_projection(df[self.features], df[t], self.labels), "pca_projection",
                    "2-D PCA projection of the clinical features by diagnosis")
        self.tracker.save_table(df.describe().T.reset_index(names="variable"), "descriptive_statistics",
                                "Descriptive statistics of the clean dataset")

    def segment(self) -> None:
        """Hito 1 objective: K-means risk segmentation (Low / Medium / High)."""
        train = self.df.loc[self.X_train.index]
        test = self.df.loc[self.X_test.index]
        self.segmenter = seg.RiskSegmenter()
        scan = self.segmenter.scan_k(train)
        self.tracker.save_table(scan, "kmeans_k_scan", "Elbow and silhouette for k = 2..8")
        self.figure(seg.plot_k_scan(scan, 3), "kmeans_k_scan", "Choice of k for K-means (k = 3 chosen)")
        self.segmenter.fit(train, self.target)
        levels_test = self.segmenter.assign(test)
        profile = self.segmenter.profile(test, levels_test, self.target)
        self.tracker.save_table(profile.reset_index(names="risk_level"), "risk_segments_profile",
                                "Clinical profile and CVD prevalence per risk segment (test set)")
        self.figure(seg.plot_segments(self.segmenter, test, levels_test), "risk_segments_pca",
                    "Risk segments in PCA space (test set)")
        self.figure(seg.plot_segment_profile(profile, self.segmenter.features), "risk_segments_profile",
                    "Mean profile and observed CVD prevalence per segment (test set)")
        self.segment_profile = profile
        p = self.ensemble.predict_proba(self.X_test)
        data = pd.DataFrame({"Risk segment": levels_test.values, "p": p})
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data, x="Risk segment", y="p", order=seg.RISK_LEVELS, hue="Risk segment",
                    palette=RISK_COLORS, ax=ax)
        ax.set(ylabel="Ensemble probability of CVD", title="Supervised vs. unsupervised view agree")
        self.figure(fig, "segments_vs_ensemble", "Ensemble probability distribution inside each K-means segment")
        self.tracker.log_param("risk segmentation", {"features": self.segmenter.features, "k": 3,
                                                     "pca_components": int(self.segmenter.pca_.n_components_),
                                                     "cluster_to_level": self.segmenter.cluster_to_level_})

    def explain(self, n: int = 4000) -> None:
        super().explain(n)
        lr = self.models[0].model_
        coef = pd.DataFrame({"feature": self.features, "coefficient": lr.named_steps["clf"].coef_[0]})
        coef["odds_ratio_per_sd"] = np.exp(coef["coefficient"])
        self.tracker.save_table(coef.sort_values("coefficient"), "logistic_coefficients",
                                "Standardised logistic-regression coefficients and odds ratios")
        c = coef.sort_values("coefficient")
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(c["feature"], c["coefficient"], color=np.where(c["coefficient"] > 0, CLASS_COLORS[1], CLASS_COLORS[0]))
        ax.axvline(0, c="black", lw=0.8)
        ax.set(xlabel="Coefficient β (per 1 SD of the feature)", title="Logistic regression: direction of each risk factor")
        self.figure(fig, "logistic_coefficients", "Standardised logistic-regression coefficients")
        self.plausibility()

    def plausibility(self, n: int = 400) -> None:
        """Clinical plausibility audit of every model (see plausibility.py)."""
        raw = self.df.loc[self.X_test.index].sample(min(n, len(self.X_test)), random_state=config.SEED)
        audit = monotonicity_audit(self.ensemble, raw, lambda r: ClinicalFeatureEngineer().transform(r)[self.features],
                                   config.PLAUSIBILITY_SWEEPS, config.PLAUSIBILITY_EXPECTED)
        self.tracker.save_table(audit, "plausibility_audit",
                                "Share of patients whose risk moves against the clinical direction (> 1 pp)")
        self.figure(plot_audit(audit, "Patients whose risk moves the WRONG way when a factor worsens"),
                    "plausibility_audit", "Clinical plausibility audit: wrong-way changes per model and factor")
        final = audit[audit["model"] == "Stacking"].set_index("feature")
        self.tracker.log_metrics("plausibility (final answer)", {
            "max_share_violating": float(final["share_violating"].max()),
            "mean_share_violating": float(final["share_violating"].mean()),
            "constraints": bool(config.CLINICAL_MONOTONIC)})
        self.tracker.log_param("clinical monotonic constraints", config.CLINICAL_MONOTONIC or "none")
        self.tracker.log_param("inputs not given to KNN and SVM", config.CLINICAL_UNCONSTRAINED_EXCLUDE
                               if config.CLINICAL_MONOTONIC else "none")
        self.results["plausibility"] = audit

    ARCHETYPES = pd.DataFrame([
        # age_years gender height weight ap_hi ap_lo chol gluc smoke alco active
        ["P1 healthy young woman", 35, 1, 165, 58, 110, 70, 1, 1, 0, 0, 1],
        ["P2 overweight man, high-normal BP", 50, 2, 175, 88, 130, 85, 2, 1, 0, 0, 1],
        ["P3 obese woman, stage-2 HTN", 60, 1, 160, 85, 150, 95, 3, 2, 0, 0, 0],
        ["P4 smoker man, severe HTN", 58, 2, 172, 80, 170, 100, 2, 1, 1, 1, 0],
        ["P5 borderline middle-aged", 48, 1, 168, 70, 125, 82, 1, 1, 0, 0, 1],
        ["P6 older but healthy", 62, 2, 170, 70, 120, 80, 1, 1, 0, 0, 1],
        ["P7 diabetic, elevated chol", 55, 1, 158, 75, 135, 85, 3, 3, 0, 0, 0],
        ["P8 young man, hypertensive", 40, 2, 180, 95, 145, 92, 1, 1, 1, 0, 1],
    ], columns=["case", "age_years", "gender", "height", "weight", "ap_hi", "ap_lo", "cholesterol", "gluc",
                "smoke", "alco", "active"])

    def simulate(self) -> None:
        self.segment()
        names = self.ensemble.names
        arch = ClinicalFeatureEngineer().transform(self.ARCHETYPES.drop(columns="case"))
        cases = list(self.ARCHETYPES["case"])
        table = simulate_cases(self.ensemble, arch[self.features], cases)
        table["risk_segment"] = self.segmenter.assign(arch).values
        self.tracker.save_table(table, "simulation_archetypes", "Simulated patient archetypes through the system")
        self.figure(plot_simulation(table, names, "Simulation: 8 hypothetical patients through the 4 models",
                                    self.positive), "simulation_archetypes",
                    "Per-model and final probability for simulated patient archetypes")
        save_case_traces(self.ensemble, arch[self.features], cases, self.tracker.root / "tables" /
                         "simulation_archetypes_trace.json",
                         {c: {"risk_segment": s} for c, s in zip(cases, table["risk_segment"])})
        sample = self.X_test.groupby(self.y_test, group_keys=False).sample(6, random_state=config.SEED)
        ids = [f"test #{i}" for i in sample.index]
        table2 = simulate_cases(self.ensemble, sample, ids, self.y_test.loc[sample.index])
        self.tracker.save_table(table2, "simulation_test_patients", "12 unseen test patients through the system")
        self.figure(plot_simulation(table2, names, "Simulation: 12 unseen test patients (ok = correct)",
                                    self.positive), "simulation_test_patients",
                    "Per-model and final probability for 12 unseen test patients")
        self.results["simulation"] = table

    def export(self) -> None:
        super().export()
        self.tracker.save_model(self.segmenter, "risk_segmenter",
                                "RiskSegmenter: PCA + K-means (k = 3) -> Low / Medium / High risk")

    def schema(self) -> dict:
        s = super().schema()
        train = self.df.loc[self.X_train.index]
        stats_cols = ["age_years", "gender", "height", "weight", "ap_hi", "ap_lo", "cholesterol", "gluc",
                      "smoke", "alco", "active", "bmi", "bp_stage"]
        s["training_stats"] = {c: {"median": float(train[c].median()), "mode": float(train[c].mode().iloc[0]),
                                   "p01": float(train[c].quantile(0.01)), "p99": float(train[c].quantile(0.99))}
                               for c in stats_cols}
        s["segment_features"] = self.segmenter.features
        s["segment_profiles"] = self.segment_profile.round(3).to_dict(orient="index")
        s["raw_inputs"] = {
            "age_years": "years", "gender": "1 = female, 2 = male", "height": "cm", "weight": "kg",
            "ap_hi": "systolic mmHg", "ap_lo": "diastolic mmHg", "cholesterol": "1 normal, 2 above, 3 well above",
            "gluc": "1 normal, 2 above, 3 well above", "smoke": "0/1", "alco": "0/1", "active": "0/1"}
        s["valid_ranges"] = config.CLINICAL_RANGES
        s["derived"] = {"bmi": "weight / (height/100)^2", "bp_stage": "AHA 2017 category 0-4"}
        return s


# =============================================================================
class ECGDiagnosisWorkflow(BaseWorkflow):
    tier = "ecg"
    target = config.ECG_TARGET
    labels = {0: "Normal", 1: "Abnormal"}
    positive = "abnormal beat"

    def __init__(self, fast: bool = False, show: bool = False) -> None:
        self.extractor = BeatExtractor()
        super().__init__(fast, show)

    @property
    def features(self) -> list[str]:
        return ECG_TABULAR + self.extractor.waveform_columns

    def build_models(self):
        from scipy.stats import randint
        # wide k range: with patient-grouped tuning a larger k (smoother vote) can win over memorising neighbours
        knn_space = {"clf__n_neighbors": randint(3, 151), "clf__weights": ["uniform", "distance"], "clf__p": [1, 2]}
        return [LogisticRegressionModel(), KNNModel(space=knn_space), RandomForestModel(), SVMModel()]

    def _region(self, lo_s: float, hi_s: float) -> list[str]:
        cols = self.extractor.waveform_columns
        t = (np.arange(len(cols)) - self.extractor.n_before) / self.extractor.fs_target
        return [c for c, ti in zip(cols, t) if lo_s <= ti < hi_s]

    def importance_groups(self):
        groups = {f: [f] for f in ECG_TABULAR}
        groups["Waveform: P-wave region (−250 to −80 ms)"] = self._region(-0.25, -0.08)
        groups["Waveform: QRS complex (−80 to +80 ms)"] = self._region(-0.08, 0.08)
        groups["Waveform: ST-T region (+80 to +450 ms)"] = self._region(0.08, 0.46)
        return groups

    def ingest(self) -> None:
        self.raw = PhysioNetECGIngester(self.tracker, extractor=self.extractor).load()
        self.raw["patient"] = self.raw["record"].replace(config.ECG_SAME_PATIENT)
        self.tracker.log_param("patients", int(self.raw["patient"].nunique()))

    def split(self) -> None:
        """Beat-level stratified split (as in most of the literature) + patient groups.

        Each patient contributes thousands of beats, so this split puts beats
        of the SAME patient in train and test ("intra-patient" protocol). The
        patient of every training beat is kept so that hyperparameter tuning
        and the stacking out-of-fold predictions are grouped by patient, and
        leakage_check() / patient_cv() measure the result on NEW patients.
        """
        super().split()
        self.groups_train = self.df.loc[self.X_train.index, "patient"]

    def clean(self) -> None:
        steps = [FiniteFilter(),
                 RangeFilter({"pre_rr": (0.2, 3.0), "post_rr": (0.2, 3.0)},
                             name="Filter implausible RR intervals (20-300 bpm)")]
        self.unbalanced = Pipeline(steps, self.tracker).run(self.raw)
        self.balancer = ClassBalancer(self.target)
        self.df = Pipeline([self.balancer], self.tracker).run(self.unbalanced)
        if self.fast:
            self.df = self.df.sample(8000, random_state=config.SEED).reset_index(drop=True)
            self.tracker.log_step("FAST MODE subsample", len(self.unbalanced), len(self.df), "testing only")
        config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        meta_cols = ["database", "record", "sample", "symbol", "aami_class", self.target] + ECG_TABULAR
        self.df[meta_cols].to_csv(config.DATA_PROCESSED / "ecg_beats_metadata.csv", index=False)
        counts = self.unbalanced.groupby(["database", "aami_class"]).size().unstack(fill_value=0)
        self.tracker.save_table(counts.reset_index(), "beats_per_class", "Beats per database and AAMI class")

    def _example_record(self) -> tuple:
        folder = config.ECG_DIR / config.PHYSIONET_DATABASES["mitdb"][1]
        raw, clean, fs, ann = self.extractor.preprocess_signal(folder / "208")
        t = np.asarray(ann.sample) / fs
        sym = np.asarray(ann.symbol)
        for start in np.arange(0, t.max() - 10, 2.0):
            win = (t >= start) & (t < start + 8)
            if np.isin(sym[win], ["V", "F"]).sum() >= 3 and (sym[win] == "N").sum() >= 3:
                return raw, fs, clean, ann, float(start)
        return raw, fs, clean, ann, 0.0

    def explore(self) -> None:
        df, t = self.df, self.target
        raw, fs, clean, ann, start = self._example_record()
        self.figure(eda.plot_ecg_strip(raw, fs, clean, self.extractor.fs_target, ann, start),
                    "ecg_etl_example", "ETL of an unstructured ECG signal: raw vs. filtered/resampled (MIT-BIH 208)")
        self.figure(eda.plot_cleaning_funnel(self.tracker.lineage_frame()), "cleaning_funnel",
                    "Rows (beats) remaining after each ETL step")
        self.figure(eda.plot_beat_counts(self.unbalanced), "beat_counts",
                    "Heartbeats per database and AAMI class before balancing")
        self.figure(eda.plot_class_balance(self.balancer.counts_before, self.balancer.counts_after,
                                           {0: "Normal beat", 1: "Abnormal beat"}),
                    "class_balance", "Class balance before and after undersampling")
        self.figure(eda.plot_mean_beats(df, self.extractor.waveform_columns, self.extractor.fs_target,
                                        self.extractor.n_before), "mean_beats", "Mean beat shape per AAMI class")
        rr_cols = ECG_TABULAR
        self.figure(eda.plot_rr_features(df, rr_cols, t), "rr_features", "RR and amplitude features by class")
        self.figure(eda.plot_distributions(df, ["rr_ratio", "post_pre_ratio", "template_corr", "template_dist"], t,
                                           eda.ECG_LABELS, ncols=2), "distributions",
                    "Histograms and KDE of rhythm features by class")
        self.figure(eda.plot_correlations(df, rr_cols + [t]), "correlations",
                    "Pearson vs. Spearman correlation of rhythm features")
        vif = eda.vif_table(df, rr_cols)
        self.tracker.save_table(vif, "vif_rhythm_features", "VIF of the rhythm features")
        self.figure(eda.plot_vif(vif), "vif_rhythm_features", "Multicollinearity of the rhythm features (VIF)")
        self.figure(eda.plot_pca_projection(df[self.features], df[t], eda.ECG_LABELS), "pca_projection",
                    "2-D PCA projection of all beat features by class")
        self.tracker.save_table(df[rr_cols + [t]].describe().T.reset_index(names="variable"),
                                "descriptive_statistics", "Descriptive statistics of rhythm features")

    def leakage_check(self, n: int = 3000) -> None:
        """Does the beat-level test measure memory of known patients? (data-leakage audit)

        Two numbers answer it:
          patient overlap = share of test beats whose patient also has beats in training;
          same-patient NN = share of test beats whose nearest training beat (Euclidean
                            distance on standardised features) belongs to the same patient.
        If both are high, a test beat is usually "recognised" through a sibling beat
        of the same person, so the beat-level accuracy is optimistic for new patients.
        """
        meta = self.df
        train_patients = set(self.groups_train)
        test_patients = meta.loc[self.X_test.index, "patient"]
        sample = self.X_test.sample(min(n, len(self.X_test)), random_state=config.SEED)
        scaler = StandardScaler().fit(self.X_train)
        nn = NearestNeighbors(n_neighbors=1).fit(scaler.transform(self.X_train))
        _, idx = nn.kneighbors(scaler.transform(sample))
        same = meta.loc[self.X_train.index[idx[:, 0]], "patient"].values == meta.loc[sample.index, "patient"].values
        audit = {"test_beats_from_training_patients": float(test_patients.isin(train_patients).mean()),
                 "nearest_training_beat_same_patient": float(same.mean()),
                 "patients_in_train": len(train_patients), "patients_in_test": int(test_patients.nunique())}
        self.tracker.log_metrics("leakage audit (beat-level split)", audit)
        self.tracker.note(f"Leakage audit: {audit['test_beats_from_training_patients']:.0%} of the beat-level test "
                          f"beats come from patients also seen in training, and for "
                          f"{audit['nearest_training_beat_same_patient']:.0%} of them the nearest training beat is "
                          "from the same patient. The beat-level accuracy therefore measures performance on KNOWN "
                          "patients; patient_cv() measures it on NEW patients.")
        self.results["leakage"] = audit
        print(json.dumps(audit, indent=2))

    def patient_cv(self, folds: int = config.CV_FOLDS) -> None:
        """Patient-wise (inter-patient) K-fold cross-validation: the strict protocol.

        Patients (not beats) are split into K groups with StratifiedGroupKFold, so a
        patient is never in training and test at the same time. For each fold k:
          train  = beats of the other K-1 groups, balanced 50/50 by undersampling;
          test_b = beats of group k, balanced 50/50 (same rule as the main test set);
          test_n = all beats of group k, natural prevalence (~11 % abnormal),
                   with the Bayes prior correction of StackingEnsemble.adjust_prior.
        A fresh copy of the 4 tuned models + meta-model is trained in every fold.
        Reported: mean +/- SD over the K folds, per-fold values and recall per AAMI class.
        """
        data, t = self.unbalanced, self.target
        splitter = StratifiedGroupKFold(folds, shuffle=True, random_state=config.SEED)
        rows, per_class = [], []
        for f, (tr, te) in enumerate(splitter.split(data, data[t], data["patient"])):
            train = ClassBalancer(t).transform(data.iloc[tr])
            if self.fast:
                train = train.sample(min(8000, len(train)), random_state=config.SEED)
            test_n = data.iloc[te]
            test_b = ClassBalancer(t, seed=config.SEED + f).transform(test_n)
            print(f"Patient-wise CV fold {f + 1}/{folds}: {train['patient'].nunique()} training patients, "
                  f"{test_n['patient'].nunique()} new test patients ...")
            models = [m.fresh() for m in self.models]
            ens = StackingEnsemble(models).fit(train[self.features], train[t], self.tracker,
                                               groups=train["patient"], log_prefix=f"patient-CV fold {f + 1} ")
            prior = float(data.iloc[tr][t].mean())
            Z = ens.base_probabilities(test_n[self.features])
            Zb = Z.loc[test_b.index]
            scores = {m.name: (Zb[m.short].values, ens.adjust_prior(Z[m.short].values, prior)) for m in models}
            scores["Stacking ensemble"] = (ens.combine(Zb), ens.combine(Z, prior))
            natural = {}
            for name, (pb, pn) in scores.items():
                mb = ev.classification_metrics(test_b[t].values, pb)
                mn = ev.classification_metrics(test_n[t].values, pn)
                natural[name] = mn
                rows.append({"fold": f + 1, "model": name, "test_patients": int(test_n["patient"].nunique()),
                             "accuracy": mb["accuracy"], "recall_sensitivity": mb["recall_sensitivity"],
                             "specificity": mb["specificity"], "roc_auc": mb["roc_auc"],
                             "natural_accuracy": mn["accuracy"], "natural_balanced_accuracy": mn["balanced_accuracy"]})
            p_raw = ens.combine(Z)
            for cls in ("N", "S", "V", "F"):
                m = test_n["aami_class"].values == cls
                if m.any():
                    per_class.append({"fold": f + 1, "aami_class": cls, "beats": int(m.sum()),
                                      "correct": int(((p_raw[m] >= 0.5) == (cls != "N")).sum())})
            if f == 0:          # kept for the deployment checks on these unseen patients
                self.inter_patient = {"ensemble": ens, "records": sorted(test_n["record"].unique()), "prior": prior}
                self.results["inter_patient"] = pd.DataFrame(natural).T
        table = pd.DataFrame(rows)
        self.tracker.save_table(table, "patient_cv_folds", "Patient-wise CV: metrics per fold and model (new patients)")
        cls = pd.DataFrame(per_class).groupby("aami_class")[["beats", "correct"]].sum()
        cls["recall"] = cls["correct"] / cls["beats"]
        cls = cls.reindex(["N", "S", "V", "F"])
        self.tracker.save_table(cls.reset_index(), "patient_cv_per_class", "Patient-wise CV: recall per AAMI class")
        summary = table.groupby("model")[["accuracy", "recall_sensitivity", "specificity", "roc_auc",
                                          "natural_accuracy", "natural_balanced_accuracy"]].agg(["mean", "std"])
        for name in summary.index:
            self.tracker.log_metrics(f"patient-wise CV {name}", {f"{k}_{s}": v for (k, s), v in summary.loc[name].items()})
        self.results["patient_cv"] = table
        self.results["patient_cv_summary"] = summary
        self.results["patient_cv_class"] = cls
        st = summary.loc["Stacking ensemble"]
        print(f"Patient-wise CV stacking accuracy: {st[('accuracy', 'mean')]:.2%} ± {st[('accuracy', 'std')]:.2%}")
        self._plot_protocols(summary, table, cls)

    def _plot_protocols(self, summary: pd.DataFrame, table: pd.DataFrame, cls: pd.DataFrame) -> None:
        """Figures: known vs. new patients per model, and the patient-wise CV folds."""
        beat = self.results["test_metrics"]["accuracy"]
        names = [m.name for m in self.models] + ["Stacking ensemble"]
        short = [m.short for m in self.models] + ["Stacking"]
        fig, ax = plt.subplots(figsize=(9, 4.4))
        x = np.arange(len(names))
        known = [beat[n] for n in names]
        new, err = summary.loc[names, ("accuracy", "mean")].values, summary.loc[names, ("accuracy", "std")].values
        b1 = ax.bar(x - 0.2, known, 0.38, color="#c3c2b7", label="Known patients (beat-level split)")
        b2 = ax.bar(x + 0.2, new, 0.38, yerr=err, capsize=4, color=CLASS_COLORS[0],
                    label="New patients (patient-wise 5-fold CV, mean ± SD)")
        ax.bar_label(b1, fmt="{:.1%}", fontsize=8, padding=2)
        ax.bar_label(b2, labels=[f"{v:.1%}" for v in new], fontsize=8, padding=10)
        ax.axhline(0.95, ls="--", c=CRITICAL, lw=1, label="Rubric target 95 %")
        ax.set_xticks(x, short)
        ax.set_ylim(max(0.0, min(min(known), min(new - err)) - 0.08), 1.03)
        ax.set(ylabel="Accuracy on a balanced test set", title="Leakage check: accuracy on known vs. new patients")
        ax.legend(loc="lower left", fontsize=8)
        self.figure(fig, "leakage_check", "Accuracy with a beat-level split (known patients) vs. patient-wise CV (new patients)")

        stk = table[table["model"] == "Stacking ensemble"]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [3, 2]})
        a1.scatter(stk["accuracy"], stk["fold"], s=70, c=ENSEMBLE_COLOR, zorder=3)
        for _, r in stk.iterrows():
            a1.annotate(f"{r['accuracy']:.1%} ({r['test_patients']} patients)", (r["accuracy"], r["fold"]),
                        xytext=(8, 0), textcoords="offset points", va="center", fontsize=9)
        a1.axvline(stk["accuracy"].mean(), c=ENSEMBLE_COLOR, lw=1)
        a1.axvline(0.95, ls="--", c=CRITICAL, lw=1)
        a1.set(yticks=stk["fold"], yticklabels=[f"Fold {k}" for k in stk["fold"]], xlabel="Accuracy (balanced test)",
               title=f"Stacking on new patients: {stk['accuracy'].mean():.1%} ± {stk['accuracy'].std():.1%}")
        a1.set_xlim(min(stk["accuracy"].min(), 0.95) - 0.03, 1.0)
        a1.invert_yaxis()
        bars = a2.bar(cls.index, cls["recall"], color=[AAMI_COLORS[c] for c in cls.index])
        a2.bar_label(bars, labels=[f"{r:.0%}\nn={int(n):,}" for r, n in zip(cls["recall"], cls["beats"])], fontsize=8)
        a2.set(ylim=(0, 1.18), ylabel="Share recognised correctly",
               title="Recall per beat type (new patients)", xlabel="AAMI class (N normal · S supra. · V ventr. · F fusion)")
        fig.tight_layout()
        self.figure(fig, "patient_cv", "Patient-wise 5-fold CV: accuracy per fold and recall per AAMI class")

    def automatic_detection_check(self, tolerance_s: float = 0.15) -> None:
        """Deployment check: locate the beats with an automatic detector (XQRS).

        The web app receives ECGs without cardiologist annotations, so the beats
        must be found automatically. On the new patients of patient-wise CV fold 1,
        each detected beat is matched to the nearest annotation (within 150 ms)
        to know its true class. The beats are then classified with the fold-1
        ensemble (trained without those patients), exactly as the app does.
        """
        ens, prior = self.inter_patient["ensemble"], self.inter_patient["prior"]
        ex = self.extractor
        per_record, frames = [], []
        print(f"Automatic beat detection on {len(self.inter_patient['records'])} unseen records ...")
        for record in self.inter_patient["records"]:
            db, name = record.split("_", 1)
            raw, _, fs, ann = ex.preprocess_signal(config.ECG_DIR / config.PHYSIONET_DATABASES[db][1] / name)
            beats, _, peaks = ex.extract_signal(raw, fs)
            symbols = np.asarray(ann.symbol)
            is_beat = np.isin(symbols, list(ex.rr_symbols))
            filtered, _ = ex.prepare(raw, fs)
            ref = ex.align_peaks(filtered, np.asarray(ann.sample)[is_beat], fs)
            ref_class = np.array([ex.symbol_to_class.get(x, "Q") for x in symbols[is_beat]])
            tol = tolerance_s * fs
            # detection quality against the reference beats
            j = np.clip(np.searchsorted(ref, peaks), 1, len(ref) - 1)
            nearest = np.where(np.abs(ref[j] - peaks) < np.abs(ref[j - 1] - peaks), j, j - 1)
            k = np.clip(np.searchsorted(peaks, ref), 1, len(peaks) - 1)
            found = np.minimum(np.abs(peaks[k] - ref), np.abs(peaks[k - 1] - ref)) <= tol
            per_record.append({"record": record, "annotated_beats": len(ref), "detected_beats": len(peaks),
                               "sensitivity": float(found.mean()),
                               "ppv": float((np.abs(ref[nearest] - peaks) <= tol).mean())})
            # true class of each detected beat that was classified
            b_near = np.clip(np.searchsorted(ref, beats["sample"].values), 1, len(ref) - 1)
            b_idx = np.where(np.abs(ref[b_near] - beats["sample"].values) < np.abs(ref[b_near - 1] - beats["sample"].values),
                             b_near, b_near - 1)
            matched = np.abs(ref[b_idx] - beats["sample"].values) <= tol
            beats["aami_class"] = np.where(matched, ref_class[b_idx], "unmatched")
            beats["record"] = record
            beats["t"] = beats["sample"] / fs
            frames.append(beats)
        detection = pd.DataFrame(per_record)
        beats = pd.concat(frames, ignore_index=True)
        valid = beats["pre_rr"].between(0.2, 3.0) & beats["post_rr"].between(0.2, 3.0)
        beats["p"] = np.nan
        beats.loc[valid, "p"] = ens.predict_proba(beats.loc[valid, self.features], prior)
        beats["pred"] = (beats["p"] >= 0.5).astype(int)
        scored = beats[valid & beats["aami_class"].isin(["N", "S", "V", "F"])]
        y = (scored["aami_class"] != "N").astype(int).values
        auto = ev.classification_metrics(y, scored["p"].values)
        reference = self.results["inter_patient"].loc["Stacking ensemble"].to_dict()   # fold-1 patients, annotated peaks
        total = {"detection_sensitivity": float((detection["sensitivity"] * detection["annotated_beats"]).sum()
                                                / detection["annotated_beats"].sum()),
                 "detection_ppv": float((detection["ppv"] * detection["detected_beats"]).sum()
                                        / detection["detected_beats"].sum()),
                 "classified_beats": int(len(scored)), "accuracy": auto["accuracy"],
                 "balanced_accuracy": auto["balanced_accuracy"], "recall_sensitivity": auto["recall_sensitivity"],
                 "specificity": auto["specificity"], "roc_auc": auto["roc_auc"]}
        self.tracker.log_metrics("automatic detection (unseen patients)", total)
        self.tracker.save_table(detection, "automatic_detection_per_record",
                                "XQRS beat detection quality per unseen record (150 ms tolerance)")
        table = pd.DataFrame({"Annotated R peaks": reference, "Automatic detection (XQRS)": auto}).T
        self.tracker.save_table(table.reset_index(names="beat_source"), "automatic_detection_metrics",
                                "Classification of unseen patients: annotated vs. automatically detected beats")
        fig = ev.plot_metric_comparison(table, metrics=("accuracy", "balanced_accuracy", "recall_sensitivity",
                                                        "specificity", "roc_auc"),
                                        colors={"Annotated R peaks": "#c3c2b7", "Automatic detection (XQRS)": CLASS_COLORS[0]})
        fig.axes[0].set_title(f"Deployment check on unseen patients: detector found {total['detection_sensitivity']:.1%} "
                              f"of the beats (PPV {total['detection_ppv']:.1%})")
        self.figure(fig, "automatic_detection", "Annotated vs. automatically detected beats on unseen patients")
        self.detection = beats
        print(json.dumps(total, indent=2))

    def monitoring_session(self, seconds: float = 12.0) -> None:
        """Simulated remote-monitoring session on an unseen patient, as the web app would run it."""
        beats_all = self.detection
        burden = beats_all.groupby("record")["aami_class"].apply(lambda c: c.isin(["S", "V", "F"]).mean())
        mid = burden[(burden > 0.05) & (burden < 0.6)]
        record = mid.index[0] if len(mid) else burden.idxmax()
        db, name = record.split("_", 1)
        _, clean, fs, _ = self.extractor.preprocess_signal(config.ECG_DIR / config.PHYSIONET_DATABASES[db][1] / name)
        beats = beats_all[(beats_all["record"] == record) & beats_all["p"].notna()].copy()
        known = beats[beats["aami_class"].isin(["N", "S", "V", "F"])]
        truth = (known["aami_class"] != "N").astype(int)
        summary = {"record": record, "beats": len(beats), "true_abnormal_burden": float(truth.mean()),
                   "predicted_abnormal_burden": float(beats["pred"].mean()),
                   "beat_accuracy": float((known["pred"] == truth).mean())}
        self.tracker.log_metrics("monitoring session (unseen patient)", summary)
        abn = known[truth == 1]
        start = max(0.0, float(abn["t"].iloc[len(abn) // 2]) - seconds / 2) if len(abn) else 0.0
        win = beats[(beats["t"] >= start) & (beats["t"] < start + seconds)]
        tt = np.arange(len(clean)) / self.extractor.fs_target
        m = (tt >= start) & (tt < start + seconds)
        fig, ax = plt.subplots(figsize=(14, 3.8))
        ax.plot(tt[m], clean[m], c="grey", lw=0.9)
        for _, b in win.iterrows():
            i = min(int(round(b["t"] * self.extractor.fs_target)), len(clean) - 1)
            true = None if b["aami_class"] not in ("N", "S", "V", "F") else int(b["aami_class"] != "N")
            ok = true is None or b["pred"] == true
            ax.scatter(b["t"], clean[i], s=70, marker="o" if ok else "X", zorder=3,
                       c=CRITICAL if b["pred"] else CLASS_COLORS[0], edgecolor="black")
            ax.annotate(f"{b['aami_class']}\n{b['p']:.2f}", (b["t"], clean[i]), xytext=(0, 10),
                        textcoords="offset points", ha="center", fontsize=7)
        ax.margins(y=0.22)
        ax.set(xlabel="Time (s)", ylabel="mV",
               title=f"Remote monitoring of unseen patient {record} (beats found automatically)\nblue = predicted "
                     f"normal · red = predicted abnormal · X = error · label: true AAMI class and P(abnormal)")
        self.figure(fig, "monitoring_session", f"Simulated monitoring session on unseen record {record}")
        self.tracker.save_table(beats[["record", "sample", "t", "aami_class", "p", "pred"]],
                                "monitoring_session_beats", f"Beat-by-beat predictions for record {record}")

    def simulate(self) -> None:
        names = self.ensemble.names
        idx = []
        meta = self.df.loc[self.X_test.index]
        for cls, n in (("N", 4), ("S", 3), ("V", 3), ("F", 2)):
            pool = meta[meta["aami_class"] == cls]
            idx += list(pool.sample(min(n, len(pool)), random_state=config.SEED).index)
        ids = [f"{meta.loc[i, 'record']} beat {meta.loc[i, 'sample']} ({meta.loc[i, 'aami_class']})" for i in idx]
        table = simulate_cases(self.ensemble, self.X_test.loc[idx], ids, self.y_test.loc[idx])
        self.tracker.save_table(table, "simulation_test_beats", "12 test beats through the system")
        self.figure(plot_simulation(table, names, "Simulation: 12 test heartbeats through the 4 models",
                                    self.positive), "simulation_test_beats",
                    "Per-model and final probability for 12 held-out beats of every AAMI class")
        save_case_traces(self.ensemble, self.X_test.loc[idx, ECG_TABULAR].join(
            self.X_test.loc[idx, self.extractor.waveform_columns]), ids,
            self.tracker.root / "tables" / "simulation_test_beats_trace.json")
        self.leakage_check()
        self.patient_cv()
        self.automatic_detection_check()
        self.monitoring_session()

    def schema(self) -> dict:
        s = super().schema()
        s["signal_processing"] = self.extractor.describe()
        s["deployment_prior"] = round(float(self.unbalanced[self.target].mean()), 4)
        s["rr_valid_range_s"] = (0.2, 3.0)
        cv = self.results["patient_cv_summary"].loc["Stacking ensemble"]
        s["evaluation_protocols"] = {
            "beat_level_split": {"accuracy": float(self.results["test_metrics"].loc["Stacking ensemble", "accuracy"]),
                                 "meaning": "random split by beat: test beats come from patients seen in training"},
            "patient_wise_cv": {"accuracy_mean": float(cv[("accuracy", "mean")]), "accuracy_std": float(cv[("accuracy", "std")]),
                                "natural_accuracy_mean": float(cv[("natural_accuracy", "mean")]),
                                "natural_balanced_accuracy_mean": float(cv[("natural_balanced_accuracy", "mean")]),
                                "roc_auc_mean": float(cv[("roc_auc", "mean")]),
                                "meaning": "5-fold CV by patient: test patients never seen in training"},
            "leakage_audit": self.results["leakage"]}
        s["usage"] = ("BeatExtractor().extract_signal(raw_mV, fs) detects the beats (XQRS) and builds the "
                      "features; then ensemble.predict_proba(beats[features], prior=deployment_prior).")
        return s
