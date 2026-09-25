"""How the four models 'talk' to each other: stacked generalisation.

Level 0: the four base models are trained separately. Each one outputs the
probability p_k(x) that the case is positive.

Level 1: a logistic-regression meta-model reads those four probabilities and
returns ONE answer:

    z(x) = b0 + b1*p_LR(x) + b2*p_KNN(x) + b3*p_RF(x) + b4*p_SVM(x)
    P(y=1 | x) = 1 / (1 + exp(-z(x)))

    What it does: a weighted vote in which the weights b_k are LEARNED from
    data instead of being equal. b_k >= 0 is imposed, so a model that sees
    more risk can never lower the final answer.

To avoid the meta-model learning from probabilities the base models produced
on rows they had already memorised, it is trained on out-of-fold (OOF)
predictions: with K-fold CV each row is scored by base models that never saw
it. The learned weights b_k tell how much the system trusts each model.

Alternatives compared in the report (same four probabilities):
    soft vote  P = (p_LR + p_KNN + p_RF + p_SVM) / 4       (equal weights)
    hard vote  y = 1 if at least 3 of the 4 models say "positive" (2-2 -> soft vote)
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from . import config
from .models import BaseModel, ConstrainedLogisticRegression
from .tracking import RunTracker


class StackingEnsemble:
    def __init__(self, base_models: list[BaseModel], cv: int = config.CV_FOLDS,
                 seed: int = config.SEED, referral_band: tuple[float, float] = config.REFERRAL_BAND) -> None:
        self.base_models = base_models
        self.cv = cv
        self.seed = seed
        self.referral_band = referral_band
        # beta_k >= 0: a base model that sees more risk can never lower the final answer
        self.meta_ = ConstrainedLogisticRegression(C=1.0, signs=[1] * len(base_models))
        self.oof_: pd.DataFrame | None = None
        self.feature_names: list[str] = []

    @property
    def names(self) -> list[str]:
        return [m.short for m in self.base_models]

    # -------------------------------------------------------------- training
    def fit(self, X: pd.DataFrame, y: pd.Series, tracker: RunTracker, groups=None,
            log_prefix: str = "") -> "StackingEnsemble":
        self.feature_names = list(X.columns)
        splitter = (StratifiedGroupKFold(self.cv, shuffle=True, random_state=self.seed) if groups is not None
                    else StratifiedKFold(self.cv, shuffle=True, random_state=self.seed))
        folds = list(splitter.split(X, y, groups))
        oof = np.zeros((len(X), len(self.base_models)))
        fold_id = np.zeros(len(X), dtype=int)
        for k, model in enumerate(self.base_models):
            start = time.perf_counter()
            for f, (tr, va) in enumerate(folds):
                fold_model = model.fresh().fit(X.iloc[tr], y.iloc[tr])
                oof[va, k] = fold_model.predict_proba(X.iloc[va])
                fold_id[va] = f
            model.fit(X, y)                       # final level-0 model on all training rows
            tracker.log_metrics(f"{log_prefix}training time {model.short}",
                                {"oof_seconds": time.perf_counter() - start, "final_fit_seconds": model.fit_seconds,
                                 "rows_used_final_fit": min(len(X), model.max_train or len(X))})
            print(f"  {model.name}: OOF + final fit done in {time.perf_counter() - start:.1f}s")
        self.oof_ = pd.DataFrame(oof, columns=self.names, index=X.index)
        self.oof_["fold"] = fold_id
        self.oof_["y_true"] = y.values
        self.meta_.fit(oof, y)
        tracker.log_param(f"{log_prefix}stacking", {"cv_folds": self.cv,
                                                     "meta_model": "LogisticRegression(C=1.0), weights beta_k >= 0",
                                       "meta_features": self.names, "referral_band": self.referral_band,
                                       "grouped_cv": groups is not None})
        tracker.log_param(f"{log_prefix}meta-model weights", self.weights())
        return self

    # ------------------------------------------------------------- inference
    def base_probabilities(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({m.short: m.predict_proba(X[self.feature_names]) for m in self.base_models},
                            index=X.index)

    def combine(self, Z: pd.DataFrame, prior: float | None = None) -> np.ndarray:
        p = self.meta_.predict_proba(Z[self.names].values)[:, 1]
        return p if prior is None else self.adjust_prior(p, prior)

    def predict_proba(self, X: pd.DataFrame, prior: float | None = None) -> np.ndarray:
        return self.combine(self.base_probabilities(X), prior)

    def predict(self, X: pd.DataFrame, threshold: float = 0.5, prior: float | None = None) -> np.ndarray:
        return (self.predict_proba(X, prior) >= threshold).astype(int)

    @staticmethod
    def adjust_prior(p: np.ndarray, prior: float, train_prior: float = 0.5) -> np.ndarray:
        """Bayes prior-shift correction for a model trained on balanced data.

        logit(p') = logit(p) + log(pi / (1 - pi)) - log(pi_train / (1 - pi_train)),
        where pi is the real prevalence of the positive class (estimated from the
        training data before undersampling, never from the test set).

        What it does: a model trained on a 50/50 dataset believes positives are
        as common as negatives. In a real Holter only ~11 % of the beats are
        abnormal, so this formula (Bayes' rule) lowers every probability by
        the same amount on the log-odds scale. It changes no ranking (same
        ROC-AUC), only where the 0.5 threshold falls.
        """
        p = np.clip(p, 1e-9, 1 - 1e-9)
        shift = np.log(prior / (1 - prior)) - np.log(train_prior / (1 - train_prior))
        return 1 / (1 + np.exp(-(np.log(p / (1 - p)) + shift)))

    # ---------------------------------------------------- alternative fusions
    @staticmethod
    def soft_vote(Z: pd.DataFrame) -> np.ndarray:
        return Z.mean(axis=1).values

    @staticmethod
    def hard_vote(Z: pd.DataFrame) -> np.ndarray:
        votes = (Z >= 0.5).sum(axis=1).values
        ties = votes * 2 == Z.shape[1]
        return np.where(ties, (Z.mean(axis=1).values >= 0.5), votes * 2 > Z.shape[1]).astype(int)

    def referral_mask(self, p: np.ndarray, Z: pd.DataFrame) -> np.ndarray:
        """Cases the system hands to a clinician: uncertain answer or a 2-2 split vote."""
        lo, hi = self.referral_band
        split_vote = (Z >= 0.5).sum(axis=1).values * 2 == Z.shape[1]
        return ((p > lo) & (p < hi)) | split_vote

    # --------------------------------------------------------- explanations
    def weights(self) -> dict:
        w = {"intercept": float(self.meta_.intercept_[0])}
        w.update({n: float(c) for n, c in zip(self.names, self.meta_.coef_[0])})
        return w

    def explain_case(self, x: pd.DataFrame) -> dict:
        """Full trace of one prediction: every model's opinion and how it was combined."""
        Z = self.base_probabilities(x)
        w = self.weights()
        z = w["intercept"] + sum(w[n] * Z[n].iloc[0] for n in self.names)
        p = float(self.combine(Z)[0])
        return {"base_probabilities": {n: round(float(Z[n].iloc[0]), 4) for n in self.names},
                "base_votes": {n: int(Z[n].iloc[0] >= 0.5) for n in self.names},
                "meta_contributions": {n: round(w[n] * float(Z[n].iloc[0]), 4) for n in self.names},
                "meta_intercept": round(w["intercept"], 4), "meta_logit": round(float(z), 4),
                "final_probability": round(p, 4), "final_prediction": int(p >= 0.5),
                "referred_to_clinician": bool(self.referral_mask(np.array([p]), Z)[0])}
