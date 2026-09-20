"""The four base learners from the syllabus (weeks 9-12).

Every learner shares one interface (BaseModel), so the ensemble can train and
query them polymorphically without knowing which algorithm is behind each one.
Each class docstring states the equation the model is built on, what it does
and why it is in the system (full derivations: docs/mathematical_model.md §4).

Shared step - standardisation (z-score), used before LR, KNN and SVM:
    z_j = (x_j - mean_j) / sd_j
    What it does: puts every input on the same scale (mean 0, SD 1).
    Why: KNN and SVM measure distances and LR penalises ||w||^2, so without it
    a variable measured in large units (mmHg) would dominate one in small units
    (cholesterol 1-3). The mean and SD are learned on the training rows only.
"""
from __future__ import annotations

import copy
import time
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import loguniform, randint
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import config
from .palette import MODEL_COLORS
from .tracking import RunTracker


class ConstrainedLogisticRegression(ClassifierMixin, BaseEstimator):
    """L2-regularised logistic regression whose coefficients have a fixed sign.

    Equation (sigmoid / logistic function):
        P(y = 1 | x) = sigma(z) = 1 / (1 + e^{-z}),   z = w.x + b
        What it does: turns any real score z into a probability between 0 and 1.

    Equation (penalised log-loss, the quantity being minimised):
        min_{w,b}  sum_i [log(1 + e^{z_i}) - y_i z_i] + ||w||^2 / (2C)
        s.t.       w_j >= 0 if sign_j = +1,   w_j <= 0 if sign_j = -1,   free if sign_j = 0
        What it does: the first term punishes confident wrong answers (it is the
        negative log-likelihood of the data); the second (L2 / ridge) keeps the
        weights small to avoid overfitting. C is the inverse regularisation
        strength: small C = simpler model.

    Same objective as sklearn's LogisticRegression, solved with L-BFGS-B (a
    quasi-Newton optimiser that accepts bounds) so the sign constraints can be
    imposed. Used to encode clinical knowledge ("a higher blood pressure never
    lowers the risk") and for the non-negative meta-model of the ensemble.
    The gradient returned to the optimiser is
        dL/dw = X^T (p - y) + w / C,      dL/db = sum_i (p_i - y_i).
    """

    def __init__(self, C: float = 1.0, signs=None, max_iter: int = 1000):
        self.C = C
        self.signs = signs
        self.max_iter = max_iter

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        target = (y == self.classes_[-1]).astype(float)
        n_features = X.shape[1]
        signs = np.zeros(n_features) if self.signs is None else np.asarray(self.signs, dtype=float)
        bounds = [(0, None) if s > 0 else (None, 0) if s < 0 else (None, None) for s in signs] + [(None, None)]

        def objective(theta):
            w, b = theta[:-1], theta[-1]
            z = X @ w + b
            p = expit(z)
            loss = np.sum(np.logaddexp(0.0, z) - target * z) + w @ w / (2 * self.C)
            grad = np.append(X.T @ (p - target) + w / self.C, np.sum(p - target))
            return loss, grad

        result = minimize(objective, np.zeros(n_features + 1), jac=True, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": self.max_iter})
        self.coef_ = result.x[:-1][None, :]
        self.intercept_ = result.x[-1:]
        self.n_iter_ = result.nit
        return self

    def decision_function(self, X):
        return np.asarray(X, dtype=float) @ self.coef_[0] + self.intercept_[0]

    def predict_proba(self, X):
        p = expit(self.decision_function(X))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return self.classes_[(self.decision_function(X) >= 0).astype(int)]


class BaseModel(ABC):
    name: str = "model"
    short: str = "M"
    color: str = "#888888"
    max_train: int | None = None     # optional cap on training rows (logged)

    def __init__(self, space: dict | None = None, seed: int = config.SEED, columns: list[str] | None = None) -> None:
        self.seed = seed
        self.columns = columns          # optional subset of input columns this model may use
        self.space = space if space is not None else self.default_space()
        self.params: dict = {}
        self.model_ = None
        self.fit_seconds: float | None = None

    # ------------------------------------------------------------ definition
    @abstractmethod
    def search_estimator(self):
        """Estimator explored by the hyperparameter search."""

    @abstractmethod
    def default_space(self) -> dict:
        """Search distributions, keyed with the 'clf__' pipeline prefix."""

    def final_estimator(self):
        """Estimator actually trained; by default the tuned search estimator."""
        return clone(self.search_estimator()).set_params(**self.params)

    def _x(self, X):
        return X[self.columns] if self.columns is not None else X

    # -------------------------------------------------------------- training
    def tune(self, X: pd.DataFrame, y: pd.Series, tracker: RunTracker,
             sample: int = config.TUNING_SAMPLE, n_iter: int = config.TUNING_ITER,
             groups: pd.Series | None = None) -> pd.DataFrame:
        """Randomized search of the hyperparameters with 3-fold cross-validation.

        With `groups` (ECG: one group per patient) the folds never split a
        patient, so the chosen hyperparameters are the ones that generalise to
        NEW patients instead of the ones that memorise known patients best.
        """
        X = self._x(X)
        if len(X) > sample:
            keep, _ = train_test_split(np.arange(len(X)), train_size=sample, stratify=y, random_state=self.seed)
            X, y = X.iloc[keep], y.iloc[keep]
            groups = None if groups is None else groups.iloc[keep]
        cv = (StratifiedGroupKFold(3, shuffle=True, random_state=self.seed) if groups is not None
              else StratifiedKFold(3, shuffle=True, random_state=self.seed))
        search = RandomizedSearchCV(
            self.search_estimator(), self.space, n_iter=n_iter, scoring="accuracy", cv=cv,
            random_state=self.seed, n_jobs=-1, refit=False)
        search.fit(X, y, groups=groups)
        self.params = search.best_params_
        results = pd.DataFrame(search.cv_results_)
        cols = [c for c in results.columns if c.startswith("param_")] + ["mean_test_score", "std_test_score",
                                                                         "rank_test_score", "mean_fit_time"]
        results = results[cols].sort_values("rank_test_score")
        tracker.log_param(f"{self.name} best hyperparameters", {k.replace("clf__", ""): v
                                                               for k, v in self.params.items()})
        tracker.log_metrics(f"tuning {self.name}", {"cv_accuracy": search.best_score_, "tuning_rows": len(X)})
        return results

    def _cap(self, X, y):
        if self.max_train and len(X) > self.max_train:
            X, _, y, _ = train_test_split(X, y, train_size=self.max_train, stratify=y, random_state=self.seed)
        return X, y

    def fit(self, X, y) -> "BaseModel":
        X, y = self._cap(self._x(X), y)
        start = time.perf_counter()
        self.model_ = clone(self.final_estimator()).fit(X, y)
        self.fit_seconds = time.perf_counter() - start
        return self

    def fresh(self) -> "BaseModel":
        """Untrained copy with the same tuned hyperparameters (used for CV folds)."""
        twin = copy.copy(self)
        twin.params = dict(self.params)
        twin.model_, twin.fit_seconds = None, None
        return twin

    def predict_proba(self, X) -> np.ndarray:
        return self.model_.predict_proba(self._x(X))[:, 1]


class LogisticRegressionModel(BaseModel):
    """Model 1 - Logistic Regression (linear model, week 9).

    Equation:  P(y = 1 | x) = 1 / (1 + exp(-(b + w_1 x_1 + ... + w_d x_d)))
    What it does: adds up every standardised input times its weight w_j and
    squashes the sum into a probability. Each w_j says how much that input
    pushes the risk up (w_j > 0) or down (w_j < 0); exp(w_j) is the odds ratio
    for a 1-SD increase. With `signs` the weights are forced to respect the
    clinical direction of each risk factor (see ConstrainedLogisticRegression).
    Why it is in the system: transparent, well calibrated and the clinical
    reference model; it captures the global linear trend.
    """
    name, short, color = "Logistic Regression", "LR", MODEL_COLORS["LR"]

    def __init__(self, signs: list[int] | None = None, **kwargs) -> None:
        self.signs = signs
        super().__init__(**kwargs)

    def search_estimator(self):
        clf = (ConstrainedLogisticRegression(signs=self.signs) if self.signs is not None
               else LogisticRegression(max_iter=3000, random_state=self.seed))
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    def default_space(self):
        return {"clf__C": loguniform(1e-3, 1e2)}


class KNNModel(BaseModel):
    """Model 2 - K-Nearest Neighbours (instance-based model, week 10).

    Equation (Minkowski distance, p = 1 Manhattan, p = 2 Euclidean):
        d(x, x') = ( sum_j |x_j - x'_j|^p )^(1/p)
    Equation (prediction):
        P(y = 1 | x) = sum_{i in N_k(x)} v_i y_i / sum_{i in N_k(x)} v_i
        with v_i = 1 (uniform) or v_i = 1 / d(x, x_i) (distance weighting).
    What it does: finds the k most similar training cases and returns the share
    of them that are positive. It makes no assumption about the shape of the
    boundary. Why it is in the system: captures local patterns ("patients like
    this one") that a linear model misses.
    """
    name, short, color = "K-Nearest Neighbors", "KNN", MODEL_COLORS["KNN"]

    def search_estimator(self):
        return Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_jobs=-1))])

    def default_space(self):
        return {"clf__n_neighbors": randint(3, 80), "clf__weights": ["uniform", "distance"], "clf__p": [1, 2]}


class RandomForestModel(BaseModel):
    """Model 3 - Random Forest (ensemble of decision trees, week 11).

    Equation (Gini impurity of a node with class shares p_c):
        G = 1 - sum_c p_c^2
        What it does: measures how mixed a node is (0 = pure). Each tree picks
        the split that lowers G the most, using a random subset of the inputs.
    Equation (forest prediction, B = 300 trees):
        P(y = 1 | x) = (1 / B) * sum_b P_b(y = 1 | x)
        What it does: averages the votes of many decorrelated trees, which
        lowers the variance of a single deep tree (bagging).
    With `monotonic_cst` no split may contradict a clinical direction.
    Why it is in the system: learns non-linear thresholds and interactions
    (e.g. "high pressure AND older age") without scaling the inputs.
    """
    name, short, color = "Random Forest", "RF", MODEL_COLORS["RF"]

    def __init__(self, monotonic_cst: list[int] | None = None, **kwargs) -> None:
        self.monotonic_cst = monotonic_cst
        super().__init__(**kwargs)

    def search_estimator(self):
        return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=self.seed,
                                      monotonic_cst=self.monotonic_cst)

    def default_space(self):
        return {"max_depth": [None, 10, 16, 24], "min_samples_leaf": [1, 2, 5, 10, 20],
                "max_features": ["sqrt", 0.3, 0.5]}


class SVMModel(BaseModel):
    """Model 4 - Support Vector Machine with RBF kernel (week 12).

    Equation (soft-margin objective):
        min_{w,b,xi}  ||w||^2 / 2 + C * sum_i xi_i   s.t.  y_i (w.phi(x_i) + b) >= 1 - xi_i,  xi_i >= 0
        What it does: finds the boundary with the widest margin between the
        classes; C sets how much a misclassified training case costs.
    Equation (RBF / Gaussian kernel):
        K(x, x') = exp(-gamma * ||x - x'||^2)
        What it does: measures similarity; it lets the SVM draw curved
        boundaries. gamma sets how local each training case's influence is.
    Equation (decision function):  f(x) = sum_i alpha_i y_i K(x_i, x) + b
    Equation (Platt scaling):        P(y = 1 | x) = 1 / (1 + exp(A f(x) + B))
        What it does: turns the SVM margin (which is not a probability) into a
        calibrated probability; A and B are fitted by 3-fold cross-validation.
    Why it is in the system: strong on high-dimensional inputs such as the
    90-sample ECG waveform.
    """
    name, short, color = "Support Vector Machine", "SVM", MODEL_COLORS["SVM"]
    max_train = config.SVM_MAX_TRAIN

    def search_estimator(self):
        return Pipeline([("scaler", StandardScaler()),
                         ("clf", SVC(kernel="rbf", cache_size=1000, random_state=self.seed))])

    def default_space(self):
        return {"clf__C": loguniform(0.1, 100), "clf__gamma": loguniform(1e-3, 1)}

    def final_estimator(self):
        svc = clone(self.search_estimator()).set_params(**self.params).named_steps["clf"]
        return Pipeline([("scaler", StandardScaler()),
                         ("clf", CalibratedClassifierCV(svc, method="sigmoid", cv=3))])


def default_models() -> list[BaseModel]:
    return [LogisticRegressionModel(), KNNModel(), RandomForestModel(), SVMModel()]
