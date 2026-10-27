# Mathematical model

This document lists every equation the code implements, in the order the data
flows through the system. Each section names the file that implements it, so the
report can trace every formula back to the code.

## 0. Notation

| Symbol | Meaning |
|---|---|
| $\mathbf{x}_i \in \mathbb{R}^d$ | feature vector of case $i$ (a patient in tier 1, a heartbeat in tier 2) |
| $y_i \in \{0,1\}$ | true label (1 = CVD in tier 1, 1 = abnormal beat in tier 2) |
| $n,\ d$ | number of cases, number of features |
| $p_k(\mathbf{x})$ | probability that base model $k$ assigns to the positive class |
| $\hat{P}(\mathbf{x})$ | final probability returned by the system |
| $\sigma(z) = 1/(1+e^{-z})$ | logistic (sigmoid) function |

## 1. System overview

```
Tier 1 (risk screening)     clinical exam ──► ETL ──► 4 base models ──► meta-model ──► P(CVD) + risk segment
Tier 2 (diagnosis)          single-lead ECG ──► ETL ──► 4 base models ──► meta-model ──► P(abnormal beat)
```

Both tiers share the same mathematical core (sections 4–6). Only the feature
extraction (sections 2–3) differs.

## 2. Tier 1 – clinical feature engineering (`preprocessing.py`)

$$\text{age}_{\text{years}} = \frac{\text{age}_{\text{days}}}{365.25} \qquad
\text{BMI} = \frac{\text{weight [kg]}}{\left(\text{height [cm]}/100\right)^2}$$

$$\text{MAP} = \frac{\text{SBP} + 2\,\text{DBP}}{3} \qquad \text{PP} = \text{SBP} - \text{DBP}$$

AHA 2017 blood-pressure stage:

$$\text{bp\_stage} = \begin{cases}
4 & \text{SBP} > 180 \ \lor\ \text{DBP} > 120 \quad\text{(crisis)}\\
3 & \text{SBP} \ge 140 \ \lor\ \text{DBP} \ge 90 \quad\text{(stage 2)}\\
2 & \text{SBP} \ge 130 \ \lor\ \text{DBP} \ge 80 \quad\text{(stage 1)}\\
1 & \text{SBP} \ge 120 \quad\text{(elevated)}\\
0 & \text{otherwise (normal)}
\end{cases}$$

**Outlier rule (EDA only).** A value is flagged when it lies outside
$[Q_1 - 1.5\,\text{IQR},\ Q_3 + 1.5\,\text{IQR}]$ with $\text{IQR} = Q_3 - Q_1$.
Records are removed only when the value is physiologically impossible
(the ranges are in `config.CLINICAL_RANGES`).

**Multicollinearity.** For each feature $j$, regress it on the others and take

$$\text{VIF}_j = \frac{1}{1 - R_j^2}.$$

MAP and PP are exact linear combinations of SBP and DBP ($\text{VIF}\to\infty$),
and weight is explained by BMI and height ($\text{VIF}\approx 100$). These three
are not used as model inputs. After removing them, every VIF is below 5.

**Class balance (random undersampling).** With $n_0$ negatives and $n_1$ positives,
keep $m = \min(n_0, n_1)$ randomly chosen cases of each class.

**Standardisation** (fitted on the training set only, inside each model pipeline):

$$z_{ij} = \frac{x_{ij} - \mu_j}{s_j}$$

## 3. Tier 2 – ECG signal processing (`ecg_etl.py`)

1. **Band-pass filter.** Butterworth order 3, 0.5–40 Hz, applied forward and
   backward (zero phase, so the R peak does not shift):
   $|H(j\omega)|^2 = \dfrac{1}{1 + (\omega/\omega_c)^{2N}}$ for each cut-off $\omega_c$, with $N = 3$.
2. **Resampling** to $f_s = 128$ Hz by polyphase filtering with ratio
   $L/M = 128/360 = 16/45$ (MIT-BIH). SVDB is already at 128 Hz.
3. **Fiducial alignment.** Every R peak, whether annotated by a cardiologist (training) or found by the XQRS
   detector (web app), is moved to the largest absolute deflection of the band-passed signal within ±75 ms:
   $r^\* = \arg\max_{|t - r| \le 75\,\text{ms}} |x_{\text{bp}}(t)|$. Training and deployment therefore cut identical windows.
4. **Segmentation.** For an R peak at sample $r$, the beat is
   $\mathbf{w} = \big(s[r-32], \dots, s[r+57]\big)$, i.e. 0.25 s before and 0.45 s after (90 samples).
5. **Per-beat normalisation:** $\tilde{w}_t = (w_t - \bar{w})/\text{sd}(\mathbf{w})$. Amplitude: $A = \max_t w_t - \min_t w_t$ (mV, before normalisation).
6. **Rhythm features.** With $R_i$ the time of beat $i$:

$$\text{RR}^{\text{pre}}_i = R_i - R_{i-1},\quad \text{RR}^{\text{post}}_i = R_{i+1} - R_i,\quad
\overline{\text{RR}}_i = \frac{1}{K}\sum_{k=1}^{K} \text{RR}^{\text{pre}}_{i-k+1}\ (K = 10)$$

$$\text{rr\_ratio}_i = \frac{\text{RR}^{\text{pre}}_i}{\overline{\text{RR}}_i},\qquad
\text{post\_pre\_ratio}_i = \frac{\text{RR}^{\text{post}}_i}{\text{RR}^{\text{pre}}_i}$$

A premature ectopic beat has $\text{rr\_ratio} < 1$ followed by a compensatory pause
($\text{post\_pre\_ratio} > 1$). This is the clinical signature the models learn.

7. **Patient-relative features.** Each beat is compared with the patient's own dominant beat,
   the median template $\mathbf{m} = \text{median}_i\,\tilde{\mathbf{w}}_i$ over all beats of the recording.
   No label is used to build it, so it is also available in deployment.

$$\text{template\_corr}_i = \rho(\tilde{\mathbf{w}}_i, \mathbf{m}),\qquad
\text{template\_dist}_i = \sqrt{\tfrac{1}{90}\textstyle\sum_t(\tilde w_{it} - m_t)^2}$$

$$\text{amplitude\_rel}_i = \frac{A_i}{\text{median}_j A_j},\qquad \text{rr\_rel}_i = \frac{\text{RR}^{\text{pre}}_i}{\text{median}_j\,\text{RR}^{\text{pre}}_j}$$

   A ventricular beat has a shape that differs from the patient's usual beat
   (low `template_corr`, high `template_dist`). These features let the models generalise to new
   patients, whose absolute amplitudes and heart rates differ from the training patients.

8. **Label (AAMI EC57).** $y = 0$ for class N (N, L, R, e, j); $y = 1$ for classes
   S (A, a, J, S), V (V, E) and F (F). Paced records 102, 104, 107 and 217 are excluded.

Final feature vector: $\mathbf{x} \in \mathbb{R}^{100}$ (5 rhythm, 1 amplitude, 4 patient-relative, 90 waveform samples).

## 4. The four base models (`models.py`)

### 4.1 Logistic regression (LR)

$$p_{\text{LR}}(\mathbf{x}) = \sigma(\mathbf{w}^\top\mathbf{x} + b)$$

It is trained by minimising the L2-regularised log-loss:

$$\min_{\mathbf{w},b}\ \frac{1}{2C}\|\mathbf{w}\|^2 - \sum_{i=1}^{n}\Big[y_i\log p_i + (1-y_i)\log(1-p_i)\Big]$$

Odds ratio per standard deviation of feature $j$: $\text{OR}_j = e^{w_j}$.

### 4.2 K-nearest neighbours (KNN)

The Minkowski distance ($q = 1$ Manhattan, $q = 2$ Euclidean) on standardised features:

$$d(\mathbf{x}, \mathbf{x}_i) = \Big(\sum_{j=1}^{d}|x_j - x_{ij}|^q\Big)^{1/q}$$

With $\mathcal{N}_k(\mathbf{x})$ the $k$ closest training cases and weights
$v_i = 1$ (uniform) or $v_i = 1/d(\mathbf{x},\mathbf{x}_i)$ (distance):

$$p_{\text{KNN}}(\mathbf{x}) = \frac{\sum_{i\in\mathcal{N}_k} v_i\,\mathbb{1}[y_i = 1]}{\sum_{i\in\mathcal{N}_k} v_i}$$

### 4.3 Random forest (RF)

$T$ CART trees are each grown on a bootstrap sample. Every split uses a random subset of
features and maximises the decrease of the Gini impurity:

$$G(S) = 1 - \sum_{c\in\{0,1\}} \pi_c^2,\qquad
\Delta G = G(S) - \frac{|S_L|}{|S|}G(S_L) - \frac{|S_R|}{|S|}G(S_R)$$

$$p_{\text{RF}}(\mathbf{x}) = \frac{1}{T}\sum_{t=1}^{T} \pi_{1}^{(t)}\big(\text{leaf}_t(\mathbf{x})\big)$$

### 4.4 Support vector machine (SVM, RBF kernel)

The soft-margin dual problem is

$$\max_{\boldsymbol\alpha}\ \sum_i \alpha_i - \frac{1}{2}\sum_{i,j}\alpha_i\alpha_j \tilde y_i \tilde y_j K(\mathbf{x}_i,\mathbf{x}_j)
\quad \text{s.t. } 0 \le \alpha_i \le C,\ \sum_i \alpha_i \tilde y_i = 0$$

with $\tilde y = 2y - 1$ and $K(\mathbf{x},\mathbf{x}') = \exp(-\gamma\|\mathbf{x}-\mathbf{x}'\|^2)$.

Decision function: $f(\mathbf{x}) = \sum_i \alpha_i \tilde y_i K(\mathbf{x}_i, \mathbf{x}) + b$.
The margin is turned into a probability with Platt scaling (fitted by 3-fold cross-validation):

$$p_{\text{SVM}}(\mathbf{x}) = \sigma\big(A f(\mathbf{x}) + B\big)$$

### 4.5 Hyperparameter search

A randomised search ($n_{\text{iter}} = 8$, 3-fold stratified CV, 12,000-row stratified subsample)
maximises the CV accuracy. The search spaces are in `models.py` and `workflows.py`, and the
chosen values are recorded in `run_manifest.json`. In tier 2 every patient contributes thousands of
beats, so the 3 folds are **grouped by patient** (`StratifiedGroupKFold`): the hyperparameters that win
are the ones that generalise to new patients, not the ones that best memorise known patients.

## 5. How the four models communicate: stacked generalisation (`ensemble.py`)

**Step 1 – out-of-fold predictions.** The training set is split into $K = 5$ stratified folds
$F_1,\dots,F_K$ (in tier 2 grouped by patient, so a beat is never scored by a model that saw other
beats of the same person). For each model $k$ and fold $f$, the model is trained on the other folds and
predicts the held-out fold:

$$z_{ik} = p_k^{(-f)}(\mathbf{x}_i)\quad \text{for } i \in F_f$$

Every case therefore gets 4 probabilities from models that never saw it. Together they form
the meta-feature matrix $\mathbf{Z} \in [0,1]^{n\times 4}$.

**Step 2 – meta-model.** A logistic regression learns how much to trust each model:

$$\boxed{\ \hat{P}(\mathbf{x}) = \sigma\Big(\beta_0 + \beta_1\,p_{\text{LR}}(\mathbf{x}) + \beta_2\,p_{\text{KNN}}(\mathbf{x}) + \beta_3\,p_{\text{RF}}(\mathbf{x}) + \beta_4\,p_{\text{SVM}}(\mathbf{x})\Big)\ }$$

$$\hat{\boldsymbol\beta} = \arg\min_{\boldsymbol\beta}\ \frac{1}{2}\|\boldsymbol\beta_{1:4}\|^2 - \sum_i \Big[y_i\log\hat P_i + (1-y_i)\log(1-\hat P_i)\Big]
\quad \text{s.t. } \beta_1,\dots,\beta_4 \ge 0$$

The non-negativity constraint means that a base model that sees more risk can never lower the final answer.
Without it, two highly correlated models (LR and SVM, $\rho = 0.99$) received weights of opposite sign.

**Step 3 – final answer.** After the meta-model is trained, the 4 base models are refitted on the
whole training set. For a new case:

$$\hat{y} = \mathbb{1}\big[\hat{P}(\mathbf{x}) \ge 0.5\big]$$

**Referral rule (clinical safety).** The system does not answer on its own, and refers the case to a
clinician, when

$$0.35 < \hat{P}(\mathbf{x}) < 0.65 \quad\text{or}\quad \sum_k \mathbb{1}[p_k(\mathbf{x}) \ge 0.5] = 2 \ \text{(2–2 split vote)}$$

**Prior-shift correction (deployment on real prevalence).** The models are trained on balanced data
($\pi_{\text{train}} = 0.5$), but in a real ECG only a fraction $\pi$ of the beats are abnormal. By Bayes'
theorem, the posterior is corrected by shifting its logit:

$$\operatorname{logit}\hat P'(\mathbf{x}) = \operatorname{logit}\hat P(\mathbf{x}) + \ln\frac{\pi}{1-\pi} - \ln\frac{\pi_{\text{train}}}{1-\pi_{\text{train}}}$$

$\pi$ is estimated on the training patients **before** undersampling, never on the test set. This
correction is used in the patient-wise validation (section 6b) and in the web app, where the ECG keeps its
natural class balance.

**Baselines the stacking is compared against.**

$$\text{soft vote: } \bar p(\mathbf{x}) = \frac{1}{4}\sum_k p_k(\mathbf{x}) \qquad
\text{hard vote: } \hat y = \mathbb{1}\Big[\sum_k \mathbb{1}[p_k \ge 0.5] > 2\Big]\ \text{(ties broken by } \bar p\text{)}$$

## 5b. Clinical plausibility constraints, tier 1 (`models.py`, `plausibility.py`)

In this dataset some associations contradict clinical evidence. Patients with very high cholesterol **and**
very high glucose show a lower CVD prevalence (69 %) than those with very high cholesterol and normal glucose
(82 %), and smokers (47 %) and alcohol drinkers (48 %) show less disease than non-smokers (50 %). These are
known biases of this self-reported dataset. Unconstrained models learn them faithfully, and in addition the
step-shaped random forest and KNN make the risk go up and down when age or weight increase. To keep the answer
clinically coherent, every risk factor $j$ gets a direction $s_j \in \{+1, -1, 0\}$ (`config.CLINICAL_MONOTONIC`):
$+1$ for age, BMI, systolic/diastolic pressure, BP stage, cholesterol, glucose, smoking and alcohol; $-1$ for
physical activity; $0$ (free) for sex and height.

* **Sign-constrained logistic regression.** The same objective as in §4.1, minimised with L-BFGS-B under box
  constraints: $s_j w_j \ge 0$ for every constrained feature.
* **Monotone random forest.** Each tree only accepts a split on a constrained feature $j$ if the predicted
  probabilities of its children respect the direction, $s_j\,(\bar p_{\text{right}} - \bar p_{\text{left}}) \ge 0$,
  and the children's values are bounded by their parents' (`monotonic_cst` in scikit-learn). Every tree, and
  therefore their average, is then monotone in $x_j$.
* **KNN and SVM** cannot be constrained, so they do not receive the three inputs whose association in the data
  contradicts the evidence (glucose, smoking, alcohol). They also use smooth settings, because with 11 noisy inputs
  flexibility only adds wiggles, not accuracy. KNN estimates the local CVD prevalence, whose standard error is
  $\sqrt{p(1-p)/k}$: with $k = 199$ it was ≈ 3.5 points and 99 % of the patients had a wrong-way dip when their
  weight rose. With $k = 2500$ it is ≈ 1 point and the dips disappear, at a cost of 0.8 accuracy points for KNN
  alone. The SVM uses a wide RBF kernel ($\gamma \le 0.05$).
* **Non-negative meta-model weights** (§5).

**Plausibility audit.** For 400 test patients, each input $x_j$ is swept over its range with everything else fixed
(age 40–65, weight 50–130 kg, systolic pressure 100–200 mmHg, the three levels of cholesterol and glucose,
smoking, alcohol, activity). A patient *violates* the direction for model $m$ if there is a step where

$$s_j\,\big(p_m(x_j^{(t+1)}) - p_m(x_j^{(t)})\big) < -0.01,$$

i.e. the probability moves more than 1 percentage point the wrong way. The share of violating patients is
reported per model and per factor (`tables/plausibility_audit.csv`), so the effect of the constraints is traceable.

## 6. Evaluation (`evaluation.py`)

With $TP, TN, FP, FN$ taken from the confusion matrix on the held-out test set (20 %):

$$\text{Accuracy} = \frac{TP+TN}{TP+TN+FP+FN}\qquad
\text{Precision} = \frac{TP}{TP+FP}\qquad
\text{Recall (sensitivity)} = \frac{TP}{TP+FN}$$

$$\text{Specificity} = \frac{TN}{TN+FP}\qquad
F_1 = \frac{2\cdot\text{Precision}\cdot\text{Recall}}{\text{Precision}+\text{Recall}}\qquad
\text{Balanced acc.} = \frac{\text{Recall}+\text{Specificity}}{2}$$

$$\text{MCC} = \frac{TP\cdot TN - FP\cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}\qquad
\text{Brier} = \frac{1}{n}\sum_i(\hat P_i - y_i)^2$$

$$\text{ROC-AUC} = P\big(\hat P(\mathbf{x}^+) > \hat P(\mathbf{x}^-)\big)$$

**95 % bootstrap confidence interval.** Resample the test set with replacement $B = 1000$ times,
recompute the accuracy each time, and take the 2.5 % and 97.5 % percentiles.

**McNemar test** (stacking vs. best base model). With $b$ = cases stacking gets right and the
base model gets wrong, and $c$ the reverse:
$b \sim \text{Binomial}(b+c,\ 0.5)$ under $H_0$ (same error rate). The exact two-sided p-value is reported.

**Selective prediction (coverage curve).** Sort the cases by confidence $|\hat P - 0.5|$ and report the
accuracy on the top fraction $c$ (coverage).

## 6b. Additional analysis: data-leakage audit and patient-wise validation, tier 2 (`workflows.py`)

**Why it is needed.** The official tier-2 accuracy (§6, 96.9 %) follows the standard MIT-BIH protocol: a stratified random split *by beat*, with test beats never used for training. A 30-minute recording has
≈ 2,000 beats of the same person, so almost every test beat has "siblings" of the same patient in the
training set. Two quantities measure this leakage:

$$\text{overlap} = \frac{1}{|T|}\sum_{i\in T} \mathbb 1\big[\text{patient}(i) \in \text{patients}(\text{train})\big]
\qquad
\text{NN}_{\text{same}} = \frac{1}{|T|}\sum_{i \in T} \mathbb 1\big[\text{patient}(i) = \text{patient}(\operatorname{NN}(i))\big]$$

where $\operatorname{NN}(i) = \arg\min_{j\in\text{train}} \|\mathbf z_i - \mathbf z_j\|_2$ on standardised features.
If both are close to 1, the beat-level test shares patients with training (the *intra-patient*
protocol used by much of the literature), and performance on completely new patients is expected to be
lower (de Chazal et al., 2004; Luz et al., 2016). The patient-wise validation below measures it and is
reported as a limitation of the system (90.5 % ± 4.4 %).

**Patient-wise K-fold cross-validation (strict, *inter-patient* protocol).** The $P$ patients are split into
$K = 5$ disjoint groups $G_1,\dots,G_K$ (`StratifiedGroupKFold`, records 201/202 of the same subject kept together).
For each fold $k$ a fresh copy of the 4 tuned models and the meta-model is trained on the balanced beats of the
other groups and evaluated on the beats of $G_k$:

$$\text{Acc}_k = \text{Acc}\big(\text{model trained on } \textstyle\bigcup_{j\ne k} G_j,\ \text{balanced test from } G_k\big),
\qquad \overline{\text{Acc}} = \frac{1}{K}\sum_k \text{Acc}_k,\quad
s = \sqrt{\tfrac{1}{K-1}\sum_k (\text{Acc}_k - \overline{\text{Acc}})^2}$$

The same folds are also scored with the natural class balance (prior-corrected, §5) and per AAMI class
(recall of N, S, V and F). The spread $s$ shows how much the result depends on *which* patients are new.

**Deployment check (tier 2).** On the new patients of fold 1, the beats are located by the XQRS detector instead of
the annotations. A detection counts as correct when it lies within 150 ms of an annotated beat
(detection sensitivity = found annotated beats / annotated beats; PPV = correct detections / detections). Detected beats
are then classified exactly as in the web app.

## 7. Explainability (`explainability.py`)

**Permutation importance** of a feature group $G$:

$$I_G = \text{Acc}(\mathbf{X}) - \frac{1}{R}\sum_{r=1}^{R}\text{Acc}\big(\mathbf{X}^{(G,\pi_r)}\big)$$

where $\mathbf{X}^{(G,\pi_r)}$ shuffles the columns of $G$ together with the random permutation $\pi_r$ ($R = 3$).

## 8. Risk segmentation, tier 1 (`segmentation.py`)

1. Standardise 7 clinical variables. Then **PCA**: keep the first $q$ eigenvectors of the covariance matrix
   $\boldsymbol\Sigma = \frac{1}{n}\mathbf{Z}^\top\mathbf{Z}$ that explain at least 90 % of the variance,
   $\sum_{j\le q}\lambda_j / \sum_j \lambda_j \ge 0.90$.
2. **K-means** ($k = 3$) minimises

$$J = \sum_{c=1}^{k}\sum_{\mathbf{u}_i \in C_c}\|\mathbf{u}_i - \boldsymbol\mu_c\|^2$$

3. Quality is measured with the silhouette $s_i = \dfrac{b_i - a_i}{\max(a_i, b_i)}$, where $a_i$ is the
   mean intra-cluster distance and $b_i$ the mean distance to the nearest other cluster.
4. The clusters are ranked by observed CVD prevalence and named Low / Medium / High risk.
