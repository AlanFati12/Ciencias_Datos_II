# CardioAI — telemedicine web application

Streamlit application that puts the trained models of [`../ml-pipeline`](../ml-pipeline) to work.
It does not train anything: it loads the stacking ensembles that the pipeline exported, so every
answer comes from the mathematical model documented in `ml-pipeline/docs/mathematical_model.md`.
The interface is bilingual (English / Spanish, selector in the sidebar).

## Design

The interface is built for a health professional with no machine-learning background:

* **Two guided paths:** *Assess a patient* and *Analyse an ECG*. Each one has two steps (data → result).
* **Answer first, in plain language:** a coloured result card (low / inconclusive / high) with one recommendation.
  The detail (why, what-if, the four models and the equation) is behind tabs and expanders.
* **Bilingual:** Spanish by default, English from the sidebar (remembered in the URL). Numbers follow each
  language's conventions (67.772 · 98,1 % in Spanish).
* **Nobody gets lost:** a 4-step welcome tour on the first visit (reopen it from *Quick guide* in the sidebar),
  an eyebrow on every page that says where you are, and a glossary of every term in *How it works*.
* **Every chart explains itself:** each chart answers one question, labels its reference lines on the chart
  (threshold, doubt zone, 95 % target, "your patient"), and has a **"How to read this chart?"** button with
  *what it shows* and *how to read it* (`core/guides.py`). The 50+ training figures in *How it works → Figures*
  show the same kind of guide next to them, read from the pipeline's `run_manifest.json`.
* **One visual system:** design tokens and a registered Plotly template (`core/theme.py`), the same
  colour-blind-safe palette as the pipeline figures (checked with a colour-vision-deficiency simulator), and
  colour never used alone (ECG beats use colour **and** marker shape; risk levels carry a text label).

## Functionalities

| # | Functionality | Page |
|---|---|---|
| 1 | Patient risk assessment: P(CVD), plain-language result, referral when uncertain | Assess a patient |
| 2 | How the 4 models decided: each model's probability, the meta-model weights, the live equation | Assess a patient → Technical detail |
| 3 | Similar-patient group (PCA + K-means) with its observed prevalence | Assess a patient → Why this result? |
| 4 | Explanation: the factors that raise or lower this patient's risk, in points | Assess a patient → Why this result? |
| 5 | What-if simulator: change blood pressure, weight or habits and see the risk move | Assess a patient → Simulate changes |
| 6 | ECG analysis: example / CSV / WFDB → automatic beat detection → normal vs. abnormal → burden | Analyse an ECG |
| 7 | ECG viewer with "previous / next abnormal" navigation, beat inspector, rhythm plot, beat table | Analyse an ECG |
| 8 | Several patients from a CSV, with downloadable results (and a comparison if the true label is present) | Several patients |
| 9 | Patient history (SQLite): filter, search, readable detail, export, delete with confirmation | History |
| 10 | Model results and traceability: accuracy per model, equation, unseen-patient checks, figures, lineage | How it works |
| + | PDF report of a risk assessment or an ECG analysis | Assess a patient / Analyse an ECG |
| + | Bilingual interface (Spanish / English) | Sidebar |

## How to run it

The models must exist first (they are not versioned because of their size):

```bash
cd ml-pipeline
python main.py            # writes outputs/clinical/models and outputs/ecg/models
```

Then:

```bash
cd web-app
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py               # opens http://localhost:8501
```

`scikit-learn` is pinned to the version that trained the models, because pickled models only load
reliably with the same version. If you change it, re-run the pipeline.

## Structure

```
web-app/
├── app.py                 # entry point: logo, navigation, language selector, styles
├── core/                  # logic without UI + the shared visual components
│   ├── model_store.py     # loads ../ml-pipeline/outputs/<tier>/models (+ the cardio_ml package)
│   ├── clinical.py        # validation, prediction, risk levels, local explanation, what-if sweeps
│   ├── ecg.py             # signal reading, beat detection + classification, rhythm summary
│   ├── storage.py         # SQLite history (data/history.db, not versioned)
│   ├── report_pdf.py      # PDF reports (fpdf2)
│   ├── theme.py           # design tokens + the 'cardioai' Plotly template (colour-blind-safe palette)
│   ├── charts.py          # the Plotly charts; each answers one question and labels itself
│   ├── guides.py          # "How to read this chart?" texts and the glossary (Spanish / English)
│   ├── ui.py              # design system: hero, cards, result card, chart header + guide, welcome tour
│   └── i18n.py            # Spanish / English texts and number formatting
├── views/                 # home, risk, ecg_analysis, history, batch, how_it_works
├── assets/                # logo and icon (SVG)
├── sample_data/           # demo ECG excerpts and batch CSV examples
└── .streamlit/config.toml # theme (colours, Inter font, radii)
```

## How an ECG is processed in the app

1. The signal is band-pass filtered (0.5–40 Hz) and resampled to 128 Hz, as in training.
2. The beats are found by the XQRS detector (`wfdb`) and aligned to their main deflection.
3. Each beat gets the same 100 features as in training (rhythm, amplitude, patient-relative, waveform).
4. The 4 models and the meta-model classify every beat. The probability is corrected to the real
   prevalence of abnormal beats (prior shift, `deployment_prior` in the model schema).

Accuracy: 96.9 % on the held-out test beats (standard MIT-BIH beat-level protocol), above the > 95 % target.
As an additional analysis, a patient-wise 5-fold cross-validation gives about 90 % ± 4 % on completely new
patients, and the automatic detector finds 99.5 % of their beats
(`ml-pipeline/outputs/ecg/tables/automatic_detection_metrics.csv`). Both are shown in *How it works → Results*.

> Academic prototype for decision support. It is not a certified medical device.
