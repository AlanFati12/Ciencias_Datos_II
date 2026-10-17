# Sample data

| File | Content |
|---|---|
| `ecg/*.csv` | 2-minute excerpts (lead MLII / ECG1, mV) of PhysioNet records, used as demo signals. `demos.json` lists the sampling rate and the cardiologist labels of each excerpt. |
| `batch_template.csv` | Template for the batch evaluation page (hypothetical patients). |
| `batch_example_test_patients.csv` | 300 patients from the held-out **test set** of the clinical tier, with their true diagnosis (`cardio`). |

ECG excerpts come from the MIT-BIH Arrhythmia Database (Moody & Mark, 2001) and the MIT-BIH Supraventricular
Arrhythmia Database (Greenwald, 1990), published on PhysioNet under the Open Data Commons Attribution License v1.0.
The clinical patients come from the Kaggle *Cardiovascular Disease dataset* (S. Ulianova, 2019).
