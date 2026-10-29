# EAC Baseline Models

Baseline models for predicting a respondent's **Emotional Addiction Cycle (EAC)** from questionnaire answers. Each person gets two labels:

- **Protect:** how they protect themselves. One of 6 labels: Anxiety, Anxiety-Apathy, Anger, Anger-Apathy, Covert Anger, Covert Anger-Apathy.
- **Escalating:** what happens when things escalate. The same 6 labels plus Frustration.

This repo holds the full path from raw data to a tested baseline:

1. Clean the data.
2. Explore it (EDA, exploratory data analysis).
3. Train six small yes/no models that follow the EAC prediction flow.
4. Test the whole flow on people the models never saw.

Every later model is measured against this baseline.

---

## Contents

- [How the prediction flow works](#how-the-prediction-flow-works)
- [Results](#results)
- [Project layout](#project-layout)
- [Setup](#setup)
- [How to run](#how-to-run)
- [Outputs](#outputs)
- [Documentation](#documentation)
- [Data and privacy](#data-and-privacy)

---

## How the prediction flow works

The flow is a short chain of yes/no questions. Each model gives a score, the score is compared with that model's own **threshold** (cut-off), and the yes/no answer decides which model runs next. The full diagram is in [`docs/eac_prediction_flow.drawio`](docs/eac_prediction_flow.drawio). Open it in draw.io; it has two pages.

### Protect chain

```
eac_protect_anger ──no──▶ eac_protect_apathy ──▶ Anxiety / Anxiety-Apathy
        │
       yes ──▶ add covert scores ──▶ eac_protect_covert
                                         ├─yes─▶ eac_protect_apathy ──▶ Covert Anger / Covert Anger-Apathy
                                         └─no──▶ eac_protect_apathy ──▶ Anger / Anger-Apathy
```

### Escalating chain

The branch is picked by the Protect anger result, inverted.

```
Protect anger     ──▶ eac_esc_apathy ──▶ Anxiety / Anxiety-Apathy
Protect not anger ──▶ eac_esc_frustration ──yes─▶ Frustration
                               └─no─▶ eac_esc_covert (flag) ─▶ eac_esc_apathy
                                      ──▶ Anger / Anger-Apathy / Covert Anger / Covert Anger-Apathy
```

### The six models

| Model | Question it answers | Trained on |
|---|---|---|
| `eac_protect_anger` | Is the Protect label in the anger family? | everyone |
| `eac_protect_covert` | Is that anger covert (hidden)? | Protect anger-family people |
| `eac_protect_apathy` | Does apathy (withdrawing) come with it? | everyone |
| `eac_esc_frustration` | Is the Escalating label Frustration? | Protect not-anger people |
| `eac_esc_covert` | Is the Escalating anger covert? | Escalating anger branch, not Frustration |
| `eac_esc_apathy` | Does apathy come with the Escalating label? | everyone except Frustration |

**Covert scores** are three numbers made from 18 yes/no questions: an overt count, a covert count and their ratio. They are added right before `eac_protect_covert`. See [`docs/covert_scores.md`](docs/covert_scores.md).

---

## Results

These come from the run on 1,280 respondents, split 80/20 (1,024 to train, 256 to test, seed 42).

- **CV accuracy** (cross-validation): the model is trained on 4/5 of the training people and scored on the remaining 1/5. This is repeated 5 times so every training person is scored once by a model that never saw them.
- **"Always no"**: the accuracy a model would get by always answering no. A useful model must beat it.

| Model | Method chosen | CV accuracy | "Always no" | Test accuracy |
|---|---|---|---|---|
| `eac_protect_anger` | XGBoost | 88.3% | 54.9% | 89.1% |
| `eac_protect_covert` | XGBoost | 71.4% | 63.2% | 70.7% |
| `eac_protect_apathy` | Logistic regression | 95.4% | 91.5% | 96.1% |
| `eac_esc_frustration` | Logistic regression | 92.2% | 92.0% | 90.7% |
| `eac_esc_covert` | "Always no" fallback | 94.6% | 94.6% | 95.3% |
| `eac_esc_apathy` | XGBoost | 93.2% | 71.1% | 94.7% |

**Whole flow on the test set**, compared with always guessing the most common label:

| Chain | Flow accuracy | Most common label |
|---|---|---|
| Protect | **73.0%** | 48.8% (Anxiety) |
| Escalating | **78.9%** | 34.4% (Anger) |

Notes:

- `eac_esc_covert` has only 34 positive examples. Neither a model nor the covert-score rule beat "always no", so it never predicts covert. This was an agreed fallback.
- `eac_esc_frustration` is kept as a model by decision, even though it only just beats "always no".
- A fairness check (accuracy by gender and age group) is in the Excel report.

---

## Project layout

```
eac-baseline-models/
├── README.md
├── requirements.txt
├── config/
│   └── excluded_ids.example.txt   format of the respondent exclusion list
├── data/
│   └── README.md                  what input file the scripts expect (no data is stored here)
├── docs/
│   ├── methodology.md             full method: column roles, cleaning, models, thresholds, tests
│   ├── eda_findings.md            what the data exploration showed
│   ├── covert_scores.md           how the overt / covert scores are computed
│   └── eac_prediction_flow.drawio Protect and Escalating flow diagram
└── src/
    ├── clean_data.py              step 1: remove test and incomplete respondents
    ├── eda_eac.py                 step 2: data exploration (column types, blanks, labels)
    ├── eda_excel.py               builds the styled EDA Excel workbook (used by eda_eac.py)
    └── eac_baseline.py            step 3: train the six models, test the flow, write the report
```

---

## Setup

Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## How to run

Run every command from the repo root.

**1. Clean the data.** This removes rows with blank timeline answers, test ids and ids on the exclusion list.

```bash
cp config/excluded_ids.example.txt config/excluded_ids.txt   # then add the real ids
python src/clean_data.py --data data/data_5.xlsx --questions data/Questions.xlsx --out data/data_6.xlsx
```

**2. Explore the data (EDA).**

```bash
python src/eda_eac.py --file data/data_6.xlsx
# optional: also recompute the pipeline's covert scores by label
python src/eda_eac.py --file data/data_6.xlsx --eac-py path/to/ml/services/eac.py
```

**3. Train and test the baseline.**

```bash
python src/eac_baseline.py --file data/data_6.xlsx
```

| Script | Main options |
|---|---|
| `clean_data.py` | `--data`, `--questions`, `--out`, `--sheet`, `--exclude` |
| `eda_eac.py` | `--file`, `--sheet`, `--out`, `--eac-py`, `--keep-branch-breaks` |
| `eac_baseline.py` | `--file`, `--sheet`, `--out` |

The baseline script stops with a clear message if a check fails. Examples: a target column is missing, a label is outside the 7 allowed values, or the per-model row counts do not match the labels.

---

## Outputs

Everything is written to `outputs/`, which git ignores.

| Path | What it is |
|---|---|
| `outputs/eda/eda_report.xlsx` | EDA workbook: overview charts, label balance, covert scores, column guide |
| `outputs/eda/eda_report.txt` | Plain-text summary of the EDA |
| `outputs/eda/column_summary.csv` | Every column with its type, % blank, role and suggested action |
| `outputs/eda/category_values.csv` | Every answer option with its count |
| `outputs/baseline/eac_baseline_report.xlsx` | Baseline report. Sheets: Summary, Columns, Excluded rows, Models, Flow, Fairness, Test predictions |
| `outputs/baseline/models/*.joblib` | One saved model per step, plus the encoder (the step that turns answers into numbers) |
| `outputs/baseline/models/models_summary.json` | Each model's method, threshold and input columns |

---

## Documentation

- [Methodology](docs/methodology.md): the agreed method, step by step.
- [EDA findings](docs/eda_findings.md): what the data looks like and why some rows and columns are dropped.
- [Covert scores](docs/covert_scores.md): the 18 questions and the counting rules.

---

## Data and privacy

No respondent data, outputs or trained models are stored in this repo. The questionnaire data has personal answers and identifiers, so:

- `data/`, `outputs/`, `*.xlsx`, `*.csv` and `*.joblib` are git-ignored.
- The real exclusion list (`config/excluded_ids.txt`) is git-ignored. Only the example file is committed.
- Protected attributes (age, gender, sexual orientation, religion, children's age and gender) are **never** model inputs. They are only used to check accuracy by group.
