# Data

No data is stored in this repo. Put input files here locally; everything in this folder except this README is git-ignored.

## Files the scripts expect

| File | Used by | Contents |
|---|---|---|
| `data_5.xlsx` | `clean_data.py` | Raw export, sheet `Ayush-Records`, one row per respondent |
| `Questions.xlsx` | `clean_data.py` | Sheet `Timeline QID` with columns `ordering` (timeline question id) and `question` (question text) |
| `data_6.xlsx` | `eda_eac.py`, `eac_baseline.py` | Cleaned output of `clean_data.py` (first sheet is read) |

## Required columns in the cleaned data

| Column | Role |
|---|---|
| `diagnostic_response_id` | Respondent id (dropped before training) |
| `Link`, `Campaign` | Identifiers (dropped before training) |
| `final_eac_protect` | Target: Protect label |
| `final_eac_esc` | Target: Escalating label |
| `final_osb`, `final_asb`, other `final_*` | Upstream labels, used as inputs |
| Question columns | One column per question, header = question text |

Allowed target labels: `Anxiety`, `Anxiety-Apathy`, `Anger`, `Anger-Apathy`, `Covert Anger`, `Covert Anger-Apathy`, and (Escalating only) `Frustration`.
