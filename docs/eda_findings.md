# EDA findings

This page summarises what the data exploration (EDA) of the cleaned questionnaire data (`data_6.xlsx`) showed. Run `python src/eda_eac.py` to regenerate the full workbook.

All numbers here are totals; no respondent-level data is included.

## Data at a glance

- 1,283 rows and 207 columns, with no duplicate respondent ids.
- 3 rows were excluded because they break the **branch rule**: a Protect anger-family label must go with an Escalating Anxiety or Anxiety-Apathy label, and the other way round. That leaves **1,280 rows**.
- The source file is not edited. `--keep-branch-breaks` turns the exclusion off.
- The branch rule held for 99.8% of rows before the exclusion and 100% after it, so the data supports the flow.

## Column roles

| Role | Columns | Used as model input? |
|---|---|---|
| Question (core) | 181 | Yes |
| Protected attribute | 11 | No: kept aside for the fairness check |
| Other `final_*` label | 10 | Yes, except `final_eac_origin` (one value in every row) |
| Identifier | 3 | No |
| Target | 2 | No: this is what we predict |

Question columns by type:

| Type | Columns |
|---|---|
| Yes / no | 102 |
| Category (pick one) | 54 |
| Multi-select (pick several) | 13 |
| Numeric | 9 |
| Free text | 4 |
| Short text | 2 |
| All blank | 8 |

## Blanks

A blank usually means "not asked": a follow-up question hidden by an earlier answer (skip logic). It does not mean "forgotten", so blanks are kept as their own value rather than filled in.

| % blank | 0–5% | 5–20% | 20–50% | 50–80% | 80–95% | 95–100% |
|---|---|---|---|---|---|---|
| Question columns | 111 | 12 | 17 | 28 | 9 | 15 |

Columns that are 95% or more blank are dropped.

## Labels

| Protect | Count |
|---|---|
| Anxiety | 627 |
| Anger | 342 |
| Covert Anger | 202 |
| Anxiety-Apathy | 75 |
| Anger-Apathy | 23 |
| Covert Anger-Apathy | 11 |

| Escalating | Count |
|---|---|
| Anger | 439 |
| Anxiety | 400 |
| Anxiety-Apathy | 178 |
| Anger-Apathy | 172 |
| Frustration | 57 |
| Covert Anger | 31 |
| Covert Anger-Apathy | 3 |

## Training rows per model

| Model | Eligible rows | Positives |
|---|---|---|
| `eac_protect_anger` | 1,280 | 578 |
| `eac_protect_covert` | 578 | 213 |
| `eac_protect_apathy` | 1,280 | 109 |
| `eac_esc_frustration` | 702 | 57 |
| `eac_esc_covert` | 645 | 34 |
| `eac_esc_apathy` | 1,223 | 353 |

Why the counts differ:

- Every Protect not-anger row (702) goes to the Escalating anger branch.
- Apathy is not checked after Frustration, so `eac_esc_apathy` gets 1,280 − 57 = 1,223 rows.

## Things to watch

- **Rare labels:** Escalating Covert Anger (31) and Covert Anger-Apathy (3), and Protect Covert Anger-Apathy (11). Expect weak models there.
- **Near-constant columns:** `final_eac_origin` (one value everywhere) and `Were you adopted?` (97.7% one answer).
- **Train/serve mismatch:** the other `final_*` columns are reviewed values in training, but at prediction time they come from upstream models.
- **Covert scores** separate Anger from Anxiety well, but separate Covert Anger from plain Anger only weakly. See [covert_scores.md](covert_scores.md).
