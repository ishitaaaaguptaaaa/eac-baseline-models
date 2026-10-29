# EAC baseline: methodology

Agreed 2026-10-15. Based on the EDA (`outputs/eda/eda_report.xlsx`, `outputs/eda/column_summary.csv`, summarised in [eda_findings.md](eda_findings.md)) and [eac_prediction_flow.drawio](eac_prediction_flow.drawio) (Protect and Escalating pages).

Goal: a simple, explainable baseline for the EAC flow. Every later model is measured against it.

Sections are in execution order: do them top to bottom. Counts in this document (rows, columns, positives) are for `data_6.xlsx`.

---

## 1. Working rules

- Stay within this method. New steps, features or models are added here first, then to the code.
- Stop and report when a check fails (the count check in section 7, a label outside the allowed values, a question text that does not match). Never fix it quietly.
- Code: `src/eac_baseline.py`, which reuses the column type rules from `src/eda_eac.py`.
- Outputs: `outputs/baseline/` (the Excel report and a `models/` folder). Outputs are not committed.
- The input `.xlsx` file is never edited.
- Libraries: pandas 2.2, openpyxl, scikit-learn 1.5.2, xgboost 3.2.0 (see `requirements.txt`).

---

## 2. The script: input and output

One self-contained script. Give it an `.xlsx` file; it cleans the data, decides every column's role itself, trains and tests the models, and writes one Excel report.

Run: `python src/eac_baseline.py --file <path to .xlsx> [--sheet 0] [--out outputs/baseline]`

### 2.1 How the script decides each column's role

Checked in this order. The first rule that matches decides; every column gets a role and a reason.

| Order | Role | Rule | Action |
|---|---|---|---|
| 1 | identifier | name is `diagnostic_response_id`, `Link` or `Campaign` | drop |
| 2 | target | name is `final_eac_protect` or `final_eac_esc` | used as labels |
| 3 | protected | name is in the protected list below | drop from inputs, keep aside for the fairness check |
| 4 | other label | any other name starting with `final_` | keep (unless rule 5 or 6 drops it) |
| 5 | blank | 95% or more blank | drop |
| 6 | single value | only one distinct answer | drop |
| 7 | text | free text or short text (type rules from `eda_eac.py`) | drop |
| 8 | feature | everything else | keep |

Protected list (11): `What is your age?`, `What is your gender?`, `How would you describe your sexual orientation?`, `Was there any religion in your home growing up?`, `How would you categorize the religion?`, `What is the gender of your oldest child?`, `What is their gender?`, `Placeholder - Gender of child 3`, `What is the age of your oldest child?`, `How old are they?`, `Placeholder - Age of child 3`.

Names are matched after the header cleaning in section 3 (step 3). If a column that is not on the list mentions age, gender, sexual orientation or religion, the script does not drop it but lists it under "Warnings" in the report for a person to review.

The script stops with a clear message if a target column is missing.

### 2.2 The Excel report

File: `outputs/baseline/eac_baseline_report.xlsx`. One sheet per topic, short and readable:

| Sheet | What it shows |
|---|---|
| Summary | input file; rows in, excluded and used; train/test sizes; column counts per role; headline accuracies (each model vs "always no", whole flow vs most common label); which method won per model; warnings |
| Columns | every input column: name, role, type, % blank, kept or dropped, reason |
| Excluded rows | each excluded row: Excel row number, id, both labels, reason |
| Models | per model: eligible rows and positives (train/test), method chosen (logistic regression, XGBoost or rule), settings, threshold, cross-validation accuracy, test accuracy, "always no" accuracy, secondary checks |
| Flow | whole-flow accuracy for Protect and Escalating vs most common label; 6x6 and 7x7 predicted vs true tables |
| Fairness | accuracy by gender and age group |
| Test predictions | each test person: id, true and predicted Protect label, true and predicted Escalating label |

---

## 3. Data cleaning

Source: the file given with `--file`. For this run: `data/data_6.xlsx` (1283 rows, 207 columns, no duplicate ids).

| # | Step | Why |
|---|---|---|
| 1 | Exclude rows that break the branch rule: a Protect anger-family label must go with an Escalating Anxiety or Anxiety-Apathy label, and a Protect not-anger label with an Escalating Anger-branch label. The script finds them itself. In `data_6.xlsx` they are Excel rows 640, 1056, 1149, leaving 1280 rows. The source file is not edited. | Their Protect and Escalating labels contradict the flow. |
| 2 | Clean label text. Use `Covert Anger-Apathy` in both chains (the drawio uses this spelling too). | One spelling for one label. |
| 3 | Clean headers: repair broken apostrophes (`don�t`) and turn curly apostrophes into straight ones before matching columns. | Otherwise columns selected by question text silently fail to match. |
| 4 | Remove the 3 identifiers and 11 protected columns (section 2.1). Keep the protected columns aside for the fairness check. | Only identifiers and protected attributes are removed up front. |
| 5 | Drop the 11 columns that are 95% or more blank, the 6 free-text and short-text columns, and `final_eac_origin` (same value in every row). | No usable signal for a baseline. |
| 6 | Yes/no answers: trim, lower-case, then `yes` = 1 and `no` = 0. | Same rule as `eac.py`. |
| 7 | Blank answers stay as their own value, "not asked". | A blank is usually a skip-logic follow-up that was not shown, not a forgotten answer. |
| 8 | Turn the remaining answers into numbers: category and multi-select answers become 0/1 columns (pandas `get_dummies`), with blank as its own column. | The models only read numbers. |
| 9 | Number columns (for example "How many siblings did you have?", "What age did the substance abuse start?"): split the answered values into 4 bands of about equal size, worked out on the training 80% (section 6). Blank is its own band, "not asked". Then turn the bands into 0/1 columns as in step 8. | No made-up number is filled in, so a blank is never mistaken for a real answer. |

Output: one clean feature table (one row per respondent) plus the two target columns.

---

## 4. Inputs

### 4.1 Which questions to use

Use every question left after cleaning. That is 173 input columns: 207 total, minus 3 identifiers, 11 protected, 11 that are 95%+ blank, 6 text and 2 targets (= 174), minus `final_eac_origin`. They are 164 questions, `final_osb`, `final_asb` and the 7 other `final_*` columns below. No hand-picking or extra grouping for the baseline.

- About 1,280 rows against 173 input columns, so strong regularisation is required.
- The 7 other `final_*` columns (`final_interaction_style`, `final_self_trust`, `final_primary`, `final_secondary`, `final_tertiary_1`, `final_tertiary_2`, `final_source_belief_modifier`) are kept. They are available when EAC runs at prediction time (confirmed). One check remains: train once with them and once without. Training uses person-reviewed values, but prediction gets upstream model outputs. If these columns add a lot, the baseline depends on upstream quality.
- After the baseline, read the top coefficients per model. A shorter question list is justified only by that evidence.

### 4.2 Covert and overt scores

The flow injects three scores right before `eac_protect_covert` (Protect step 2b, "Inject covert scores"). They are calculated from 18 yes/no questions already in the data, as in `eac.py`.

| Score | Calculation |
|---|---|
| `overt_score` | number of YES answers among the 9 overt questions (0 to 9) |
| `covert_score` | number of YES answers among the 9 covert questions (0 to 9) |
| `behavioral_bias_ratio` | `overt_score / (covert_score + 1)` |

Counting rules:

- An answer counts only if, trimmed and upper-cased, it equals `YES`.
- Blank answers are not counted (they add 0).
- A question missing from the row is treated as NO (adds 0).
- Match the question texts exactly after the header cleaning in section 3 (step 3). The typo "point it our" is part of the real column name.

Overt questions (9):

1. Would you describe yourself as direct?
2. Would you describe yourself as desiring for things to be fair or just?
3. Do you have a pattern of saying things impulsively that you would later regret and feel the desire to fix things or smooth them over?
4. Even if you feel like you're trying to help, do people blame you for being overly direct, controlling or quick to anger?
5. Are you quick to correct or step in verbally in a way that often creates conflict or hurt feelings with others?
6. Do you ask direct or overt questions?
7. When a boundary has been crossed do you find yourself quick to point it our or stand up for yourself?
8. Do you get upset if something or someone attempts to derail your idea, plan or goal?
9. Do you ideally prefer control over a situation but expect things to go wrong?

Covert questions (9):

1. Do you tend to be a people-pleaser or say what someone needs to hear rather than what is actually on your mind?
2. Do you tend to keep things to yourself because you don't want to upset others or rock the boat?
3. Is it challenging for you to voice your wants, needs or concerns?
4. Do you struggle with expressing boundaries or expectations?
5. Do you often people-please and not say anything until you can't take it anymore, and then say everything that you've been holding in?
6. Do you go through prolonged periods of keeping things in and staying quiet followed by needing to stand up for yourself or what feels fair?
7. Do you find it challenging to trust other people's judgment or ability to follow through on plans or a commitment?
8. Do you replay fights in your head or imagine what you should have said instead?
9. If you experience apathy and withdraw, do you tend to isolate and willingly self-sabotage?

Where the scores are used:

- `eac_protect_covert`: as 3 extra inputs (173 + 3 = 176 columns), as the flow shows. Calculated for the Protect anger-family people who reach this step.
- Escalating covert fallback (section 8.3): calculated for Escalating Anger-branch people only if the fallback rule is needed.

Average scores from the EDA (by label) show they separate Anger from Anxiety, but separate Covert Anger from plain Anger only weakly:

| Protect label | Overt | Covert |
|---|---|---|
| Anger | 6.3 | 5.5 |
| Covert Anger | 5.2 | 5.4 |
| Anxiety | 4.8 | 6.1 |

| Escalating label | Overt | Covert |
|---|---|---|
| Anger | 4.8 | 5.9 |
| Covert Anger | 4.6 | 6.1 |

---

## 5. Targets per model

Each model needs two things from the label columns: which rows it is trained on (eligible rows) and what counts as a yes (positive).

### 5.1 Turn each label into yes/no flags

Clean the label text first (trim, one spelling: `Covert Anger-Apathy`). Stop if any label is outside the 7 allowed values. Then look each label up in this fixed table. Use the table, not text parsing, so every value is explicit.

| Label | anger | covert | apathy | frustration |
|---|---|---|---|---|
| Anxiety | 0 | 0 | 0 | 0 |
| Anxiety-Apathy | 0 | 0 | 1 | 0 |
| Anger | 1 | 0 | 0 | 0 |
| Anger-Apathy | 1 | 0 | 1 | 0 |
| Covert Anger | 1 | 1 | 0 | 0 |
| Covert Anger-Apathy | 1 | 1 | 1 | 0 |
| Frustration | 0 | 0 | 0 | 1 |

Apply it to both columns. This gives `p_anger`, `p_covert`, `p_apathy` from `final_eac_protect`, and `e_covert`, `e_apathy`, `e_frustration` from `final_eac_esc`. These flags are targets only, never inputs.

### 5.2 Eligible rows and target per model

Eligibility follows the flow. The Escalating branch is chosen by the Protect anger result, so Escalating eligibility uses `p_anger`.

| Model | Eligible rows (filter) | Target | Positive labels | Eligible | Positives |
|---|---|---|---|---|---|
| `eac_protect_anger` | all rows | `p_anger` | Anger, Covert Anger, Anger-Apathy, Covert Anger-Apathy | 1280 | 578 |
| `eac_protect_covert` | `p_anger = 1` | `p_covert` | Covert Anger, Covert Anger-Apathy | 578 | 213 |
| `eac_protect_apathy` | all rows | `p_apathy` | Anxiety-Apathy, Anger-Apathy, Covert Anger-Apathy | 1280 | 109 |
| `eac_esc_frustration` | `p_anger = 0` (Escalating Anger branch) | `e_frustration` | Frustration | 702 | 57 |
| `eac_esc_covert` | `p_anger = 0` and `e_frustration = 0` | `e_covert` | Covert Anger, Covert Anger-Apathy | 645 | 34 |
| `eac_esc_apathy` | `e_frustration = 0` (both branches) | `e_apathy` | Anxiety-Apathy, Anger-Apathy, Covert Anger-Apathy | 1223 | 353 |

---

## 6. Train/test split: put 20% of people aside

- Pick 20% of the 1,280 people at random (about 256). The other 80% (about 1,024) are for training. Use a fixed random seed so the same people are picked every run.
- Do this once, on all 1,280 people, before the per-model filters in section 7. Each person is then "train" or "test" for all six models. That way, at the end, the test people can go through the whole flow and their predicted labels can be compared with their true labels.
- Keep the label mix the same in both parts. Example: Frustration is 4.5% of all people, so it should be about 4.5% of the test set too. A purely random pick could leave a rare label with almost no test rows. Use the Protect + Escalating label combination for this; combinations with fewer than 5 people are grouped together.

---

## 7. Build the per-model data sets

1. Clean the labels and add the 6 flag columns (5.1).
2. The train/test split is already made on all 1280 rows (section 6), so a respondent is in the same split for every model.
3. For each model, filter train and test by its eligibility rule and take its target flag (5.2). Keep these as one config per model, not hand-made files.
4. Check: eligible and positive counts over train + test must match the labels (for `data_6.xlsx`, the table in 5.2). Stop if they do not. This is really important.
5. At prediction time eligibility is not used. The predicted path through the flow decides which model runs.

Training vs testing the flow:

- Training: each downstream model is trained on rows chosen by the true labels.
- Testing: the full flow runs on predicted upstream decisions, including the Escalating branch chosen by the predicted Protect anger result.

---

## 8. Train and choose the models

Six binary models, three per chain, as in the flow. Each apathy model is one model shared across its branches.

| Model | Trained on | Positives / rows | Strength | Accuracy if it always says "no" |
|---|---|---|---|---|
| `eac_protect_anger` | all rows | 578 / 1280 (45%) | good | 54.8% |
| `eac_protect_covert` | Protect anger-family rows | 213 / 578 (37%) | good | 63.1% |
| `eac_protect_apathy` | all rows | 109 / 1280 (8.5%) | weak | 91.5% |
| `eac_esc_frustration` | Protect not-anger rows | 57 / 702 (8%) | weak | 91.9% |
| `eac_esc_covert` | Escalating Anger branch, not Frustration | 34 / 645 (5%) | very weak | 94.7% |
| `eac_esc_apathy` | all rows except Frustration | 353 / 1223 (29%) | good | 71.1% |

Accuracy is the main measure (section 10). The last column is the accuracy a model gets by always saying "no". A model is only useful if its accuracy is clearly above that number.

"Clearly" (used throughout this document) means at least 1 percentage point higher accuracy, measured on the cross-validation scores from section 8.1.

### 8.1 Choose settings using the training 80% only

Each model needs some settings chosen (how strongly to regularise, and its threshold). The test set must not be used for this. One extra hold-out slice would be too small for the weak models, so instead:

- Cut the training 80% into 5 equal parts.
- Train on 4 parts and score the 5th. Repeat 5 times, so every training person gets one score from a model that did not see them.
- Use those scores to compare settings by accuracy, and pick each model's threshold as the one with the highest accuracy.
- Then train the final model on the whole 80%.

### 8.2 Algorithms, in order

1. Logistic regression with L2 regularisation and balanced class weights, for all 6 models.
2. Gradient boosting (XGBoost or LightGBM) for the 3 "good" models only. Keep it only if it clearly beats logistic regression on accuracy (at least 1 percentage point, cross-validation scores from 8.1).
3. Check every model's accuracy against two simple baselines:
   - always predict the majority class (the last column above);
   - a single simple rule. For the covert models: covert if `covert_score` (section 4.2) is at or above a cut-off, with the cut-off chosen on the training 80% by accuracy.
   If a weak model cannot beat these on accuracy, use the rule as the fallback. Where there is no rule (apathy, frustration) or the rule does not clearly beat "always no" either, the fallback is "always no", and the script reports it.

### 8.3 Fallbacks (decided)

- `eac_esc_covert`: Escalating `Covert Anger` (31 rows) and `Covert Anger-Apathy` (3 rows) are too rare for a reliable model. If it does not beat the covert-score rule on accuracy, the rule replaces it; if the rule does not clearly beat "always no" either, use "always no". In the first run on `data_6.xlsx` neither beat "always no", so "always no" is used (accepted). The apathy step after it is unchanged.
- `eac_esc_frustration`: always use the trained model (logistic regression), even if it does not beat "always no" by 1 point. The script still reports when it does not. In the first run it did not (cross-validation 92.2% vs 92.0%).

### 8.4 Things the flow and data already tell us

- `eac_protect_anger` is the most important model. Its prediction also chooses the Escalating branch, so its errors spread to both chains.
- Protect anger-family rows always have an Escalating Anxiety or Anxiety-Apathy label.
- Apathy almost never appears in both chains (1 row).
- Escalating `Covert Anger-Apathy` has 3 rows. Expect the baseline to almost never predict it.

---

## 9. Test once

- Run the final models and the full flow on the test 20% once, at the end. Do not go back and change settings after looking at the test results.
- Weak models have very few test positives (about 7 for `eac_esc_covert`, 11 for `eac_esc_frustration`). For those, the section 8.1 numbers are more reliable than the test numbers.
- Whole-flow dumb comparison: compare the flow's final labels with always predicting the most common label. That is `Anxiety` for Protect (49.0% of people) and `Anger` for Escalating (34.3%). The flow is only useful if its accuracy is clearly above these (at least 1 percentage point). The whole flow is only run on the test set, so here the test accuracy is used.

---

## 10. What to report

Written to the Excel report (section 2.2).

Main measure: accuracy. It is used to choose settings and thresholds (section 8.1), to compare models (section 8.2), and as the headline result.

- Per model: accuracy at the chosen threshold, next to the "always no" accuracy from section 8.
- Whole flow: accuracy of the final Protect label and of the final Escalating label, next to the most-common-label accuracy from section 9, plus predicted vs true label tables (6x6 for Protect, 7x7 for Escalating).
- Fairness check: accuracy by gender and age group, using the protected columns kept aside.

Secondary checks, reported but not used to choose: PR-AUC, precision and recall on the positive class per model, and macro-F1 and recall per label for the whole flow.

---

## 11. Save the trained models

In `outputs/baseline/models/`. For each of the 6 models (or its fallback rule), save:

- the final trained model;
- its threshold (or the rule's cut-off);
- the exact list of input columns, in order.

This makes every prediction reproducible later and is what the pipeline will need.

---

## 12. Order of work (summary)

1. Column roles and cleaning (sections 2.1 and 3). Saves the clean feature table.
2. Inputs and covert/overt scores (section 4).
3. Label flags and one config per model: eligible rows and target (section 5). All 6 targets come from `final_eac_protect` and `final_eac_esc`.
4. Train/test split (section 6), then the per-model data sets and count check (section 7).
5. Logistic regression for all 6 models with cross-validation, then thresholds (section 8).
6. Gradient boosting for the 3 good models, baselines and fallback (section 8).
7. Full flow on the test set, plus the report (sections 9, 10 and 2.2).
8. Save the models (section 11).

---

## 13. Decisions (former open points)

- [x] Were the `final_*` labels reviewed by a person? Yes, they are person-reviewed ground truth.
- [x] Are the 7 other `final_*` columns available when EAC runs at prediction time? Yes.
- [x] One shared `eac_protect_apathy` model, or one per branch? One shared model for now.
- [x] Fallback for Escalating `Covert Anger-Apathy` (3 rows) and Escalating `Covert Anger` (31 rows)? Covert-score rule, then "always no" (section 8.3). `eac_esc_covert` ended on "always no" and that is accepted.
- [x] `eac_esc_frustration`: always use the trained model, even when it does not clearly beat "always no" (section 8.3).
- [x] Fairness age groups: under 18 (only if present), 18-29, 30-44, 45-59, 60+. Gender as given.
- [x] Main results use all 173 input columns including the 7 other `final_*` columns. The "without them" accuracy is a side check in the Models sheet. Only `final_eac_*` columns are banned as inputs; the flag columns made for the targets are never inputs.
- [x] Fixed settings: C in {0.01, 0.1, 1, 10}, seed 42, XGBoost depth 3 / 200 trees / learning rate 0.1 with balanced class weights.
- [x] Update the drawio label to `Covert Anger-Apathy`. Done on both pages.
- [x] Covert scores on the Escalating side: only for Escalating Anger-branch people, only if the fallback rule is needed.
- [x] Blanks in number columns: no made-up number; number columns become bands with "not asked" as its own band (section 3, step 9).
