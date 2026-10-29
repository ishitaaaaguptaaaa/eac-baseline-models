"""Make a cleaned copy of the questionnaire data workbook.

A row is removed when any of these is true (checked in this order; each row is
counted once, at the first check that removes it):

1. any timeline (TID) answer is blank
2. the respondent id contains the word "test"
3. the respondent id is on the exclusion list (``config/excluded_ids.txt``)

The output workbook has two sheets: the kept rows (same sheet name as the
input) and ``Removed_rows`` with a ``removal_reason`` column.

Usage:
    python src/clean_data.py --data data/data_5.xlsx --questions data/Questions.xlsx \
        [--out data/data_6.xlsx] [--exclude config/excluded_ids.txt]
"""

import argparse
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# ---- sheet names ------------------------------------------------------------
QUESTIONS_SHEET = "Timeline QID"
DATA_SHEET = "Ayush-Records"
REMOVED_SHEET = "Removed_rows"

# ---- column names in the questions workbook ---------------------------------
QID_COL = "ordering"
QUESTION_TEXT_COL = "question"

# ---- column name in the data file that uniquely identifies a row -----------
DIAG_ID_COL = "diagnostic_response_id"

# ---- text that marks a test ID (matched case-insensitively) ----------------
TEST_WORD = "test"


# ============================================================
# Question text -> data column mapping (same logic as the classifier)
# ============================================================

def _normalize_text(s):
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def build_question_mapping(questions_path, data_columns):
    """Returns (qid_to_col dict, unmatched list of (qid, question_text))."""
    q_df = pd.read_excel(questions_path, sheet_name=QUESTIONS_SHEET)
    norm_to_actual = {_normalize_text(c): c for c in data_columns}

    qid_to_col, unmatched = {}, []
    for _, row in q_df.iterrows():
        qid_raw = row.get(QID_COL)
        question_text = row.get(QUESTION_TEXT_COL)
        if pd.isna(qid_raw):
            continue
        try:
            qid = int(qid_raw)
        except (ValueError, TypeError):
            continue

        norm_q = _normalize_text(question_text)
        actual_col = norm_to_actual.get(norm_q)
        if actual_col is None:
            # loose "contains" fallback both directions, only if unique
            candidates = [
                actual for norm, actual in norm_to_actual.items()
                if norm_q and (norm_q in norm or norm in norm_q)
            ]
            if len(candidates) == 1:
                actual_col = candidates[0]

        if actual_col is None:
            unmatched.append((qid, question_text))
        else:
            qid_to_col[qid] = actual_col
    return qid_to_col, unmatched


# ============================================================
# Helpers
# ============================================================

def load_excluded_ids(path):
    """Read one id per line; blank lines and lines starting with # are ignored.

    Ids are compared against the whole id, ignoring case and surrounding spaces.
    """
    path = Path(path)
    if not path.exists():
        print(f"No exclusion list at '{path}' -- the remove-list step is skipped.")
        return set()
    lines = path.read_text(encoding="utf-8").splitlines()
    return {ln.strip().casefold() for ln in lines if ln.strip() and not ln.strip().startswith("#")}


def is_blank(series):
    """True where the cell is empty/NaN or only whitespace (one column)."""
    return series.isna() | series.astype(str).str.strip().eq("")


def any_blank(frame):
    """Per row: True if ANY of the given columns is blank."""
    return frame.apply(is_blank).any(axis=1)


def id_key(series):
    """Trimmed, case-folded ID text -- the key used to match the exclusion list."""
    return series.fillna("").astype(str).str.strip().str.casefold()


def find_column(df, wanted):
    """Exact match first, then case/space-insensitive match."""
    if wanted in df.columns:
        return wanted
    norm = _normalize_text(wanted)
    hits = [c for c in df.columns if _normalize_text(c) == norm]
    if len(hits) == 1:
        return hits[0]
    raise ValueError(
        f"Column '{wanted}' not found in sheet '{DATA_SHEET}'. "
        f"Available columns: {list(df.columns)}"
    )


# ============================================================
# MAIN
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True, help="input data workbook (.xlsx)")
    ap.add_argument("--questions", required=True, help="questions workbook with the 'Timeline QID' sheet")
    ap.add_argument("--out", default=str(ROOT / "data" / "data_6.xlsx"), help="cleaned workbook to write")
    ap.add_argument("--sheet", default=DATA_SHEET, help="sheet in the data workbook")
    ap.add_argument("--exclude", default=str(ROOT / "config" / "excluded_ids.txt"),
                    help="text file of respondent ids to remove, one per line")
    return ap.parse_args()


def main():
    args = parse_args()
    excluded = load_excluded_ids(args.exclude)

    df = pd.read_excel(args.data, sheet_name=args.sheet)
    start_rows = len(df)
    print(f"Loaded {start_rows} rows from '{args.data}' / '{args.sheet}'.")

    id_col = find_column(df, DIAG_ID_COL)

    qid_to_col, unmatched = build_question_mapping(args.questions, df.columns)
    tid_cols = sorted(set(qid_to_col.values()), key=list(df.columns).index)
    print(f"Matched {len(qid_to_col)} TID question(s) to {len(tid_cols)} data column(s).")
    if unmatched:
        print(f"!! {len(unmatched)} TID question(s) could NOT be matched -- "
              f"they are NOT checked for blanks:")
        for qid, qtext in unmatched:
            print(f"   TID {qid}: {qtext!r}")

    # ---- Step 1: any blank TID answer --------------------------------------
    blank_tid = any_blank(df[tid_cols]) if tid_cols else pd.Series(False, index=df.index)
    # ---- Step 2: "test" anywhere in the ID (case-insensitive) --------------
    test_id = df[id_col].fillna("").astype(str).str.contains(TEST_WORD, case=False, regex=False)
    # ---- Step 3: ID is on the exclusion list (whole-ID match) ---------------
    id_keys = id_key(df[id_col])
    listed_id = id_keys.isin(excluded)

    # Each row is attributed to the FIRST step that removes it, so the step
    # counts add up exactly to the total removed.
    reason = pd.Series("", index=df.index, dtype=object)
    reason[blank_tid] = "blank_tid_answer"
    reason[(reason == "") & test_id] = "test_id"
    reason[(reason == "") & listed_id] = "removed_id"

    kept = df[reason == ""].copy()
    removed = df[reason != ""].copy()
    removed.insert(0, "removal_reason", reason[reason != ""])

    n1 = int((reason == "blank_tid_answer").sum())
    n2 = int((reason == "test_id").sum())
    n3 = int((reason == "removed_id").sum())

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.out, engine="openpyxl") as xw:
        kept.to_excel(xw, sheet_name=args.sheet, index=False)
        removed.to_excel(xw, sheet_name=REMOVED_SHEET, index=False)

    print("\n---- Rows removed (each row counted once, at the first step that removes it) ----")
    print(f"  Step 1  blank TID answer      : {n1}")
    print(f"  Step 2  '{TEST_WORD}' in {id_col} : {n2}")
    print(f"  Step 3  ID on exclusion list  : {n3}")
    print(f"  Total removed                 : {n1 + n2 + n3}")
    print("\n---- Overlap info (rows matching each check in the ORIGINAL data) ----")
    print(f"  blank TID answer : {int(blank_tid.sum())}")
    print(f"  test ID          : {int(test_id.sum())}")
    print(f"  ID on exclusion list: {int(listed_id.sum())}")
    missing = len(excluded - set(id_keys))
    if missing:
        print(f"\n{missing} id(s) on the exclusion list were not found in the data.")
    print(f"\nFinal rows: {len(kept)}  (started with {start_rows})")
    print(f"Wrote '{args.out}' -> sheets '{args.sheet}' (kept) and '{REMOVED_SHEET}' (removed).")


if __name__ == "__main__":
    main()
