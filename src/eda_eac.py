"""EDA for the EAC baseline: column types, blanks, category values, label balance.

Usage: python src/eda_eac.py [--file PATH_TO_XLSX] [--sheet 0] [--out outputs/eda] [--eac-py PATH_TO_eac.py]
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ID_COLS = ["diagnostic_response_id", "Link", "Campaign"]
TARGETS = ["final_eac_protect", "final_eac_esc"]

# Protected attributes are removed from model inputs (names matched after whitespace normalising)
PROTECTED = [
    "What is your age?",
    "What is your gender?",
    "How would you describe your sexual orientation?",
    "Was there any religion in your home growing up?",
    "How would you categorize the religion?",
    "What is the gender of your oldest child?",
    "What is their gender?",
    "Placeholder - Gender of child 3",
    "What is the age of your oldest child?",
    "How old are they?",
    "Placeholder - Age of child 3",
]
# Label sets from the agreed flow (Covert Anger-Apathy spelling for both chains)
PROTECT_LABELS = [
    "Anxiety", "Anxiety-Apathy", "Anger", "Anger-Apathy",
    "Covert Anger", "Covert Anger-Apathy",
]  # fmt: skip
ESC_LABELS = PROTECT_LABELS + ["Frustration"]
ANGER = {"Anger", "Anger-Apathy", "Covert Anger", "Covert Anger-Apathy"}
COVERT = {"Covert Anger", "Covert Anger-Apathy"}
ANGER_BRANCH = ANGER | {"Frustration"}  # Escalating labels that come from the Anger branch

# final_osb / final_asb are injected as model inputs by the repo (ml/services/eac.py)
PIPELINE_INPUT_LABELS = {"final_osb", "final_asb"}

BLANK_BANDS = [0, 5, 20, 50, 80, 95, 100.01]
BLANK_BAND_NAMES = ["0-5%", "5-20%", "20-50%", "50-80%", "80-95%", "95-100%"]

_lines: list[str] = []


def out(line: str = "") -> None:
    """Print a line and keep it for the saved report."""
    print(line)
    _lines.append(line)


def norm_col(name: str) -> str:
    """Collapse whitespace in a column name."""
    return re.sub(r"\s+", " ", str(name)).strip()


def clean_cell(v):
    """Strip strings; turn empty or NaN into NA."""
    if pd.isna(v):
        return pd.NA
    if isinstance(v, str):
        v = v.strip()
        return v if v else pd.NA
    return v


def norm_label(v, allowed: dict[str, str]):
    """Map a raw label to its canonical spelling, or return the cleaned text if unknown."""
    if pd.isna(v):
        return pd.NA
    s = re.sub(r"\s*[-–—]\s*", "-", re.sub(r"\s+", " ", str(v))).strip()
    return allowed.get(s.lower(), s)


def classify(s: pd.Series) -> str:
    """Guess a column type from its non-blank values."""
    vals = s.dropna().astype(str)
    n = len(vals)
    if n == 0:
        return "all_blank"
    low = set(vals.str.lower().unique())
    if low <= {"yes", "no"}:
        return "yes_no"
    if pd.to_numeric(vals, errors="coerce").notna().mean() >= 0.95:
        return "numeric"
    if vals.str.contains(r" \| ", regex=True).mean() >= 0.02:
        return "multiselect"
    nunique = vals.nunique()
    if vals.str.len().median() >= 40 and nunique / n > 0.5:
        return "free_text"
    if nunique <= 30:
        return "category"
    return "short_text"


def group_of(col: str, groups: dict[str, set[str]]) -> str:
    """Return the review group a column belongs to."""
    if col in ID_COLS:
        return "identifier"
    if col in TARGETS:
        return "target"
    if col.startswith("final_"):
        return "other_label"
    for name, members in groups.items():
        if col in members:
            return name
    return "feature"


def suggest(group: str, ctype: str, pct_blank: float, n_unique: int, col: str = "") -> str:
    """Baseline default action: only identifiers and protected attributes are removed up front."""
    if group == "identifier":
        return "drop: identifier"
    if group == "target":
        return "target"
    if group == "protected":
        return "drop: protected attribute"
    if group == "other_label":
        if col in PIPELINE_INPUT_LABELS:
            return "keep (pipeline input)"
        return "keep (confirm available at prediction time)"
    if ctype == "all_blank" or pct_blank >= 95:
        return "drop: >=95% blank"
    if n_unique <= 1:
        return "drop: single value"
    if ctype in ("free_text", "short_text"):
        return "skip in baseline (text)"
    if pct_blank >= 50:
        return "keep (blank = 'not asked')"
    return "keep"


def option_counts(s: pd.Series, ctype: str) -> pd.Series:
    """Count each value; multiselect cells are split into options."""
    vals = s.dropna().astype(str)
    if ctype == "multiselect":
        vals = vals.str.split(r"\s*\|\s*").explode().str.strip()
        vals = vals[vals != ""]
    return vals.value_counts()


def node_table(p: pd.Series, e: pd.Series) -> pd.DataFrame:
    """Rows eligible and positives for each cascade model."""
    nodes = [
        ("eac_protect_anger", p, p.notna(), p.isin(ANGER)),
        ("eac_protect_covert", p, p.isin(ANGER), p.isin(COVERT)),
        ("eac_protect_apathy", p, p.notna(), p.str.endswith("Apathy", na=False)),
        ("eac_esc_frustration", e, e.isin(ANGER_BRANCH), e.eq("Frustration")),
        ("eac_esc_covert", e, e.isin(ANGER_BRANCH - {"Frustration"}), e.isin(COVERT)),
        ("eac_esc_apathy", e, e.notna() & e.ne("Frustration"), e.str.endswith("Apathy", na=False)),
    ]  # fmt: skip
    rows = []
    for name, _, elig, pos in nodes:
        elig = elig.fillna(False).astype(bool)
        pos = (pos.fillna(False) & elig).astype(bool)
        n, k = int(elig.sum()), int(pos.sum())
        rows.append({
            "model": name, "eligible_rows": n, "positives": k,
            "negatives": n - k, "pct_positive": round(100 * k / n, 1) if n else 0.0,
        })  # fmt: skip
    return pd.DataFrame(rows)


def covert_score_table(df: pd.DataFrame, p: pd.Series, eac_py: Path | None, labels: list[str] = PROTECT_LABELS):
    """Recompute the pipeline's overt/covert scores (from eac.py) and summarise by label."""
    if eac_py is None:
        return None, "Skipped: no --eac-py path given", None
    if not eac_py.exists():
        return None, f"Skipped: {eac_py} not found", None
    import importlib.util  # noqa: PLC0415

    spec = importlib.util.spec_from_file_location("eac_pipeline", eac_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cols = {norm_col(c): c for c in df.columns}
    overt_q = [norm_col(q) for q in mod.OVERT_QUESTIONS]
    covert_q = [norm_col(q) for q in mod.COVERT_QUESTIONS]
    absent = [q for q in overt_q + covert_q if q not in cols]

    def yes_count(qs):
        present = [q for q in qs if q in cols]
        return sum(df[q].astype(str).str.upper().eq("YES") for q in present)

    overt, covert = yes_count(overt_q), yes_count(covert_q)
    work = pd.DataFrame({"label": p, "overt_score": overt, "covert_score": covert})
    work["behavioral_bias_ratio"] = work["overt_score"] / (work["covert_score"] + 1)
    g = work.groupby("label").agg(
        rows=("label", "size"),
        mean_overt_score=("overt_score", "mean"),
        mean_covert_score=("covert_score", "mean"),
        mean_bias_ratio=("behavioral_bias_ratio", "mean"),
    ).round(2).reset_index()  # fmt: skip
    order = {lab: i for i, lab in enumerate(labels)}
    g = g.sort_values("label", key=lambda s: s.map(order)).reset_index(drop=True)
    note = f"Each score counts YES answers among 9 overt / 9 covert questions; ratio = overt / (covert + 1). Questions missing from data: {absent or 'none'}"
    return g, note, work


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(ROOT / "data" / "data_6.xlsx"))
    ap.add_argument("--sheet", default="0", help="sheet name or index")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "eda"))
    ap.add_argument("--eac-py", default=None, help="path to the pipeline's ml/services/eac.py (enables the covert score check)")
    ap.add_argument("--keep-branch-breaks", action="store_true", help="do not exclude the rows that break the branch rule")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    outdir = Path(args.out)
    eac_py = Path(args.eac_py) if args.eac_py else None
    outdir.mkdir(parents=True, exist_ok=True)

    sheet = int(args.sheet) if args.sheet.isdigit() else args.sheet
    raw = pd.read_excel(args.file, sheet_name=sheet, dtype=object)
    raw.columns = [norm_col(c) for c in raw.columns]
    df = raw.map(clean_cell)
    n_src_rows = len(df)

    # ---- exclude rows where the Protect/Escalating branch rule breaks (source file is not changed)
    p0 = df["final_eac_protect"].map(lambda v: norm_label(v, {x.lower(): x for x in PROTECT_LABELS}))
    e0 = df["final_eac_esc"].map(lambda v: norm_label(v, {x.lower(): x for x in ESC_LABELS}))
    broke = p0.notna() & e0.notna() & (p0.isin(ANGER) == e0.isin(ANGER_BRANCH))
    breaks_df = pd.DataFrame({
        "excel_row": df.index[broke] + 2,  # +1 for the header row, +1 for 1-based rows
        "protect_label": p0[broke].values,
        "escalating_label": e0[broke].values,
        "what_the_rule_expected": [
            "Escalating Anxiety / Anxiety-Apathy" if pr in ANGER else "an Anger-branch label (Anger, Covert, Frustration)"
            for pr in p0[broke]
        ],
    })  # fmt: skip
    if not args.keep_branch_breaks:
        df = df.loc[~broke]
    n_rows, n_cols = df.shape

    groups = {
        "protected": {norm_col(c) for c in PROTECTED},
    }
    out(f"File: {Path(args.file).name} | rows in file: {n_src_rows} | rows analysed: {n_rows} | columns: {n_cols}")
    if not args.keep_branch_breaks:
        out(f"Excluded {len(breaks_df)} rows where the branch rule breaks (kept in the source file):")
        out(breaks_df.to_string(index=False))
    missing = {g: sorted(m - set(df.columns)) for g, m in groups.items()}
    for g, names in missing.items():
        if names:
            out(f"NOTE: {g} names not found in data (fix the list): {names}")

    # ---- per-column summary
    recs = []
    for i, col in enumerate(df.columns, start=1):
        s = df[col]
        ctype = classify(s)
        n_blank = int(s.isna().sum())
        pct_blank = round(100 * n_blank / n_rows, 1)
        n_unique = int(s.dropna().astype(str).nunique())
        grp = group_of(col, groups)
        recs.append({
            "no": i, "column": col, "group": grp, "type": ctype,
            "n_answered": n_rows - n_blank, "n_blank": n_blank,
            "pct_blank": pct_blank, "pct_answered": round(100 - pct_blank, 1),
            "n_unique": n_unique, "suggestion": suggest(grp, ctype, pct_blank, n_unique, col),
        })  # fmt: skip
    summ = pd.DataFrame(recs)
    summ.to_csv(outdir / "column_summary.csv", index=False, encoding="utf-8-sig")

    feats = summ[summ["group"].isin(["feature", "protected"])]

    role_df = summ["group"].value_counts().rename_axis("role").reset_index(name="columns")
    type_df = feats["type"].value_counts().rename_axis("type").reset_index(name="columns")
    bands = pd.cut(feats["pct_blank"], BLANK_BANDS, labels=BLANK_BAND_NAMES, right=False)
    band_df = bands.value_counts().reindex(BLANK_BAND_NAMES).rename_axis("pct_blank").reset_index(name="columns")
    sugg_df = summ["suggestion"].value_counts().rename_axis("baseline_action").reset_index(name="columns")

    out("\n=== 1. Columns by role ===")
    out(role_df.to_string(index=False))
    out("\n=== 2. Question columns by data type ===")
    out(type_df.to_string(index=False))
    out("\n=== 3. Question columns by % blank ===")
    out(band_df.to_string(index=False))
    out("(blank = nobody answered; many are skip-logic follow-ups, so blank often means 'not asked')")
    out("\n=== 4. Baseline action counts (only identifiers and protected attributes are removed up front) ===")
    out(sugg_df.to_string(index=False))

    # ---- numbered category values
    lines, long_rows = [], []
    listable = summ[summ["type"].isin(["category", "yes_no", "multiselect"])]
    for _, r in listable.iterrows():
        counts = option_counts(df[r["column"]], r["type"])
        total = int(counts.sum())
        lines.append(f"[{r['no']}] {r['column']}  ({r['type']}, {len(counts)} values, {r['pct_blank']}% blank)")
        for j, (val, c) in enumerate(counts.items(), start=1):
            pct = round(100 * c / total, 1) if total else 0.0
            lines.append(f"    {j}. {val}  (n={c}, {pct}%)")
            long_rows.append({"column_no": r["no"], "column": r["column"], "type": r["type"],
                              "value_no": j, "value": val, "count": int(c), "pct": pct})  # fmt: skip
        lines.append("")
    (outdir / "category_values.txt").write_text("\n".join(lines), encoding="utf-8")
    pd.DataFrame(long_rows).to_csv(outdir / "category_values.csv", index=False, encoding="utf-8-sig")
    out(f"\n=== 5. Category values: {len(listable)} columns listed, see category_values.txt / .csv ===")

    # ---- labels
    out("\n=== 6. Target labels ===")
    allowed_p = {x.lower(): x for x in PROTECT_LABELS}
    allowed_e = {x.lower(): x for x in ESC_LABELS}
    p = df["final_eac_protect"].map(lambda v: norm_label(v, allowed_p))
    e = df["final_eac_esc"].map(lambda v: norm_label(v, allowed_e))
    label_tables = {}
    for name, ser, allowed in [("final_eac_protect", p, PROTECT_LABELS), ("final_eac_esc", e, ESC_LABELS)]:
        out(f"\n{name}: {int(ser.isna().sum())} blank")
        vc = ser.value_counts()
        tbl = pd.DataFrame({
            "label": vc.index, "rows": vc.values,
            "pct": (100 * vc.values / n_rows).round(1),
            "in_allowed_set": ["yes" if lab in allowed else "NO" for lab in vc.index],
        })  # fmt: skip
        label_tables[name] = tbl
        out(tbl.to_string(index=False))
        raw_vals = df[name].dropna().astype(str).nunique()
        out(f"  raw distinct spellings: {raw_vals} -> after cleaning: {vc.shape[0]}")

    ctab = pd.crosstab(p, e)
    out("\nProtect x Escalating (rows = Protect, cols = Escalating):")
    out(ctab.to_string())
    both = p.notna() & e.notna()
    consistent = (p[both].isin(ANGER) == ~e[both].isin(ANGER_BRANCH)).mean() if both.any() else float("nan")
    out(f"\nBranch rule (Protect anger <-> Escalating Anxiety branch) holds for {100 * consistent:.1f}% of {int(both.sum())} rows")

    nodes_df = node_table(p, e)
    out("\n=== 7. Training rows per cascade model ===")
    out(nodes_df.to_string(index=False))

    # ---- covert scores as the pipeline computes them (eac.py)
    scores_df, scores_note, work_p = covert_score_table(df, p, eac_py)
    scores_esc_df, _, work_e = covert_score_table(df, e, eac_py, ESC_LABELS)
    out("\n=== 7b. Covert scores (overt_score, covert_score, behavioral_bias_ratio) by Protect label ===")
    out(scores_note)
    if scores_df is not None:
        out(scores_df.to_string(index=False))
        out("\nSame scores by Escalating label:")
        out(scores_esc_df.to_string(index=False))

    # ---- row quality
    out("\n=== 8. Row quality ===")
    ids = df["diagnostic_response_id"]
    out(f"Duplicate respondent ids: {int(ids.dropna().duplicated().sum())} | blank ids: {int(ids.isna().sum())}")
    dup_cols = [c for c in raw.columns if re.search(r"\.\d+$", c)]
    out(f"Repeated question headers (pandas suffix .N): {dup_cols}")
    q_cols = feats["column"].tolist()
    answered = df[q_cols].notna().sum(axis=1)
    out(f"Answered questions per respondent: min {answered.min()}, median {answered.median():.0f}, max {answered.max()} (of {len(q_cols)})")
    yn_core = feats[(feats["type"] == "yes_no") & (feats["pct_blank"] < 20)]["column"].tolist()
    if yn_core:
        yes_share = df[yn_core].apply(lambda c: c.astype(str).str.lower().eq("yes")).sum(axis=1) / df[yn_core].notna().sum(axis=1).replace(0, pd.NA)
        same = ((yes_share >= 0.95) | (yes_share <= 0.05)).sum()
        out(f"Core YES/NO block: {len(yn_core)} questions; respondents answering >=95% the same way: {int(same)}")

    quality = {
        "Respondents analysed": n_rows,
        "Rows in source file": n_src_rows,
        "Rows excluded (branch rule breaks)": 0 if args.keep_branch_breaks else len(breaks_df),
        "Columns in file": n_cols,
        "Duplicate respondent ids": int(ids.dropna().duplicated().sum()),
        "Blank respondent ids": int(ids.isna().sum()),
        "Repeated question headers": ", ".join(dup_cols) or "none",
        "Questions answered per respondent (min)": int(answered.min()),
        "Questions answered per respondent (median)": int(answered.median()),
        "Questions answered per respondent (max)": int(answered.max()),
        "Question columns counted": len(q_cols),
        "Respondents answering >=95% of core YES/NO the same way": int(same) if yn_core else "n/a",
        "Branch rule holds (Protect anger <-> Escalating Anxiety)": f"{100 * consistent:.1f}%",
    }

    (outdir / "eda_report.txt").write_text("\n".join(_lines), encoding="utf-8")
    from eda_excel import build_workbook  # noqa: PLC0415

    xlsx = outdir / "eda_report.xlsx"
    build_workbook(xlsx, Path(args.file).name, summ, role_df, type_df, band_df, sugg_df,
                   pd.DataFrame(long_rows), label_tables, ctab, nodes_df, scores_df, scores_esc_df,
                   scores_note, quality, breaks_df, work_p, work_e)  # fmt: skip
    out(f"\nSaved to {outdir}: eda_report.xlsx, eda_report.txt, column_summary.csv, category_values.txt, category_values.csv")


if __name__ == "__main__":
    main()
