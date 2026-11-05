#!/usr/bin/env python3
"""Prepare the retained 2025 datasets for the corrected Figure 2 rerun.

The five retained files are the *un-composited* 2025 snapshots, i.e. they still
carry the 50 raw modelling columns (47 manuscript features + Weight/MLP/Mo) and
their pre-cleaning row counts (1,356 / 2,468 / 9,829 / 6,070 / 587).  The
``composited_*`` variants were merged with composite columns and lost rows; we
therefore start from the raw snapshots and derive the ten composites ourselves.

The SMOTE pool is trimmed to 2,368 records after this stage, discarding 100
generated IVIG-resistant records (seed 20261104) so that the prespecified
1,184 : 1,184 design is restored; empirical records are never dropped.

Corrections applied to every dataset
------------------------------------
1. Unit / decimal-point repair where the intended value is unambiguous
   (creatinine mg/dL -> umol/L, albumin g/dL -> g/L, calcium mg/dL -> mmol/L,
   haematocrit fraction/percent, chloride/LDH/RDW/PDW/P-LCR decimal shifts,
   plateletcrit 10x/100x, magnesium 10x, fever days 10x).
2. Values still outside the physiological screen (the same deliberately wide
   ranges used in Part 2) become missing, including albumin = 0.
3. Binary symptoms are rounded to {0, 1}; day variables to whole days.
4. Missing cells are filled with the median of the cleaned Original training
   partition (no validation information is used).

Synthetic quality filter
-----------------------
The Synthetic and SPA pools are additionally filtered: a record is dropped if
any of the 47 base features lies outside the range observed in the cleaned
Original training partition.  The filter is applied *after* unit repair and
*before* imputation, so an impossible generated value cannot be rescued by
imputation.  SMOTE is not range-filtered (its points are interpolations of
observed patients).

Outputs (data/retained/)
------------------------
{ori,smote,syn,spa,val}_panel_{a,b}.csv
corrections.csv, filter_log.csv, manifest.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "legacy" / "data_snapshots"
OUT = ROOT / "data" / "retained"
RES = ROOT / "results" / "retained"

MANIFEST = json.loads((ROOT / "data" / "processed" / "part2_manifest.json").read_text())
PANEL_A = MANIFEST["panel_a"]
COMPOSITES = MANIFEST["composites"]
PANEL_B = MANIFEST["panel_b"]
BINARY = MANIFEST["binary"]
DAYS = MANIFEST["day_vars"]
LABEL = "Label"

RAW_FILES = {
    "Ori": SNAP / "ori_train_data_20250305.csv",
    "SMOTE": SNAP / "train_smoted_20250305.csv",
    "Syn": SNAP / "tabula_middle_padding_synthetic_data_20250305.csv",
    "SPA": SNAP / "sds_data_20250305.csv",
    "Val": SNAP / "val_data_20250305.csv",
}

# Deliberately wide plausibility ranges: flag the impossible, not the extreme.
RANGES = {
    "WBC": (0.5, 100), "ANC": (0.01, 80), "Lym": (0.01, 60), "RBC": (0.5, 8.5),
    "HGB": (30, 260), "HCT": (10, 75), "PLT": (5, 5000), "RDW-CV": (8, 35),
    "RDW-SD": (20, 90), "PDW": (5, 60), "MPV": (4, 25), "P-LCR": (5, 90),
    "PCT": (0.01, 1.5), "CRP": (0, 1000), "ESR": (0, 300), "ALT": (0, 8000),
    "AST": (0, 8000), "TB": (0.1, 600), "DBIL": (0, 500), "IDIL": (0, 500),
    "ALB": (5, 70), "GLB": (5, 80), "LDH": (20, 30000), "UN": (0.5, 400),
    "Cr": (5, 1500), "K": (1.0, 10), "Na": (100, 190), "Cl": (60, 150),
    "Ca": (0.5, 5.0), "P": (0.1, 6.0), "Mg": (0.2, 4.0), "r-GT": (0, 8000),
    "FeverDay": (0, 30), "IVIGDay": (0, 30), "LabTestDay": (0, 30),
    "Age": (0, 18),
}

# Zero is not a possible measurement for these variables.
ZERO_IMPOSSIBLE = {
    "WBC", "ANC", "Lym", "RBC", "HGB", "PLT", "HCT", "RDW-CV", "RDW-SD",
    "PDW", "MPV", "P-LCR", "PCT", "TB", "ALB", "GLB", "LDH", "UN", "Cr",
    "K", "Na", "Cl", "Ca", "P", "Mg",
}


def _log(changes: list, dataset: str, feature: str, index, old, new, action: str):
    changes.append({
        "dataset": dataset, "row_id": int(index), "feature": feature,
        "old": float(old), "new": np.nan if pd.isna(new) else float(new),
        "action": action,
    })


def _assign(df, dataset, feature, mask, new_values, action, changes):
    """Assign new_values where mask, logging old -> new per row."""
    old = df.loc[mask, feature]
    for idx, ov, nv in zip(old.index, old.values, np.asarray(new_values)):
        if not (pd.isna(ov) and pd.isna(nv)) and not (pd.notna(ov) and pd.notna(nv) and np.isclose(ov, nv)):
            _log(changes, dataset, feature, idx, ov, nv, action)
    df.loc[mask, feature] = new_values


def repair_units(df: pd.DataFrame, dataset: str, changes: list) -> pd.DataFrame:
    """Repair unambiguous unit / decimal-point errors.  Vectorised per rule."""
    df = df.copy()

    def apply(feature, mask, transform, action):
        if feature not in df or not mask.any():
            return
        new = transform(df.loc[mask, feature].astype(float))
        _assign(df, dataset, feature, mask, new.values, action, changes)

    s = df["Cr"].astype(float)
    apply("Cr", (s > 0) & (s < 5), lambda v: v * 88.4, "creatinine mg/dL -> umol/L")

    s = df["ALB"].astype(float)
    apply("ALB", (s > 0) & (s < 5), lambda v: v * 10.0, "albumin g/dL -> g/L")

    s = df["Ca"].astype(float)
    apply("Ca", (s > 5) & (s < 30), lambda v: v / 4.008, "calcium mg/dL -> mmol/L")

    s = df["HCT"].astype(float)
    apply("HCT", (s > 0) & (s < 1), lambda v: v * 100.0, "haematocrit fraction -> %")
    s = df["HCT"].astype(float)
    apply("HCT", (s >= 1) & (s < 10), lambda v: v * 10.0, "haematocrit decimal shift x10")

    s = df["Cl"].astype(float)
    apply("Cl", (s >= 5) & (s < 20), lambda v: v * 10.0, "chloride decimal shift x10")

    s = df["LDH"].astype(float)
    apply("LDH", (s > 0) & (s < 20), lambda v: v * 10.0, "LDH decimal shift x10")

    s = df["RDW-SD"].astype(float)
    apply("RDW-SD", (s > 0) & (s < 20), lambda v: v * 10.0, "RDW-SD decimal shift x10")

    s = df["RDW-CV"].astype(float)
    apply("RDW-CV", (s > 0) & (s < 8), lambda v: v * 10.0, "RDW-CV decimal shift x10")

    s = df["PDW"].astype(float)
    apply("PDW", (s > 0) & (s < 5), lambda v: v * 10.0, "PDW decimal shift x10")

    # Plateletcrit: 4.915 -> 0.4915, 46.1 -> 0.461
    s = df["PCT"].astype(float)
    mask = (s > 1.5) & (s < 100)
    new = s[mask].copy()
    for _ in range(3):
        still = new > 1.5
        if not still.any():
            break
        new[still] = new[still] / 10.0
    apply("PCT", mask, lambda v: new, "plateletcrit decimal shift /10")

    # P-LCR: 0.41 -> 41, 2.3 -> 23, 2.6 -> 26
    s = df["P-LCR"].astype(float)
    mask = (s > 0) & (s < 5)
    new = s[mask].copy()
    for _ in range(3):
        still = new < 5
        if not still.any():
            break
        new[still] = new[still] * 10.0
    apply("P-LCR", mask, lambda v: new, "P-LCR percentage decimal shift x10")

    # Haemoglobin 862 -> 86.2; 319/291 are ambiguous and stay missing.
    s = df["HGB"].astype(float)
    apply("HGB", (s > 500) & (s < 1000), lambda v: v / 10.0, "haemoglobin decimal shift /10")

    # Magnesium 8.69 -> 0.869
    s = df["Mg"].astype(float)
    apply("Mg", (s > 4) & (s < 10), lambda v: v / 10.0, "magnesium decimal shift /10")

    s = df["FeverDay"].astype(float)
    apply("FeverDay", (s > 30) & (s <= 100), lambda v: v / 10.0, "fever days decimal shift /10")

    return df


def screen_impossible(df: pd.DataFrame, dataset: str, changes: list) -> pd.DataFrame:
    """Set values outside the plausibility screen (and impossible zeros) to NaN."""
    df = df.copy()
    for feature, (low, high) in RANGES.items():
        if feature not in df:
            continue
        s = df[feature].astype(float)
        bad = s.notna() & ((s < low) | (s > high))
        if bad.any():
            _assign(df, dataset, feature, bad, np.full(int(bad.sum()), np.nan),
                    f"outside physiological range [{low}, {high}]", changes)
        if feature in ZERO_IMPOSSIBLE:
            zero = df[feature].astype(float) == 0
            if zero.any():
                _assign(df, dataset, feature, zero, np.full(int(zero.sum()), np.nan),
                        "impossible zero", changes)
    return df


def fix_categoricals_and_days(df: pd.DataFrame, dataset: str, changes: list,
                              round_days: bool = True) -> pd.DataFrame:
    df = df.copy()
    for feature in BINARY:
        if feature not in df:
            continue
        s = df[feature].astype(float)
        bad = s.notna() & ~s.isin([0, 1])
        if bad.any():
            new = s[bad].round().clip(0, 1)
            _assign(df, dataset, feature, bad, new.values, "binary rounded to {0,1}", changes)
        df[feature] = df[feature].round().astype("Int64")
    for feature in DAYS:
        if feature not in df:
            continue
        s = df[feature].astype(float)
        bad = s.notna() & (s < 0)
        if bad.any():
            _assign(df, dataset, feature, bad, np.full(int(bad.sum()), np.nan),
                    "negative day", changes)
        if round_days:
            df[feature] = df[feature].round().astype("Int64")
        else:
            # Perturbed / interpolated day variables are continuous; rounding them
            # destroys information (see devlog 10.2).
            df[feature] = df[feature].astype(float)
    return df


def derive_composites(df: pd.DataFrame) -> pd.DataFrame:
    """Documented composite formulas (Table S1), on the corrected base variables."""
    V = pd.DataFrame(index=df.index)
    V["ALB-dNLR"] = expit(df["ALB"] * (df["WBC"] - df["ANC"]) / df["ANC"] / 10.0)
    V["ALBI"] = expit(0.66 * np.log10(df["TB"]) - 0.085 * df["ALB"])
    V["CALLY"] = df["ALB"] * df["Lym"]
    V["RPR"] = df["PLT"] / df["RBC"]
    V["PVR"] = df["PLT"] / df["ALB"]
    V["BA"] = df["GLB"] / df["ALB"]
    V["CAR"] = df["CRP"] / df["ALB"]
    V["CLI"] = df["ANC"] / df["Lym"]
    V["L1"] = df["Lym"] / df["WBC"]
    V["N1"] = df["ANC"] / df["WBC"]
    return V[COMPOSITES]


def filter_to_original_range(df: pd.DataFrame, ranges: dict, dataset: str):
    """Drop records with any base feature outside the cleaned Original range."""
    keep = np.ones(len(df), dtype=bool)
    detail = {}
    for feature in PANEL_A:
        low, high = ranges[feature]
        bad = ((df[feature] < low) | (df[feature] > high)).to_numpy()
        keep &= ~bad
        detail[feature] = int(bad.sum())
    detail = {k: v for k, v in sorted(detail.items(), key=lambda kv: -kv[1]) if v}
    return df.loc[keep].copy(), int((~keep).sum()), detail


def drop_physiologically_impossible(df: pd.DataFrame, dataset: str):
    """Drop records containing a physiologically impossible value.

    A value outside the *training* min/max is not by itself invalid: the training
    range is a sample boundary, not a clinical bound.  Only values outside the
    deliberately wide physiological ranges (RANGES) are treated as invalid.
    Returns the kept frame, the number dropped, per-feature violation counts, and
    a row-level log.
    """
    viol = pd.DataFrame(index=df.index)
    for feature in PANEL_A:
        low, high = (0, 1) if feature in BINARY else RANGES[feature]
        s = df[feature].astype(float)
        viol[feature] = (s < low) | (s > high)
    bad = viol.any(axis=1)
    detail = viol.loc[bad].sum().sort_values(ascending=False)
    detail = {k: int(v) for k, v in detail.items() if v}
    dropped = df.loc[bad]
    records = []
    for idx, row in dropped.iterrows():
        violated = []
        for feature in PANEL_A:
            value = float(row[feature])
            low, high = (0, 1) if feature in BINARY else RANGES[feature]
            if value < low or value > high:
                violated.append(f"{feature}={value:g} (allowed {low}-{high})")
        records.append({
            "dataset": dataset, "row_id": int(idx), "Label": int(row[LABEL]),
            "violations": "; ".join(violated),
        })
    return df.loc[~bad].copy(), int(bad.sum()), detail, pd.DataFrame(records)


def fill_with_training_medians(df: pd.DataFrame, medians: pd.Series, dataset: str, changes: list):
    df = df.copy()
    for feature in PANEL_A:
        s = df[feature]
        if feature in BINARY or feature in DAYS:
            s = s.astype(float)
        missing = s.isna()
        if missing.any():
            value = medians[feature]
            if feature in BINARY:
                value = float(value >= 0.5)
            if feature in DAYS:
                value = float(round(float(value)))
            _assign(df, dataset, feature, missing, np.full(int(missing.sum()), value),
                    "median imputation from cleaned Original", changes)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)

    raw = {k: pd.read_csv(v, index_col=0) for k, v in RAW_FILES.items()}
    for name, df in raw.items():
        df.columns = [c.strip() for c in df.columns]
        assert set(PANEL_A + ["Label"]) <= set(df.columns), name
        print(f"{name:6s} raw {df.shape} labels={df['Label'].value_counts().to_dict()}")

    changes: list = []
    fixed = {}
    for name, df in raw.items():
        print(f"  repairing {name} ...")
        d = repair_units(df, name, changes)
        fixed[name] = d

    # Original ranges are defined on the fully screened, non-imputed training data.
    ori_screened = screen_impossible(fixed["Ori"], "Ori", changes)
    ori_screened = fix_categoricals_and_days(ori_screened, "Ori", changes)
    ranges = {f: (float(ori_screened[f].min()), float(ori_screened[f].max())) for f in PANEL_A}

    # Synthetic pools: drop only records with a physiologically impossible value,
    # keep plausible outliers (training min/max is not a clinical bound), and keep
    # fractional day variables (they are perturbed / interpolated, not integers).
    filter_rows = []
    dropped_records = []
    for name in ("Syn", "SPA"):
        before = len(fixed[name])
        labels_before = fixed[name]["Label"].value_counts().to_dict()
        kept, dropped, detail, rows = drop_physiologically_impossible(fixed[name], name)
        filter_rows.append({
            "dataset": name, "n_before": before, "n_dropped": dropped,
            "n_after": len(kept),
            "pct_dropped": round(100 * dropped / before, 2),
            "labels_before": json.dumps({str(k): int(v) for k, v in labels_before.items()}),
            "labels_after": json.dumps({str(k): int(v) for k, v in kept["Label"].value_counts().items()}),
            "top_features": "; ".join(f"{k} {v}" for k, v in list(detail.items())[:10]),
        })
        fixed[name] = fix_categoricals_and_days(kept, name, changes, round_days=False)
        dropped_records.append(rows)
        print(f"  {name}: dropped {dropped}/{before} ({100*dropped/before:.2f}%)")

    # Original / validation / SMOTE: screen + impute, round day variables.
    for name in ("Ori", "Val", "SMOTE"):
        if name == "Ori":
            fixed[name] = ori_screened
            continue
        fixed[name] = screen_impossible(fixed[name], name, changes)
        fixed[name] = fix_categoricals_and_days(fixed[name], name, changes, round_days=True)

    # Medians from the cleaned Original training partition only.
    medians = ori_screened[PANEL_A].median()
    for name in ("Ori", "Val", "SMOTE"):
        fixed[name] = fill_with_training_medians(fixed[name], medians, name, changes)

    prepared = {}
    for name, df in fixed.items():
        X = df[PANEL_A].astype(float)
        comp = derive_composites(X)
        panel_b = pd.concat([X, comp], axis=1)[PANEL_B]
        assert not X.isna().any().any(), name
        assert not panel_b.isna().any().any(), name
        assert np.isfinite(panel_b.to_numpy()).all(), name
        label = df[LABEL].round().astype(int)
        a = X.copy(); a[LABEL] = label
        b = panel_b.copy(); b[LABEL] = label
        a.to_csv(out / f"{name.lower()}_panel_a.csv", index_label="row_id")
        b.to_csv(out / f"{name.lower()}_panel_b.csv", index_label="row_id")
        prepared[name] = {
            "n": int(len(df)), "n_resistant": int(label.sum()),
            "panel_a": int(a.shape[1] - 1), "panel_b": int(b.shape[1] - 1),
        }
        print(f"{name:6s} prepared n={len(df)} resistant={int(label.sum())}")

    pd.DataFrame(changes).to_csv(RES / "corrections.csv", index=False)
    pd.DataFrame(filter_rows).to_csv(RES / "filter_log.csv", index=False)
    if dropped_records:
        pd.concat(dropped_records, ignore_index=True).to_csv(RES / "dropped_records.csv", index=False)
    manifest = {
        "source": {k: str(v.relative_to(ROOT)) for k, v in RAW_FILES.items()},
        "filter_standard": {
            "Syn/SPA": "drop records with any value outside the physiological ranges; "
                       "keep plausible values outside the training min/max; keep fractional day variables",
            "Ori/Val/SMOTE": "physiological screen + training-median imputation; integer day variables",
        },
        "corrections": {
            "n_cells_changed": int(len(changes)),
            "by_action": pd.DataFrame(changes).groupby("action").size().to_dict() if changes else {},
            "by_dataset": pd.DataFrame(changes).groupby("dataset").size().to_dict() if changes else {},
        },
        "ranges_from_cleaned_original": {k: list(v) for k, v in ranges.items()},
        "datasets": prepared,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"\nwritten: {out}")
    print(f"written: {RES/'corrections.csv'}")
    print(f"written: {RES/'filter_log.csv'}")


if __name__ == "__main__":
    main()
