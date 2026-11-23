#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
import json
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import numpy as np
import pandas as pd

ALLOY_ELEMENTS = ["C", "Si", "Mn", "Cr", "V", "Al", "Ni", "Mo", "Cu", "W"]

ATOMIC_MASS = {
    "C": 12.011,
    "Si": 28.086,
    "Mn": 54.938,
    "Cr": 51.996,
    "V": 50.942,
    "Al": 26.982,
    "Ni": 58.693,
    "Mo": 95.94,
    "Cu": 63.546,
    "W": 183.84,
    "Fe": 55.845,
}

ELEMENT_DESCRIPTOR = {
    "C":  {"Z": 6,  "Ve": 4,  "chiP": 2.55, "delta_bcc": 3.053, "delta_fcc": 0.481, "dEN_Fe": 0.72,  "dEN_C": 0.00},
    "Si": {"Z": 14, "Ve": 4,  "chiP": 1.90, "delta_bcc": 0.056, "delta_fcc": 0.071, "dEN_Fe": 0.07,  "dEN_C": -0.65},
    "Mn": {"Z": 25, "Ve": 7,  "chiP": 1.55, "delta_bcc": 0.048, "delta_fcc": 0.032, "dEN_Fe": -0.28, "dEN_C": -1.00},
    "Cr": {"Z": 24, "Ve": 6,  "chiP": 1.66, "delta_bcc": 0.032, "delta_fcc": 0.016, "dEN_Fe": -0.17, "dEN_C": -0.89},
    "V":  {"Z": 23, "Ve": 5,  "chiP": 1.63, "delta_bcc": 0.065, "delta_fcc": 0.048, "dEN_Fe": -0.20, "dEN_C": -0.92},
    "Al": {"Z": 13, "Ve": 3,  "chiP": 1.61, "delta_bcc": 0.153, "delta_fcc": 0.135, "dEN_Fe": -0.22, "dEN_C": -0.94},
    "Ni": {"Z": 28, "Ve": 10, "chiP": 1.91, "delta_bcc": 0.000, "delta_fcc": 0.016, "dEN_Fe": 0.08,  "dEN_C": -0.64},
    "Mo": {"Z": 42, "Ve": 6,  "chiP": 2.16, "delta_bcc": 0.121, "delta_fcc": 0.103, "dEN_Fe": 0.33,  "dEN_C": -0.39},
    "Cu": {"Z": 29, "Ve": 11, "chiP": 1.90, "delta_bcc": 0.032, "delta_fcc": 0.016, "dEN_Fe": 0.07,  "dEN_C": -0.65},
    "W":  {"Z": 74, "Ve": 6,  "chiP": 2.36, "delta_bcc": 0.137, "delta_fcc": 0.119, "dEN_Fe": 0.53,  "dEN_C": -0.19},
}

FEATURE_28 = [
    "C", "Si", "Mn", "Cr", "V", "Al", "Ni", "Mo", "Cu", "W",
    "F_Z", "F_Ve", "F_chiP", "F_delta_bcc", "F_delta_fcc", "F_dEN_Fe", "F_dEN_C",
    "Ta", "Ti", "ti", "dTA", "dTI", "tau", "Ms", "Tt", "tt", "sph", "tmp",
]

TRAINING_28 = [
    "C", "Si", "Mn", "Cr", "V", "Al", "Ni", "Mo", "Cu", "W",
    "F_Z", "F_Ve", "F_χP", "F_δ_bcc", "F_δ_fcc", "F_ΔEN_Fe", "F_ΔEN_C",
    "T_austenitizing", "T_isothermal ", "t_isothermal", "ΔTA", "ΔTI", "τ_norm",
    "Ms", "T_tem", "T_tim", "has_spheroidizing", "has_tempering",
]

FEATURE_BASE_26 = [c for c in FEATURE_28 if c not in ["ti", "Ms"]]
FEATURE_31 = FEATURE_BASE_26 + ["Ceq", "Ta_Ti_diff", "C_DTI", "DC_Ti", "C_Cr"]

TARGET_FEATURES = {
    "Rm": ["Ti", "tau", "F_dEN_Fe", "Mo", "Mn", "dTI", "F_Ve", "Ta", "Si", "F_delta_bcc", "DC_Ti", "Ta_Ti_diff", "Ceq", "C_Cr", "C_DTI"],
    "Re": ["Ti", "tau", "F_Z", "F_Ve", "Ni", "dTI", "F_delta_bcc", "F_dEN_Fe", "Al", "Si", "DC_Ti", "Ta_Ti_diff", "Ceq", "C_Cr", "C_DTI"],
    "A":  ["F_delta_fcc", "Ti", "sph", "Si", "Cr", "Mn", "dTA", "F_dEN_C", "dTI", "F_Ve", "F_Z", "DC_Ti", "Ceq", "Ta_Ti_diff", "C_Cr"],
    "Ak": ["F_delta_fcc", "Si", "Ti", "F_dEN_C", "Ta", "dTI", "tau", "Mn", "Al", "dTA", "Ta_Ti_diff", "Ceq", "C_Cr", "C_DTI", "DC_Ti"],
}

TARGET_FEATURES_TRAINING_NAMES = {
    "Rm": ["Ti", "τ", "ΔFe", "Mo", "Mn", "ΔTI", "Ve", "Ta", "Si", "δb", "DC_Ti", "Ta_Ti_diff", "Ceq", "C_Cr", "C_DTI"],
    "Re": ["Ti", "τ", "Z", "Ve", "Ni", "ΔTI", "δb", "ΔFe", "Al", "Si", "DC_Ti", "Ta_Ti_diff", "Ceq", "C_Cr", "C_DTI"],
    "A":  ["δf", "Ti", "sph", "Si", "Cr", "Mn", "ΔTA", "ΔC", "ΔTI", "Ve", "Z", "DC_Ti", "Ceq", "Ta_Ti_diff", "C_Cr"],
    "Ak": ["δf", "Si", "Ti", "ΔC", "Ta", "ΔTI", "τ", "Mn", "Al", "ΔTA", "Ta_Ti_diff", "Ceq", "C_Cr", "C_DTI", "DC_Ti"],
}

EXPORT_NAME_MAP = dict(zip(FEATURE_28, TRAINING_28))
EXPORT_NAME_MAP.update({
    "Ceq": "Ceq",
    "Ta_Ti_diff": "Ta_Ti_diff",
    "C_DTI": "C_DTI",
    "DC_Ti": "DC_Ti",
    "C_Cr": "C_Cr",
})


def clean_number(value, default=np.nan) -> float:
    if pd.isna(value):
        return default
    try:
        return float(value)
    except Exception:
        return default


def excel_xls_to_csv_with_com(path: Path) -> pd.DataFrame:
    """Read old .xls through Excel COM on Windows when xlrd is unavailable."""
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "excel_export.csv"
        ps = f"""
$excel = New-Object -ComObject Excel.Application
$excel.Visible = $false
$excel.DisplayAlerts = $false
$wb = $excel.Workbooks.Open('{str(path)}')
$wb.Worksheets.Item(1).SaveAs('{str(csv_path)}', 6)
$wb.Close($false)
$excel.Quit()
"""
        subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps],
            check=True,
            capture_output=True,
            text=True,
        )
        return pd.read_csv(csv_path)


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in [".csv", ".txt", ".tsv"]:
        return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    if suffix in [".xlsx", ".xlsm"]:
        return pd.read_excel(path)
    if suffix == ".xls":
        try:
            return pd.read_excel(path)
        except ImportError:
            return excel_xls_to_csv_with_com(path)
    raise ValueError(f"Unsupported input format: {path.suffix}")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().replace("\ufeff", "").replace("\u3000", "") for c in out.columns]
    return out


def first_existing(row: pd.Series, names: Iterable[str], required: bool = True) -> float:
    for name in names:
        if name in row.index and pd.notna(row[name]):
            return clean_number(row[name])
    if required:
        raise ValueError(f"Missing required column. Tried: {list(names)}")
    return np.nan


def extract_wt_phase_table(df: pd.DataFrame, trace_threshold: float) -> pd.DataFrame:
    df = normalize_columns(df)
    rows = []
    for idx, row in df.iterrows():
        out = {"source_index": idx}
        out["Ms"] = first_existing(row, ["Ms temperature", "Ms", "MS", "M_s"])
        out["Ac3"] = first_existing(row, ["A3 Temperature", "Ac3", "A3", "A_c3"])
        for el in ALLOY_ELEMENTS:
            val = first_existing(row, [f"Mass percent {el}", f"mass percent {el}", el], required=False)
            if pd.isna(val):
                val = 0.0
            if abs(val) <= trace_threshold:
                val = 0.0
            out[f"{el}_wt"] = max(float(val), 0.0)
        fe = first_existing(row, ["Mass percent Fe", "Fe"], required=False)
        out["Fe_wt"] = fe if pd.notna(fe) else 100.0 - sum(out[f"{el}_wt"] for el in ALLOY_ELEMENTS)
        rows.append(out)
    return pd.DataFrame(rows)


def wt_to_at_percent(row: pd.Series) -> Dict[str, float]:
    wt = {el: clean_number(row[f"{el}_wt"], 0.0) for el in ALLOY_ELEMENTS}
    fe_wt = clean_number(row.get("Fe_wt", np.nan), np.nan)
    if pd.isna(fe_wt):
        fe_wt = 100.0 - sum(wt.values())
    if fe_wt < -1e-8:
        raise ValueError(f"Alloy wt.% sum exceeds 100: {sum(wt.values()):.6f}")
    fe_wt = max(float(fe_wt), 0.0)
    moles = {el: wt[el] / ATOMIC_MASS[el] for el in ALLOY_ELEMENTS}
    moles["Fe"] = fe_wt / ATOMIC_MASS["Fe"]
    total = sum(moles.values())
    if total <= 0:
        raise ValueError("Total moles are zero.")
    return {el: moles[el] / total * 100.0 for el in ALLOY_ELEMENTS + ["Fe"]}


def weighted_descriptors(at_pct: Dict[str, float]) -> Dict[str, float]:
    solute_sum = sum(at_pct[el] for el in ALLOY_ELEMENTS)
    if solute_sum <= 0:
        raise ValueError("Solute at.% sum is zero.")
    props = {
        "F_Z": "Z",
        "F_Ve": "Ve",
        "F_chiP": "chiP",
        "F_delta_bcc": "delta_bcc",
        "F_delta_fcc": "delta_fcc",
        "F_dEN_Fe": "dEN_Fe",
        "F_dEN_C": "dEN_C",
    }
    return {
        out_name: sum(at_pct[el] * ELEMENT_DESCRIPTOR[el][prop] for el in ALLOY_ELEMENTS) / solute_sum
        for out_name, prop in props.items()
    }


def sample_by_zones(
    rng: np.random.Generator,
    low: float,
    q1: float,
    q2: float,
    high: float,
    n_per_zone: int,
    labels: Tuple[str, str, str] = ("low", "mid", "high"),
) -> Tuple[List[float], List[str]]:
    ranges = [
        (labels[0], low, q1),
        (labels[1], q1, q2),
        (labels[2], q2, high),
    ]
    values: List[float] = []
    labels: List[str] = []
    for label, lo, hi in ranges:
        if hi <= lo:
            raise ValueError(f"Invalid sampling range for {label}: {lo} - {hi}")
        for _ in range(n_per_zone):
            values.append(float(rng.uniform(lo, hi)))
            labels.append(label)
    order = rng.permutation(len(values))
    return [values[i] for i in order], [labels[i] for i in order]


def add_interactions(row: Dict[str, float]) -> None:
    row["Ceq"] = row["C"] + row["Mn"] / 6.0 + (row["Cr"] + row["Mo"] + row["V"]) / 5.0 + (row["Ni"] + row["Cu"]) / 15.0
    row["Ta_Ti_diff"] = row["Ta"] - row["Ti"]
    row["C_DTI"] = row["C"] * row["dTI"]
    row["DC_Ti"] = row["F_dEN_C"] * row["Ti"]
    row["C_Cr"] = row["C"] * row["Cr"]


def build_candidate_features(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(args.seed)
    raw = read_table(Path(args.input))
    wt_phase = extract_wt_phase_table(raw, trace_threshold=args.trace_threshold)

    feature_rows: List[Dict[str, float]] = []
    meta_rows: List[Dict[str, float | str | int]] = []

    for _, src in wt_phase.iterrows():
        at = wt_to_at_percent(src)
        desc = weighted_descriptors(at)
        dti_values, dti_zones = sample_by_zones(
            rng, args.dti_min, args.dti_q1, args.dti_q2, args.dti_max, args.n_per_zone
        )
        tau_values, tau_zones = sample_by_zones(
            rng, args.tau_min, args.tau_q1, args.tau_q2, args.tau_max, args.n_per_zone,
            labels=("short", "mid", "long"),
        )

        for variant_idx, (dti, dti_zone, tau, tau_zone) in enumerate(zip(dti_values, dti_zones, tau_values, tau_zones), start=1):
            ta_offset = float(rng.choice(args.ta_offsets))
            tempering = bool(rng.random() < args.tempering_probability)
            Ta = float(src["Ac3"] + ta_offset)
            Ti = float(src["Ms"] + dti)
            ti = np.nan
            Tt = 320.0 if tempering else 0.0
            tt = 60.0 if tempering else 0.0

            feat = {el: at[el] for el in ALLOY_ELEMENTS}
            feat.update(desc)
            feat.update({
                "Ta": Ta,
                "Ti": Ti,
                "ti": ti,
                "dTA": ta_offset,
                "dTI": dti,
                "tau": tau,
                "Ms": float(src["Ms"]),
                "Tt": Tt,
                "tt": tt,
                "sph": 0.0,
                "tmp": 1.0 if tempering else 0.0,
            })
            add_interactions(feat)
            feature_rows.append(feat)

            meta = {
                "source_index": int(src["source_index"]),
                "variant": variant_idx,
                "dTI_zone": dti_zone,
                "tau_zone": tau_zone,
                "Ac3": float(src["Ac3"]),
                "Ms": float(src["Ms"]),
                "Ta_offset": ta_offset,
                "tempering": int(tempering),
            }
            for el in ALLOY_ELEMENTS:
                meta[f"{el}_wt"] = src[f"{el}_wt"]
                meta[f"{el}_at"] = at[el]
            meta["Fe_wt"] = src["Fe_wt"]
            meta["Fe_at"] = at["Fe"]
            meta_rows.append(meta)

    features = pd.DataFrame(feature_rows)
    metadata = pd.DataFrame(meta_rows)
    return features, metadata


def round_numeric(df: pd.DataFrame, digits: int) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].round(digits)
    return out


def export_outputs(features: pd.DataFrame, metadata: pd.DataFrame, args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    features = round_numeric(features, args.decimals)
    metadata = round_numeric(metadata, args.decimals)

    x28_internal = features[FEATURE_28]
    x31_internal = features[FEATURE_31]

    x28_training = x28_internal.rename(columns=EXPORT_NAME_MAP)
    x31_training = x31_internal.rename(columns=EXPORT_NAME_MAP)

    metadata.to_csv(out_dir / "candidate_metadata.csv", index=False, encoding="utf-8-sig")
    x28_training.to_csv(out_dir / "candidate_X_28dim.csv", index=False, encoding="utf-8-sig")
    x31_training.to_csv(out_dir / "candidate_X_31dim.csv", index=False, encoding="utf-8-sig")
    np.save(out_dir / "candidate_X_28dim.npy", x28_training.to_numpy(dtype=float))
    np.save(out_dir / "candidate_X_31dim.npy", x31_training.to_numpy(dtype=float))

    for target, cols in TARGET_FEATURES.items():
        target_df = features[cols].copy()
        target_df.columns = TARGET_FEATURES_TRAINING_NAMES[target]
        target_df = round_numeric(target_df, args.decimals)
        target_df.to_csv(out_dir / f"candidate_model_input_{target}_15dim.csv", index=False, encoding="utf-8-sig")
        np.save(out_dir / f"candidate_model_input_{target}_15dim.npy", target_df.to_numpy(dtype=float))

    summary = {
        "input": str(args.input),
        "rows_in": int(metadata["source_index"].nunique()),
        "variants_per_composition": int(args.n_per_zone * 3),
        "rows_out": int(len(features)),
        "decimals": int(args.decimals),
        "dTI_ranges": {
            "low": [args.dti_min, args.dti_q1],
            "mid": [args.dti_q1, args.dti_q2],
            "high": [args.dti_q2, args.dti_max],
        },
        "tau_ranges": {
            "short": [args.tau_min, args.tau_q1],
            "mid": [args.tau_q1, args.tau_q2],
            "long": [args.tau_q2, args.tau_max],
        },
        "ta_offsets": args.ta_offsets,
        "tempering_rule": {
            "probability": args.tempering_probability,
            "if_tempered": {"Tt": 320.0, "tt": 60.0, "tmp": 1.0},
            "if_not_tempered": {"Tt": 0.0, "tt": 0.0, "tmp": 0.0},
        },
        "t_isothermal_rule": "unknown for design candidates; not sampled; excluded from 31D and target-specific 15D model inputs",
        "outputs": [
            "candidate_metadata.csv",
            "candidate_X_28dim.csv",
            "candidate_X_31dim.csv",
            "candidate_X_28dim.npy",
            "candidate_X_31dim.npy",
            "candidate_model_input_{target}_15dim.csv/npy",
        ],
    }
    with open(out_dir / "candidate_feature_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def validate_original_conversion(args: argparse.Namespace) -> None:
    if not args.validate_wt or not args.validate_at:
        return
    wt = read_table(Path(args.validate_wt))
    at_ref = read_table(Path(args.validate_at))
    wt_phase_like = normalize_columns(wt)
    rows = []
    for idx, row in wt_phase_like.iterrows():
        src = {"source_index": idx}
        for el in ALLOY_ELEMENTS:
            val = clean_number(row.get(el, 0.0), 0.0)
            src[f"{el}_wt"] = 0.0 if pd.isna(val) else val
        src["Fe_wt"] = 100.0 - sum(src[f"{el}_wt"] for el in ALLOY_ELEMENTS)
        rows.append(wt_to_at_percent(pd.Series(src)))
    at_calc = pd.DataFrame(rows)
    at_ref = normalize_columns(at_ref)
    common = [el for el in ALLOY_ELEMENTS if el in at_ref.columns]
    diff = (at_calc[common].round(3).to_numpy() - at_ref[common].fillna(0).to_numpy())
    print("\nValidation: wt -> at conversion")
    print(f"  rows compared: {len(at_calc)}")
    print(f"  max abs diff after 3-dec rounding: {float(np.nanmax(np.abs(diff))):.6g}")

    if args.validate_x:
        X = np.load(args.validate_x)
        desc_rows = []
        for _, row in at_ref.iterrows():
            at = {el: clean_number(row.get(el, 0.0), 0.0) for el in ALLOY_ELEMENTS}
            desc_rows.append(weighted_descriptors(at))
        desc_df = pd.DataFrame(desc_rows)
        x_desc = X[:, 10:17]
        calc_desc = desc_df[["F_Z", "F_Ve", "F_chiP", "F_delta_bcc", "F_delta_fcc", "F_dEN_Fe", "F_dEN_C"]].to_numpy()
        # Compare as unordered rows because project CSV/X row ordering may differ after cleaning.
        first_diff = np.max(np.abs(calc_desc[0] - x_desc[0]))
        print("\nValidation: physical descriptor formula")
        print(f"  first-row descriptor max abs diff vs X_28dim: {float(first_diff):.6g}")
        print("  note: full row alignment is not assumed because local data5.csv ordering differs from X_28dim.")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Build candidate model features from wt.% + Ac3/Ms.")
    parser.add_argument("--input", default=str(script_dir / "07002 Ms and A3.xls"), help="Thermo-Calc wt.% + Ms/A3 table.")
    parser.add_argument("--output-dir", default=str(script_dir / "candidate_model_features"), help="Output directory.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decimals", type=int, default=3, help="Decimal places to keep in CSV/NPY outputs.")
    parser.add_argument("--trace-threshold", type=float, default=1e-4, help="Composition values at or below this wt.% are treated as 0.")
    parser.add_argument("--n-per-zone", type=int, default=2, help="Samples per zone. Default gives 6 variants.")
    parser.add_argument("--dti-min", type=float, default=0.0)
    parser.add_argument("--dti-q1", type=float, default=25.9928)
    parser.add_argument("--dti-q2", type=float, default=52.0144)
    parser.add_argument("--dti-max", type=float, default=110.8)
    parser.add_argument("--tau-min", type=float, default=0.133333)
    parser.add_argument("--tau-q1", type=float, default=1.0)
    parser.add_argument("--tau-q2", type=float, default=1.6139454)
    parser.add_argument("--tau-max", type=float, default=3.754972)
    parser.add_argument("--ta-offsets", type=float, nargs="+", default=[30.0, 40.0, 50.0])
    parser.add_argument("--tempering-probability", type=float, default=0.5)
    parser.add_argument("--validate-wt", default=None, help="Optional original wt.% file for conversion validation.")
    parser.add_argument("--validate-at", default=None, help="Optional converted at.% reference file for validation.")
    parser.add_argument("--validate-x", default=None, help="Optional X_28dim.npy for descriptor validation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_original_conversion(args)
    features, metadata = build_candidate_features(args)
    export_outputs(features, metadata, args)
    print("\nDone.")
    print(f"  input rows: {metadata['source_index'].nunique()}")
    print(f"  output rows: {len(features)}")
    print(f"  output dir: {args.output_dir}")
    print("  decimals: 3")


if __name__ == "__main__":
    main()
