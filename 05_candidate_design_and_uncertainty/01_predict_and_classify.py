#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import ast
import json
import pickle
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd

# 项目根目录与输出路径设置
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "design_outputs"
OUTPUT_PREDICTION_TABLE = OUTPUT_DIR / "candidate_prediction_summary_simplified.csv"

# 候选特征路径
CANDIDATE_DIR = PROJECT_ROOT / "data_clean" / "design" / "candidate_model_features"

TARGETS = ["Rm", "Re", "A", "Ak"]
ALLOY_ELEMENTS = ["C", "Si", "Mn", "Cr", "V", "Al", "Ni", "Mo", "Cu", "W"]
STRENGTH_TARGETS = ["Rm", "Re"]
DUCTILITY_TARGETS = ["A", "Ak"]
TARGET_SENSITIVE_ZONES = {"Rm": {0, 3}, "Re": {0, 3}, "A": {7, 8}, "Ak": {7, 8}}

MODEL_SUBDIR = {
    "Rm": "Rm_EXP_3_Stability_Rm",
    "Re": "Re_EXP_3_Stability_Re",
    "A": "A_EXP_3_Stability_A",
    "Ak": "Ak_EXP_3_Stability_Ak",
}

DTI_ZONE_INDEX = {"low": 0, "mid": 1, "high": 2}
TAU_ZONE_INDEX = {"short": 0, "mid": 1, "long": 2}
ZONE_LABELS = {
    0: "Low Delta_TI x Short tau", 1: "Low Delta_TI x Mid tau", 2: "Low Delta_TI x Long tau",
    3: "Mid Delta_TI x Short tau", 4: "Mid Delta_TI x Mid tau", 5: "Mid Delta_TI x Long tau",
    6: "High Delta_TI x Short tau", 7: "High Delta_TI x Mid tau", 8: "High Delta_TI x Long tau",
}

FEATURE_ALIAS = {
    "F_Z": "Z", "F_Ve": "Ve", "F_χP": "chiP", "F_δ_bcc": "delta_b", "F_δ_fcc": "delta_f",
    "F_ΔEN_Fe": "Delta_Fe", "F_ΔEN_C": "Delta_C", "T_austenitizing": "Ta",
    "T_isothermal ": "Ti", "t_isothermal": "ti", "ΔTA": "Delta_TA", "ΔTI": "Delta_TI",
    "τ_norm": "tau", "T_tem": "Tt", "T_tim": "tt", "has_spheroidizing": "sph", "has_tempering": "tmp",
}

PROCESS_COLUMNS_28 = [
    "T_austenitizing", "T_isothermal ", "ΔTA", "ΔTI", "τ_norm",
    "Ms", "T_tem", "T_tim", "has_spheroidizing", "has_tempering",
]


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def zone_id_from_metadata(meta: pd.DataFrame) -> pd.Series:
    dti = meta["dTI_zone"].astype(str).str.lower().map(DTI_ZONE_INDEX)
    tau = meta["tau_zone"].astype(str).str.lower().map(TAU_ZONE_INDEX)
    if dti.isna().any() or tau.isna().any():
        bad = meta.loc[dti.isna() | tau.isna(), ["dTI_zone", "tau_zone"]].drop_duplicates()
        raise ValueError(f"Unknown zone labels in metadata:\n{bad}")
    return (dti.astype(int) * 3 + tau.astype(int)).astype(int)


def load_candidate_tables(candidate_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    meta = pd.read_csv(candidate_dir / "candidate_metadata.csv")
    meta["zone_id"] = zone_id_from_metadata(meta)
    meta["zone_label"] = meta["zone_id"].map(ZONE_LABELS)

    x28 = pd.read_csv(candidate_dir / "candidate_X_28dim.csv")
    for col in PROCESS_COLUMNS_28:
        if col in x28.columns:
            meta[col] = x28[col].round(3)

    x31 = pd.read_csv(candidate_dir / "candidate_X_31dim.csv")
    formula_features = x31.rename(columns=FEATURE_ALIAS).copy()

    target_inputs = {}
    for target in TARGETS:
        target_inputs[target] = pd.read_csv(candidate_dir / f"candidate_model_input_{target}_15dim.csv")

    return meta, formula_features, target_inputs


def predict_lopzo(model_root: Path, target_inputs: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
    from catboost import CatBoostRegressor
    predictions: Dict[str, np.ndarray] = {}
    for target in TARGETS:
        subdir = model_root / MODEL_SUBDIR[target]
        model_path = subdir / f"{target}_final_model.cbm"
        config = read_json(subdir / f"{target}_model_config.json")
        features = list(config["feature_names"])

        model = CatBoostRegressor()
        model.load_model(str(model_path))
        pred = np.asarray(model.predict(target_inputs[target][features]), dtype=float)

        if bool(config.get("use_log_transform", False)):
            offset = float(config.get("log_offset") or 0.0)
            pred = np.exp(pred) - offset
        predictions[target] = pred
    return predictions


def safe_eval_equation(expr: str, features: pd.DataFrame) -> np.ndarray:
    env = {c: features[c].to_numpy(dtype=float) for c in features.columns}
    env.update({
        "abs": np.abs, "sqrt": np.sqrt, "square": np.square, "log": np.log,
        "exp": np.exp, "sin": np.sin, "cos": np.cos, "maximum": np.maximum,
        "minimum": np.minimum, "np": np,
    })
    ast.parse(expr, mode="eval")
    out = eval(compile(expr, "<symbolic_equation>", "eval"), {"__builtins__": {}}, env)
    return np.asarray(out, dtype=float)


def apply_residual_patch(
        predictions: Dict[str, np.ndarray],
        meta: pd.DataFrame,
        formula_features: pd.DataFrame,
        preprocess_root: Path,
        sr_root: Path,
        sr_norm_limit: float,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], pd.DataFrame]:
    pred_bias: Dict[str, np.ndarray] = {}
    pred_comp: Dict[str, np.ndarray] = {}
    pred_rank: Dict[str, np.ndarray] = {}
    patch_cols = pd.DataFrame(index=meta.index)

    for target in TARGETS:
        pkl_path = preprocess_root / target / f"{target}_normalized_residuals.pkl"
        with open(pkl_path, "rb") as f:
            patch_data = pickle.load(f)

        zone_stats = {int(k): v for k, v in patch_data.get("zone_stats", {}).items()}
        calibration_zones = set([int(z) for z in patch_data.get("calibration_zones", [])])
        warning_zones = set([int(z) for z in patch_data.get("warning_zones", [])])
        train_features = pd.DataFrame(patch_data["X_clean"], columns=patch_data["feature_names"])
        train_df = patch_data["df"].copy()

        y_base = predictions[target]
        y_bias = y_base.copy()
        stage = np.full(len(meta), "base", dtype=object)
        gate = np.full(len(meta), "not_applicable", dtype=object)
        trend_out = np.full(len(meta), np.nan, dtype=float)
        trend_norm_out = np.full(len(meta), np.nan, dtype=float)

        for zid in calibration_zones:
            if zid in zone_stats:
                mask = meta["zone_id"].to_numpy() == zid
                y_bias[mask] = y_base[mask] + float(zone_stats[zid]["mu"])
                stage[mask] = "bias_only"

        y_comp = y_bias.copy()
        y_rank = y_base.copy()
        selected = pd.read_csv(sr_root / target / f"{target}_selected_equation_zone_effects.csv")

        for _, row in selected.iterrows():
            if not bool(row.get("activated", False)):
                continue
            zid = int(row["zone_id"])
            if zid in warning_zones or zid not in zone_stats:
                continue
            mask = meta["zone_id"].to_numpy() == zid
            if not np.any(mask):
                continue

            equation = str(row["equation"])
            trend = safe_eval_equation(equation, formula_features)
            trend_mu = float(row["trend_mean_applied"])
            trend_std = float(row["trend_std_applied"])
            if not np.isfinite(trend_std) or abs(trend_std) < 1e-12:
                continue
            trend_norm = (trend - trend_mu) / trend_std
            train_idx = train_df.loc[train_df["zone_id"].astype(int).eq(zid), "global_idx"].astype(int).to_numpy()
            train_trend = safe_eval_equation(equation, train_features.iloc[train_idx])
            train_trend = train_trend[np.isfinite(train_trend)]
            if len(train_trend) == 0:
                continue
            train_min = float(np.nanmin(train_trend))
            train_max = float(np.nanmax(train_trend))
            sigma_z = float(zone_stats[zid]["sigma"])
            w_adaptive = float(row["w_adaptive"])
            y_comp[mask] = y_bias[mask] + w_adaptive * sigma_z * trend_norm[mask]

            in_support = (
                    np.isfinite(trend)
                    & np.isfinite(trend_norm)
                    & (trend >= train_min)
                    & (trend <= train_max)
                    & (np.abs(trend_norm) <= sr_norm_limit)
            )
            usable = mask & in_support
            extrapolated = mask & ~in_support
            y_rank[usable] = y_comp[usable]
            stage[mask] = "bias_plus_centered_sr"
            gate[usable] = "used_for_ranking"
            gate[extrapolated] = "extrapolation_warning"
            trend_out[mask] = trend[mask]
            trend_norm_out[mask] = trend_norm[mask]

        warn_mask = meta["zone_id"].isin(warning_zones).to_numpy()
        stage[warn_mask] = "warning_only"
        gate[warn_mask] = "warning_zone_no_sr"
        y_rank[warn_mask] = y_base[warn_mask]

        pred_bias[target] = y_bias
        pred_comp[target] = y_comp
        pred_rank[target] = y_rank
        patch_cols[f"{target}_sr_gate"] = gate
        patch_cols[f"{target}_sr_trend_norm"] = np.round(trend_norm_out, 6)

    return pred_bias, pred_comp, pred_rank, patch_cols


def attach_uq(meta: pd.DataFrame, output: pd.DataFrame, uq_root: Path) -> pd.DataFrame:
    out = output.copy()
    for target in TARGETS:
        with open(uq_root / target / "uq_zone_table.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        zones = {int(k): v for k, v in data["zones"].items()}
        global_q = float(data["global_q_ref"])
        q = meta["zone_id"].map(lambda z: float(zones[int(z)]["q_final"]))
        role = meta["zone_id"].map(lambda z: str(zones[int(z)]["role"]))

        out[f"{target}_uq_q90"] = q.round(3)
        out[f"{target}_uq_norm"] = (q / global_q).round(3)
        out[f"{target}_uq_role"] = role
        out[f"{target}_ranking_lower90"] = (out[f"{target}_pred_for_ranking"] - q).round(3)
        out[f"{target}_ranking_upper90"] = (out[f"{target}_pred_for_ranking"] + q).round(3)
    return out


def risk_level_to_num(level: str) -> int:
    return {"low": 0, "medium": 1, "high": 2}.get(str(level), 2)


def target_risk_level(row: pd.Series, target: str) -> str:
    zone_id = int(row["zone_id"])
    if zone_id == 6 or str(row.get(f"{target}_uq_role", "")) == "warning":
        return "high"
    if str(row.get(f"{target}_sr_gate", "")) == "extrapolation_warning":
        return "medium"
    uq_norm = float(row.get(f"{target}_uq_norm", np.nan))
    if np.isfinite(uq_norm) and uq_norm >= 2.0:
        return "high"
    if zone_id in TARGET_SENSITIVE_ZONES[target]:
        return "medium"
    if np.isfinite(uq_norm) and uq_norm >= 1.2:
        return "medium"
    return "low"


def design_tier(row: pd.Series) -> str:
    if int(row["zone_id"]) == 6 or row["strength_risk"] == "high":
        return "C_not_recommended"
    if row["strength_risk"] == "medium" or row["ductility_toughness_risk"] == "medium":
        return "B_review"
    if row["ductility_toughness_risk"] == "high":
        return "B_review"
    return "A_primary"


def main(output_dir: Path = OUTPUT_DIR) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_table = output_dir / OUTPUT_PREDICTION_TABLE.name

    model_root = PROJECT_ROOT / "results_final" / "EXP_3_Stability"
    preprocess_root = PROJECT_ROOT / "sr_patch_preprocessing_v5_bias_separated"
    sr_root = PROJECT_ROOT / "sr_patch_results_v5_bias_separated"
    uq_root = PROJECT_ROOT / "uq_results_v5_bias_separated"

    print("[核心] 正在读取候选设计特征与参数表...")
    meta, formula_features, target_inputs = load_candidate_tables(CANDIDATE_DIR)

    print("[模型] 启动 CatBoost 多目标预测机制...")
    base_pred = predict_lopzo(model_root, target_inputs)

    print("[修正] 计算符号回归偏差校准补丁...")
    _, _, ranking_pred, patch_cols = apply_residual_patch(
        base_pred, meta, formula_features, preprocess_root, sr_root, sr_norm_limit=3.0
    )

    out = meta.copy()
    for target in TARGETS:
        out[f"{target}_pred_for_ranking"] = np.round(ranking_pred[target], 3)
    out = pd.concat([out, patch_cols], axis=1)

    print("[区间] 正在附着 Mondrian 置信区间与不确定度...")
    out = attach_uq(meta, out, uq_root)

    print("[决策] 正在进行材料适用域决策评级与打分...")
    for target in TARGETS:
        out[f"{target}_target_risk"] = out.apply(lambda r: target_risk_level(r, target), axis=1)
        out[f"{target}_risk_num"] = out[f"{target}_target_risk"].map(risk_level_to_num)

    out["strength_risk_num"] = out[[f"{t}_risk_num" for t in STRENGTH_TARGETS]].max(axis=1)
    out["ductility_toughness_risk_num"] = out[[f"{t}_risk_num" for t in DUCTILITY_TARGETS]].max(axis=1)
    inv_risk = {0: "low", 1: "medium", 2: "high"}
    out["strength_risk"] = out["strength_risk_num"].map(inv_risk)
    out["ductility_toughness_risk"] = out["ductility_toughness_risk_num"].map(inv_risk)
    out["sr_extrapolation_any"] = out[[f"{t}_sr_gate" for t in TARGETS]].eq("extrapolation_warning").any(axis=1)
    out["design_tier"] = out.apply(design_tier, axis=1)
    out["design_tier_order"] = out["design_tier"].map({"A_primary": 0, "B_review": 1, "C_not_recommended": 2})

    # 直接使用冶金强塑积进行全局排序与分配排名 (强塑积 PSE = Rm * A / 100.0)
    out["strong_plastic_product"] = np.round((out["Rm_pred_for_ranking"] * out["A_pred_for_ranking"]) / 100.0, 3)
    out = out.sort_values(["design_tier_order", "strong_plastic_product"], ascending=[True, False]).reset_index(
        drop=True)
    out["candidate_rank"] = np.arange(1, len(out) + 1)

    keep_cols = [
        "candidate_rank", "source_index", "variant", "design_tier", "zone_id", "zone_label",
        "dTI_zone", "tau_zone", "Ac3", "Ms", "T_austenitizing", "T_isothermal ", "ΔTA", "ΔTI",
        "τ_norm", "T_tem", "T_tim", "has_spheroidizing", "has_tempering"
    ]
    for el in ALLOY_ELEMENTS:
        keep_cols.append(f"{el}_wt")

    for target in TARGETS:
        keep_cols.extend([
            f"{target}_pred_for_ranking", f"{target}_uq_norm",
            f"{target}_ranking_lower90", f"{target}_ranking_upper90", f"{target}_target_risk"
        ])
    keep_cols.extend(["strength_risk", "ductility_toughness_risk", "sr_extrapolation_any", "strong_plastic_product"])

    final_output = out[keep_cols].copy()
    final_output.to_csv(output_table, index=False, encoding="utf-8-sig")
    print(f"\n[任务结束] 预测计算及分类表单已经输出至:\n >> {output_table}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    arguments = parser.parse_args()
    main(arguments.output_dir)