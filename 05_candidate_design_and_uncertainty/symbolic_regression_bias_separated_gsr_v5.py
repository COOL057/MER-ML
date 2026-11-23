# symbolic_regression_bias_separated_gsr_v5.py
# v5: Bias-separated gated symbolic regression (BS-GSR).
#     Base CatBoost, Base + Zone bias, and Base + Zone bias + centered SR
#     are exported separately for ablation. Z6 is warning-only.

import argparse
import pickle
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score

try:
    from pysr import PySRRegressor
    PYSR_AVAILABLE = True
except ImportError:
    PYSR_AVAILABLE = False

warnings.filterwarnings("ignore")


# ════════════════════════════════════════════════════════════════
# Static configuration
# ════════════════════════════════════════════════════════════════

BATCH_CONFIG = {
    # Z6 = High Delta_TI x Short tau.
    # It is an overlap-risk warning zone and is not calibrated.
    "Rm": {"Short_tau_Regime": [0, 3]},
    "Re": {"Short_tau_Regime": [0, 3]},
    "A": {"High_DTI_Regime": [7, 8]},
    "Ak": {"High_DTI_Regime": [7, 8]},
}

WARNING_ZONES = {
    "Rm": [6],
    "Re": [6],
    "A": [6],
    "Ak": [6],
}

ZONE_LABELS = {
    0: "Low Delta_TI x Short tau",
    3: "Mid Delta_TI x Short tau",
    6: "High Delta_TI x Short tau",
    7: "High Delta_TI x Mid tau",
    8: "High Delta_TI x Long tau",
}

EPS = 1e-9


def make_pysr_safe(name: str) -> str:
    """Replace symbols and add x_ prefix to avoid Julia/SymPy name conflicts."""
    safe = (
        name.replace("Δ", "Delta")
        .replace("τ", "tau")
        .replace("χ", "chi")
        .replace("δ", "delta")
    )
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", safe)
    return f"x_{safe}"


def restore_feature_names(equation: str, features: list) -> str:
    """Replace PySR-safe variable names with original feature names."""
    eq_display = str(equation)
    for f in features:
        eq_display = re.sub(r"\b" + make_pysr_safe(f) + r"\b", f, eq_display)
    return eq_display


def permutation_spearmanr(x, y, n_perm: int = 200, seed: int = 42):
    """Permutation-based two-sided Spearman-rank test."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    finite_mask = np.isfinite(x) & np.isfinite(y)
    x = x[finite_mask]
    y = y[finite_mask]

    if len(x) < 4 or np.std(x) <= EPS or np.std(y) <= EPS:
        return 0.0, 1.0

    rho_obs, _ = spearmanr(x, y)
    if np.isnan(rho_obs):
        return 0.0, 1.0

    rng = np.random.default_rng(seed)
    count = 0

    for _ in range(n_perm):
        rho_perm, _ = spearmanr(rng.permutation(x), y)
        if np.isnan(rho_perm):
            rho_perm = 0.0
        if abs(rho_perm) >= abs(rho_obs):
            count += 1

    p_perm = (count + 1) / (n_perm + 1)
    return float(rho_obs), float(p_perm)


def zscore_vector(values, eps: float = EPS):
    """Center and standardize a vector; return normalized values, mean, std."""
    values = np.asarray(values, dtype=float)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    mean_val = float(np.mean(values))
    std_val = float(np.std(values))

    if std_val <= eps:
        return np.zeros_like(values), mean_val, std_val + eps

    return (values - mean_val) / (std_val + eps), mean_val, std_val + eps


def compute_metrics(y_true, y_pred):
    """Common regression metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return {
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else np.nan,
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(np.mean(np.abs(y_true - y_pred))),
    }


def compute_reliability_shrinkage(rho, p_val, n_z, alpha, rho_threshold):
    """
    Reliability shrinkage for the symbolic trend term.

    Unlike v4, shrinkage no longer depends on sigma/|mu|, because SR is only
    asked to explain centered residuals. The factor is based on trend reliability
    and local sample size.
    """
    if p_val >= alpha or abs(rho) < rho_threshold:
        return 0.0

    rho_factor = min(1.0, 0.35 + 0.65 * abs(rho))
    n_factor = np.sqrt(n_z / (n_z + 10.0))

    return float(np.clip(rho_factor * n_factor, 0.0, 1.0))


def build_bias_prediction(df, y_pred_cb, zone_stats, calibration_zones):
    """Apply only zone-bias correction to calibration zones."""
    y_bias = y_pred_cb.copy()

    for zid in calibration_zones:
        if zid not in zone_stats:
            continue
        z_mask = (df["zone_id"].values == zid)
        if not np.any(z_mask):
            continue
        y_bias[z_mask] = y_pred_cb[z_mask] + float(zone_stats[zid]["mu"])

    return y_bias


def multi_seed_evolutionary_consensus(
    X_sr,
    y_sr,
    top_features,
    n_subsets: int = 50,
    subset_size: int = 4,
    niterations: int = 80,
    maxsize: int = 12,
    max_complexity_keep: int = 10,
    base_seed: int = 42,
):
    """
    Random-subspace symbolic regression ensemble.

    Each run samples a subset of features and returns equations up to the
    prescribed complexity. Duplicate equations are removed.
    """
    tf_list = list(top_features)
    rng = np.random.default_rng(base_seed)

    all_candidates = []
    seen_equations = set()

    if len(tf_list) < subset_size:
        subset_size = len(tf_list)

    print(
        f"  [PySR] 随机子空间 SR 集成: "
        f"{n_subsets} 个子集 × {subset_size} 特征 | "
        f"每个子集 {niterations} iterations | 来自 {len(tf_list)} 个候选特征"
    )

    for i in range(n_subsets):
        sub_idx = sorted(rng.choice(len(tf_list), size=subset_size, replace=False))
        sub_features = [tf_list[j] for j in sub_idx]
        safe_sub = [make_pysr_safe(f) for f in sub_features]
        X_sub = X_sr[:, sub_idx]

        parsimony = float(rng.uniform(0.01, 0.05))
        seed = int(rng.integers(0, 100000))

        model = PySRRegressor(
            niterations=niterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sqrt", "square", "abs"],
            maxsize=maxsize,
            parsimony=parsimony,
            verbosity=0,
            deterministic=True,
            random_state=seed,
            parallelism="serial",
        )

        try:
            model.fit(X_sub, y_sr, variable_names=safe_sub)
        except Exception as exc:
            print(f"    [跳过] 子集{i + 1} SR失败: {exc}")
            continue

        valid = model.equations_[model.equations_["complexity"] <= max_complexity_keep]

        for _, row in valid.iterrows():
            eq_str = str(row["equation"])
            if eq_str in seen_equations:
                continue

            seen_equations.add(eq_str)
            eq_idx = model.equations_.index.get_loc(row.name)

            all_candidates.append({
                "equation": eq_str,
                "model": model,
                "index": eq_idx,
                "complexity": int(row["complexity"]),
                "group": f"子集{i + 1}({','.join(sub_features)})",
                "sub_features": sub_features,
                "sub_col_idx": sub_idx,
            })

    if not all_candidates:
        safe_names = [make_pysr_safe(f) for f in tf_list]
        fallback = PySRRegressor(
            niterations=max(40, niterations // 2),
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sqrt", "square", "abs"],
            maxsize=maxsize,
            parsimony=0.05,
            verbosity=0,
            deterministic=True,
            random_state=42,
            parallelism="serial",
        )
        fallback.fit(X_sr, y_sr, variable_names=safe_names)
        all_candidates.append({
            "equation": str(fallback.equations_.iloc[0]["equation"]),
            "model": fallback,
            "index": 0,
            "complexity": int(fallback.equations_.iloc[0]["complexity"]),
            "group": "兜底",
            "sub_features": tf_list,
            "sub_col_idx": list(range(len(tf_list))),
        })

    return all_candidates


def predict_candidate(cand, X_clean, global_idx, top_indices):
    """Predict a candidate equation on given global sample indices."""
    col_idx = cand["sub_col_idx"]
    X_in = X_clean[global_idx][:, top_indices][:, col_idx]
    trend = cand["model"].predict(X_in, index=cand["index"])
    trend = np.clip(
        np.nan_to_num(trend, nan=0.0, posinf=0.0, neginf=0.0),
        -1e3,
        1e3,
    )
    return trend


def evaluate_candidate_by_zone(
    cand,
    df,
    X_clean,
    top_indices,
    y_true,
    y_pred_cb,
    y_pred_bias,
    zone_stats,
    z_list,
    alpha,
    rho_threshold,
    min_delta_rmse,
    w_bound,
    n_perm,
):
    """
    Evaluate one symbolic equation after zone-bias correction.

    Activation requires:
    1. Spearman(|rho|) >= threshold on centered residuals,
    2. permutation p < alpha,
    3. adaptive SR improves RMSE compared with bias-only.
    """
    per_zone = {}
    activated = 0
    rho_sum = 0.0
    delta_rmse_sum = 0.0
    delta_rmse_rel_sum = 0.0

    for zid in z_list:
        z_mask = df["zone_id"].values == zid
        if not np.any(z_mask) or zid not in zone_stats:
            continue

        gi_z = df.loc[z_mask, "global_idx"].values.astype(int)
        trend = predict_candidate(cand, X_clean, gi_z, top_indices)
        trend_norm, trend_mu, trend_sigma = zscore_vector(trend)

        res_norm_z = df.loc[z_mask, "res_norm"].values.astype(float)
        valid = np.isfinite(res_norm_z)
        trend_norm = trend_norm[valid]
        res_norm_z = res_norm_z[valid]

        local_indices = np.where(z_mask)[0][valid]
        yt = y_true[local_indices]
        yb = y_pred_cb[local_indices]
        ybias = y_pred_bias[local_indices]

        if len(yt) < 4:
            rho, p_val = 0.0, 1.0
            w_opt = 0.0
            w_adaptive = 0.0
            lambda_shrink = 0.0
            rmse_base = compute_metrics(yt, yb)["RMSE"] if len(yt) else np.nan
            rmse_bias = compute_metrics(yt, ybias)["RMSE"] if len(yt) else np.nan
            rmse_bias_sr = rmse_bias
            delta_rmse = 0.0
            activated_flag = False
        else:
            rho, p_val = permutation_spearmanr(
                trend_norm,
                res_norm_z,
                n_perm=n_perm,
                seed=42,
            )

            sigma_z = float(zone_stats[zid]["sigma"])
            n_z = int(len(yt))

            def objective(w_trial):
                y_corr = ybias + w_trial * sigma_z * trend_norm
                return mean_squared_error(yt, y_corr)

            if p_val < alpha and abs(rho) >= rho_threshold and np.std(trend_norm) > EPS:
                opt = minimize_scalar(objective, bounds=(-w_bound, w_bound), method="bounded")
                w_opt = float(opt.x)
            else:
                w_opt = 0.0

            lambda_shrink = compute_reliability_shrinkage(
                rho=rho,
                p_val=p_val,
                n_z=n_z,
                alpha=alpha,
                rho_threshold=rho_threshold,
            )
            w_adaptive = w_opt * lambda_shrink

            rmse_base = compute_metrics(yt, yb)["RMSE"]
            rmse_bias = compute_metrics(yt, ybias)["RMSE"]
            rmse_bias_sr = compute_metrics(yt, ybias + w_adaptive * sigma_z * trend_norm)["RMSE"]
            delta_rmse = rmse_bias - rmse_bias_sr
            activated_flag = bool(
                p_val < alpha
                and abs(rho) >= rho_threshold
                and delta_rmse > min_delta_rmse
                and abs(w_adaptive) > EPS
            )

            if not activated_flag:
                w_adaptive = 0.0
                rmse_bias_sr = rmse_bias
                delta_rmse = 0.0

        if activated_flag:
            activated += 1
            rho_sum += abs(rho)
            delta_rmse_sum += delta_rmse
            delta_rmse_rel_sum += delta_rmse / (rmse_bias + EPS)

        per_zone[zid] = {
            "zone_id": int(zid),
            "zone_label": ZONE_LABELS.get(int(zid), f"Zone {zid}"),
            "sample_size": int(len(yt)),
            "rho_centered": float(rho),
            "p_value": float(p_val),
            "trend_mean": float(trend_mu),
            "trend_std": float(trend_sigma),
            "w_opt": float(w_opt),
            "lambda_shrink": float(lambda_shrink),
            "w_adaptive": float(w_adaptive),
            "rmse_base": float(rmse_base),
            "rmse_bias_only": float(rmse_bias),
            "rmse_bias_plus_sr": float(rmse_bias_sr),
            "delta_rmse_from_sr": float(delta_rmse),
            "activated": bool(activated_flag),
        }

    return {
        "activated_zones": activated,
        "rho_sum": rho_sum,
        "delta_rmse_sum": delta_rmse_sum,
        "delta_rmse_rel_sum": delta_rmse_rel_sum,
        "per_zone": per_zone,
    }


def apply_best_candidate(
    best_cand,
    df,
    X_clean,
    top_indices,
    y_true,
    y_pred_cb,
    y_pred_bias,
    zone_stats,
    z_list,
    eval_info,
):
    """Apply the selected symbolic equation only in activated zones."""
    y_final = y_pred_bias.copy()
    per_zone_application = {}

    for zid in z_list:
        z_mask = df["zone_id"].values == zid
        if not np.any(z_mask) or zid not in zone_stats:
            continue

        z_eval = eval_info["per_zone"].get(zid, {})
        activated = bool(z_eval.get("activated", False))
        w_adaptive = float(z_eval.get("w_adaptive", 0.0))
        sigma_z = float(zone_stats[zid]["sigma"])

        gi_z = df.loc[z_mask, "global_idx"].values.astype(int)
        trend = predict_candidate(best_cand, X_clean, gi_z, top_indices)
        trend_norm, trend_mu, trend_sigma = zscore_vector(trend)

        ybias = y_pred_bias[z_mask]
        if activated:
            y_corr = ybias + w_adaptive * sigma_z * trend_norm
        else:
            y_corr = ybias

        y_final[z_mask] = y_corr

        per_zone_application[zid] = {
            **z_eval,
            "trend_mean_applied": float(trend_mu),
            "trend_std_applied": float(trend_sigma),
            "applied_action": "bias_plus_centered_sr" if activated else "bias_only",
            "metrics_base": compute_metrics(y_true[z_mask], y_pred_cb[z_mask]),
            "metrics_bias_only": compute_metrics(y_true[z_mask], y_pred_bias[z_mask]),
            "metrics_bias_plus_sr": compute_metrics(y_true[z_mask], y_corr),
        }

    return y_final, per_zone_application


def process_single_target(
    target,
    pre_dir,
    output_root,
    top_k: int = 10,
    n_subsets: int = 50,
    subset_size: int = 4,
    niterations: int = 80,
    alpha: float = 0.05,
    rho_threshold: float = 0.2,
    min_delta_rmse: float = 0.0,
    w_bound: float = 3.0,
    n_perm: int = 200,
):
    """
    Stage 2 BS-GSR for one target.

    Final decomposition:
        y_final = y_CB + B_z * mu_z + G_z * lambda_z * w_z * sigma_z * f_tilde_SR,z(x)

    B_z is enabled for calibration zones only.
    G_z is enabled only when the centered symbolic trend passes the activation tests.
    Z6 has neither B_z nor G_z and is exported as warning-only.
    """
    target_in = Path(pre_dir) / target
    target_out = output_root / target
    target_out.mkdir(parents=True, exist_ok=True)

    data_path = target_in / f"{target}_normalized_residuals.pkl"
    ranking_path = target_in / f"{target}_rm_smi_ranking.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"未找到预处理数据: {data_path}")

    if not ranking_path.exists():
        raise FileNotFoundError(f"未找到特征排序文件: {ranking_path}")

    with open(data_path, "rb") as f:
        data = pickle.load(f)

    df = data["df"].copy()
    X_clean = data["X_clean"]
    kept_names = data["feature_names"]
    zone_stats = data["zone_stats"]
    warning_stats = data.get("warning_stats", {})
    warning_zones = data.get("warning_zones", WARNING_ZONES.get(target, []))
    calibration_zones = data.get("calibration_zones", [])

    excluded_features = set(data.get("excluded_features", []))
    ranked_df = pd.read_csv(ranking_path)

    top_features = [
        f for f in ranked_df["feature"].tolist()
        if f not in excluded_features and f in kept_names
    ][:top_k]

    if not top_features:
        raise ValueError(f"{target}: 没有可用的 top_features，请检查 ranking 文件。")

    top_indices = [kept_names.index(f) for f in top_features]

    print("\n==========================================================================================")
    print(
        f" 开始处理目标: {target} | "
        f"校准 Zones: {calibration_zones} | "
        f"预警 Zones: {warning_zones} | "
        f"输入 SR 特征: {top_features}"
    )
    print("==========================================================================================")

    y_pred_cb = df["y_pred"].values.astype(float).copy()
    y_true = df["y_true"].values.astype(float)
    y_pred_bias = build_bias_prediction(df, y_pred_cb, zone_stats, calibration_zones)
    y_final = y_pred_bias.copy()

    active_patches = {}

    for batch_name, z_list in BATCH_CONFIG.get(target, {}).items():
        batch_mask = df["zone_id"].isin(z_list)
        batch_df = df[batch_mask].copy()

        if len(batch_df) == 0:
            print(f"  [跳过] {target} - {batch_name} 没有校准样本。")
            continue

        gi = batch_df["global_idx"].values.astype(int)
        X_sr = X_clean[gi][:, top_indices]
        y_sr = batch_df["res_norm"].values.astype(float)

        valid_mask = np.isfinite(y_sr)
        X_sr = X_sr[valid_mask]
        y_sr = y_sr[valid_mask]

        if len(y_sr) < 4:
            print(f"  [跳过] {target} - {batch_name} 有效 SR 样本过少: n={len(y_sr)}")
            continue

        candidates = multi_seed_evolutionary_consensus(
            X_sr=X_sr,
            y_sr=y_sr,
            top_features=top_features,
            n_subsets=n_subsets,
            subset_size=subset_size,
            niterations=niterations,
            base_seed=42,
        )

        print(f"  [公式选择] 从 {len(candidates)} 个候选公式中按 bias-only 后增益筛选最优...")

        scored_candidates = []
        candidate_records = []

        for cand in candidates:
            eval_info = evaluate_candidate_by_zone(
                cand=cand,
                df=df,
                X_clean=X_clean,
                top_indices=top_indices,
                y_true=y_true,
                y_pred_cb=y_pred_cb,
                y_pred_bias=y_pred_bias,
                zone_stats=zone_stats,
                z_list=z_list,
                alpha=alpha,
                rho_threshold=rho_threshold,
                min_delta_rmse=min_delta_rmse,
                w_bound=w_bound,
                n_perm=n_perm,
            )

            eq_display = restore_feature_names(cand["equation"], cand["sub_features"])

            scored_candidates.append((cand, eval_info, eq_display))

            candidate_records.append({
                "equation": eq_display,
                "complexity": cand["complexity"],
                "activated_zones": eval_info["activated_zones"],
                "rho_sum_centered": eval_info["rho_sum"],
                "delta_rmse_sum_from_sr": eval_info["delta_rmse_sum"],
                "delta_rmse_rel_sum_from_sr": eval_info["delta_rmse_rel_sum"],
                "group": cand.get("group", "?"),
                "sub_features": ",".join(cand["sub_features"]),
            })

            for zid, zinfo in eval_info["per_zone"].items():
                candidate_records[-1][f"Z{zid}_activated"] = zinfo["activated"]
                candidate_records[-1][f"Z{zid}_rho"] = zinfo["rho_centered"]
                candidate_records[-1][f"Z{zid}_p"] = zinfo["p_value"]
                candidate_records[-1][f"Z{zid}_delta_RMSE_SR"] = zinfo["delta_rmse_from_sr"]
                candidate_records[-1][f"Z{zid}_w_adaptive"] = zinfo["w_adaptive"]

        candidate_df = (
            pd.DataFrame(candidate_records)
            .sort_values(
                by=[
                    "activated_zones",
                    "delta_rmse_rel_sum_from_sr",
                    "rho_sum_centered",
                    "complexity",
                ],
                ascending=[False, False, False, True],
            )
            .reset_index(drop=True)
        )

        candidate_csv = target_out / f"{target}_candidate_equations_bias_separated.csv"
        candidate_df.to_csv(candidate_csv, index=False)
        print(f"  [公式选择] 候选公式评分已保存至: {candidate_csv.name}")

        def candidate_sort_key(item):
            cand, eval_info, _ = item
            return (
                eval_info["activated_zones"],
                eval_info["delta_rmse_rel_sum"],
                eval_info["rho_sum"],
                -cand["complexity"],
            )

        best_cand, best_eval, best_eq = max(scored_candidates, key=candidate_sort_key)

        print(
            f"  [公式选择] 最优公式覆盖 {best_eval['activated_zones']}/{len(z_list)} 个 zone "
            f"| 来源: {best_cand.get('group', '?')} "
            f"| 复杂度: {best_cand['complexity']} "
            f"| f(x) = {best_eq}"
        )

        y_batch_final, per_zone_application = apply_best_candidate(
            best_cand=best_cand,
            df=df,
            X_clean=X_clean,
            top_indices=top_indices,
            y_true=y_true,
            y_pred_cb=y_pred_cb,
            y_pred_bias=y_pred_bias,
            zone_stats=zone_stats,
            z_list=z_list,
            eval_info=best_eval,
        )

        for zid in z_list:
            z_mask = df["zone_id"].values == zid
            if np.any(z_mask):
                y_final[z_mask] = y_batch_final[z_mask]

        active_patches[batch_name] = {
            "eq": best_eq,
            "equation_raw": best_cand["equation"],
            "sub_features": best_cand["sub_features"],
            "complexity": best_cand["complexity"],
            "zone_params": per_zone_application,
        }

    # Print calibration panel.
    print("\n  [Bias-separated 局部校准反馈面板]")
    print("  公式: y_final = y_CB + B_z*mu_z + G_z*lambda_z*w_z*sigma_z*f_tilde_SR,z(x)")

    for batch_name, info in active_patches.items():
        print(f"\n  机制区间: {batch_name} | 中心化符号趋势项: f(x) = {info['eq']}")
        print(
            f"  {'Zone':<27} | {'n':>4} | {'rho':>7} | {'p':>7} | "
            f"{'RMSE_base':>10} | {'RMSE_bias':>10} | {'RMSE_bias+SR':>13} | "
            f"{'SR增益':>8} | {'lambda':>7} | {'w_adapt':>8} | {'action':>20}"
        )
        print("  " + "-" * 140)

        for zid, stats in info["zone_params"].items():
            z_label = ZONE_LABELS.get(zid, f"Zone {zid}")
            print(
                f"  Z{zid} {z_label:<24} | "
                f"{stats['sample_size']:>4} | "
                f"{stats['rho_centered']:>7.3f} | "
                f"{stats['p_value']:>7.4f} | "
                f"{stats['rmse_base']:>10.3f} | "
                f"{stats['rmse_bias_only']:>10.3f} | "
                f"{stats['rmse_bias_plus_sr']:>13.3f} | "
                f"{stats['delta_rmse_from_sr']:>8.3f} | "
                f"{stats['lambda_shrink']:>7.3f} | "
                f"{stats['w_adaptive']:>8.3f} | "
                f"{stats['applied_action']:>20}"
            )

    print("\n  [宏观消融结果]")
    print(
        f"  Base R²={compute_metrics(y_true, y_pred_cb)['R2']:.4f}, "
        f"Bias-only R²={compute_metrics(y_true, y_pred_bias)['R2']:.4f}, "
        f"Bias+SR R²={compute_metrics(y_true, y_final)['R2']:.4f}"
    )

    # Warning-only zone summary.
    warning_records = []

    for wz in warning_zones:
        z_mask = df["zone_id"].values == wz
        if not np.any(z_mask):
            continue

        yt = y_true[z_mask]
        yp = y_pred_cb[z_mask]
        res = yt - yp

        record = {
            "target": target,
            "zone_id": int(wz),
            "zone_label": ZONE_LABELS.get(int(wz), f"Zone {wz}"),
            "role": "warning_only",
            "sample_size": int(np.sum(z_mask)),
            "mean_residual": float(np.mean(res)),
            "std_residual": float(np.std(res)),
            "MAE_base": float(np.mean(np.abs(res))),
            "RMSE_base": float(np.sqrt(mean_squared_error(yt, yp))),
            "R2_base": float(r2_score(yt, yp)) if len(yt) >= 2 else np.nan,
            "action": (
                "No bias correction and no SR compensation; "
                "use risk warning and uncertainty quantification."
            ),
        }

        warning_records.append(record)
        print(
            f"  ⚠ Z{wz} {record['zone_label']} 被标记为 warning-only: "
            f"n={record['sample_size']}, MAE={record['MAE_base']:.3f}, "
            f"RMSE={record['RMSE_base']:.3f}"
        )

    if warning_records:
        pd.DataFrame(warning_records).to_csv(
            target_out / f"{target}_warning_zone_summary.csv",
            index=False,
        )

    # Final dataframe with all three prediction columns.
    final_df = df.copy()
    final_df["is_warning_zone"] = final_df["zone_id"].isin(warning_zones)
    final_df["is_calibration_zone"] = final_df["zone_id"].isin(calibration_zones)

    final_df["calibration_action"] = "base_prediction"
    final_df.loc[final_df["is_calibration_zone"], "calibration_action"] = "bias_only_or_bias_plus_sr"
    final_df.loc[final_df["is_warning_zone"], "calibration_action"] = "warning_only_no_compensation"

    # Mark the activated SR zones explicitly.
    final_df["sr_trend_activated"] = False
    final_df["local_calibration_stage"] = "base"
    final_df.loc[final_df["is_calibration_zone"], "local_calibration_stage"] = "bias_only"
    final_df.loc[final_df["is_warning_zone"], "local_calibration_stage"] = "warning_only"

    for _, info in active_patches.items():
        for zid, stats in info["zone_params"].items():
            z_mask = final_df["zone_id"] == zid
            if stats.get("activated", False):
                final_df.loc[z_mask, "sr_trend_activated"] = True
                final_df.loc[z_mask, "local_calibration_stage"] = "bias_plus_centered_sr"

    final_df["y_pred_base"] = y_pred_cb
    final_df["y_pred_bias_only"] = y_pred_bias
    final_df["y_pred_bias_plus_sr"] = y_final
    final_df["y_pred_compensated"] = y_final

    final_df["residual_base"] = y_true - y_pred_cb
    final_df["residual_bias_only"] = y_true - y_pred_bias
    final_df["residual_bias_plus_sr"] = y_true - y_final
    final_df["residual_compensated"] = y_true - y_final

    final_df.to_csv(target_out / f"{target}_compensated_results_bias_separated.csv", index=False)

    with open(target_out / f"{target}_compensated_residuals_bias_separated.pkl", "wb") as f:
        pickle.dump({
            "df": final_df,
            "feature_names": kept_names,
            "active_patches": active_patches,
            "warning_zones": warning_zones,
            "warning_stats": warning_stats,
            "calibration_zones": calibration_zones,
            "method": "Bias-separated gated symbolic regression",
            "formula": (
                "y_final=y_CB+B_z*mu_z+G_z*lambda_z*w_z*sigma_z*f_tilde_SR,z(x); "
                "Z6 is warning-only."
            ),
        }, f)

    # Metrics summary: base vs bias-only vs bias+SR.
    metrics_records = []

    def add_metrics_record(subset_name, mask, zones=None):
        if not np.any(mask):
            return
        rec = {
            "target": target,
            "subset": subset_name,
            "zones": "" if zones is None else ",".join(map(str, zones)),
            "n": int(np.sum(mask)),
        }
        for prefix, pred in [
            ("base", y_pred_cb),
            ("bias_only", y_pred_bias),
            ("bias_plus_sr", y_final),
        ]:
            m = compute_metrics(y_true[mask], pred[mask])
            rec.update({f"{prefix}_{k}": v for k, v in m.items()})
        rec["delta_RMSE_bias_vs_base"] = rec["base_RMSE"] - rec["bias_only_RMSE"]
        rec["delta_RMSE_sr_vs_bias"] = rec["bias_only_RMSE"] - rec["bias_plus_sr_RMSE"]
        rec["delta_RMSE_total_vs_base"] = rec["base_RMSE"] - rec["bias_plus_sr_RMSE"]
        rec["delta_R2_bias_vs_base"] = rec["bias_only_R2"] - rec["base_R2"]
        rec["delta_R2_sr_vs_bias"] = rec["bias_plus_sr_R2"] - rec["bias_only_R2"]
        rec["delta_R2_total_vs_base"] = rec["bias_plus_sr_R2"] - rec["base_R2"]
        metrics_records.append(rec)

    all_mask = np.ones(len(final_df), dtype=bool)
    calib_mask = final_df["zone_id"].isin(calibration_zones).values
    warn_mask = final_df["zone_id"].isin(warning_zones).values

    add_metrics_record("all_samples", all_mask)
    add_metrics_record("calibration_zones_only", calib_mask, calibration_zones)
    add_metrics_record("warning_zones_only", warn_mask, warning_zones)

    for zid in calibration_zones:
        z_mask = (final_df["zone_id"].values == zid)
        add_metrics_record(f"zone_{zid}", z_mask, [zid])

    for wz in warning_zones:
        z_mask = (final_df["zone_id"].values == wz)
        add_metrics_record(f"warning_zone_{wz}", z_mask, [wz])

    metrics_df = pd.DataFrame(metrics_records)
    metrics_df.to_csv(target_out / f"{target}_compensation_metrics_ablation.csv", index=False)

    # Per-zone application table for the chosen equations.
    zone_records = []
    for batch_name, info in active_patches.items():
        for zid, stats in info["zone_params"].items():
            zone_records.append({
                "target": target,
                "batch_name": batch_name,
                "equation": info["eq"],
                "zone_id": int(zid),
                **{
                    k: v for k, v in stats.items()
                    if not isinstance(v, dict)
                },
            })

    if zone_records:
        pd.DataFrame(zone_records).to_csv(
            target_out / f"{target}_selected_equation_zone_effects.csv",
            index=False,
        )

    print(f"  ✓ 修正数据集已保存至: {target_out / f'{target}_compensated_residuals_bias_separated.pkl'}")
    print(f"  ✓ 消融指标已保存至: {target_out / f'{target}_compensation_metrics_ablation.csv'}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target",
        type=str,
        default="all",
        choices=["Rm", "Re", "A", "Ak", "all"],
    )
    parser.add_argument("--pre-dir", type=str, default=str(Path(__file__).resolve().parent / 'sr_patch_preprocessing_v5_bias_separated'))
    parser.add_argument("--output-dir", type=str, default=str(Path(__file__).resolve().parent / 'sr_patch_results_v5_bias_separated'))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--n-subsets", type=int, default=50)
    parser.add_argument("--subset-size", type=int, default=4)
    parser.add_argument("--niterations", type=int, default=80)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--rho-threshold", type=float, default=0.2)
    parser.add_argument("--min-delta-rmse", type=float, default=0.0)
    parser.add_argument("--w-bound", type=float, default=3.0)
    parser.add_argument("--n-perm", type=int, default=200)

    args = parser.parse_args()

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    targets = ["Rm", "Re", "A", "Ak"] if args.target == "all" else [args.target]

    if not PYSR_AVAILABLE:
        print("✗ 错误: 当前环境未检测到 PySR 包，请先安装 PySR 后再运行。")
        return

    print("🚀 启动阶段二：Bias-separated Gated Symbolic Regression 管线...")

    for target in targets:
        try:
            process_single_target(
                target=target,
                pre_dir=args.pre_dir,
                output_root=output_root,
                top_k=args.top_k,
                n_subsets=args.n_subsets,
                subset_size=args.subset_size,
                niterations=args.niterations,
                alpha=args.alpha,
                rho_threshold=args.rho_threshold,
                min_delta_rmse=args.min_delta_rmse,
                w_bound=args.w_bound,
                n_perm=args.n_perm,
            )
        except Exception as exc:
            print(f"  ✗ 处理目标 {target} 失败，原因: {exc}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
