# residual_normalization_and_collinearity_v5_bias_separated.py
# v5: Bias-separated local calibration preprocessing.
#     Z6 is treated as an overlap-risk warning zone and is excluded from
#     both zone-bias compensation and symbolic-regression trend calibration.

import argparse
import pickle
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression

matplotlib.use("Agg")
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

try:
    import seaborn as sns  # noqa: F401
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

warnings.filterwarnings("ignore")


# ════════════════════════════════════════════════════════════════
# Static configuration and domain priors
# ════════════════════════════════════════════════════════════════

BATCH_CONFIG = {
    # Z6 = High Delta_TI x Short tau.
    # It is an overlap-risk warning zone and is not used for local calibration.
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

BASE_FEATURE_NAMES = [
    "C", "Si", "Mn", "Cr", "V", "Al", "Ni", "Mo", "Cu", "W",
    "Z", "Ve", "chiP", "delta_b", "delta_f", "Delta_Fe", "Delta_C",
    "Ta", "Ti", "ti", "Delta_TA", "Delta_TI", "tau",
    "Ms", "Tt", "tt", "sph", "tmp",
]
BASE_FEATURE_INDEX = {name: idx for idx, name in enumerate(BASE_FEATURE_NAMES)}

EXP2_BASE_FEATURES = [
    "C", "Si", "Mn", "Cr", "V", "Al", "Ni", "Mo", "Cu", "W",
    "Z", "Ve", "chiP", "delta_b", "delta_f", "Delta_Fe", "Delta_C",
    "Ta", "Ti", "Delta_TA", "Delta_TI", "tau", "Tt", "tt", "sph", "tmp",
]

EXP2_INTERACTION_FEATURES = [
    "Ceq",
    "Ta_Ti_diff",
    "C_DTI",
    "DC_Ti",
    "C_Cr",
]

COLLINEARITY_REMOVE_GLOBAL = ["C", "tmp", "chiP"]
COLLINEARITY_REMOVE_STRENGTH = ["delta_f", "Delta_C"]
COLLINEARITY_REMOVE_DUCTILITY = ["delta_b", "Delta_Fe"]

EPS = 1e-6


def compute_interaction_features(X_28: np.ndarray) -> dict:
    """Construct physically motivated interaction features."""
    feat_dict = {}

    feat_dict["Ceq"] = (
        X_28[:, 0]
        + X_28[:, 2] / 6
        + (X_28[:, 3] + X_28[:, 7] + X_28[:, 4]) / 5
        + (X_28[:, 6] + X_28[:, 8]) / 15
    )

    feat_dict["Ta_Ti_diff"] = X_28[:, 17] - X_28[:, 18]
    feat_dict["C_DTI"] = X_28[:, 0] * X_28[:, 21]
    feat_dict["DC_Ti"] = X_28[:, 16] * X_28[:, 18]
    feat_dict["C_Cr"] = X_28[:, 0] * X_28[:, 3]

    return feat_dict


def run_stratified_bootstrap_rm_smi(
    df: pd.DataFrame,
    X_clean: np.ndarray,
    feature_names: list,
    active_zones: list,
    n_bootstrap: int = 100,
    top_k_threshold: int = 10,
):
    """
    Stratified perturbation-resampling feature stability selection.

    The target is the centered and standardized residual, res_norm.
    Z6 warning-only samples are not included.
    """
    n_features = len(feature_names)
    inclusion_counts = {f: 0 for f in feature_names}

    zone_indices = {
        zid: df[df["zone_id"] == zid].index.tolist()
        for zid in active_zones
    }

    print(f"    开始分层扰动评估，进行 {n_bootstrap} 轮 Stratified Bootstrap 抽样...")

    rng = np.random.default_rng(42)

    for round_idx in range(n_bootstrap):
        bootstrap_df_indices = []

        for _, idx_list in zone_indices.items():
            if len(idx_list) == 0:
                continue

            sample_size = max(int(len(idx_list) * 0.8), 2)
            sampled_idx = rng.choice(idx_list, size=sample_size, replace=True)
            bootstrap_df_indices.extend(sampled_idx)

        if not bootstrap_df_indices:
            continue

        bootstrap_df = df.loc[bootstrap_df_indices].reset_index(drop=True)

        feature_scores = np.zeros(n_features)
        total_samples = 0

        for zid in active_zones:
            z_mask = bootstrap_df["zone_id"] == zid

            if not np.any(z_mask):
                continue

            n_z = int(np.sum(z_mask))
            total_samples += n_z

            gi = bootstrap_df.loc[z_mask, "global_idx"].values.astype(int)
            X_z = X_clean[gi]
            y_signed = bootstrap_df.loc[z_mask, "res_norm"].values
            y_abs = np.abs(y_signed)

            try:
                mi_z = mutual_info_regression(
                    X_z,
                    y_abs,
                    random_state=42 + round_idx,
                )
            except Exception:
                mi_z = np.zeros(n_features)

            rho_z = np.zeros(n_features)
            for i in range(n_features):
                rho, _ = spearmanr(X_z[:, i], y_signed)
                rho_z[i] = abs(rho) if not np.isnan(rho) else 0.0

            score_z = 0.6 * rho_z + 0.4 * mi_z
            feature_scores += score_z * n_z

        feature_scores /= total_samples + EPS
        round_rank = np.argsort(feature_scores)[::-1]

        for rank_pos in range(min(top_k_threshold, len(round_rank))):
            feature_name = feature_names[round_rank[rank_pos]]
            inclusion_counts[feature_name] += 1

    results = [
        {
            "feature": f,
            "inclusion_frequency": inclusion_counts[f] / max(n_bootstrap, 1),
        }
        for f in feature_names
    ]

    return (
        pd.DataFrame(results)
        .sort_values(by="inclusion_frequency", ascending=False)
        .reset_index(drop=True)
    )


def execute_preprocessing(
    target: str,
    data_dir: Path,
    res_dir: Path,
    output_dir: Path,
    n_bootstrap: int = 100,
):
    """
    Stage 1 preprocessing.

    1. Read LOPZO residuals.
    2. Construct 31-dimensional physical feature set.
    3. Remove collinear/redundant variables by domain priors.
    4. Estimate zone-bias mu_z only for calibration zones.
    5. Build centered standardized residual res_norm for symbolic trend modeling.
    6. Save Z6 as warning-only, without normalization or compensation.
    """
    X28_path = data_dir / "X_28dim.npy"
    pkl_path = res_dir / f"{target}_lopzo_residuals.pkl"
    csv_path = res_dir / f"{target}_lopzo_predictions.csv"

    if not X28_path.exists():
        raise FileNotFoundError(f"未找到特征文件: {X28_path}")

    X28 = np.load(X28_path)
    if csv_path.exists():
        source_path = csv_path
        df = pd.read_csv(csv_path)
    elif pkl_path.exists():
        source_path = pkl_path
        with open(pkl_path, "rb") as f:
            raw_data = pickle.load(f)
        df = (raw_data["df"] if isinstance(raw_data, dict) else raw_data).copy()
    else:
        raise FileNotFoundError(f"未找到残差文件: {csv_path} 或 {pkl_path}")

    required_cols = {"zone_id", "global_idx", "residual", "y_true", "y_pred"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"{source_path} 缺少必要列: {sorted(missing_cols)}")

    # Build 31-dimensional features.
    base_indices = [BASE_FEATURE_INDEX[name] for name in EXP2_BASE_FEATURES]
    X_base = X28[:, base_indices]

    interactions = compute_interaction_features(X28)
    X_interactions = np.hstack([
        interactions[name].reshape(-1, 1)
        for name in EXP2_INTERACTION_FEATURES
    ])

    X31 = np.hstack([X_base, X_interactions])
    all_names = EXP2_BASE_FEATURES + EXP2_INTERACTION_FEATURES

    # Domain-prior dimensionality reduction.
    remove_set = set(COLLINEARITY_REMOVE_GLOBAL)

    if target in ["Rm", "Re"]:
        remove_set.update(COLLINEARITY_REMOVE_STRENGTH)
    elif target in ["A", "Ak"]:
        remove_set.update(COLLINEARITY_REMOVE_DUCTILITY)

    kept_indices = [
        i for i, name in enumerate(all_names)
        if name not in remove_set
    ]

    kept_names = [all_names[i] for i in kept_indices]
    X_clean = X31[:, kept_indices]

    regime_config = BATCH_CONFIG.get(target, {})
    if not regime_config:
        print(f"目标 {target} 未设置 BATCH_CONFIG，跳过。")
        return

    regime_name = list(regime_config.keys())[0]
    active_zones = regime_config[regime_name]
    warning_zones = WARNING_ZONES.get(target, [])

    df["is_warning_zone"] = df["zone_id"].isin(warning_zones)
    df["calibration_role"] = "other"
    df.loc[df["zone_id"].isin(active_zones), "calibration_role"] = "calibration"
    df.loc[df["zone_id"].isin(warning_zones), "calibration_role"] = "warning_only"

    print("\n======================================================================")
    print(
        f" 目标: {target} | 工艺区段: {regime_name} | "
        f"校准 Zones: {active_zones} | 预警 Zones: {warning_zones}"
    )
    print("======================================================================")
    print(f"  ✓ 领域先验剔除特征: {sorted(remove_set)}")
    print(f"  ✓ 剩余候选物理特征数: {len(kept_names)}")

    # Bias-separated residual normalization.
    df["residual_raw"] = df["residual"].astype(float)
    df["zone_bias_mu"] = np.nan
    df["zone_residual_sigma"] = np.nan
    df["res_centered"] = np.nan
    df["res_norm"] = np.nan

    zone_stats = {}

    for zid in active_zones:
        z_mask = df["zone_id"] == zid

        if not np.any(z_mask):
            continue

        residuals = df.loc[z_mask, "residual_raw"].values.astype(float)
        mu = float(np.mean(residuals))
        sigma = float(np.std(residuals) + EPS)
        centered = residuals - mu
        res_norm = centered / sigma

        df.loc[z_mask, "zone_bias_mu"] = mu
        df.loc[z_mask, "zone_residual_sigma"] = sigma
        df.loc[z_mask, "res_centered"] = centered
        df.loc[z_mask, "res_norm"] = res_norm

        zone_stats[zid] = {
            "mu": mu,
            "sigma": sigma,
            "sample_size": int(len(residuals)),
            "role": "calibration",
            "zone_label": ZONE_LABELS.get(zid, f"Zone {zid}"),
            "message": (
                "Bias mu_z is estimated for this calibration zone; "
                "symbolic regression is trained only on centered residuals."
            ),
        }

        print(
            f"  ✓ Zone {zid} 校准残差分解完成: "
            f"样本数={len(residuals):<3} | "
            f"Mu={mu:+.3f} | Sigma={sigma:.3f} | SR目标=res_norm"
        )

    warning_stats = {}

    for zid in warning_zones:
        z_mask = df["zone_id"] == zid

        if not np.any(z_mask):
            continue

        residuals = df.loc[z_mask, "residual_raw"].values.astype(float)

        warning_stats[zid] = {
            "mu": float(np.mean(residuals)),
            "sigma": float(np.std(residuals) + EPS),
            "sample_size": int(len(residuals)),
            "role": "warning_only",
            "zone_label": ZONE_LABELS.get(zid, f"Zone {zid}"),
            "message": (
                "Z6 is treated as an overlap-risk warning zone and excluded "
                "from both zone-bias and symbolic-trend calibration."
            ),
        }

        print(
            f"  ⚠ Zone {zid} 风险叠加区: "
            f"样本数={len(residuals):<3} | "
            f"Mu={np.mean(residuals):+.3f} | "
            f"Sigma={np.std(residuals) + EPS:.3f} | 不参与校准"
        )

    print("\n  [稳定性选择] 启动分层扰动重采样并计算特征入选概率...")

    ranked_df = run_stratified_bootstrap_rm_smi(
        df=df,
        X_clean=X_clean,
        feature_names=kept_names,
        active_zones=active_zones,
        n_bootstrap=n_bootstrap,
        top_k_threshold=10,
    )

    print(f"    {'排名':<4} | {'特征名':<15} | {'稳定性入选概率':^30}")
    print(f"    {'-' * 60}")

    for idx, row in ranked_df.head(10).iterrows():
        print(
            f"    Top-{idx + 1:<2} | "
            f"{row['feature']:<15} | "
            f"{row['inclusion_frequency']:^30.1%}"
        )

    output_target_dir = output_dir / target
    output_target_dir.mkdir(parents=True, exist_ok=True)

    norm_pkl_path = output_target_dir / f"{target}_normalized_residuals.pkl"

    with open(norm_pkl_path, "wb") as f:
        pickle.dump({
            "df": df,
            "zone_stats": zone_stats,
            "warning_stats": warning_stats,
            "calibration_zones": active_zones,
            "warning_zones": warning_zones,
            "regime_name": regime_name,
            "X_clean": X_clean,
            "feature_names": kept_names,
            "excluded_features": list(remove_set),
            "normalization": (
                "res_norm=(residual_raw-zone_bias_mu)/(zone_residual_sigma+eps); "
                "Z6 warning-only has no res_norm."
            ),
        }, f)

    ranked_df.to_csv(output_target_dir / f"{target}_rm_smi_ranking.csv", index=False)

    zone_summary = pd.DataFrame([
        {"target": target, "zone_id": zid, **stats}
        for zid, stats in zone_stats.items()
    ])
    if len(zone_summary) > 0:
        zone_summary.to_csv(output_target_dir / f"{target}_calibration_zone_stats.csv", index=False)

    warning_summary = pd.DataFrame([
        {"target": target, "zone_id": zid, **stats}
        for zid, stats in warning_stats.items()
    ])
    if len(warning_summary) > 0:
        warning_summary.to_csv(output_target_dir / f"{target}_warning_zone_stats.csv", index=False)

    print(f"\n  ✓ 中心化归一残差与特征数据已保存至: {norm_pkl_path}")
    print(f"  ✓ 特征稳定性排序已导出至: {target}_rm_smi_ranking.csv")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target",
        type=str,
        default="all",
        choices=["Rm", "Re", "A", "Ak", "all"],
    )
    parser.add_argument("--data-dir", type=str, default=str(Path(__file__).resolve().parent.parent / '01_model_training_and_evaluation' / 'data_clean'))
    parser.add_argument("--res-dir", type=str, default=str(Path(__file__).resolve().parent.parent / '01_model_training_and_evaluation' / 'unified_results_clean' / 'residuals'))
    parser.add_argument("--output-dir", type=str, default=str(Path(__file__).resolve().parent / 'sr_patch_preprocessing_v5_bias_separated'))
    parser.add_argument("--n-bootstrap", type=int, default=100)

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    res_dir = Path(args.res_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    targets = ["Rm", "Re", "A", "Ak"] if args.target == "all" else [args.target]

    for target in targets:
        execute_preprocessing(
            target=target,
            data_dir=data_dir,
            res_dir=res_dir,
            output_dir=output_dir,
            n_bootstrap=args.n_bootstrap,
        )


if __name__ == "__main__":
    main()
