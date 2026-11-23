#!/usr/bin/env python3
"""Plot UQ scatter plots with error bars in 1x4 layout.

Expected input layout under ``--patched-dir`` and ``--uq-dir``::

    sr_patch_results_v5_bias_separated/
        Rm/Rm_compensated_residuals_bias_separated.pkl
        Re/Re_compensated_residuals_bias_separated.pkl
        A/A_compensated_residuals_bias_separated.pkl
        Ak/Ak_compensated_residuals_bias_separated.pkl
    uq_results_v5_bias_separated/
        Rm/uq_zone_table.json
        Re/uq_zone_table.json
        A/uq_zone_table.json
        Ak/uq_zone_table.json
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# =========================================================
# Global style -
# =========================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 13.0,
    "axes.titlesize": 14.0,
    "axes.labelsize": 13.0,
    "xtick.labelsize": 13.0,
    "ytick.labelsize": 13.0,
    "legend.fontsize": 11.0,
    "axes.linewidth": 0.65,
    "xtick.major.width": 0.55,
    "ytick.major.width": 0.55,
    "xtick.major.size": 2.4,
    "ytick.major.size": 2.4,
    "figure.dpi": 200,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
})

TARGETS = ["Rm", "Re", "A", "Ak"]

ROLE_COLORS = {
    "calibration": "#5FA6D9",
    "other": "#8FA4AA",
    "warning": "#B89A7A",
}

ROLE_LABELS = {
    "calibration": "Calibration zone",
    "other": "Normal zone",
    "warning": "Warning zone",
}


# =========================================================
def load_pkl(pkl_path: Path):
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    df = obj["df"] if isinstance(obj, dict) else obj
    return df.copy(), obj


def load_uq_json(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def calc_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot != 0 else np.nan


def calc_mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def calc_rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# =========================================================
def save_metrics_csv(all_metrics, output_dir):
    """保存所有指标到一个CSV文件 (全部保留2位小数)"""
    output_dir.mkdir(exist_ok=True, parents=True)
    csv_path = output_dir / "metrics_summary.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "target", "level", "group", "sample_count",
            "r2", "mae", "rmse", "q_final"
        ])

        for metric in all_metrics:
            target = metric["target"]

            writer.writerow([
                target, "global", "all",
                metric["total_samples"],
                f"{metric['global']['r2']:.2f}",
                f"{metric['global']['mae']:.2f}",
                f"{metric['global']['rmse']:.2f}",
                ""
            ])

            for role, role_data in metric["by_role"].items():
                writer.writerow([
                    target, "by_role", role,
                    role_data["sample_count"],
                    f"{role_data['r2']:.2f}",
                    f"{role_data['mae']:.2f}",
                    f"{role_data['rmse']:.2f}",
                    ""
                ])

            for zone_id, zone_data in sorted(metric["by_zone"].items()):
                writer.writerow([
                    target, "by_zone", zone_data["zone_label"],
                    zone_data["sample_count"],
                    f"{zone_data['r2']:.2f}" if not np.isnan(zone_data['r2']) else "NaN",
                    f"{zone_data['mae']:.2f}" if not np.isnan(zone_data['mae']) else "NaN",
                    f"{zone_data['rmse']:.2f}" if not np.isnan(zone_data['rmse']) else "NaN",
                    f"{zone_data['q_final']:.2f}" if not np.isnan(zone_data['q_final']) else "NaN"
                ])

    print(f" Metrics saved to: {csv_path}")
    return csv_path


# =========================================================
def collect_metrics(patched_dir, uq_dir, target):
    """收集单个目标变量的所有指标数据"""
    pkl_path = patched_dir / target / f"{target}_compensated_residuals_bias_separated.pkl"
    json_path = uq_dir / target / "uq_zone_table.json"

    df, _ = load_pkl(pkl_path)
    uq = load_uq_json(json_path)

    df["calibration_role"] = df["calibration_role"].replace({"warning_only": "warning"})
    pred_col = "y_pred_bias_plus_sr"

    q_vals = []
    for zid in df["zone_id"].values:
        q_info = uq["zones"].get(str(int(zid)))
        q_vals.append(np.nan if not q_info else q_info["q_final"])

    df["q_final"] = q_vals
    df = df.dropna(subset=["q_final"])

    x = df["y_true"].values
    y = df[pred_col].values

    r2_global = calc_r2(x, y)
    mae_global = calc_mae(x, y)
    rmse_global = calc_rmse(x, y)

    zone_metrics = {}
    for zone_id in df["zone_id"].unique():
        zone_df = df[df["zone_id"] == zone_id]
        z_x = zone_df["y_true"].values
        z_y = zone_df[pred_col].values

        zone_info = uq["zones"].get(str(int(zone_id)), {})
        zone_label = zone_df["zone_label"].iloc[0] if len(zone_df) > 0 else f"Zone_{zone_id}"

        if len(z_x) >= 2:
            zone_metrics[int(zone_id)] = {
                "zone_id": int(zone_id),
                "zone_label": zone_label,
                "sample_count": len(zone_df),
                "q_final": zone_info.get("q_final", np.nan),
                "r2": calc_r2(z_x, z_y),
                "mae": calc_mae(z_x, z_y),
                "rmse": calc_rmse(z_x, z_y),
            }

    role_metrics = {}
    for role in ["calibration", "other", "warning"]:
        role_df = df[df["calibration_role"] == role]
        if len(role_df) >= 2:
            r_x = role_df["y_true"].values
            r_y = role_df[pred_col].values
            role_metrics[role] = {
                "role": role,
                "sample_count": len(role_df),
                "r2": calc_r2(r_x, r_y),
                "mae": calc_mae(r_x, r_y),
                "rmse": calc_rmse(r_x, r_y),
            }

    return {
        "target": target,
        "total_samples": len(df),
        "global": {
            "r2": r2_global,
            "mae": mae_global,
            "rmse": rmse_global,
        },
        "by_role": role_metrics,
        "by_zone": zone_metrics,
    }


# =========================================================
def plot_combined_calibrated_scatter(patched_dir, uq_dir, output_dir):

    width_mm = 340.0
    height_mm = 110.0

    fig, axes = plt.subplots(
        1, 4,
        figsize=(width_mm / 25.4, height_mm / 25.4),
        facecolor="white",
    )
    fig.subplots_adjust(
        left=0.080,
        right=0.985,
        bottom=0.180,
        top=0.900,
        wspace=0.18,
        hspace=0.0,
    )

    all_metrics = []

    for idx, (ax, target) in enumerate(zip(axes.flat, TARGETS)):
        pkl_path = patched_dir / target / f"{target}_compensated_residuals_bias_separated.pkl"
        json_path = uq_dir / target / "uq_zone_table.json"

        df, _ = load_pkl(pkl_path)
        uq = load_uq_json(json_path)

        df["calibration_role"] = df["calibration_role"].replace({"warning_only": "warning"})
        pred_col = "y_pred_bias_plus_sr"

        q_vals = []
        for zid in df["zone_id"].values:
            q_info = uq["zones"].get(str(int(zid)))
            q_vals.append(np.nan if not q_info else q_info["q_final"])

        df["q_final"] = q_vals
        df = df.dropna(subset=["q_final"])

        x = df["y_true"].values
        y = df[pred_col].values

        vmin, vmax = np.min(x), np.max(x)
        pad = (vmax - vmin) * 0.06
        lim = (vmin - pad, vmax + pad)

        # 绘制误差条
        ax.errorbar(
            x, y,
            yerr=df["q_final"].values,
            fmt="none",
            ecolor="0.75",
            elinewidth=0.6,
            alpha=0.5,
            zorder=1
        )

        # 按角色绘制散点
        for role in ["calibration", "other", "warning"]:
            sub = df[df["calibration_role"] == role]
            ax.scatter(
                sub["y_true"],
                sub[pred_col],
                s=28,
                color=ROLE_COLORS[role],
                edgecolors="white",
                linewidths=0.3,
                alpha=0.9,
                zorder=3
            )

        # 对角线
        ax.plot(lim, lim, "--", color="0.35", lw=1.0, zorder=2)

        # 计算指标
        r2 = calc_r2(x, y)
        mae = calc_mae(x, y)
        rmse = calc_rmse(x, y)

        note_lines = [
            f"{target}",
            f"R² = {r2:.2f}",
            f"MAE = {mae:.2f}",
            f"RMSE = {rmse:.2f}",
        ]
        note_text = "\n".join(note_lines)

        ax.text(
            0.020,
            0.975,
            note_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12.0,
            linespacing=1.20,
            color="#20272D",
            zorder=9,
        )

        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")

        # 删除标题
        # ax.set_title(target, fontsize=14.0, pad=6)  # 已删除

        # 横纵坐标标签 - 只有第一个子图显示纵坐标
        if idx == 0:
            ax.set_ylabel("Predicted value", fontsize=13.0, labelpad=2.4)
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=True)

        # 所有子图都显示横坐标
        ax.set_xlabel("True value", fontsize=13.0, labelpad=2.2)

        # 刻度样式
        ax.tick_params(
            axis="both",
            which="both",
            direction="in",
            length=2.4,
            width=0.55,
            pad=1.5,
            bottom=True,
            left=True,
            top=False,
            right=False,
            labelsize=13.0,
        )
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.65)
        ax.grid(alpha=0.2, color="#E8EBED", linewidth=0.45)

        # 图例放在第4幅图右下角
        if target == "Ak":
            handles = [
                Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=ROLE_COLORS["other"],
                       label="Normal zone", markersize=7),
                Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=ROLE_COLORS["calibration"],
                       label="Calibration zone", markersize=7),
                Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=ROLE_COLORS["warning"],
                       label="Warning zone", markersize=7),
            ]
            ax.legend(
                handles=handles,
                loc="lower right",
                bbox_to_anchor=(0.98, 0.02),
                frameon=False,
                fontsize=11.0,
                ncol=1,
                handlelength=1.5,
                handletextpad=0.55,
                labelspacing=0.38,
                borderaxespad=0.0,
            )

        # 收集指标
        target_metrics = collect_metrics(patched_dir, uq_dir, target)
        all_metrics.append(target_metrics)

    # 保存图形
    output_dir.mkdir(exist_ok=True, parents=True)
    out_path = output_dir / "uq_scatter_1x4.png"
    fig.savefig(out_path, dpi=600, facecolor="white")
    plt.close()

    print(f"Saved: {out_path}")

    # 保存指标到CSV
    save_metrics_csv(all_metrics, output_dir)


# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description="Draw UQ scatter plots in 1x4 layout with large fonts."
    )
    parser.add_argument(
        "--patched-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / 'sr_patch_results_v5_bias_separated'),
        help="Directory containing SR-patch results.",
    )
    parser.add_argument(
        "--uq-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / 'uq_results_v5_bias_separated'),
        help="Directory containing UQ results.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / 'uq_final_1x4'),
        help="Output directory for figures and CSV.",
    )

    args = parser.parse_args()

    plot_combined_calibrated_scatter(
        Path(args.patched_dir),
        Path(args.uq_dir),
        Path(args.output_dir),
    )


if __name__ == "__main__":
    main()