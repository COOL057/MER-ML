#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

import argparse
import json
import warnings
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =========================================================================
# =========================================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "figure.dpi": 200,
    "savefig.dpi": 600,
})

PROJECT_ROOT = Path(__file__).resolve().parent
UQ_DIR = PROJECT_ROOT / "uq_results_v5_bias_separated"
PATCHED_DIR = PROJECT_ROOT / "sr_patch_results_v5_bias_separated"
OUTPUT_DIR = PROJECT_ROOT / "uq_final"

TARGETS = ["Rm", "Re", "A", "Ak"]
TARGET_TITLES = {"Rm": "Rm", "Re": "Re", "A": "A", "Ak": "Ak"}

ORIGINAL_COLOR = "#CBD5E1"
ZONE_WISE_COLOR = "#0B2545"

def load_pkl(pkl_path: Path):
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    df = obj["df"] if isinstance(obj, dict) else obj
    return df.copy(), obj


def load_uq_json(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# -------------------------------------------------------------
# -------------------------------------------------------------
def plot_conformal_coverage(patched_dir: Path, uq_dir: Path, output_dir: Path):
    # 标准两栏 17.5 cm 宽（6.89 in）
    fig, axes = plt.subplots(1, 4, figsize=(6.89, 2.5), sharey=True)

    for i, (ax, target) in enumerate(zip(axes, TARGETS)):
        json_path = uq_dir / target / "uq_zone_table.json"
        pkl_path = patched_dir / target / f"{target}_compensated_residuals_bias_separated.pkl"
        if not json_path.exists() or not pkl_path.exists():
            continue

        data = load_uq_json(json_path)
        df, _ = load_pkl(pkl_path)
        df["abs_res_final"] = df["residual_bias_plus_sr"].abs()

        q_global = data["global_q_ref"]
        zones = data["zones"]
        z_ids = sorted([int(k) for k in zones.keys()])

        coverages_after = []
        coverages_before = []
        for zid in z_ids:
            z_info = zones[str(zid)]
            coverages_after.append(z_info["actual_coverage"] * 100.0)

            sub_df = df[df["zone_id"] == zid]
            if len(sub_df) > 0:
                cov_before = np.mean(sub_df["abs_res_final"] <= q_global) * 100.0
            else:
                cov_before = np.nan
            coverages_before.append(cov_before)

        x_ticks = [f"Z{zid}" for zid in z_ids]
        y_after = np.array(coverages_after)
        y_before = np.array(coverages_before)

        # 标称 90% 置信度基准线
        ax.axhline(90.0, linestyle="--", color="#2F3E46", linewidth=0.8, alpha=0.9, zorder=2)

        # 双棒棒糖并排坐标设置
        x_indices = np.arange(len(z_ids))
        offset = 0.18

        # 1. 绘制左棒：原经验半宽（统一中性浅灰，空心小圆点）
        ax.vlines(x_indices - offset, ymin=50.0, ymax=y_before, colors=ORIGINAL_COLOR, linewidth=0.7, alpha=0.6,
                  zorder=1)
        ax.scatter(x_indices - offset, y_before, s=8, facecolors='white', edgecolors=ORIGINAL_COLOR, linewidths=1.0,
                   zorder=3)

        # 2. 绘制右棒：分区段经验半宽（统一深蓝，空心圆点）
        ax.vlines(x_indices + offset, ymin=50.0, ymax=y_after, colors=ZONE_WISE_COLOR, linewidth=0.8, alpha=0.8,
                  zorder=1)
        ax.scatter(x_indices + offset, y_after, s=10, facecolors='white', edgecolors=ZONE_WISE_COLOR, linewidths=1.2,
                   zorder=3)

        ax.set_facecolor("#FFFFFF")
        ax.set_xlim(-0.6, 8.6)
        ax.set_ylim(50.0, 105.0)          # 【修改】Y 轴下限 45 -> 50
        ax.set_xticks(x_indices)

        # 加大 X 轴刻度标签字号
        ax.set_xticklabels(x_ticks, fontsize=9, color="#000000")

        if i == 0:
            # 加大 Y 轴标签字号
            ax.set_ylabel("Empirical Coverage (%)", fontsize=11, fontweight='bold', color="#000000")
            ax.tick_params(axis='y', labelsize=9, colors="#000000")

        # 全局刻度参数字号加大
        ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize=9, colors="#000000")
        for spine in ax.spines.values():
            spine.set_color("#000000")
            spine.set_linewidth(0.5)

        ax.grid(True, linestyle='--', color="#F3F4F6", alpha=0.8, linewidth=0.5)

        # 加大小标题字号（【修改】11 -> 10）
        ax.text(0.95, 0.94, TARGET_TITLES[target], transform=ax.transAxes, ha='right', va='top', fontsize=10,
                fontweight='bold', color='#000000')

        if target == "Rm":
            handles = [
                mlines.Line2D([], [], color=ORIGINAL_COLOR, marker='o', linestyle='None', markersize=2,   # 【修改】4 -> 3
                              label='Original empirical half-widths'),
                mlines.Line2D([], [], color=ZONE_WISE_COLOR, marker='o', linestyle='None', markersize=2,  # 【修改】4 -> 3
                              label='Zone-wise empirical half-widths'),
            ]
            ax.legend(
                handles=handles,
                loc="upper left",
                bbox_to_anchor=(-0.05, 1.0),
                frameon=False,
                facecolor="white",
                edgecolor="none",
                framealpha=0.9,
                fontsize=5.5,
                handletextpad=0.0,
                labelspacing=0.15,
                borderaxespad=0.2,
            )

    plt.tight_layout(pad=0.3)
    save_png = output_dir / "uq_conformal_coverage_FINAL.png"
    save_pdf = output_dir / "uq_conformal_coverage_FINAL.pdf"
    save_svg = output_dir / "uq_conformal_coverage_FINAL.svg"
    save_tiff = output_dir / "uq_conformal_coverage_FINAL.tiff"
    fig.savefig(save_png, dpi=600, bbox_inches='tight')
    fig.savefig(save_pdf, bbox_inches='tight')
    fig.savefig(save_svg, bbox_inches='tight')
    fig.savefig(save_tiff, dpi=600, bbox_inches='tight')
    plt.close()
    print(f" >> 双棒棒糖覆盖率比对图已无损保存: {save_png}")


# -------------------------------------------------------------
# -------------------------------------------------------------
def plot_conformal_interval_comparison(uq_dir: Path, output_dir: Path):
    # 共享 Y 轴，保持完美的对称排版 (17.5 cm 宽)
    fig, axes = plt.subplots(1, 4, figsize=(6.89, 2.5), sharey=True)

    for i, (ax, target) in enumerate(zip(axes, TARGETS)):
        csv_path = uq_dir / target / "uq_comparison_report.csv"
        json_path = uq_dir / target / "uq_zone_table.json"
        if not csv_path.exists() or not json_path.exists():
            continue

        df = pd.read_csv(csv_path)
        uq_data = load_uq_json(json_path)
        q_global = uq_data["global_q_ref"]

        z_ids = df["zone_id"].values
        q_base = df["q_base"].values
        q_final = df["q_final"].values
        q_base_norm = q_base / q_global
        q_final_norm = q_final / q_global

        x_ticks = [f"Z{zid}" for zid in z_ids]

        ax.axhline(1.0, linestyle="--", color="#2F3E46", linewidth=0.8, alpha=0.9, zorder=2)

        x_indices = np.arange(len(z_ids))
        bar_width = 0.35

        # 绘制分组柱状图
        ax.bar(x_indices - bar_width / 2, q_base_norm, width=bar_width, color=ORIGINAL_COLOR, zorder=3)
        ax.bar(x_indices + bar_width / 2, q_final_norm, width=bar_width, color=ZONE_WISE_COLOR, zorder=3)

        ax.set_facecolor("#FFFFFF")
        ax.set_xlim(-0.5, 8.5)
        ax.set_ylim(0.0, 1.9)             # 【修改】Y 轴上限 3.2 -> 1.9
        ax.set_xticks(x_indices)

        # 加大 X 轴刻度标签字号
        ax.set_xticklabels(x_ticks, fontsize=9, color="#000000")

        # 全局刻度参数字号加大
        ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize=9, colors="#000000")
        ax.grid(True, linestyle='--', color="#F3F4F6", alpha=0.8, linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_color("#000000")
            spine.set_linewidth(0.5)

        if i == 0:
            # 加大 Y 轴标签字号
            ax.set_ylabel("Normalized 90% Half-width", fontsize=11, fontweight='bold', color="#000000")
            ax.tick_params(axis='y', labelsize=9, colors="#000000")

        # 加大小标题字号（【修改】11 -> 10）
        ax.text(0.95, 0.94, TARGET_TITLES[target], transform=ax.transAxes, ha='right', va='top', fontsize=10,
                fontweight='bold', color='#000000')

        if target == "Rm":
            handles = [
                mlines.Line2D([], [], color=ORIGINAL_COLOR, marker='s', linestyle='None', markersize=4,   # 【修改】5 -> 4
                              label='Original empirical half-widths'),
                mlines.Line2D([], [], color=ZONE_WISE_COLOR, marker='s', linestyle='None', markersize=4,  # 【修改】5 -> 4
                              label='Zone-wise empirical half-widths'),
            ]
            ax.legend(
                handles=handles,
                loc="upper left",
                frameon=True,
                facecolor="white",
                edgecolor="none",
                framealpha=0.9,
                fontsize=6.0,
                handletextpad=0.2,
                labelspacing=0.15,
                borderaxespad=0.2,
            )

    plt.tight_layout(pad=0.3)
    save_png = output_dir / "uq_interval_comparison_FINAL.png"
    save_pdf = output_dir / "uq_interval_comparison_FINAL.pdf"
    save_svg = output_dir / "uq_interval_comparison_FINAL.svg"
    save_tiff = output_dir / "uq_interval_comparison_FINAL.tiff"
    fig.savefig(save_png, dpi=600, bbox_inches='tight')
    fig.savefig(save_pdf, bbox_inches='tight')
    fig.savefig(save_svg, bbox_inches='tight')
    fig.savefig(save_tiff, dpi=600, bbox_inches='tight')
    plt.close()
    print(f" >> 区间对比图已无损保存: {save_png}")


# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(exist_ok=True, parents=True)
    plot_conformal_coverage(PATCHED_DIR, UQ_DIR, arguments.output_dir)
    plot_conformal_interval_comparison(UQ_DIR, arguments.output_dir)
    print("\n[顺利完成] uq_final ")


if __name__ == "__main__":
    main()