#!/usr/bin/env python3
"""Plot calibration-zone predictions and empirical q90 error half-widths.

Expected input layout under ``--project-root``::

    sr_patch_results_v5_bias_separated/
        Rm/Rm_compensated_results_bias_separated.csv
        Re/Re_compensated_results_bias_separated.csv
        A/A_compensated_results_bias_separated.csv
        Ak/Ak_compensated_results_bias_separated.csv
    uq_results_v5_bias_separated/
        Rm/uq_comparison_report.csv
        Re/uq_comparison_report.csv
        A/uq_comparison_report.csv
        Ak/uq_comparison_report.csv

The shaded bands are descriptive empirical bands around the identity line:
``y = x +/- q90(|y_true - y_pred|)``. They are not confidence intervals.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D

BEFORE = "#8A98A8"
BEFORE_EDGE = "#9AA5AF"
BEFORE_BAND = "#E2E7EA"
AFTER = "#17847E"
AFTER_BAND = "#CBE7E2"
IDENTITY = "#505B64"
GRID = "#E8EBED"
TEXT = "#20272D"

PANEL_SPECS = [
    {"target": "Rm", "zones": [0, 3], "unit": "MPa", "digits": 1},
    {"target": "Re", "zones": [0, 3], "unit": "MPa", "digits": 1},
    {"target": "A", "zones": [7, 8], "unit": "%", "digits": 2},
    {"target": "Ak", "zones": [7, 8], "unit": "J/cm²", "digits": 2},
]

SR_REQUIRED = {"zone_id", "y_true", "y_pred_base", "y_pred_bias_plus_sr"}
UQ_REQUIRED = {"zone_id", "q_base", "q_zone_raw"}


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    default_root = script_path.parents[2]
    default_output = script_path.parent / "output" / "calibration_1x4"

    parser = argparse.ArgumentParser(
        description="Draw a 1x4 calibration before/after figure using existing CSV results."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=default_root,
        help="Project root containing the SR-patch and UQ result directories.",
    )
    parser.add_argument(
        "--sr-results-dir",
        type=Path,
        default=None,
        help="Optional explicit path to sr_patch_results_v5_bias_separated.",
    )
    parser.add_argument(
        "--uq-results-dir",
        type=Path,
        default=None,
        help="Optional explicit path to uq_results_v5_bias_separated.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help="Directory for figure exports.",
    )
    parser.add_argument(
        "--output-name",
        default="calibration_before_after_1x4",
        help="Output filename stem without extension.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "svg", "pdf", "tiff"),
        default=("png",),
        help="One or more export formats; default: png.",
    )
    parser.add_argument("--dpi", type=int, default=600, help="Raster export resolution.")
    parser.add_argument(
        "--font-family",
        default="Times New Roman",
        help="Requested primary font family.",
    )
    parser.add_argument(
        "--font-file",
        type=Path,
        default=None,
        help="Optional .ttf/.otf file to register, useful on a headless server.",
    )
    return parser.parse_args()


def configure_font(font_family: str, font_file: Path | None) -> str:
    if font_file is not None:
        font_file = font_file.expanduser().resolve()
        if not font_file.is_file():
            raise FileNotFoundError(f"Font file not found: {font_file}")
        font_manager.fontManager.addfont(str(font_file))
        font_family = font_manager.FontProperties(fname=str(font_file)).get_name()

    installed = {item.name for item in font_manager.fontManager.ttflist}
    if font_family not in installed:
        warnings.warn(
            f"Font '{font_family}' is not installed; matplotlib will use a serif fallback. "
            "Use --font-file /path/to/times.ttf on the server for exact typography.",
            stacklevel=2,
        )

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [font_family, "Arial", "Liberation Serif", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 13.0,
            "axes.titlesize": 14.0,
            "axes.labelsize": 13.0,
            "xtick.labelsize": 13.0,  # 从11.0增大到13.0
            "ytick.labelsize": 13.0,  # 从11.0增大到13.0
            "legend.fontsize": 11.0,
            "axes.linewidth": 0.65,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.dpi": 600,
        }
    )
    return font_family


def require_columns(data: pd.DataFrame, required: set[str], path: Path) -> None:
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Missing columns in {path}: {', '.join(missing)}")


def q90_higher(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("q90 input must be non-empty and finite.")
    return float(np.percentile(array, 90, method="higher"))


def fmt_value(value: float, digits: int) -> str:
    return f"{value:.{digits}f}"


def resolve_input_dirs(args: argparse.Namespace) -> tuple[Path, Path]:
    root = args.project_root.expanduser().resolve()
    sr_dir = (
        args.sr_results_dir.expanduser().resolve()
        if args.sr_results_dir is not None
        else root / "sr_patch_results_v5_bias_separated"
    )
    uq_dir = (
        args.uq_results_dir.expanduser().resolve()
        if args.uq_results_dir is not None
        else root / "uq_results_v5_bias_separated"
    )
    if not sr_dir.is_dir():
        raise FileNotFoundError(f"SR results directory not found: {sr_dir}")
    if not uq_dir.is_dir():
        raise FileNotFoundError(f"UQ results directory not found: {uq_dir}")
    return sr_dir, uq_dir


def load_target_data(
        sr_dir: Path,
        uq_dir: Path,
        target: str,
        zones: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sr_path = sr_dir / target / f"{target}_compensated_results_bias_separated.csv"
    uq_path = uq_dir / target / "uq_comparison_report.csv"
    if not sr_path.is_file():
        raise FileNotFoundError(f"Prediction file not found: {sr_path}")
    if not uq_path.is_file():
        raise FileNotFoundError(f"UQ report not found: {uq_path}")

    data = pd.read_csv(sr_path)
    uq = pd.read_csv(uq_path)
    require_columns(data, SR_REQUIRED, sr_path)
    require_columns(uq, UQ_REQUIRED, uq_path)

    data = data.loc[data["zone_id"].isin(zones)].copy()
    uq = uq.loc[uq["zone_id"].isin(zones)].copy()
    if data.empty:
        raise ValueError(f"No {target} rows found for zones {zones} in {sr_path}")
    if set(uq["zone_id"].astype(int)) != set(zones):
        raise ValueError(f"UQ report for {target} does not contain exactly zones {zones}")
    return data, uq


def draw_panel(
        ax: plt.Axes,
        spec: dict[str, object],
        sr_dir: Path,
        uq_dir: Path,
        is_first: bool = False,
) -> None:
    target = str(spec["target"])
    zones = [int(value) for value in spec["zones"]]
    unit = str(spec["unit"])
    digits = int(spec["digits"])
    data, uq = load_target_data(sr_dir, uq_dir, target, zones)

    x = data["y_true"].to_numpy(float)
    before = data["y_pred_base"].to_numpy(float)
    after = data["y_pred_bias_plus_sr"].to_numpy(float)
    if not np.isfinite(np.concatenate([x, before, after])).all():
        raise ValueError(f"Non-finite prediction values found for target {target}")

    q_before = q90_higher(np.abs(x - before))
    q_after = q90_higher(np.abs(x - after))
    reduction = 100.0 * (1.0 - q_after / q_before)

    data_min = float(np.min(np.concatenate([x, before, after])))
    data_max = float(np.max(np.concatenate([x, before, after])))
    span = data_max - data_min
    if span <= 0:
        raise ValueError(f"Cannot draw target {target}: all plotted values are identical.")
    lim_lo = data_min - 0.08 * span
    lim_hi = data_max + 0.08 * span
    grid_x = np.linspace(lim_lo, lim_hi, 300)

    ax.fill_between(
        grid_x,
        grid_x - q_before,
        grid_x + q_before,
        color=BEFORE_BAND,
        alpha=0.75,
        edgecolor=BEFORE_EDGE,
        linewidth=0.45,
        zorder=0,
    )
    ax.fill_between(
        grid_x,
        grid_x - q_after,
        grid_x + q_after,
        color=AFTER_BAND,
        alpha=0.72,
        edgecolor=AFTER,
        linewidth=0.50,
        zorder=1,
    )
    ax.plot(grid_x, grid_x, ls=(0, (3, 2)), lw=0.75, color=IDENTITY, zorder=2)
    ax.scatter(
        x,
        before,
        s=12,
        facecolors="white",
        edgecolors=BEFORE_EDGE,
        linewidths=0.65,
        alpha=0.82,
        zorder=3,
    )
    ax.scatter(
        x,
        after,
        s=12,
        color=AFTER,
        edgecolors="white",
        linewidths=0.25,
        alpha=0.90,
        zorder=4,
    )

    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color=GRID, linewidth=0.45, alpha=0.85)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.65)
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
    )

    # 横坐标：所有子图都显示
    ax.set_xlabel("True value", labelpad=2.2)

    # 纵坐标：只有第一个子图显示
    if is_first:
        ax.set_ylabel("Predicted value", labelpad=2.4)
    else:
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelleft=True)

    # 备注框：单位移到第一行性能符号后面
    note_lines = [
        f"{target} ({unit})  |  Zone {zones[0]}+{zones[1]}  |  n = {len(data)}",
        f"q₉₀: {fmt_value(q_before, digits)} → {fmt_value(q_after, digits)}  (−{reduction:.0f}%)",
    ]
    note_text = "\n".join(note_lines)

    ax.text(
        0.020,
        0.975,
        note_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12.0,  # 从10.0增大到12.0
        linespacing=1.20,
        color=TEXT,
        zorder=9,
    )


def add_legend_to_panel(ax: plt.Axes) -> None:
    """在第4幅图右下角添加图例（无背景边框，符号缩小）"""
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=4.5,  # 从6.0缩小到4.5
            markerfacecolor="white",
            markeredgecolor=BEFORE_EDGE,
            markeredgewidth=0.8,
            label="Before prediction",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=4.5,  # 从6.0缩小到4.5
            markerfacecolor=AFTER,
            markeredgecolor="white",
            markeredgewidth=0.3,
            label="After BS-GSR",
        ),
    ]
    legend = ax.legend(
        handles=handles,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.02),
        ncol=1,
        frameon=False,  # 取消背景和边框
        handlelength=1.5,
        handletextpad=0.55,
        labelspacing=0.38,
        borderaxespad=0.0,
        fontsize=11.0,
    )
    legend.set_zorder(12)


def save_figure(
        fig: plt.Figure,
        output_dir: Path,
        output_name: str,
        formats: list[str] | tuple[str, ...],
        dpi: int = 600,
) -> list[Path]:
    if dpi < 300:
        raise ValueError("--dpi must be at least 300 for publication-quality raster output.")
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for fmt in formats:
        if fmt == "png":
            destination = output_dir / f"{output_name}.png"
            fig.savefig(destination, dpi=dpi, facecolor="white", bbox_inches="tight")
        elif fmt == "svg":
            destination = output_dir / f"{output_name}.svg"
            fig.savefig(destination, facecolor="white", bbox_inches="tight")
        elif fmt == "pdf":
            destination = output_dir / f"{output_name}.pdf"
            fig.savefig(destination, facecolor="white", bbox_inches="tight")
        elif fmt == "tiff":
            destination = output_dir / f"{output_name}.tiff"
            fig.savefig(destination, dpi=dpi, facecolor="white", bbox_inches="tight")
        else:
            raise ValueError(f"Unsupported format: {fmt}")
        saved.append(destination)
    return saved


def main() -> None:
    args = parse_args()
    configure_font(args.font_family, args.font_file)
    sr_dir, uq_dir = resolve_input_dirs(args)

    # 1行4列：适当调整尺寸
    width_mm = 340.0  # 稍微增大
    height_mm = 110.0  # 稍微增大

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

    # 绘制所有子图，标记第一个子图
    for idx, (ax, spec) in enumerate(zip(axes.flat, PANEL_SPECS)):
        draw_panel(ax, spec, sr_dir, uq_dir, is_first=(idx == 0))

    # 在第4幅图（索引3）右下角添加图例
    add_legend_to_panel(axes.flat[3])

    saved = save_figure(
        fig,
        args.output_dir,
        args.output_name,
        args.formats,
        args.dpi,
    )
    plt.close(fig)
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()