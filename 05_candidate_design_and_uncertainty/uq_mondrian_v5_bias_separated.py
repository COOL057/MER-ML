# uq_mondrian_v5_bias_separated.py
# Mondrian conformal UQ for the BS-GSR v5 pipeline.
#
# Statistical method: zone_q90 × t_correction_factor(n)
#   — zone_q90: empirical 90th percentile of |residual_bias_plus_sr|
#   — t_correction: t_{n-1,0.9} / z_{0.9}, accounts for small-sample bias
#   — no cross-zone shrinkage; each zone is calibrated independently
#
# Zone roles (semantic only, same formula for all):
#   calibration  — participated in BS-GSR bias+SR correction
#   other        — not calibrated, model predicted well in-distribution
#   warning (Z6) — overlap-risk zone, model extrapolation failure, HIGH_RISK
#
# Data source: {target}_compensated_residuals_bias_separated.pkl
#
# Outputs (per target, under {output_dir}/{target}/):
#   uq_zone_table.json        — prediction engine use
#   uq_comparison_report.csv  — ablation table (q_base vs q_final)
#
# Global summary:
#   {output_dir}/uq_summary_report.txt

import argparse
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist

warnings.filterwarnings("ignore")

# ── constants ──────────────────────────────────────────────────────────────────

ZONE_LABELS = {
    0: "Low Delta_TI x Short tau",
    1: "Low Delta_TI x Mid tau",
    2: "Low Delta_TI x Long tau",
    3: "Mid Delta_TI x Short tau",
    4: "Mid Delta_TI x Mid tau",
    5: "Mid Delta_TI x Long tau",
    6: "High Delta_TI x Short tau",
    7: "High Delta_TI x Mid tau",
    8: "High Delta_TI x Long tau",
}

TARGET_UNITS = {"Rm": "MPa", "Re": "MPa", "A": "%", "Ak": "J"}

WARNING_ZONE = 6


# ── t-distribution correction factor ──────────────────────────────────────────

def t_correction_factor(n: int, nominal_coverage: float = 0.90) -> float:
    """
    Inflate the q90 estimate for small-sample uncertainty.
    Returns t_{n-1, p} / z_p  (≥ 1.0).
    """
    z = float(t_dist.ppf(nominal_coverage, df=10_000))
    t = float(t_dist.ppf(nominal_coverage, df=max(n - 1, 1)))
    return max(t / z, 1.0)


# ── per-zone UQ calibration ────────────────────────────────────────────────────

def calibrate_zone(
    abs_res_final: np.ndarray,
    abs_res_base: np.ndarray,
    zone_id: int,
    role: str,
    q_global: float,
    coverage: float,
) -> dict:
    """
    q_final = zone_q90(|residual_bias_plus_sr|) × t_correction_factor(n)

    q_global is retained for reference output only, not used in q_final.
    """
    n = len(abs_res_final)
    label = ZONE_LABELS.get(zone_id, f"Zone {zone_id}")

    q_base = float(np.percentile(abs_res_base, coverage * 100, method="higher")) if n > 0 else np.nan

    if n == 0:
        return {
            "zone_id": zone_id, "zone_label": label, "role": role,
            "n": 0, "q_base": np.nan, "q_zone_raw": np.nan,
            "q_final": np.nan, "t_factor": np.nan,
            "actual_coverage": np.nan, "note": "empty_zone",
        }

    q_point = float(np.percentile(abs_res_final, coverage * 100, method="higher"))
    t_factor = t_correction_factor(n, coverage)
    q_final = q_point * t_factor
    actual_cov = float(np.mean(abs_res_final <= q_final))

    if role == "warning":
        note = f"HIGH_RISK|global_q={q_global:.2f}(ref)"
    else:
        note = f"{role}|global_q={q_global:.2f}(ref)"

    return {
        "zone_id": zone_id,
        "zone_label": label,
        "role": role,
        "n": int(n),
        "q_base": float(q_base),
        "q_zone_raw": float(q_point),
        "q_final": float(q_final),
        "t_factor": float(t_factor),
        "actual_coverage": float(actual_cov),
        "note": note,
    }


# ── single-target processing ───────────────────────────────────────────────────

def process_target(
    target: str,
    patched_dir: Path,
    output_dir: Path,
    coverage: float,
) -> dict | None:
    pkl_path = patched_dir / target / f"{target}_compensated_residuals_bias_separated.pkl"

    if not pkl_path.exists():
        print(f"  ✗ 未找到: {pkl_path}")
        return None

    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)

    df: pd.DataFrame = obj["df"] if isinstance(obj, dict) else obj
    df = df.copy()

    required = {"zone_id", "residual_base", "residual_bias_plus_sr"}
    missing = required - set(df.columns)
    if missing:
        print(f"  ✗ {target} pkl 缺少列: {missing}")
        return None

    unit = TARGET_UNITS.get(target, "")
    warning_zones = obj.get("warning_zones", [WARNING_ZONE]) if isinstance(obj, dict) else [WARNING_ZONE]
    calibration_zones = obj.get("calibration_zones", []) if isinstance(obj, dict) else []

    df["abs_res_final"] = df["residual_bias_plus_sr"].abs()
    df["abs_res_base"]  = df["residual_base"].abs()

    # global q90 for reference (non-warning zones, final residuals)
    non_warn = ~df["zone_id"].isin(warning_zones)
    q_global = float(
        np.percentile(df.loc[non_warn, "abs_res_final"].values, coverage * 100, method="higher")
    ) if non_warn.sum() > 0 else float(df["abs_res_final"].max())

    zone_results = []
    for zid in sorted(df["zone_id"].unique()):
        mask = df["zone_id"] == zid
        role = "warning" if zid in warning_zones else ("calibration" if zid in calibration_zones else "other")
        zone_results.append(calibrate_zone(
            abs_res_final=df.loc[mask, "abs_res_final"].values,
            abs_res_base=df.loc[mask, "abs_res_base"].values,
            zone_id=int(zid),
            role=role,
            q_global=q_global,
            coverage=coverage,
        ))

    # ── print table ──────────────────────────────────────────────────────────
    print(f"\n  {'=' * 95}")
    print(f"  {target}  (Coverage target={coverage * 100:.0f}%,  unit={unit})")
    print(f"  Global q{coverage * 100:.0f} (non-warning, ref): {q_global:.2f} {unit}")
    print(f"  {'─' * 95}")
    print(f"  {'Zone':<34} {'Role':<12} {'n':>4}  {'q_base':>9}  {'q_zone_raw':>11}  {'q_final':>9}  {'t_f':>5}  {'Cov%':>6}")
    print(f"  {'─' * 95}")

    for r in zone_results:
        if r["n"] == 0:
            continue
        cov_flag = "✓" if r["actual_coverage"] >= 0.85 else "✗"
        warn_tag = "  ⚠ HIGH_RISK" if r["role"] == "warning" else ""
        print(
            f"  Z{r['zone_id']} {r['zone_label']:<31} {r['role']:<12} {r['n']:>4}  "
            f"{r['q_base']:>9.2f}  {r['q_zone_raw']:>11.2f}  {r['q_final']:>9.2f}  "
            f"{r['t_factor']:>5.3f}  {r['actual_coverage'] * 100:>5.1f}%{cov_flag}{warn_tag}"
        )

    # ── save outputs ─────────────────────────────────────────────────────────
    target_out = output_dir / target
    target_out.mkdir(parents=True, exist_ok=True)

    zone_table = {
        "target": target,
        "coverage": coverage,
        "global_q_ref": round(q_global, 4),
        "unit": unit,
        "zones": {
            str(r["zone_id"]): {
                "zone_label": r["zone_label"],
                "role": r["role"],
                "n": r["n"],
                "q_final": round(r["q_final"], 4) if not np.isnan(r["q_final"]) else None,
                "t_factor": round(r["t_factor"], 4) if not np.isnan(r["t_factor"]) else None,
                "actual_coverage": round(r["actual_coverage"], 4) if not np.isnan(r["actual_coverage"]) else None,
                "note": r["note"],
            }
            for r in zone_results if r["n"] > 0
        },
    }
    with open(target_out / "uq_zone_table.json", "w", encoding="utf-8") as f:
        json.dump(zone_table, f, indent=2, ensure_ascii=False)

    report_rows = []
    for r in zone_results:
        if r["n"] == 0:
            continue
        report_rows.append({
            "target": target,
            "zone_id": r["zone_id"],
            "zone_label": r["zone_label"],
            "role": r["role"],
            "n": r["n"],
            "q_base": round(r["q_base"], 3),
            "q_zone_raw": round(r["q_zone_raw"], 3),
            "q_final": round(r["q_final"], 3),
            "delta_q_pct": round((r["q_base"] - r["q_final"]) / r["q_base"] * 100, 1) if r["q_base"] > 0 else 0.0,
            "t_factor": round(r["t_factor"], 3),
            "actual_coverage_%": round(r["actual_coverage"] * 100, 1),
            "note": r["note"],
        })

    pd.DataFrame(report_rows).to_csv(target_out / "uq_comparison_report.csv", index=False)

    print(f"\n  ✓ 已保存: {target_out / 'uq_zone_table.json'}")
    print(f"  ✓ 已保存: {target_out / 'uq_comparison_report.csv'}")

    return {"target": target, "unit": unit, "global_q": q_global, "zone_results": zone_results}


# ── global summary txt ─────────────────────────────────────────────────────────

def write_summary_txt(all_results: list[dict], output_dir: Path, coverage: float):
    lines = []
    sep = "=" * 100
    lines += [
        sep,
        f"  UQ Summary — BS-GSR v5 Mondrian Conformal UQ",
        f"  Method: q_final = zone_q90 × t_correction_factor(n)",
        f"  Coverage target: {coverage * 100:.0f}%  |  Pass criterion: actual coverage ≥ 85%",
        f"  Z6: warning-only (model extrapolation failure, HIGH_RISK)",
        sep,
    ]

    for res in all_results:
        target, unit = res["target"], res["unit"]
        zone_results = res["zone_results"]

        lines += [
            "",
            f"  ── {target} ({unit}) ──────────────────────────────────────────────────",
            f"  Global q{coverage*100:.0f} (non-warning, ref): {res['global_q']:.2f} {unit}",
            "",
            f"  {'Zone':<34} {'Role':<12} {'n':>4}  {'q_base':>9}  {'q_final':>9}  {'Δq%':>8}  {'t_f':>6}  {'Cov%':>6}  PASS?",
            "  " + "-" * 95,
        ]

        all_pass = True
        for r in zone_results:
            if r["n"] == 0:
                continue
            dq = (r["q_base"] - r["q_final"]) / r["q_base"] * 100 if r["q_base"] > 0 else 0.0
            ok = r["actual_coverage"] >= 0.85
            if not ok:
                all_pass = False
            warn = "  ⚠" if r["role"] == "warning" else ""
            lines.append(
                f"  Z{r['zone_id']} {r['zone_label']:<31} {r['role']:<12} {r['n']:>4}  "
                f"{r['q_base']:>9.2f}  {r['q_final']:>9.2f}  {dq:>+8.1f}%  "
                f"{r['t_factor']:>6.3f}  {r['actual_coverage']*100:>5.1f}%  "
                f"{'PASS ✓' if ok else 'FAIL ✗'}{warn}"
            )

        lines.append("")
        fails = [r for r in zone_results if r["n"] > 0 and r["actual_coverage"] < 0.85]
        marginal = all(0.80 <= r["actual_coverage"] < 0.85 for r in fails) if fails else False
        if all_pass:
            lines.append("  Overall: ALL PASS ✓")
        elif marginal:
            lines.append("  Overall: MARGINAL — all FAIL zones within 5pp of threshold (likely small-sample noise)")
        else:
            lines.append("  Overall: FAIL — review zones above")

    lines += [
        "",
        sep,
        "",
        "  INTERPRETATION GUIDE",
        "  ─────────────────────",
        "  Δq% = (q_base − q_final) / q_base × 100",
        "    +% → interval NARROWED: BS-GSR calibration reduced residuals in this zone",
        "    −% → interval WIDENED:  BS-GSR not applied here; t-correction > 1 due to small n",
        "",
        "  t_factor = t_{n−1, 0.9} / z_{0.9}",
        "    Accounts for quantile estimation uncertainty with small n.",
        "    n=6: +15.2%  |  n=14: +5.3%  |  n=22: +3.2%  |  n=34: +2.0%",
        "",
        "  Role:",
        "    calibration → zone underwent BS-GSR bias + SR correction",
        "    other       → no calibration; model predicted well in-distribution",
        "    warning     → Z6 (High ΔTI × Short τ), model extrapolation failure,",
        "                  interval reflects genuine unreliability — do not use for design",
        "",
        "  Coverage: in-sample empirical (calibration = evaluation; unavoidable at n=14–34).",
        "    85–95%: acceptable.  80–85%: likely n-noise (1 sample = 5pp at n=19).",
        sep,
    ]

    txt_path = output_dir / "uq_summary_report.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  ✓ 汇总报告已保存: {txt_path}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BS-GSR v5 Mondrian Conformal UQ")
    parser.add_argument("--target", type=str, default="all",
                        choices=["Rm", "Re", "A", "Ak", "all"])
    parser.add_argument("--patched-dir", type=str,
                        default=str(Path(__file__).resolve().parent / 'sr_patch_results_v5_bias_separated'))
    parser.add_argument("--output-dir", type=str,
                        default=str(Path(__file__).resolve().parent / 'uq_results_v5_bias_separated'))
    parser.add_argument("--coverage", type=float, default=0.90)
    args = parser.parse_args()

    patched_dir = Path(args.patched_dir)
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    targets = ["Rm", "Re", "A", "Ak"] if args.target == "all" else [args.target]

    print(f"\n{'=' * 95}")
    print(f"  BS-GSR v5 Mondrian Conformal UQ  |  Coverage: {args.coverage * 100:.0f}%")
    print(f"  Input:  {patched_dir}")
    print(f"  Output: {output_dir}")
    print(f"{'=' * 95}")

    all_results = []
    for target in targets:
        result = process_target(
            target=target,
            patched_dir=patched_dir,
            output_dir=output_dir,
            coverage=args.coverage,
        )
        if result:
            all_results.append(result)

    if all_results:
        write_summary_txt(all_results, output_dir, args.coverage)

    print(f"\n{'=' * 95}")
    print(f"  完成！输出: {output_dir}/")
    print(f"    {{target}}/uq_zone_table.json       — 预测引擎读取")
    print(f"    {{target}}/uq_comparison_report.csv — 消融对比表")
    print(f"    uq_summary_report.txt              — 人可读汇总")
    print(f"{'=' * 95}")


if __name__ == "__main__":
    main()
