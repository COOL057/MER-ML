#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns

# =========================================================================
# 🎨 【全局常量与高级科技感色系控制台】
# =========================================================================
ALLOY_ELEMENTS = ["C", "Si", "Mn", "Cr", "V", "Al", "Ni", "Mo", "Cu", "W"]

PALETTES = {
    # 方案 4：浩瀚深空探索风 (Cosmic DeepSpace) - 默认启用，大深度空间感，对比鲜明
    "Cosmic_DeepSpace": {
        "A_Confidence": "#0B2545",  # 深空极夜蓝 (Midnight Blue)
        "B_Exploratory": "#F26419",  # 超新星耀橙 (Supernova Orange)
        "C_Risk": "#94A3B8",  # 星云尘埃灰 (Cosmos Gray - 净化背景)
        "bg": "#FFFFFF",
        "grid": "#F3F4F6",
        "expert_star": "#FFC857",  # 恒星熠黄
        "star_edge": "#0B2545"
    },
    # 方案 1：北欧极简冷科技风 (Nordic Minimalist) - 干净、严谨、大方
    "Nordic_Minimalist": {
        "A_Confidence": "#2B4C7E",  # 冰川深蓝 (Ice Glacier Blue)
        "B_Exploratory": "#A77A50",  # 精铸黄铜 (Warm Brass)
        "C_Risk": "#94A3B8",  # 优雅板岩灰 (Slate Gray - 净化背景)
        "bg": "#FFFFFF",  # 纯白底
        "grid": "#F3F4F6",  # 极轻网格
        "expert_star": "#DFB15B",  # 经典香槟金
        "star_edge": "#111827"  # 炭灰边界
    },
    # 方案 2：钛金钴蓝实验室风 (Titanium Cobalt) - 材料相变物理科技感
    "Titanium_Cobalt": {
        "A_Confidence": "#0F2080",  # 纯正钴蓝 (Pure Cobalt Blue)
        "B_Exploratory": "#D97706",  # 激光耀橘 (Laser Orange)
        "C_Risk": "#64748B",  # 钛合金灰 (Titanium Gray - 净化背景)
        "bg": "#F8FAFC",  # 科技微灰
        "grid": "#F1F5F9",
        "expert_star": "#FFD700",  # 闪耀黄金
        "star_edge": "#0F172A"
    },
    # 方案 3：高精翡翠光电风 (Precision Emerald) - SCI 期浅绿色系
    "Precision_Emerald": {
        "A_Confidence": "#0A5C36",  # 高精翡翠绿 (Precision Emerald)
        "B_Exploratory": "#EA580C",  # 聚能耀橘 (Energetic Orange)
        "C_Risk": "#A3A3A3",  # 精密石材灰 (Stone Gray - 净化背景)
        "bg": "#FFFFFF",
        "grid": "#F3F4F6",
        "expert_star": "#FFE600",
        "star_edge": "#171717"
    },
    # 方案 5：哑光碳纤工业风 (Matte Carbon) - 工业机械沉稳质感
    "Matte_Carbon": {
        "A_Confidence": "#334155",  # 碳纤深灰 (Slate Charcoal)
        "B_Exploratory": "#B25E29",  # 曜铜金属色 (Metallic Copper)
        "C_Risk": "#CBD5E1",  # 哑光炭素灰 (Carbon Gray - 净化背景)
        "bg": "#FAFAFA",
        "grid": "#E5E7EB",
        "expert_star": "#FBBF24",
        "star_edge": "#1E293B"
    }
}

# 👉 【一键换色配置】 默认启用高不透明度的 "Cosmic_DeepSpace"
ACTIVE_PALETTE = "Cosmic_DeepSpace"

# -------------------------------------------------------------------------
# 学术格式初始化配置 (Times New Roman & 刻度方向控制)
# -------------------------------------------------------------------------
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'

# 刻度朝内设置 (Inward Ticks)
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

P = PALETTES[ACTIVE_PALETTE]
TEXT_COLOR = '#000000'  # 纯学术黑
GRID_COLOR = P["grid"]
BG_COLOR = P["bg"]
HIGHLIGHT_GOLD = P["expert_star"]
STAR_EDGE = P["star_edge"]

# ==================== 1. 读取数据 ====================
PROJECT_ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "design_outputs" / "candidate_prediction_summary_simplified.csv")
parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "design_outputs")
arguments = parser.parse_args()
csv_path = arguments.input
output_dir = arguments.output_dir

# 载入数据并保留完整列信息
df = pd.read_csv(csv_path)
df_plot = df.copy()

# 坐标极限
A_min, A_max = df_plot['A_pred_for_ranking'].min() - 0.8, df_plot['A_pred_for_ranking'].max() + 0.8
Rm_min, Rm_max = df_plot['Rm_pred_for_ranking'].min() - 50, df_plot['Rm_pred_for_ranking'].max() + 50

# 建立色彩分类表
SCI_PALETTE = {
    'A_primary': P["A_Confidence"],
    'B_review': P["B_Exploratory"],
    'C_not_recommended': P["C_Risk"]
}

# 创建三联画布：双栏标准 17 cm 宽 (6.69 in)，紧凑高度微调至 2.7 in (给更大字体留出呼吸感)
fig, axes = plt.subplots(1, 3, figsize=(6.69, 2.7), sharey=True)

# -------------------------------------------------------------
# 子图 A：全局景观与分区图 (图一：添加三色无边框图例，点极小)
# -------------------------------------------------------------
ax1 = axes[0]
ax1.set_facecolor(BG_COLOR)
sns.scatterplot(
    data=df_plot,
    x='A_pred_for_ranking',
    y='Rm_pred_for_ranking',
    hue='design_tier',
    palette=SCI_PALETTE,
    alpha=0.85,  # 颜色饱满不偏浅
    s=1.2,  # 背景散点极小
    ax=ax1,
    edgecolor='none',
    legend=False  # 屏蔽 seaborn 默认图例，使用我们高定制的纯净图例
)
ax1.set_xlim(A_min, A_max)
ax1.set_ylim(Rm_min, Rm_max)
ax1.set_xlabel('')  # 隐藏左侧子图横坐标标签
ax1.set_ylabel('Predicted Tensile Strength Rm (MPa)', fontsize=8.5, fontweight='bold', color=TEXT_COLOR)

# 📌 针对可信、探索、风险样本的高清无框图例绘制 (Bottom-Left，带有优雅的半透明背景)
handle_A = mlines.Line2D([], [], color=P["A_Confidence"], marker='o', linestyle='None', markersize=1,
                         label='Tier_A Confidence')
handle_B = mlines.Line2D([], [], color=P["B_Exploratory"], marker='o', linestyle='None', markersize=1,
                         label='Tier_B Exploratory')
handle_C = mlines.Line2D([], [], color=P["C_Risk"], marker='o', linestyle='None', markersize=1, label='Tier_C Risk')

ax1.legend(
    handles=[handle_A, handle_B, handle_C],
    loc='lower left',
    bbox_to_anchor=(0.005, 0.01),
    fontsize=7,
    frameon=True,             # 启用背景框
    facecolor='white',         # 白色背景
    framealpha=0.9,            # 60% 适度半透明遮罩
    edgecolor='none',          # 取消黑色边线，呈现现代轻盈感
    handletextpad=0.005,
    labelspacing=0.05
)

# -------------------------------------------------------------
# 子图 B：置信生产轨 (Confidence Domain - Tier A)
# -------------------------------------------------------------
ax2 = axes[1]
ax2.set_facecolor(BG_COLOR)
a_primary_df = df_plot[df_plot['design_tier'] == 'A_primary'].copy()
ax2.scatter(
    a_primary_df['A_pred_for_ranking'],
    a_primary_df['Rm_pred_for_ranking'],
    color=P["A_Confidence"],
    alpha=0.45,  # 调高背景点不透明度
    s=1.2,
    edgecolor='none'
)

# 筛选前10个安全稳健点
top_safe = a_primary_df.sort_values(by='strong_plastic_product', ascending=False).head(10).reset_index(drop=True)

for idx, row in top_safe.iterrows():
    rank_id = idx + 1

    if rank_id == 2:
        # S2 为特挑最低成本方案：黄金五角星缩至 55，极致精致
        ax2.scatter(
            row['A_pred_for_ranking'],
            row['Rm_pred_for_ranking'],
            marker='*',
            color=HIGHLIGHT_GOLD,
            s=55,
            edgecolors=STAR_EDGE,
            linewidths=0.5,
            zorder=12,
            label='Cost-Selected'
        )
        # 完美居中悬挂在五角星正下方：拉近间距至 -7，字号放大至 7.5
        ax2.annotate(
            "S2",
            xy=(row['A_pred_for_ranking'], row['Rm_pred_for_ranking']),
            xytext=(0, -7),
            textcoords='offset points',
            ha='center',
            va='top',
            fontsize=7.5,
            fontweight='bold',
            color='#000000',
            bbox=dict(boxstyle="round,pad=0.15", fc="#FFFDF0", ec=HIGHLIGHT_GOLD, lw=0.7, alpha=0.95),
            zorder=13
        )
    else:
        # 普通优选点：空心圈尺寸降至 10，去除 S1, S3 等重叠标签文字，清爽无暇
        ax2.scatter(
            row['A_pred_for_ranking'],
            row['Rm_pred_for_ranking'],
            marker='o',
            color='white',
            edgecolors=P["A_Confidence"],
            linewidths=0.8,
            s=10,
            zorder=10,
            label='Top 10 Strength-Ductility Product' if idx == 0 else ""
        )

ax2.set_xlim(A_min, A_max)
# 全图仅在中间子图下方保留横坐标标签 (字号放大至 8.5)
ax2.set_xlabel('Predicted Elongation A (%)', fontsize=8.5, fontweight='bold', color=TEXT_COLOR)

# -------------------------------------------------------------
# 子图 C：前沿探索轨 (Exploratory Domain - Tier B)
# -------------------------------------------------------------
ax3 = axes[2]
ax3.set_facecolor(BG_COLOR)
b_review_df = df_plot[df_plot['design_tier'] == 'B_review'].copy()
ax3.scatter(
    b_review_df['A_pred_for_ranking'],
    b_review_df['Rm_pred_for_ranking'],
    color=P["B_Exploratory"],
    alpha=0.45,  # 调高背景点不透明度
    s=1.2,
    edgecolor='none'
)

# 筛选前10个探索点
top_explore = b_review_df.sort_values(by='strong_plastic_product', ascending=False).head(10).reset_index(drop=True)

for idx, row in top_explore.iterrows():
    rank_id = idx + 1

    if rank_id == 2:
        # E2 选定的低成本优选点，五角星尺寸缩至 55
        ax3.scatter(
            row['A_pred_for_ranking'],
            row['Rm_pred_for_ranking'],
            marker='*',
            color=HIGHLIGHT_GOLD,
            s=55,
            edgecolors=STAR_EDGE,
            linewidths=0.5,
            zorder=12,
            label='Cost-Selected'
        )
        # 完美居中悬挂在五角星正下方：拉近间距至 -7，字号放大至 7.5
        ax3.annotate(
            "E2",
            xy=(row['A_pred_for_ranking'], row['Rm_pred_for_ranking']),
            xytext=(0, -7),
            textcoords='offset points',
            ha='center',
            va='top',
            fontsize=7.5,
            fontweight='bold',
            color='#000000',
            bbox=dict(boxstyle="round,pad=0.15", fc="#FFFDF0", ec=HIGHLIGHT_GOLD, lw=0.7, alpha=0.95),
            zorder=13
        )
    else:
        # 普通点：空心圈(o)尺寸缩小至 10，不带有任何标注文字，极简、清爽
        ax3.scatter(
            row['A_pred_for_ranking'],
            row['Rm_pred_for_ranking'],
            marker='o',
            color='white',
            edgecolors=P["B_Exploratory"],
            linewidths=0.8,
            s=10,
            zorder=10,
            label='Top 10 Strength-Ductility Product' if idx == 0 else ""
        )

ax3.set_xlim(A_min, A_max)
ax3.set_xlabel('')  # 隐藏右侧子图横坐标标签

# -------------------------------------------------------------
# 坐标轴刻度及右上角黑色加粗标注 (Times New Roman, Bold, Ticks Inside)
# -------------------------------------------------------------
for ax in axes:
    # 启用上下左右双向刻度，且方向朝内 (in)，刻度大小放大至 7.5
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize=7.5, colors=TEXT_COLOR)
    # 格网及轴框线粗细微调
    ax.grid(True, linestyle='--', color=GRID_COLOR, alpha=0.8, linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_color(TEXT_COLOR)
        spine.set_linewidth(0.5)

# 在各子图右上角放置纯黑色加粗标注 (字号放大至 8)
axes[0].text(0.95, 0.94, "Global Candidate Space", transform=axes[0].transAxes, ha='right', va='top', fontsize=8,
             fontweight='bold', color='#000000')
axes[1].text(0.95, 0.94, "Confidence Domain", transform=axes[1].transAxes, ha='right', va='top', fontsize=8,
             fontweight='bold', color='#000000')
axes[2].text(0.95, 0.94, "Exploratory Domain", transform=axes[2].transAxes, ha='right', va='top', fontsize=8,
             fontweight='bold', color='#000000')

# 精细化调整图例：启用无边框半透明背景框（与图一保持完美一致，移除了多余的 framealpha）
axes[1].legend(
    loc='lower left',
    fontsize=7,
    frameon=True,             # 启用背景框
    facecolor='white',         # 白色背景
    framealpha=0.9,            # 60% 适度半透明遮罩
    edgecolor='none',          # 取消黑色边线，呈现现代轻盈感
    handletextpad=0.2,
    markerscale=0.75

)
axes[2].legend(
    loc='lower left',
    fontsize=7,
    frameon=True,             # 启用背景框
    facecolor='white',         # 白色背景
    framealpha=0.9,            # 60% 适度半透明遮罩
    edgecolor='none',          # 取消黑色边线，呈现现代轻盈感
    handletextpad=0.2,
    markerscale=0.75
)

# 学术紧密排版
plt.tight_layout(pad=0.3)

# ==================== 5. 图形保存 ====================
output_dir.mkdir(parents=True, exist_ok=True)
save_png = output_dir / "academic_decision_chart.png"
save_pdf = output_dir / "academic_decision_chart.pdf"

# 无损保存
plt.savefig(save_png, dpi=600, bbox_inches='tight')
plt.savefig(save_pdf, bbox_inches='tight')

print(f"\n[完成] 17cm 宽学术图谱（高对比 Cosmic_DeepSpace、无重复标注、无图例边框）已保存至:")
print(f" >> PNG 格式: {save_png}")
print(f" >> PDF 格式: {save_pdf}")

# ==================== 6. 自动导出精选设计方案 S2 & E2 到 Excel 表格 ====================
try:
    # 提取第 2 行候选数据 (DataFrame 索引为 1)
    s2_row = top_safe.iloc[1].copy()
    e2_row = top_explore.iloc[1].copy()

    # 构建输出表
    excel_df = pd.DataFrame([s2_row, e2_row])
    excel_df.insert(0, "Selection_ID", ["S2 (Confidence Domain)", "E2 (Exploratory Domain)"])

    # 格式化球化、回火状态输出
    excel_df["has_spheroidizing"] = excel_df["has_spheroidizing"].map({1: "已球化", 0: "未球化"})
    excel_df["has_tempering"] = excel_df["has_tempering"].map({1: "已回火", 0: "未回火"})

    # 中英文双语对照标准表头字典（完美集成了您指定的10大工艺全套参数，及Ac3, Ms等）
    EXCEL_HEADER_MAPPING = {
        "Selection_ID": "推荐代号 (Design Code)",
        "candidate_rank": "全局候选排名 (Global Rank)",
        "design_tier": "设计准入级别 (Design Tier)",
        # 化学组分
        "C_wt": "C (wt.%)", "Si_wt": "Si (wt.%)", "Mn_wt": "Mn (wt.%)",
        "Cr_wt": "Cr (wt.%)", "V_wt": "V (wt.%)", "Al_wt": "Al (wt.%)",
        "Ni_wt": "Ni (wt.%)", "Mo_wt": "Mo (wt.%)", "Cu_wt": "Cu (wt.%)", "W_wt": "W (wt.%)",
        # 物理特性临界点
        "Ac3": "临界转变温度 Ac3 (°C)",
        "Ms": "马氏体开始温度 Ms (°C)",
        # 完整的 10 项工艺控制链
        "T_austenitizing": "奥氏体化温度 Ta (°C)",
        "T_isothermal ": "等温相变温度 Ti (°C)",
        "ΔTA": "奥氏体温度偏差 ΔTA (°C)",
        "ΔTI": "等温温度偏差 ΔTI (°C)",
        "τ_norm": "无量纲等温时间 τ_norm",
        "has_spheroidizing": "球化退火状态 (Spheroidizing Status)",
        "has_tempering": "回火状态 (Tempering Status)",
        "T_tem": "回火温度 Tt (°C)",
        "T_tim": "回火时间 tt (min)",
        # 预测性能数据
        "Rm_pred_for_ranking": "抗拉强度 Rm 预测值 (MPa)",
        "Re_pred_for_ranking": "屈服强度 Re 预测值 (MPa)",
        "A_pred_for_ranking": "延伸率 A 预测值 (%)",
        "Ak_pred_for_ranking": "冲击功 Ak 预测值 (J/cm²)",
        "strong_plastic_product": "强塑积 (GPa·%)"
    }

    keep_cols = list(EXCEL_HEADER_MAPPING.keys())
    keep_cols = [c for c in keep_cols if c in excel_df.columns]

    final_excel_df = excel_df[keep_cols].rename(columns=EXCEL_HEADER_MAPPING)

    excel_path = output_dir / "selected_optimal_designs.xlsx"
    final_excel_df.to_excel(excel_path, index=False)
    print(f"[完成] 最优低成本钢种 S2 & E2 的全套工艺控制参数（含 10 大工艺细节）已成功导出至 Excel 表格:")
    print(f" >> Excel 格式: {excel_path}")
except Exception as exc:
    print(f"[警告] 自动导出 Excel 出现微小偏差: {exc}")

# ==================== 7. 控制台推荐卡片打印 ====================
print("\n" + "=" * 102)
print("      🔬 【材料主动学习人机协同】双轨并行（10+10）钢种设计实验推荐方案库 🔬")
print("=" * 102)

# --- A. Confidence Domain 推荐卡 ---
print("\n🟢 【轨道一：置信安全生产轨 (Confidence Domain - Tier A)】")
print("-" * 102)
for i, row in top_safe.iterrows():
    rank_idx = i + 1
    wt_list = [f"{el}: {row[f'{el}_wt']:.3f}%" for el in ALLOY_ELEMENTS if
               f"{el}_wt" in row and row[f"{el}_wt"] > 0.001]

    uq_Rm = row['Rm_pred_for_ranking'] - row['Rm_ranking_lower90']
    uq_Re = row['Re_pred_for_ranking'] - row['Re_ranking_lower90']
    uq_A = row['A_pred_for_ranking'] - row['A_ranking_lower90']
    uq_Ak = row['Ak_pred_for_ranking'] - row['Ak_ranking_lower90']

    # 特殊标识
    expert_tag = " ★ [材料学家终筛：低成本优选设计] ★" if rank_idx == 2 else ""

    print(
        f"推荐代号 [ S{rank_idx:<2} ] | 全局候选 Rank: {row['candidate_rank']:<4} | 强塑积: {row['strong_plastic_product']:.2f} GPa%{expert_tag}")
    print(f"  ● 合金组分 (wt.%): {', '.join(wt_list)}")
    print(
        f"  ● 热处理工艺: 奥氏体化 Ta={row['T_austenitizing']:.1f}°C | 等温相变 Ti={row['T_isothermal ']:.1f}°C | 回火: {'320°C×60min' if row['has_tempering'] == 1 else '未回火'}")
    print(f"  ● 四大性能预测均值与双向置信区间 (90%置信度):")
    print(f"    - 抗拉强度 Rm: {row['Rm_pred_for_ranking']:.1f} ± {uq_Rm:.1f} MPa")
    print(f"    - 屈服强度 Re: {row['Re_pred_for_ranking']:.1f} ± {uq_Re:.1f} MPa")
    print(f"    - 延伸率 A  : {row['A_pred_for_ranking']:.1f} ± {uq_A:.1f} %")
    print(f"    - 冲击功 Ak  : {row['Ak_pred_for_ranking']:.1f} ± {uq_Ak:.1f} J/cm²")
    print("." * 102)

# --- B. Exploratory Domain 推荐卡 ---
print("\n🟡 【轨道二：前沿突破探索轨 (Exploratory Domain - Tier B)】")
print("-" * 102)
for i, row in top_explore.iterrows():
    rank_idx = i + 1
    wt_list = [f"{el}: {row[f'{el}_wt']:.3f}%" for el in ALLOY_ELEMENTS if
               f"{el}_wt" in row and row[f"{el}_wt"] > 0.001]

    uq_Rm = row['Rm_pred_for_ranking'] - row['Rm_ranking_lower90']
    uq_Re = row['Re_pred_for_ranking'] - row['Re_ranking_lower90']
    uq_A = row['A_pred_for_ranking'] - row['A_ranking_lower90']
    uq_Ak = row['Ak_pred_for_ranking'] - row['Ak_ranking_lower90']

    # 特殊标识
    expert_tag = " ★ [材料学家终筛：低成本优选设计] ★" if rank_idx == 2 else ""

    print(
        f"推荐代号 [ E{rank_idx:<2} ] | 全局候选 Rank: {row['candidate_rank']:<4} | 强塑积: {row['strong_plastic_product']:.2f} GPa%{expert_tag}")
    print(f"  ● 合金组分 (wt.%): {', '.join(wt_list)}")
    print(
        f"  ● 热处理工艺: 奥氏体化 Ta={row['T_austenitizing']:.1f}°C | 等温相变 Ti={row['T_isothermal ']:.1f}°C | 回火: {'320°C×60min' if row['has_tempering'] == 1 else '未回火'}")
    print(f"  ● 四大性能预测均值与双向置信区间 (90%置信度):")
    print(f"    - 抗拉强度 Rm: {row['Rm_pred_for_ranking']:.1f} ± {uq_Rm:.1f} MPa")
    print(f"    - 屈服强度 Re: {row['Re_pred_for_ranking']:.1f} ± {uq_Re:.1f} MPa")
    print(f"    - 延伸率 A  : {row['A_pred_for_ranking']:.1f} ± {uq_A:.1f} %")
    print(f"    - 冲击功 Ak  : {row['Ak_pred_for_ranking']:.1f} ± {uq_Ak:.1f} J/cm²")
    print("." * 102)

# 渲染绘图结果
plt.show()