from pathlib import Path
import argparse
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
TARGETS = ['Rm', 'Re', 'A', 'Ak']
RISK = {'Rm': {0, 3, 6}, 'Re': {0, 3, 6}, 'A': {6, 7, 8}, 'Ak': {6, 7, 8}}
ZONE_COLORS = {0: '#D9947C', 3: '#CA6D54', 6: '#B84235', 7: '#8E2A3A', 8: '#571C35'}
NAVY, OTHER, BOX, EDGE = '#17304C', '#91A3AE', '#EAF2F7', '#9FB1BD'


def fingerprints(source):
    data = pd.read_csv(source)
    keys = ['target', 'held_zone', 'training_zone', 'feature']
    assert len(data) == 4320 and not data.duplicated(keys).any()
    assert set(data.target) == set(TARGETS)
    assert np.isfinite(data.mean_abs_shap).all() and (data.mean_abs_shap > 0).all()
    assert (data.held_zone != data.training_zone).all()
    for (_, zone), block in data.groupby(['target', 'held_zone']):
        assert len(block) == 120 and block.feature.nunique() == 15
        assert set(block.training_zone) == set(range(9)) - {zone}
        assert (block.groupby('feature').size() == 8).all()
    fp = data.groupby(['target', 'held_zone', 'feature']).mean_abs_shap.mean()
    return data, fp.rename('v').reset_index().rename(columns={'held_zone': 'zone'})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=HERE / 's5_training_zone_fingerprints.csv')
    parser.add_argument('--output-dir', type=Path, default=HERE)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    font_manager.findfont('Times New Roman', fallback_to_default=False)
    mpl.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman'], 'font.size': 8,
        'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 6.7,
        'legend.fontsize': 6.7, 'legend.frameon': False, 'axes.linewidth': .65,
        'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.unicode_minus': False,
        'mathtext.fontset': 'custom', 'mathtext.rm': 'Times New Roman',
        'mathtext.it': 'Times New Roman:italic', 'mathtext.bf': 'Times New Roman:bold',
    })
    raw, fp = fingerprints(args.data)
    fp.to_csv(args.output_dir / 's5_zone_fingerprint_derived.csv', index=False, encoding='utf-8-sig')
    fig = plt.figure(figsize=(14 / 2.54, 10 / 2.54))
    bottom = np.array(mpl.colors.to_rgb('#F0F5FC'))
    top = np.array(mpl.colors.to_rgb('#FBF0EF'))
    gradient = (bottom[None, :] + np.linspace(0, 1, 512)[:, None] * (top-bottom)[None, :])[:, None, :]
    selected_count = missing_count = 0
    panels = []
    axes = []
    for index, target in enumerate(TARGETS):
        row, column = divmod(index, 2)
        ax = fig.add_axes([.115 + column * .445, .555 if row == 0 else .105, .38 if column == 0 else .385, .41])
        axes.append(ax)
        piv = fp.loc[fp.target == target].pivot(index='zone', columns='feature', values='v').reindex(range(9))
        feats = list(piv.median().sort_values(ascending=False).index)
        lo, hi = piv.min().min() * .30, piv.max().max() * 1.40
        rng = np.random.default_rng(7)
        for i, feature in enumerate(feats):
            values = piv[feature].dropna().to_numpy()
            selected_count += len(values)
            if len(values) >= 3:
                ax.boxplot([values], positions=[i], widths=.64, vert=False,
                    showfliers=False, patch_artist=True, zorder=2,
                    medianprops={'color': NAVY, 'linewidth': .85},
                    boxprops={'facecolor': BOX, 'edgecolor': EDGE, 'linewidth': .6},
                    whiskerprops={'color': EDGE, 'linewidth': .6},
                    capprops={'color': EDGE, 'linewidth': .6})
            jitter = rng.normal(0, .115, 9)
            missing = piv.index[piv[feature].isna()].tolist()
            missing_count += len(missing)
            for k, zone in enumerate(missing):
                color = ZONE_COLORS[zone] if zone in RISK[target] else OTHER
                ax.scatter(lo * 1.35, i-.30+.075*k, s=9, facecolors='none',
                    edgecolors=color, linewidths=.6, alpha=.9, zorder=5)
            for zone in piv.index[piv[feature].notna()]:
                focus = zone in RISK[target]
                ax.scatter(piv.loc[zone, feature], i+jitter[zone],
                    s=13 if focus else 7, color=ZONE_COLORS[zone] if focus else OTHER,
                    alpha=.95 if focus else .55, edgecolors='white' if focus else 'none',
                    linewidths=.25 if focus else 0, zorder=4 if focus else 3)
        ax.set_xscale('log')
        ax.set_xlim(lo, hi)
        ax.set_ylim(len(feats)-.38, -0.55)
        ax.set_yticks(range(len(feats)), feats)
        plt.setp(ax.get_yticklabels(), rotation=45,  # 倾斜 45 度
                 ha='right', rotation_mode='anchor')
        if row == 1:
            ax.set_xlabel('Mean |SHAP|', labelpad=0)
            ax.xaxis.set_label_coords(0.5, -0.05)
        ax.tick_params(axis='both', which='both', direction='in', top=False,
            right=False, bottom=True, left=True, length=3, width=.6, pad=2)
        ax.tick_params(axis='x', which='minor', length=1.5, width=.45)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(.65)
            spine.set_color('black')
        ax.text(.95, .09, target, transform=ax.transAxes, ha='right',
            va='top', fontsize=8, fontweight='bold')
        ax.imshow(gradient, extent=(0, 1, 0, 1), transform=ax.transAxes,
            origin='lower', interpolation='bilinear', aspect='auto', zorder=0)
        ax.set_axisbelow(True)
        ax.grid(axis='x', which='major', color='#E8EBED', linewidth=.4)
        panels.append({'target': target, 'features': feats, 'selected_values': int(piv.count().sum()),
                       'unselected_markers': int(piv.isna().sum().sum())})
    handles = [Line2D([], [], marker='o', linestyle='', markersize=3.5,
        markerfacecolor=ZONE_COLORS[z], markeredgewidth=0, label=f'Z{z}') for z in ZONE_COLORS]
    handles.extend([
        Line2D([], [], marker='o', linestyle='', markersize=3,
            markerfacecolor=OTHER, markeredgewidth=0, label='Other'),
        Line2D([], [], marker='o', linestyle='', markersize=3.3,
            markerfacecolor='none', markeredgecolor=OTHER, markeredgewidth=.6, label='Not selected'),
    ])
    axes[0].legend(handles=handles, loc='upper left', bbox_to_anchor=(.015, .985), ncol=2,
        handlelength=.8, handletextpad=.35, columnspacing=.7, labelspacing=.3, borderaxespad=.15)
    prefix = args.output_dir / 'S5_lopzo_fingerprint_boxscatter_optimized'
    fig.savefig(prefix.with_suffix('.png'), dpi=600, facecolor='white')
    fig.savefig(prefix.with_suffix('.pdf'), facecolor='white')
    fig.savefig(prefix.with_suffix('.svg'), facecolor='white')
    fig.savefig(prefix.with_suffix('.tif'), dpi=600, facecolor='white', pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)
    qa = {'input_rows': len(raw), 'fingerprint_values': len(fp), 'plotted_selected_values': selected_count,
        'unselected_markers': missing_count, 'panels': panels, 'size_cm': [14, 14], 'layout': '2x2, compact spacing, bottom-row x-axis titles only', 'legend_position': 'First panel, upper left, two columns',
        'input_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
        'aggregation': 'Equal-weight mean across the eight training zones for each held-zone model and selected feature.',
        'boxplot': 'Selected-model values only; median, IQR and 1.5-IQR whiskers. All values are also shown as points.',
        'missing': 'An unselected feature is an open marker in the left margin, not a numeric zero.',
        'ordering': 'Per-target median importance descending, identical to the source plotting method.',
        'font': 'Times New Roman; mathematical Greek letters are retained in feature names.'}
    assert selected_count == len(fp) == 540
    (args.output_dir / 's5_plot_qa.json').write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Saved: {prefix}')
    print(f'All {len(raw)} input rows retained; {selected_count} fingerprint values and {missing_count} unselected markers plotted.')


if __name__ == '__main__':
    main()