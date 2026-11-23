import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TARGETS = ('Rm', 'Re', 'A', 'Ak')
RISK_ZONES = {'Rm': {0, 3, 6}, 'Re': {0, 3, 6}, 'A': {6, 7, 8}, 'Ak': {6, 7, 8}}
OFFSETS = {
    'Rm': {0: (6, 5), 1: (-3, 11), 2: (5, 3), 3: (5, 6), 4: (5, 5), 5: (5, 4), 6: (-10, 5), 7: (-3, -9), 8: (-12, 11)},
    'Re': {0: (6, 5), 1: (-3, 11), 2: (5, -8), 3: (5, 6), 4: (5, 7), 5: (-15, -5), 6: (-10, 5), 7: (-3, -8), 8: (-15, -2)},
    'A': {0: (-7, 6), 1: (-15, 6), 2: (5, -12), 3: (-17, 5), 4: (5, 5), 5: (-15, 8), 6: (-10, 5), 7: (-18, 7), 8: (-18, 7)},
    'Ak': {0: (-16, 4), 1: (-17, -2), 2: (4, 4), 3: (-8, -10), 4: (4, -9), 5: (4, 4), 6: (-17, -6), 7: (-10, 5), 8: (-16, 4)},
}


def summarize(data):
    required = {'target', 'global_idx', 'comp_id', 'physical_zone', 'abs_error',
                'nearest_distance', 'training_loo_percentile', 'nearest_same_composition'}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    if data.duplicated(['target', 'global_idx']).any():
        raise ValueError('Each target and sample must have one held-out prediction')
    rows = []
    for (target, zone), group in data.groupby(['target', 'physical_zone'], sort=True):
        rows.append({
            'target': target,
            'zone_id': int(zone),
            'n': len(group),
            'n_compositions': group.comp_id.nunique(),
            'mae': group.abs_error.mean(),
            'rmse': np.sqrt(np.mean(np.square(group.abs_error))),
            'distance_median': group.nearest_distance.median(),
            'distance_mean': group.nearest_distance.mean(),
            'percentile_median': group.training_loo_percentile.median(),
            'high_percentile_fraction': (group.training_loo_percentile > 2 / 3).mean(),
            'same_comp_nearest_fraction': group.nearest_same_composition.astype(bool).mean(),
        })
    result = pd.DataFrame(rows)
    if set(result.target) != set(TARGETS) or not all(set(g.zone_id) == set(range(9)) for _, g in result.groupby('target')):
        raise ValueError('Expected nine process zones for each target')
    return result


def plot(summary, output):
    mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 9, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 4, figsize=(6.65, 2.0), layout='constrained')
    titles = {'Rm': r'$R_m$', 'Re': r'$R_e$', 'A': r'$A$', 'Ak': r'$A_k$'}
    cmap = mpl.colors.LinearSegmentedColormap.from_list('blue_red', ['#e7eff6', '#f7eeee', '#f7e1df'])
    for ax, target in zip(axes, TARGETS):
        values = summary[summary.target == target]
        upper = values.mae.max() * 1.18
        ax.imshow(np.linspace(0, 1, 512)[None, :], extent=[.12, 1.02, 0, upper],
                  origin='lower', cmap=cmap, aspect='auto', interpolation='bilinear', zorder=0)
        for row in values.itertuples(index=False):
            risk = row.zone_id in RISK_ZONES[target]
            color = '#111111' if risk else '#7890a0'
            ax.scatter(row.percentile_median, row.mae, s=25, facecolors=color,
                       edgecolors=color, linewidths=.6, zorder=3)
            ax.annotate(f'Z{row.zone_id}', (row.percentile_median, row.mae),
                        xytext=OFFSETS[target][row.zone_id], textcoords='offset points',
                        fontsize=7.5, color='#111111' if risk else '#45555f')
        ax.set(xlim=(.12, 1.02), ylim=(0, upper), xlabel='Distance percentile', ylabel='MAE')
        ax.xaxis.label.set_size(8)
        ax.yaxis.label.set_size(8)
        ax.set_title(titles[target], loc='center', fontsize=11, fontweight='bold')
        ax.set_box_aspect(1)
        ax.grid(True, color='#ffffff', linewidth=.65, alpha=.75)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(True)
        ax.tick_params(axis='both', direction='in', top=False, right=False, labelsize=7.5)
    for ext in ('svg', 'pdf', 'png'):
        fig.savefig(output / f'zone_coverage.{ext}', dpi=600, facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = pd.read_csv(args.input, dtype={'comp_id': str})
    summary = summarize(data)
    args.output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output / 'zone_summary.csv', index=False, encoding='utf-8-sig')
    plot(summary, args.output)


if __name__ == '__main__':
    main()
