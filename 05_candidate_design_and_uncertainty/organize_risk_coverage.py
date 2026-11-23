"""Consolidate existing 15-descriptor held-out errors and coverage distances."""

from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'rcv_descriptor_contribution'
LOPZO = ROOT.parent / '03_robustness_and_coverage' / 'lopzo_coverage'
parser = argparse.ArgumentParser()
parser.add_argument('--output-dir', type=Path, default=ROOT / 'risk_coverage_results')
OUT = parser.parse_args().output_dir.resolve()
OUT.mkdir(parents=True, exist_ok=True)

inputs = {
    'RCV': BASE / 'rcv_coverage' / 'sample_distances.csv',
    'LOCO': BASE / 'loco_coverage' / 'sample_distances.csv',
    'LOPZO': LOPZO / 'sample_distances.csv',
}
reference = pd.read_csv(BASE / 'three_scenario_risk_concentration.csv')
targets = ['Rm', 'Re', 'A', 'Ak']
bands = ['low', 'middle', 'high']
frames = []
for scenario, path in inputs.items():
    d = pd.read_csv(path)
    if scenario == 'LOPZO':
        d = d.rename(columns={'zone_id': 'physical_zone', 'outer_fold': 'fold'})
        d['repeat'] = 0
        d['specified_risk'] = [int(z in ({0, 3, 6} if t in ('Rm', 'Re') else {6, 7, 8})) for t, z in zip(d.target, d.physical_zone)]
    d['scenario'] = scenario
    d['n_source_rows'] = len(d)
    keep = ['scenario', 'target', 'repeat', 'fold', 'global_idx', 'comp_id', 'physical_zone',
            'specified_risk', 'y_true', 'y_pred', 'abs_error', 'nearest_distance',
            'training_loo_percentile', 'distance_band', 'nearest_train_global_idx',
            'nearest_same_composition']
    d = d[keep]
    assert d.target.isin(targets).all() and d.distance_band.isin(bands).all()
    assert d.abs_error.notna().all() and np.allclose(d.abs_error, abs(d.y_true - d.y_pred))
    assert not d.duplicated(['target', 'repeat', 'global_idx']).any()
    for target in targets:
        x = d[d.target == target]
        assert x.global_idx.nunique() == {'Rm': 189, 'Re': 176, 'A': 189, 'Ak': 199}[target]
        assert len(x) == x.global_idx.nunique() * (3 if scenario == 'RCV' else 1)
    frames.append(d)

all_samples = pd.concat(frames, ignore_index=True)
assert len(all_samples) == 3765
all_samples.to_csv(OUT / 'sample_level_all_scenarios.csv', index=False, encoding='utf-8-sig')


def summarize(g):
    a = g[g.specified_risk == 1]
    b = g[g.specified_risk == 0]
    ne, ae = len(g), g.abs_error.sum()
    if len(a) == 0 or len(b) == 0:
        risk_mae = np.nan if len(a) == 0 else a.abs_error.mean()
        other_mae = np.nan if len(b) == 0 else b.abs_error.mean()
        ratio = np.nan
    else:
        risk_mae, other_mae = a.abs_error.mean(), b.abs_error.mean()
        ratio = risk_mae / other_mae
    return pd.Series({
        'n_predictions': ne, 'n_unique': g.global_idx.nunique(), 'n_compositions': g.comp_id.nunique(),
        'risk_n_predictions': len(a), 'risk_n_unique': a.global_idx.nunique(),
        'risk_n_compositions': a.comp_id.nunique(),
        'other_n_predictions': len(b), 'other_n_unique': b.global_idx.nunique(),
        'other_n_compositions': b.comp_id.nunique(),
        'risk_sample_share': len(a) / ne,
        'risk_absolute_error_sum': a.abs_error.sum(),
        'other_absolute_error_sum': b.abs_error.sum(),
        'total_absolute_error_sum': ae,
        'risk_absolute_error_share': a.abs_error.sum() / ae,
        'risk_mae': risk_mae, 'other_mae': other_mae, 'risk_other_mae_ratio': ratio,
        'error_minus_sample_share_pp': 100 * (a.abs_error.sum() / ae - len(a) / ne),
    })

overall = all_samples.groupby(['scenario', 'target'], sort=False).apply(summarize, include_groups=False).reset_index()
strata = all_samples.groupby(['scenario', 'target', 'distance_band'], sort=False).apply(summarize, include_groups=False).reset_index()
overall['scenario'] = pd.Categorical(overall.scenario, ['RCV', 'LOCO', 'LOPZO'], ordered=True)
overall['target'] = pd.Categorical(overall.target, targets, ordered=True)
overall = overall.sort_values(['target', 'scenario']).reset_index(drop=True)
strata['scenario'] = pd.Categorical(strata.scenario, ['RCV', 'LOCO', 'LOPZO'], ordered=True)
strata['target'] = pd.Categorical(strata.target, targets, ordered=True)
strata['distance_band'] = pd.Categorical(strata.distance_band, bands, ordered=True)
strata = strata.sort_values(['target', 'scenario', 'distance_band']).reset_index(drop=True)

for row in overall.itertuples():
    ref = reference[(reference.scenario == row.scenario) & (reference.target == row.target)].iloc[0]
    assert row.n_predictions == ref.n_predictions
    assert row.risk_n_predictions == ref.risk_n_predictions
    assert np.isclose(row.risk_sample_share, ref.risk_sample_share)
    assert np.isclose(row.risk_absolute_error_share, ref.risk_absolute_error_share)
    assert np.isclose(row.risk_mae, ref.risk_mae)
    assert np.isclose(row.other_mae, ref.other_mae)

overall.to_csv(OUT / 'overall_12.csv', index=False, encoding='utf-8-sig')
strata.to_csv(OUT / 'distance_strata_all.csv', index=False, encoding='utf-8-sig')
visible = strata[(strata.scenario != 'LOCO') | (strata.distance_band == 'high')].copy()
visible.to_csv(OUT / 'distance_strata_for_figures.csv', index=False, encoding='utf-8-sig')

def md_table(data, cols, headers, fmt):
    lines = ['| ' + ' | '.join(headers) + ' |', '|' + '|'.join(['---'] * len(cols)) + '|']
    for row in data.itertuples(index=False):
        vals = [fmt[c](getattr(row, c)) if c in fmt else str(getattr(row, c)) for c in cols]
        lines.append('| ' + ' | '.join(vals) + ' |')
    return '\n'.join(lines)

overall_cols = ['target', 'scenario', 'risk_n_unique', 'other_n_unique', 'risk_n_compositions',
                'other_n_compositions', 'risk_sample_share', 'risk_absolute_error_share',
                'risk_mae', 'other_mae', 'risk_other_mae_ratio']
overall_headers = ['性能', '留出', '风险区记录', '其余区记录', '风险区成分', '其余区成分', '风险区样本份额',
                   '风险区绝对误差份额', '风险区MAE', '其余区MAE', 'MAE比']
fmt = {'risk_sample_share': lambda x: f'{100*x:.1f}%', 'risk_absolute_error_share': lambda x: f'{100*x:.1f}%',
       'risk_mae': lambda x: f'{x:.2f}', 'other_mae': lambda x: f'{x:.2f}',
       'risk_other_mae_ratio': lambda x: '—' if pd.isna(x) else f'{x:.2f}'}

strata_cols = ['target', 'scenario', 'distance_band', 'risk_n_predictions', 'other_n_predictions',
               'risk_n_unique', 'other_n_unique', 'risk_n_compositions', 'other_n_compositions',
               'risk_sample_share', 'risk_absolute_error_share', 'risk_mae', 'other_mae', 'risk_other_mae_ratio']
strata_headers = ['性能', '留出', '距离层', '风险区预测行', '其余区预测行', '风险区唯一记录', '其余区唯一记录',
                  '风险区成分', '其余区成分', '风险区样本份额', '风险区误差份额', '风险区MAE', '其余区MAE', 'MAE比']
report = [
    '# 审稿意见 1–2：原稿 15 维模型三场景误差与覆盖度完整整理',
    '',
    '## 口径',
    '',
    '- 原稿 15 维知识特征模型的外层留出预测：RCV 为 3 次重复 × 10 折，LOCO 和 LOPZO 每个有效记录各预测一次。',
    '- 风险区沿用原稿 LOPZO 讨论的区域：Rm/Re 为 Zone 0、3、6（Short-τ）；A/Ak 为 Zone 6、7、8（High-ΔTI）。',
    '- 覆盖度按每个外层训练集中的原始 17 维变量最近邻距离计算；RCV、LOCO、LOPZO 均采用训练折内拟合的距离参照。',
    '- “绝对误差份额”＝风险区绝对误差之和／本组全部绝对误差之和。“样本份额”按预测行计算，RCV 的三次预测不当作三条独立材料记录。',
    '- 全部逐样本结果、12 行总表、全部距离层及绘图选择表均另存为 CSV；LOCO 的低、中距离层只保留在全表，不进入主图。',
    '',
    '## 总体：四性能 × 三场景', '',
    md_table(overall, overall_cols, overall_headers, fmt), '',
    '## RCV：低／中／高距离层', '',
    md_table(strata[strata.scenario == 'RCV'], strata_cols, strata_headers, fmt), '',
    '## LOCO：高距离层（主图使用）', '',
    md_table(strata[(strata.scenario == 'LOCO') & (strata.distance_band == 'high')], strata_cols, strata_headers, fmt), '',
    'LOCO 低、中距离层的完整原始统计见 `distance_strata_all.csv`；风险区在这些层中极少或不存在。', '',
    '## LOPZO：低／中／高距离层（完整归档）', '',
    md_table(strata[strata.scenario == 'LOPZO'], strata_cols, strata_headers, fmt), '',
    '## 读图重点', '',
    '1. 总体上，四项目标的指定区域仅含 23.9%–36.2% 的有效记录，却在三种验证设置下承担 31.1%–51.7% 的绝对误差。',
    '2. RCV 分层后，Rm、Re、A 的风险区 MAE 在低、中、高三层均高于同层其余区；Ak 的低距离层方向不同，中、高层差异较弱。',
    '3. LOCO 留出记录约九成处于高距离层；在这一层，风险区 MAE／其余区 MAE 为 Rm 2.59、Re 1.69、A 1.45、Ak 1.18。',
    '4. LOPZO 的完整距离分层保留在表中，主图是否放入可根据版面选择；本分析用于描述已有误差的区域组织，不承担独立预报能力的结论。',
]
(OUT / '完整数据与读图说明.md').write_text('\n'.join(report), encoding='utf-8')

manifest = {'inputs': {key: {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for key, path in inputs.items()},
            'reference': str(BASE / 'three_scenario_risk_concentration.csv'),
            'output_rows': {'sample_level': len(all_samples), 'overall': len(overall), 'strata_all': len(strata), 'strata_for_figures': len(visible)}}
(OUT / 'analysis_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
print('PASS', manifest['output_rows'])
print(OUT)
