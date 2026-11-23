import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = ['Rm', 'Re', 'A', 'Ak']
BASELINE = ['C', 'Si', 'Mn', 'Cr', 'V', 'Al', 'Ni', 'Mo', 'Cu', 'W',
            'Ta', 'Ti', 'ti', 'Tt', 'tt', 'sph', 'tmp']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def metrics(frame, prediction='y_pred'):
    y, pred = frame.y_true.to_numpy(), frame[prediction].to_numpy()
    error = y - pred
    return {'r2': float(1 - np.sum(error ** 2) / np.sum((y - y.mean()) ** 2)),
            'mae': float(np.mean(np.abs(error))), 'rmse': float(np.sqrt(np.mean(error ** 2)))}


def shap_summary(frame, names, selected, path):
    values = frame[['shap_' + name for name in names]].abs().mean().to_numpy()
    shares = values / values.sum() * 100 if values.sum() > 0 else np.zeros(len(values))
    result = pd.DataFrame({'feature': names, 'shap_importance': values, 'shap_share_percent': shares,
                           'selected_folds': [sum(name in s for s in selected) for name in names],
                           'n_folds': len(selected),
                           'selection_frequency': [sum(name in s for s in selected) / len(selected) for name in names]})
    result = result.sort_values('shap_importance', ascending=False).reset_index(drop=True)
    result.insert(0, 'rank', np.arange(1, len(result) + 1))
    result.to_csv(path, index=False, encoding='utf-8-sig')
    return result


def main():
    package = Path(__file__).resolve().parent
    baseline, knowledge = package / 'unified_results_17Baseline', package / 'unified_results_rcv3'
    bm, km = read(baseline / 'run_manifest.json'), read(knowledge / 'run_manifest.json')
    changes = {key: [value, bm['parameters'].get(key)] for key, value in km['parameters'].items()
               if value != bm['parameters'].get(key)}
    if changes != {'feature_mode': ['knowledge15', 'baseline17']} or bm['input_sha256'] != km['input_sha256'] or bm['versions'] != km['versions']:
        raise ValueError('Baseline and knowledge protocols do not match')
    output = baseline / 'evaluation'
    shap_output = package / 'shap_analysis_17_vs_15_nested'
    shap_output.mkdir(exist_ok=True)
    rows, repeat_rows, paired_frames, comparisons = [], [], [], []
    for target in TARGETS:
        b = pd.read_csv(baseline / 'residuals' / f'{target}_cv_predictions.csv')
        k = pd.read_csv(knowledge / 'residuals' / f'{target}_cv_predictions.csv')
        keys = ['bootstrap_id', 'global_idx']
        if b.duplicated(keys).any() or k.duplicated(keys).any():
            raise ValueError(f'Duplicate outer-test predictions: {target}')
        paired = b.merge(k, on=keys, suffixes=('_17Baseline', '_15Knowledge'), how='outer', indicator=True)
        if not (paired['_merge'] == 'both').all() or not np.allclose(paired.y_true_17Baseline, paired.y_true_15Knowledge):
            raise ValueError(f'Unmatched samples or labels: {target}')
        if not (paired.outer_fold_17Baseline == paired.outer_fold_15Knowledge).all():
            raise ValueError(f'Outer random folds differ: {target}')
        expected = bm['parameters']['rcv_repeats'] * bm['parameters']['rcv_splits']
        paths = sorted((baseline / 'folds' / target / 'rcv').glob('*/complete.json'))
        if len(paths) != expected:
            raise ValueError(f'Baseline fold count differs: {target}')
        bs, ks, candidates = [], [], None
        for marker in paths:
            bp = marker.parent
            kp = knowledge / 'folds' / target / 'rcv' / bp.name
            ba, ka = read(bp / 'split_audit.json'), read(kp / 'split_audit.json')
            for field in ['outer_train_indices', 'outer_test_indices']:
                if ba[field] != ka[field]:
                    raise ValueError(f'Outer partitions differ: {target}/{bp.name}')
            bmeta, kmeta = read(bp / 'model_config.json'), read(kp / 'model_config.json')
            if bmeta['feature_names'] != BASELINE or len(kmeta['feature_names']) != 15:
                raise ValueError(f'Unexpected feature lists: {target}/{bp.name}')
            if bmeta['actual_model_params']['random_seed'] != kmeta['actual_model_params']['random_seed']:
                raise ValueError(f'Model seed differs: {target}/{bp.name}')
            if len(ba['inner_splits']) != len(ka['inner_splits']) or any(
                bi['train_indices'] != ki['train_indices'] or bi['validation_indices'] != ki['validation_indices']
                for bi, ki in zip(ba['inner_splits'], ka['inner_splits'])
            ):
                raise ValueError(f'Inner random splits differ: {target}/{bp.name}')
            bs.append(set(bmeta['feature_names']))
            ks.append(set(kmeta['feature_names']))
            names = read(kp / 'outer_features.json')['candidate_features']
            if candidates is not None and candidates != names:
                raise ValueError(f'Knowledge feature coordinates differ: {target}')
            candidates = names
        values = {'17Baseline': metrics(b), '15Knowledge': metrics(k)}
        for label, root in [('17Baseline', baseline), ('15Knowledge', knowledge)]:
            saved = read(root / 'evaluation' / target / 'rcv_unfiltered_overall.json')
            if not saved['complete'] or not all(np.isclose(value, saved['overall'][name]) for name, value in values[label].items()):
                raise ValueError(f'Saved metric differs: {target}/{label}')
        for repeat in sorted(b.bootstrap_id.unique()):
            bv, kv = metrics(b[b.bootstrap_id == repeat]), metrics(k[k.bootstrap_id == repeat])
            repeat_rows.append({'target': target, 'repeat_id': int(repeat),
                                **{name + '_17Baseline': value for name, value in bv.items()},
                                **{name + '_15Knowledge': value for name, value in kv.items()},
                                'delta_r2_15_minus_17': kv['r2'] - bv['r2']})
        br = [row['r2_17Baseline'] for row in repeat_rows if row['target'] == target]
        kr = [row['r2_15Knowledge'] for row in repeat_rows if row['target'] == target]
        rows.append({'target': target, 'n_valid_samples': b.global_idx.nunique(), 'n_predictions_each': len(b),
                     'n_repeats': bm['parameters']['rcv_repeats'], 'n_outer_folds_each': expected,
                     **{name + '_17Baseline': value for name, value in values['17Baseline'].items()},
                     **{name + '_15Knowledge': value for name, value in values['15Knowledge'].items()},
                     'delta_r2_15_minus_17': values['15Knowledge']['r2'] - values['17Baseline']['r2'],
                     'r2_retention_percent': values['15Knowledge']['r2'] / values['17Baseline']['r2'] * 100
                     if values['17Baseline']['r2'] > 0 else None,
                     'r2_sd_17Baseline_ddof1': float(np.std(br, ddof=1)),
                     'r2_sd_15Knowledge_ddof1': float(np.std(kr, ddof=1))})
        paired.insert(0, 'target', target)
        paired_frames.append(paired[['target', *keys, 'outer_fold_17Baseline', 'y_true_17Baseline',
                                     'y_pred_17Baseline', 'y_pred_15Knowledge',
                                     'residual_17Baseline', 'residual_15Knowledge']])
        bshap = shap_summary(b, BASELINE, bs, shap_output / f'{target}_17Baseline_SHAP_summary.csv')
        kshap = shap_summary(k, candidates, ks, shap_output / f'{target}_15Knowledge_SHAP_summary.csv')
        combined = bshap.merge(kshap, on='feature', how='outer', suffixes=('_17Baseline', '_15Knowledge'))
        combined.insert(0, 'target', target)
        combined['in_17Baseline'] = combined.shap_importance_17Baseline.notna()
        combined['in_15Knowledge_candidates'] = combined.shap_importance_15Knowledge.notna()
        combined.to_csv(shap_output / f'{target}_17Baseline_vs_15Knowledge_SHAP_compare.csv', index=False, encoding='utf-8-sig')
        comparisons.append(combined)
    pd.DataFrame(rows).to_csv(output / '17Baseline_vs_15Knowledge_metrics.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(repeat_rows).to_csv(output / '17Baseline_vs_15Knowledge_per_repeat.csv', index=False, encoding='utf-8-sig')
    pd.concat(paired_frames, ignore_index=True).to_csv(output / '17Baseline_vs_15Knowledge_paired_predictions.csv', index=False, encoding='utf-8-sig')
    pd.concat(comparisons, ignore_index=True).to_csv(shap_output / 'ALL_TARGETS_17Baseline_vs_15Knowledge_SHAP_compare.csv', index=False, encoding='utf-8-sig')
    report = {'checks_passed': True, 'metric_source': 'matched outer-test predictions', 'results': rows,
              'r2_retention_note': '100 * R2_15 / R2_17, only if R2_17 > 0; not capped at 100%',
              'repeat_sd_note': 'ddof=1 over complete repeats; split variation, not an independent-sample confidence interval',
              'shap_source': 'outer-test mean absolute SHAP; bias excluded; unselected knowledge features contribute zero',
              'knowledge_feature_note': '15 features per fold; the cross-fold summary uses all 26 candidate coordinates',
              'physical_category_note': 'Apply a documented fixed feature grouping before reporting thermodynamic/kinetic/coupling shares'}
    (output / '17Baseline_vs_15Knowledge_report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(pd.DataFrame(rows)[['target', 'r2_17Baseline', 'r2_15Knowledge', 'r2_retention_percent']].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
