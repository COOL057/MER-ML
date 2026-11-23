import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
from catboost import CatBoostRegressor, Pool
from sklearn.covariance import LedoitWolf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from feature_stability_selection import select_features, export_selection
from train_engineering_228 import fit_pipeline

TARGETS = ['Rm', 'Re', 'A', 'Ak']
BASE = ['C', 'Si', 'Mn', 'Cr', 'V', 'Al', 'Ni', 'Mo', 'Cu', 'W',
        'Z', 'Ve', 'χP', 'δb', 'δf', 'ΔFe', 'ΔC', 'Ta', 'Ti', 'ti',
        'ΔTA', 'ΔTI', 'τ', 'Ms', 'Tt', 'tt', 'sph', 'tmp']
FULL_BASE = [f for f in BASE if f not in ['ti', 'Ms']]
INTERACTIONS = ['Ceq', 'Ta_Ti_diff', 'C_DTI', 'DC_Ti', 'C_Cr']
FULL = FULL_BASE + INTERACTIONS
BASELINE = ['C', 'Si', 'Mn', 'Cr', 'V', 'Al', 'Ni', 'Mo', 'Cu', 'W',
            'Ta', 'Ti', 'ti', 'Tt', 'tt', 'sph', 'tmp']
ZONE_LABELS = [f'{d} × {t}' for d in ['Low ΔTI', 'Mid ΔTI', 'High ΔTI']
               for t in ['Short τ', 'Mid τ', 'Long τ']]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(tmp, path)


def metrics(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    return {'r2': float(r2_score(y, pred)) if len(y) > 1 and np.var(y) > 0 else None,
            'mae': float(mean_absolute_error(y, pred)),
            'rmse': float(np.sqrt(mean_squared_error(y, pred))), 'n_samples': len(y)}


def load_data(args):
    root = Path(args.data_dir).resolve()
    files = [root / 'X_28dim.npy', Path(args.composition_csv).resolve(),
             Path(args.process_data).resolve(), Path(args.feature_config).resolve()]
    files += [root / f'y_{t}.npy' for t in TARGETS]
    for f in files:
        if not f.is_file():
            raise FileNotFoundError(f)
    X = np.load(files[0], allow_pickle=False)
    if X.ndim != 2 or X.shape[1] != 28 or np.isinf(X).any():
        raise ValueError('X_28dim.npy must be a finite-or-NaN n × 28 matrix')
    feat_cfg = json.loads(files[3].read_text(encoding='utf-8'))
    if feat_cfg['base_features'] != BASE:
        raise ValueError('feature_config base_features do not match the X_28dim column contract')
    comp = pd.read_csv(files[1]).apply(pd.to_numeric, errors='raise').fillna(0).round(3)
    proc = pd.read_excel(files[2])
    if len(comp) != len(X) or len(proc) != len(X):
        raise ValueError('Composition/process row counts must match X_28dim')
    dti = proc['ΔTI'].to_numpy(dtype=float)
    tau = proc['τ_norm'].to_numpy(dtype=float)
    if not np.isfinite(dti).all() or not np.isfinite(tau).all():
        raise ValueError('Zone coordinates must be finite')
    if not np.allclose(dti, X[:, 21], rtol=1e-6, atol=1e-5) or not np.allclose(tau, X[:, 22], rtol=1e-6, atol=1e-5):
        raise ValueError('Process coordinates disagree with X_28dim; check row alignment/version')
    comp_id = comp.apply(lambda row: '_'.join(f'{v:.3f}' for v in row), axis=1).to_numpy()
    counts = pd.Series(comp_id).value_counts()
    tier = np.array(['A' if counts[c] >= 6 else 'B' if counts[c] >= 3 else 'C' for c in comp_id])
    if args.zone_boundaries is None:
        args.zone_boundaries = np.percentile(dti, [33.33, 66.67]).tolist() + np.percentile(tau, [33.33, 66.67]).tolist()
        args.zone_partition_source = 'input process coordinates, label-free 33.33/66.67 percentiles; frozen before folds'
    else:
        args.zone_partition_source = 'explicit CLI boundaries; frozen before folds'
    b = args.zone_boundaries
    if not b[0] < b[1] or not b[2] < b[3]:
        raise ValueError('Zone boundaries must be increasing')
    zone = np.digitize(dti, b[:2]) * 3 + np.digitize(tau, b[2:])
    interactions = [X[:, 0] + X[:, 2] / 6 + (X[:, 3] + X[:, 7] + X[:, 4]) / 5 + (X[:, 6] + X[:, 8]) / 15,
                    X[:, 17] - X[:, 18], X[:, 0] * X[:, 21], X[:, 16] * X[:, 18], X[:, 0] * X[:, 3]]
    X31 = np.column_stack([X[:, [BASE.index(f) for f in FULL_BASE]]] + interactions)
    y = {t: np.load(root / f'y_{t}.npy', allow_pickle=False).reshape(-1) for t in TARGETS}
    if any(len(v) != len(X) or np.isinf(v).any() for v in y.values()):
        raise ValueError('Invalid target lengths or infinite target values')
    return {'X28': X, 'X31': X31, 'y': y, 'comp_id': comp_id, 'zone': zone,
            'dti': dti, 'tau': tau, 'tier': tier, 'files': files}


def feature_matrix(data, target, mode):
    if mode == 'baseline17':
        return data['X28'][:, [BASE.index(f) for f in BASELINE]], BASELINE.copy()
    if mode == 'full31':
        return data['X31'], FULL.copy()
    remove = {'C', 'tmp', 'χP'} | ({'δf', 'ΔC'} if target in ['Rm', 'Re'] else {'δb', 'ΔFe'})
    names = [f for f in FULL if f not in remove]
    return data['X31'][:, [FULL.index(f) for f in names]], names


def split_indices(indices, groups, n_splits, seed):
    indices = np.asarray(indices, dtype=int)
    if groups is None:
        k = min(n_splits, len(indices))
        if k < 2:
            raise ValueError('At least two training samples are required')
        return [(indices[a], indices[b]) for a, b in KFold(k, shuffle=True, random_state=seed).split(indices)]
    unique = np.unique(groups[indices])
    k = min(n_splits, len(unique))
    if k < 2:
        raise ValueError('At least two training groups are required')
    np.random.default_rng(seed).shuffle(unique)
    result = []
    for bucket in np.array_split(unique, k):
        mask = np.isin(groups[indices], bucket)
        result.append((indices[~mask], indices[mask]))
    return result


def model(params, iterations, args, seed, early_stop=None):
    params = {k: v for k, v in params.items() if k not in ['iterations', 'avg_best_iteration']}
    m = CatBoostRegressor(**params, iterations=iterations, loss_function='RMSE',
                          random_seed=seed, task_type='CPU', thread_count=args.threads,
                          verbose=False, allow_writing_files=False)
    if early_stop is not None:
        m.set_params(early_stopping_rounds=early_stop)
    return m




def cached_selection(path, X, y, names, ids, groups, args, seed):
    if path.exists():
        d = json.loads(path.read_text(encoding='utf-8'))
        if d['train_indices'] != ids.tolist() or d['candidate_features'] != names:
            raise ValueError(f'Incompatible selector cache: {path}')
        return d
    result = select_features(X, y, names, ids, groups, args, seed)
    write_json(path, result)
    return result




def novelty(data, train_ids, test_ids):
    ids = data['comp_id']
    selected = np.unique(ids[train_ids])
    vectors = np.array([data['X28'][train_ids[ids[train_ids] == c][0], :10] for c in selected])
    if not np.isfinite(vectors).all():
        raise ValueError('Element composition values must be finite')
    cov = LedoitWolf().fit(vectors)
    dif = data['X28'][test_ids, :10] - cov.location_
    return np.sqrt(np.maximum(0, np.einsum('ij,jk,ik->i', dif, cov.precision_, dif)))


def outer_splits(data, target, scenario, args):
    valid = np.flatnonzero(np.isfinite(data['y'][target]))
    if scenario == 'rcv':
        for b in range(args.rcv_repeats):
            for i, (tr, te) in enumerate(split_indices(valid, None, args.rcv_splits, args.seed + b)):
                yield f'repeat_{b:02d}_fold_{i:02d}', tr, te, None, b, i
    else:
        groups = data['comp_id'] if scenario == 'loco' else data['zone']
        for i, g in enumerate(np.unique(groups[valid])):
            yield f'fold_{i:03d}', valid[groups[valid] != g], valid[groups[valid] == g], groups, 0, i


def records(data, target, scenario, test_ids, pred, shap, meta, fold, repeat, fold_id, train_ids):
    y = data['y'][target]
    nov = novelty(data, train_ids, test_ids)
    rows = []
    for j, idx in enumerate(test_ids):
        c, z = data['comp_id'][idx], int(data['zone'][idx])
        r = float(y[idx] - pred[j])
        row = {'global_idx': int(idx), 'comp_id': c, 'comp_short': c, 'tier': data['tier'][idx],
               'dti': float(data['dti'][idx]), 'tau': float(data['tau'][idx]), 'zone_id': z,
               'zone_label': ZONE_LABELS[z], 'novelty': float(nov[j]), 'y_true': float(y[idx]),
               'y_pred': float(pred[j]), 'residual': r, 'abs_residual': abs(r),
               'outer_fold': fold, 'fold_id': fold_id, 'bootstrap_id': repeat, 'scenario': scenario}
        if shap is not None:
            row.update({f'shap_{name}': 0. for name in list(dict.fromkeys(FULL + BASELINE))})
            row.update({f'shap_{name}': float(shap[j, k]) for k, name in enumerate(meta['feature_names'])})
            row['shap_bias'] = float(shap[j, -1])
        rows.append(row)
    return rows


def verify_complete(path):
    marker = json.loads((path / 'complete.json').read_text(encoding='utf-8'))
    for name, digest in marker['files_sha256'].items():
        if not (path / name).is_file() or sha256(path / name) != digest:
            raise ValueError(f'Completed fold file changed or missing: {path / name}')


def mark_complete(path, **fields):
    names = ['model.cbm', 'model_config.json', 'outer_features.json', 'split_audit.json', 'optimization_trials.csv']
    if (path / 'predictions.csv').exists():
        names.append('predictions.csv')
    names += [p.name for p in sorted(path.glob('inner_*_features.json'))]
    write_json(path / 'complete.json', {**fields, 'files_sha256': {n: sha256(path / n) for n in names}})




def summarize_features(output, data, target, scenario, paths, args):
    root = output.parent / 'stability_selection_results' if output.name == 'unified_results_clean' else output / 'stability_selection_results'
    directory = root / target / scenario
    directory.mkdir(parents=True, exist_ok=True)
    _, names = feature_matrix(data, target, args.feature_mode)
    rows, sets, sizes = [], [], []
    for p in paths:
        selection = json.loads((p / 'outer_features.json').read_text(encoding='utf-8'))
        export_selection(selection, directory / p.name, target)
        audit = json.loads((p / 'split_audit.json').read_text(encoding='utf-8'))
        selected = set(selection['selected_features'])
        sets.append(selected)
        sizes.append(len(audit['outer_train_indices']))
        held_ids = audit['outer_test_indices']
        held = sorted(set(data['comp_id'][held_ids].tolist())) if scenario == 'loco' else sorted(set(data['zone'][held_ids].tolist()))
        row = {'target': target, 'scenario': scenario, 'outer_fold': p.name,
               'n_train': len(audit['outer_train_indices']), 'n_test': len(held_ids),
               'held_out_groups': json.dumps(held, ensure_ascii=False),
               'selected_features': json.dumps(selection['selected_features'], ensure_ascii=False)}
        row.update({f'selected_{f}': int(f in selected) for f in names})
        rows.append(row)
    pd.DataFrame(rows).to_csv(directory / 'fold_selected_features.csv', index=False)
    frequency = [{'target': target, 'scenario': scenario, 'feature': f,
                  'selected_folds': sum(f in s for s in sets), 'n_folds': len(sets),
                  'selection_frequency': sum(f in s for s in sets) / len(sets),
                  'selected_in_all_folds': all(f in s for s in sets)} for f in names]
    pd.DataFrame(frequency).sort_values(['selection_frequency', 'feature'], ascending=[False, True]).to_csv(
        directory / 'feature_selection_frequency.csv', index=False)
    overlaps = []
    for i, a in enumerate(sets):
        for j in range(i + 1, len(sets)):
            b = sets[j]
            overlaps.append({'fold_a': paths[i].name, 'fold_b': paths[j].name,
                             'n_shared': len(a & b), 'jaccard': len(a & b) / len(a | b)})
    pd.DataFrame(overlaps, columns=['fold_a', 'fold_b', 'n_shared', 'jaccard']).to_csv(directory / 'pairwise_feature_overlap.csv', index=False)
    write_json(directory / 'feature_stability_summary.json', {
        'target': target, 'scenario': scenario, 'n_folds': len(sets),
        'features_in_all_folds': sorted(set.intersection(*sets)),
        'features_in_any_fold': sorted(set.union(*sets)),
        'mean_pairwise_jaccard': float(np.mean([r['jaccard'] for r in overlaps])) if overlaps else None,
        'interpretation': 'Descriptive cross-fold consistency; not a confidence interval or independent replicate count.'})
    all_freq, all_fold = [], []
    for target_dir in sorted(root.iterdir()):
        if not target_dir.is_dir():
            continue
        for scenario_dir in sorted(target_dir.iterdir()):
            if scenario_dir.is_dir() and (scenario_dir / 'feature_selection_frequency.csv').exists():
                all_freq.append(pd.read_csv(scenario_dir / 'feature_selection_frequency.csv'))
                all_fold.append(pd.read_csv(scenario_dir / 'fold_selected_features.csv'))
    pd.concat(all_freq, ignore_index=True).to_csv(root / 'all_targets_feature_selection_frequency.csv', index=False)
    pd.concat(all_fold, ignore_index=True).to_csv(root / 'all_targets_fold_selected_features.csv', index=False)


def aggregate(output, data, target, scenario, paths, args):
    df = pd.concat([pd.read_csv(p / 'predictions.csv', dtype={'comp_id': str, 'comp_short': str}) for p in paths], ignore_index=True)
    expected = np.flatnonzero(np.isfinite(data['y'][target]))
    count = args.rcv_repeats if scenario == 'rcv' else 1
    coverage = df.groupby('global_idx').size()
    complete = set(coverage.index) == set(expected) and bool((coverage == count).all())
    evaldir = output / 'evaluation' / target
    evaldir.mkdir(parents=True, exist_ok=True)
    result = {'overall': metrics(df.y_true, df.y_pred), 'complete': complete,
              'scenario': scenario, 'n_completed_folds': len(paths), 'feature_mode': args.feature_mode}
    if scenario == 'rcv':
        repeats = [{'repeat_id': int(b), **metrics(g.y_true, g.y_pred)} for b, g in df.groupby('bootstrap_id')]
        r2 = [d['r2'] for d in repeats if d['r2'] is not None]
        result.update({'per_repeat': repeats, 'mean_repeat_r2': float(np.mean(r2)) if r2 else None,
                       'std_repeat_r2': float(np.std(r2)) if r2 else None})
    else:
        group_key = 'comp_id' if scenario == 'loco' else 'zone_id'
        per = [{group_key: g.item() if isinstance(g, np.generic) else g, **metrics(a.y_true, a.y_pred)}
               for g, a in df.groupby(group_key)]
        write_json(evaldir / f'{scenario}_unfiltered_per_{"composition" if scenario == "loco" else "zone"}.json', per)
        if scenario == 'loco':
            result['n_eligible'] = len(per)
    if scenario == 'lopzo':
        result.update({'dti_thresholds': args.zone_boundaries[:2], 'tau_thresholds': args.zone_boundaries[2:]})
        if args.compute_shap:
            cols = [f'shap_{f}' for f in FULL]
            fingerprints = []
            for z, a in df.groupby('zone_id'):
                v = a[cols].abs().mean().to_numpy()
                fingerprints.append({'zone_id': int(z), 'n_samples': len(a), 'mean_abs_shap': v.tolist(),
                                     'mean_shap': a[cols].mean().tolist(),
                                     'top_features': [FULL[i] for i in np.argsort(-v)[:5]]})
            write_json(evaldir / 'shap_zone_analysis.json', {'source': 'nested_lopzo_fold_local',
                         'feature_names': FULL, 'missing_feature_contribution': 0, 'per_zone': fingerprints,
                         'global_mean_abs_shap': df[cols].abs().mean().tolist()})
    write_json(evaldir / f'{scenario}_unfiltered_overall.json', result)
    residuals = output / 'residuals'
    residuals.mkdir(exist_ok=True)
    label = 'cv' if scenario == 'rcv' else scenario
    df.to_csv(residuals / f'{target}_{label}_predictions.csv', index=False)
    feature_names = list(dict.fromkeys(FULL + BASELINE)) if args.feature_mode == 'baseline17' else FULL
    if complete:
        pd.to_pickle({'df': df, 'feature_names': feature_names, 'nested_validation': True,
                      'feature_names_by_fold': {p.name: json.loads((p / 'model_config.json').read_text(encoding='utf-8'))['feature_names'] for p in paths},
                      'n_bootstrap': args.rcv_repeats, 'n_splits': args.rcv_splits},
                     residuals / f'{target}_{label}_residuals.pkl')
    summarize_features(output, data, target, scenario, paths, args)
    summary_rows = []
    for result_path in sorted((output / 'evaluation').glob('*/*_unfiltered_overall.json')):
        item = json.loads(result_path.read_text(encoding='utf-8'))
        summary_rows.append({'target': result_path.parent.name, 'scenario': item['scenario'].upper(),
                             'complete': item['complete'], **item['overall']})
    pd.DataFrame(summary_rows).to_csv(output / 'evaluation' / 'unfiltered_summary.csv', index=False)
    return result
