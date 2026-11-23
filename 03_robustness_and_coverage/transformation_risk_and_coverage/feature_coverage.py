import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr


FEATURES = ['C', 'Si', 'Mn', 'Cr', 'V', 'Al', 'Ni', 'Mo', 'Cu', 'W', 'Z', 'Ve', 'χP', 'δb', 'δf', 'ΔFe', 'ΔC', 'Ta', 'Ti', 'ti', 'ΔTA', 'ΔTI', 'τ', 'Ms', 'Tt', 'tt', 'sph', 'tmp']
COMPOSITION = FEATURES[:10]
PROCESS = ['Ta', 'Ti', 'ti', 'Tt', 'tt']
BINARY = ['sph', 'tmp']
TARGETS = ['Rm', 'Re', 'A', 'Ak']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scenario', choices=['rcv', 'loco', 'lopzo'], required=True)
    args = parser.parse_args()
    root = args.project
    args.output.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(np.load(root / 'data_clean/X_28dim.npy'), columns=FEATURES)
    for name in ['ti', 'tt']:
        if (raw[name].dropna() < 0).any():
            raise ValueError(name)
        raw[name] = np.log1p(raw[name])
    comp = pd.read_csv(root / 'data_clean/composition_at_percent.csv').fillna(0).round(3)
    ids = comp.apply(lambda row: '_'.join(f'{value:.3f}' for value in row), axis=1).to_numpy()
    rows = []
    for target in TARGETS:
        y = np.load(root / f'data_clean/y_{target}.npy').reshape(-1)
        valid = set(np.flatnonzero(np.isfinite(y)))
        if args.scenario == 'rcv':
            paths = [(rep, fold, root / f'unified_results_rcv3/folds/{target}/rcv/repeat_{rep:02d}_fold_{fold:02d}') for rep in range(3) for fold in range(10)]
        else:
            paths = [(0, fold, path) for fold, path in enumerate(sorted((root / f'unified_results_clean/folds/{target}/{args.scenario}').glob('fold_*')))]
        if not paths:
            raise ValueError(f'No folds: {target} {args.scenario}')
        for repeat, fold, path in paths:
            cfg = json.loads((path / 'model_config.json').read_text(encoding='utf-8'))
            train_ids = np.array(cfg['outer_train_indices'], dtype=int)
            test_ids = np.array(cfg['outer_test_indices'], dtype=int)
            if set(train_ids) & set(test_ids) or set(train_ids) | set(test_ids) != valid:
                raise ValueError(f'Invalid split: {path}')
            pred = pd.read_csv(path / 'predictions.csv').set_index('global_idx').loc[test_ids]
            if not np.allclose(pred.y_true, y[test_ids]):
                raise ValueError(f'Invalid labels: {path}')
            train = raw.iloc[train_ids][COMPOSITION + PROCESS]
            test = raw.iloc[test_ids][COMPOSITION + PROCESS]
            medians = train.median()
            if medians.isna().any():
                raise ValueError(f'Missing training column: {path}')
            means = train.fillna(medians).mean()
            std = train.fillna(medians).std(ddof=0)
            constant = std[std < 1e-12].index.tolist()
            std = std.mask(std < 1e-12, 1.0)
            train_scaled = (train.fillna(medians) - means) / std
            test_scaled = (test.fillna(medians) - means) / std
            if constant:
                train_scaled.loc[:, constant] = 0
                test_scaled.loc[:, constant] = 0
            train_binary = raw.iloc[train_ids][BINARY].to_numpy(float)
            test_binary = raw.iloc[test_ids][BINARY].to_numpy(float)
            if not np.isfinite(train_binary).all() or not np.isfinite(test_binary).all():
                raise ValueError(f'Missing binary variable: {path}')
            train_missing = train[['Tt', 'tt']].isna().to_numpy(float)
            test_missing = test[['Tt', 'tt']].isna().to_numpy(float)
            train_vec = np.column_stack([train_scaled[COMPOSITION].to_numpy() * np.sqrt(.5 / 10), train_scaled[PROCESS].to_numpy() * np.sqrt(.5 / 9), train_binary * np.sqrt(.5 / 9), train_missing * np.sqrt(.5 / 9)])
            test_vec = np.column_stack([test_scaled[COMPOSITION].to_numpy() * np.sqrt(.5 / 10), test_scaled[PROCESS].to_numpy() * np.sqrt(.5 / 9), test_binary * np.sqrt(.5 / 9), test_missing * np.sqrt(.5 / 9)])
            distances = cdist(test_vec, train_vec)
            nearest = distances.argmin(axis=1)
            minimum = distances[np.arange(len(test_ids)), nearest]
            reference = cdist(train_vec, train_vec)
            np.fill_diagonal(reference, np.inf)
            percentile = np.searchsorted(np.sort(reference.min(axis=1)), minimum, side='right') / len(train_ids)
            errors = np.abs(y[test_ids] - pred.y_pred.to_numpy())
            zones = np.digitize(raw.loc[test_ids, 'ΔTI'], [25.992800000000003, 52.01440000000002]) * 3 + np.digitize(raw.loc[test_ids, 'τ'], [1.0, 1.6139453980627756])
            if 'zone_id' in pred and not np.array_equal(zones, pred.zone_id.to_numpy()):
                raise ValueError(f'Invalid zones: {path}')
            risk = np.isin(zones, [0, 3, 6] if target in ['Rm', 'Re'] else [6, 7, 8])
            for j, index in enumerate(test_ids):
                rows.append(dict(target=target, repeat=repeat, fold=fold, global_idx=int(index), comp_id=ids[index], physical_zone=int(zones[j]), specified_risk=int(risk[j]), y_true=float(y[index]), y_pred=float(pred.y_pred.iloc[j]), abs_error=float(errors[j]), nearest_distance=float(minimum[j]), training_loo_percentile=float(percentile[j]), distance_band='low' if percentile[j] <= 1/3 else 'middle' if percentile[j] <= 2/3 else 'high', nearest_train_global_idx=int(train_ids[nearest[j]]), nearest_same_composition=int(ids[index] == ids[train_ids[nearest[j]]])))
    d = pd.DataFrame(rows)
    if d.duplicated(['target', 'repeat', 'global_idx']).any() or not d.groupby(['target', 'global_idx']).size().eq(3 if args.scenario == 'rcv' else 1).all():
        raise ValueError('Invalid prediction counts')
    d.to_csv(args.output / 'sample_distances.csv', index=False, encoding='utf-8-sig')
    cor = []
    for (target, repeat), group in d.groupby(['target', 'repeat']):
        cor.append(dict(target=target, repeat=repeat, n=len(group), n_compositions=group.comp_id.nunique(), spearman_distance_error=spearmanr(group.nearest_distance, group.abs_error).statistic, spearman_percentile_error=spearmanr(group.training_loo_percentile, group.abs_error).statistic))
    pd.DataFrame(cor).to_csv(args.output / 'correlations_by_repeat.csv', index=False, encoding='utf-8-sig')
    strata = []
    for (target, band, risk), group in d.groupby(['target', 'distance_band', 'specified_risk']):
        strata.append(dict(target=target, distance_band=band, specified_risk=risk, n_predictions=len(group), n_unique=group.global_idx.nunique(), n_compositions=group.comp_id.nunique(), mae=group.abs_error.mean(), median_abs_error=group.abs_error.median()))
    pd.DataFrame(strata).to_csv(args.output / 'coverage_risk_strata.csv', index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    main()
