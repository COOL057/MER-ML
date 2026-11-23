import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nested_core import TARGETS, sha256, verify_complete


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default=str(Path(__file__).resolve().parent / 'unified_results_clean'))
    parser.add_argument('--target', choices=['all'] + TARGETS, default='all')
    parser.add_argument('--scenarios', nargs='+', choices=['loco', 'lopzo', 'rcv'], default=['loco', 'lopzo'])
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    root = Path(args.output_dir).resolve()
    manifest = json.loads((root / 'run_manifest.json').read_text(encoding='utf-8'))
    for name, digest in manifest['input_sha256'].items():
        if not Path(name).is_file() or sha256(name) != digest:
            raise ValueError(f'Input or code changed: {name}')
    parameters = manifest['parameters']
    targets = TARGETS if args.target == 'all' else [args.target]
    comp = pd.read_csv(parameters['composition_csv']).apply(pd.to_numeric, errors='raise').fillna(0).round(3)
    groups = comp.apply(lambda row: '_'.join(f'{v:.3f}' for v in row), axis=1).to_numpy()
    proc = pd.read_excel(parameters['process_data'])
    bounds = parameters['zone_boundaries']
    zones = np.digitize(proc['ΔTI'], bounds[:2]) * 3 + np.digitize(proc['τ_norm'], bounds[2:])
    reports = []
    for target in targets:
        y = np.load(Path(parameters['data_dir']) / f'y_{target}.npy').reshape(-1)
        expected_ids = set(np.flatnonzero(np.isfinite(y)))
        for scenario in args.scenarios:
            paths = sorted((root / 'folds' / target / scenario).glob('*/complete.json'))
            if not paths:
                raise ValueError(f'No completed folds: {target}/{scenario}')
            frames, selections = [], []
            for marker in paths:
                path = marker.parent
                verify_complete(path)
                audit = json.loads((path / 'split_audit.json').read_text(encoding='utf-8'))
                tr, te = set(audit['outer_train_indices']), set(audit['outer_test_indices'])
                if tr & te or tr | te != expected_ids:
                    raise ValueError(f'Invalid outer partition: {path}')
                g = groups if scenario == 'loco' else zones if scenario == 'lopzo' else None
                if g is not None and set(g[list(tr)]) & set(g[list(te)]):
                    raise ValueError(f'Outer groups overlap: {path}')
                selection = json.loads((path / 'outer_features.json').read_text(encoding='utf-8'))
                if set(selection['train_indices']) != tr:
                    raise ValueError(f'Outer selector scope mismatch: {path}')
                if parameters['feature_mode'] == 'knowledge15' and len(selection['selected_features']) != parameters['top_k']:
                    raise ValueError(f'Unexpected feature count: {path}')
                inner_validation = []
                for inner in audit['inner_splits']:
                    a, b = set(inner['train_indices']), set(inner['validation_indices'])
                    if a & b or (a | b) != tr or set(inner['selector_train_indices']) != a:
                        raise ValueError(f'Inner selector/partition scope mismatch: {path}')
                    if g is not None and set(g[list(a)]) & set(g[list(b)]):
                        raise ValueError(f'Inner groups overlap: {path}')
                    inner_validation.extend(b)
                if len(inner_validation) != len(tr) or set(inner_validation) != tr:
                    raise ValueError(f'Inner validation coverage mismatch: {path}')
                frame = pd.read_csv(path / 'predictions.csv')
                if set(frame.global_idx) != te or len(frame) != len(te):
                    raise ValueError(f'Test prediction coverage mismatch: {path}')
                if not np.allclose(frame.y_true, y[frame.global_idx]):
                    raise ValueError(f'Target alignment mismatch: {path}')
                cols = [c for c in frame if c.startswith('shap_')]
                if cols and not np.allclose(frame[cols].sum(axis=1), frame.y_pred, rtol=1e-7, atol=1e-7):
                    raise ValueError(f'SHAP additivity mismatch: {path}')
                meta = json.loads((path / 'model_config.json').read_text(encoding='utf-8'))
                if meta['feature_names'] != selection['selected_features'] or meta['early_stopping_scope'] != 'inner validation only':
                    raise ValueError(f'Model/selector metadata mismatch: {path}')
                frames.append(frame)
                selections.append(set(selection['selected_features']))
            df = pd.concat(frames, ignore_index=True)
            counts = df.groupby('global_idx').size()
            expected_count = parameters['rcv_repeats'] if scenario == 'rcv' else 1
            complete = set(counts.index) == expected_ids and bool((counts == expected_count).all())
            saved = json.loads((root / 'evaluation' / target / f'{scenario}_unfiltered_overall.json').read_text(encoding='utf-8'))
            if saved['complete'] != complete:
                raise ValueError(f'Saved completeness mismatch: {target}/{scenario}')
            if not complete and not args.allow_partial:
                raise ValueError(f'Partial evaluation: {target}/{scenario}; finish all folds')
            stability = root.parent / 'stability_selection_results' if root.name == 'unified_results_clean' else root / 'stability_selection_results'
            frequency = pd.read_csv(stability / target / scenario / 'feature_selection_frequency.csv')
            for row in frequency.itertuples():
                n = sum(row.feature in s for s in selections)
                if row.selected_folds != n or row.n_folds != len(selections) or not np.isclose(row.selection_frequency, n / len(selections)):
                    raise ValueError(f'Feature frequency mismatch: {target}/{scenario}/{row.feature}')
            reports.append({'target': target, 'scenario': scenario, 'n_folds': len(paths), 'complete': complete, 'checks_passed': True})
    (root / 'verification_report.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
