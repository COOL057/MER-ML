import argparse
import importlib.metadata
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from nested_core import (TARGETS, aggregate, feature_matrix, fit_pipeline, load_data,
                         outer_splits, records, sha256, write_json, mark_complete,
                         verify_complete, export_selection, select_features)


def arguments(default_mode='evaluate'):
    package = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default=str(package / 'data_clean'))
    p.add_argument('--config-dir', default=str(package / 'configs_wide'))
    p.add_argument('--composition-csv')
    p.add_argument('--process-data')
    p.add_argument('--feature-config')
    p.add_argument('--output-dir')
    p.add_argument('--target', choices=['all'] + TARGETS, default='all')
    p.add_argument('--mode', choices=['evaluate', 'deploy', 'select'], default=default_mode)
    p.add_argument('--scenarios', nargs='+', choices=['loco', 'lopzo', 'rcv'], default=['loco', 'lopzo', 'rcv'])
    p.add_argument('--feature-mode', choices=['knowledge15', 'baseline17', 'full31'], default='knowledge15')
    p.add_argument('--n-trials', type=int, default=400)
    p.add_argument('--inner-splits', type=int, default=10)
    p.add_argument('--stability-rounds', type=int, default=50)
    p.add_argument('--stability-splits', type=int, default=5)
    p.add_argument('--selector-iterations', type=int, default=500)
    p.add_argument('--selector-early-stopping', type=int, default=100)
    p.add_argument('--top-k', type=int, default=15)
    p.add_argument('--n-bootstrap', '--rcv-repeats', dest='rcv_repeats', type=int, default=10)
    p.add_argument('--n-splits', '--rcv-splits', dest='rcv_splits', type=int, default=10)
    p.add_argument('--early-stopping', type=int, default=200)
    p.add_argument('--iteration-multiplier', type=float, default=1.2)
    p.add_argument('--zone-boundaries', nargs=4, type=float)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--skip-shap', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--max-outer-folds', type=int)
    p.add_argument('--iteration-cap', type=int)
    a = p.parse_args()
    if a.output_dir is None:
        folder = {'evaluate': 'unified_results_clean', 'deploy': 'results_final/EXP_3_Stability',
                  'select': 'stability_selection_results/global'}[a.mode]
        a.output_dir = str(package / folder)
    a.compute_shap = not a.skip_shap
    a.composition_csv = a.composition_csv or str(Path(a.data_dir) / 'composition_at_percent.csv')
    a.process_data = a.process_data or str(Path(a.data_dir) / 'data5.xlsx')
    a.feature_config = a.feature_config or str(Path(a.config_dir) / 'feature_engineering_config.json')
    for key in ['n_trials', 'inner_splits', 'stability_rounds', 'stability_splits', 'selector_iterations',
                'top_k', 'rcv_repeats', 'rcv_splits', 'early_stopping', 'selector_early_stopping', 'threads']:
        if getattr(a, key) < (2 if 'splits' in key else 1):
            p.error(f'Invalid {key}')
    if a.iteration_multiplier <= 0 or (a.iteration_cap is not None and a.iteration_cap < 1):
        p.error('Invalid iteration settings')
    if a.max_outer_folds is not None and a.max_outer_folds < 1:
        p.error('max-outer-folds must be positive')
    return a


def main(default_mode='evaluate'):
    args = arguments(default_mode)
    package = Path(__file__).resolve().parent
    output = Path(args.output_dir).resolve()
    if package not in output.parents:
        raise ValueError('output-dir must be a subdirectory of this new script package')
    data = load_data(args)
    configs = {}
    config_paths = []
    for target in TARGETS:
        f = Path(args.config_dir).resolve() / f'{target}_config.json'
        config = json.loads(f.read_text(encoding='utf-8'))
        if config.get('use_log_transform', False) or config.get('use_winsorize', False) or config.get('loss_function', 'RMSE') != 'RMSE':
            raise ValueError(f'{target}: this protocol requires RMSE, no log transform, no Winsorize')
        configs[target] = config
        config_paths.append(f)
    effective = {k: v for k, v in vars(args).items() if k not in
                 ['target', 'mode', 'scenarios', 'resume', 'preflight', 'max_outer_folds', 'output_dir', 'skip_shap']}
    for key in ['data_dir', 'config_dir', 'composition_csv', 'process_data', 'feature_config']:
        effective[key] = str(Path(effective[key]).resolve())
    versions = {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'catboost', 'optuna', 'openpyxl']}
    sources = data['files'] + config_paths + [package / n for n in
              ['evaluate_and_collect_residuals.py', 'nested_core.py', 'feature_stability_selection.py',
               'train_engineering_228.py']]
    manifest = {'protocol': 'nested-feature-selection-and-tuning-v1', 'parameters': effective,
                'input_sha256': {str(f): sha256(f) for f in sources}, 'versions': versions,
                'zone_partition': args.zone_partition_source, 'pruning': 'fixed original domain rules',
                'selector_params_source': 'fixed literals, no existing fitted model or optimization result read',
                'inner_cv': 'composition groups for LOCO/deploy, zone groups for LOPZO, random folds for RCV',
                'stability_resampling': 'repeated shuffled CV, not bootstrap with replacement',
                'scope': 'base-model feature selection and tuning only; BS-GSR and UQ not validated'}
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'run_manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding='utf-8')) != manifest:
            raise ValueError('Inputs/settings/environment/code changed. Choose a NEW output-dir.')
    elif any(output.iterdir()):
        raise ValueError('Nonempty output-dir has no compatible manifest')
    else:
        write_json(manifest_path, manifest)
    report = {'python': sys.executable, 'versions': versions, 'sample_count': len(data['X28']),
              'composition_count': len(np.unique(data['comp_id'])),
              'valid_targets': {t: int(np.isfinite(data['y'][t]).sum()) for t in TARGETS},
              'zone_counts': {str(z): int((data['zone'] == z).sum()) for z in range(9)}}
    write_json(output / 'preflight.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.preflight:
        return
    lock = output / 'running.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f'Lock exists: {lock}. Confirm the old process is stopped before removing this lock.')
    with os.fdopen(fd, 'w') as f:
        f.write(f'{os.getpid()}\n')
    try:
        targets = TARGETS if args.target == 'all' else [args.target]
        summary = []
        for t in targets:
            X, names = feature_matrix(data, t, args.feature_mode)
            y = data['y'][t]
            if args.mode == 'select':
                path = output / t
                selection = select_features(X, y, names, np.flatnonzero(np.isfinite(y)), None, args, args.seed)
                export_selection(selection, path, t)
                continue
            if args.mode == 'deploy':
                path = output / f'{t}_EXP_3_Stability_{t}'
                if (path / 'complete.json').exists():
                    if not args.resume:
                        raise RuntimeError('Existing deployment; use --resume or a new output-dir')
                    verify_complete(path)
                    continue
                ids = np.flatnonzero(np.isfinite(y))
                _, _, meta = fit_pipeline(X, y, names, ids, np.array([], dtype=int), data['comp_id'], configs[t],
                                          args, path, args.seed + TARGETS.index(t) * 100000)
                import shutil
                shutil.copy2(path / 'model.cbm', path / f'{t}_final_model.cbm')
                write_json(path / f'{t}_model_config.json', meta)
                selection = json.loads((path / 'outer_features.json').read_text(encoding='utf-8'))
                export_selection(selection, package / 'stability_selection_results' / 'global' / t, t)
                mark_complete(path, target=t, deployment_only=True)
                continue
            for scenario in args.scenarios:
                paths = []
                for count, (fold, tr, te, groups, repeat, fold_id) in enumerate(outer_splits(data, t, scenario, args)):
                    if args.max_outer_folds is not None and count >= args.max_outer_folds:
                        break
                    path = output / 'folds' / t / scenario / fold
                    print(f'{datetime.now(timezone.utc).isoformat()} {t} {scenario} {fold} train={len(tr)} test={len(te)}', flush=True)
                    if (path / 'complete.json').exists():
                        if not args.resume:
                            raise RuntimeError('Existing fold; use --resume or a new output-dir')
                        verify_complete(path)
                    else:
                        seed = args.seed + TARGETS.index(t) * 100000 + ['loco', 'lopzo', 'rcv'].index(scenario) * 10000 + count
                        pred, shap, meta = fit_pipeline(X, y, names, tr, te, groups, configs[t], args, path, seed)
                        import pandas as pd
                        frame = pd.DataFrame(records(data, t, scenario, te, pred, shap, meta, fold, repeat, fold_id, tr))
                        frame.to_csv(path / 'predictions.csv', index=False)
                        mark_complete(path, target=t, scenario=scenario, fold=fold)
                    paths.append(path)
                if not paths:
                    raise ValueError(f'No folds available for {t}/{scenario}')
                summary.append({'target': t, **aggregate(output, data, t, scenario, paths, args)})
        write_json(output / f'{args.mode}_summary_{args.target}_{"_".join(args.scenarios)}.json', summary)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
