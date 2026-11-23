import argparse
import csv
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

from run_parallel_evaluation import digest


TARGETS = ['Rm', 'Re', 'A', 'Ak']
SCENARIOS = ['rcv', 'loco', 'lopzo']


def check_protocol(package):
    root = package / 'unified_results_clean'
    manifest = json.loads((root / 'run_manifest.json').read_text(encoding='utf-8'))
    if manifest['protocol'] != 'nested-feature-selection-and-tuning-v1':
        raise ValueError('Unexpected protocol')
    parameters = manifest['parameters']
    if parameters['feature_mode'] != 'knowledge15':
        raise ValueError('This entry requires the knowledge15 protocol')
    for name, expected in manifest['input_sha256'].items():
        path = Path(name)
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f'Input/code changed or unavailable: {name}. Use the original server directory.')
    for name, expected in manifest['versions'].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError(f'Environment changed: {name}. Activate the original training environment.')
    for target in TARGETS:
        for scenario in ['loco', 'lopzo']:
            result = json.loads((root / 'evaluation' / target / f'{scenario}_unfiltered_overall.json').read_text(encoding='utf-8'))
            if not result['complete']:
                raise ValueError(f'Finish the existing evaluation first: {target}/{scenario}')
    return root, parameters


def summarize(root, parameters):
    rows = []
    for target in TARGETS:
        for scenario in SCENARIOS:
            result = json.loads((root / 'evaluation' / target / f'{scenario}_unfiltered_overall.json').read_text(encoding='utf-8'))
            if not result['complete']:
                raise ValueError(f'Incomplete result: {target}/{scenario}')
            repeats = parameters['rcv_repeats'] if scenario == 'rcv' else 1
            metrics = result['overall']
            if metrics['n_samples'] % repeats:
                raise ValueError(f'Invalid prediction count: {target}/{scenario}')
            if scenario == 'rcv' and result['n_completed_folds'] != repeats * parameters['rcv_splits']:
                raise ValueError(f'Invalid RCV fold count: {target}')
            rows.append({'target': target, 'scenario': scenario.upper(),
                         'n_valid_samples': metrics['n_samples'] // repeats,
                         'n_predictions': metrics['n_samples'], 'n_repeats': repeats,
                         'n_outer_folds': result['n_completed_folds'],
                         'r2': metrics['r2'], 'mae': metrics['mae'], 'rmse': metrics['rmse'],
                         'mean_repeat_r2': result.get('mean_repeat_r2'),
                         'std_repeat_r2': result.get('std_repeat_r2')})
    directory = root / 'evaluation'
    with (directory / 'three_scenario_comparison.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (directory / 'three_scenario_comparison.json').write_text(
        json.dumps({'metrics_source': 'pooled outer-test predictions',
                    'rcv_resampling': 'repeated shuffled KFold, not bootstrap with replacement',
                    'repeat_std_note': 'split variability; not an independent-sample confidence interval',
                    'results': rows}, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\nTarget  Scenario       R2          MAE         RMSE', flush=True)
    for row in rows:
        print(f'{row["target"]:6}  {row["scenario"]:8}  {row["r2"]:9.4f}  {row["mae"]:11.4f}  {row["rmse"]:11.4f}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, choices=[1, 2, 3, 4], default=4)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    package = Path(__file__).resolve().parent
    root, parameters = check_protocol(package)
    folds = parameters['rcv_repeats'] * parameters['rcv_splits']
    print(f'RCV: {parameters["rcv_repeats"]} repeats x {parameters["rcv_splits"]} folds; '
          f'{folds} folds/target; {folds * len(TARGETS)} outer models in total.', flush=True)
    print(f'{parameters["n_trials"]} trials/fold; {parameters["stability_rounds"]} selection rounds; '
          f'{args.workers} concurrent targets x {parameters["threads"]} threads/model.', flush=True)
    print('Existing server protocol verified. Only RCV is added; completed folds resume.', flush=True)
    if args.check_only:
        return
    subprocess.run([sys.executable, str(package / 'run_parallel_evaluation.py'),
                    '--workers', str(args.workers), '--scenarios', 'rcv'], cwd=package, check=True)
    subprocess.run([sys.executable, str(package / 'verify_results.py'), '--output-dir', str(root),
                    '--target', 'all', '--scenarios', *SCENARIOS], cwd=package, check=True)
    summarize(root, parameters)
    print(f'Finished. Three-scenario comparison: {root / "evaluation" / "three_scenario_comparison.csv"}', flush=True)


if __name__ == '__main__':
    main()
