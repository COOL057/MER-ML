import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

from run_parallel_evaluation import command, digest


TARGETS = ['Rm', 'Re', 'A', 'Ak']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, choices=[1, 2, 3, 4], default=4)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    package = Path(__file__).resolve().parent
    knowledge = package / 'unified_results_rcv3'
    baseline = package / 'unified_results_17Baseline'
    manifest = json.loads((knowledge / 'run_manifest.json').read_text(encoding='utf-8'))
    if manifest['protocol'] != 'nested-feature-selection-and-tuning-v1':
        raise ValueError('Unexpected knowledge-model protocol')
    parameters = dict(manifest['parameters'])
    if parameters['feature_mode'] != 'knowledge15' or parameters['rcv_repeats'] != 3 or not parameters['compute_shap']:
        raise ValueError('Expected completed three-repeat knowledge15 RCV with SHAP')
    for name, expected in manifest['input_sha256'].items():
        path = Path(name)
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f'Input/code changed: {name}. Use the original server package.')
    for name, expected in manifest['versions'].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError(f'Environment changed: {name}. Activate the original training environment.')
    for target in TARGETS:
        saved = json.loads((knowledge / 'evaluation' / target / 'rcv_unfiltered_overall.json').read_text(encoding='utf-8'))
        if not saved['complete'] or saved['n_completed_folds'] != 3 * parameters['rcv_splits']:
            raise ValueError(f'Finish knowledge RCV first: {target}')
    for name in ['run_parallel_baseline17.py', 'compare_17baseline_and_15knowledge.py']:
        if not (package / name).is_file():
            raise FileNotFoundError(f'Upload the required new file: {name}')
    print(f'17Baseline: fixed 17 raw features; no stability-selection model fits.', flush=True)
    print(f'RCV: 3 repeats x {parameters["rcv_splits"]} folds; '
          f'{12 * parameters["rcv_splits"]} outer models; {parameters["n_trials"]} trials/fold; '
          f'{parameters["inner_splits"]} inner folds.', flush=True)
    print(f'{args.workers} concurrent targets x {parameters["threads"]} threads/model. '
          'Knowledge model and old baseline results remain unchanged.', flush=True)
    if args.check_only:
        return
    parameters['feature_mode'] = 'baseline17'
    if not args.summarize_only:
        subprocess.run(command(package, parameters, baseline, 'all', ['rcv']) + ['--preflight'],
                       cwd=package, check=True)
        subprocess.run([sys.executable, str(package / 'run_parallel_baseline17.py'),
                        '--workers', str(args.workers), '--scenarios', 'rcv'],
                       cwd=package, check=True)
    subprocess.run([sys.executable, str(package / 'verify_results.py'), '--output-dir', str(baseline),
                    '--target', 'all', '--scenarios', 'rcv'], cwd=package, check=True)
    subprocess.run([sys.executable, str(package / 'compare_17baseline_and_15knowledge.py')],
                   cwd=package, check=True)
    print(f'Finished. 17Baseline vs 15Knowledge: {baseline / "evaluation" / "17Baseline_vs_15Knowledge_metrics.csv"}', flush=True)


if __name__ == '__main__':
    main()
