import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from run_parallel_evaluation import command, digest, verify_fold
from run_rcv_evaluation import TARGETS, summarize


def old_processes(package):
    if os.name != 'posix':
        return []
    result = subprocess.run(['ps', '-u', str(os.getuid()), '-o', 'pid=,args='],
                            check=True, capture_output=True, text=True)
    jobs = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2 or int(parts[0]) == os.getpid():
            continue
        pid, text = int(parts[0]), parts[1]
        original_worker = str(package / 'parallel_results') + '/'
        original_master = str(package / 'unified_results_clean')
        if str(package / 'evaluate_and_collect_residuals.py') in text:
            if original_worker in text or original_master in text:
                jobs.append((pid, text))
        elif str(package / 'run_rcv_evaluation.py') in text:
            jobs.append((pid, text))
        elif str(package / 'run_parallel_evaluation.py') in text and str(package / 'parallel_results_rcv3') not in text:
            jobs.append((pid, text))
    return jobs


def finish_old_processes(package, stop):
    jobs = old_processes(package)
    if jobs and not stop:
        raise RuntimeError('Original workers still run. Use --stop-current to interrupt them gracefully and reuse saved progress.')
    if jobs:
        if any(' --scenarios rcv ' not in text + ' ' and 'run_rcv_evaluation.py' not in text
               for _, text in jobs):
            raise RuntimeError('An original evaluation is not RCV-only; do not stop it automatically.')
        for pid, _ in jobs:
            current = dict(old_processes(package))
            if pid in current:
                print(f'Sending SIGINT to original RCV process {pid}.', flush=True)
                try:
                    os.kill(pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 60
        while old_processes(package):
            if time.monotonic() >= deadline:
                raise RuntimeError('Original processes have not exited yet. Wait and rerun; no forced termination or copying was performed.')
            time.sleep(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, choices=[1, 2, 3, 4], default=4)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--stop-current', action='store_true')
    args = parser.parse_args()
    package = Path(__file__).resolve().parent
    original = package / 'unified_results_clean'
    output = package / 'unified_results_rcv3'
    parallel = package / 'parallel_results_rcv3'
    manifest = json.loads((original / 'run_manifest.json').read_text(encoding='utf-8'))
    if manifest['protocol'] != 'nested-feature-selection-and-tuning-v1' or manifest['parameters']['feature_mode'] != 'knowledge15':
        raise ValueError('Unexpected original protocol')
    for name, expected in manifest['input_sha256'].items():
        if not Path(name).is_file() or digest(name) != expected:
            raise ValueError(f'Original input/code changed: {name}')
    for name, expected in manifest['versions'].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError(f'Original environment changed: {name}')
    parameters = dict(manifest['parameters'])
    if parameters['rcv_repeats'] < 3:
        raise ValueError('Original RCV must define at least three repeats')
    parameters['rcv_repeats'] = 3
    splits = parameters['rcv_splits']
    fold_names = [f'repeat_{r:02d}_fold_{i:02d}' for r in range(3) for i in range(splits)]
    for target in TARGETS:
        for scenario in ['loco', 'lopzo']:
            result = json.loads((original / 'evaluation' / target / f'{scenario}_unfiltered_overall.json').read_text(encoding='utf-8'))
            if not result['complete']:
                raise ValueError(f'Original evaluation incomplete: {target}/{scenario}')
        count = sum(any((source / fold / 'complete.json').is_file() for source in
                        [package / 'parallel_results' / target / 'folds' / target / 'rcv',
                         original / 'folds' / target / 'rcv']) for fold in fold_names)
        print(f'{target}: {count}/{len(fold_names)} required folds already completed in original outputs.', flush=True)
    print(f'Plan: 3 repeats x {splits} folds; {len(fold_names) * 4} outer models; '
          f'{parameters["n_trials"]} trials/fold; original selector and inner CV retained.', flush=True)
    jobs = old_processes(package)
    print(f'Original running processes: {[pid for pid, _ in jobs]}. New output: {output}', flush=True)
    if args.check_only:
        return
    finish_old_processes(package, args.stop_current)
    if (output / 'running.lock').exists() or any(parallel.glob('*/running.lock')):
        raise RuntimeError('New-run lock exists. Confirm the new workers are stopped before handling this lock.')
    subprocess.run(command(package, parameters, output, 'all', ['rcv']) + ['--preflight'], cwd=package, check=True)
    for target in TARGETS:
        sources = [package / 'parallel_results' / target / 'folds' / target / 'rcv',
                   original / 'folds' / target / 'rcv']
        for fold in fold_names:
            destination = output / 'folds' / target / 'rcv' / fold
            if destination.exists():
                continue
            candidates = [source / fold for source in sources if (source / fold).is_dir()]
            if not candidates:
                continue
            source = next((p for p in candidates if (p / 'complete.json').is_file()), candidates[0])
            if (source / 'complete.json').is_file():
                verify_fold(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, destination)
    subprocess.run([sys.executable, str(package / 'run_parallel_evaluation.py'), '--output-dir', str(output),
                    '--parallel-dir', str(parallel), '--workers', str(args.workers), '--scenarios', 'rcv'],
                   cwd=package, check=True)
    subprocess.run([sys.executable, str(package / 'verify_results.py'), '--output-dir', str(output),
                    '--target', 'all', '--scenarios', 'rcv'], cwd=package, check=True)
    for target in TARGETS:
        for scenario in ['loco', 'lopzo']:
            source = original / 'evaluation' / target / f'{scenario}_unfiltered_overall.json'
            destination = output / 'evaluation' / target / source.name
            shutil.copy2(source, destination)
    summarize(output, parameters)
    comparison = output / 'evaluation' / 'three_scenario_comparison.json'
    report = json.loads(comparison.read_text(encoding='utf-8'))
    report['sources'] = {'RCV': str(output), 'LOCO': str(original), 'LOPZO': str(original)}
    report['scope_note'] = 'Only RCV evaluation repeats changed from 10 to 3. Per-fold model-development settings unchanged.'
    comparison.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Finished. Three-scenario comparison: {output / "evaluation" / "three_scenario_comparison.csv"}', flush=True)


if __name__ == '__main__':
    main()
