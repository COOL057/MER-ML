import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
from datetime import datetime


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def verify_fold(path):
    marker = json.loads((path / 'complete.json').read_text(encoding='utf-8'))
    for name, expected in marker['files_sha256'].items():
        if not (path / name).is_file() or digest(path / name) != expected:
            raise ValueError(f'Changed or missing completed file: {path / name}')


def command(package, parameters, output, target, scenarios):
    cmd = [sys.executable, str(package / 'evaluate_and_collect_residuals.py')]
    keys = ['data_dir', 'config_dir', 'composition_csv', 'process_data', 'feature_config',
            'feature_mode', 'n_trials', 'inner_splits', 'stability_rounds', 'stability_splits',
            'selector_iterations', 'selector_early_stopping', 'top_k', 'early_stopping',
            'iteration_multiplier', 'threads', 'seed']
    for key in keys:
        cmd += ['--' + key.replace('_', '-'), str(parameters[key])]
    cmd += ['--n-bootstrap', str(parameters['rcv_repeats']), '--n-splits', str(parameters['rcv_splits'])]
    if parameters.get('iteration_cap') is not None:
        cmd += ['--iteration-cap', str(parameters['iteration_cap'])]
    if parameters['zone_partition_source'].startswith('explicit'):
        cmd += ['--zone-boundaries'] + [str(v) for v in parameters['zone_boundaries']]
    if not parameters['compute_shap']:
        cmd += ['--skip-shap']
    return cmd + ['--output-dir', str(output), '--target', target, '--scenarios', *scenarios, '--resume']


def main():
    package = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default=str(package / 'unified_results_17Baseline'))
    parser.add_argument('--parallel-dir', default=str(package / 'parallel_results_17Baseline'))
    parser.add_argument('--workers', type=int, choices=[1, 2, 3, 4], default=2)
    parser.add_argument('--targets', nargs='+', choices=['Rm', 'Re', 'A', 'Ak'], default=['Rm', 'Re', 'A', 'Ak'])
    parser.add_argument('--scenarios', nargs='+', choices=['loco', 'lopzo', 'rcv'], default=['rcv'])
    args = parser.parse_args()
    root, parallel = Path(args.output_dir).resolve(), Path(args.parallel_dir).resolve()
    if package not in root.parents or package not in parallel.parents:
        raise ValueError('Both result directories must stay inside this package')
    if root == parallel or root in parallel.parents or parallel in root.parents:
        raise ValueError('The original and parallel directories must not overlap')
    if len(set(args.targets)) != len(args.targets) or len(set(args.scenarios)) != len(args.scenarios):
        raise ValueError('Duplicate target/scenario')
    manifest_path = root / 'run_manifest.json'
    if not manifest_path.is_file():
        raise ValueError('Run the original preflight first; its manifest defines the training protocol')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    parameters = manifest['parameters']
    for name, expected in manifest['input_sha256'].items():
        if not Path(name).is_file() or digest(name) != expected:
            raise ValueError(f'Input/code changed: {name}')
    if parameters.get('feature_mode') != 'baseline17':
        raise ValueError('This launcher is for the fixed baseline17 evaluation')
    lock = root / 'running.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError('Original evaluation is still running or its lock remains. Stop it first; never run both launchers together.')
    with os.fdopen(fd, 'w') as stream:
        stream.write(f'{os.getpid()}\n')
    processes, guard, stop = [], threading.Lock(), threading.Event()
    executor = None
    try:
        parallel.mkdir(parents=True, exist_ok=True)
        logs = package / 'logs'
        logs.mkdir(exist_ok=True)
        jobs = []
        for target in args.targets:
            work = parallel / target
            work.mkdir(parents=True, exist_ok=True)
            if (work / 'running.lock').exists():
                raise RuntimeError(f'Worker lock exists: {work / "running.lock"}. Confirm the worker has stopped first.')
            existing_manifest = work / 'run_manifest.json'
            if existing_manifest.exists():
                if json.loads(existing_manifest.read_text(encoding='utf-8')) != manifest:
                    raise ValueError(f'Incompatible worker output: {work}')
            elif any(work.iterdir()):
                raise ValueError(f'Worker directory has files without a matching manifest: {work}')
            else:
                shutil.copy2(manifest_path, existing_manifest)
                source = root / 'folds' / target
                if source.exists():
                    shutil.copytree(source, work / 'folds' / target)
            jobs.append((target, work))

        def run(job):
            target, work = job
            if stop.is_set():
                return target, 130
            logfile = logs / f'parallel_17Baseline_{target}.log'
            with open(logfile, 'a', encoding='utf-8') as stream:
                stream.write(f'\n{datetime.now().isoformat()} start/resume {target}\n')
                stream.flush()
                env = os.environ.copy()
                env['PYTHONUNBUFFERED'] = '1'
                with guard:
                    if stop.is_set():
                        return target, 130
                    process = subprocess.Popen(command(package, parameters, work, target, args.scenarios),
                                               cwd=package, stdout=stream, stderr=subprocess.STDOUT,
                                               env=env, start_new_session=os.name == 'posix')
                    processes.append(process)
                print(f'{target}: pid={process.pid}, log={logfile}', flush=True)
                code = process.wait()
            return target, code

        print(f'{args.workers} concurrent targets; {parameters["threads"]} threads/model; existing protocol retained.', flush=True)
        executor = ThreadPoolExecutor(max_workers=args.workers)
        futures = [executor.submit(run, job) for job in jobs]
        failures = []
        for future in as_completed(futures):
            target, code = future.result()
            print(f'{target}: exit={code}', flush=True)
            if code:
                failures.append(target)
        executor.shutdown(wait=True)
        executor = None
        if failures:
            raise RuntimeError(f'Workers failed: {failures}. Their saved progress remains; inspect logs and rerun this launcher.')

        for target, work in jobs:
            report = work / 'worker_verification.json'
            check = [sys.executable, str(package / 'verify_results.py'), '--output-dir', str(work),
                     '--target', target, '--scenarios', *args.scenarios]
            with open(report, 'w', encoding='utf-8') as stream:
                subprocess.run(check, cwd=package, stdout=stream, check=True)
            for scenario in args.scenarios:
                source = work / 'folds' / target / scenario
                for marker in sorted(source.glob('*/complete.json')):
                    fold = marker.parent
                    verify_fold(fold)
                    destination = root / 'folds' / target / scenario / fold.name
                    if (destination / 'complete.json').exists():
                        verify_fold(destination)
                        continue
                    if destination.exists():
                        backup = parallel / '_previous_partial' / target / scenario / (fold.name + '_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
                        backup.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(destination), str(backup))
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(fold, destination)
        lock.unlink()
        print('Completed folds imported. Rebuilding original-format summaries without retraining completed folds.', flush=True)
        for target in args.targets:
            subprocess.run(command(package, parameters, root, target, args.scenarios), cwd=package, check=True)
        for target in args.targets:
            subprocess.run([sys.executable, str(package / 'verify_results.py'), '--output-dir', str(root),
                            '--target', target, '--scenarios', *args.scenarios], cwd=package, check=True)
        print(f'Finished. Results: {root}', flush=True)
    except KeyboardInterrupt:
        stop.set()
        with guard:
            for process in processes:
                if process.poll() is None:
                    if os.name == 'posix':
                        os.killpg(process.pid, signal.SIGINT)
                    else:
                        process.terminate()
        print('Stopping workers; saved folds, selectors and completed trials can resume.', flush=True)
        for process in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        raise
    finally:
        if executor is not None:
            stop.set()
            executor.shutdown(wait=True, cancel_futures=True)
        if lock.exists():
            try:
                if lock.read_text().strip() == str(os.getpid()):
                    lock.unlink()
            except FileNotFoundError:
                pass


if __name__ == '__main__':
    main()
