
import argparse
import ast
import json
import re
import sys
import warnings
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import rankdata, spearmanr
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / '01_model_training_and_evaluation'))
TARGETS = ['Rm', 'Re', 'A', 'Ak']
ZONES = {'Rm': [0, 3], 'Re': [0, 3], 'A': [7, 8], 'Ak': [7, 8]}
FUNCTIONS = {'square': np.square, 'sqrt': np.sqrt, 'abs': np.abs}
EPS = 1e-9
PERMUTATIONS = {}
RIDGE_ALPHA = 0.25
CLIP_Z = 3.
SEED = 42
SR_NAMES = ['C', 'Si', 'Mn', 'Cr', 'V', 'Al', 'Ni', 'Mo', 'Cu', 'W',
            'Z', 'Ve', 'chiP', 'delta_b', 'delta_f', 'Delta_Fe', 'Delta_C',
            'Ta', 'Ti', 'Delta_TA', 'Delta_TI', 'tau', 'Tt', 'tt', 'sph',
            'tmp', 'Ceq', 'Ta_Ti_diff', 'C_DTI', 'DC_Ti', 'C_Cr']


def make_splits(data, target, n_splits, seed):
    valid = np.flatnonzero(np.isfinite(data['y'][target]))
    active = valid[np.isin(data['zone'][valid], ZONES[target])]
    labels = data['zone'][active]
    if any(np.sum(labels == z) < n_splits for z in ZONES[target]):
        raise ValueError(f'{target}: too few samples per zone for {n_splits} folds')
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = [(active[tr], active[te]) for tr, te in splitter.split(active, labels)]
    if not all(all(np.sum(data['zone'][tr] == z) >= 4 and np.sum(data['zone'][te] == z) >= 1
                   for z in ZONES[target]) for tr, te in splits):
        raise ValueError(f'{target}: insufficient calibration training samples per zone')
    return valid, splits, seed

def run_stratified_bootstrap_rm_smi(
    df: pd.DataFrame,
    X_clean: np.ndarray,
    feature_names: list,
    active_zones: list,
    n_bootstrap: int = 100,
    top_k_threshold: int = 10,
):
    """
    Stratified perturbation-resampling feature stability selection.

    The target is the centered and standardized residual, res_norm.
    Z6 warning-only samples are not included.
    """
    n_features = len(feature_names)
    inclusion_counts = {f: 0 for f in feature_names}

    zone_indices = {
        zid: df[df["zone_id"] == zid].index.tolist()
        for zid in active_zones
    }


    rng = np.random.default_rng(42)

    for round_idx in range(n_bootstrap):
        bootstrap_df_indices = []

        for _, idx_list in zone_indices.items():
            if len(idx_list) == 0:
                continue

            sample_size = max(int(len(idx_list) * 0.8), 2)
            sampled_idx = rng.choice(idx_list, size=sample_size, replace=True)
            bootstrap_df_indices.extend(sampled_idx)

        if not bootstrap_df_indices:
            continue

        bootstrap_df = df.loc[bootstrap_df_indices].reset_index(drop=True)

        feature_scores = np.zeros(n_features)
        total_samples = 0

        for zid in active_zones:
            z_mask = bootstrap_df["zone_id"] == zid

            if not np.any(z_mask):
                continue

            n_z = int(np.sum(z_mask))
            total_samples += n_z

            gi = bootstrap_df.loc[z_mask, "global_idx"].values.astype(int)
            X_z = X_clean[gi]
            y_signed = bootstrap_df.loc[z_mask, "res_norm"].values
            y_abs = np.abs(y_signed)

            try:
                mi_z = mutual_info_regression(
                    X_z,
                    y_abs,
                    random_state=42 + round_idx,
                )
            except Exception:
                mi_z = np.zeros(n_features)

            rho_z = np.zeros(n_features)
            for i in range(n_features):
                rho, _ = spearmanr(X_z[:, i], y_signed)
                rho_z[i] = abs(rho) if not np.isnan(rho) else 0.0

            score_z = 0.6 * rho_z + 0.4 * mi_z
            feature_scores += score_z * n_z

        feature_scores /= total_samples + 1e-6
        round_rank = np.argsort(feature_scores)[::-1]

        for rank_pos in range(min(top_k_threshold, len(round_rank))):
            feature_name = feature_names[round_rank[rank_pos]]
            inclusion_counts[feature_name] += 1

    results = [
        {
            "feature": f,
            "inclusion_frequency": inclusion_counts[f] / max(n_bootstrap, 1),
        }
        for f in feature_names
    ]

    return (
        pd.DataFrame(results)
        .sort_values(by="inclusion_frequency", ascending=False)
        .reset_index(drop=True)
    )

def make_pysr_safe(name: str) -> str:
    """Replace symbols and add x_ prefix to avoid Julia/SymPy name conflicts."""
    safe = (
        name.replace("Δ", "Delta")
        .replace("τ", "tau")
        .replace("χ", "chi")
        .replace("δ", "delta")
    )
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", safe)
    return f"x_{safe}"


def restore_feature_names(equation: str, features: list) -> str:
    """Replace PySR-safe variable names with original feature names."""
    eq_display = str(equation)
    for f in features:
        eq_display = re.sub(r"\b" + make_pysr_safe(f) + r"\b", f, eq_display)
    return eq_display

def prepare_calibration(train, X31, target):
    remove = {'C', 'tmp', 'chiP'}
    remove.update(['delta_f', 'Delta_C'] if target in ('Rm', 'Re') else ['delta_b', 'Delta_Fe'])
    names = [name for name in SR_NAMES if name not in remove]
    X = X31[:, [SR_NAMES.index(name) for name in names]]
    ids = train.global_idx.to_numpy(dtype=int)
    medians = np.nanmedian(X[ids], axis=0)
    if not np.isfinite(medians).all():
        raise ValueError('Missing calibration-training median')
    X = np.where(np.isnan(X), medians, X)
    frame = train.reset_index(drop=True).copy()
    stats = {}
    for zone in ZONES[target]:
        mask = frame.zone_id == zone
        residual = frame.loc[mask, 'residual'].to_numpy()
        mu = float(np.mean(residual))
        sigma = float(np.std(residual) + 1e-6)
        frame.loc[mask, 'res_norm'] = (residual - mu) / sigma
        stats[str(zone)] = {'mu': mu, 'sigma': sigma, 'sample_size': len(residual)}
    ranks = run_stratified_bootstrap_rm_smi(frame, X, names, ZONES[target])
    state = {'feature_names': names, 'imputation_medians': medians.tolist(),
             'top_features': ranks.feature.tolist()[:10], 'training_indices': ids.tolist(),
             'zone_stats': stats}
    return frame, X, state


def search_equations(frame, X, state, scratch):
    from pysr import PySRRegressor

    rng = np.random.default_rng(SEED)
    top = state['top_features']
    columns = [state['feature_names'].index(n) for n in top]
    values = X[frame.global_idx.to_numpy(int)][:, columns]
    responses = frame.res_norm.to_numpy()
    candidates = {}
    for run in range(50):
        subset = sorted(rng.choice(len(top), size=4, replace=False))
        names = [top[i] for i in subset]
        model = PySRRegressor(
            niterations=80, binary_operators=['+', '-', '*', '/'],
            unary_operators=['sqrt', 'square', 'abs'], maxsize=12,
            parsimony=float(rng.uniform(.01, .05)), verbosity=0, deterministic=True,
            random_state=int(rng.integers(0, 100000)), parallelism='serial',
            output_directory=str(scratch), run_id=f'subset_{run:02d}',
        )
        try:
            model.fit(values[:, subset], responses, variable_names=[make_pysr_safe(n) for n in names])
        except Exception as exc:
            warnings.warn(f'PySR subset {run + 1}: {exc}', RuntimeWarning)
            continue
        for row in model.equations_.itertuples(index=False):
            if row.complexity <= 10:
                equation = restore_feature_names(str(row.equation), names)
                candidates.setdefault(equation, int(row.complexity))
    if not candidates:
        fallback = PySRRegressor(
            niterations=40, binary_operators=['+', '-', '*', '/'],
            unary_operators=['sqrt', 'square', 'abs'], maxsize=12, parsimony=.05,
            verbosity=0, deterministic=True, random_state=42, parallelism='serial',
            output_directory=str(scratch), run_id='all_features',
        )
        fallback.fit(values, responses, variable_names=[make_pysr_safe(n) for n in top])
        row = fallback.equations_.iloc[0]
        candidates[restore_feature_names(str(row.equation), top)] = int(row.complexity)
    return candidates.items()


def compiled_equation(equation, allowed):
    tree = ast.parse(equation, mode='eval')
    symbols = set()
    nodes = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
             ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.UAdd, ast.USub)
    for node in ast.walk(tree):
        if not isinstance(node, nodes):
            raise ValueError(f'Unsupported equation syntax: {equation}')
        if isinstance(node, ast.Constant):
            if type(node.value) not in (int, float) or not np.isfinite(node.value):
                raise ValueError('Invalid equation constant')
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS or len(node.args) != 1 or node.keywords:
                raise ValueError('Invalid equation function')
        if isinstance(node, ast.Name) and node.id not in FUNCTIONS:
            if node.id not in allowed:
                raise ValueError(f'Unknown equation feature {node.id}')
            symbols.add(node.id)
    return compile(tree, '<symbolic-equation>', 'eval'), symbols


def predict_expression(equation, X, names, ids):
    code, symbols = compiled_equation(equation, set(names))
    ids = np.asarray(ids, dtype=int)
    values = {name: X[ids, names.index(name)] for name in symbols}
    with np.errstate(all='ignore'):
        raw = np.asarray(eval(code, {'__builtins__': {}, **FUNCTIONS}, values), dtype=float)
    raw = np.broadcast_to(raw, (len(ids),)).copy()
    return np.clip(np.nan_to_num(raw, nan=0., posinf=0., neginf=0.), -1000., 1000.)


def normalize(raw, clip_z):
    mean, std = float(np.mean(raw)), float(np.std(raw))
    scale = std + EPS
    norm = (raw - mean) / scale if std > EPS else np.zeros_like(raw)
    return np.clip(norm, -clip_z, clip_z), mean, scale

def permutation_spearman(x, y, n_perm=200, seed=42):
    if len(x) < 4 or np.std(x) <= EPS or np.std(y) <= EPS:
        return 0., 1.
    # Ranking once preserves ties. Permuting ranks leaves both rank variances fixed.
    rx, ry = rankdata(x), rankdata(y)
    rx -= rx.mean()
    ry -= ry.mean()
    key = (len(x), n_perm, seed)
    if key not in PERMUTATIONS:
        rng = np.random.default_rng(seed)
        PERMUTATIONS[key] = np.array([rng.permutation(len(x)) for _ in range(n_perm)])
    count = np.count_nonzero(np.abs(rx[PERMUTATIONS[key]] @ ry) >= abs(rx @ ry))
    return float(spearmanr(x, y).statistic), float((count + 1) / (n_perm + 1))


def evaluate_candidate(equation, train, X, state):
    zones, rho_sum, gain_sum, gain_rel, active = {}, 0., 0., 0., 0
    ids = train.global_idx.to_numpy(dtype=int)
    raw = predict_expression(equation, X, state['feature_names'], ids)
    for z in sorted(train.zone_id.unique()):
        mask = (train.zone_id == z).to_numpy()
        stats = state['zone_stats'][str(int(z))]
        trend, mean, std = normalize(raw[mask], CLIP_Z)
        residual = train.loc[mask, 'residual'].to_numpy()
        centered = (residual - stats['mu']) / stats['sigma']
        truth = train.loc[mask, 'y_true'].to_numpy()
        base = train.loc[mask, 'y_pred'].to_numpy()
        bias = base + stats['mu']
        rho, p = permutation_spearman(trend, centered, 200, 42)
        w, shrink = 0., 0.
        eligible = p < .05 and abs(rho) >= .2 and np.std(trend) > EPS
        if eligible:
            def objective(weight):
                error = (truth - bias) / stats['sigma'] - weight * trend
                return float(np.mean(error**2) + RIDGE_ALPHA * weight**2)

            w = float(minimize_scalar(objective, bounds=(-3., 3.), method='bounded').x)
            shrink = float(min(1., .35 + .65 * abs(rho)) * np.sqrt(len(truth) / (len(truth) + 10.)))
        adaptive = w * shrink
        rb = float(np.sqrt(np.mean((truth - base)**2)))
        rbi = float(np.sqrt(np.mean((truth - bias)**2)))
        rfull = float(np.sqrt(np.mean((truth - bias - adaptive * stats['sigma'] * trend)**2)))
        gain = rbi - rfull
        activated = bool(eligible and gain > 0. and abs(adaptive) > EPS)
        if not activated:
            adaptive, rfull, gain = 0., rbi, 0.
        if activated:
            active += 1
            rho_sum += abs(rho)
            gain_sum += gain
            gain_rel += gain / (rbi + EPS)
        zones[str(int(z))] = {'zone_id': int(z), 'sample_size': len(truth), 'rho_centered': rho,
                             'p_value': p, 'trend_mean': mean, 'trend_std': std,
                             'clip_z': CLIP_Z, 'w_opt': w, 'lambda_shrink': shrink,
                             'ridge_alpha': RIDGE_ALPHA,
                             'w_adaptive': adaptive, 'activated': activated,
                             'rmse_base': rb, 'rmse_bias_only': rbi, 'rmse_bias_plus_sr': rfull,
                             'delta_rmse_from_sr': gain,
                             'training_clipped_n': int(np.sum(np.abs((raw[mask] - mean) / std) > CLIP_Z))}
    return {'activated_zones': active, 'rho_sum': rho_sum, 'delta_rmse_sum': gain_sum,
            'delta_rmse_rel_sum': gain_rel, 'per_zone': zones}

def select_calibration(candidates, train, X, state):
    best_key = None
    selected = None
    for index, (equation, complexity) in enumerate(candidates):
        fitted = evaluate_candidate(equation, train, X, state)
        key = (fitted['activated_zones'], fitted['delta_rmse_rel_sum'], fitted['rho_sum'], -complexity)
        if best_key is None or key > best_key:
            best_key = key
            selected = {
                **state, 'equation': equation, 'complexity': int(complexity),
                'candidate_index': index,
                'variant_settings': {'variant': 'ridge', 'ridge_alpha': RIDGE_ALPHA,
                                     'max_complexity': None, 'allow_variable_denominator': True},
                'training_selection': fitted, 'clip_z': CLIP_Z,
                'normalization_scope': 'calibration training samples only',
                'selection_scope': 'training-generated candidate pool scored with bounded trend and ridge weights',
                'p_value_role': 'training activation heuristic, not selection-adjusted significance',
            }
    if selected is None:
        raise ValueError('Empty symbolic candidate pool')
    return selected


def apply_calibration(state, X, frame):
    result = frame[['global_idx', 'zone_id', 'y_pred']].copy()
    raw = predict_expression(state['equation'], X, state['feature_names'], result.global_idx)
    for z in sorted(result.zone_id.unique()):
        mask = (result.zone_id == z).to_numpy()
        stats = state['zone_stats'][str(int(z))]
        gate = state['training_selection']['per_zone'][str(int(z))]
        unbounded = (raw[mask] - gate['trend_mean']) / gate['trend_std']
        bounded = np.clip(unbounded, -state['clip_z'], state['clip_z'])
        bias = result.loc[mask, 'y_pred'].to_numpy() + stats['mu']
        correction = gate['w_adaptive'] * stats['sigma'] * bounded if gate['activated'] else np.zeros(np.sum(mask))
        result.loc[mask, 'y_pred_bias_only'] = bias
        result.loc[mask, 'y_pred_bias_plus_sr'] = bias + correction
        result.loc[mask, 'trend_normalized_unbounded'] = unbounded
        result.loc[mask, 'trend_normalized_bounded'] = bounded
        result.loc[mask, 'trend_clipped'] = np.abs(unbounded) > state['clip_z']
        result.loc[mask, 'sr_correction'] = correction
        result.loc[mask, 'sr_activated'] = gate['activated']
    if not np.isfinite(result[['y_pred', 'y_pred_bias_only', 'y_pred_bias_plus_sr']].to_numpy(dtype=float)).all():
        raise ValueError('Nonfinite prediction')
    return result


def attach_labels(result, data, target, fold):
    ids = result.global_idx.to_numpy(dtype=int)
    result = result.copy()
    result['y_true'] = data['y'][target][ids]
    result['composition_id'] = data['comp_id'][ids]
    result['target'], result['fold'] = target, fold
    for stage, col in [('base', 'y_pred'), ('bias_only', 'y_pred_bias_only'), ('bias_plus_sr', 'y_pred_bias_plus_sr')]:
        result['residual_' + stage] = result.y_true - result[col]
    return result


def metrics(truth, prediction):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    error = truth - prediction
    sst = float(np.sum((truth - np.mean(truth))**2))
    return {'r2': float(1. - np.sum(error**2) / sst) if len(truth) > 1 and sst > 0 else None,
            'rmse': float(np.sqrt(np.mean(error**2))), 'mae': float(np.mean(np.abs(error)))}


def metric_rows(frame, target, fold='pooled_oos'):
    rows = []
    for label, group in [('calibration_zones_only', frame)] + [(f'zone_{z}', frame[frame.zone_id == z]) for z in sorted(frame.zone_id.unique())]:
        row = {'target': target, 'fold': fold, 'subset': label, 'n': len(group)}
        for stage, col in [('base', 'y_pred'), ('bias_only', 'y_pred_bias_only'), ('bias_plus_sr', 'y_pred_bias_plus_sr')]:
            row.update({f'{stage}_{name}': value for name, value in metrics(group.y_true, group[col]).items()})
        row['delta_rmse_bias_vs_base'] = row['base_rmse'] - row['bias_only_rmse']
        row['delta_rmse_sr_vs_bias'] = row['bias_only_rmse'] - row['bias_plus_sr_rmse']
        row['delta_rmse_total_vs_base'] = row['base_rmse'] - row['bias_plus_sr_rmse']
        rows.append(row)
    return rows

def base_predictions(data, target, config, params, valid, held, fold, scratch):
    from nested_core import feature_matrix, fit_pipeline

    X, names = feature_matrix(data, target, 'knowledge15')
    y = data['y'][target].copy()
    y[held] = np.nan
    frames, training = [], {}
    for zone in ZONES[target]:
        fit_ids = valid[(data['zone'][valid] != zone) & ~np.isin(valid, held)]
        pred_ids = valid[data['zone'][valid] == zone]
        seed = SEED + TARGETS.index(target) * 100000 + fold * 1000 + zone * 100
        pred, _, _ = fit_pipeline(X, y, names, fit_ids, pred_ids, data['zone'], config,
                                  params, scratch / f'zone_{zone}', seed)
        frames.append(pd.DataFrame({'global_idx': pred_ids, 'zone_id': zone, 'y_pred': pred}))
        training[str(zone)] = fit_ids.tolist()
    return pd.concat(frames, ignore_index=True).set_index('global_idx'), training


def write_target(output, target, predictions, models):
    from nested_core import write_json

    path = output / target
    path.mkdir()
    pooled = pd.concat(predictions, ignore_index=True)
    if pooled.global_idx.duplicated().any():
        raise ValueError('Duplicate held-out sample')
    pooled.to_csv(path / f'{target}_oos_predictions.csv', index=False)
    for fold, selected in enumerate(models):
        write_json(path / f'fold_{fold:03d}' / 'selected_calibration.json', selected)
    return metric_rows(pooled, target)


def run_target(output, data, target, config, params):
    predictions, models = [], []
    valid, splits, _ = make_splits(data, target, 3, SEED)
    for fold, (train_ids, held) in enumerate(splits):
        print(f'{target}, fold {fold + 1}/3', flush=True)
        with TemporaryDirectory(dir=output) as working:
            scratch = Path(working)
            base, training = base_predictions(data, target, config, params, valid, held, fold, scratch)
            train = base.loc[train_ids].reset_index()
            train['y_true'] = data['y'][target][train_ids]
            train['residual'] = train.y_true - train.y_pred
            frame, X, state = prepare_calibration(train, data['X31'], target)
            candidates = search_equations(frame, X, state, scratch / 'pysr')
            selected = select_calibration(candidates, train, X, state)
            result = apply_calibration(selected, X, base.loc[held].reset_index())
            predictions.append(attach_labels(result, data, target, fold))
            models.append({**selected, 'test_indices': held.tolist(), 'base_training_indices': training})
    return write_target(output, target, predictions, models)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-dir', type=Path, default=ROOT.parent / '01_model_training_and_evaluation')
    parser.add_argument('--data-dir', type=Path, help='Input snapshot (default: PROJECT_DIR/data_clean)')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'sr_patch_results_oos_ridge')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    project, output = args.project_dir.resolve(), args.output_dir.resolve()
    if output.exists():
        parser.error(f'Choose a new output directory: {output}')
    try:
        pysr_version = version('pysr')
    except PackageNotFoundError:
        parser.error('Install PySR 1.5.9 and its Julia backend for symbolic fitting.')
    if pysr_version.split('.')[0] == '0':
        parser.error(f'Found PySR {pysr_version}. The archived search used PySR 1.5.9.')
    from nested_core import load_data

    manifest = json.loads((project / 'unified_results_clean/run_manifest.json').read_text(encoding='utf-8'))
    data_dir = args.data_dir.resolve() if args.data_dir else project / 'data_clean'
    configs = project / 'configs_wide'
    params = SimpleNamespace(**{**manifest['parameters'], 'threads': args.threads, 'compute_shap': False,
                                'data_dir': data_dir, 'process_data': data_dir / 'data5.xlsx',
                                'composition_csv': data_dir / 'composition_at_percent.csv',
                                'feature_config': configs / 'feature_engineering_config.json'})
    data = load_data(params)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as working:
        stage = Path(working) / output.name
        stage.mkdir()
        rows = []
        for target in TARGETS:
            config = json.loads((configs / f'{target}_config.json').read_text(encoding='utf-8'))
            rows.extend(run_target(stage, data, target, config, params))
        pd.DataFrame(rows).to_csv(stage / 'oos_summary_all.csv', index=False)
        stage.rename(output)


if __name__ == '__main__':
    main()
