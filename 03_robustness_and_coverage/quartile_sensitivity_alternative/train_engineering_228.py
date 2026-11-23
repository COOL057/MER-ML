import numpy as np
import pandas as pd
import optuna
from catboost import Pool


def fit_pipeline(X, y, names, train_ids, test_ids, groups, config, args, path, seed):
    from nested_core import split_indices, cached_selection, write_json, model, metrics
    path.mkdir(parents=True, exist_ok=True)
    if np.intersect1d(train_ids, test_ids).size:
        raise ValueError('Outer train/test overlap')
    if groups is not None and set(groups[train_ids]) & set(groups[test_ids]):
        raise ValueError('Outer group overlap')
    inner = split_indices(train_ids, groups, args.inner_splits, seed)
    prepared, split_audit = [], []
    for i, (tr, va) in enumerate(inner):
        if np.intersect1d(tr, va).size or not (set(tr) | set(va)) <= set(train_ids):
            raise ValueError('Invalid inner split')
        selection = cached_selection(path / f'inner_{i:02d}_features.json', X, y, names, tr, groups, args, seed + i * 1000)
        columns = selection['selected_indices']
        selected_names = selection['selected_features']
        prepared.append((tr, va, columns, selected_names))
        split_audit.append({'inner_fold': i, 'train_indices': tr.tolist(), 'validation_indices': va.tolist(),
                            'selector_train_indices': selection['train_indices'], 'selected_features': selected_names})
    write_json(path / 'split_audit.json', {'outer_train_indices': train_ids.tolist(),
                'outer_test_indices': test_ids.tolist(), 'grouped_inner_cv': groups is not None, 'inner_splits': split_audit})
    space = config['search_space']

    def objective(trial):
        params = {}
        for key, spec in space.items():
            if spec['type'] == 'int':
                params[key] = trial.suggest_int(key, spec['low'], spec['high'])
            elif spec['type'] == 'float':
                params[key] = trial.suggest_float(key, spec['low'], spec['high'], log=spec.get('log', False))
            else:
                raise ValueError(f'Unsupported search space: {key}')
        cap = int(params.get('iterations', 2000))
        if args.iteration_cap is not None:
            cap = min(cap, args.iteration_cap)
        true, pred, counts = [], [], []
        for tr, va, cols, selected_names in prepared:
            m = model(params, cap, args, seed, args.early_stopping)
            m.fit(Pool(X[np.ix_(tr, cols)], y[tr], feature_names=selected_names),
                  eval_set=Pool(X[np.ix_(va, cols)], y[va], feature_names=selected_names))
            true.extend(y[va].tolist())
            pred.extend(m.predict(X[np.ix_(va, cols)]).tolist())
            counts.append(int(m.get_best_iteration()) + 1 if m.get_best_iteration() >= 0 else int(m.tree_count_))
        trial.set_user_attr('inner_tree_counts', counts)
        score = metrics(true, pred)['r2']
        if score is None or not np.isfinite(score):
            raise ValueError('Inner CV R² is undefined; insufficient target variation')
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed),
                               storage=f'sqlite:///{(path / "optuna.sqlite3").resolve().as_posix()}',
                               study_name='nested', load_if_exists=True)
    for old in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)):
        study.tell(old.number, state=optuna.trial.TrialState.FAIL)
    while len(study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))) < args.n_trials:
        trial_id = len(study.trials)
        study.sampler = optuna.samplers.TPESampler(seed=seed + trial_id)
        study.optimize(objective, n_trials=1)
        if (trial_id + 1) % 25 == 0 or trial_id == 0:
            print(f'  Optuna {path.name}: trial={trial_id + 1}/{args.n_trials}, inner R2={study.best_value:.4f}', flush=True)
    trial = study.best_trial
    counts = trial.user_attrs['inner_tree_counts']
    avg = int(np.mean(counts))
    iterations = max(100, min(2000, int(avg * args.iteration_multiplier)))
    if args.iteration_cap is not None:
        iterations = min(iterations, args.iteration_cap)
    selection = cached_selection(path / 'outer_features.json', X, y, names, train_ids, groups, args, seed + 50000)
    cols, selected_names = selection['selected_indices'], selection['selected_features']
    m = model(trial.params, iterations, args, seed)
    m.fit(Pool(X[np.ix_(train_ids, cols)], y[train_ids], feature_names=selected_names))
    m.save_model(str(path / 'model.cbm'))
    metadata = {'best_params': trial.params, 'final_iterations': iterations, 'avg_best_iteration': avg,
                'inner_tree_counts': counts, 'feature_names': selected_names,
                'n_trials': len(study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))),
                'inner_splits_actual': len(inner),
                'inner_selection_r2': float(trial.value), 'loss_function': 'RMSE',
                'use_log_transform': False, 'log_offset': None, 'actual_model_params': m.get_all_params(),
                'outer_train_indices': train_ids.tolist(), 'outer_test_indices': test_ids.tolist(),
                'feature_selection_scope': 'inner-training folds and outer-training refit',
                'early_stopping_scope': 'inner validation only'}
    write_json(path / 'model_config.json', metadata)
    pd.DataFrame([{'trial': t.number, 'state': t.state.name, 'inner_r2': t.value, **t.params}
                  for t in study.trials]).to_csv(path / 'optimization_trials.csv', index=False)
    pred = m.predict(X[np.ix_(test_ids, cols)]) if len(test_ids) else np.array([])
    shap = None
    if len(test_ids) and args.compute_shap:
        shap = m.get_feature_importance(Pool(X[np.ix_(test_ids, cols)], feature_names=selected_names), type='ShapValues')
    return pred, shap, metadata
