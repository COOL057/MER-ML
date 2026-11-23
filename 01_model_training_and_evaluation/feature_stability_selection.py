import numpy as np
import pandas as pd
from catboost import Pool


def select_features(X, y, names, train_ids, groups, args, seed):
    from nested_core import model, split_indices
    if args.feature_mode != 'knowledge15':
        return {'selected_features': names.copy(), 'selected_indices': list(range(len(names))),
                'train_indices': train_ids.tolist(), 'candidate_features': names.copy()}
    if args.top_k > len(names):
        raise ValueError('top_k exceeds the candidate feature count')
    fixed = {'depth': 5, 'learning_rate': .06, 'l2_leaf_reg': 30., 'min_data_in_leaf': 10,
             'subsample': .7, 'bagging_temperature': 1., 'random_strength': 3., 'colsample_bylevel': .8}
    importance, ranks = [], []
    for r in range(args.stability_rounds):
        if r == 0 or (r + 1) % 10 == 0:
            print(f'  Feature selection n={len(train_ids)} round={r + 1}/{args.stability_rounds}', flush=True)
        fi = []
        splits = split_indices(train_ids, None, args.stability_splits, 42 + r)
        for j, (tr, va) in enumerate(splits):
            m = model(fixed, args.selector_iterations, args, 42 + r * 100 + j, args.selector_early_stopping)
            m.fit(Pool(X[tr], y[tr], feature_names=names), eval_set=Pool(X[va], y[va], feature_names=names))
            fi.append(m.get_feature_importance())
        avg = np.mean(fi, axis=0)
        importance.append(avg)
        rank = len(names) - np.argsort(np.argsort(avg))
        ranks.append(rank)
    importance, ranks = np.asarray(importance), np.asarray(ranks)
    mean = importance.mean(axis=0)
    freq = (ranks <= args.top_k).mean(axis=0)
    score = (mean / mean.max() if mean.max() > 0 else mean) * freq
    selected = np.argsort(score)[::-1][:args.top_k]
    return {'selected_features': [names[i] for i in selected], 'selected_indices': selected.tolist(),
            'candidate_features': names.copy(), 'train_indices': train_ids.tolist(),
            'mean_importance': mean.tolist(), 'std_importance': importance.std(axis=0).tolist(),
            'importance_matrix': importance.tolist(), 'top_k_frequency': freq.tolist(),
            'stability_score': score.tolist(), 'mean_rank': ranks.mean(axis=0).tolist(),
            'std_rank': ranks.std(axis=0).tolist(), 'n_rounds': args.stability_rounds,
            'n_splits_requested': args.stability_splits, 'top_k': args.top_k,
            'n_splits_actual': min(args.stability_splits, len(train_ids)),
            'resampling': 'repeated shuffled sample-wise KFold within supplied training indices',
            'selector_params': fixed, 'selector_iterations': args.selector_iterations,
            'selector_early_stopping': args.selector_early_stopping}



def export_selection(selection, directory, target):
    from nested_core import write_json
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / f'{target}_stability_scores.json', selection)
    rows = []
    selected = set(selection['selected_features'])
    for i, name in enumerate(selection['candidate_features']):
        row = {'feature': name, 'selected': name in selected}
        for key in ['mean_importance', 'std_importance', 'top_k_frequency', 'stability_score', 'mean_rank', 'std_rank']:
            if key in selection:
                row[key] = selection[key][i]
        rows.append(row)
    df = pd.DataFrame(rows)
    if 'stability_score' in df:
        df = df.sort_values('stability_score', ascending=False)
    df.to_csv(directory / f'{target}_stability_ranking.csv', index=False)



if __name__ == '__main__':
    from evaluate_and_collect_residuals import main
    main(default_mode='select')
