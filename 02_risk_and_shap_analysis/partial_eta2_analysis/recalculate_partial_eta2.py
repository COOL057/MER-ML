import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm


TARGETS = ['Rm', 'Re', 'A', 'Ak']
TERMS = {'Novelty': 'C(novelty_layer, Sum)', 'Process zone': 'C(zone_id, Sum)',
         'Novelty x Zone': 'C(novelty_layer, Sum):C(zone_id, Sum)'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit(df, target, analysis, interaction=True):
    df['novelty_layer'] = df['novelty_layer'].astype(object)
    df['zone_id'] = df['zone_id'].astype(object)
    formula = 'log_abs_residual ~ C(novelty_layer, Sum) ' + ('*' if interaction else '+') + ' C(zone_id, Sum)'
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model = ols(formula, data=df).fit()
        table = anova_lm(model, typ=3)
    design = model.model.exog
    rank = int(np.linalg.matrix_rank(design))
    rows = []
    for effect, term in TERMS.items():
        if term not in table.index:
            continue
        ss, error = float(table.loc[term, 'sum_sq']), float(table.loc['Residual', 'sum_sq'])
        rows.append({'target': target, 'analysis': analysis, 'effect': effect,
                     'partial_eta2': ss / (ss + error), 'ss_effect': ss, 'ss_error': error,
                     'df_effect_nominal': float(table.loc[term, 'df']), 'df_error': float(model.df_resid),
                     'nominal_F': float(table.loc[term, 'F']),
                     'nominal_p_value_not_cluster_adjusted': float(table.loc[term, 'PR(>F)']),
                     'n_observations': len(df), 'n_compositions': df.comp_id.nunique(),
                     'design_rank': rank, 'design_columns': design.shape[1],
                     'rank_deficient': rank < design.shape[1]})
    diagnostics = {'target': target, 'analysis': analysis, 'formula': formula,
                   'rank': rank, 'columns': design.shape[1], 'df_error': float(model.df_resid),
                   'n_observations': len(df), 'n_compositions': int(df.comp_id.nunique()),
                   'warnings': [str(w.message) for w in caught], 'ss_error': float(model.ssr)}
    return rows, diagnostics, table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package-dir', type=Path, default=Path(__file__).resolve().parents[2] / '01_model_training_and_evaluation')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    all_rows, diagnostics, hashes, thresholds, cell_rows, inputs = [], [], [], [], [], []
    for target in TARGETS:
        residual = args.package_dir / 'unified_results_clean/residuals' / f'{target}_lopzo_predictions.csv'
        df = pd.read_csv(residual, dtype={'comp_id': str})
        if not df.global_idx.is_unique or not np.allclose(df.abs_residual, np.abs(df.y_true-df.y_pred)):
            raise ValueError(f'Invalid prediction coverage/residuals: {target}')
        if not np.isfinite(df[['novelty', 'abs_residual']]).all().all():
            raise ValueError(f'Nonfinite values: {target}')
        low, high = np.quantile(df.novelty, [1/3, 2/3])
        df['novelty_layer'] = np.where(df.novelty <= low, 'Core', np.where(df.novelty <= high, 'Transition', 'Frontier'))
        df['zone_id'] = df.zone_id.astype(int).astype(str)
        df['log_abs_residual'] = np.log1p(df.abs_residual)
        thresholds.append({'target': target, 'low': low, 'high': high,
                           'source': 'new fold-local LOPZO novelty empirical 1/3 and 2/3 quantiles'})
        counts = pd.crosstab(df.novelty_layer, df.zone_id).reindex(index=['Core', 'Transition', 'Frontier'], columns=[str(i) for i in range(9)], fill_value=0)
        for layer in counts.index:
            for zone in counts.columns:
                cell_rows.append({'target': target, 'novelty_layer': layer, 'zone_id': int(zone), 'n': int(counts.loc[layer, zone])})
        for analysis, subset, interaction in [('original_method_all9', df, True),
                                             ('complete_crossing_zones_sensitivity', df[df.zone_id.isin(counts.columns[(counts > 0).all(axis=0)])], True),
                                             ('additive_all9_sensitivity', df, False)]:
            rows, diagnostic, table = fit(subset.copy(), target, analysis, interaction)
            all_rows.extend(rows)
            diagnostics.append(diagnostic)
            table.to_csv(out / f'{target}_{analysis}_anova.csv', encoding='utf-8-sig')
        df[['global_idx', 'comp_id', 'zone_id', 'novelty', 'novelty_layer', 'abs_residual', 'log_abs_residual']].assign(target=target).to_csv(out / f'{target}_analysis_data.csv', index=False, encoding='utf-8-sig')
        hashes.append({'file': str(residual), 'sha256': digest(residual)})
    frame = pd.DataFrame(all_rows)
    frame.to_csv(out / 'partial_eta2_detailed.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(thresholds).to_csv(out / 'novelty_thresholds.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(cell_rows).to_csv(out / 'cell_counts.csv', index=False, encoding='utf-8-sig')
    for analysis in frame.analysis.unique():
        pivot = frame[frame.analysis == analysis].pivot(index='target', columns='effect', values='partial_eta2').reindex(TARGETS)
        pivot.to_csv(out / f'table_{analysis}.csv', encoding='utf-8-sig')
    report = {'versions': {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scipy', 'statsmodels', 'patsy']},
              'source_files': hashes + inputs, 'diagnostics': diagnostics,
              'note': 'Original all9 Type III table has a rank-deficient factorial design. No reported p value is composition-cluster adjusted; exploratory/descriptive reanalysis only. Sensitivity subsets do not replace original data.'}
    (out / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(frame[['target', 'analysis', 'effect', 'partial_eta2', 'design_rank', 'design_columns']].to_string(index=False))


if __name__ == '__main__':
    main()
