import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rcv', type=Path, required=True)
    parser.add_argument('--loco', type=Path, required=True)
    parser.add_argument('--lopzo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for scenario, path in [('RCV', args.rcv), ('LOCO', args.loco), ('LOPZO', args.lopzo)]:
        data = pd.read_csv(path)
        for target, group in data.groupby('target'):
            risk = group.specified_risk.eq(1)
            mapped = group.loc[risk, 'abs_error']
            other = group.loc[~risk, 'abs_error']
            share = len(mapped) / len(group)
            error_share = mapped.sum() / group.abs_error.sum()
            rows.append(dict(scenario=scenario, target=target, n_predictions=len(group), n_unique=group.global_idx.nunique(), risk_n_predictions=len(mapped), risk_n_unique=group.loc[risk, 'global_idx'].nunique(), risk_sample_share=share, risk_absolute_error_share=error_share, enrichment_factor=error_share / share, risk_mae=mapped.mean(), other_mae=other.mean(), mae_ratio=mapped.mean() / other.mean()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    main()
