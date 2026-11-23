import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    x = np.load(args.data / 'X_28dim.npy')
    composition = pd.read_csv(args.data / 'composition_at_percent.csv').fillna(0).round(3)
    ids = composition.apply(lambda row: '_'.join(f'{value:.3f}' for value in row), axis=1)
    data = pd.DataFrame(dict(global_idx=np.arange(len(x)), comp_id=ids, Ti=x[:, 18], ti=x[:, 19], dti=x[:, 21], tau=x[:, 22]))
    data['zone'] = np.digitize(data.dti, [25.992800000000003, 52.01440000000002]) * 3 + np.digitize(data.tau, [1.0, 1.6139453980627756])
    summary = []
    records = []
    for (temperature, time), group in data.groupby(['Ti', 'ti']):
        if group.comp_id.nunique() < 2:
            continue
        summary.append(dict(Ti=temperature, ti=time, n_records=len(group), n_compositions=group.comp_id.nunique(), dti_min=group.dti.min(), dti_max=group.dti.max(), tau_min=group.tau.min(), tau_max=group.tau.max(), n_zones=group.zone.nunique(), sample_ids=','.join(map(str, group.global_idx))))
        records.append(group)
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(args.output / 'same_processing_summary.csv', index=False, encoding='utf-8-sig')
    pd.concat(records, ignore_index=True).to_csv(args.output / 'same_processing_samples.csv', index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    main()
