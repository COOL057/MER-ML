

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
TARGETS=('Rm','Re','A','Ak')

def stats(df, grouping):
    rows=[]
    for key,g in df.groupby(grouping,dropna=False,sort=True):
        key=key if isinstance(key,tuple) else (key,)
        r=dict(zip(grouping,key)); err=(g.y_true-g.y_pred).to_numpy(float); a=np.abs(err)
        r.update(n_samples=len(g),n_compositions=g.comp_id.nunique(),mae=float(a.mean()),
                 rmse=float(np.sqrt(np.mean(err**2))),median_abs_error=float(np.median(a)),
                 q25_abs_error=float(np.quantile(a,.25)),q75_abs_error=float(np.quantile(a,.75)),
                 observed_min=float(g.y_true.min()),observed_max=float(g.y_true.max()),
                 sparse_warning=(len(g)<5 or g.comp_id.nunique()<2))
        rows.append(r)
    return rows

def checked_predictions(path,target,labels,n):
    f=pd.read_csv(path,dtype={'comp_id':str})
    need={'global_idx','comp_id','y_true','y_pred','dti','tau'}
    if not need.issubset(f): raise ValueError(f'Missing columns in {path}: {need-set(f)}')
    if f.global_idx.duplicated().any(): raise ValueError(f'Duplicate sample IDs in {path}')
    idx=f.global_idx.to_numpy(dtype=int)
    valid=np.flatnonzero(np.isfinite(labels))
    if set(idx)!=set(valid): raise ValueError(f'{target}: {path} does not contain exactly all valid labels')
    if np.any(idx<0) or np.any(idx>=n) or not np.allclose(f.y_true,labels[idx],rtol=1e-6,atol=1e-7):
        raise ValueError(f'{target}: target alignment mismatch in {path}')
    if not np.isfinite(f[['dti','tau','y_true','y_pred']].to_numpy()).all(): raise ValueError(f'Non-finite values: {path}')
    return f

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=ROOT.parent.parent/'01_model_training_and_evaluation'/'data_clean'); p.add_argument('--results-dir',type=Path,default=ROOT/'results')
    p.add_argument('--tertile-dir',type=Path,default=ROOT.parent.parent/'01_model_training_and_evaluation'/'unified_results_clean'/'residuals')
    p.add_argument('--output-dir',type=Path,default=ROOT/'analysis'); p.add_argument('--original-only',action='store_true')
    a=p.parse_args(); d=pd.read_excel(a.data_dir/'data5.xlsx'); dti=d['ΔTI'].to_numpy(float); tau=d['τ_norm'].to_numpy(float)
    edges={k:np.percentile(x,[25,33.33,50,66.67,75]) for k,x in [('dti',dti),('tau',tau)]}
    summaries=[]; paired=[]
    for target in TARGETS:
        labels=np.load(a.data_dir/f'y_{target}.npy',allow_pickle=False).reshape(-1)
        original=checked_predictions(a.tertile_dir/f'{target}_lopzo_predictions.csv',target,labels,len(d))
        sources=[('tertile_lopzo',original)]
        if not a.original_only:
            run=a.results_dir/target
            meta=json.loads((run/'run_summary.json').read_text(encoding='utf-8'))
            if not meta['complete']: raise ValueError(f'Incomplete quartile result: {target}')
            new=checked_predictions(run/'predictions.csv',target,labels,len(d)); sources.append(('quartile_lopzo',new))
            join=original[['global_idx','y_true','y_pred']].merge(new[['global_idx','y_pred']],on='global_idx',validate='one_to_one',suffixes=('_tertile','_quartile'))
            join.insert(0,'target',target); paired.append(join)
        for scenario,raw in sources:
            f=raw.copy(); ix=f.global_idx.to_numpy(int)
            if not (np.allclose(f.dti,dti[ix]) and np.allclose(f.tau,tau[ix])): raise ValueError(f'{scenario} feature coordinate mismatch: {target}')
            for kind,br in [('quartile',[25,50,75]),('tertile',[33.33,66.67])]:
                f['dti_group']=np.digitize(f.dti,np.percentile(dti,br))+1
                f['tau_group']=np.digitize(f.tau,np.percentile(tau,br))+1
                for group in (['dti_group'],['tau_group'],['dti_group','tau_group']):
                    for r in stats(f,group):
                        r.update(target=target,prediction_source=scenario,grouping=kind,axis='+'.join(group)); summaries.append(r)
            for axis,values,direction in [('tau',tau,'short'),('dti',dti,'high')]:
                for quant in (.25,.3333):
                    threshold=float(np.quantile(values,quant if direction=='short' else 1-quant))
                    
                    flag=f[axis]<threshold if direction=='short' else f[axis]>=threshold
                    f['extreme_group']=np.where(flag,'extreme','other')
                    for r in stats(f,['extreme_group']):
                        r.update(target=target,prediction_source=scenario,grouping=f'tail_{quant:.4f}',axis=axis,threshold=threshold,direction=direction); summaries.append(r)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(summaries).to_csv(a.output_dir/'descriptive_metrics.csv',index=False,encoding='utf-8-sig')
    if paired: pd.concat(paired,ignore_index=True).to_csv(a.output_dir/'paired_predictions.csv',index=False,encoding='utf-8-sig')
    (a.output_dir/'boundaries.json').write_text(json.dumps({k:v.tolist() for k,v in edges.items()},indent=2),encoding='utf-8')
    print(f'Saved {a.output_dir} (descriptive only; sample rows are not independent compositions).')

if __name__=='__main__': main()
