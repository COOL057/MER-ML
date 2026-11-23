#!/usr/bin/env python3
"""Quartile LOPZO using the manuscript's nested training functions."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
try:
    import fcntl
except ImportError:
    fcntl = None
    import msvcrt

import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str((ROOT.parent.parent / '01_model_training_and_evaluation')))
import nested_core as core

TARGETS = core.TARGETS

def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target', required=True, choices=TARGETS)
    p.add_argument('--data-dir', type=Path, default=(ROOT.parent.parent / '01_model_training_and_evaluation')/'data_clean')
    p.add_argument('--config-dir', type=Path, default=(ROOT.parent.parent / '01_model_training_and_evaluation')/'configs_wide')
    p.add_argument('--output-dir', type=Path, default=ROOT/'results')
    p.add_argument('--n-trials', type=int, default=400)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--inner-splits', type=int, default=10)
    p.add_argument('--stability-rounds', type=int, default=50)
    p.add_argument('--stability-splits', type=int, default=5)
    p.add_argument('--selector-iterations', type=int, default=500)
    p.add_argument('--selector-early-stopping', type=int, default=100)
    p.add_argument('--top-k', type=int, default=15)
    p.add_argument('--early-stopping', type=int, default=200)
    p.add_argument('--iteration-multiplier', type=float, default=1.2)
    p.add_argument('--iteration-cap', type=int)
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--max-outer-folds', type=int, help='Smoke test only; writes to a separate output directory')
    a=p.parse_args()
    for n in ['n_trials','threads','stability_rounds','selector_iterations','selector_early_stopping','top_k','early_stopping']:
        if getattr(a,n)<1: p.error(f'{n} must be positive')
    if min(a.inner_splits,a.stability_splits)<2 or a.iteration_multiplier<=0: p.error('Invalid split or iteration settings')
    if a.iteration_cap is not None and a.iteration_cap<1: p.error('Invalid iteration cap')
    if a.max_outer_folds is not None and a.max_outer_folds<1: p.error('Invalid max folds')
    a.data_dir=a.data_dir.resolve(); a.config_dir=a.config_dir.resolve(); a.output_dir=a.output_dir.resolve()
    a.composition_csv=a.data_dir/'composition_at_percent.csv'; a.process_data=a.data_dir/'data5.xlsx'
    a.feature_config=a.config_dir/'feature_engineering_config.json'; a.feature_mode='knowledge15'
    a.zone_boundaries=None; a.compute_shap=False
    return a

def main():
    a=arguments(); data=core.load_data(a)
    d=data['dti']; t=data['tau']
    db=np.percentile(d,[25,50,75]); tb=np.percentile(t,[25,50,75])
    if not (np.all(np.diff(db)>0) and np.all(np.diff(tb)>0)):
        raise ValueError('Duplicated quartile boundaries; equal-width bin protocol is invalid')
    data['zone']=np.digitize(d,db)*4+np.digitize(t,tb)
    core.ZONE_LABELS=[f'DTI_Q{di+1} × tau_Q{ti+1}' for di in range(4) for ti in range(4)]
    target=a.target; y=data['y'][target]; valid=np.flatnonzero(np.isfinite(y)); zone=data['zone']
    if len(valid)==0: raise ValueError('No finite labels')
    counts=[]
    for z in range(16):
        ids=valid[zone[valid]==z]
        counts.append({'zone_id':z,'dti_quartile':z//4+1,'tau_quartile':z%4+1,
                       'n_all':int((zone==z).sum()),'n_valid':len(ids),
                       'n_compositions':len(np.unique(data['comp_id'][ids]))})
    report={'target':target,'n_total':len(y),'n_valid':len(valid),'dti_edges':db.tolist(),'tau_edges':tb.tolist(),
            'boundary_rule':'np.digitize right=False; equal values go to upper bin',
            'partition':'unlabelled full-sample quartiles fixed before folds; transductive',
            'zone_counts':counts}
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    if a.preflight: return
    out=a.output_dir/target; out.mkdir(parents=True,exist_ok=True)
    # flock is released by the OS after termination, including after SIGKILL.
    with open(out/'running.lock','a+') as lock:
        try:
            if fcntl is not None:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            else:
                lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        except (BlockingIOError,OSError): raise RuntimeError(f'Another process is running for {target}: {out}')
        config_path=a.config_dir/f'{target}_config.json'; config=json.loads(config_path.read_text(encoding='utf-8'))
        if config.get('use_log_transform',False) or config.get('use_winsorize',False) or config.get('loss_function','RMSE')!='RMSE':
            raise ValueError('Only original-scale RMSE protocol supported')
        packages=['numpy','pandas','scikit-learn','catboost','optuna','openpyxl']
        sources=data['files']+[config_path]+[(ROOT.parent.parent / '01_model_training_and_evaluation')/n for n in ['nested_core.py','train_engineering_228.py','feature_stability_selection.py']]+[ROOT/'quartile_lopzo.py']
        parameters={k:v for k,v in vars(a).items() if k in ['target','n_trials','threads','seed','inner_splits','stability_rounds','stability_splits','selector_iterations','selector_early_stopping','top_k','early_stopping','iteration_multiplier','iteration_cap','max_outer_folds']}
        manifest={'protocol':'quartile-nested-lopzo-v1','parameters':parameters,'source_sha256':{str(p):core.sha256(p) for p in sources},
                  'versions':{p:importlib.metadata.version(p) for p in packages},'python':sys.version,
                  'boundary_rule':report['boundary_rule'],'partition':report['partition'],
                  'dti_edges':db.tolist(),'tau_edges':tb.tolist(), 'shap':False}
        mp=out/'manifest.json'
        if mp.exists():
            if json.loads(mp.read_text(encoding='utf-8'))!=manifest: raise ValueError('Manifest mismatch: choose a NEW output directory')
        else:
            if any(x.name not in ['running.lock'] for x in out.iterdir()): raise ValueError('Output not empty and no manifest')
            core.write_json(mp,manifest)
        core.write_json(out/'preflight.json',report)
        pd.DataFrame(counts).to_csv(out/'zone_coverage.csv',index=False,encoding='utf-8-sig')
        X,names=core.feature_matrix(data,target,a.feature_mode); paths=[]
        zones=np.unique(zone[valid])
        for count,z in enumerate(zones):
            if a.max_outer_folds is not None and count>=a.max_outer_folds: break
            tr=valid[zone[valid]!=z]; te=valid[zone[valid]==z]; fold=f'zone_{int(z):02d}'; path=out/'folds'/fold
            if len(np.unique(zone[tr]))<2: raise ValueError('At least two training zones required')
            print(f'{target} {fold} train={len(tr)} test={len(te)}',flush=True)
            if (path/'complete.json').exists():
                core.verify_complete(path); print(f'{fold}: verified and skipped',flush=True)
            else:
                seed=a.seed+TARGETS.index(target)*100000+10000+count
                pred,shap,meta=core.fit_pipeline(X,y,names,tr,te,zone,config,a,path,seed)
                frame=pd.DataFrame(core.records(data,target,'lopzo',te,pred,shap,meta,fold,0,int(z),tr))
                frame.to_csv(path/'predictions.csv',index=False)
                core.mark_complete(path,target=target,scenario='quartile_lopzo',zone=int(z))
            paths.append(path)
        df=pd.concat([pd.read_csv(p/'predictions.csv',dtype={'comp_id':str,'comp_short':str}) for p in paths],ignore_index=True)
        if df.global_idx.duplicated().any(): raise ValueError('Duplicate held-out sample IDs')
        complete=set(df.global_idx)==set(valid)
        df.sort_values('global_idx').to_csv(out/'predictions.csv',index=False,encoding='utf-8-sig')
        core.write_json(out/'run_summary.json',{'target':target,'complete':bool(complete),'n_folds':len(paths),
                        'expected_folds':len(zones),'overall':core.metrics(df.y_true,df.y_pred)})
        print(f'{target} complete={complete} folds={len(paths)}/{len(zones)}',flush=True)

if __name__=='__main__': main()
