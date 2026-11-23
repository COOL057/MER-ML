from pathlib import Path
import argparse
import json, hashlib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

parser = argparse.ArgumentParser()
parser.add_argument('--project-dir', type=Path, default=Path(__file__).resolve().parents[2] / '01_model_training_and_evaluation')
parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
arguments = parser.parse_args()
ROOT = arguments.project_dir.resolve()
OUT = arguments.output_dir.resolve()
OUT.mkdir(parents=True, exist_ok=True)
BASE=['C','Si','Mn','Cr','V','Al','Ni','Mo','Cu','W','Z','Ve','χP','δb','δf','ΔFe','ΔC','Ta','Ti','ti','ΔTA','ΔTI','τ','Ms','Tt','tt','sph','tmp']
COMP=BASE[:10]; PROC=['Ta','Ti','ti','Tt','tt']; CAT=['sph','tmp']; expected={'Rm':189,'Re':176,'A':189,'Ak':199}
X=np.load(ROOT/'data_clean'/'X_28dim.npy',allow_pickle=False); assert X.shape==(217,28)
raw=pd.DataFrame(X,columns=BASE)
comp=pd.read_csv(ROOT/'data_clean'/'composition_at_percent.csv').apply(pd.to_numeric,errors='raise').fillna(0).round(3)
ids=comp.apply(lambda row:'_'.join(f'{v:.3f}' for v in row),axis=1).to_numpy()
for name in ['ti','tt']:
    if (raw[name].dropna()<0).any(): raise ValueError(f'Negative time {name}')
    raw[name]=np.log1p(raw[name])
all_data=[]; audits=[]; zones=[]; strata=[]; correlations=[]
for target in expected:
    y=np.load(ROOT/'data_clean'/f'y_{target}.npy',allow_pickle=False)
    pred=pd.read_csv(ROOT/'unified_results_clean'/'residuals'/f'{target}_lopzo_predictions.csv',dtype={'comp_id':str})
    valid=set(np.flatnonzero(np.isfinite(y))); assert len(pred)==expected[target] and not pred.global_idx.duplicated().any() and set(pred.global_idx)==valid
    assert np.allclose(pred.y_true,y[pred.global_idx.to_numpy(int)])
    for fold,sub in pred.groupby('outer_fold',sort=True):
        path=ROOT/'unified_results_clean'/'folds'/target/'lopzo'/fold
        audit=json.loads((path/'split_audit.json').read_text(encoding='utf8'))
        tr=np.array(audit['outer_train_indices'],dtype=int); te=np.array(audit['outer_test_indices'],dtype=int)
        assert set(te)==set(sub.global_idx) and not set(tr)&set(te) and set(tr)|set(te)==valid
        sub=sub.set_index('global_idx').loc[te].reset_index()
        assert np.array_equal(sub.comp_id.to_numpy(str),ids[te])
        columns=COMP+PROC; train=raw.iloc[tr][columns].copy(); test=raw.iloc[te][columns].copy()
        med=train.median();
        if med.isna().any(): raise ValueError(f'All values missing in training: {target} {fold}')
        means=train.fillna(med).mean(); std=train.fillna(med).std(ddof=0)
        zero=std[std<1e-12].index.tolist(); std=std.mask(std<1e-12,1.0)
        trn=(train.fillna(med)-means)/std; ten=(test.fillna(med)-means)/std
        # A training-constant dimension has no learned scale; omit it and disclose it.
        if zero: trn.loc[:,zero]=0; ten.loc[:,zero]=0
        tc=raw.iloc[tr][CAT].to_numpy(float); ec=raw.iloc[te][CAT].to_numpy(float)
        if not np.isfinite(tc).all() or not np.isfinite(ec).all(): raise ValueError('Missing categorical indicators')
        tm=train[['Tt','tt']].isna().to_numpy(float); em=test[['Tt','tt']].isna().to_numpy(float)
        # Equal block weights: composition 0.5 and processing 0.5.
        # Processing has 5 numerical, 2 binary and 2 missingness dimensions.
        tvec=np.column_stack([trn[COMP].to_numpy()*np.sqrt(.5/10),trn[PROC].to_numpy()*np.sqrt(.5/9),tc*np.sqrt(.5/9),tm*np.sqrt(.5/9)])
        evec=np.column_stack([ten[COMP].to_numpy()*np.sqrt(.5/10),ten[PROC].to_numpy()*np.sqrt(.5/9),ec*np.sqrt(.5/9),em*np.sqrt(.5/9)])
        dist=cdist(evec,tvec); nearest=dist.argmin(axis=1); mindist=dist[np.arange(len(te)),nearest]
        ref=cdist(tvec,tvec); np.fill_diagonal(ref,np.inf); refmin=ref.min(axis=1)
        pct=np.searchsorted(np.sort(refmin),mindist,side='right')/len(refmin)
        sub['nearest_distance']=mindist;sub['training_loo_percentile']=pct;sub['nearest_train_global_idx']=tr[nearest]
        sub['nearest_same_composition']=(ids[te]==ids[tr[nearest]])
        sub['abs_error']=abs(sub.y_true-sub.y_pred);sub['target']=target
        sub['distance_band']=pd.cut(pct,[-.001,1/3,2/3,1],labels=['low','middle','high'],include_lowest=True).astype(str)
        all_data.append(sub[['target','global_idx','outer_fold','zone_id','comp_id','dti','tau','y_true','y_pred','abs_error','nearest_distance','training_loo_percentile','distance_band','nearest_train_global_idx','nearest_same_composition']])
        audits.append({'target':target,'outer_fold':fold,'zone_id':int(sub.zone_id.iloc[0]),'n_train':len(tr),'n_test':len(te),'training_constant_features':';'.join(zero),'training_median':json.dumps(med.to_dict(),ensure_ascii=False),'training_mean':json.dumps(means.to_dict(),ensure_ascii=False),'training_sd':json.dumps(std.to_dict(),ensure_ascii=False)})
        assert sub.zone_id.nunique()==1
    df=pd.concat([v for v in all_data if v.target.iloc[0]==target],ignore_index=True)
    df['risk_group']=np.where(df.zone_id.isin([0,3,6] if target in ['Rm','Re'] else [6,7,8]),'specified_risk','other')
    for z,g in df.groupby('zone_id'):
        zones.append({'target':target,'zone_id':int(z),'n':len(g),'n_compositions':g.comp_id.nunique(),'mae':g.abs_error.mean(),'rmse':np.sqrt(np.mean(g.abs_error**2)),'distance_median':g.nearest_distance.median(),'distance_mean':g.nearest_distance.mean(),'percentile_median':g.training_loo_percentile.median(),'high_percentile_fraction':(g.training_loo_percentile>2/3).mean(),'same_comp_nearest_fraction':g.nearest_same_composition.mean()})
    for (band,group),g in df.groupby(['distance_band','risk_group'],observed=True):
        strata.append({'target':target,'distance_band':band,'risk_group':group,'n':len(g),'n_compositions':g.comp_id.nunique(),'n_zones':g.zone_id.nunique(),'mae':g.abs_error.mean(),'rmse':np.sqrt(np.mean(g.abs_error**2)),'median_abs_error':g.abs_error.median()})
    correlations.append({'target':target,'n':len(df),'n_compositions':df.comp_id.nunique(),'spearman_distance_error':spearmanr(df.nearest_distance,df.abs_error).statistic,'spearman_percentile_error':spearmanr(df.training_loo_percentile,df.abs_error).statistic,'note':'descriptive; no p-values, dependent records and fold differences'})
all_df=pd.concat(all_data,ignore_index=True); all_df.to_csv(OUT/'sample_distances.csv',index=False,encoding='utf-8-sig')
for name,values in [('fold_preprocessing',audits),('zone_summary',zones),('distance_strata',strata),('correlations',correlations)]: pd.DataFrame(values).to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig')
ex=all_df[all_df.zone_id!=6].copy()
ex['risk_group']=[('specified_risk' if z in ([0,3] if target in ['Rm','Re'] else [7,8]) else 'other') for target,z in zip(ex.target,ex.zone_id)]
out=[]
for (target,band,group),g in ex.groupby(['target','distance_band','risk_group']):
    out.append({'target':target,'distance_band':band,'risk_group':group,'n':len(g),'n_compositions':g.comp_id.nunique(),'mae':g.abs_error.mean(),'rmse':np.sqrt(np.mean(g.abs_error**2))})
pd.DataFrame(out).to_csv(OUT/'distance_strata_excluding_zone6.csv',index=False,encoding='utf-8-sig')
source=[ROOT/'data_clean'/'X_28dim.npy',ROOT/'data_clean'/'composition_at_percent.csv']+[ROOT/'data_clean'/f'y_{t}.npy' for t in expected]+[ROOT/'unified_results_clean'/'residuals'/f'{t}_lopzo_predictions.csv' for t in expected]
manifest={'features_composition':COMP,'features_process_numeric':PROC,'features_binary':CAT,'additional_flags':['Tt_missing','tt_missing'],'log1p':['ti','tt'],'block_weights':{'composition':.5,'processing':.5},'process_dimensions':9,'sample_percentile':'ECDF of training-set leave-one-row-out nearest distances per outer fold; not an independent percentile or calibrated error probability','risk_groups':{'Rm_Re':[0,3,6],'A_Ak':[6,7,8]},'note':'exploratory; sample-level repeated compositions, leave-zone-out test, empirical globally frozen zones; associations not causal; metric choice and R2 limitations','source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source}}
(OUT/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf8')
print(pd.DataFrame(correlations).to_string(index=False));print('\nSTRATA\n',pd.DataFrame(strata).to_string(index=False))
