"""Exploratory grouped cross-fitted risk ranking, fixed 17D property predictor.

Outer GroupKFold by composition; nested inner OOF property residuals supply
the risk model target. Both risk models share all data, scores and tuning,
except that the second receives two extra transformation-referenced variables.
This is exploratory, not the manuscript's nested LOPZO or a prospective test.
"""
from pathlib import Path
import argparse
import json,hashlib,platform,importlib.metadata
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parent.parent / '01_model_training_and_evaluation' / 'data_clean')
parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'raw17_vs_raw17plus_physics_risk')
arguments = parser.parse_args()
ROOT = arguments.data_dir.resolve()
OUT = arguments.output_dir.resolve()
OUT.mkdir(parents=True, exist_ok=True)
X=np.load(ROOT/'X_28dim.npy'); comp=pd.read_csv(ROOT/'composition_at_percent.csv').fillna(0).round(3)
groups=comp.apply(lambda r:'_'.join(f'{v:.3f}' for v in r),axis=1).to_numpy()
base=[*range(10),17,18,19,24,25,26,27]
assert X.shape==(217,28) and len(groups)==217 and np.isfinite(X[:,[18,19,21,22]]).all()
assert (X[:,19]>=0).all() and (X[:,22]>-1).all()
raw=X[:,base].copy();raw[:,12]=np.log1p(raw[:,12]) # base position 12: ti
aug=np.column_stack([raw,X[:,21],np.log1p(X[:,22])])
assert raw.shape[1]==17 and aug.shape[1]==19

def property_predict(train,test,y,seed):
    model=CatBoostRegressor(iterations=500,depth=6,learning_rate=.05,l2_leaf_reg=3,loss_function='RMSE',
        random_seed=seed,thread_count=4,verbose=False,allow_writing_files=False)
    model.fit(X[train][:,base],y[train]);return model.predict(X[test][:,base])

def risk_predict(features,train,test,error):
    imp=SimpleImputer(strategy='median',keep_empty_features=True).fit(features[train])
    train_x=imp.transform(features[train]);test_x=imp.transform(features[test])
    sc=StandardScaler().fit(train_x);train_x=sc.transform(train_x);test_x=sc.transform(test_x)
    _,inv,counts=np.unique(groups[train],return_inverse=True,return_counts=True)
    weights=1/counts[inv];weights=weights/weights.mean()
    model=Ridge(alpha=10).fit(train_x,np.log1p(error),sample_weight=weights)
    return model.predict(test_x)

records=[];audit=[]
for target in ['Rm','Re','A','Ak']:
    y=np.load(ROOT/f'y_{target}.npy').reshape(-1);valid=np.flatnonzero(np.isfinite(y))
    for fold,(trp,tep) in enumerate(GroupKFold(n_splits=5).split(valid,groups=groups[valid])):
        tr,te=valid[trp],valid[tep];assert not set(groups[tr]) & set(groups[te])
        inner_prediction=np.full(len(tr),np.nan);inner_fold=np.full(len(tr),-1)
        for k,(it,iv) in enumerate(GroupKFold(n_splits=4).split(tr,groups=groups[tr])):
            assert not set(groups[tr[it]]) & set(groups[tr[iv]])
            inner_prediction[iv]=property_predict(tr[it],tr[iv],y,42+10*fold+k);inner_fold[iv]=k
        assert np.isfinite(inner_prediction).all() and (inner_fold>=0).all()
        train_error=np.abs(y[tr]-inner_prediction);threshold=float(np.quantile(train_error,.8))
        prediction=property_predict(tr,te,y,420+fold);err=np.abs(y[te]-prediction)
        score17=risk_predict(raw,tr,te,train_error);score19=risk_predict(aug,tr,te,train_error)
        for j,idx in enumerate(te):
            records.append(dict(target=target,fold=fold,global_idx=int(idx),comp_id=groups[idx],
                y_true=y[idx],y_pred=prediction[j],abs_error=err[j],training_error_q80=threshold,
                high_error=int(err[j]>threshold),score_raw17=score17[j],score_raw17_plus_physics=score19[j]))
        audit.append(dict(target=target,fold=fold,train_ids=tr.tolist(),test_ids=te.tolist(),
            inner_assignments=[dict(global_idx=int(i),fold=int(k)) for i,k in zip(tr,inner_fold)],
            threshold=threshold))
        print(target,'fold',fold,'test_n',len(te),'high',int((err>threshold).sum()),flush=True)

df=pd.DataFrame(records);rows=[];summ=[]
for (target,fold),g in df.groupby(['target','fold'],sort=False):
    g=g.sort_values('global_idx');n=len(g);k=int(np.ceil(.2*n));actual=g.high_error.to_numpy()
    for model in ['raw17','raw17_plus_physics']:
        score=g['score_'+model].to_numpy();chosen=np.argsort(-score,kind='stable')[:k]
        mask=(df.target==target)&(df.fold==fold);chosen_ids=set(g.iloc[chosen].global_idx)
        df.loc[mask,'selected_'+model]=df.loc[mask,'global_idx'].isin(chosen_ids).astype(int)
        rows.append(dict(target=target,fold=fold,method=model,n=n,n_selected=k,n_high=int(actual.sum()),
            hits=int(actual[chosen].sum()),precision=float(actual[chosen].mean()),
            recall=float(actual[chosen].sum()/actual.sum()) if actual.sum() else np.nan,
            auc=float(roc_auc_score(actual,score)) if len(np.unique(actual))==2 else np.nan,
            selected_mae=float(g.abs_error.to_numpy()[chosen].mean())))
folds=pd.DataFrame(rows)
for (target,method),g in folds.groupby(['target','method'],sort=False):
    summ.append(dict(target=target,method=method,n=int(g.n.sum()),n_selected=int(g.n_selected.sum()),
        n_high=int(g.n_high.sum()),hits=int(g.hits.sum()),pooled_precision=g.hits.sum()/g.n_selected.sum(),
        pooled_recall=g.hits.sum()/g.n_high.sum(),mean_fold_auc=g.auc.mean(),
        mean_fold_recall=g.recall.mean(),mean_fold_precision=g.precision.mean()))
summary=pd.DataFrame(summ)
df.to_csv(OUT/'heldout_predictions.csv',index=False,encoding='utf-8-sig')
folds.to_csv(OUT/'fold_metrics.csv',index=False,encoding='utf-8-sig')
summary.to_csv(OUT/'target_summary.csv',index=False,encoding='utf-8-sig')
(OUT/'split_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
paths=[ROOT/'X_28dim.npy',ROOT/'composition_at_percent.csv']+[ROOT/f'y_{t}.npy' for t in ['Rm','Re','A','Ak']]
metadata={'status':'exploratory; design informed by previous results; no independent external cohort',
 'outer':'GroupKFold(5) by composition; not original process-zone LOPZO',
 'inner':'GroupKFold(4) by composition within outer training to generate out-of-fold residual targets',
 'property_model':'fixed 17-feature CatBoost 500 trees depth6 lr.05 l2=3; same predictions shared by risk models; not original tuned predictor',
 'risk_model':'Ridge alpha=10 on 17 original variables vs same 17 plus deltaTI and log1p(tau); training-only median imputation and standardization; equal total weight per training composition',
 'feature_transformation':'log1p(ti) in both risk models; add deltaTI,log1p(tau) in augmented model',
 'label':'outer abs error greater than the 80th percentile of inner-OOF training errors for that fold',
 'budget':'highest ceil(20% outer fold size) per model; deterministic sample-ID tie breaking',
 'cautions':['Original feature-only model includes composition and other process features. 19-feature model tests augmentation under chosen Ridge, not all possible classifiers.',
  'Inner property models train on fewer observations than the outer predictor, so error-distribution thresholds can be miscalibrated.',
  'Outer folds share training data; groups or data sources may be dependent; exploratory no confirmatory p values.',
  'No external dataset; group holdout addresses unseen composition, not necessarily process region extrapolation.',
  'Physical descriptors may derive from experimental or computed transformation information; acquisition and deployment assumptions matter.',
  'Small event counts. No claim of causation, equivalence, or universal superiority.'],
 'versions':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','catboost']},
 'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
 'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(OUT/'manifest.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
print(summary.round(4).to_string(index=False),flush=True)
