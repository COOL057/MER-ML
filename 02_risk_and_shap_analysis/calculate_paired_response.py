import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent.parent / '01_model_training_and_evaluation')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records, feature_records, checks, inputs = [], [], [], []
    for target in ['Rm', 'Re', 'A', 'Ak']:
        rbase = args.root / 'unified_results_rcv3'
        lbase = args.root / 'unified_results_clean'
        paths = [rbase / 'residuals' / f'{target}_cv_predictions.csv',
                 lbase / 'residuals' / f'{target}_lopzo_predictions.csv']
        rcv, lopzo = [pd.read_csv(p) for p in paths]
        for p in paths:
            inputs.append({'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
        assert set(rcv.scenario) == {'rcv'} and set(lopzo.scenario) == {'lopzo'}
        assert not lopzo.global_idx.duplicated().any()
        rcv['repeat'] = rcv.outer_fold.str.extract(r'repeat_(\d+)_')[0].astype(int)
        assert set(rcv.repeat) == {0, 1, 2}
        cols = [c for c in lopzo if c.startswith('shap_') and c != 'shap_bias']
        assert set(cols) == {c for c in rcv if c.startswith('shap_') and c != 'shap_bias'}
        for frame in [rcv, lopzo]:
            assert np.isfinite(frame[cols + ['shap_bias', 'y_pred', 'y_true']].to_numpy()).all()
            assert np.allclose(frame[cols].sum(axis=1) + frame.shap_bias, frame.y_pred, atol=1e-8, rtol=0)
        cache = {}
        def fold_info(base, scenario, fold):
            key = (str(base), scenario, fold)
            if key not in cache:
                directory = base / 'folds' / target / scenario / fold
                audit = json.loads((directory / 'split_audit.json').read_text(encoding='utf-8'))
                selected = json.loads((directory / 'outer_features.json').read_text(encoding='utf-8'))['selected_features']
                assert len(selected) == 15 and len(set(selected)) == 15
                cache[key] = (audit, set(selected))
            return cache[key]
        for repeat, block in rcv.groupby('repeat'):
            assert not block.global_idx.duplicated().any()
            assert set(block.global_idx) == set(lopzo.global_idx)
            joined = lopzo.merge(block, on='global_idx', suffixes=('_l', '_r'), validate='one_to_one')
            assert np.allclose(joined.y_true_l, joined.y_true_r, atol=1e-10, rtol=0)
            assert (joined.zone_id_l == joined.zone_id_r).all()
            assert (joined.comp_id_l == joined.comp_id_r).all()
            for _, row in joined.iterrows():
                idx = int(row.global_idx)
                la, ls = fold_info(lbase, 'lopzo', row.outer_fold_l)
                ra, rs = fold_info(rbase, 'rcv', row.outer_fold_r)
                for audit in [la, ra]:
                    assert idx in audit['outer_test_indices'] and idx not in audit['outer_train_indices']
                    assert not set(audit['outer_train_indices']) & set(audit['outer_test_indices'])
                assert set(la['outer_test_indices']) == set(lopzo.loc[lopzo.zone_id == row.zone_id_l, 'global_idx'])
                lv = np.array([row[c + '_l'] for c in cols], dtype=float)
                rv = np.array([row[c + '_r'] for c in cols], dtype=float)
                for j, c in enumerate(cols):
                    if c[5:] not in ls:
                        assert abs(lv[j]) < 1e-10
                    if c[5:] not in rs:
                        assert abs(rv[j]) < 1e-10
                delta = lv - rv
                dp = float(row.y_pred_l - row.y_pred_r)
                db = float(row.shap_bias_l - row.shap_bias_r)
                decomposition_error = abs(dp - db - delta.sum())
                assert decomposition_error < 1e-8
                both = (np.abs(lv) > 1e-8) & (np.abs(rv) > 1e-8)
                flips = both & (lv * rv < 0)
                nl, nr = np.linalg.norm(lv), np.linalg.norm(rv)
                records.append(dict(target=target, zone=int(row.zone_id_l), repeat=int(repeat),
                    global_idx=idx, comp_id=row.comp_id_l, y_true=row.y_true_l,
                    pred_lopzo=row.y_pred_l, pred_rcv=row.y_pred_r,
                    prediction_delta=dp, abs_prediction_delta=abs(dp),
                    ae_lopzo=abs(row.y_true_l-row.y_pred_l), ae_rcv=abs(row.y_true_r-row.y_pred_r),
                    error_increment=abs(row.y_true_l-row.y_pred_l)-abs(row.y_true_r-row.y_pred_r),
                    bias_delta=db, abs_bias_delta=abs(db), shap_delta_sum=float(delta.sum()),
                    shap_delta_l1=float(np.abs(delta).sum()), shap_delta_l2=float(np.linalg.norm(delta)),
                    signed_dcos=float(1 - lv.dot(rv)/(nl*nr)) if nl*nr > 1e-12 else np.nan,
                    mr=nl/nr if nr > 1e-12 else np.nan,
                    comparable_feature_count=int(both.sum()), flip_count=int(flips.sum()),
                    selection_changes=len(ls ^ rs), decomposition_error=decomposition_error))
                for j, c in enumerate(cols):
                    feature_records.append(dict(target=target,zone=int(row.zone_id_l),repeat=int(repeat),
                        global_idx=idx,feature=c[5:],shap_lopzo=lv[j],shap_rcv=rv[j],
                        shap_delta=delta[j],abs_shap_delta=abs(delta[j]),
                        selected_lopzo=c[5:] in ls,selected_rcv=c[5:] in rs,
                        comparable=bool(both[j]),sign_flip=bool(flips[j])))
            checks.append(dict(target=target,repeat=int(repeat),n_samples=len(joined),checks_passed=True))
    paired = pd.DataFrame(records)
    features = pd.DataFrame(feature_records)
    paired.to_csv(args.output / 'paired_samples.csv',index=False,encoding='utf-8-sig')
    features.to_csv(args.output / 'paired_feature_changes.csv',index=False,encoding='utf-8-sig')
    aggregate = dict(n=('global_idx','nunique'),mae_lopzo=('ae_lopzo','mean'),mae_rcv=('ae_rcv','mean'),
        error_increment=('error_increment','mean'),prediction_change=('abs_prediction_delta','mean'),
        signed_prediction_change=('prediction_delta','mean'),shap_change_l1=('shap_delta_l1','mean'),
        shap_change_l2=('shap_delta_l2','mean'),signed_dcos=('signed_dcos','mean'),mr=('mr','mean'),
        bias_delta=('bias_delta','mean'),abs_bias_delta=('abs_bias_delta','mean'),
        comparable_features=('comparable_feature_count','sum'),flips=('flip_count','sum'),
        selection_changes=('selection_changes','mean'))
    per_repeat = paired.groupby(['target','zone','repeat']).agg(**aggregate).reset_index()
    per_repeat['flip_fraction'] = per_repeat.flips / per_repeat.comparable_features
    per_repeat.to_csv(args.output / 'zone_by_repeat.csv',index=False,encoding='utf-8-sig')
    zones = paired.groupby(['target','zone']).agg(**aggregate).reset_index()
    zones['flip_fraction'] = zones.flips / zones.comparable_features
    bounds = per_repeat.groupby(['target','zone']).agg(
        error_increment_min=('error_increment','min'),error_increment_max=('error_increment','max'),
        prediction_change_min=('prediction_change','min'),prediction_change_max=('prediction_change','max')).reset_index()
    zones = zones.merge(bounds,on=['target','zone'],validate='one_to_one')
    zones.to_csv(args.output / 'zone_summary.csv',index=False,encoding='utf-8-sig')
    fg = features.groupby(['target','zone','feature']).agg(
        mean_abs_delta=('abs_shap_delta','mean'),mean_signed_delta=('shap_delta','mean'),
        selected_lopzo_fraction=('selected_lopzo','mean'),selected_rcv_fraction=('selected_rcv','mean'),
        comparable=('comparable','sum'),flips=('sign_flip','sum')).reset_index()
    fg['flip_fraction'] = fg.flips / fg.comparable.replace(0,np.nan)
    fg.to_csv(args.output / 'zone_feature_summary.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(checks).to_csv(args.output / 'verification.csv',index=False)
    (args.output / 'manifest.json').write_text(json.dumps(dict(method='paired LOPZO versus three out-of-fold RCV repeats',
        input_files=inputs,checks=checks,max_decomposition_error=float(paired.decomposition_error.max()),
        shap_features=cols,repeat_policy='compare each repeat before aggregation; no ensemble MAE',
        caveats=['Different training size, selected features, tuning and SHAP baseline: whole pipeline contrast, not a controlled zone-only causal effect.',
                 'SHAP differences are descriptive; not a material mechanism or an error attribution.',
                 'Three repeats are not independent samples; no significance tests.',
                 'Zone 6: six samples from one composition.']),ensure_ascii=False,indent=2),encoding='utf-8')
    text = ['# 同样本 LOPZO—RCV 配对分析','',
        '## 主要发现','',
        'Rm、Re的平均误差增量最高三区均为Z0/Z3/Z6，Ak为Z6/Z7/Z8；这些三区集合在三个RCV重复中均保持一致。A的Z8/Z7平均误差增量较大，但Z6仅增加0.042个百分点，且不同重复中增量有正有负。A的Z7虽三个重复均恶化，却不是每次都排前三。',
        'Rm的SHAP贡献变化L2最高三区为Z6/Z3/Z0；Re为Z6/Z3/Z2，Z0不是最高三区。带符号Dcos方面：Rm/Re均以Z6最突出，Ak最高三区为Z8/Z7/Z6；A最高三区为Z8/Z2/Z7，Z6不突出。不同响应指标没有完全一致地识别全部风险区。',
        'τ在Rm/Re的Z6、Re的Z3以及Ak的Z6/Z8中均是贡献变化最大的特征之一。但SHAP基准、特征入选、调参和训练数据也同时变化，不能据此断言τ的材料物理机制发生反转或单独导致误差。',
        '可支持的论文结论：相较随机留出预测，区域整体留出产生了具有区域依赖性的误差增量，并在部分目标和区域伴随明显的特征归因重分配。不能支持所有指定风险区均呈现一致的响应改变。',
        'Z6为单成分6样本，仍须保留这一范围限制。此为对现有结果的探索性补充；此前训练区参考结果保留，不以当前方法替换、隐去不符合结果。','',
        '## 计算定义','',
        '对每个样本分别比较LOPZO预测与三个RCV留出预测，然后按区汇总。RCV样本自身不参与训练。',
        '预测变化=平均|LOPZO预测−RCV预测|；误差增量=LOPZO MAE−RCV MAE。RCV MAE先分别计入三个重复的绝对误差，未先平均预测。',
        '带符号SHAP按完整特征名称对齐，未入选特征贡献为零；基准值差异另列，满足预测差=基准值差+SHAP贡献差之和。',
        '符号翻转率只在两模型贡献均非零时统计。Dcos是逐样本带符号向量距离的平均，与此前平均绝对SHAP的Dcos含义不同，不能混称同一指标。',
        '所有36个目标×区域同时报告；指标均为探索性描述，不以能否突出指定区域来选择指标。','',
        '## 全部区域结果','',
        '|目标|区|n|LOPZO MAE|RCV MAE|误差增量（三重复范围）|预测变化|SHAP变化L2|带符号Dcos|翻转率|平均基准值差|',
        '|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|']
    for _,r in zones.iterrows():
        text.append(f'|{r.target}|{r.zone}|{r.n}|{r.mae_lopzo:.3f}|{r.mae_rcv:.3f}|{r.error_increment:.3f} ({r.error_increment_min:.3f}—{r.error_increment_max:.3f})|{r.prediction_change:.3f}|{r.shap_change_l2:.3f}|{r.signed_dcos:.3f}|{r.flip_fraction:.1%}|{r.bias_delta:.3f}|')
    text += ['', '## 排名与预先关注区域','']
    groups=[]
    for target,block in zones.groupby('target',sort=False):
        risk={0,3,6} if target in ['Rm','Re'] else {6,7,8}
        text.append(f'### {target}（关注区：{sorted(risk)}）')
        for metric in ['error_increment','prediction_change','shap_change_l2','signed_dcos','flip_fraction']:
            ranked=block.sort_values(metric,ascending=False)
            text.append(f'- {metric}最高三区：'+', '.join(f'Z{int(r.zone)}={r[metric]:.3f}' for _,r in ranked.head(3).iterrows()))
            groups.append(dict(target=target,metric=metric,risk_mean=block.loc[block.zone.isin(risk),metric].mean(),
                other_mean=block.loc[~block.zone.isin(risk),metric].mean()))
        text.append('贡献变化最大的特征：')
        for z in sorted(risk):
            top=fg[(fg.target==target)&(fg.zone==z)].nlargest(3,'mean_abs_delta')
            text.append(f'- Z{z}: '+', '.join(f'{r.feature}={r.mean_abs_delta:.3f}' for _,r in top.iterrows()))
        text.append('')
    pd.DataFrame(groups).to_csv(args.output / 'predefined_group_summary.csv',index=False,encoding='utf-8-sig')
    text += ['## 验证与解释边界','',
        f'四目标、三个重复均已逐样本匹配标签/区/成分，检查样本位于相应外层测试集、训练测试不重叠，验证SHAP加和及未入选特征为零。最大预测差分解误差：{paired.decomposition_error.max():.3g}。',
        '对比的是不同训练集规模、特征筛选、调参结果和SHAP背景下的整个模型开发流程，不能把全部差异独立归因于是否见过该工艺区。',
        '贡献变化大不等于预测失败；带符号变化也不是对损失的解释。贡献翻转受背景基准变化影响，不能直接认定物理规律反转。',
        'Z6只有6个样本且来自一个成分。三次RCV重复不是新增独立样本，不作伪重复显著性检验。',
        '本次只读取已有预测与SHAP，不重新训练或修改原文件。']
    (args.output / '配对响应分析汇总.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(zones[['target','zone','mae_lopzo','mae_rcv','error_increment','prediction_change','shap_change_l2','signed_dcos','flip_fraction']].to_string(index=False))
    print('CHECKS PASSED',len(paired),'pairs; max decomposition error',paired.decomposition_error.max())


if __name__ == '__main__':
    main()
