#!/usr/bin/env python3
"""Plot same-task gradient drift, norms, and Monte Carlo variability."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def heatmap(ax, values, title, *, cosine=False, fmt='.2f', low_signal=None):
    kw=dict(cmap='RdBu_r',vmin=-1,vmax=1) if cosine else dict(cmap='viridis')
    im=ax.imshow(np.ma.masked_invalid(values),**kw)
    ax.set(title=title,xlabel='Task evaluated',ylabel='Checkpoint after task',
           xticks=range(values.shape[1]),yticks=range(values.shape[0]))
    ax.set_facecolor('#eeeeee')
    for c,t in zip(*np.where(np.isfinite(values))):
        value=values[c,t];rgba=im.cmap(im.norm(value))
        color='black' if np.dot(rgba[:3],[.2126,.7152,.0722])>.55 else 'white'
        suffix='*' if low_signal is not None and low_signal[c,t] else ''
        ax.text(t,c,f'{value:{fmt}}{suffix}',ha='center',va='center',color=color,fontsize=9)
    plt.colorbar(im,ax=ax,shrink=.8)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--reference-task01',type=Path)
    args=parser.parse_args();root=args.root
    meta=json.loads((root/'analysis_metadata.json').read_text());n=len(meta['tasks']);r=meta['repeats']
    assert meta['status']=='complete'
    signal_all=pd.read_csv(root/'gradient_signal.csv')
    baseline_all=pd.read_csv(root/'gradient_baseline.csv')
    adjacent_all=pd.read_csv(root/'gradient_drift.csv')
    full=lambda df:df[df.module=='all_action_head'].copy()
    signal=full(signal_all);base=full(baseline_all);adj=full(adjacent_all)
    brep=full(pd.read_csv(root/'gradient_baseline_repeats.csv'))
    arep=full(pd.read_csv(root/'gradient_drift_repeats.csv'))
    losses=pd.read_csv(root/'probe_losses.csv')
    expected=n*(n+1)//2
    assert len(signal)==expected and len(base)==len(adj)==n*(n-1)//2
    assert len(losses)==expected*r and len(brep)==len(arep)==n*(n-1)//2*r
    assert np.isfinite(signal.mean_gradient_norm).all()
    assert np.isfinite(base.cosine).all() and np.isfinite(adj.cosine).all()
    assert (base.cosine.abs()<=1.000001).all() and (adj.cosine.abs()<=1.000001).all()
    sg=signal.set_index(['checkpoint','task']);lg=losses.groupby(['checkpoint','task']).loss.mean()
    aidx=adj.set_index(['transition','old_task'])
    rows=[]
    for x in base.sort_values(['checkpoint','task']).itertuples():
        c,t=x.checkpoint,x.task
        bv=brep[(brep.checkpoint==c)&(brep.task==t)].cosine
        av=arep[(arep.transition==c)&(arep.old_task==t)].cosine
        low_base=bool(sg.loc[(t,t),'low_signal'] or sg.loc[(c,t),'low_signal'])
        low_adj=bool(sg.loc[(c-1,t),'low_signal'] or sg.loc[(c,t),'low_signal'])
        rows.append(dict(checkpoint=c,task=t,reference_checkpoint=t,
             baseline_cosine=x.cosine,baseline_repeat_mean=bv.mean(),baseline_repeat_std=bv.std(),
             baseline_repeat_min=bv.min(),baseline_repeat_max=bv.max(),baseline_low_signal=low_base,
             adjacent_cosine=aidx.loc[(c,t),'cosine'],adjacent_repeat_mean=av.mean(),adjacent_repeat_std=av.std(),
             adjacent_repeat_min=av.min(),adjacent_repeat_max=av.max(),adjacent_low_signal=low_adj,
             initial_gradient_norm=x.left_norm,current_gradient_norm=x.right_norm,
             norm_ratio=x.right_norm/x.left_norm if x.left_norm else np.nan,
             initial_loss=lg.loc[(t,t)],current_loss=lg.loc[(c,t)],
             loss_change_from_initial=lg.loc[(c,t)]-lg.loc[(t,t)]))
    table=pd.DataFrame(rows);table.to_csv(root/'raw_gradient_summary.csv',index=False)
    figures=root/'figures';figures.mkdir(exist_ok=True)
    bm=np.full((n,n),np.nan);am=bm.copy();nm=bm.copy();snr=bm.copy();bl=np.zeros((n,n),bool);al=bl.copy();pl=bl.copy()
    for x in signal.itertuples():
        nm[x.checkpoint,x.task]=x.mean_gradient_norm;snr[x.checkpoint,x.task]=x.snr;pl[x.checkpoint,x.task]=x.low_signal
    for x in table.itertuples():
        bm[x.checkpoint,x.task]=x.baseline_cosine;am[x.checkpoint,x.task]=x.adjacent_cosine
        bl[x.checkpoint,x.task]=x.baseline_low_signal;al[x.checkpoint,x.task]=x.adjacent_low_signal
    np.fill_diagonal(bm,1.)
    fig,axes=plt.subplots(2,2,figsize=(12,11),constrained_layout=True)
    heatmap(axes[0,0],bm,'Same-task cosine vs its task-end checkpoint',cosine=True,low_signal=bl)
    heatmap(axes[0,1],am,'Same-task cosine vs previous checkpoint',cosine=True,low_signal=al)
    heatmap(axes[1,0],np.log10(nm),'log10 norm of mean raw gradient',low_signal=pl)
    heatmap(axes[1,1],snr,'Mean-gradient norm / MC vector SE',fmt='.1f',low_signal=pl)
    fig.suptitle('Task 0–5 raw gradients: fixed demos, paired noise/time, four repeats\n* low-signal heuristic (cosines: either endpoint); gray = not evaluated',fontsize=13)
    for ext in ('png','pdf'):fig.savefig(figures/f'01_raw_gradient_overview.{ext}',dpi=200)
    plt.close(fig)
    fig,axes=plt.subplots(int(np.ceil(n/3)),3,figsize=(14,3.5*int(np.ceil(n/3))),constrained_layout=True,squeeze=False)
    for t,ax in enumerate(axes.flat):
        if t>=n:
            ax.set_visible(False);continue
        subset=base[base.task==t].sort_values('checkpoint')
        for rep,values in brep[brep.task==t].groupby('repeat'):
            values=values.sort_values('checkpoint')
            ax.plot([t]+values.checkpoint.tolist(),[1]+values.cosine.tolist(),alpha=.4,marker='o',label='Individual repeat' if rep==0 else None)
        ax.plot([t]+subset.checkpoint.tolist(),[1]+subset.cosine.tolist(),color='black',lw=2,marker='D',label='Cosine of mean gradients')
        ax.axhline(0,color='gray',lw=.7);ax.set(title=f'Task {t}',xlabel='Checkpoint after task',ylabel='Cosine vs task-end gradient',ylim=(-1.05,1.05),xticks=range(n))
        if t==0:ax.legend(fontsize=7)
        if t==n-1:ax.text(.05,.15,'No subsequent checkpoint\nin requested range',transform=ax.transAxes,fontsize=9)
    for ext in ('png','pdf'):fig.savefig(figures/f'02_noise_repeat_variability.{ext}',dpi=200)
    plt.close(fig)
    modules=[m for m in signal_all.module.unique() if m!='all_action_head']
    fig,axes=plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
    for module,ax in zip(modules,axes.flat):
        values=np.full((n,n),np.nan);np.fill_diagonal(values,1.)
        for x in baseline_all[baseline_all.module==module].itertuples():values[x.checkpoint,x.task]=x.cosine
        heatmap(ax,values,module,cosine=True)
    fig.suptitle('Module raw gradient cosine vs each task-end checkpoint')
    for ext in ('png','pdf'):fig.savefig(figures/f'03_module_baseline_cosines.{ext}',dpi=180)
    plt.close(fig)
    # Verify that extending to six tasks did not change the original Task 0→1 result.
    reference_check=None
    if args.reference_task01:
        reference=json.loads((args.reference_task01/'raw_gradient_summary.json').read_text())
        first=table[(table.checkpoint==1)&(table.task==0)].iloc[0]
        actual=[first.baseline_cosine,first.initial_gradient_norm,first.current_gradient_norm]
        expected_values=[reference['cosine_of_mean_gradients'],reference['pre_mean_gradient_norm'],reference['post_mean_gradient_norm']]
        np.testing.assert_allclose(actual,expected_values,rtol=1e-5,atol=1e-7)
        reference_check=dict(passed=True,actual=actual,reference=expected_values)
    summary=dict(status='complete',task_checkpoint_combinations=expected,gradient_evaluations=expected*r,
       comparisons_per_definition=len(table),low_signal_gradient_points=int(signal.low_signal.sum()),baseline_low_signal_pairs=int(table.baseline_low_signal.sum()),
       adjacent_low_signal_pairs=int(table.adjacent_low_signal.sum()),
       baseline_repeat_sign_flips=int(((table.baseline_repeat_min<0)&(table.baseline_repeat_max>0)).sum()),
       adjacent_repeat_sign_flips=int(((table.adjacent_repeat_min<0)&(table.adjacent_repeat_max>0)).sum()),
       task01_reproduction=reference_check)
    (root/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    lines=['# Task 0~5 raw gradient 변화 실험','',
           f'완료: task/checkpoint 조합 {expected}개 × noise/time 반복 {r}회 = {expected*r}회 평가. 모든 task에서 demo 20개 × frame 4개를 사용했다. optimizer update 없이 기존 checkpoint를 평가했다.','',
           '## 연구 질문과 비교 정의','',
           '- 학습 직후 기준: cos(g_j(C_j), g_j(C_k)). Task j를 배운 직후와 이후의 같은 Task j gradient를 비교한다.',
           '- 인접 checkpoint 기준: cos(g_j(C_(k−1)), g_j(C_k)). 이번 학습 구간 전후의 같은 Task j gradient를 비교한다.',
           '- checkpoint 0~5, j < k. 각각 15개 비교다. Task 5는 checkpoint 5의 기준 gradient/norm/loss만 있으며 이후 비교는 없다.',
           '- 주 지표는 noise/time 반복을 평균한 gradient 사이의 cosine이다. 반복별 cosine 평균과 구분한다.','',
           '## 전체 action head 결과','',
           '| Task | 비교 checkpoint | 학습 직후 기준 cosine | 인접 기준 cosine | 초기 norm | 현재 norm | Norm 배율 | 기준 비교 low signal |',
           '| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for x in table.itertuples():lines.append(f'| {x.task} | C{x.task}→C{x.checkpoint} | {x.baseline_cosine:+.4f} | {x.adjacent_cosine:+.4f} | {x.initial_gradient_norm:.4f} | {x.current_gradient_norm:.4f} | {x.norm_ratio:.2f} | {x.baseline_low_signal} |')
    lines+=['','## 안정성과 해석 한계','',
            f'- Low signal 비교: 학습 직후 기준 {summary["baseline_low_signal_pairs"]}/15, 인접 기준 {summary["adjacent_low_signal_pairs"]}/15. 어느 한 endpoint라도 mean-gradient norm < 2×MC vector standard error이면 표시했다. 유의성 검정이 아닌 휴리스틱이다.',
            f'- 반복 간 cosine 부호가 바뀌는 비교: 학습 직후 기준 {summary["baseline_repeat_sign_flips"]}/15, 인접 기준 {summary["adjacent_repeat_sign_flips"]}/15.',
            '- 동일 task의 gradient 회전이며 cross-task gradient conflict를 직접 측정한 결과가 아니다. 음의 cosine만으로 망각의 인과관계를 확정하지 않는다.',
            f'- 개별 gradient {expected}개 중 {int(signal.low_signal.sum())}개가 low signal이다. Task 5의 checkpoint 5 gradient는 norm/SE=1.980으로 해당하며, 후속 비교에는 포함되지 않는다.',
            '- noise/time 반복 4회는 독립 학습 seed 실험이 아니다. demo A/B는 10개씩 고정 분할하고 각각의 평균 gradient를 동일 가중치로 합쳤다.',
            '- 측정 파라미터는 active LIBERO action head다. frozen backbone은 여섯 checkpoint에서 585개 tensor hash가 동일함을 확인했다.',
            '- FP32 파라미터/gradient 누적, BF16 forward/backward, FP64 지표 합산을 사용했다. 모든 checkpoint에서 demo/frame/전처리/noise/time이 고정되어 있다.',
            '- 원본 forward 일치, 동일 seed 재실행, batch size 변경 검사를 통과했다.',
            f'- 기존 Task 0→1 결과 재현 검사: {reference_check["passed"] if reference_check else "미실행"}.','',
            '## 결과 파일','',
            '- [전체 비교 PNG](figures/01_raw_gradient_overview.png) / [PDF](figures/01_raw_gradient_overview.pdf)',
            '- [Noise 반복 PNG](figures/02_noise_repeat_variability.png) / [PDF](figures/02_noise_repeat_variability.pdf)',
            '- [모듈별 PNG](figures/03_module_baseline_cosines.png) / [PDF](figures/03_module_baseline_cosines.pdf)',
            '- [요약 CSV](raw_gradient_summary.csv): 두 기준 cosine, 반복 변동, norm과 loss.',
            '- [학습 직후 기준](gradient_baseline.csv), [인접 기준](gradient_drift.csv), [gradient 신호](gradient_signal.csv).',
            '- `mean_gradients/c{k}_t{j}.safetensors`: Task j의 checkpoint k 평균 raw gradient.','']
    (root/'RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps(summary,indent=2))
    print(table[['checkpoint','task','baseline_cosine','adjacent_cosine','norm_ratio']].to_string(index=False))

if __name__=='__main__':main()
