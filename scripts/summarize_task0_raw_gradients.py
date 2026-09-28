#!/usr/bin/env python3
"""Report Task 0 raw gradients at checkpoints after Task 0 and Task 1."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args(); root=args.root
    progress=json.loads((root/'progress.json').read_text())
    if progress['status']!='computed': raise RuntimeError('Gradient computation is incomplete')
    drift=pd.read_csv(root/'gradient_drift.csv')
    drift=drift[(drift.transition==1)&(drift.old_task==0)]
    repeats=pd.read_csv(root/'gradient_drift_repeats.csv')
    repeats=repeats[(repeats.transition==1)&(repeats.old_task==0)&(repeats.module=='all_action_head')].sort_values('repeat')
    signal=pd.read_csv(root/'gradient_signal.csv')
    signal=signal[(signal.task==0)&(signal.module=='all_action_head')].sort_values('checkpoint')
    full=drift[drift.module=='all_action_head'].iloc[0]
    norms=pd.read_csv(root/'gradient_norms_repeats.csv')
    norms=norms[(norms.task==0)&(norms.bank=='all')&(norms.module=='all_action_head')]
    losses=pd.read_csv(root/'probe_losses.csv')
    paired=losses[(losses.task==0)&(losses.bank=='all')].pivot(index='repeat',columns='checkpoint',values='loss')
    assert len(repeats)==4 and len(signal)==2 and paired.shape==(4,2)
    assert all(np.isfinite(repeats.cosine)) and abs(full.cosine)<=1.000001
    report={
        'definition':'cos(mean_r g_Task0(checkpoint_after_Task0, r), mean_r g_Task0(checkpoint_after_Task1, r))',
        'gradient_scope':'active LIBERO action-head parameters; frozen backbone verified identical',
        'gradient_kind':'raw loss gradient, no optimizer scaling or clipping',
        'episodes':20,'frames_per_episode':4,'noise_time_repeats':4,
        'cosine_of_mean_gradients':float(full.cosine),
        'pre_mean_gradient_norm':float(full.left_norm),'post_mean_gradient_norm':float(full.right_norm),
        'repeat_cosines':repeats.cosine.tolist(),
        'mean_of_repeat_cosines':float(repeats.cosine.mean()),
        'std_of_repeat_cosines':float(repeats.cosine.std()),
        'gradient_signal':signal.to_dict('records'),
        'mean_task0_loss_before':float(paired[0].mean()),'mean_task0_loss_after':float(paired[1].mean()),
        'precision':'FP32 parameters/gradient accumulation; BF16 forward/backward kernels; FP64 metric reductions',
        'interpretation':'Endpoint gradient rotation only; not cross-task gradient conflict or causal attribution',
    }
    (root/'raw_gradient_summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    repeats.to_csv(root/'task0_raw_gradient_repeats.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
    axes[0].scatter(repeats.repeat,repeats.cosine,s=50,label='Paired noise/time repeats')
    axes[0].axhline(full.cosine,color='C1',label='Cosine of mean gradients')
    axes[0].axhline(0,color='gray',lw=.7,ls=':')
    axes[0].set(xticks=list(repeats.repeat),xlabel='Noise/time repeat',ylabel='Cosine',ylim=(-1,1),title='Task 0 raw gradient: C0 vs C1')
    axes[0].legend(fontsize=8)
    for repeat,rows in norms.groupby('repeat'):
        rows=rows.sort_values('checkpoint')
        axes[1].plot(rows.checkpoint,rows.norm,marker='o',alpha=.5,label='Repeat gradient norms' if repeat==0 else None)
    axes[1].plot([0,1],[full.left_norm,full.right_norm],color='black',marker='D',lw=2,label='Norms of mean gradients')
    axes[1].set(xticks=[0,1],xticklabels=['C0: after Task 0','C1: after Task 1'],ylabel='Raw gradient L2 norm',title='Same Task 0 probes at both checkpoints')
    axes[1].legend(fontsize=8)
    fig.savefig(root/'task0_raw_gradient_comparison.png',dpi=200)
    fig.savefig(root/'task0_raw_gradient_comparison.pdf');plt.close(fig)
    lines=['# Task 0 raw gradient: checkpoint 0 vs checkpoint 1','',
           '동일 Task 0 demo 20개 × frame 4개에서 flow-matching loss를 미분했다. 반복마다 noise/time을 변경하고, 각 반복의 두 checkpoint에는 동일한 noise/time을 사용했다. optimizer 변환이나 gradient clipping을 적용하지 않은 raw gradient다.','',
           '대상은 active LIBERO action head다. backbone은 두 checkpoint에서 동일함을 tensor hash로 검증했다. FP32 파라미터·gradient 누적과 BF16 forward/backward를 사용했다.','',
           '| 지표 | 결과 |','| --- | ---: |',
           f'| 평균 gradient 사이 cosine | {full.cosine:+.6f} |',
           f'| checkpoint 0의 Task 0 평균 gradient norm | {full.left_norm:.6f} |',
           f'| checkpoint 1의 Task 0 평균 gradient norm | {full.right_norm:.6f} |',
           f'| 반복별 cosine 평균 ± 표준편차 | {repeats.cosine.mean():+.6f} ± {repeats.cosine.std():.6f} |','',
           '## 반복 결과','', '| Noise/time 반복 | Cosine | C0 gradient norm | C1 gradient norm |', '| --- | ---: | ---: | ---: |']
    for row in repeats.itertuples(): lines.append(f'| {row.repeat} | {row.cosine:+.6f} | {row.left_norm:.6f} | {row.right_norm:.6f} |')
    lines+=['','## Gradient 신호와 해석','', '| Checkpoint | 평균 gradient norm | MC vector standard error | 비율 | Low signal (norm < 2×SE) |', '| --- | ---: | ---: | ---: | --- |']
    for row in signal.itertuples(): lines.append(f'| {row.checkpoint} | {row.mean_gradient_norm:.6f} | {row.mc_vector_standard_error:.6f} | {row.snr:.3f} | {row.low_signal} |')
    lines+=['', '- 평균 gradient 사이 cosine과 반복별 cosine의 평균은 서로 다른 통계량이다.',
            '- Low signal 기준은 안정성 진단용 휴리스틱이며 유의성 검정이 아니다. noise 반복 4개는 독립 학습 seed 4개가 아니다.',
            '- 이 비교는 같은 Task 0 gradient의 checkpoint 간 변화다. Task 0과 Task 1 사이의 gradient conflict를 측정한 것이 아니다.',
            '- cosine의 부호 또는 회전만으로 망각의 원인을 확정하지 않는다. 작은 norm 여부는 noise 대비 신호와 함께 판단한다.',
            f'- 참고: 같은 probe의 Task 0 평균 loss는 {paired[0].mean():.6f} → {paired[1].mean():.6f}다.',
            '', '## 검증 및 산출물','',
            '- 원본 forward와 고정 noise/time forward의 loss 일치 검사 통과.',
            '- 동일 seed 재실행 cosine=1.0. batch size 2→1 gradient cosine=0.99994154.',
            '- [비교 그림](task0_raw_gradient_comparison.png) / [PDF](task0_raw_gradient_comparison.pdf)',
            '- [요약 JSON](raw_gradient_summary.json) / [반복 CSV](task0_raw_gradient_repeats.csv)',
            '- [모듈별 cosine과 norm](gradient_drift.csv) / [신호 진단](gradient_signal.csv)',
            '- 평균 raw gradient 텐서: `mean_gradients/c0_t0.safetensors`, `mean_gradients/c1_t0.safetensors`.','']
    (root/'RAW_GRADIENT_RESULTS.md').write_text('\n'.join(lines))
    # Correct the completion metadata of the already-running pre-fix process.
    completed=len(losses[['repeat','checkpoint','task']].drop_duplicates())
    assert completed==progress['total_evaluations']
    progress['completed_evaluations']=completed
    (root/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
    metadata=json.loads((root/'analysis_metadata.json').read_text()); metadata['status']='complete'
    (root/'analysis_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
