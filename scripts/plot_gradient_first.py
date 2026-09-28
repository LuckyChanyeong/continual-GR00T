#!/usr/bin/env python3
"""Readable annotated maps and tables for the gradient-first experiment."""
import argparse, json, math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

GROUPS=['all_action_head','state_encoder','action_encoder','vl_projector','future_tokens','dit','action_decoder']
LABELS={'all_action_head':'Whole action head','state_encoder':'State encoder','action_encoder':'Action encoder + position',
'vl_projector':'VL normalization + attention','future_tokens':'Future tokens','dit':'DiT','action_decoder':'Action decoder'}


def signal_lookup(sig):
    return {(int(r.checkpoint),int(r.task),r.module):r for r in sig.itertuples()}


def format_value(x):
    if abs(x)<1e-12: return '0'
    if abs(x)<.001: return f'{x:.1e}'
    return f'{x:+.3f}'


def heatmap(ax,rows,key,title,limit=None,flag=None,fmt=None):
    values=np.full((5,5),np.nan); flags={}
    for row in rows.to_dict('records'):
        i=int(row['transition'])-1; j=int(row['old_task']); values[i,j]=row[key]
        flags[i,j]=bool(row.get(flag,False)) if flag else False
    valid=values[np.isfinite(values)]
    vmax=limit if limit is not None else max(float(np.max(np.abs(valid))) if len(valid) else 0.,1e-6)
    norm=TwoSlopeNorm(vmin=-vmax,vcenter=0,vmax=vmax)
    cmap=plt.get_cmap('RdBu_r').copy(); cmap.set_bad('#eeeeee')
    im=ax.imshow(np.ma.masked_invalid(values),cmap=cmap,norm=norm,aspect='equal')
    for i in range(5):
        for j in range(5):
            v=values[i,j]
            if np.isfinite(v):
                color=cmap(norm(v)); lum=.2126*color[0]+.7152*color[1]+.0722*color[2]
                label=(fmt(v) if fmt else format_value(v))+('*' if flags[i,j] else '')
                ax.text(j,i,label,ha='center',va='center',fontsize=11,fontweight='bold',color='white' if lum<.5 else '#101010')
            else: ax.text(j,i,'—',ha='center',va='center',color='#aaa',fontsize=12)
    ax.set_xticks(range(5),[f'T{j}' for j in range(5)])
    ax.set_yticks(range(5),[f'C{k-1} → C{k}' for k in range(1,6)])
    ax.set_xticks(np.arange(-.5,5,1),minor=True); ax.set_yticks(np.arange(-.5,5,1),minor=True)
    ax.grid(which='minor',color='white',linewidth=1.5); ax.tick_params(which='minor',bottom=False,left=False)
    ax.set_xlabel('Old task'); ax.set_ylabel('Checkpoint transition'); ax.set_title(title,fontsize=13,pad=12)
    cb=ax.figure.colorbar(im,ax=ax,shrink=.83,pad=.03); cb.ax.tick_params(labelsize=10)
    cb.set_label({'conflict_cosine':'Cosine (<0: conflict)', 'drift_cosine':'Cosine (<0: direction reversal)', 'loss_change':'Flow loss change (>0: worse)', 'success_drop_pp':'Percentage points (>0: worse)'}.get(key,key),fontsize=9)
    return im


def save(fig,out,name):
    fig.savefig(out/f'{name}.png',dpi=240,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,default=Path('outputs/analysis/libero_spatial/none/gradient_first_v1'))
    args=ap.parse_args(); root=args.input; out=root/'figures'; out.mkdir(exist_ok=True)
    p=pd.read_csv(root/'gradient_pairs.csv'); d=pd.read_csv(root/'gradient_drift.csv')
    pr=pd.read_csv(root/'gradient_pairs_repeats.csv'); dr=pd.read_csv(root/'gradient_drift_repeats.csv')
    losses=pd.read_csv(root/'probe_losses.csv'); signals=pd.read_csv(root/'gradient_signal.csv'); lookup=signal_lookup(signals)
    success=pd.read_csv('outputs/eval_output/libero_cl/libero_spatial/none/continual_metrics_task0_to_9/success_matrix.csv',index_col=0)
    merged=p.rename(columns={'cosine':'conflict_cosine','dot':'conflict_dot','left_norm':'old_pre_norm','right_norm':'new_pre_norm'}).merge(
        d.rename(columns={'cosine':'drift_cosine','dot':'drift_dot','left_norm':'drift_pre_norm','right_norm':'old_post_norm'}),on=['transition','old_task','module'],validate='one_to_one')
    extra=[]; phase=[]
    for row in merged.itertuples():
        k=int(row.transition); j=int(row.old_task); g=row.module
        pre=losses[(losses.checkpoint==k-1)&(losses.task==j)&(losses.bank=='all')].sort_values('repeat')
        post=losses[(losses.checkpoint==k)&(losses.task==j)&(losses.bank=='all')].sort_values('repeat')
        delta=post.loss.to_numpy()-pre.loss.to_numpy()
        rr=pr[(pr.transition==k)&(pr.old_task==j)&(pr.module==g)]
        rrall=rr[rr.bank=='all']
        dd=dr[(dr.transition==k)&(dr.old_task==j)&(dr.module==g)&(dr.bank=='all')]
        a=lookup[k-1,j,g]; b=lookup[k-1,k,g]; c=lookup[k,j,g]
        conflict_low=bool(a.low_signal or b.low_signal); drift_low=bool(a.low_signal or c.low_signal)
        negative_count=int((rrall['dot']<0).sum())
        split_negative=all(rr[rr.bank==bank]['dot'].mean()<0 for bank in ['A','B'])
        stable=(negative_count>=3 and split_negative and row.conflict_dot<0 and not conflict_low)
        sb=float(success.loc[f'checkpoint_task_{k-1}',f'eval_task_{j}']); sa=float(success.loc[f'checkpoint_task_{k}',f'eval_task_{j}'])
        extra.append(dict(pre_loss=float(pre.loss.mean()),post_loss=float(post.loss.mean()),loss_change=float(delta.mean()),
            loss_change_mc_sd=float(delta.std(ddof=1)),success_before=sb,success_after=sa,success_drop_pp=100*(sb-sa),
            conflict_mc_sd=float(rrall.cosine.std(ddof=1)),drift_mc_sd=float(dd.cosine.std(ddof=1)),
            negative_repeats=negative_count,split_negative=split_negative,stable_negative=stable,
            conflict_low_signal=conflict_low,drift_low_signal=drift_low,
            old_pre_snr=a.snr,new_pre_snr=b.snr,old_post_snr=c.snr,
            norm_ratio=row.old_post_norm/row.old_pre_norm if row.old_pre_norm>0 else np.nan))
    merged=pd.concat([merged.reset_index(drop=True),pd.DataFrame(extra)],axis=1)
    merged.to_csv(root/'combined_metrics_by_module.csv',index=False)
    main_rows=merged[merged.module=='all_action_head'].copy(); main_rows.to_csv(root/'combined_metrics.csv',index=False)
    for k in range(1,6):
        for j in range(k):
            for bank in ['all','A','B']:
                for r in range(4):
                    pre=float(losses[(losses.checkpoint==k-1)&(losses.task==j)&(losses.bank==bank)&(losses.repeat==r)].loss.iloc[0])
                    post=float(losses[(losses.checkpoint==k)&(losses.task==j)&(losses.bank==bank)&(losses.repeat==r)].loss.iloc[0])
                    phase.append(dict(transition=k,old_task=j,bank=bank,repeat=r,pre_loss=pre,post_loss=post,loss_change=post-pre))
    pd.DataFrame(phase).to_csv(root/'phase_losses.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(14,12),layout='constrained')
    heatmap(axes[0,0],main_rows,'conflict_cosine','Before training: old vs incoming gradient',flag='conflict_low_signal')
    heatmap(axes[0,1],main_rows,'drift_cosine','Same old task: before vs after gradient',flag='drift_low_signal')
    heatmap(axes[1,0],main_rows,'loss_change','Old-task flow loss: after − before')
    heatmap(axes[1,1],main_rows,'success_drop_pp','Old-task success drop (percentage points)',limit=100,fmt=lambda x:f'{x:+.1f}')
    fig.suptitle('GR00T · Task 0–5 · Whole action head\n20 demonstration episodes/task · 4 probes/episode · 4 noise/time repeats',fontsize=17)
    fig.supxlabel('Cosines use noise-averaged gradients. Each panel has its own labeled scale; no clipping.\n* Low gradient signal relative to Monte Carlo variation. Gray cells: not applicable.',fontsize=10)
    save(fig,out,'00_overview')
    for key,title,flag in [('conflict_cosine','Old/new conflict','conflict_low_signal'),('drift_cosine','Old-task gradient drift','drift_low_signal')]:
        limit=max(1e-6,float(merged[key].abs().max()))
        fig,axes=plt.subplots(2,4,figsize=(26,12),layout='constrained')
        for ax,g in zip(axes.flat,GROUPS): heatmap(ax,merged[merged.module==g],key,LABELS[g],limit=limit,flag=flag)
        axes.flat[-1].axis('off'); axes.flat[-1].text(.05,.8,'Shared symmetric color scale\nacross all modules.\n\n* Low signal: interpret cautiously.\n\nGray: not applicable.',va='top',fontsize=14)
        fig.suptitle(title+' · whole head and six parameter groups',fontsize=21)
        save(fig,out,'01_'+key+'_modules')
        for g in GROUPS:
            fig,ax=plt.subplots(figsize=(8,7),layout='constrained')
            heatmap(ax,merged[merged.module==g],key,title+' · '+LABELS[g],limit=limit,flag=flag)
            fig.supxlabel('Shared scale across modules. * Low Monte Carlo gradient signal.',fontsize=10)
            save(fig,out,'module_'+g+'_'+key)
        # Absolute reference scale also provided to avoid magnifying tiny effects.
        fig,ax=plt.subplots(figsize=(8,7),layout='constrained')
        heatmap(ax,main_rows,key,title+' · fixed cosine scale [-1, 1]',limit=1,flag=flag)
        save(fig,out,'02_'+key+'_fixed_scale')
    labels=[f'{int(x.transition)-1}→{int(x.transition)} / T{int(x.old_task)}' for x in main_rows.itertuples()]
    fig,axes=plt.subplots(2,1,figsize=(16,9),layout='constrained')
    for ax,repeat_df,key,title in [(axes[0],pr,'conflict_cosine','Old/new conflict: individual noise/time repetitions'),(axes[1],dr,'drift_cosine','Same-task drift: individual noise/time repetitions')]:
        for r in range(4):
            vals=[]
            for row in main_rows.itertuples():
                sub=repeat_df[(repeat_df.repeat==r)&(repeat_df.transition==row.transition)&(repeat_df.old_task==row.old_task)&(repeat_df.module=='all_action_head')&(repeat_df.bank=='all')]
                vals.append(float(sub.cosine.iloc[0]))
            ax.scatter(np.arange(15)+(r-1.5)*.07,vals,s=30,alpha=.8,label=f'Noise repeat {r+1}')
        ax.scatter(range(15),main_rows[key],marker='D',color='black',s=24,label='Cosine of mean gradient',zorder=4)
        ax.axhline(0,color='gray',ls='--'); ax.set_xticks(range(15),labels,rotation=40,ha='right'); ax.set_ylabel('Cosine'); ax.set_title(title); ax.grid(axis='y',alpha=.2)
    axes[0].legend(ncols=5,fontsize=9)
    save(fig,out,'03_noise_repeat_variability')
    # Absolute values complement signed differences and avoid hiding near-optimum norms.
    from matplotlib.colors import Normalize, LogNorm
    fig,axes=plt.subplots(1,2,figsize=(15,7),layout='constrained')
    pooled_losses=losses[losses.bank=='all'].groupby(['checkpoint','task']).loss.mean()
    pooled_norms=signals[signals.module=='all_action_head'].set_index(['checkpoint','task']).mean_gradient_norm
    for ax,series,title,logscale in [(axes[0],pooled_losses,'Flow-matching loss (mean over noise)',False),(axes[1],pooled_norms,'Norm of mean gradient (log color scale)',True)]:
        matrix=np.full((6,6),np.nan)
        for (c,t),v in series.items(): matrix[int(c),int(t)]=v
        valid=matrix[np.isfinite(matrix)]
        norm=LogNorm(vmin=float(valid[valid>0].min()),vmax=float(valid.max())) if logscale else Normalize(vmin=0,vmax=float(valid.max()))
        cmap=plt.get_cmap('viridis').copy(); cmap.set_bad('#eee')
        im=ax.imshow(np.ma.masked_invalid(matrix),norm=norm,cmap=cmap)
        for c in range(6):
            for t in range(6):
                v=matrix[c,t]
                if np.isfinite(v):
                    rgb=cmap(norm(v)); lum=.2126*rgb[0]+.7152*rgb[1]+.0722*rgb[2]
                    ax.text(t,c,f'{v:.3g}',ha='center',va='center',fontsize=11,fontweight='bold',color='white' if lum<.5 else 'black')
                else: ax.text(t,c,'—',ha='center',va='center',color='#aaa')
        ax.set_xticks(range(6),[f'T{t}' for t in range(6)]); ax.set_yticks(range(6),[f'C{c}' for c in range(6)])
        ax.set_xlabel('Probe task'); ax.set_ylabel('Checkpoint after learning task c'); ax.set_title(title)
        cb=fig.colorbar(im,ax=ax,shrink=.82)
        if logscale:
            ticks=np.geomspace(float(valid[valid>0].min()),float(valid.max()),6)
            cb.set_ticks(ticks); cb.set_ticklabels([f'{x:.3g}' for x in ticks])
            cb.set_label('L2 norm (logarithmic spacing)')
        else:
            cb.set_label('Mean masked flow loss')
    fig.supxlabel('Lower triangle: learned tasks. One cell above each diagonal: next incoming task. Gray: not evaluated.',fontsize=11)
    save(fig,out,'04_absolute_loss_and_gradient_norm')
    corr=main_rows[['conflict_cosine','drift_cosine','loss_change','success_drop_pp']].corr()
    corr.to_csv(root/'descriptive_correlations.csv')
    nstable=int(main_rows.stable_negative.sum()); nlow=int(main_rows.conflict_low_signal.sum())
    text=['# Gradient-first experiment results','',
        'Completed: task 0–5, 26 checkpoint/task evaluations × 4 noise repeats; 20 demonstration episodes and 80 probes per task.',
        'No training updates or new rollouts were performed. Existing 30-episode/task rollout results are behavioral context.',
        '',f'- Stable negative full-head conflict: {nstable}/15 cells (exploratory rule; not a significance test).',
        f'- Low-signal full-head conflict: {nlow}/15 cells (mean gradient norm < 2 × MC vector standard error).',
        f'- Positive old loss change: {int((main_rows.loss_change>0).sum())}/15 cells.',
        f'- Positive rollout success drop: {int((main_rows.success_drop_pp>0).sum())}/15 cells.',
        '', '## Per-transition results','',
        '| New task | Old task | Conflict cosine | Drift cosine | Loss change | Success drop (pp) | Stable negative |',
        '|---|---|---|---|---|---|---|']
    for x in main_rows.itertuples(): text.append(f'| {x.transition} | {x.old_task} | {x.conflict_cosine:+.4f} | {x.drift_cosine:+.4f} | {x.loss_change:+.4f} | {x.success_drop_pp:+.1f} | {x.stable_negative} |')
    text+=['','## Interpretation limits','',
        '- Conflict compares different tasks at the same pre-training checkpoint; drift compares the same old task across checkpoints.',
        '- Phase-end observations do not establish what happened at intermediate optimizer steps or prove causality.',
        '- Raw gradients do not equal AdamW updates. Single-step logging/intervention remain follow-up work.',
        '- Noise repeats are not independent training seeds. Episode A/B reproducibility is a diagnostic, not a full episode-bootstrap confidence interval.',
        '- Demonstrations may overlap historical training. Split B is not a held-out training test set.',
        '- Cells share tasks/checkpoints; correlations are descriptive. Zero baseline success causes a floor effect.',
        '- Cosine of the noise-averaged gradient can differ markedly from the average of per-repeat cosines. Both are exported.',
        '- Modules share one loss; their statistics are not independent causal loss contributions.',
        '- Overview panels use independently labeled symmetric scales; fixed [-1,1] reference maps are also included.',
        '', 'Figures: [overview](figures/00_overview.png), [noise variability](figures/03_noise_repeat_variability.png).']
    (root/'RESULTS.md').write_text('\n'.join(text)+'\n')
    meta=json.loads((root/'analysis_metadata.json').read_text()); meta['status']='complete'; meta['completed_checkpoint_task_evaluations']=104
    (root/'analysis_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(main_rows[['transition','old_task','conflict_cosine','drift_cosine','loss_change','success_drop_pp','stable_negative']].to_string(index=False))
    print('Figures:',out)

if __name__=='__main__': main()
