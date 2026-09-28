#!/usr/bin/env python3
"""Same-task raw gradient rotation across continual checkpoints; no training."""
import argparse
import gc
import json
import os
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
from run_gradient_first import (
    FlowmatchingActionHead, FlowmatchingActionHeadConfig, GROUP_ORDER,
    add_running_mean, build_probes, capture, check_backbone, checkpoint,
    compare, dump, load_head_weights, loss_gradient, parameter_group,
    prepare_cache, save_gradient, validate, write_rows,
)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--experiment-root',type=Path,default=Path('outputs/libero_cl/libero_spatial/none/step10000_bs16_adamw_bnb_8bits42'))
    parser.add_argument('--dataset-root',type=Path,default=Path('datasets/libero'))
    parser.add_argument('--num-tasks',type=int,default=6,choices=range(2,11))
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--repeats',type=int,default=4)
    parser.add_argument('--batch-size',type=int,default=4)
    args=parser.parse_args()
    if args.repeats<2: parser.error('At least two repeats required')
    args.prepare_only=False
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'progress.json').exists():
        raise FileExistsError('Use a new output directory to preserve existing gradient results')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True,warn_only=False)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    config=check_backbone(args)
    rows=build_probes(args,config)
    prepare_cache(args,config,rows)
    n=args.num_tasks; total=n*(n+1)//2*args.repeats
    metadata=dict(status='running',tasks=list(range(n)),episodes_per_task=20,probes_per_task=80,
        repeats=args.repeats,batch_size=args.batch_size,seed=args.seed,
        scope='Same-task gradients at its own task-end checkpoint and every subsequent checkpoint',
        definitions={'adjacent':'cos(g_j(C_(k-1)), g_j(C_k)), j<k',
                     'baseline':'cos(g_j(C_j), g_j(C_k)), j<k'},
        objective='masked flow-matching loss in eval mode',
        noise='per-task/probe/repeat noise and flow time; shared across checkpoints',
        precision='FP32 parameters and gradient accumulation; BF16 forward/backward kernels; FP64 metric reductions',
        gradient_storage='exact FP32 active LIBERO action-head gradients',
        experiment_root=str(args.experiment_root.resolve()),
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        no_parameter_updates=True)
    dump(args.output/'analysis_metadata.json',metadata)
    means=args.output/'mean_gradients';means.mkdir()
    temp=args.output/'temporary_gradients';temp.mkdir()
    head=FlowmatchingActionHead(FlowmatchingActionHeadConfig(**config['action_head_cfg'])).float().cuda().eval()
    params=[dict(name='action_head.'+name,shape=list(p.shape),numel=p.numel(),requires_grad=p.requires_grad,module=parameter_group('action_head.'+name)) for name,p in head.named_parameters()]
    dump(args.output/'trainable_parameters.json',params)
    load_head_weights(head,checkpoint(args.experiment_root,0))
    data=torch.load(args.output/'cache/task_0.pt',weights_only=False)
    validate(head,data,args);del data
    losses=[];norms=[];adjacent=[];baseline=[]
    for r in range(args.repeats):
        previous={};reference={}
        for c in range(n):
            load_head_weights(head,checkpoint(args.experiment_root,c))
            current={}
            for t in range(c+1):
                data=torch.load(args.output/f'cache/task_{t}.pt',weights_only=False)
                embodiment=int(data['embodiment_id'][0]); bank_grads={};bank_losses={}
                for bank in ('A','B'):
                    ids=[i for i,b in enumerate(data['banks']) if b==bank]
                    bank_losses[bank]=loss_gradient(head,data,ids,r,args.batch_size)
                    bank_grads[bank]=capture(head,embodiment)
                grad={name:(bank_grads['A'][name]+bank_grads['B'][name])*.5 for name in bank_grads['A']}
                loss=(bank_losses['A']+bank_losses['B'])*.5
                del bank_grads
                head.zero_grad(set_to_none=True)
                add_running_mean(means/f'c{c}_t{t}.safetensors',grad,r)
                for module,stats in compare(grad,grad).items():
                    norms.append(dict(repeat=r,checkpoint=c,task=t,bank='all',module=module,norm=stats['left_norm']))
                losses.append(dict(repeat=r,checkpoint=c,task=t,bank='all',loss=loss))
                path=temp/f'r{r}_c{c}_t{t}.safetensors';save_gradient(path,grad);current[t]=path
                if c==t:
                    ref=temp/f'baseline_r{r}_t{t}.safetensors'
                    os.link(path,ref);reference[t]=ref
                else:
                    for module,stats in compare(previous[t],grad).items():
                        adjacent.append(dict(repeat=r,transition=c,old_task=t,bank='all',module=module,**stats))
                    for module,stats in compare(reference[t],grad).items():
                        baseline.append(dict(repeat=r,checkpoint=c,task=t,reference_checkpoint=t,module=module,**stats))
                    previous[t].unlink()
                for name,values in [('probe_losses',losses),('gradient_norms_repeats',norms),('gradient_drift_repeats',adjacent),('gradient_baseline_repeats',baseline)]:
                    if values:write_rows(args.output/f'{name}.csv',values)
                dump(args.output/'progress.json',dict(status='running',repeat=r,checkpoint=c,task=t,completed_evaluations=len(losses),total_evaluations=total))
                print(f'DONE {len(losses)}/{total}: repeat={r} checkpoint={c} task={t} loss={loss:.6f}',flush=True)
                del data,grad;gc.collect()
            previous=current
        for path in previous.values():path.unlink()
        for path in reference.values():path.unlink()
    del head;gc.collect();torch.cuda.empty_cache()
    drift_means=[];baseline_means=[];signals=[]
    for c in range(n):
        for t in range(c+1):
            path=means/f'c{c}_t{t}.safetensors'
            for module,stats in compare(path,path).items():
                values=[x['norm'] for x in norms if x['checkpoint']==c and x['task']==t and x['module']==module]
                second=sum(v*v for v in values)/args.repeats
                se=float(np.sqrt(max(0.,(second-stats['left_norm']**2)/(args.repeats-1))))
                signals.append(dict(checkpoint=c,task=t,module=module,mean_gradient_norm=stats['left_norm'],mc_vector_standard_error=se,snr=stats['left_norm']/se if se else float('inf'),low_signal=stats['left_norm']<2*se))
            if c>t:
                for module,stats in compare(means/f'c{c-1}_t{t}.safetensors',path).items():
                    drift_means.append(dict(transition=c,old_task=t,module=module,**stats))
                for module,stats in compare(means/f'c{t}_t{t}.safetensors',path).items():
                    baseline_means.append(dict(checkpoint=c,task=t,reference_checkpoint=t,module=module,**stats))
    for name,values in [('gradient_drift',drift_means),('gradient_baseline',baseline_means),('gradient_signal',signals)]:write_rows(args.output/f'{name}.csv',values)
    dump(args.output/'progress.json',dict(status='computed',completed_evaluations=total,total_evaluations=total))
    metadata['status']='complete';dump(args.output/'analysis_metadata.json',metadata)
    print('COMPLETE',flush=True)

if __name__=='__main__':main()
