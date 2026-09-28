#!/usr/bin/env python3
"""Fixed-demo gradient study. Does not update model parameters or run training.

Uses explicit per-probe noise/time, verifies the full frozen backbone, caches
backbone outputs, and stores exact FP32 gradients temporarily on disk. Designed
for the 48 GiB host-memory limit and a 24 GiB GPU.
"""
from __future__ import annotations
import argparse, csv, gc, hashlib, json, math, os, random, subprocess, time
from pathlib import Path
from contextlib import ExitStack
import numpy as np
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from transformers.feature_extraction_utils import BatchFeature
from analyze_demo_gradients import (load_model, make_dataset, materialize_samples,
    parameter_group, GROUP_ORDER, write_rows)
from build_demo_probe_bank import LIBERO_SPATIAL_TASKS
from gr00t.model.transforms import DefaultDataCollator
from gr00t.model.action_head.flow_matching_action_head import (
    FlowmatchingActionHead, FlowmatchingActionHeadConfig, CategorySpecificLinear)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def sha_tensor(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy()).hexdigest()


def checkpoint(root,c): return root/f'task_{c}'/'checkpoint-10000'


def weight_map(cp): return json.loads((cp/'model.safetensors.index.json').read_text())['weight_map']


def check_backbone(args):
    out=args.output/'backbone_verification.json'
    hashes={}; config0=None
    for c in range(args.num_tasks):
        cp=checkpoint(args.experiment_root,c); mapping=weight_map(cp)
        config=json.loads((cp/'config.json').read_text())
        if config0 is None: config0=config
        assert config['backbone_cfg']==config0['backbone_cfg']
        assert config['action_head_cfg']==config0['action_head_cfg']
        current={}
        for shard in sorted(set(mapping.values())):
            with safe_open(cp/shard, framework='pt',device='cpu') as h:
                for name in sorted(n for n in h.keys() if n.startswith('backbone.')):
                    current[name]=sha_tensor(h.get_tensor(name))
        if c: assert current==hashes, f'Backbone changed at checkpoint {c}; cannot reuse features'
        else: hashes=current
        print(f'BACKBONE {c}: {len(current)} tensors verified',flush=True)
    dump(out,{'all_six_identical':args.num_tasks==6,'all_checked_identical':True,'num_tasks':args.num_tasks,'tensor_count':len(hashes),'checkpoint0_tensor_sha256':hashes})
    return config0


def build_probes(args,config):
    rows=[]; H=config['action_horizon']
    for t,name in enumerate(LIBERO_SPATIAL_TASKS[:args.num_tasks]):
        folder=(args.dataset_root/name).resolve()
        eps=[json.loads(l) for l in (folder/'meta/episodes.jsonl').read_text().splitlines() if l.strip()]
        rng=np.random.default_rng(args.seed+t)
        selected=np.sort(rng.choice(len(eps),20,replace=False))
        # Random episode split, not the first/last episode IDs.
        a_ranks=set(rng.permutation(20)[:10].tolist())
        for rank,pos in enumerate(selected):
            e=eps[pos]; limit=int(e['length'])-H
            assert limit>=3, 'Episode too short for four distinct full action chunks'
            for f,pct in enumerate([.2,.4,.6,.8]):
                rows.append(dict(task_id=t,task_name=name,dataset_path=str(folder),
                    episode_id=int(e['episode_index']),episode_length=int(e['length']),
                    progress=pct,base_index=round(pct*limit),probe_id=4*rank+f,
                    bank='A' if rank in a_ranks else 'B'))
    write_rows(args.output/'probe_manifest.csv',rows)
    return rows


def prepare_cache(args,config,rows):
    cache=args.output/'cache'; cache.mkdir(exist_ok=True)
    model=load_model(checkpoint(args.experiment_root,0),'cuda')
    assert all(not p.requires_grad for p in model.backbone.parameters())
    collator=DefaultDataCollator()
    for t in range(args.num_tasks):
        selected=[r for r in rows if r['task_id']==t]
        path=cache/f'task_{t}.pt'
        dataset=make_dataset(selected[0]); samples=materialize_samples(dataset,selected)
        del dataset
        chunks=[]
        with torch.no_grad():
            for start in range(0,80,args.batch_size):
                b=collator(samples[start:start+args.batch_size])
                bi,ai=model.prepare_input(b)
                with torch.autocast('cuda',dtype=torch.bfloat16): bo=model.backbone(bi)
                chunks.append({k:bo[k].detach().cpu() for k in ['backbone_features','backbone_attention_mask']})
                chunks[-1].update({k:ai[k].detach().cpu() for k in ['state','action','action_mask','embodiment_id']})
                del b,bi,ai,bo
        data={k:torch.cat([ch[k] for ch in chunks]) for k in chunks[0]}
        assert len(torch.unique(data['embodiment_id']))==1
        assert bool(torch.all(data['action_mask'].flatten(1).sum(1)==data['action_mask'][0].sum()))
        noise=[]; ts=[]
        for r in range(args.repeats):
            nr=[]; tr=[]
            for row in selected:
                seed=args.seed+10_000_019*t+100_003*r+101*row['probe_id']
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed)
                    nr.append(torch.randn(data['action'][0].shape,dtype=data['action'].dtype))
                    beta=torch.distributions.Beta(config['action_head_cfg']['noise_beta_alpha'],config['action_head_cfg']['noise_beta_beta']).sample()
                    tr.append((config['action_head_cfg']['noise_s']-beta)/config['action_head_cfg']['noise_s'])
            noise.append(torch.stack(nr)); ts.append(torch.stack(tr).to(data['action'].dtype))
        data['noise']=torch.stack(noise); data['flow_time']=torch.stack(ts)
        data['banks']=[x['bank'] for x in selected]
        torch.save(data,path)
        print(f'CACHE task {t}: features {tuple(data["backbone_features"].shape)}',flush=True)
        del data,samples,chunks
    del model; gc.collect(); torch.cuda.empty_cache()


def load_head_weights(head,cp):
    mapping=weight_map(cp); state={}
    for shard in sorted(set(mapping.values())):
        with safe_open(cp/shard,framework='pt',device='cpu') as f:
            for n in sorted(k for k in f.keys() if k.startswith('action_head.')):
                state[n.removeprefix('action_head.')]=f.get_tensor(n).float()
    head.load_state_dict(state,strict=True)
    del state; gc.collect()


def fixed_forward(head,back,ai,noise,t):
    """Repository head.forward with the identical loss and explicit noise/time."""
    assert head.config.expand_batch is None
    bo=head.process_backbone_output(BatchFeature(data=dict(back)))
    state=head.state_encoder(ai['state'],ai['embodiment_id'])
    actions=ai['action']; time=t[:,None,None]
    noisy=(1-time)*noise+time*actions
    bucket=(t*head.num_timestep_buckets).long()
    action=head.action_encoder(noisy,bucket,ai['embodiment_id'])
    if head.config.add_pos_embed:
        action=action+head.position_embedding(torch.arange(action.shape[1],device=action.device))[None]
    future=head.future_tokens.weight[None].expand(len(actions),-1,-1)
    hidden=head.model(hidden_states=torch.cat((state,future,action),dim=1),
        encoder_hidden_states=bo['backbone_features'],encoder_attention_mask=bo['backbone_attention_mask'],
        timestep=bucket,return_all_hidden_states=False)
    pred=head.action_decoder(hidden,ai['embodiment_id'])[:,-actions.shape[1]:]
    return (F.mse_loss(pred,actions-noise,reduction='none')*ai['action_mask']).sum()/ai['action_mask'].sum()


def inputs(data,idx,repeat):
    back={k:data[k][idx].cuda() for k in ['backbone_features','backbone_attention_mask']}
    ai={k:data[k][idx].cuda() for k in ['state','action','action_mask','embodiment_id']}
    return back,ai,data['noise'][repeat,idx].cuda(),data['flow_time'][repeat,idx].cuda()


def loss_gradient(head,data,idx,repeat,bs):
    head.zero_grad(set_to_none=True); total=0.
    for start in range(0,len(idx),bs):
        ids=idx[start:start+bs]; b,a,n,t=inputs(data,ids,repeat)
        with torch.autocast('cuda',dtype=torch.bfloat16): loss=fixed_forward(head,b,a,n,t)
        w=len(ids)/len(idx); total+=loss.detach().float().item()*w
        (loss*w).backward()
    return total


def capture(head,emb):
    result={}; active={}
    for module_name,module in head.named_modules():
        if isinstance(module,CategorySpecificLinear):
            active[module_name+'.W']=emb; active[module_name+'.b']=emb
    for name,p in head.named_parameters():
        assert p.dtype==torch.float32
        if p.grad is None: raise RuntimeError(f'No gradient: {name}')
        value=p.grad.detach()
        if name in active: value=value[emb]
        result['action_head.'+name]=value.to(device='cpu',dtype=torch.float32).contiguous().clone()
    return result


def compare(left,right):
    # Dictionaries stay resident for current/incoming tasks; old pooled vectors
    # are streamed from disk. Computations are identical for both representations.
    dots={g:0. for g in GROUP_ORDER}; a={g:0. for g in GROUP_ORDER}; b={g:0. for g in GROUP_ORDER}
    same=left is right
    with ExitStack() as stack:
        lf=left if isinstance(left,dict) else stack.enter_context(safe_open(left,framework='pt',device='cpu'))
        rf=right if isinstance(right,dict) else stack.enter_context(safe_open(right,framework='pt',device='cpu'))
        assert set(lf.keys())==set(rf.keys())
        for name in sorted(lf.keys()):
            x=lf[name] if isinstance(lf,dict) else lf.get_tensor(name)
            y=rf[name] if isinstance(rf,dict) else rf.get_tensor(name)
            if same:
                d=xx=yy=float(torch.sum(x*x,dtype=torch.float64))
            else:
                d=float(torch.sum(x*y,dtype=torch.float64)); xx=float(torch.sum(x*x,dtype=torch.float64)); yy=float(torch.sum(y*y,dtype=torch.float64))
            for group in [parameter_group(name),'all_action_head']:
                dots[group]+=d; a[group]+=xx; b[group]+=yy
    return {g:dict(dot=dots[g],left_norm=math.sqrt(a[g]),right_norm=math.sqrt(b[g]),
        cosine=dots[g]/math.sqrt(a[g]*b[g]) if a[g]*b[g]>0 else float('nan')) for g in GROUP_ORDER}


def save_gradient(path,grad):
    path.parent.mkdir(exist_ok=True,parents=True)
    save_file(grad,str(path))


def add_running_mean(path,gradient,r):
    if r==0: save_gradient(path,gradient); return
    with safe_open(path,framework='pt',device='cpu') as f:
        updated={n:(f.get_tensor(n)*r+v)/(r+1) for n,v in gradient.items()}
    tmp=path.with_suffix('.tmp'); save_gradient(tmp,updated); tmp.replace(path)
    del updated


def validate(head,data,args):
    ids=[0,1]; back,ai,noise,t=inputs(data,ids,0)
    # Match repository RNG order to verify the explicit forward implementation.
    torch.manual_seed(123); torch.cuda.manual_seed_all(123)
    n=torch.randn(ai['action'].shape,device='cuda',dtype=ai['action'].dtype)
    tm=head.sample_time(2,'cuda',ai['action'].dtype)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16): fixed=fixed_forward(head,back,ai,n,tm).item()
    torch.manual_seed(123); torch.cuda.manual_seed_all(123)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16): original=head(BatchFeature(data=dict(back)),BatchFeature(data=dict(ai)))['loss'].item()
    assert math.isclose(original,fixed,rel_tol=1e-6,abs_tol=1e-7),(original,fixed)
    emb=int(data['embodiment_id'][0]); checks={}
    for name,bs in [('batch2',2),('repeat_batch2',2),('batch1',1)]:
        loss=loss_gradient(head,data,ids,0,bs)
        if name=='batch2':
            for mod in head.modules():
                if isinstance(mod,CategorySpecificLinear):
                    for param in [mod.W,mod.b]:
                        g=param.grad
                        assert not bool(torch.count_nonzero(g[:emb])) and not bool(torch.count_nonzero(g[emb+1:]))
        path=args.output/f'validation_{name}.safetensors'; save_gradient(path,capture(head,emb))
        checks[name]={'loss':loss,'path':path}
    for name in ['repeat_batch2','batch1']:
        metrics=compare(checks['batch2']['path'],checks[name]['path'])['all_action_head']
        checks[name].update(metrics)
    assert abs(checks['batch2']['loss']-checks['repeat_batch2']['loss'])<1e-7
    assert checks['repeat_batch2']['cosine']>.999999
    assert checks['batch1']['cosine']>.995,checks
    assert math.isclose(checks['batch1']['loss'],checks['batch2']['loss'],rel_tol=.02,abs_tol=1e-5)
    for x in checks.values(): x.pop('path').unlink()
    result={'original_forward_loss':original,'explicit_forward_loss':fixed,'inactive_embodiment_gradients_exactly_zero':True,'batch_checks':checks,
        'note':'BF16 kernels may differ slightly with batch size; exact noise/time per sample is unchanged.'}
    dump(args.output/'validation.json',result)
    print('VALIDATION',json.dumps(result),flush=True)
    head.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()


def run(args,config):
    total_evaluations=(args.num_tasks*(args.num_tasks+1)//2+args.num_tasks-1)*args.repeats
    temp=args.output/'temporary_gradients'; means=args.output/'mean_gradients'
    temp.mkdir(exist_ok=True); means.mkdir(exist_ok=True)
    head=FlowmatchingActionHead(FlowmatchingActionHeadConfig(**config['action_head_cfg'])).float().cuda().eval()
    params=[{'name':'action_head.'+n,'shape':list(p.shape),'numel':p.numel(),'requires_grad':p.requires_grad,'module':parameter_group('action_head.'+n)} for n,p in head.named_parameters()]
    dump(args.output/'trainable_parameters.json',params)
    group_counts={g:sum(x['numel'] for x in params if g=='all_action_head' or x['module']==g) for g in GROUP_ORDER}
    load_head_weights(head,checkpoint(args.experiment_root,0))
    data=torch.load(args.output/'cache/task_0.pt',weights_only=False)
    validate(head,data,args); del data
    if args.prepare_only: return
    pair_rows=[]; drift_rows=[]; loss_rows=[]; norm_rows=[]
    for r in range(args.repeats):
        previous={}
        for c in range(args.num_tasks):
            begin=time.time(); load_head_weights(head,checkpoint(args.experiment_root,c))
            order=([c+1] if c<args.num_tasks-1 else [])+list(range(c+1))
            current={}; incoming={}
            for t in order:
                data=torch.load(args.output/f'cache/task_{t}.pt',weights_only=False)
                losses={}; gradients={}; emb=int(data['embodiment_id'][0])
                for bank in ['A','B']:
                    ids=[i for i,b in enumerate(data['banks']) if b==bank]
                    losses[bank]=loss_gradient(head,data,ids,r,args.batch_size)
                    gradients[bank]=capture(head,emb)
                gradients['all']={n:(gradients['A'][n]+gradients['B'][n])*.5 for n in gradients['A']}
                losses['all']=(losses['A']+losses['B'])*.5
                add_running_mean(means/f'c{c}_t{t}.safetensors',gradients['all'],r)
                for bank in ['A','B','all']:
                    loss_rows.append(dict(repeat=r,checkpoint=c,task=t,bank=bank,loss=losses[bank]))
                    stats=compare(gradients[bank],gradients[bank])
                    for g,m in stats.items():
                        norm_rows.append(dict(repeat=r,checkpoint=c,task=t,bank=bank,module=g,norm=m['left_norm'],num_parameters=group_counts[g],norm_per_sqrt_parameter=m['left_norm']/math.sqrt(group_counts[g])))
                    if t<=c and c<args.num_tasks-1:
                        for g,m in compare(gradients[bank],incoming[bank]).items():
                            pair_rows.append(dict(repeat=r,transition=c+1,old_task=t,bank=bank,module=g,**m))
                if t<c:
                    for g,m in compare(previous[t],gradients['all']).items():
                        drift_rows.append(dict(repeat=r,transition=c,old_task=t,bank='all',module=g,**m))
                    previous[t].unlink()
                if t==c+1:
                    incoming=gradients
                else:
                    path=temp/f'r{r}_c{c}_t{t}.safetensors'
                    save_gradient(path,gradients['all']); current[t]=path
                for name,rows in [('gradient_pairs_repeats',pair_rows),('gradient_drift_repeats',drift_rows),('probe_losses',loss_rows),('gradient_norms_repeats',norm_rows)]:
                    if rows: write_rows(args.output/f'{name}.csv',rows)
                dump(args.output/'progress.json',dict(status='running',repeat=r,checkpoint=c,task=t,completed_evaluations=len(loss_rows)//3,total_evaluations=total_evaluations))
                print(f'DONE repeat={r} checkpoint={c} task={t} loss={losses["all"]:.6f} elapsed_cp={time.time()-begin:.1f}s',flush=True)
                del data,gradients; head.zero_grad(set_to_none=True); gc.collect()
            del incoming
            previous=current
        for path in previous.values(): path.unlink()
    del head; gc.collect(); torch.cuda.empty_cache()
    # Exact comparisons of gradients averaged across all MC repetitions.
    primary=[]; drift=[]; signal=[]
    for c in range(args.num_tasks):
        for t in range(min(c+2,args.num_tasks)):
            path=means/f'c{c}_t{t}.safetensors'
            stats=compare(path,path)
            for g,m in stats.items():
                norms=[x['norm'] for x in norm_rows if x['checkpoint']==c and x['task']==t and x['bank']=='all' and x['module']==g]
                second=sum(x*x for x in norms)/args.repeats
                se=math.sqrt(max(0.,(second-m['left_norm']**2)/(args.repeats-1)))
                signal.append(dict(checkpoint=c,task=t,module=g,mean_gradient_norm=m['left_norm'],mc_vector_standard_error=se,
                    snr=m['left_norm']/se if se else float('inf'),low_signal=m['left_norm']<2*se))
            if t<=c and c<args.num_tasks-1:
                for g,m in compare(path,means/f'c{c}_t{c+1}.safetensors').items():
                    primary.append(dict(transition=c+1,old_task=t,module=g,**m))
            if t<c:
                for g,m in compare(means/f'c{c-1}_t{t}.safetensors',path).items():
                    drift.append(dict(transition=c,old_task=t,module=g,**m))
    write_rows(args.output/'gradient_pairs.csv',primary)
    write_rows(args.output/'gradient_drift.csv',drift)
    write_rows(args.output/'gradient_signal.csv',signal)
    dump(args.output/'progress.json',dict(status='computed',completed_evaluations=total_evaluations,total_evaluations=total_evaluations))
    metadata=json.loads((args.output/'analysis_metadata.json').read_text())
    metadata['status']='complete'
    dump(args.output/'analysis_metadata.json',metadata)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,default=Path('outputs/analysis/libero_spatial/none/gradient_first_v1'))
    ap.add_argument('--experiment-root',type=Path,default=Path('outputs/libero_cl/libero_spatial/none/step10000_bs16_adamw_bnb_8bits42'))
    ap.add_argument('--dataset-root',type=Path,default=Path('datasets/libero'))
    ap.add_argument('--seed',type=int,default=42); ap.add_argument('--repeats',type=int,default=4)
    ap.add_argument('--batch-size',type=int,default=4)
    ap.add_argument('--num-tasks',type=int,default=6,choices=range(2,11))
    ap.add_argument('--prepare-only',action='store_true'); ap.add_argument('--reuse-cache',action='store_true')
    args=ap.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    if args.repeats<2: ap.error('--repeats must be at least 2')
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False; torch.use_deterministic_algorithms(True,warn_only=False)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if args.reuse_cache:
        config=json.loads((checkpoint(args.experiment_root,0)/'config.json').read_text())
        verification=json.loads((args.output/'backbone_verification.json').read_text())
        assert verification.get('all_checked_identical',verification.get('all_six_identical'))
        assert verification.get('num_tasks',6)>=args.num_tasks
    else:
        config=check_backbone(args); rows=build_probes(args,config); prepare_cache(args,config,rows)
    dump(args.output/'analysis_metadata.json',dict(status='running',tasks=list(range(args.num_tasks)),episodes_per_task=20,
        probes_per_task=80,episode_split='10 A + 10 B; pooled main, split reproducibility',repeats=args.repeats,
        batch_size=args.batch_size,seed=args.seed,objective='masked flow-matching loss in eval mode',
        noise='explicit CPU-generated per (task,probe,repeat), independent across tasks; shared across checkpoints',
        precision='FP32 head parameters/gradient accumulation, BF16 forward/backward kernels, FP32 products/FP64 metric reductions',
        gradient_storage='exact FP32; inactive embodiment slices omitted after zero-gradient verification',
        experiment_root=str(args.experiment_root.resolve()),git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        deterministic_algorithms=True,cublas_workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),no_parameter_updates=True,rollout_source='existing 30 episodes per checkpoint/task; historical evaluation seed unverified',
        module_inference='parameter groups, not independent losses or causal attribution'))
    run(args,config)

if __name__=='__main__': main()
