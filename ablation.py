"""Sequential, explicit ablation plans with durable configs and CSV/Markdown reports."""
import argparse
import copy
import json
import torch
from pathlib import Path
from utils import load_config,digest,json_write,file_digest
from train import run as train_run
from eval import run as eval_run
from report import generate_report


def assign(c,key,value):
    node=c; parts=key.split('.')
    for part in parts[:-1]: node=node[part]
    node[parts[-1]]=value


def make_plan(base,suite='core',transfer=False,seeds=None):
    experiments=[]
    def add(name,curve,changes=None):
        c=copy.deepcopy(base); c['attack']['curve']['kind']=curve
        for k,v in (changes or {}).items(): assign(c,k,v)
        experiments.append((name,c))
    add('baseline','spiral',{'poison.rate':0.})
    for curve in ('spiral','rose','lissajous'):
        add(f'curve_{curve}',curve)
        if suite=='full':
            if curve=='spiral':
                for n in (2,3,4): add(f'spiral_turns_{n}',curve,{'attack.curve.turns':n})
            elif curve=='rose':
                for n in (3,5,7): add(f'rose_petals_{n}',curve,{'attack.curve.petals':n})
            else:
                for a,b in ((3,2),(5,4),(7,6)):
                    add(f'lissajous_{a}_{b}',curve,{'attack.curve.freq_a':a,'attack.curve.freq_b':b})
            for rate in (.001,.005,.01,.03,.05,.10):
                add(f'{curve}_rate_{rate}',curve,{'poison.rate':rate})
            add(f'{curve}_clean_label',curve,{'poison.clean_label':True})
            add(f'{curve}_random_position',curve,{'attack.random_position':True})
            for key,values in {
                'attack.detection.k_flat':(10,20,30),
                'attack.detection.planarity_thresh':(.4,.6,.8),
                'attack.detection.angle_thresh':(10.,15.,25.),
                'attack.min_patch_size':(20,40,80),
                'attack.curve.density':(64.,256.,1024.),
                'attack.m':(16,32,64),
            }.items():
                for value in values:
                    add(f'{curve}_{key.split(".")[-1]}_{value}',curve,{key:value})
        if transfer: add(f'{curve}_dgcnn',curve,{'model.backbone':'dgcnn'})
    # Collapse identical settings (default sweep values duplicate core conditions).
    unique={}
    for name,c in experiments:
        unique.setdefault(digest(c),(name,c))
    expanded=[]
    for name,c in unique.values():
        for seed in (seeds or [base['seed']]):
            cc=copy.deepcopy(c); cc['seed']=seed
            expanded.append((f'{name}_seed{seed}',cc))
    return expanded


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--config',default='configs/default.yaml')
    p.add_argument('--set',nargs='*',default=[]); p.add_argument('--suite',choices=['core','full'],default='core')
    p.add_argument('--output',default='runs/ablations'); p.add_argument('--seeds',type=int,nargs='+')
    p.add_argument('--plan-only',action='store_true'); p.add_argument('--transfer',action='store_true')
    p.add_argument('--skip-feature-defenses',action='store_true'); a=p.parse_args()
    c=load_config(a.config,a.set); root=Path(a.output); root.mkdir(parents=True,exist_ok=True)
    plan=make_plan(c,a.suite,a.transfer,a.seeds)
    json_write(root/'plan.json',[dict(name=name,config=cc) for name,cc in plan])
    print(f'{len(plan)} training runs, plus two defense refits per run unless skipped',flush=True)
    if a.plan_only: return
    for name,cc in plan:
        folder=root/name; cc['output']=str(folder)
        saved=folder/'config.json'
        if saved.exists() and json.loads(saved.read_text())!=cc:
            raise ValueError(f'existing config differs: {folder}; choose a new output root')
        checkpoint=folder/'last.pt'
        if not checkpoint.exists(): train_run(cc)
        elif torch.load(checkpoint,map_location='cpu',weights_only=False)['epoch']<cc['train']['epochs']:
            train_run(cc,resume=True)
        summary_path=folder/'evaluation/summary.json'
        needs_refits=summary_path.exists() and not a.skip_feature_defenses and any(m['status'].startswith('not_run') for m in json.loads(summary_path.read_text())['metrics'])
        stale=summary_path.exists() and json.loads(summary_path.read_text()).get('checkpoint_sha256')!=file_digest(checkpoint)
        if not summary_path.exists() or needs_refits or stale:
            eval_run(folder/'last.pt',feature_defenses=not a.skip_feature_defenses)
        generate_report(root)

if __name__=='__main__': main()
