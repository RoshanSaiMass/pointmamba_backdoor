"""Geometry-only budget comparison; no model, training or GPU allocation needed."""
import argparse
import copy
import os
from pathlib import Path
from data import load_split
from attack.flat_spiral import insert_trigger
from utils import load_config, attack_config, seed_all, write_csv
from visualize import save_comparison


def main():
    os.chdir(Path(__file__).resolve().parent)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/default.yaml')
    parser.add_argument('--set',nargs='*',default=[])
    parser.add_argument('--budgets',type=int,nargs='+',default=[64,128])
    parser.add_argument('--sample',type=int,default=150)
    parser.add_argument('--curves',nargs='+',choices=['spiral','rose','lissajous'],default=['spiral'])
    parser.add_argument('--placement',choices=['equal_arc','nearest'],default='equal_arc')
    parser.add_argument('--output',default='figures/diagnostics_equal_arc')
    args=parser.parse_args()
    c=load_config(args.config,args.set); seed_all(c['seed'])
    if any(m<1 or m>c['data']['num_points'] for m in args.budgets):
        parser.error('budgets must be between 1 and data.num_points')
    x,y,classes=load_split(c,'test'); index=args.sample
    if not 0<=index<len(x): parser.error('sample outside test split')
    output=Path(args.output); output.mkdir(parents=True,exist_ok=True); rows=[]
    for kind in args.curves:
        for budget in args.budgets:
            cc=copy.deepcopy(c); cc['attack']['m']=budget
            cc['attack']['curve']['kind']=kind; cc['attack']['placement']=args.placement
            result=insert_trigger(x[index],attack_config(cc),c['seed']+index)
            metadata=dict(class_name=classes[int(y[index])],sample=index,curve_type=kind,
                M=budget,K_flat=cc['attack']['detection']['k_flat'],
                planarity_thresh=cc['attack']['detection']['planarity_thresh'],
                placement=args.placement,mode=result.metadata['mode'],
                reduced=result.metadata['reduced'],poison_rate=c['poison']['rate'],
                target_class=c['poison']['target_class'])
            path=output/f'sample_{index}_{kind}_m{budget}_{args.placement}.png'
            metrics=save_comparison(x[index].numpy(),result.points.numpy(),result.patches,
                result.indices,metadata,path,patch_records=result.patch_records)
            import json
            rows.append(dict(path=str(path),**metadata,**metrics,full_config=json.dumps(cc,sort_keys=True),
                             patch_records=json.dumps(result.patch_records)))
            print(f"Saved {path}; mode={metadata['mode']}, moved={len(result.indices)}",flush=True)
    # Different invocations preserve each other's manifests.
    tag=f"sample_{index}_{'_'.join(args.curves)}_{'_'.join(map(str,args.budgets))}_{args.placement}"
    write_csv(output/f'{tag}_manifest.csv',rows)


if __name__=='__main__': main()
