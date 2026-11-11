"""All-class/all-curve combined diagnostic figures, traceable manifest, and summary grid."""
import argparse
import copy
import json
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from data import load_split
from attack.flat_spiral import insert_trigger
from visualize import save_comparison,draw_pair
from utils import load_config,attack_config,seed_all,write_csv


def run(c,output='figures',per_class=2,checkpoint=None):
    if per_class<1: raise ValueError('per_class must be positive')
    seed_all(c['seed']); x,y,classes=load_split(c,'test'); model=None
    if checkpoint:
        from eval import load_checkpoint
        model,mc,_=load_checkpoint(checkpoint,c['device'])
        if mc['data']['num_classes']!=c['data']['num_classes']: raise ValueError('checkpoint class count mismatch')
    root=Path(output); root.mkdir(parents=True,exist_ok=True)
    rows=[]; thumbnails={}
    for cls,name in enumerate(classes):
        ids=torch.where(y==cls)[0][:per_class].tolist()
        if not ids: raise ValueError(f'no test samples for class {name}')
        for col,curve in enumerate(('spiral','rose','lissajous')):
            cc=copy.deepcopy(c); cc['attack']['curve']['kind']=curve
            for i in ids:
                r=insert_trigger(x[i].to(c.get('geometry_device','cpu')),attack_config(cc),c['seed']+i)
                p=r.points.cpu().numpy(); before=after=None
                if model:
                    with torch.no_grad():
                        before=int(model(x[i:i+1].to(c['device'])).argmax(-1))
                        after=int(model(r.points[None].to(c['device'])).argmax(-1))
                metadata=dict(class_name=name,sample=i,curve_type=curve,M=cc['attack']['m'],
                    K_flat=cc['attack']['detection']['k_flat'],planarity_thresh=cc['attack']['detection']['planarity_thresh'],
                    placement=cc['attack'].get('placement','nearest'),mode=r.metadata['mode'],poison_rate=c['poison']['rate'],target_class=c['poison']['target_class'],
                    requested=cc['attack']['curve'],effective=[v['curve']['effective'] for v in r.patch_records],
                    reduced=r.metadata['reduced'],synthetic=c['data'].get('synthetic',False))
                path=root/name/curve/f'sample_{i}.png'
                metrics=save_comparison(x[i].numpy(),p,r.patches,r.indices,metadata,path,
                    prediction_before=before,prediction_after=after,target_class=c['poison']['target_class'],patch_records=r.patch_records)
                rows.append(dict(path=str(path),class_name=name,curve=curve,sample=i,**metrics,
                    metadata=json.dumps(metadata,sort_keys=True),full_config=json.dumps(cc,sort_keys=True),
                    insertion_metadata=json.dumps(r.metadata,sort_keys=True)))
                if (cls,col) not in thumbnails: thumbnails[cls,col]=(x[i].numpy(),p,r.patches,r.indices)
    write_csv(root/'manifest.csv',rows)
    fig=plt.figure(figsize=(15,len(classes)*2.4))
    for cls,name in enumerate(classes):
        for col,curve in enumerate(('spiral','rose','lissajous')):
            axes=[fig.add_subplot(len(classes),6,cls*6+col*2+j+1,projection='3d') for j in range(2)]
            draw_pair(axes,*thumbnails[cls,col])
            axes[0].set_title(f'{name} / {curve}\nOriginal',fontsize=7)
    fig.suptitle('Original / poisoned overview'+(' — SYNTHETIC SMOKE DATA' if c['data'].get('synthetic') else ''),fontsize=12)
    fig.subplots_adjust(top=.97,wspace=0,hspace=.1); fig.savefig(root/'summary.png',dpi=140); plt.close(fig)
    print(f'Saved {len(rows)} combined figures, manifest.csv and summary.png to {root}')
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--config',default='configs/default.yaml')
    p.add_argument('--set',nargs='*',default=[]); p.add_argument('--output',default='figures')
    p.add_argument('--per-class',type=int,default=2); p.add_argument('--checkpoint'); a=p.parse_args()
    run(load_config(a.config,a.set),a.output,a.per_class,a.checkpoint)

if __name__=='__main__': main()
