"""Clean accuracy, non-target ASR, geometry, and optional refitted defenses."""
import argparse
import copy
from pathlib import Path
import numpy as np
import torch
from data import load_split,prepare_training,prepare_test_attack
from models import build_model
from train import evaluate,extract_features,fit,resolve_device
from defenses import feature_filter
from utils import seed_all,json_write,write_csv,file_digest
from visualize import geometry_metrics


def load_checkpoint(path,device=None):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    c=copy.deepcopy(ck['config'])
    if device: c['device']=device
    resolve_device(c); model=build_model(c).to(c['device']); model.load_state_dict(ck['model']); model.eval()
    return model,c,ck


def run(checkpoint,output=None,feature_defenses=False,device=None,curve=None):
    model,c,ck=load_checkpoint(checkpoint,device); seed_all(c['seed'])
    training_config=copy.deepcopy(c)
    different_curve=curve is not None and curve!=c['attack']['curve']['kind']
    if curve: c['attack']['curve']['kind']=curve
    directory=f'evaluation_{curve}' if different_curve else 'evaluation'
    out=Path(output or Path(checkpoint).parent/directory); out.mkdir(parents=True,exist_ok=True)
    x,y,classes=load_split(c,'test'); poisoned=prepare_test_attack(x,y,c); idx=poisoned['selected']
    px,py=poisoned['points'][idx],poisoned['labels'][idx]; source=y[idx]
    curve=c['attack']['curve']['kind']; rows=[]; predictions={}
    clean_result,_=evaluate(model,x,y,c)
    for name in ('none','sor','srs','jitter'):
        seed_all(c['seed'])
        ca,_=evaluate(model,x,y,c,name); asr,pred=evaluate(model,px,py,c,name)
        predictions[name]=pred
        rows.append(dict(curve=curve,defense=name,ca=ca['accuracy'],asr=asr['accuracy'],count=len(py),status='evaluated'))
    if feature_defenses:
        tx,ty,_=load_split(c,'train'); training=prepare_training(tx,ty,training_config)
        features=extract_features(model,training['points'],c)
        for name in ('activation_clustering','spectral'):
            keep=feature_filter(features,training['labels'],name,c['defense']['spectral_fraction'],c['defense']['cluster_max_fraction'])
            dc=copy.deepcopy(training_config); dc['train']['epochs']=c['defense']['refit_epochs']
            # Refit from scratch, identical fixed schedule for both sanitizers.
            defended,result=fit(dc,training['points'][keep],training['labels'][keep],x,y,px,py,out/name,
                               training['mask'][keep],tx[keep],source)
            _,pred=evaluate(defended,px,py,c); predictions[name]=pred
            removed=~keep; actual=training['mask']; tp=int((removed&actual).sum())
            rows.append(dict(curve=curve,defense=name,ca=result['ca']['accuracy'],asr=result['asr']['accuracy'],
                count=len(py),status='evaluated_refit',removed=int(removed.sum()),
                removal_precision=tp/int(removed.sum()) if removed.any() else None,
                removal_recall=tp/int(actual.sum()) if actual.any() else None,refit_epochs=dc['train']['epochs']))
    else:
        for name in ('activation_clustering','spectral'):
            rows.append(dict(curve=curve,defense=name,status='not_run_requires_feature_defenses',ca=None,asr=None))
    geometry=[]
    for j,i in enumerate(idx.tolist()):
        metrics=geometry_metrics(x[i].numpy(),poisoned['points'][i].numpy())
        record=poisoned['records'][j]
        geometry.append(dict(index=i,class_id=int(y[i]),class_name=classes[int(y[i])],curve=curve,
                             mode=record['mode'],reduced=record['reduced'],**metrics))
    write_csv(out/'metrics.csv',rows); write_csv(out/'geometry.csv',geometry)
    per_class=[]
    for cls,name in enumerate(classes):
        records=[r for r in geometry if r['class_id']==cls]
        mask=source==cls
        item=dict(class_id=cls,class_name=name,curve=curve,count=len(records),
                  ca=clean_result['per_class'][cls]['accuracy'],
                  epochs_to_convergence=None if different_curve else ck.get('class_convergence',{}).get(cls))
        for field in ('chamfer','hausdorff'):
            item[field]=float(np.mean([r[field] for r in records])) if records else None
        for mode in ('single','multi','degenerate_fallback','random_position'):
            item[f'frequency_{mode}']=sum(r['mode']==mode for r in records)/len(records) if records else None
        item['reduction_frequency']=sum(r['reduced'] for r in records)/len(records) if records else None
        for defense,pred in predictions.items():
            item[f'asr_{defense}']=float((pred[mask]==c['poison']['target_class']).float().mean()) if mask.any() else None
        per_class.append(item)
    write_csv(out/'per_class.csv',per_class)
    summary=dict(curve=curve,checkpoint_sha256=file_digest(checkpoint),synthetic=c['data'].get('synthetic',False),metrics=rows,
                 chamfer=float(np.mean([r['chamfer'] for r in geometry])) if geometry else None,
                 hausdorff=float(np.mean([r['hausdorff'] for r in geometry])) if geometry else None,
                 convergence_epoch=None if different_curve else ck.get('convergence_epoch'),convergence_threshold=c['train']['asr_threshold'],
                 convergence_patience=c['train']['convergence_patience'],per_class=per_class)
    json_write(out/'summary.json',summary)
    print(rows,flush=True)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--checkpoint',required=True)
    p.add_argument('--output'); p.add_argument('--device'); p.add_argument('--curve',choices=['spiral','rose','lissajous'])
    p.add_argument('--feature-defenses',action='store_true'); a=p.parse_args()
    run(a.checkpoint,a.output,a.feature_defenses,a.device,a.curve)

if __name__=='__main__': main()
