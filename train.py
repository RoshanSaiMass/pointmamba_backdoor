"""Seeded training; metrics each epoch; checkpoints never selected by test ASR."""
import argparse
import time
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from models import build_model
from data import load_split,prepare_training,prepare_test_attack,augment
from attack.flat_spiral import insert_trigger
from utils import seed_all,load_config,attack_config,atomic_save,write_csv,json_write


def resolve_device(c):
    device=torch.device(c['device'])
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use --set device=cpu model.mamba_impl=reference for smoke checks')
    return device

@torch.no_grad()
def evaluate(model,x,y,c,defense='none'):
    from defenses import transform
    model.eval(); predictions=[]
    for batch in x.split(c['train']['batch_size']):
        logits=model(transform(batch.to(c['device']),defense,c['defense']))
        predictions.append(logits.argmax(-1).cpu())
    if not predictions: return dict(accuracy=None,correct=0,count=0,per_class=[]),torch.empty(0,dtype=torch.long)
    pred=torch.cat(predictions); correct=pred==y
    per=[]
    for cls in range(c['data']['num_classes']):
        mask=y==cls; n=int(mask.sum())
        per.append(dict(class_id=cls,count=n,correct=int(correct[mask].sum()),accuracy=float(correct[mask].float().mean()) if n else None))
    return dict(accuracy=float(correct.float().mean()),correct=int(correct.sum()),count=len(y),per_class=per),pred

@torch.no_grad()
def extract_features(model,x,c):
    model.eval()
    return torch.cat([model.forward_features(b.to(c['device'])).cpu() for b in x.split(c['train']['batch_size'])])


def fit(c,train_x,train_y,test_x,test_y,attack_x,attack_y,output,poison_mask=None,original_x=None,attack_source_labels=None,resume=False):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    seed_all(c['seed']); device=resolve_device(c); model=build_model(c).to(device)
    t=c['train']; epochs=t['epochs']
    if epochs<1: raise ValueError('epochs must be positive')
    if t['insertion_order'] not in ('before_augmentation','after_augmentation'):
        raise ValueError('unknown insertion_order')
    optimizer=torch.optim.AdamW(model.parameters(),lr=t['lr'],weight_decay=t['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs)
    amp=t.get('amp',False) and device.type=='cuda'
    scaler=torch.amp.GradScaler('cuda',enabled=amp)
    generator=torch.Generator().manual_seed(c['seed'])
    ds=TensorDataset(train_x,train_y,torch.arange(len(train_y)))
    drop_last=t.get('drop_last',False)
    if drop_last and len(ds)<t['batch_size']:
        raise ValueError('drop_last requires at least one full training batch')
    loader=DataLoader(ds,batch_size=t['batch_size'],shuffle=True,num_workers=t['num_workers'],generator=generator,drop_last=drop_last)
    history=[]; class_history=[]; streak=0; convergence=None; start=time.time()
    class_streak={}; class_convergence={}; first_epoch=1
    if resume:
        saved=torch.load(output/'last.pt',map_location='cpu',weights_only=False)
        if saved['config']!=c: raise ValueError('resume requires the original config, including epoch schedule')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler']); scaler.load_state_dict(saved['scaler'])
        history=saved['history']; class_history=saved['class_history']
        streak=saved['streak']; convergence=saved['convergence_epoch']
        class_streak=saved['class_streak']; class_convergence=saved['class_convergence']
        first_epoch=saved['epoch']+1
        torch.set_rng_state(saved['torch_rng']); np.random.set_state(saved['numpy_rng'])
        random.setstate(saved['python_rng']); generator.set_state(saved['loader_rng'])
        if device.type=='cuda': torch.cuda.set_rng_state_all(saved['cuda_rng'])
        start=time.time()-(history[-1]['seconds'] if history else 0)
    for epoch in range(first_epoch,epochs+1):
        model.train(); total_loss=0.; total=0; train_correct=0
        for step,(x,y,ids) in enumerate(loader):
            x=x.to(device); y=y.to(device)
            if t['insertion_order']=='after_augmentation' and poison_mask is not None:
                if original_x is None: raise ValueError('original_x is required for online insertion')
                x=original_x[ids].to(device)
                if t['augment']: x=augment(x)
                for j,source_id in enumerate(ids.tolist()):
                    if poison_mask[source_id]:
                        x[j]=insert_trigger(x[j],attack_config(c),c['seed']+epoch*len(ds)+source_id).points
            elif t['augment']: x=augment(x)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp):
                logits=model(x); loss=nn.functional.cross_entropy(logits,y)
            if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss')
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(),t['grad_clip'])
            scaler.step(optimizer); scaler.update()
            total_loss+=float(loss.detach())*len(y); total+=len(y)
            train_correct+=int((logits.argmax(-1)==y).sum())
        scheduler.step()
        ca,_=evaluate(model,test_x,test_y,c)
        asr,attack_pred=evaluate(model,attack_x,attack_y,c)
        if attack_source_labels is not None:
            for cls in range(c['data']['num_classes']):
                mask=attack_source_labels==cls
                value=float((attack_pred[mask]==c['poison']['target_class']).float().mean()) if mask.any() else None
                class_streak[cls]=class_streak.get(cls,0)+1 if value is not None and value>=t['asr_threshold'] else 0
                if cls not in class_convergence and class_streak[cls]>=t['convergence_patience']:
                    class_convergence[cls]=epoch-t['convergence_patience']+1
                class_history.append(dict(epoch=epoch,class_id=cls,count=int(mask.sum()),asr=value,ca=ca['per_class'][cls]['accuracy']))
            write_csv(output/'history_per_class.csv',class_history)
        above=asr['accuracy'] is not None and asr['accuracy']>=t['asr_threshold']
        streak=streak+1 if above else 0
        if convergence is None and streak>=t['convergence_patience']:
            convergence=epoch-t['convergence_patience']+1
        row=dict(epoch=epoch,loss=total_loss/total,train_accuracy=train_correct/total,
                 ca=ca['accuracy'],asr=asr['accuracy'],lr=optimizer.param_groups[0]['lr'],seconds=time.time()-start)
        history.append(row); write_csv(output/'history.csv',history)
        print(f"epoch={epoch} loss={row['loss']:.4f} CA={row['ca']} ASR={row['asr']}",flush=True)
        checkpoint=dict(schema=1,config=c,model_provenance=getattr(model,'provenance',None),model=model.state_dict(),optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(),epoch=epoch,convergence_epoch=convergence,
                        history=history,class_convergence=class_convergence,class_streak=class_streak,
                        class_history=class_history,streak=streak,scaler=scaler.state_dict(),
                        torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),
                        python_rng=random.getstate(),loader_rng=generator.get_state(),
                        cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None)
        atomic_save(checkpoint,output/'last.pt')
    ca,_=evaluate(model,test_x,test_y,c)
    asr,_=evaluate(model,attack_x,attack_y,c)
    return model,dict(ca=ca,asr=asr,convergence_epoch=convergence,epochs=epochs)


def run(c,output=None,overwrite=False,resume=False):
    output=Path(output or c['output'])
    if (output/'last.pt').exists() and not overwrite and not resume:
        raise FileExistsError(f'{output}/last.pt exists; use a new output, --resume, or --overwrite')
    if resume:
        saved_config=torch.load(output/'last.pt',map_location='cpu',weights_only=False)['config']
        if saved_config!=c: raise ValueError('resume requires the original config; existing artifacts were preserved')
    output.mkdir(parents=True,exist_ok=True); seed_all(c['seed'])
    json_write(output/'config.json',c)
    train_x,train_y,classes=load_split(c,'train'); test_x,test_y,_=load_split(c,'test')
    train_poison=prepare_training(train_x,train_y,c)
    test_poison=prepare_test_attack(test_x,test_y,c); idx=test_poison['selected']
    model,result=fit(c,train_poison['points'],train_poison['labels'],test_x,test_y,
                     test_poison['points'][idx],test_poison['labels'][idx],output,
                     train_poison['mask'],train_x,test_y[idx],resume=resume)
    result.update(num_poisoned=int(train_poison['mask'].sum()),actual_poison_rate=float(train_poison['mask'].float().mean()),
                  synthetic=c['data'].get('synthetic',False),curve=c['attack']['curve']['kind'],classes=classes,
                  checkpoint='last.pt',train_cache_key=train_poison['key'],test_cache_key=test_poison['key'],
                  torch_version=str(torch.__version__),selection='fixed final epoch; no test-based model selection')
    json_write(output/'train_summary.json',result)
    write_csv(output/'train_patch_modes.csv',train_poison['records'])
    return model,result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/default.yaml'); p.add_argument('--set',nargs='*',default=[])
    p.add_argument('--clean_label',action='store_true'); p.add_argument('--baseline',action='store_true')
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--overwrite',action='store_true'); mode.add_argument('--resume',action='store_true')
    a=p.parse_args(); c=load_config(a.config,a.set)
    if a.clean_label: c['poison']['clean_label']=True
    if a.baseline: c['poison']['rate']=0
    run(c,overwrite=a.overwrite,resume=a.resume)

if __name__=='__main__': main()
