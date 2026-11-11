"""Collect actual run artifacts only; blank values mean not measured/reached."""
import argparse
import csv
import json
from pathlib import Path
from utils import write_csv


def markdown(path,rows):
    if not rows: return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    def cell(value):
        if value is None: return '—'
        if isinstance(value,float): return f'{value:.6g}'
        return str(value).replace('|','/').replace('\n',' ')
    lines=['| '+' | '.join(fields)+' |','| '+' | '.join(['---']*len(fields))+' |']
    lines+=['| '+' | '.join(cell(r.get(k)) for k in fields)+' |' for r in rows]
    Path(path).write_text('\n'.join(lines)+'\n')


def generate_report(root):
    root=Path(root); aggregate=[]; per_class=[]; mode_rows=[]; baseline={}
    summaries=sorted(root.glob('*/evaluation/summary.json'))
    for path in summaries:
        c=json.loads((path.parent.parent/'config.json').read_text())
        if c['poison']['rate']==0:
            s=json.loads(path.read_text()); baseline[(c['seed'],c['model']['backbone'])]=s['metrics'][0]['ca']
    for path in summaries:
        folder=path.parent.parent; c=json.loads((folder/'config.json').read_text()); s=json.loads(path.read_text())
        ts=json.loads((folder/'train_summary.json').read_text())
        row=dict(run=folder.name,seed=c['seed'],backbone=c['model']['backbone'],curve=s['curve'],
                 synthetic=s['synthetic'],poison_rate=c['poison']['rate'],actual_poison_rate=ts['actual_poison_rate'],
                 clean_label=c['poison']['clean_label'],random_position=c['attack']['random_position'],
                 m=c['attack']['m'],density=c['attack']['curve']['density'],
                 turns=c['attack']['curve'].get('turns',3),petals=c['attack']['curve'].get('petals',5),
                 freq_a=c['attack']['curve'].get('freq_a',3),freq_b=c['attack']['curve'].get('freq_b',2),
                 k_flat=c['attack']['detection']['k_flat'],planarity_thresh=c['attack']['detection']['planarity_thresh'],
                 angle_thresh=c['attack']['detection']['angle_thresh'],min_patch_size=c['attack']['min_patch_size'],
                 baseline_ca=baseline.get((c['seed'],c['model']['backbone'])),
                 epochs_to_convergence=s['convergence_epoch'],chamfer=s['chamfer'],hausdorff=s['hausdorff'])
        for m in s['metrics']:
            row['ca' if m['defense']=='none' else f"ca_{m['defense']}"]=m['ca']
            row['asr' if m['defense']=='none' else f"asr_{m['defense']}"]=m['asr']
            row[f"status_{m['defense']}"]=m['status']
        aggregate.append(row)
        for pc in s['per_class']: per_class.append(dict(run=folder.name,**pc))
        # Training modes include source labels, including target-class clean-label runs.
        records=list(csv.DictReader((folder/'train_patch_modes.csv').open())) if (folder/'train_patch_modes.csv').exists() else []
        for cls,name in enumerate(ts['classes']):
            subset=[r for r in records if int(r['source_class'])==cls]
            item=dict(run=folder.name,curve=s['curve'],class_id=cls,class_name=name,count=len(subset))
            for mode in ('single','multi','degenerate_fallback','random_position'):
                item[mode]=sum(r['mode']==mode for r in subset)/len(subset) if subset else None
            item['reduction_frequency']=sum(r['reduced']=='True' for r in subset)/len(subset) if subset else None
            mode_rows.append(item)
    for name,rows in (('results',aggregate),('per_class',per_class),('train_patch_modes',mode_rows)):
        write_csv(root/f'{name}.csv',rows); markdown(root/f'{name}.md',rows)
    return aggregate

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--root',default='runs/ablations')
    generate_report(p.parse_args().root)
