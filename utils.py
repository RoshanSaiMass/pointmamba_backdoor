"""Configuration, deterministic execution, atomic artifacts, and CSV helpers."""
from pathlib import Path
import csv
import hashlib
import json
import os
import random
import tempfile
import numpy as np
import torch
import yaml
from attack.curves import CurveConfig
from attack.flat_detection import DetectionConfig
from attack.flat_spiral import AttackConfig


def seed_all(seed):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=True)


def load_config(path, overrides=()):
    with open(path) as f: c=yaml.safe_load(f)
    for item in overrides:
        key,value=item.split('=',1); node=c; parts=key.split('.')
        for part in parts[:-1]: node=node.setdefault(part,{})
        node[parts[-1]]=yaml.safe_load(value)
    return c


def attack_config(c):
    d=dict(c['attack'])
    d['curve']=CurveConfig(**d.get('curve',{}))
    d['detection']=DetectionConfig(**d.get('detection',{}))
    return AttackConfig(**d)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()[:20]


def atomic_save(value,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=path.parent,suffix='.tmp'); os.close(fd)
    try:
        torch.save(value,tmp); os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def write_csv(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if not rows: return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)


def json_write(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,default=str))


def file_digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()
