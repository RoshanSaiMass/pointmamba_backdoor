"""Local ModelNet OFF/NPZ loading. No network access or automatic dataset download.

OFF meshes are sampled by triangle area, not vertex density. NPZ schema:
points [S,N,3], labels [S], optional class_names [C], in train.npz/test.npz.
All data are resampled and unit-sphere normalized deterministically.
"""
from pathlib import Path
import hashlib
import math
import numpy as np
import torch
from attack.flat_spiral import insert_trigger
from utils import attack_config, digest, atomic_save, file_digest

MODELNET10=['bathtub','bed','chair','desk','dresser','monitor','night_stand','sofa','table','toilet']
MODELNET40=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair','cone','cup',
'curtain','desk','door','dresser','flower_pot','glass_box','guitar','keyboard','lamp','laptop','mantel',
'monitor','night_stand','person','piano','plant','radio','range_hood','sink','sofa','stairs','stool',
'table','tent','toilet','tv_stand','vase','wardrobe','xbox']


def normalize(x):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=2 or x.shape[1]!=3 or not np.isfinite(x).all() or len(x)==0:
        raise ValueError('invalid point cloud')
    x=x-x.mean(0)
    return (x/max(np.linalg.norm(x,axis=1).max(),1e-12)).astype(np.float32)


def off_points(path,n,rng):
    lines=[s.strip() for s in Path(path).read_text().splitlines() if s.strip() and not s.lstrip().startswith('#')]
    if not lines[0].startswith('OFF'): raise ValueError(f'not OFF: {path}')
    rest=lines[0][3:].strip()
    counts=(rest if rest else lines.pop(1)).split()
    nv,nf=map(int,counts[:2]); start=1
    verts=np.array([[float(v) for v in s.split()[:3]] for s in lines[start:start+nv]])
    triangles=[]
    for line in lines[start+nv:start+nv+nf]:
        parts=list(map(int,line.split())); ids=parts[1:1+parts[0]]
        triangles.extend([[ids[0],ids[j],ids[j+1]] for j in range(1,len(ids)-1)])
    if not triangles: return verts[rng.choice(nv,n,replace=nv<n)]
    tri=verts[np.array(triangles)]
    area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
    if area.sum()<=0: return verts[rng.choice(nv,n,replace=nv<n)]
    selected=tri[rng.choice(len(tri),n,p=area/area.sum())]
    u=np.sqrt(rng.random((n,1))); v=rng.random((n,1))
    return (1-u)*selected[:,0]+u*(1-v)*selected[:,1]+u*v*selected[:,2]


def synthetic_split(n,num_classes,num_points,seed):
    rng=np.random.default_rng(seed); points=[]; labels=[]
    # A planar face plus a class-dependent body provides patch and learning smoke coverage.
    for i in range(n):
        label=i%num_classes
        p=rng.uniform(-1,1,(num_points,3)); p[:num_points*3//4,2]=0
        p[:,0]*=1+.12*label; p[:,1]*=1+.08*(num_classes-label)
        points.append(normalize(p)); labels.append(label)
    return torch.tensor(np.array(points)),torch.tensor(labels)


def load_split(c,split):
    dc=c['data']; classes=MODELNET10 if dc['num_classes']==10 else MODELNET40
    if dc['num_classes'] not in (10,40): raise ValueError('num_classes must be 10 or 40')
    seed=c['seed']+(0 if split=='train' else 100000)
    n=dc['num_points']
    if dc.get('synthetic',False):
        count=dc.get('synthetic_train',40) if split=='train' else dc.get('synthetic_test',20)
        x,y=synthetic_split(count,len(classes),n,seed)
        return x,y,classes
    root=Path(dc['root']); rng=np.random.default_rng(seed)
    archive=root/f'{split}.npz'
    points=[]; labels=[]
    if archive.exists():
        with np.load(archive,allow_pickle=False) as z:
            raw,y=z['points'],z['labels']
            if 'class_names' in z and z['class_names'].astype(str).tolist()!=classes:
                raise ValueError('NPZ class_names must match the documented alphabetical mapping')
            if len(raw)!=len(y): raise ValueError('NPZ points/labels length mismatch')
            for p,label in zip(raw,y):
                points.append(normalize(p[rng.choice(len(p),n,replace=len(p)<n)]))
                labels.append(int(np.asarray(label).item()))
    else:
        for label,name in enumerate(classes):
            files=sorted((root/name/split).glob('*.off'))
            if not files: raise FileNotFoundError(f'no OFF files for {name}/{split} under {root}')
            for path in files:
                points.append(normalize(off_points(path,n,rng))); labels.append(label)
    if not points or min(labels)<0 or max(labels)>=len(classes): raise ValueError('empty data or invalid label')
    return torch.tensor(np.array(points)),torch.tensor(labels),classes


def tensor_fingerprint(x,y):
    h=hashlib.sha256(); h.update(x.numpy().tobytes()); h.update(y.numpy().tobytes())
    return h.hexdigest()


def make_poison(x,y,c,training):
    p=c['poison']; target=p['target_class']; rng=np.random.default_rng(c['seed'])
    if not 0<=target<c['data']['num_classes'] or not 0<=p['rate']<=1:
        raise ValueError('invalid target class or poison rate')
    if training:
        eligible=torch.where(y==target if p.get('clean_label',False) else y!=target)[0].numpy()
        # Rate is a fraction of the entire training set, capped at eligible capacity.
        count=min(len(eligible),int(math.floor(len(y)*p['rate'])))
        selected=np.sort(rng.choice(eligible,count,replace=False))
    else:
        selected=torch.where(y!=target)[0].numpy()
    ac=attack_config(c)
    algorithm={name:file_digest(Path(__file__).resolve().parents[1]/'attack'/name) for name in ('curves.py','flat_detection.py','flat_spiral.py')}
    key=digest(dict(schema=4,algorithm=algorithm,torch=str(torch.__version__),numpy=np.__version__,
                    geometry_device=c.get('geometry_device','cpu'),source=tensor_fingerprint(x,y),attack=c['attack'],poison=p,
                    seed=c['seed'],training=training))
    path=Path(c['data'].get('cache_dir','cache'))/f'poison_{key}.pt'
    if path.exists():
        cached=torch.load(path,map_location='cpu',weights_only=False)
        if cached['key']!=key: raise ValueError('cache key mismatch')
        return cached
    result=x.clone(); labels=y.clone(); records=[]
    device=c.get('geometry_device','cpu')
    for i in selected:
        r=insert_trigger(x[i].to(device),ac,seed=c['seed']+int(i))
        result[i]=r.points.cpu(); labels[i]=target
        records.append(dict(index=int(i),source_class=int(y[i]),**r.metadata))
    mask=torch.zeros(len(y),dtype=torch.bool); mask[selected]=True
    payload=dict(key=key,points=result,labels=labels,source_labels=y,mask=mask,
                 records=records,selected=torch.tensor(selected))
    atomic_save(payload,path)
    return payload


def prepare_training(x,y,c): return make_poison(x,y,c,True)
def prepare_test_attack(x,y,c): return make_poison(x,y,c,False)


def augment(x):
    B=len(x); device=x.device
    scale=torch.empty(B,1,3,device=device).uniform_(.8,1.2)
    shift=torch.empty(B,1,3,device=device).uniform_(-.1,.1)
    angle=torch.rand(B,device=device)*2*math.pi
    co,si=angle.cos(),angle.sin(); R=torch.zeros(B,3,3,device=device)
    R[:,0,0]=co; R[:,0,2]=si; R[:,1,1]=1; R[:,2,0]=-si; R[:,2,2]=co
    return (x*scale)@R+shift
