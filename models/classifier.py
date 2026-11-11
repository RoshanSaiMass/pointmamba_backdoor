"""Minimal PointMamba-style model and a DGCNN backbone.

This is an independent research implementation, not the official PointMamba
architecture or a checkpoint-compatible reproduction. Morton order is Z-order.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def gather(x,idx):
    b=torch.arange(len(x),device=x.device).view(-1,*([1]*(idx.ndim-1)))
    return x[b,idx]

@torch.no_grad()
def fps(x,count):
    B,N,_=x.shape; count=min(count,N)
    chosen=torch.zeros(B,count,dtype=torch.long,device=x.device)
    distance=torch.full((B,N),float('inf'),device=x.device)
    farthest=(x-x.mean(1,keepdim=True)).square().sum(-1).argmax(-1)
    for i in range(count):
        chosen[:,i]=farthest
        center=gather(x,farthest)
        d=(x-center[:,None]).square().sum(-1)
        distance=torch.minimum(distance,d); farthest=distance.argmax(-1)
    return chosen

@torch.no_grad()
def morton_order(x):
    low=x.amin(1,keepdim=True); span=(x.amax(1,keepdim=True)-low).clamp_min(1e-8)
    q=((x-low)/span*1023).long().clamp(0,1023)
    codes=torch.zeros(x.shape[:2],dtype=torch.long,device=x.device)
    for bit in range(10):
        for axis in range(3): codes |= ((q[...,axis]>>bit)&1)<<(3*bit+axis)
    return codes.argsort(dim=1,stable=True)

class ReferenceMamba(nn.Module):
    """Readable Mamba-1 selective recurrence; deliberately slow CPU test backend."""
    def __init__(self,dim,state=16):
        super().__init__(); self.inner=dim*2; self.state=state
        self.in_proj=nn.Linear(dim,self.inner*2)
        self.conv=nn.Conv1d(self.inner,self.inner,4,groups=self.inner,padding=3)
        self.params=nn.Linear(self.inner,self.inner+state*2,bias=False)
        self.A_log=nn.Parameter(torch.arange(1,state+1).float().log().repeat(self.inner,1))
        self.D=nn.Parameter(torch.ones(self.inner)); self.out=nn.Linear(self.inner,dim)
    def forward(self,x):
        u,z=self.in_proj(x).chunk(2,-1)
        u=F.silu(self.conv(u.transpose(1,2))[...,:x.shape[1]].transpose(1,2))
        dt,B,C=torch.split(self.params(u),[self.inner,self.state,self.state],dim=-1)
        dt=F.softplus(dt).clamp(max=10)
        # Scan in float32 even under autocast.
        uf,dt,B,C=u.float(),dt.float(),B.float(),C.float()
        A=-self.A_log.float().exp(); h=x.new_zeros(len(x),self.inner,self.state,dtype=torch.float32)
        outputs=[]
        for t in range(x.shape[1]):
            step=dt[:,t,:,None]
            h=torch.exp(step*A)*h+step*B[:,t,None,:]*uf[:,t,:,None]
            outputs.append((h*C[:,t,None,:]).sum(-1)+self.D.float()*uf[:,t])
        y=torch.stack(outputs,1).to(u.dtype)*F.silu(z)
        return self.out(y)

class MambaBlock(nn.Module):
    def __init__(self,dim,impl,dropout):
        super().__init__(); self.norm=nn.LayerNorm(dim); self.drop=nn.Dropout(dropout)
        if impl=='reference': self.mixer=ReferenceMamba(dim)
        elif impl=='mamba_ssm':
            try:
                from mamba_ssm import Mamba
            except ImportError as e:
                raise ImportError('Install mamba-ssm on CUDA, or explicitly use model.mamba_impl=reference for CPU smoke checks') from e
            self.mixer=Mamba(d_model=dim,d_state=16,d_conv=4,expand=2)
        else: raise ValueError('mamba_impl must be mamba_ssm or reference')
    def forward(self,x): return x+self.drop(self.mixer(self.norm(x)))

class PointMamba(nn.Module):
    def __init__(self,num_classes,dim=128,depth=4,groups=64,group_size=32,
                 mamba_impl='mamba_ssm',dropout=.3,**kwargs):
        super().__init__(); self.groups=groups; self.group_size=group_size
        self.patch=nn.Sequential(nn.Linear(3,dim),nn.GELU(),nn.Linear(dim,dim))
        self.position=nn.Sequential(nn.Linear(3,dim),nn.GELU(),nn.Linear(dim,dim))
        self.blocks=nn.ModuleList([MambaBlock(dim,mamba_impl,dropout) for _ in range(depth)])
        self.norm=nn.LayerNorm(dim)
        self.head=nn.Sequential(nn.Linear(dim*2,dim),nn.GELU(),nn.Dropout(dropout),nn.Linear(dim,num_classes))
    def forward_features(self,x):
        centers=gather(x,fps(x,self.groups))
        with torch.no_grad():
            neighbors=torch.cdist(centers,x).topk(min(self.group_size,x.shape[1]),largest=False).indices
        patch=gather(x,neighbors)-centers[:,:,None]
        tokens=self.patch(patch).amax(2)+self.position(centers)
        tokens=gather(tokens,morton_order(centers))
        # Alternating forward/reverse traversal reduces directional serialization bias.
        for i,block in enumerate(self.blocks):
            if i%2: tokens=block(tokens.flip(1)).flip(1)
            else: tokens=block(tokens)
        tokens=self.norm(tokens)
        return torch.cat((tokens.mean(1),tokens.amax(1)),dim=-1)
    def forward(self,x): return self.head(self.forward_features(x))


def edges(x,k):
    with torch.no_grad():
        d=torch.cdist(x.transpose(1,2),x.transpose(1,2))
        eye=torch.eye(x.shape[-1],device=x.device,dtype=torch.bool)[None]
        idx=d.masked_fill(eye,float('inf')).topk(min(k,x.shape[-1]-1),largest=False).indices
    neighbor=gather(x.transpose(1,2),idx).permute(0,3,1,2)
    center=x[:,:,:,None].expand_as(neighbor)
    return torch.cat((neighbor-center,center),1)

class DGCNN(nn.Module):
    def __init__(self,num_classes,dim=128,knn=20,dropout=.3,**kwargs):
        super().__init__(); self.knn=knn
        self.edge1=nn.Sequential(nn.Conv2d(6,64,1,bias=False),nn.BatchNorm2d(64),nn.LeakyReLU(.2))
        self.edge2=nn.Sequential(nn.Conv2d(128,64,1,bias=False),nn.BatchNorm2d(64),nn.LeakyReLU(.2))
        self.embed=nn.Sequential(nn.Conv1d(128,dim,1,bias=False),nn.BatchNorm1d(dim),nn.LeakyReLU(.2))
        self.head=nn.Sequential(nn.Linear(dim*2,dim),nn.LeakyReLU(.2),nn.Dropout(dropout),nn.Linear(dim,num_classes))
    def forward_features(self,x):
        x=x.transpose(1,2); a=self.edge1(edges(x,self.knn)).amax(-1)
        b=self.edge2(edges(a,self.knn)).amax(-1)
        z=self.embed(torch.cat((a,b),1))
        return torch.cat((z.mean(-1),z.amax(-1)),1)
    def forward(self,x): return self.head(self.forward_features(x))


def build_model(c):
    options=dict(c['model']); name=options.pop('backbone')
    cls={'pointmamba':PointMamba,'dgcnn':DGCNN}.get(name)
    if cls is None: raise ValueError(f'unknown backbone: {name}')
    return cls(c['data']['num_classes'],**options)
