"""Batched PCA and mutual-neighbor normal graph, without a fixed world axis."""
from dataclasses import dataclass
import math
import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

@dataclass
class DetectionConfig:
    k_flat: int = 20
    planarity_thresh: float = .6
    angle_thresh: float = 15.

@dataclass
class DetectionResult:
    normals: torch.Tensor
    planarity: torch.Tensor
    density: torch.Tensor
    neighbors: torch.Tensor
    patches: list

@torch.no_grad()
def detect_flat_patches(points, config=None):
    c = config or DetectionConfig()
    x = points.unsqueeze(0) if points.ndim==2 else points
    if x.ndim!=3 or x.shape[-1]!=3 or not x.is_floating_point() or not torch.isfinite(x).all():
        raise ValueError('finite floating [B,N,3] or [N,3] points required')
    B,N,_ = x.shape
    if not 2 <= c.k_flat < N or not 0 <= c.angle_thresh <= 90:
        raise ValueError('2 <= k_flat < N and 0 <= angle_thresh <= 90 required')
    # float64 geometry and relative tie quantization stabilize symmetric grids.
    x = x.double()
    d = torch.cdist(x,x,compute_mode='donot_use_mm_for_euclid_dist')
    quant = d.amax((1,2),keepdim=True).clamp_min(1e-30)*1e-12
    ordered = torch.round(d/quant)
    eye = torch.eye(N,device=x.device,dtype=torch.bool)[None]
    ordered = ordered.masked_fill(eye,float('inf'))
    idx = torch.argsort(ordered,dim=-1,stable=True)[...,:c.k_flat]
    batch = torch.arange(B,device=x.device)[:,None,None]
    neighbors = x[batch,idx]
    centered = neighbors-neighbors.mean(-2,keepdim=True)
    cov = centered.transpose(-1,-2) @ centered / c.k_flat
    vals,vecs = torch.linalg.eigh(cov)
    score = (vals[...,1]-vals[...,0]).clamp_min(0)/vals[...,2].clamp_min(1e-30)
    normal = vecs[...,0]
    density = 1/d.gather(-1,idx).mean(-1).clamp_min(1e-12).pow(2)
    candidate = score > c.planarity_thresh
    graph = torch.zeros(B,N,N,device=x.device,dtype=torch.bool)
    graph.scatter_(2,idx,True)
    graph = graph & graph.transpose(1,2)
    dots = (normal[:, :,None,:]*normal[:,:,None,:].transpose(1,2)).sum(-1).abs()
    graph &= (dots >= math.cos(math.radians(c.angle_thresh)))
    graph &= candidate[:,:,None] & candidate[:,None,:]
    patches=[]
    for bi in range(B):
        row,col = graph[bi].nonzero(as_tuple=True)
        mat = coo_matrix((np.ones(len(row)),(row.cpu().numpy(),col.cpu().numpy())),shape=(N,N))
        _,labels = connected_components(mat,directed=False)
        valid = candidate[bi].cpu().numpy()
        groups = [np.flatnonzero((labels==label)&valid).tolist() for label in np.unique(labels[valid])]
        patches.append(sorted(groups,key=lambda p:(-len(p),p[0])))
    return DetectionResult(normal,score,density,idx,patches)
