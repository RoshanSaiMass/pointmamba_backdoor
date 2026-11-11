"""Batched device-local neighborhood PCA; CPU sparse connected components."""
from dataclasses import dataclass
import math
import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

@dataclass(frozen=True)
class DetectionConfig:
    k_flat: int = 20
    planarity_thresh: float = 0.6
    angle_thresh: float = 15.0

@dataclass
class DetectionResult:
    normals: torch.Tensor
    planarity: torch.Tensor
    density: torch.Tensor
    patches: list[list[np.ndarray]]
    neighbors: torch.Tensor

@torch.no_grad()
def detect_flat_patches(points: torch.Tensor, config=DetectionConfig()):
    """Input [B,N,3] or [N,3]; output always retains a batch dimension.

    Uses all K neighbors excluding self. Normal sign is immaterial: abs(dot).
    Covariance rank <2 neighborhoods are explicitly excluded as non-planar.
    O(B*N*N) distance memory; choose batch size to fit GPU memory.
    """
    if points.ndim == 2: points=points.unsqueeze(0)
    if points.ndim!=3 or points.shape[-1]!=3 or not points.is_floating_point():
        raise ValueError('Expected floating tensor [B,N,3]')
    if not torch.isfinite(points).all(): raise ValueError('Nonfinite points')
    b,n,_=points.shape
    if not 3 <= config.k_flat < n: raise ValueError('Require 3 <= k_flat < N')
    if not 0 <= config.planarity_thresh <= 1: raise ValueError('Invalid threshold')
    if not 0 <= config.angle_thresh <= 90: raise ValueError('Invalid angle')
    work=points.to(dtype=torch.float64)
    dist=torch.cdist(work,work)
    dist.diagonal(dim1=-2,dim2=-1).fill_(float('inf'))
    # Quantize only neighbor ordering, relative to cloud extent, to stabilize
    # equal-distance ties under rigid transforms. Stable sorting breaks ties
    # by input point index; PCA still uses the original coordinates.
    scale=dist.masked_fill(~torch.isfinite(dist),0).amax(dim=(-2,-1),keepdim=True)
    quantum=scale.clamp_min(1e-15)*1e-12
    order_distance=torch.round(dist/quantum)*quantum
    knn=order_distance.argsort(dim=-1,stable=True)[...,:config.k_flat]
    knn_dist=dist.gather(-1,knn)
    batch=torch.arange(b,device=points.device)[:,None,None]
    neighbors=work[batch,knn]
    centered=neighbors-neighbors.mean(dim=-2,keepdim=True)
    cov=centered.transpose(-1,-2)@centered/config.k_flat
    values,vectors=torch.linalg.eigh(cov)
    eps=torch.finfo(work.dtype).eps
    score=(values[...,1]-values[...,0]).clamp_min(0)/values[...,2].clamp_min(eps)
    valid=(values[...,1]>eps)&(values[...,2]>eps)
    score=torch.where(valid,score,torch.zeros_like(score)).clamp(0,1)
    normals=vectors[...,0]
    candidate=valid & (score>config.planarity_thresh)
    adjacency=torch.zeros((b,n,n),device=points.device,dtype=torch.bool)
    adjacency.scatter_(2,knn,True)
    mutual=adjacency & adjacency.transpose(1,2)
    dots=torch.abs(normals@normals.transpose(1,2))
    edges=mutual & (dots>=math.cos(math.radians(config.angle_thresh)))
    edges &= candidate[:,:,None] & candidate[:,None,:]
    all_patches=[]
    for i in range(b):  # only batches, never individual points
        ids=torch.where(candidate[i])[0].cpu().numpy()
        sub=edges[i].cpu().numpy()[np.ix_(ids,ids)]
        if len(ids):
            count,labels=connected_components(coo_matrix(sub),directed=False)
            patches=[ids[labels==j] for j in range(count)]
            patches.sort(key=lambda p:(-len(p),int(p[0])))
        else: patches=[]
        all_patches.append(patches)
    return DetectionResult(normals,score,1/knn_dist.mean(-1).clamp_min(eps),all_patches,knn)
