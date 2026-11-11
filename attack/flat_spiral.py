"""Hull-safe in-place insertion, retaining N and explicit capacity accounting."""
from dataclasses import dataclass, field, asdict
import numpy as np
import torch
from scipy.spatial import ConvexHull, QhullError
from .curves import CurveConfig, generate_curve
from .flat_detection import DetectionConfig, detect_flat_patches

@dataclass
class AttackConfig:
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    curve: CurveConfig = field(default_factory=CurveConfig)
    m: int = 32
    min_patch_size: int = 40
    num_subpatches: int = 8
    min_subpatch_size: int = 6
    min_points_per_subpatch: int = 3
    scale_fraction: float = .8
    random_position: bool = False

@dataclass
class AttackResult:
    points: torch.Tensor
    indices: list
    patches: list
    metadata: dict
    patch_records: list


def allocate_budget(sizes, budget, floor):
    if not sizes: return []
    budget=min(budget,sum(sizes))
    if budget < floor*len(sizes):
        raise ValueError('select fewer patches before allocating minimum budgets')
    allocations=np.minimum(sizes,floor).astype(int)
    while allocations.sum() < budget:
        available=np.array(sizes)-allocations
        priorities=np.array(sizes)/(allocations+1)
        priorities[available==0]=-1
        allocations[np.argmax(priorities)]+=1
    return allocations.tolist()


def local_frame(points):
    center=points.mean(0)
    _,_,vh=np.linalg.svd(points-center,full_matrices=True)
    # Canonical signs improve deterministic replay; normal sign has no semantics.
    for i in range(3):
        j=np.argmax(np.abs(vh[i]))
        if vh[i,j]<0: vh[i]*=-1
    axes=vh[:2].T
    return center,axes,(points-center)@axes


def clamp_hull(curve, uv):
    """Radial intersection with convex halfplanes, centered at the patch mean."""
    try:
        hull=ConvexHull(uv)
        A,b=hull.equations[:,:2],hull.equations[:,2]
        dots=curve@A.T
        limits=np.where(dots>1e-15, -b/np.maximum(dots,1e-15),np.inf)
        t=np.minimum(1,np.min(limits,axis=1)).clip(0,1)
        return curve*t[:,None]*(1-1e-10)
    except QhullError:
        # Rank-one hull: project to the actual line segment; rank-zero -> mean.
        _,s,vh=np.linalg.svd(uv,full_matrices=False)
        if len(s)==0 or s[0]<1e-12: return np.zeros_like(curve)
        direction=vh[0]
        coord=uv@direction
        return np.clip(curve@direction,coord.min(),coord.max())[:,None]*direction

@torch.no_grad()
def insert_trigger(points, config=None, seed=0):
    c=config or AttackConfig()
    if points.ndim!=2 or points.shape[1]!=3 or c.m<1 or c.m>len(points):
        raise ValueError('expected [N,3] and 1 <= m <= N')
    if not 0 < c.scale_fraction <= 1 or c.min_points_per_subpatch<1 or c.num_subpatches<1:
        raise ValueError('invalid scale or subpatch configuration')
    detection=detect_flat_patches(points,c.detection)
    patches=detection.patches[0]
    score=(detection.planarity[0]/detection.density[0]).cpu().numpy()
    rng=np.random.default_rng(seed)
    if c.random_position:
        chosen=[rng.choice(len(points),size=max(c.m,c.min_subpatch_size),replace=False).tolist()]
        mode='random_position'
    elif patches and len(patches[0])>=max(c.min_patch_size,c.m):
        chosen=[patches[0]]; mode='single'
    else:
        chosen=[p for p in patches if len(p)>=max(c.min_subpatch_size,c.min_points_per_subpatch)][:c.num_subpatches]
        chosen=chosen[:max(1,c.m//c.min_points_per_subpatch)]
        mode='multi'
        if not chosen:
            chosen=[np.argsort(-score,kind='stable')[:max(c.m,c.min_subpatch_size)].tolist()]
            mode='degenerate_fallback'
    # If capacity is insufficient, use the documented global fallback to preserve M.
    capacity_fallback=sum(map(len,chosen))<c.m
    if capacity_fallback:
        chosen=[np.argsort(-score,kind='stable')[:max(c.m,c.min_subpatch_size)].tolist()]
        mode='degenerate_fallback'
    budgets=allocate_budget(list(map(len,chosen)),c.m,min(c.min_points_per_subpatch,c.m))
    original=points.detach().cpu().double().numpy()
    poisoned=original.copy(); indices=[]; records=[]
    for patch,budget in zip(chosen,budgets):
        patch=np.asarray(patch,dtype=int)
        center,axes,uv=local_frame(original[patch])
        radius=float(np.linalg.norm(uv,axis=1).max())*c.scale_fraction
        curve=generate_curve(radius,budget,c.curve)
        bounded=clamp_hull(curve.points,uv)
        order=rng.permutation(len(patch)) if c.random_position else np.argsort(-score[patch],kind='stable')
        selected=patch[order[:budget]]
        local=(original[selected]-center)@axes
        nearest=np.argmin(((local[:,None,:]-bounded[None,:,:])**2).sum(-1),axis=1)
        snapped=bounded[nearest]
        poisoned[selected]=center+snapped@axes.T
        indices.extend(selected.tolist())
        records.append(dict(indices=selected.tolist(),patch=patch.tolist(),center=center.tolist(),
            axes=axes.tolist(),hull_uv=uv.tolist(),curve_uv=bounded.tolist(),snapped_uv=snapped.tolist(),
            curve=curve.metadata))
    meta=dict(mode=mode,num_subpatches=len(chosen),requested_m=c.m,actual_m=len(indices),
              capacity_fallback=capacity_fallback,reduced=any(r['curve']['reduced'] for r in records),
              config=asdict(c))
    return AttackResult(torch.as_tensor(poisoned,dtype=points.dtype,device=points.device),indices,
                        [list(p) for p in chosen],meta,records)
