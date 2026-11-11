from dataclasses import replace
import numpy as np
import pytest
import torch
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from attack.curves import CurveConfig,generate_curve
from attack.flat_detection import DetectionConfig,detect_flat_patches
from attack.flat_spiral import AttackConfig,insert_trigger,allocate_budget


def grid(side=12):
    u,v=np.meshgrid(np.linspace(-1,1,side),np.linspace(-1,1,side))
    return torch.tensor(np.column_stack((u.ravel(),v.ravel(),np.zeros(u.size))),dtype=torch.float64)

@pytest.mark.parametrize('kind',['spiral','rose','lissajous'])
def test_radius_and_reduction(kind):
    c=CurveConfig(kind=kind,turns=7,petals=7,freq_a=7,freq_b=6)
    r=generate_curve(.3,8,c)
    assert np.linalg.norm(r.points,axis=1).max()<=.3+1e-12
    assert r.metadata['reduced'] and r.metadata['features']<=2
    tiny=generate_curve(.3,1,c)
    assert not tiny.metadata['feature_budget_satisfied']


def test_orientation():
    cloud=grid()
    rotations=[np.eye(3),Rotation.from_euler('x',90,degrees=True).as_matrix(),Rotation.from_euler('xyz',[31,47,19],degrees=True).as_matrix()]
    batch=torch.stack([cloud@torch.tensor(r.T) for r in rotations])
    result=detect_flat_patches(batch,DetectionConfig(k_flat=12))
    counts=[len(p[0]) for p in result.patches]
    assert counts[0]>len(cloud)//2 and counts==[counts[0]]*3
    torch.testing.assert_close(result.planarity[0],result.planarity[2],atol=1e-10,rtol=1e-10)

@pytest.mark.parametrize('kind',['spiral','rose','lissajous'])
@pytest.mark.parametrize('mode',['single','multi','degenerate_fallback'])
@pytest.mark.parametrize('placement',['nearest','equal_arc'])
def test_insertion_hulls_count(kind,mode,placement):
    cloud=grid()
    if mode=='multi': cloud=torch.cat((grid(8)-torch.tensor([3.,0.,0.]),grid(8)+torch.tensor([3.,0.,1.])))
    c=AttackConfig(placement=placement,curve=CurveConfig(kind=kind),m=24,detection=DetectionConfig(k_flat=8),
                   min_patch_size=40 if mode=='single' else 10000,min_subpatch_size=6)
    if mode=='degenerate_fallback': c.detection.planarity_thresh=2
    r=insert_trigger(cloud,c)
    assert r.metadata['mode']==mode
    assert r.points.shape==cloud.shape and len(set(r.indices))==24
    if mode=='multi': assert len(r.patch_records)==2
    changed=torch.ones(len(cloud),dtype=torch.bool); changed[r.indices]=False
    torch.testing.assert_close(r.points[changed],cloud[changed])
    for rec in r.patch_records:
        hull=ConvexHull(rec['hull_uv']); eq=hull.equations
        for name in ('snapped_uv','curve_uv'):
            uv=np.asarray(rec[name]); assert (uv@eq[:,:2].T+eq[:,2]).max()<1e-8
        center=np.array(rec['center']); axes=np.array(rec['axes'])
        local=(r.points[rec['indices']].numpy()-center)@axes
        np.testing.assert_allclose(local,rec['snapped_uv'],atol=1e-8)


def test_rank_one_and_zero():
    for p in (torch.zeros(40,3),torch.stack((torch.linspace(-1,1,40),torch.zeros(40),torch.zeros(40)),1)):
        r=insert_trigger(p,AttackConfig(m=8))
        assert torch.isfinite(r.points).all() and len(r.indices)==8
        assert torch.count_nonzero(r.points[:,1:])==0


def test_allocation():
    assert sum(allocate_budget([6,12,30],32,3))==32
    assert all(a<=b for a,b in zip(allocate_budget([6,12,30],32,3),[6,12,30]))
    with pytest.raises(ValueError): allocate_budget([6,6],3,3)


def test_equal_arc_distances_and_closed_seam():
    from attack.flat_spiral import sample_equal_arc
    # Unequal input sample density; targets follow arc length, not vertex indices.
    path=np.array([[0.,0.],[.1,0.],[1.,0.],[1.,2.]])
    target,record=sample_equal_arc(path,7)
    np.testing.assert_allclose(np.diff(record['arc_positions']),.5)
    np.testing.assert_allclose(target,[[0,0],[.5,0],[1,0],[1,.5],[1,1],[1,1.5],[1,2]])
    square=np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.],[0.,0.]])
    target,record=sample_equal_arc(square,4)
    np.testing.assert_allclose(target,square[:-1])
    assert record['closed'] and len(np.unique(target,axis=0))==4
    degenerate,record=sample_equal_arc(np.zeros((20,2)),8)
    assert record['degenerate_curve'] and degenerate.shape==(8,2)


def test_unique_minimum_cost_assignment_and_coverage():
    from scipy.optimize import linear_sum_assignment
    from attack.flat_spiral import sample_equal_arc
    cloud=grid(18)
    config=AttackConfig(placement='equal_arc',m=64,detection=DetectionConfig(k_flat=8))
    r=insert_trigger(cloud,config)
    assert r.metadata['mode']=='single' and len(set(r.indices))==64
    rec=r.patch_records[0]
    target,spacing=sample_equal_arc(rec['curve_uv'],64)
    np.testing.assert_allclose(rec['snapped_uv'],target)
    np.testing.assert_allclose(np.diff(spacing['arc_positions']),spacing['total_arc_length']/63)
    xyz=np.asarray(rec['center'])+target@np.asarray(rec['axes']).T
    patch=cloud[rec['patch']].numpy()
    cost=((xyz[:,None]-patch[None])**2).sum(-1)
    rows,columns=linear_sum_assignment(cost)
    actual=((cloud[rec['indices']].numpy()-xyz)**2).sum()
    np.testing.assert_allclose(actual,cost[rows,columns].sum(),rtol=1e-12)
    np.testing.assert_allclose(r.points[rec['indices']].numpy(),xyz,atol=1e-12)
