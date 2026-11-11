import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation
from attack.flat_detection import DetectionConfig,detect_flat_patches


def grid():
    u,v=np.meshgrid(np.linspace(-1,1,20),np.linspace(-1,1,20))
    return np.column_stack((u.ravel(),v.ravel(),np.zeros(u.size)))


def test_no_world_axis_bias():
    xyz=grid()
    rotations=[np.eye(3),Rotation.from_euler('x',90,degrees=True).as_matrix(),
               Rotation.from_euler('xyz',[31,47,19],degrees=True).as_matrix()]
    batch=torch.tensor(np.stack([xyz@r.T for r in rotations]),dtype=torch.float64)
    result=detect_flat_patches(batch)
    counts=[len(p[0]) for p in result.patches]
    assert counts[0]>200
    assert counts==[counts[0]]*3
    for i in (1,2):
        torch.testing.assert_close(result.planarity[0],result.planarity[i],atol=1e-10,rtol=1e-10)


def test_collinear_not_flat():
    line=torch.zeros(40,3,dtype=torch.float64)
    line[:,0]=torch.arange(40)
    result=detect_flat_patches(line)
    assert result.patches==[[]]


def test_translation_and_normal_sign():
    cloud=torch.tensor(grid(),dtype=torch.float64)
    result=detect_flat_patches(torch.stack((cloud,cloud+torch.tensor([9.,-3.,6.]))))
    assert len(result.patches[0][0])>200
    assert len(result.patches[0][0])==len(result.patches[1][0])
    assert torch.abs(result.normals[:,:,2]).min()>.999


def test_validation():
    with pytest.raises(ValueError,match='k_flat'):
        detect_flat_patches(torch.zeros(10,3))
