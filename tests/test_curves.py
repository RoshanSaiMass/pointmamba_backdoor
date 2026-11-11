import numpy as np
import pytest
from attack.curves import CurveConfig, generate_curve, spiral, rose, lissajous

@pytest.mark.parametrize('fn',[spiral,rose,lissajous])
@pytest.mark.parametrize('radius',[0.,0.01,1.])
def test_radius(fn,radius):
    result=fn(radius,32)
    assert result.points.shape[1]==2
    assert np.isfinite(result.points).all()
    assert (np.linalg.norm(result.points,axis=1)<=radius+1e-12).all()

@pytest.mark.parametrize('kind',['spiral','rose','lissajous'])
def test_budget_reduction(kind):
    result=generate_curve(1,4,CurveConfig(curve_type=kind))
    assert result.metadata['reduced']
    assert not result.metadata['minimum_unmet']

@pytest.mark.parametrize('kind',['spiral','rose','lissajous'])
def test_impossible_budget_explicit(kind):
    with pytest.raises(ValueError,match='even one feature'):
        generate_curve(1,3,CurveConfig(curve_type=kind))
    result=generate_curve(1,3,CurveConfig(curve_type=kind,insufficient_budget='simplest'))
    assert result.metadata['minimum_unmet']


def test_spiral_endpoint_and_determinism():
    first=spiral(.7,32)
    assert np.linalg.norm(first.points[0])==0
    assert np.linalg.norm(first.points[-1])==pytest.approx(.7)
    np.testing.assert_array_equal(first.points,spiral(.7,32).points)


def test_non_coprime_rejected():
    with pytest.raises(ValueError,match='coprime'):
        lissajous(1,32,freq_a=4,freq_b=2)


def test_clipped_rose_all_lobes_present():
    points=rose(1,32,k_petals=5).points
    angle=np.arctan2(points[:,1],points[:,0])
    for petal in range(5):
        target=2*np.pi*petal/5
        delta=np.angle(np.exp(1j*(angle-target)))
        assert np.any((np.abs(delta)<.03)&(np.linalg.norm(points,axis=1)>.98))
