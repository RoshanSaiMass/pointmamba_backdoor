import copy
import numpy as np
import torch
import pytest
from data.modelnet import normalize,off_points,prepare_training,prepare_test_attack
from utils import load_config,seed_all
from models import build_model
from defenses import statistical_outlier_removal,simple_random_sampling,gaussian_jitter,feature_filter
from visualize import geometry_metrics,save_comparison

@pytest.mark.parametrize('backbone',['pointmamba','dgcnn'])
def test_model_backward(backbone):
    c=load_config('configs/smoke.yaml'); c['model']['backbone']=backbone
    seed_all(1); model=build_model(c); x=torch.randn(3,64,3)
    y=model(x); assert y.shape==(3,10)
    torch.nn.functional.cross_entropy(y,torch.tensor([1,2,3])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_poison_semantics_cache(tmp_path):
    c=load_config('configs/smoke.yaml'); c['data']['cache_dir']=str(tmp_path)
    x=torch.randn(10,64,3); y=torch.arange(10)
    dirty=prepare_training(x,y,c)
    assert dirty['mask'].sum()==2
    assert (dirty['labels'][dirty['mask']]==0).all()
    assert not (y[dirty['mask']]==0).any()
    again=prepare_training(x,y,c); torch.testing.assert_close(again['points'],dirty['points'])
    c['poison']['clean_label']=True
    clean=prepare_training(x,y,c)
    assert clean['mask'].sum()==1 and torch.equal(clean['labels'],y)
    test=prepare_test_attack(x,y,c)
    assert len(test['selected'])==9 and not (y[test['selected']]==0).any()
    assert len(list(tmp_path.glob('*.pt')))==3


def test_defenses():
    x=torch.randn(2,32,3); x[:,-1]=100
    assert statistical_outlier_removal(x,k=8).abs().max()<100
    assert simple_random_sampling(x).shape==(2,24,3)
    assert (gaussian_jitter(x)-x).abs().max()<=.05001
    f=torch.cat((torch.zeros(18,4),torch.ones(2,4)*20))
    labels=torch.zeros(20,dtype=torch.long)
    for method in ('spectral','activation_clustering'):
        keep=feature_filter(f,labels,method)
        assert (~keep).sum()==2 and not keep[-2:].any()


def test_off_sampling(tmp_path):
    p=tmp_path/'triangle.off'; p.write_text('OFF\n3 1 0\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n')
    x=off_points(p,200,np.random.default_rng(1))
    assert x.shape==(200,3) and (x[:,:2]>=0).all() and (x.sum(1)<=1+1e-8).all()
    assert np.linalg.norm(normalize(x),axis=1).max()<=1.000001


def test_renderer(tmp_path):
    from PIL import Image
    x=np.random.default_rng(1).normal(size=(32,3))
    m=geometry_metrics(x,x); assert m==dict(chamfer=0.,hausdorff=0.)
    path=tmp_path/'pair.png'
    out=save_comparison(x,x,[],[0],{'test':True},path,prediction_before=1,prediction_after=0,target_class=0)
    assert out['attack_succeeded']
    with Image.open(path) as im: im.verify()


def test_local_mesh_dataset(tmp_path):
    from data.modelnet import MODELNET10,load_split
    c=load_config('configs/smoke.yaml'); c['data'].update(synthetic=False,root=str(tmp_path))
    for cls in MODELNET10:
        folder=tmp_path/cls/'train'; folder.mkdir(parents=True)
        (folder/'sample.off').write_text('OFF3 1 0\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n')
    a,y,names=load_split(c,'train'); b,z,_=load_split(c,'train')
    assert a.shape==(10,64,3) and y.tolist()==list(range(10)) and names==MODELNET10
    torch.testing.assert_close(a,b,rtol=0,atol=0)
