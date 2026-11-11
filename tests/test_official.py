"""CPU integration tests use an explicitly substituted mixer, not CUDA Mamba."""
import ast
import sys
import types
import numpy as np
import pytest
import torch
from torch import nn
from models.official_pointmamba import ROOT, classifier_ast, verify_source, torch_fps, TorchKNN


def test_source_and_architecture():
    record=verify_source()
    assert record['commit']=='80858e096e33ff7470640fe1f2132e5c1e667074'
    original=ast.parse((ROOT/'point_mamba_scan.py').read_text())
    selected={n.name:n for n in classifier_ast().body}
    for node in original.body:
        if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in selected:
            node.decorator_list=[] if node.name=='PointMambaScan' else node.decorator_list
            assert ast.dump(node)==ast.dump(selected[node.name])


def test_grouping_backends():
    x=torch.tensor([[[0.,0.,0.],[1.,0.,0.],[2.,0.,0.],[3.,0.,0.]]])
    selected=torch_fps(x,3)
    torch.testing.assert_close(selected,x[:,[0,3,1]])
    distances,indices=TorchKNN(2)(x,x[:,[0,3]])
    assert indices.tolist()==[[[0,1],[3,2]]]
    torch.testing.assert_close(distances,torch.tensor([[[0.,1.],[0.,1.]]]))


def test_upstream_forward_and_features_with_test_doubles(monkeypatch):
    import yaml
    # Import substitutes are confined to this CPU test. Production never substitutes Mamba.
    class Mixer(nn.Module):
        def __init__(self,d_model,**kwargs):
            super().__init__(); self.proj=nn.Linear(d_model,d_model)
        def forward(self,x,inference_params=None): return self.proj(x)
    def module(name,**attrs):
        obj=types.ModuleType(name)
        for key,value in attrs.items(): setattr(obj,key,value)
        monkeypatch.setitem(sys.modules,name,obj)
        return obj
    module('mamba_ssm',__version__='TEST_DOUBLE')
    module('mamba_ssm.modules'); module('mamba_ssm.modules.mamba_simple',Mamba=Mixer)
    module('timm'); module('timm.models')
    module('timm.models.layers',trunc_normal_=nn.init.trunc_normal_,DropPath=lambda prob:nn.Identity())
    # addict is only a convenience dictionary wrapper; use real installed package.
    from models.official_pointmamba import OfficialPointMamba
    original=yaml.safe_load
    def small_config(stream):
        result=original(stream)
        result['model'].update(trans_dim=24,encoder_dims=24,depth=1,num_group=8,group_size=8,drop_path=0.)
        return result
    monkeypatch.setattr(yaml,'safe_load',small_config)
    model=OfficialPointMamba(10).eval()
    x=torch.randn(2,64,3)
    features=model.forward_features(x)
    assert features.shape==(2,24)
    torch.testing.assert_close(model(x),model.model.cls_head_finetune(features))
    model.train(); model(x).sum().backward()
    assert all(p.grad is not None for p in model.parameters())
    # Remove dependency-bearing upstream module so later tests cannot retain doubles.
    monkeypatch.delitem(sys.modules,'_pinned_official_pointmamba.block_scan')
