"""Official PointMambaScan source adapter, with explicit grouping backend.

Classifier definitions are compiled verbatim from a pinned, hash-checked upstream
snapshot. Only unrelated pretraining classes/imports are excluded. Official
Encoder, Group, Hilbert serialization, OrderScale, blocks and head are retained.
The default torch grouping replaces extension kernels, NOT the architecture.
This is training from scratch with Mamba 2.2.4, not paper-protocol reproduction.
"""
import ast
import hashlib
import importlib
import importlib.util
import json
import math
import random
import sys
import types
from functools import partial
from pathlib import Path
from typing import Optional, Union
import numpy as np
import torch
from torch import nn
import yaml

ROOT = Path(__file__).resolve().parents[1] / 'third_party/pointmamba_official'
DEFINITIONS = {'Encoder', 'Group', '_init_weights', 'serialization', 'create_block',
               'MixerModel', 'PointMambaScan', 'init_OrderScale', 'apply_OrderScale',
               'serialization_func'}


def verify_source():
    record = json.loads((ROOT / 'SOURCE.json').read_text())
    for name, expected in record['files'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Official source checksum mismatch: {name}')
    return record


def classifier_ast():
    tree = ast.parse((ROOT / 'point_mamba_scan.py').read_text())
    definitions = [node for node in tree.body
                   if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in DEFINITIONS]
    if {node.name for node in definitions} != DEFINITIONS:
        raise RuntimeError('Pinned upstream classifier definitions are incomplete')
    for node in definitions:
        # Registration has no effect on architecture; our factory handles selection.
        if isinstance(node, ast.ClassDef) and node.name == 'PointMambaScan':
            node.decorator_list = []
    return ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[]))


@torch.no_grad()
def torch_fps(x, count):
    """FPS with initial index zero, matching pointnet2's kernel convention."""
    if count > x.shape[1]:
        raise ValueError('num_group exceeds input point count')
    b, n, _ = x.shape
    idx = torch.zeros(b, count, dtype=torch.long, device=x.device)
    distances = torch.full((b, n), 1e10, device=x.device)
    farthest = torch.zeros(b, dtype=torch.long, device=x.device)
    batch = torch.arange(b, device=x.device)
    for i in range(count):
        idx[:, i] = farthest
        d = (x - x[batch, farthest][:, None]).square().sum(-1)
        distances = torch.minimum(distances, d)
        farthest = distances.argmax(-1)
    return x[batch[:, None], idx]


class TorchKNN:
    def __init__(self, k, transpose_mode=True):
        if not transpose_mode:
            raise ValueError('Only upstream transpose_mode=True is supported')
        self.k = k

    @torch.no_grad()
    def __call__(self, reference, query):
        if self.k > reference.shape[1]:
            raise ValueError('group_size exceeds input point count')
        return torch.cdist(query, reference).topk(self.k, largest=False, sorted=True)


def upstream_class(grouping_backend='torch'):
    verify_source()
    from mamba_ssm.modules.mamba_simple import Mamba
    from timm.models.layers import trunc_normal_
    package = '_pinned_official_pointmamba'
    if package not in sys.modules:
        namespace = types.ModuleType(package)
        namespace.__path__ = [str(ROOT)]
        namespace.__spec__ = importlib.util.spec_from_loader(package, loader=None, is_package=True)
        sys.modules[package] = namespace
    Point = importlib.import_module(package + '.serialization').Point
    Block = importlib.import_module(package + '.block_scan').Block
    if grouping_backend == 'torch':
        fps, knn = torch_fps, TorchKNN
    elif grouping_backend == 'native':
        from pointnet2_ops import pointnet2_utils
        from knn_cuda import KNN
        def fps(x, count):
            indices = pointnet2_utils.furthest_point_sample(x.contiguous(), count)
            return pointnet2_utils.gather_operation(x.transpose(1, 2).contiguous(), indices).transpose(1, 2).contiguous()
        knn = KNN
    else:
        raise ValueError('grouping_backend must be torch or native')
    env = dict(torch=torch, nn=nn, F=torch.nn.functional, np=np, math=math,
               random=random, partial=partial, Optional=Optional, Union=Union,
               Mamba=Mamba, Point=Point, Block=Block, KNN=knn,
               misc=types.SimpleNamespace(fps=fps), trunc_normal_=trunc_normal_,
               print_log=lambda message, **kwargs: print(message, flush=True))
    exec(compile(classifier_ast(), str(ROOT / 'point_mamba_scan.py'), 'exec'), env)
    return env['PointMambaScan']


class OfficialPointMamba(nn.Module):
    def __init__(self, num_classes, grouping_backend='torch'):
        super().__init__()
        from types import SimpleNamespace
        settings = yaml.safe_load((ROOT / 'finetune_modelnet.yaml').read_text())['model']
        settings['cls_dim'] = num_classes
        self.provenance = dict(**verify_source(), grouping_backend=grouping_backend,
                               mamba_version=importlib.import_module('mamba_ssm').__version__)
        self.model = upstream_class(grouping_backend)(SimpleNamespace(**settings))
        self.model.load_model_from_ckpt(None)
        print(f"OFFICIAL PointMambaScan: commit={self.provenance['commit']}, "
              f"dim={settings['trans_dim']}, depth={settings['depth']}, "
              f"grouping={grouping_backend}, training=from_scratch", flush=True)

    def forward(self, x):
        return self.model(x.contiguous())

    def forward_features(self, x):
        # Capture the EXACT tensor consumed by the upstream classification head.
        features = []
        hook = self.model.cls_head_finetune.register_forward_pre_hook(
            lambda module, args: features.append(args[0]))
        try:
            self.forward(x)
        finally:
            hook.remove()
        return features[0]
