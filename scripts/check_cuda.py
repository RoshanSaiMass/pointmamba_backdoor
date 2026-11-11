"""Run on the target machine before spending GPU time on experiments."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from models import build_model
from utils import load_config,seed_all
from attack.flat_spiral import insert_trigger
from utils import attack_config


def main():
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    seed_all(1); c=load_config('configs/default.yaml')
    x=torch.randn(2,1024,3,device='cuda'); labels=torch.tensor([1,2],device='cuda')
    for backbone in ('pointmamba','dgcnn'):
        c['model']['backbone']=backbone
        model=build_model(c).cuda().train(); logits=model(x)
        loss=torch.nn.functional.cross_entropy(logits,labels); loss.backward()
        if not torch.isfinite(loss) or not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError(f'{backbone}: nonfinite loss or gradients')
        print(backbone,'forward/backward passed; loss',float(loss.detach()))
        del model
    result=insert_trigger(x[0],attack_config(c))
    assert result.points.shape==x[0].shape and len(result.indices)==c['attack']['m']
    torch.cuda.synchronize()
    print('CUDA insertion passed; device:',torch.cuda.get_device_name(0))
    print('Peak allocated MB:',torch.cuda.max_memory_allocated()/2**20)

if __name__=='__main__': main()
