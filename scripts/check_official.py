"""Mandatory target-GPU readiness and feature/checkpoint checks."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from models import build_model
from utils import load_config, seed_all


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('Run inside an allocated GPU session')
    seed_all(42)
    config = load_config('configs/official.yaml')
    model = build_model(config).cuda()
    x = torch.randn(2, 1024, 3, device='cuda')
    y = torch.tensor([1, 2], device='cuda')
    model.train()
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    parameters = [p for p in model.parameters() if p.requires_grad]
    assert logits.shape == (2, 10) and torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
    model.eval()
    with torch.no_grad():
        features = model.forward_features(x)
        torch.testing.assert_close(model(x), model.model.cls_head_finetune(features))
    print('Official upstream class:', type(model.model).__name__)
    print('Feature shape:', tuple(features.shape))
    print('Official forward/backward and feature parity PASSED; GPU:', torch.cuda.get_device_name())


if __name__ == '__main__':
    main()
