import copy
import json
import torch
from utils import load_config
from ablation import make_plan
from train import run
from eval import run as evaluate_run
from report import generate_report


def test_sweep_coverage():
    c=load_config('configs/default.yaml'); plan=make_plan(c,'full',True,[1,2])
    configs=[cc for _,cc in plan]
    assert {cc['seed'] for cc in configs}=={1,2}
    for kind in ('spiral','rose','lissajous'):
        curves=[cc for cc in configs if cc['attack']['curve']['kind']==kind]
        assert {cc['poison']['rate'] for cc in curves}>={.001,.005,.01,.03,.05,.1}
        assert any(cc['poison']['clean_label'] for cc in curves)
        assert any(cc['attack']['random_position'] for cc in curves)
        assert any(cc['model']['backbone']=='dgcnn' for cc in curves)
    assert {cc['attack']['curve']['petals'] for cc in configs}>={3,5,7}
    assert {(cc['attack']['curve']['freq_a'],cc['attack']['curve']['freq_b']) for cc in configs}>={(3,2),(5,4),(7,6)}


def test_training_checkpoint_eval_report(tmp_path):
    c=load_config('configs/smoke.yaml'); c['train']['epochs']=1
    c['data']['cache_dir']=str(tmp_path/'cache')
    folder=tmp_path/'experiment'; c['output']=str(folder)
    model,summary=run(c)
    assert summary['epochs']==1 and (folder/'last.pt').exists()
    result=evaluate_run(folder/'last.pt',feature_defenses=True)
    assert len(result['metrics'])==6
    assert all(r['status'].startswith('evaluated') for r in result['metrics'])
    assert (folder/'history_per_class.csv').exists()
    report=generate_report(tmp_path)
    assert len(report)==1 and (tmp_path/'results.md').exists()
    assert report[0]['synthetic'] and report[0]['asr_activation_clustering'] is not None
    with torch.no_grad():
        x=torch.randn(2,64,3); model.eval(); a=model(x)
        checkpoint=torch.load(folder/'last.pt',weights_only=False)
        model.load_state_dict(checkpoint['model']); b=model(x)
        torch.testing.assert_close(a,b)


def test_resume_matches_uninterrupted(tmp_path,monkeypatch):
    import train as trainer
    c=load_config('configs/smoke.yaml'); c['data']['cache_dir']=str(tmp_path/'cache')
    c['output']=str(tmp_path/'full')
    _,_=trainer.run(c)
    full=torch.load(tmp_path/'full/last.pt',weights_only=False)
    c['output']=str(tmp_path/'resumed')
    original_save=trainer.atomic_save
    def stop_after_first(value,path):
        original_save(value,path)
        if value['epoch']==1: raise InterruptedError('simulated interruption after atomic checkpoint')
    monkeypatch.setattr(trainer,'atomic_save',stop_after_first)
    import pytest
    with pytest.raises(InterruptedError): trainer.run(c)
    monkeypatch.setattr(trainer,'atomic_save',original_save)
    trainer.run(c,resume=True)
    resumed=torch.load(tmp_path/'resumed/last.pt',weights_only=False)
    assert full['epoch']==resumed['epoch']==2
    for name in full['model']:
        torch.testing.assert_close(full['model'][name],resumed['model'][name],rtol=0,atol=0)
