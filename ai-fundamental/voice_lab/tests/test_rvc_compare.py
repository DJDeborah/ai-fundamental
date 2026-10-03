import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import voicelab.rvc_bridge as bridge
from voicelab.core import sha256
from voicelab.rvc_compare import feature_filelist, report, verify_corpus
from launch_rvc import allowance


def test_budget_counts_entire_job_and_reserves_shutdown():
    seconds, reserve = allowance(50, 20, 2)
    assert 2 + (seconds + 90) / 3600 * 20 < 50 - reserve
    seconds, _ = allowance(50, 1.88, 0)
    assert seconds == 3 * 3600 - 120
    for invalid in (float('nan'), 0, -1):
        with pytest.raises(ValueError):
            allowance(50, invalid, 0)


def test_instrumentation_keeps_loss_equations_and_fails_on_drift():
    original = (bridge.RVC / 'train/train.py').read_text(encoding='utf-8')
    patched = bridge.instrument(original)
    assert 'loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl' in patched
    assert patched.index('_lab_losses(locals())') < patched.index('if loss_mel > 75:')
    assert patched.index('global_step += 1') < patched.index('_lab_step(locals(), global_step)')
    with pytest.raises(ValueError):
        bridge.instrument(original.replace('        global_step += 1\n', ''))


def test_raw_loss_logging_and_nonfinite_gradients(tmp_path):
    recorder = bridge.Recorder(tmp_path, 10, 60, 10)
    ctx = {k: torch.tensor(99.) for k in ('loss_gen_all','loss_disc','loss_gen','loss_fm','loss_mel','loss_kl')}
    ctx.update(grad_norm_g=torch.tensor(1.),grad_norm_d=torch.tensor(1.))
    recorder.losses(ctx)
    assert recorder.last_losses['loss_mel'] == 99
    assert recorder.last_losses['loss_kl'] == 99
    ctx['grad_norm_g'] = torch.tensor(float('inf'))
    with pytest.raises(FloatingPointError):
        recorder.losses(ctx)


def test_missing_or_nonfinite_features_fail_closed(tmp_path):
    for folder in ('0_gt_wavs','3_feature768','2a_f0','2b-f0nsf'):
        (tmp_path/folder).mkdir()
    wav = tmp_path/'0_gt_wavs/a.wav'
    wav.write_bytes(b'fixture')
    with pytest.raises(ValueError, match='Missing feature'):
        feature_filelist(tmp_path)
    np.save(tmp_path/'3_feature768/a.npy',np.zeros((200,768),np.float32))
    np.save(tmp_path/'2a_f0/a.wav.npy',np.ones(400,np.int64))
    np.save(tmp_path/'2b-f0nsf/a.wav.npy',np.ones(400,np.float32)*220)
    assert feature_filelist(tmp_path).endswith('|0\n')
    np.save(tmp_path/'2b-f0nsf/a.wav.npy',np.ones(400,np.float32)*np.nan)
    with pytest.raises(ValueError, match='nonfinite'):
        feature_filelist(tmp_path)


def test_no_undeclared_training_files(tmp_path):
    (tmp_path/'train').mkdir()
    a = tmp_path/'train/a.wav'
    a.write_bytes(b'fixture')
    manifest = {'records':[{'path':'train/a.wav','split':'train','group':'song1','sha256':sha256(a)}]}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    verify_corpus(tmp_path)
    (tmp_path/'train/test_recording.wav').write_bytes(b'heldout')
    with pytest.raises(ValueError, match='undeclared'):
        verify_corpus(tmp_path)


def test_step_stop_exports_before_complete_and_preserves_status(tmp_path,monkeypatch):
    vendor = tmp_path/'vendor'
    (vendor/'assets/weights').mkdir(parents=True)
    monkeypatch.setattr(bridge,'RVC',vendor)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0)
    model = torch.nn.Linear(1,1)
    optim = torch.optim.AdamW(model.parameters())
    hps = SimpleNamespace(name='fixture',sample_rate='40k',if_f0=1,version='v2',model_dir=str(tmp_path))
    ctx = {'net_g':model,'net_d':model,'optim_g':optim,'optim_d':optim,'hps':hps,'epoch':1,'writers':[]}
    def savee(state,sr,f0,name,epoch,version,hps):
        torch.save(state,vendor/f'assets/weights/{name}.pth')
    def save_full(net,opt,lr,epoch,path):
        torch.save({'model':net.state_dict()},path)
    recorder = bridge.Recorder(tmp_path/'audit',2,60,2)
    recorder.scope = {'savee':savee,'utils':SimpleNamespace(save_checkpoint=save_full)}
    recorder.last_losses = {'loss_mel':90.0}
    recorder.step(ctx,1)
    with pytest.raises(bridge.TrainingFinished):
        recorder.step(ctx,2)
    result = json.loads((recorder.output/'summary.json').read_text())
    assert result['status'] == 'complete' and result['steps'] == 2
    assert Path(result['model']).is_file()
    assert (tmp_path/'G_2.pth').is_file() and (tmp_path/'D_2.pth').is_file()


def test_report_does_not_call_unequal_or_partial_runs_complete(tmp_path):
    (tmp_path/'corpus.json').write_text(json.dumps({'seconds':{'train':1,'val':1,'test':1},'grouping_confirmed':True}))
    for arm,steps in [('scratch',10),('finetune',20)]:
        (tmp_path/arm).mkdir()
        (tmp_path/arm/'summary.json').write_text(json.dumps({'status':'complete','steps':steps,'wall_s':1}))
    assert report(tmp_path)['equal_steps_complete'] is False
    assert report(tmp_path)['perceptual_winner'] is None
