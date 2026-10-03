"""Source + explicitly authorized singing data. No SSH key or pretrained weights."""
import json
import zipfile
from pathlib import Path

from voicelab.core import ROOT, sha256, write_json
from voicelab.rvc_assets import RVC


def main():
    vendor = [p for p in RVC.rglob('*') if p.is_file() and
              not any(x in {'.git','__pycache__','logs'} for x in p.relative_to(RVC).parts) and
              p.suffix.lower() not in {'.pt','.pth','.bin','.npy','.pyc','.zip','.partial'}]
    # Only source and small UI/documentation assets already in the pinned checkout.
    snapshot = {'sha':'81eed5e8f68b6bed1789f682fe78cdd324495afc',
                'files':{p.relative_to(RVC).as_posix():sha256(p) for p in sorted(vendor)}}
    write_json(ROOT/'RVC_SNAPSHOT.json',snapshot)
    own = [ROOT/name for name in ('RVC_START_HERE.md','RVC_EXPERIMENT_RECORD.md','requirements-rvc.txt',
           'setup_rvc.sh','cloud_rvc_pipeline.sh','launch_rvc.py','RVC_SNAPSHOT.json','rvc_listen.ipynb','build_rvc_bundle.py')]
    own += list((ROOT/'voicelab').glob('*.py'))
    own += [ROOT/'tests/test_rvc_compare.py']
    own += [p for p in (ROOT/'data/self_rvc_v1').rglob('*') if p.is_file()]
    own += [ROOT/'runs/self_recordings_20261002/rvc_local_validation.json']
    files = sorted(set(vendor+own))
    checks = {'files':{p.relative_to(ROOT).as_posix():sha256(p) for p in files},
              'personal_recordings_included':True,'pretrained_weights_included':False,
              'ssh_credentials_included':False,'gpu_training_verified':False,
              'purpose':'Own vocal recordings uploaded to existing AutoDL instance for scratch/fine-tuning comparison'}
    target = ROOT.parent/'dist/voice_self_rvc_compare.zip'
    target.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f,f.relative_to(ROOT).as_posix())
        z.writestr('BUNDLE_MANIFEST.json',json.dumps(checks,ensure_ascii=False,indent=2))
    result = {'path':str(target),'bytes':target.stat().st_size,'sha256':sha256(target),'files':len(files)}
    print(json.dumps(result,ensure_ascii=False))


if __name__ == '__main__':
    main()
