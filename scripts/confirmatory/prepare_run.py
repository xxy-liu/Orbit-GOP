"""Prepare a new isolated S3-S5 run without training or evaluating."""
import argparse,json,shutil
from .preflight import ROOT,OUT,sha
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',required=True,type=__import__('pathlib').Path)
    p.add_argument('--tiny-root',required=True,type=__import__('pathlib').Path)
    a=p.parse_args()
    if OUT.exists():raise FileExistsError('Use a new ORBIT_RUN_DIR')
    if not a.data_root.is_dir() or not a.tiny_root.is_dir():raise FileNotFoundError('Supply existing authorized data directories')
    OUT.mkdir(parents=True)
    shutil.copytree(ROOT/'configs/confirmatory',OUT/'config')
    shutil.copytree(ROOT/'splits/confirmatory',OUT/'splits')
    (OUT/'manifests').mkdir()
    cfg=json.loads((OUT/'config/confirmatory_locked.yaml').read_text());cfg['data_root']=str(a.data_root.resolve())
    (OUT/'config/confirmatory_locked.yaml').write_text(json.dumps(cfg,indent=2))
    manifest=json.loads((ROOT/'provenance/confirmatory/manifests/EVALUATION_MANIFEST.json').read_text())
    manifest['ID']['root']=str(a.data_root.resolve()/'CIFAR-10')
    manifest['CIFAR100']['root']=str(a.data_root.resolve()/'CIFAR-100')
    manifest['Tiny']['root']=str(a.tiny_root.resolve())
    (OUT/'manifests/EVALUATION_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
    inputs={x.relative_to(OUT).as_posix():sha(x) for x in OUT.rglob('*') if x.is_file()}
    code={x.relative_to(ROOT).as_posix():sha(x) for folder in ('models','training','orbit_gop','baselines','scripts') for x in (ROOT/folder).rglob('*.py')}
    (OUT/'RUN_INPUT_LOCK.json').write_text(json.dumps(dict(scope='NEW_LOCAL_REPRODUCTION_NOT_HISTORICAL_FREEZE',inputs=inputs,code=code),indent=2))
    print('Prepared a new local reproduction input lock; no historical confirmation is asserted.')
if __name__=='__main__':main()
