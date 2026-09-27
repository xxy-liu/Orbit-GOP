"""Portable run I/O. Historical pre-run approval is retained only as provenance."""
import os,json,csv,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(os.environ.get('ORBIT_RUN_DIR',str(ROOT/'outputs/confirmatory'))).resolve()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(name,value):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def csvwrite(name,fields,rows):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def verify_run():
    lock=json.loads((OUT/'RUN_INPUT_LOCK.json').read_text())
    if lock['scope']!='NEW_LOCAL_REPRODUCTION_NOT_HISTORICAL_FREEZE':raise ValueError('Invalid run scope')
    for relative,digest in lock['inputs'].items():
        if sha(OUT/relative)!=digest:raise RuntimeError('Run input hash mismatch: '+relative)
    for relative,digest in lock['code'].items():
        if sha(ROOT/relative)!=digest:raise RuntimeError('Candidate code changed: '+relative)
    cfg=json.loads((OUT/'config/confirmatory_locked.yaml').read_text())
    if cfg['seeds']!=[3,4,5]:raise ValueError('Only confirmatory seeds are permitted')
    return cfg
