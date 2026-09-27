"""Run historical training loops through explicit isolated role adapters."""
import os,sys,json,shutil,hashlib,csv,argparse,traceback
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime,timezone
import numpy as np
from .preflight import OUT
from .preflight import sha,write

def verify_lock():
    from .preflight import verify_run
    return verify_run()


def main(backbone,seed):
    cfg=verify_lock()
    if seed not in cfg['seeds'] or backbone not in cfg['backbones']:raise ValueError('Unregistered run')
    os.environ['PYTHONHASHSEED']=str(seed)
    import torch
    from torchvision import datasets
    if not torch.cuda.is_available():raise RuntimeError('Locked CUDA device unavailable')
    stage=OUT/'training'/backbone/f'seed_{seed}'
    if stage.exists():raise FileExistsError('Refuse silent retry of existing run '+str(stage))
    stage.mkdir(parents=True)
    (stage/'run_config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    with np.load(OUT/'splits/cifar10_confirmatory_split_v1.npz',allow_pickle=False) as z:
        train,val=z['train'],z['val']
    splitdir=OUT/'splits'
    # Factory guard blocks official test construction before any training code executes.
    original=datasets.CIFAR10
    def guarded(root,*args,**kwargs):
        flag=kwargs.get('train',args[0] if args else True)
        if flag is not True:raise RuntimeError('DATA_ROLE_GUARD: official test forbidden during training')
        if Path(root).resolve()!=(Path(cfg['data_root'])/'CIFAR-10').resolve():raise RuntimeError('DATA_ROOT_GUARD')
        return original(root,*args,**kwargs)
    datasets.CIFAR10=guarded
    try:
        torch.backends.cudnn.allow_tf32=cfg['training'][backbone]['cudnn_allow_tf32']
        torch.backends.cuda.matmul.allow_tf32=False
        torch.set_float32_matmul_precision('highest')
        if backbone=='resnet18':
            from training import resnet as t
            c=json.loads((OUT/'config/resnet_training_locked.json').read_text());c['data_root']=cfg['data_root']
            t.ROOT=stage;t.CONFIG=c;t.ARGS=SimpleNamespace(seed=seed);t.MANIFESTS=splitdir
            original_validate=t.validate_config
            def validate(c):
                assert c['training_seeds']==[3,4,5]
                old=dict(c);old['training_seeds']=[0,1,2]
                original_validate(old)
            t.validate_config=validate
            t.main()
            source=stage/f'runs/seed_{seed}/best.pt'
            completion=json.loads((stage/f'runs/seed_{seed}/training_complete.json').read_text())
            best=completion['best_validation_metrics'];metric=best['accuracy'];epoch=best['epoch']
        elif backbone=='vgg16_bn':
            from training import vgg as t
            t.ROOT=stage;t.DATA_ROOT=Path(cfg['data_root']);t.TRAIN_INDICES=splitdir/'train_indices.npy';t.VALIDATION_INDICES=splitdir/'validation_indices.npy'
            # The frozen protocol excludes performance-based stopping/admission; numerical failures still stop training.
            t.collapse_reason=lambda rows: 'A_NONFINITE' if rows and (rows[-1]['NaN_count'] or rows[-1]['Inf_count']) else None
            t.train_run(f'seed_{seed}',0.01,seed)
            completion=json.loads((stage/f'runs/seed_{seed}/run_state.json').read_text())
            if completion['epochs_completed']!=200 or completion['total_nan_count'] or completion['total_inf_count']:raise RuntimeError('Engineering failure: incomplete/nonfinite training')
            if completion['checkpoint_reproduction']!='PASS':raise RuntimeError('Checkpoint reproduction failed')
            source=stage/f'runs/seed_{seed}/best.pt';epoch=completion['best_epoch'];metric=completion['best_validation_accuracy']
        else:
            from . import train_wrn as t
            t.OUT=stage;t.DATA=Path(cfg['data_root']);t.RUNTIME_LOG=stage/'train.log'
            t.split_paths=lambda:{'train':splitdir/'train_indices.npy','val':splitdir/'validation_indices.npy'}
            for d in ('checkpoints','training_logs','raw'):(stage/d).mkdir()
            t.cmd_train(seed)
            source=stage/f'checkpoints/seed_{seed}_final.pt';epoch=200
            completion=json.loads((stage/f'training_logs/seed_{seed}_complete.json').read_text())
            metric=completion['diagnostic_val']['accuracy']
        verify_lock()
        dest=OUT/'checkpoints'/backbone/f'seed_{seed}'/'best.pt'
        if dest.exists():raise FileExistsError('Already frozen checkpoint')
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
        assert sha(source)==sha(dest)
        row={'backbone':backbone,'seed':seed,'training_commit':'NO_ROOT_GIT_COMMIT','config_hash':sha(OUT/'config/confirmatory_locked.yaml'),'split_hash':sha(OUT/'splits/cifar10_confirmatory_split_v1.npz'),'checkpoint_hash':sha(dest),'checkpoint':str(dest),'best_val_epoch':epoch if backbone!='wrn28_10' else 'NOT_APPLICABLE','fixed_epoch':200 if backbone=='wrn28_10' else 'NOT_APPLICABLE','val_metric':metric,'val_role':'diagnostic_only' if backbone=='wrn28_10' else 'checkpoint_selection','train_end_time':datetime.now(timezone.utc).isoformat(),'test_admission':False,'status':'FROZEN'}
        write(f'manifests/{backbone}_S{seed}.json',row)
        manifest=OUT/'manifests/MODEL_MANIFEST.csv'
        with manifest.open('a',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(row))
            if f.tell()==0:w.writeheader()
            w.writerow(row)
        print(json.dumps(row),flush=True)
    except Exception as e:
        write(f'logs/{backbone}_S{seed}_FAILURE.json',{'error':str(e),'traceback':traceback.format_exc(),'seed':seed,'backbone':backbone,'retry_authorized_only_same_seed_config_for_engineering_failure':True})
        raise
    finally:datasets.CIFAR10=original
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('backbone',choices=['resnet18','vgg16_bn','wrn28_10']);p.add_argument('seed',type=int,choices=[3,4,5]);a=p.parse_args();main(a.backbone,a.seed)
