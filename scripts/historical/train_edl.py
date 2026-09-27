"""Run the extracted original training loop with portable input/output paths."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.historical.common import ROOT,sha256,verify_splits

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backbone',required=True,choices=['resnet18','vgg16_bn','wrn28_10'])
    p.add_argument('--seed',type=int,required=True,choices=[0,1,2])
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--initial-checkpoint',type=Path,help='Required original frozen initialization for VGG S0')
    args=p.parse_args();verify_splits()
    if args.backbone=='vgg16_bn' and args.seed==0:
        init=json.loads((ROOT/'splits/historical/vgg_initialization.json').read_text())
        if args.initial_checkpoint is None or sha256(args.initial_checkpoint)!=init['sha256']:
            p.error('VGG S0 requires the recorded frozen initial checkpoint and matching SHA256')
    if not (args.data_root/'CIFAR-10/cifar-10-batches-py').is_dir():p.error('CIFAR-10 Python files are missing under --data-root/CIFAR-10')
    import torch
    if not torch.cuda.is_available():p.error('The archived formal training path requires CUDA')
    out=args.output_dir.resolve()
    if out.exists():p.error('Use a new output directory; training never resumes or overwrites an existing run')
    out.mkdir(parents=True)
    config=json.loads((ROOT/'configs/historical/resnet_training.json').read_text())
    config['data_root']=str(args.data_root.resolve())
    if args.backbone!='resnet18':
        config['experiment_name']=args.backbone+'-EDL'
        config['model']='VGG16-BN with GAP EDL head' if args.backbone=='vgg16_bn' else 'WRN-28-10 EDL'
        config['feature_dim']=512 if args.backbone=='vgg16_bn' else 640
        config['learning_rate']=0.01 if args.backbone=='vgg16_bn' else 0.1
        config['checkpoint_selection']='epoch_200' if args.backbone=='wrn28_10' else config['checkpoint_selection']
    metadata=dict(backbone=args.backbone,seed=args.seed,scope='new training run from extracted source; no historical approval artifacts',initial_lr=0.01 if args.backbone=='vgg16_bn' else 0.1)
    (out/'run_config.json').write_text(json.dumps({'training':config,'run':metadata},indent=2),encoding='utf-8')
    if args.backbone=='resnet18':
        from training import resnet as trainer
        trainer.ROOT=out;trainer.CONFIG=config;trainer.ARGS=args;trainer.main()
    elif args.backbone=='vgg16_bn':
        from training import vgg as trainer
        trainer.ROOT=out;trainer.DATA_ROOT=args.data_root.resolve();trainer.INITIAL_CHECKPOINT=args.initial_checkpoint
        trainer.train_run(f'seed_{args.seed}',0.01,args.seed)
    else:
        from training import wrn as trainer
        trainer.OUT=out;trainer.DATA=args.data_root.resolve();trainer.RUNTIME_LOG=out/'train.log'
        for name in ['checkpoints','training_logs','raw']:(out/name).mkdir()
        trainer.cmd_train(args.seed)

if __name__=='__main__':main()
