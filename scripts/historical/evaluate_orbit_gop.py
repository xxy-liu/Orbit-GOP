"""Evaluate frozen Orbit-GOP equations with an explicitly supplied reference."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from torch.utils.data import DataLoader
from scripts.historical.common import configure,dataset,load_model,save_new,sha256
from orbit_gop.functional_signature import extract_loader
from orbit_gop.inference import apply_reference

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backbone',required=True,choices=['resnet18','vgg16_bn','wrn28_10'])
    p.add_argument('--seed',type=int,required=True,choices=[0,1,2])
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--dataset',choices=['cifar10','cifar100','svhn','tiny'],required=True)
    p.add_argument('--tiny-manifest',type=Path,help='Frozen OpenOOD test_tin.txt; required for tiny')
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    args=p.parse_args();configure()
    with np.load(args.reference,allow_pickle=False) as f:ref={k:f[k] for k in f.files}
    if str(ref['backbone'])!=args.backbone or int(ref['seed'])!=args.seed:raise ValueError('Reference model/seed mismatch')
    if str(ref['checkpoint_sha256'])!=sha256(args.checkpoint):raise ValueError('Reference checkpoint identity mismatch')
    model=load_model(args.backbone,args.checkpoint,args.device)
    if args.dataset=='tiny':
        from scripts.historical.tiny_dataset import TinyManifest
        if args.tiny_manifest is None or sha256(args.tiny_manifest)!='a44fd34e3c925d984063d2ad64969ff71e8fa6f5524fbe9da481c8b0c87b27cd':
            p.error('Tiny requires the frozen 7,793-row test_tin.txt with the recorded SHA256')
        ds=TinyManifest(args.tiny_manifest,args.data_root/'images_classic')
    else:ds=dataset(args.dataset,args.data_root)
    values=extract_loader(model,DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=0),args.device,args.backbone)
    result=apply_reference(ref,values)
    result['base_prediction']=values.pop('prediction')
    save_new(args.output,**values,**result,backbone=args.backbone,seed=args.seed,beta=0.40,
             checkpoint_sha256=sha256(args.checkpoint),reference_sha256=sha256(args.reference),
             reference_mode=ref['reference_mode'],precision='FP32',dataset=args.dataset,batch_size=args.batch_size)
    print('Saved sample scores. Operational prediction preserved; full paper reproduction was not run.')

if __name__=='__main__':main()
