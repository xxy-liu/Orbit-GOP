"""Build a NEW FP32 reference on the frozen 3,000 CIFAR-10 calibration IDs."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from torch.utils.data import DataLoader
from scripts.historical.common import configure,dataset,load_model,save_new,sha256
from orbit_gop.functional_signature import extract_loader
from orbit_gop.calibration import fit_global_scale
from orbit_gop._scores import fit_c_global

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backbone',required=True,choices=['resnet18','vgg16_bn','wrn28_10'])
    p.add_argument('--seed',type=int,required=True,choices=[0,1,2])
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    args=p.parse_args();configure()
    model=load_model(args.backbone,args.checkpoint,args.device)
    ds=dataset('cifar10',args.data_root,calibration=True)
    values=extract_loader(model,DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=0),args.device,args.backbone)
    fit=(fit_c_global if args.backbone=='wrn28_10' else fit_global_scale)(values['evidence'],values['label'])
    if not fit.get('pass',fit.get('success',False)):raise RuntimeError('Calibration fit failed')
    save_new(args.output,**values,c_global=fit['c_global'],backbone=args.backbone,seed=args.seed,
             checkpoint_sha256=sha256(args.checkpoint),reference_mode='NEW_FP32_CALIBRATION',precision='FP32',batch_size=args.batch_size)
    print('Created a new calibration reference; this does not replace frozen paper references.')

if __name__=='__main__':main()
