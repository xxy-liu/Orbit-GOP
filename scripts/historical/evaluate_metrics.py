"""Compute canonical OOD metrics from two saved Orbit-GOP score files."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from orbit_gop.metrics import ood_metrics

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--id-scores',type=Path,required=True)
    p.add_argument('--ood-scores',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    with np.load(args.id_scores,allow_pickle=False) as a,np.load(args.ood_scores,allow_pickle=False) as b:
        for key in ['backbone','seed','checkpoint_sha256','reference_sha256','precision','beta']:
            if not np.array_equal(a[key],b[key]):raise ValueError('Incompatible score identity: '+key)
        if str(a['dataset'])!='cifar10' or str(b['dataset']) not in ['cifar100','tiny','svhn']:
            raise ValueError('Expected CIFAR-10 ID and an OOD/diagnostic endpoint')
        values={method:ood_metrics(a[method],b[method]) for method in ['global_uncertainty','orbit_uncertainty']}
        values['scope']='Single supplied model/seed; no bootstrap or paper-level aggregate'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as f:json.dump(values,f,indent=2,allow_nan=False)

if __name__=='__main__':main()
