"""Frozen calibration and complete paired per-sample evaluation, without tuning."""
import sys,json,argparse,hashlib,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision import datasets,transforms
from PIL import Image
from scipy.special import logsumexp
from .preflight import OUT
from .preflight import sha,write
from .train_confirmatory import verify_lock
from .resume_integrity import expected_identity, raw_metadata, validate_cached_raw, verify_or_save_npz, verify_or_write_json
from models.registry import create_model
from orbit_gop.views import make_views,normalize
from orbit_gop._resnet_signature import stable_margin
from orbit_gop.functional_signature import signature_from_gradient,LAYERS
from orbit_gop.functional_risk import generalized_js,midrank
from orbit_gop.calibration import fit_global_scale,scaled_probability
from baselines.cedl_offline import score_from_view_evidence
from baselines.gradnorm import AuditedGradNorm,EDLAdapter,state_hash
from baselines.knn import l2_normalize,kth_distance_formula
from .cedl_views import transformed_batch

def save_npz(path,**arrays):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError('Immutable output already exists '+str(path))
    np.savez_compressed(path,**arrays)
class RoleDataset(Dataset):
    def __init__(self,role,cfg,manifest):
        self.role=role;self.cfg=cfg;self.manifest=manifest
        if role in ('train','cal'):
            self.base=datasets.CIFAR10(str(Path(cfg['data_root'])/'CIFAR-10'),train=True,download=False,transform=transforms.ToTensor())
            with np.load(OUT/'splits/cifar10_confirmatory_split_v1.npz') as z:self.indices=z[role]
            assert self.base.train
        elif role in ('ID','CIFAR100'):
            record=manifest[role];cls=datasets.CIFAR10 if role=='ID' else datasets.CIFAR100
            self.base=cls(record['root'],train=False,download=False,transform=transforms.ToTensor());self.indices=np.asarray(record['indices'])
            assert not self.base.train and np.array_equal(self.base.targets,record['labels'])
        elif role=='Tiny':
            self.indices=np.arange(len(manifest['Tiny']['rows']))
            self.tx=transforms.Compose([transforms.Resize(32,interpolation=transforms.InterpolationMode.BILINEAR),transforms.CenterCrop(32),transforms.ToTensor()])
        else:raise ValueError('Role forbidden for evaluation '+role)
    def __len__(self):return len(self.indices)
    def __getitem__(self,pos):
        i=int(self.indices[pos])
        if self.role=='Tiny':
            row=self.manifest['Tiny']['rows'][i]
            path=Path(self.manifest['Tiny']['root'])/row['relative_path']
            if sha(path)!=row['raw_sha256']:raise RuntimeError('Tiny source hash changed: '+row['sample_id'])
            with Image.open(path) as img:x=self.tx(img.convert('RGB'))
            return x,int(row['label']),i
        x,y=self.base[i];return x,int(y),i

def original_and_orbit(model,images,backbone):
    signatures=[];fixed=None;original=None
    with torch.enable_grad(),torch.autocast('cuda',enabled=False):
        for view in make_views(images):
            captured={}
            h=getattr(model,LAYERS[backbone]).register_forward_hook(lambda _m,_i,o:captured.update(feature=o))
            try:out=model(normalize(view).detach().requires_grad_(True))
            finally:h.remove()
            if fixed is None:
                fixed=out['probability'].argmax(1).detach()
                original={k:out[k].detach() for k in ('raw_output','evidence','probability','features')}
                original['prediction']=fixed
            grad=torch.autograd.grad(stable_margin(out['raw_output'],fixed).sum(),captured['feature'])[0]
            signatures.append(signature_from_gradient(grad).detach())
    original['d_gop']=generalized_js(torch.stack(signatures,1)).detach()
    assert all(torch.isfinite(v).all() for v in original.values())
    assert all(p.grad is None for p in model.parameters())
    return original

def gaia(model,images):
    gradients={};handles=[]
    def hook(name):
        def f(_m,_i,o):o.register_hook(lambda g:gradients.update({name:g.detach()}))
        return f
    for name,module in model.named_modules():
        if isinstance(module,nn.BatchNorm2d):handles.append(module.register_forward_hook(hook(name)))
    try:
        x=normalize(images).detach().requires_grad_(True)
        out=model(x)
        torch.autograd.grad(out['raw_output'].max(1).values.sum(),x)
        assert len(gradients)==len(handles)>0
        value=torch.cat([(g!=0).float().mean((-1,-2)) for g in gradients.values()],1).square().mean(1)
        return value.detach(),out['evidence'].detach()
    finally:
        for h in handles:h.remove()

def extract(model,role,cfg,manifest,backbone,full):
    ds=RoleDataset(role,cfg,manifest)
    loader=DataLoader(ds,batch_size=cfg['evaluation']['batch_size'],shuffle=False,num_workers=2,pin_memory=True,drop_last=False)
    chunks={};rng=np.random.RandomState(cfg['cedl_transform_seed'])
    gn=AuditedGradNorm(EDLAdapter(model)) if full else None
    for batch,(images,labels,indices) in enumerate(loader):
        cpu_images=images;images=images.cuda(non_blocking=True)
        if role=='train':
            with torch.no_grad():o=model(normalize(images))
            values={'features':o['features'],'raw_output':o['raw_output']}
        else:values=original_and_orbit(model,images,backbone)
        if full:
            g,e=gaia(model,images)
            assert torch.equal(e,values['evidence']),'GAIA original evidence path differs'
            values['gaia_z']=g
            raw_g,_=gn.score_features(values['features']);values['gradnorm']=torch.from_numpy(-raw_g)
            views,trace=transformed_batch(cpu_images,rng)
            evidence=[values['evidence']]
            with torch.no_grad():
                for v in views[1:5]:evidence.append(model(normalize(v.cuda()))['evidence'])
            values['cedl_views']=torch.stack(evidence,1);values['transform_trace']=torch.from_numpy(trace)
        values.update(label=labels,original_index=indices)
        for k,v in values.items():chunks.setdefault(k,[]).append(v.detach().cpu().numpy())
        if batch%100==0:print(json.dumps({'event':'extract','backbone':backbone,'role':role,'batch':batch,'rows':min((batch+1)*cfg['evaluation']['batch_size'],len(ds))}),flush=True)
    if gn:assert gn.unchanged()
    result={k:np.concatenate(v) for k,v in chunks.items()}
    assert np.array_equal(result['original_index'],ds.indices)
    assert all(np.isfinite(v).all() for v in result.values())
    return result

def run(backbone,seed):
    cfg=verify_lock();manifest=json.loads((OUT/'manifests/EVALUATION_MANIFEST.json').read_text())
    # Prevent early test evaluation: all nine checkpoints must be frozen first.
    for b in cfg['backbones']:
        for s in cfg['seeds']:
            if not (OUT/f'manifests/{b}_S{s}.json').exists():raise RuntimeError('All nine models must be frozen before calibration/evaluation')
    m=json.loads((OUT/f'manifests/{backbone}_S{seed}.json').read_text())
    cp=Path(m['checkpoint']);assert sha(cp)==m['checkpoint_hash']
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(True,warn_only=True)
    model=create_model(backbone);payload=torch.load(cp,map_location='cpu',weights_only=True);model.load_state_dict(payload['model_state'],strict=True)
    model.cuda().eval()
    for p in model.parameters():p.requires_grad_(False);p.grad=None
    before=state_hash(model)
    rawdir=OUT/'raw'/backbone/f'seed_{seed}'
    rawdir.mkdir(parents=True,exist_ok=True)
    def cached(role,full=False):
        path=rawdir/f'{role}.npz'
        # Same locked run can resume verified complete artifacts after an engineering stop.
        side=path.with_suffix('.json')
        identity=expected_identity(role,manifest,OUT/'splits/cifar10_confirmatory_split_v1.csv')
        metadata=raw_metadata(role,identity,backbone,seed,m['checkpoint_hash'],m['config_hash'],full)
        if path.exists():
            return validate_cached_raw(path,side,metadata,identity,sha)
        if side.exists():raise RuntimeError('Raw cache metadata exists without archive: '+str(side))
        value=extract(model,role,cfg,manifest,backbone,full);save_npz(path,**value)
        side.write_text(json.dumps({**metadata,'sha256':sha(path)},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        return validate_cached_raw(path,side,metadata,identity,sha)
    cal=cached('cal');bank=cached('train')
    assert len(cal['label'])==3000 and len(bank['label'])==44000
    fit=fit_global_scale(cal['evidence'],cal['label'])
    if not fit['optimizer_success']:raise RuntimeError('c_global fit failed')
    c=fit['c_global'];refs={i:np.sort(cal['d_gop'][cal['prediction']==i].astype(np.float64)) for i in range(10)}
    def risk(x):
        q=np.empty(len(x['prediction']))
        for i,r in refs.items():
            mask=x['prediction']==i
            if not mask.any():continue
            if len(r)==0:raise RuntimeError('Empty reference for used predicted class '+str(i))
            q[mask]=midrank(r,x['d_gop'][mask])
        assert np.all((q>=0)&(q<=1))
        return q
    qc=risk(cal)
    assert np.all((cal['d_gop']>=0)&(cal['d_gop']<=1))
    verify_or_write_json(OUT/'calibration'/f'{backbone}_S{seed}.json',{'fit':fit,'class_reference_sizes':{i:len(r) for i,r in refs.items()},'q_func_range':[float(qc.min()),float(qc.max())],'dGOP_range':[float(cal['d_gop'].min()),float(cal['d_gop'].max())],'retention_range':[float(np.exp(-.4*qc).min()),float(np.exp(-.4*qc).max())],'calibration_hash':sha(rawdir/'cal.npz')})
    # ReAct calibration-only. KNN and ViM train-only.
    threshold=float(np.quantile(cal['features'].astype(np.float64),.9,method='linear'))
    banknorm=l2_normalize(bank['features'])
    w=model.classifier.weight.detach().cpu().numpy().astype(np.float64);bias=model.classifier.bias.detach().cpu().numpy().astype(np.float64)
    feature=bank['features'].astype(np.float64);u=-np.linalg.pinv(w)@bias
    centered=feature-u;cov=centered.T@centered/len(feature)
    vals,vec=np.linalg.eigh(cov);D=cfg['baselines']['ViM']['dimension'][backbone];ns=vec[:,np.argsort(vals)[::-1][D:]]
    denom=np.linalg.norm(centered@ns,axis=1).mean()
    if not np.isfinite(denom) or denom<=0:raise RuntimeError('Invalid ViM train residual')
    vim_alpha=float(bank['raw_output'].max(1).mean()/denom)
    savepath=OUT/'calibration'/f'{backbone}_S{seed}_references.npz'
    reference_arrays={**{f'class_{i}':r for i,r in refs.items()},'c_global':np.array(c),'react_threshold':np.array(threshold),'vim_origin':u,'vim_residual_basis':ns,'vim_alpha':np.array(vim_alpha)}
    reference_meta={'dataset':'CIFAR10/cal+train','backbone':backbone,'seed':seed,'checkpoint_hash':m['checkpoint_hash'],'config_hash':m['config_hash'],'calibration_hash':sha(rawdir/'cal.npz'),'train_hash':sha(rawdir/'train.npz'),'reference_fields':sorted(reference_arrays)}
    verify_or_save_npz(savepath,reference_arrays,save_npz,sha,savepath.with_suffix('.json'),reference_meta)
    for role in ('ID','CIFAR100','Tiny'):
        x=cached(role,full=True);n=len(x['label']);q=risk(x);r=np.exp(-.4*q)
        assert np.all((x['d_gop']>=0)&(x['d_gop']<=1));assert np.all((r>=np.exp(-.4))&(r<=1))
        raw=x['raw_output'].astype(np.float64);feat=x['features'].astype(np.float64);base=x['prediction']
        logp=raw-logsumexp(raw,axis=1)[:,None];p=np.exp(logp)
        globalp,globalu=scaled_probability(x['evidence'],c)
        orbitp,orbitu=scaled_probability(x['evidence'],c*r)
        clipped=np.minimum(feat,threshold)@w.T+bias
        methods={'MSP':(1-p.max(1),base,None),'Predictive_Entropy':(-(p*logp).sum(1),base,None),'Energy':(-logsumexp(raw,axis=1),base,None),'ReAct_Energy':(-logsumexp(clipped,axis=1),clipped.argmax(1),None),'KNN':(kth_distance_formula(banknorm,l2_normalize(x['features'])),base,None),'ViM':(vim_alpha*np.linalg.norm((feat-u)@ns,axis=1)-logsumexp(raw,axis=1),base,None),'GAIA_Z':(x['gaia_z'],base,None),'GradNorm':(x['gradnorm'],base,None),'Global_EDL':(globalu,base,globalu),'Orbit_GOP':(orbitu,base,orbitu)}
        for name,configfile in [('C_EDL_Default','cedl_default_locked.yaml'),('C_EDL_Fixed_Dev_Tuned','cedl_devtuned_locked.yaml')]:
            cconf=json.loads((OUT/'config'/configfile).read_text())
            out=score_from_view_evidence(x['cedl_views'],**{k:cconf[k] for k in ('conflict_beta','cedl_lambda','delta')})
            methods[name]=(out['uncertainty'],out['prediction'],out['uncertainty'])
        assert np.array_equal(methods['Orbit_GOP'][1],base)
        identity=expected_identity(role,manifest,OUT/'splits/cifar10_confirmatory_split_v1.csv')
        sampleids=identity['sample_id']
        if not np.array_equal(x['original_index'],identity['original_index']) or not np.array_equal(x['label'],identity['label']):raise RuntimeError('Evaluation cache sample identity changed: '+role)
        for name,(score,pred,uncertainty) in methods.items():
            assert score.shape==(n,) and np.isfinite(score).all()
            path=OUT/'scores'/backbone/str(seed)/name/f'{role}.npz'
            correctness=(pred==x['label']).astype(np.int8) if role=='ID' else np.full(n,-1,dtype=np.int8)
            # OOD labels have a different ontology; -1 denotes not applicable, not a false match to ID class IDs.
            extra={'q_func':q,'dGOP':x['d_gop'],'retention':r} if name=='Orbit_GOP' else {}
            score_arrays={'sample_id':np.array(sampleids),'original_index':x['original_index'],'ground_truth':x['label'],'dataset':np.array(identity['dataset']),'backbone':np.array(backbone),'seed':np.array(seed),'method':np.array(name),'prediction_field':np.array('base_prediction'),'score_field':np.array('score'),'base_prediction':base,'calibrated_prediction':pred,'score':score,'uncertainty':score if uncertainty is None else uncertainty,'uncertainty_semantics':np.array('method_risk_score' if uncertainty is None else 'Dirichlet_vacuity'),'correctness':correctness,'checkpoint_hash':np.array(m['checkpoint_hash']),'config_hash':np.array(m['config_hash']),'split_hash':np.array(m['split_hash']),**extra}
            verify_or_save_npz(path,score_arrays,save_npz)
    assert state_hash(model)==before and sha(cp)==m['checkpoint_hash']
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    verify_or_write_json(OUT/'audit'/f'{backbone}_S{seed}_SANITY.json',{'status':'PASS','frozen_model_unchanged':True,'parameter_grad_accumulation':False,'prediction_change_rate':0,'shared_original_evidence':True,'references_cal_only':True,'bank_train_only':True,'paired_order_verified':True,'bounds_finite_verified':True,'all_12_methods_3_datasets_saved':True})
    print('EVALUATION COMPLETE',backbone,seed,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('backbone');p.add_argument('seed',type=int);a=p.parse_args();run(a.backbone,a.seed)
