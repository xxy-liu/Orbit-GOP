"""Locked three-model synchronized and supplementary hierarchical inference."""
import sys,json,csv,argparse
from pathlib import Path
import numpy as np
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.confirmatory.preflight import OUT, write, csvwrite
    from scripts.confirmatory.train_confirmatory import verify_lock
    from scripts.confirmatory.bootstrap_kernel import WeightedLayout, bootstrap_weights
else:
    from .preflight import OUT, write, csvwrite
    from .train_confirmatory import verify_lock
    from .bootstrap_kernel import WeightedLayout, bootstrap_weights
from orbit_gop.metrics import ood_metrics,aurc
CORE=['Global_EDL','C_EDL_Default','C_EDL_Fixed_Dev_Tuned','Orbit_GOP']
METHODS=['MSP','Predictive_Entropy','Energy','ReAct_Energy','KNN','ViM','GAIA_Z','GradNorm',*CORE]

def load(b,s,m,d):
    with np.load(OUT/'scores'/b/str(s)/m/f'{d}.npz',allow_pickle=False) as z:return {k:z[k] for k in z.files}
def weighted_aurc(x,w):
    order=np.lexsort((x['original_index'],x['score']))
    weights=w[:,order].astype(np.int64);errors=1-x['correctness'][order].astype(np.float64)
    n=len(order);h=np.r_[0.,np.cumsum(1./np.arange(1,n+1))]
    end=np.cumsum(weights,axis=1);start=end-weights
    errprev=np.cumsum(weights*errors,axis=1)-weights*errors
    return (errors*weights+(errprev-errors*start)*(h[end]-h[start])).sum(1)/n
def summary(values):return float(np.mean(values)),float(np.std(values,ddof=1))
def interpretation(mean,lo,hi,metric):
    direction=1 if metric=='AUROC' else -1
    if direction*mean<0:return 'Contradictory result'
    if (lo>0 if direction>0 else hi<0):return 'Strong confirmation'
    if direction*mean>0:return 'Directional confirmation; CI includes zero'
    return 'No clear confirmation; CI includes zero'

def main():
    cfg=verify_lock();seeds=cfg['seeds'];assert seeds==[3,4,5];B=cfg['statistics']['replicates'];assert B==5000
    allrows=[];effects=[];summaries=[];aggregate=[]
    for backbone in cfg['backbones']:
        iddata={s:{m:load(backbone,s,m,'ID') for m in METHODS} for s in seeds}
        metrics={}
        for s in seeds:
            for method in METHODS:
                x=iddata[s][method];acc=float(np.mean(x['correctness']));pcr=float(np.mean(x['base_prediction']!=x['calibrated_prediction']))
                a=aurc(x['score'],x['correctness'].astype(bool),x['original_index']);metrics[(s,method,'ID')]={'AURC':a}
                common={'backbone':backbone,'seed':s,'method':method,'checkpoint_hash':str(x['checkpoint_hash']),'config_hash':str(x['config_hash']),'split_hash':str(x['split_hash'])}
                allrows.append({**common,'dataset':'ID','AUROC':'NA','FPR95':'NA','AURC':a,'accuracy':acc,'prediction_change_rate':pcr})
                if method=='Orbit_GOP':assert pcr==0
                for d in ('CIFAR100','Tiny'):
                    y=load(backbone,s,method,d);vals=ood_metrics(x['score'],y['score']);metrics[(s,method,d)]=vals
                    allrows.append({**common,'dataset':d,**vals,'AURC':'NA','accuracy':'NA','prediction_change_rate':float(np.mean(y['base_prediction']!=y['calibrated_prediction']))})
        for d in ('ID','CIFAR100','Tiny'):
            for method in METHODS:
                for met in (['AURC'] if d=='ID' else ['AUROC','FPR95']):
                    mean,sd=summary([metrics[(s,method,d)][met] for s in seeds]);aggregate.append({'backbone':backbone,'dataset':d,'method':method,'metric':met,'mean':mean,'SD':sd,'n_models':3})
            for s in seeds:
                for comparator in CORE[:-1]:
                    row={'backbone':backbone,'dataset':d,'seed':s,'comparison':'Orbit_GOP - '+comparator}
                    for met in ('AUROC','FPR95','AURC'):row['delta_'+met]=metrics[(s,'Orbit_GOP',d)][met]-metrics[(s,comparator,d)][met] if met in metrics[(s,'Orbit_GOP',d)] else 'NA'
                    effects.append(row)
        # Fixed score layouts; no threshold selection from performance.
        for d in ('ID','CIFAR100','Tiny'):
            endpoints=['AURC'] if d=='ID' else ['AUROC','FPR95'];nmet=len(endpoints)
            oid=None if d=='ID' else {s:{m:load(backbone,s,m,d) for m in CORE} for s in seeds}
            layouts={} if d=='ID' else {(si,mi):WeightedLayout(iddata[s][m]['score'],oid[s][m]['score']) for si,s in enumerate(seeds) for mi,m in enumerate(CORE)}
            labels=iddata[seeds[0]][CORE[0]]['ground_truth']
            for s in seeds:
                for m in CORE:
                    assert np.array_equal(labels,iddata[s][m]['ground_truth'])
                    assert np.array_equal(iddata[seeds[0]][CORE[0]]['sample_id'],iddata[s][m]['sample_id'])
            olabels=None if d=='ID' else oid[seeds[0]][CORE[0]]['ground_truth']
            if d!='ID':
                for s in seeds:
                    for m in CORE:assert np.array_equal(oid[seeds[0]][CORE[0]]['sample_id'],oid[s][m]['sample_id'])
            rng=np.random.Generator(np.random.PCG64(cfg['statistics']['bootstrap_seed']))
            hrng=np.random.Generator(np.random.PCG64(cfg['statistics']['hierarchical_seed']))
            conditional=np.empty((B,3,nmet));hierarchical=np.empty_like(conditional)
            for start in range(0,B,16):
                count=min(16,B-start)
                iw=bootstrap_weights(labels,rng,count,True)
                ow=None if d=='ID' else bootstrap_weights(olabels,rng,count,d=='CIFAR100')
                per=np.empty((count,3,4,nmet))
                for si,s in enumerate(seeds):
                    for mi,m in enumerate(CORE):
                        if d=='ID':per[:,si,mi,0]=weighted_aurc(iddata[s][m],iw)
                        else:per[:,si,mi,:]=np.stack(layouts[(si,mi)].evaluate(iw,ow),1)
                conditional[start:start+count]=(per[:,:,3,None,:]-per[:,:,:3,:]).mean(1)
                # Only 3 independently trained models; supplementary analysis.
                seed_draw=hrng.integers(0,3,size=(count,3));hper=np.empty((count,3,4,nmet))
                for slot in range(3):
                    hiw=bootstrap_weights(labels,hrng,count,True)
                    how=None if d=='ID' else bootstrap_weights(olabels,hrng,count,d=='CIFAR100')
                    for si,s in enumerate(seeds):
                        which=seed_draw[:,slot]==si
                        if not which.any():continue
                        for mi,m in enumerate(CORE):
                            v=weighted_aurc(iddata[s][m],hiw)[:,None] if d=='ID' else np.stack(layouts[(si,mi)].evaluate(hiw,how),1)
                            hper[which,slot,mi,:]=v[which]
                hierarchical[start:start+count]=(hper[:,:,3,None,:]-hper[:,:,:3,:]).mean(1)
                if start%800==0:print(json.dumps({'event':'bootstrap','backbone':backbone,'dataset':d,'replicates':start+count}),flush=True)
            dest=OUT/'results/bootstrap';dest.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest/f'{backbone}_{d}.npz',sample_conditional=conditional,hierarchical=hierarchical,comparators=np.array(CORE[:-1]),metrics=np.array(endpoints),seeds=np.array(seeds))
            for ci,comparator in enumerate(CORE[:-1]):
                for ei,met in enumerate(endpoints):
                    values=[metrics[(s,'Orbit_GOP',d)][met]-metrics[(s,comparator,d)][met] for s in seeds];mean,sd=summary(values)
                    lo,hi=np.quantile(conditional[:,ci,ei],[.025,.975]);hlo,hhi=np.quantile(hierarchical[:,ci,ei],[.025,.975])
                    summaries.append({'backbone':backbone,'dataset':d,'comparison':'Orbit_GOP - '+comparator,'metric':met,'mean':mean,'SD':sd,'n_models':3,'CI_sample_conditional_low':float(lo),'CI_sample_conditional_high':float(hi),'CI_hierarchical_low':float(hlo),'CI_hierarchical_high':float(hhi),'primary_CI':'sample_conditional','replicates':B,'interpretation':interpretation(mean,lo,hi,met)})
    csvwrite('results/confirmatory_all_runs.csv',list(allrows[0]),allrows)
    csvwrite('results/confirmatory_pairwise_effects.csv',list(effects[0]),effects)
    csvwrite('results/confirmatory_summary.csv',list(summaries[0]),summaries)
    csvwrite('results/confirmatory_method_means.csv',list(aggregate[0]),aggregate)
    lines=['# Confirmatory results','', 'Three independent models per backbone: S3, S4, S5. All prespecified models and datasets are reported. The primary synchronized paired 95% bootstrap CI is conditional on these trained models. The supplementary hierarchical CI uses only three independent training models and is not a replacement for the primary CI. All figures and effects use these newly generated scores.','', '| Backbone | Dataset | Comparison | Metric | Mean | SD | Sample 95% CI | Hierarchical 95% CI | Interpretation |','|---|---|---|---|---:|---:|---|---|---|']
    for r in summaries:lines.append(f"|{r['backbone']}|{r['dataset']}|{r['comparison']}|{r['metric']}|{r['mean']:.6f}|{r['SD']:.6f}|[{r['CI_sample_conditional_low']:.6f}, {r['CI_sample_conditional_high']:.6f}]|[{r['CI_hierarchical_low']:.6f}, {r['CI_hierarchical_high']:.6f}]|{r['interpretation']}|")
    lines+=['','Historical adaptive development remains disclosed. CIFAR-100/Tiny replication endpoints were used historically. Two preidentified exact cross-dataset duplicate images are retained according to the locked replication manifest, including one duplicate of a training image. No unused standard near-OOD source was admitted. These results do not eliminate all development or dataset bias.']
    (OUT/'results/CONFIRMATORY_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('ANALYSIS COMPLETE',len(allrows),len(summaries),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Analyze a complete new confirmatory run using the locked protocol.')
    parser.parse_args()
    main()
