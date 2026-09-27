# Independent numeric argmax diagnostic

resnet18 / S4 / CIFAR100 / cifar100/test/7986 (zero-based row 7986). Raw original index, source label and checkpoint identity match the score file and original manifest. Base prediction=5; formal Orbit prediction=5; saved numerical diagnostic=6.

原因未证实。No attribution to floating-point error is made. The isolated saved-array float64 postprocessing copy reproduces numeric argmax=6 and the exact saved uncertainty. This does not rerun the FP32 model or establish a hardware-level cause. No precision, formula or formal score is changed.

Saved raw output/evidence/probability, dtypes, dGOP/q/retention, full calibration parameters, raw/identity paths and hashes, and reproduced probability are in audit/ARGMAX_ROW.json.

Prediction generation path excerpts:
```
PROJECT\outputs\Orbit-GOP-confirmatory-20260917-v2\code\evaluate_confirmatory.py:19: from orbit_gop.calibration import fit_global_scale,scaled_probability
PROJECT\outputs\Orbit-GOP-confirmatory-20260917-v2\code\evaluate_confirmatory.py:64:                 fixed=out['probability'].argmax(1).detach()
PROJECT\outputs\Orbit-GOP-confirmatory-20260917-v2\code\evaluate_confirmatory.py:65:                 original={k:out[k].detach() for k in ('raw_output','evidence','probability','features')}
PROJECT\outputs\Orbit-GOP-confirmatory-20260917-v2\code\evaluate_confirmatory.py:66:                 original['prediction']=fixed
PROJECT\outputs\Orbit-GOP-confirmatory-20260917-v2\code\evaluate_confirmatory.py:67:             grad=torch.autograd.grad(stable_margin(out['raw_output'],fixed).sum(),captured['feature'])[0]
PROJECT\outputs\Orbit-GOP-confirmatory-20260917-v2\code\evaluate_confirmatory.py:84:         torch.autograd.grad(out['raw_output'].max(1).values.sum(),x)
PROJECT\outputs\Orbit-GOP-confirmatory-20260917-v2\code\evaluate_confirmatory.py:100:             values={'features':o['features'],'raw_output':o['raw_output']}
PROJECT\outputs\Orbit-GOP-confirmatory-20260917-v2\code\evaluate_confirmatory.py:151:     c=fit['c_global'];refs={i:np.sort(cal['d_gop'][cal['prediction']==i].astype(np.float64)) for i in range(10)}
PROJECT\outputs\Orbit-GOP-confirmatory-20260917-v2\code\evaluate_confirmatory.py:153:         q=np.empty(len(x['prediction']))
PROJECT\outputs\Orbit-GOP-confirmatory-20260917-v2\code\evaluate_confirmatory.py:155:             mask=x['prediction']==i
PROJECT\outputs\Orbit-GOP-confirmatory-20260917-v2\code\evaluate_confirmatory.py:173:     vim_alpha=float(bank['raw_output'].max(1).mean()/denom)
PROJECT\outputs\Orbit-GOP-confirmatory-20260917-v2\code\evaluate_confirmatory.py:179:         raw=x['raw_output'].astype(np.float64);feat=x['features'].astype(np.float64);base=x['prediction']
PROJECT\outputs\Orbit-GOP-confirmatory-20260917-v2\code\evaluate_confirmatory.py:181:         globalp,globalu=scaled_probability(x['evidence'],c)
PROJECT\outputs\Orbit-GOP-confirmatory-20260917-v2\code\evaluate_confirmatory.py:182:         orbitp,orbitu=scaled_probability(x['evidence'],c*r)
PROJECT\outputs\Orbit-GOP-confirmatory-20260917-v2\code\evaluate_confirmatory.py:184:         methods={'MSP':(1-p.max(1),base,None),'Predictive_Entropy':(-(p*logp).sum(1),base,None),'Energy':(-logsumexp(raw,axis=1),base,None),'ReAct_Energy':(-logsumexp(clipped,axis=1),clipped.argmax(1),None),'KNN':(kth_distance_formula(banknorm,l2_normalize(x['features'])),base,None),'ViM':(vim_alpha*np.linalg.norm((feat-u)@ns,axis=1)-logsumexp(raw,axis=1),base,None),'GAIA_Z':(x['gaia_z'],base,None),'GradNorm':(x['gradnorm'],base,None),'Global_EDL':(globalu,base,globalu),'Orbit_GOP':(orbitu,base,orbitu)}
PROJECT\outputs\Orbit-GOP-confirmatory-20260917-v2\code\evaluate_confirmatory.py:188:             methods[name]=(out['uncertainty'],out['prediction'],out['uncertainty'])
PROJECT\outputs\Orbit-GOP-confirmatory-20260917-v2\code\evaluate_confirmatory.py:195:                 with np.load(path) as old:assert np.array_equal(old['score'],score) and np.array_equal(old['base_prediction'],base)
PROJECT\outputs\Orbit-GOP-confirmatory-20260917-v2\code\evaluate_confirmatory.py:200:             save_npz(path,sample_id=np.array(sampleids),original_index=x['original_index'],ground_truth=x['label'],base_prediction=base,calibrated_prediction=pred,score=score,uncertainty=score if uncertainty is None else uncertainty,uncertainty_semantics=np.array('method_risk_score' if uncertainty is None else 'Dirichlet_vacuity'),correctness=correctness,checkpoint_hash=np.array(m['checkpoint_hash']),config_hash=np.array(m['config_hash']),split_hash=np.array(m['split_hash']),**extra)
PROJECT\outputs\Orbit-GOP-confirmatory-20260917-v2\code\evaluate_confirmatory.py:203:     write(f'audit/{backbone}_S{seed}_SANITY.json',{'status':'PASS','frozen_model_unchanged':True,'parameter_grad_accumulation':False,'prediction_change_rate':0,'shared_original_evidence':True,'references_cal_only':True,'bank_train_only':True,'paired_order_verified':True,'bounds_finite_verified':True,'all_12_methods_3_datasets_saved':True})
PROJECT\outputs\Orbit-GOP-confirmatory-20260917-v2\code\vendor\orbit_gop\calibration.py:45: def scaled_probability(evidence: np.ndarray, scale: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
PROJECT\outputs\Orbit-GOP-confirmatory-20260917-stage7\stage7.py:126:         if n in ('Global_EDL','Orbit_GOP'):definition='K/sum(1+c_global*e)' if n=='Global_EDL' else 'K/sum(1+c_global*exp(-0.40*q_func)*e); prediction=base'
PROJECT\outputs\Orbit-GOP-confirmatory-20260917-stage7\stage7.py:231:             if name=='Orbit_GOP':assert np.array_equal(arrays['calibrated_prediction'],arrays['base_prediction'])
PROJECT\outputs\Orbit-GOP-confirmatory-20260917-stage7\stage7.py:232:             _,baseu=e.scaled_probability(x['evidence'],1.)
PROJECT\outputs\Orbit-GOP-confirmatory-20260917-stage7\stage7.py:235:                 diag=env['orbitp'].argmax(1);arrays['numeric_argmax_diagnostic']=diag;arrays['numeric_argmax_differs_from_base']=diag!=arrays['base_prediction']
PROJECT\outputs\Orbit-GOP-confirmatory-20260917-stage7\stage7.py:286:                         ids=z['sample_id'];idx=z['original_index'];base=z['base_prediction']
PROJECT\outputs\Orbit-GOP-confirmatory-20260917-stage7\stage7.py:293:                             assert np.array_equal(z['calibrated_prediction'],base)
PROJECT\outputs\Orbit-GOP-confirmatory-20260917-stage7\stage7.py:297:         write('audit/FINAL_VERIFICATION.json',{'status':'PASS','time':now(),'combinations':324,'source_hashes_reverified':True,'alignment_finite_bounds_predictions_checked':True})
```

## Saved class-5/class-6 evidence (zero-based classes)

|Saved array|dtype|class 5|class 6|
|---|---|---:|---:|
|raw_output|float32|-4.892866134643555|-4.892864227294922|
|evidence|float32|0.007471911609172821|0.007471926044672728|
|probability|float32|0.10030095279216766|0.10030095279216766|

The saved FP32 base probabilities are tied at classes 5 and 6; base prediction is 5. The saved FP32 evidence values differ; the unchanged float64 calibration-copy probabilities are 0.1022300171183987 and 0.102230027767505. These are observed saved values, not a new model computation or proof of the original numerical cause. 原因未证实。

Calibration c_global=11.205565750166224, q_func=0.9616724738675958, retention=0.6806759097597896. Full ten-class raw output/evidence/probability and provenance remain in audit/ARGMAX_ROW.json.
