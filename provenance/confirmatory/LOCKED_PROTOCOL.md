# Prospectively locked confirmatory study — protocol v1

Number of independently trained models per backbone: 3
Prespecified confirmatory seeds: 3, 4, 5

S3=3, S4=4, S5=5; 3 backbones × 3 seeds = 9 newly trained EDL models. The user's seed amendment supersedes the initial five-model plan before any confirmatory training or performance outcome. Seeds 6/7 are excluded and cannot be added to improve results. All three models must be reported, including adverse outcomes. No performance-based retry, seed replacement, backbone exclusion or dataset exclusion.

## Primary and secondary questions

With new models, separated training/checkpoint-selection/calibration roles and fixed methods, does Orbit-GOP retain incremental near-OOD uncertainty ranking over Global EDL? For each backbone and each CIFAR-100/Tiny endpoint, report ΔAUROC=Orbit−Global (>0 favorable) and ΔFPR95=Orbit−Global (<0 favorable). On official CIFAR-10 test, operational Orbit prediction change must be exactly zero. Secondary questions: Default and Fixed-Dev-Tuned C-EDL comparisons, backbone consistency, AURC and training-seed variability. Mixed outcomes remain mixed.

Strong confirmation: primary paired 95% CI wholly in the favorable direction. Directional confirmation: favorable mean, CI includes zero. No clear confirmation: zero/near-zero effect with CI across zero (no post-hoc numerical equivalence margin will be invented). Contradictory result: unfavorable mean. These labels apply to individual prespecified endpoints; no global superiority or familywise significance is inferred from one favorable endpoint.

## Immutable training/data contract

CIFAR-10 train 50,000: class-stratified 44,000 train, 3,000 validation, 3,000 calibration; per class 4,400/300/300; split_seed=20260917, PCG64, exact algorithm and original IDs saved. Official test 10,000 is excluded from optimizer, scheduler, early stopping, checkpoint selection, model admission and calibration fitting. Calibration cannot affect model training or selection. No OOD development is rerun.

ResNet18 and VGG16-BN use their historical accuracy→lower ECE15→lower categorical NLL→earlier epoch selection on the new validation role, with their original 200-epoch recipes. VGG uses the final P8-R1 learning rate 0.01, not the earlier P5-3R 0.03 recipe. No performance collapse/admission gate is retained. NaN/Inf remains an engineering stop.

WRN-28-10 checkpoint policy:
Fixed epoch = 200.
The prespecified validation subset is diagnostic only and is not used
for checkpoint selection, early stopping, model admission, or seed replacement.

This WRN clarification was explicitly approved by the user before any confirmatory performance result. Validation diagnostic is recorded after epoch 200, without influencing the training RNG or chosen epoch. The deliverable `best.pt` for WRN is merely the fixed epoch-200 checkpoint; it does not denote a validation optimum. Official CIFAR-10 test admission is removed.

All original architecture, initialization scheme, SGD/momentum/Nesterov/weight decay, learning rates, 5-epoch warmup/cosine schedule, batch 128, 200 epochs, Softplus head, squared-error Bayes-risk EDL loss and 10-epoch masked-KL schedule are inherited. Training uses original AMP/FP32 EDL mathematics. Exact numerical dependencies and source mappings are preserved separately; historical files are never imported as running experiment executors.

## Frozen method and baselines

Full Orbit = Original, brightness clip(0.9*x,0,1), reflection pad 1 then crop [...,0:32,0:32], before normalization. Fixed original-view reference class across all views; one-vs-rest raw-logit margin; per-sample spatial mean squared feature gradient; channel normalization epsilon=1e-12; generalized JSD/ln(3); class-conditional empirical midrank; beta=0.40; r=exp(-0.40*q_func); alpha=1+c_global*r*e. Operational prediction is copied from base, and numerical alpha-argmax is not allowed to redefine it. Layers: ResNet layer3, VGG block4 before pool4, WRN block3 before final BN/ReLU. No parameter or view search.

New calibration 3,000 alone fits c_global and class references. c_global minimizes categorical NLL in log-c [-8,8], bounded scalar optimization xatol=1e-12. Only predicted classes actually used must have a nonempty reference; encountering a used empty class stops that run. q_func and dGOP must be in [0,1]; retention in [exp(-0.4),1]. No NaN/Inf.

All 12 requested methods use the same new checkpoint and source-qualified sample order. FP32 model arithmetic with TF32/autocast off for all inference, common float64 postprocessing where applicable, original exact FP32 KNN accumulation. ReAct fits only calibration; KNN/ViM only the new training role. ViM author's low-dimensional rule D=N//2 gives 256/256/320, without search. Default C-EDL=(beta=1.5,lambda=.5,delta=1,T=5); fixed C009=(1,.75,.5,T=5). C009 was historically common to ResNet/VGG and is prospectively transferred unchanged to WRN; it is not represented as a historical WRN-specific tuning result. No extra c_global is applied to C-EDL. All detailed baseline definitions are in confirmatory_locked.yaml.

## Evaluation identity and contamination disclosure

Replication endpoints: official CIFAR-10 test 10,000; official CIFAR-100 test 10,000; exactly the historical 7,793-path Tiny manifest. Actual source IDs, labels, file/normalized hashes and order are frozen before inference. OOD labels belong to different class ontologies; ID classification correctness is not fabricated for OOD images (saved as not-applicable −1).

Exact image audit found CIFAR100/test/2203 identical to CIFAR10/test/9345, and CIFAR100/test/2224 identical to CIFAR10/train/3021 (the latter is in the new training role). Both are retained in the replication benchmark, with IDs and hashes disclosed before outcomes, consistent with the requested fixed benchmark replication and no performance-dependent deletion. Thus replication is not described as an entirely contamination-free OOD dataset. The new CIFAR-10 train/val/cal/test roles have zero exact image cross-role duplicates. No near-duplicate claim is made.

Fresh-source branch remains paused. The current OpenOOD CIFAR-10 near-OOD taxonomy lists CIFAR-100 and Tiny; both have historical use. CIFAR-10.1 and the remaining Tiny 968 were evaluated on 2026-09-12 and cannot now count as never-used sources. No far-OOD substitution or candidate inference is authorized by this freeze. Core replication continues independently.

## Statistics fixed before outcomes

Individual S3/S4/S5 estimates, three-seed mean and sample SD (ddof=1). Primary CI: 5,000 synchronized paired sample-level bootstrap draws, shared by all three seeds and methods. CIFAR-10 and CIFAR-100 true-class stratification; ordinary Tiny resampling. OOD-positive first tied threshold achieving TPR≥.95, no interpolation; AUROC gives half credit to ties. AURC uses stable original-index tie ordering. Percentile 2.5/97.5 bounds. Bootstrap PCG64 seed 20260917.

Supplementary hierarchical bootstrap: 5,000 replicates, PCG64 seed 20260918, resample three model seeds with replacement and paired samples within each sampled occurrence; mean paired effect over the three occurrences. Only three independent training models are available. This training-variability sensitivity analysis does not replace the primary synchronized sample CI.

## Execution and integrity

The recorded execution uses a local single GPU with sequential training. Train ResNet S3/S4/S5, VGG S3/S4/S5, WRN S3/S4/S5. All nine checkpoints must be frozen before final evaluation. Save model/config/split hashes, selected epoch or fixed epoch and validation role; then calibration, 12-method per-sample scoring, statistics, confirmation-only figures and final audit.

Only documented engineering faults permit same-seed same-config restart. Completed outcomes are never used to alter configs, manifests or choices. A code bug requires BUGFIX_LOG, reason independent of performance, protocol version bump and renewed checksums; the original freeze must remain auditable. A changed config hash is a stop. A performance failure is reported, not rescued.

Freeze timestamp and checksums are in PRE_RUN_FREEZE_REPORT.md and checksums/SHA256SUMS.txt. This protocol is not retrospectively dated. Project root has no Git history; the nested public-release repository has an unborn HEAD and preexisting staged files and is kept untouched. Source SHA-256, not a fabricated commit, records this checkout.
