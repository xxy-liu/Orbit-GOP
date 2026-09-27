# 当前执行状态：LOCAL_EVALUATION_AUTHORIZED

# 本地运行授权范围

记录时间：2026-09-12T14:24:49.699181+08:00（Asia/Shanghai）。

本地执行范围：
> 按已经通过旧样本技术检查的V2方案，执行一次完整的本地新增样本固定配置评估；本地执行获准，外部数据使用依据仍待确认。

LOCAL_EVALUATION_AUTHORIZED。批准1994张ID、968张OOD、两骨干各三个固定种子、Global EDL / C-EDL Dev-tuned C009 / Function-only Orbit-GOP及V2规定的5000次配对重采样；正常预检后连续完成。

当前批准仅替代本轮本地执行的旧PENDING工作流闸门，旧协议与记录保持不动，不倒签。数据使用依据仍NEEDS_AUTHOR_OR_INSTITUTION_CONFIRMATION；未获得外部新答复，未声明图像MIT或外部许可，不上传或再分发原图。

V2技术/数值/统计内容不变。旧FAIL、NOT_REPLAYED、A001历史披露不变。

---

以下保留V2协议的技术内容和历史状态（发布副本的非科学措辞已规范化），内含历史待批准/未执行状态，作为历史原文保留；这些本地执行状态由以上当前授权替代，技术、数据与统计约定原样执行，来源核实状态不变。

# 新增样本固定配置评估协议 V2（待统一批准）

**PENDING_SOURCE_AND_PROTOCOL_APPROVAL — 新增候选模型推断及正式评估未获批准。**

本技术修订已由作者明确授权并仅在旧技术/旧开发样本检验。两张图的保守资格处置完成，视觉同源关系仍未确定；CIFAR-10.1图像/标签使用依据继续NEEDS_AUTHOR_OR_INSTITUTION_CONFIRMATION。V2工程PASS不等于数据使用许可，不等于整份评估协议已批准。

实际目录：`PROJECT\outputs\NEW_VALIDATION_TARGETED_FIX_20260912_133901`。下文data/和sources/旧数据来源相对于`PROJECT\outputs\NEW_VALIDATION_DATA_PREP_20260912_120837`；旧ASSET_MAPPING.csv位于`PROJECT\outputs\NEW_VALIDATION_READINESS_CLOSURE_20260912_130032`。本轮采用TECHNICAL_CONTRACT_V2.json，旧合同、旧FAIL与旧协议保留。

## 1. 研究问题和历史边界

在冻结模型/配置下，以新增同类ID图像和按事先规则通过的Tiny图像，评价Orbit-GOP相对Global EDL的增量；相对C-EDL (Dev-tuned)单独报告。ResNet18和VGG16-BN各S0–S2，不增加模型或挑选种子。每个骨干分别报告，不合成跨骨干单一优势。

称为“新增样本的固定配置评估”，不称全新类别的无条件独立证明。CIFAR-10.1不是训练/校准集，不预先宣称与CIFAR-10原测试集严格同分布。有限历史追查已完成，不循环扫描，不要求作者保证从未使用。原A001和既有测试反馈继续单独披露。本轮新发现的训练图像同源关系只作定向资格证据，没有扩大历史扫描。

## 2. 数据、资格与统一顺序

作者来源提交、实际地址、获取时间、Git blob及SHA256见sources/download_log.json。CIFAR-10.1明确v6，实际原始2000张；原数组不改。作者来源索引cifar10.1_v6_ti_indices.json按utils.py定义与数组同位置相连，2000个唯一索引。Tiny仅val_tin原0-based 32–999，968身份逐项核对原清单、val_annotations、文件/像素hash。7793正式、前32技术、原1207排除身份不补回；不扩展Tiny train，不纳入无标签test。参考库里的train/test仅用于重复比较。

原data/DATA_ELIGIBILITY_RULES.json v1.0及原未决清单保持只读。本轮以ELIGIBILITY_RULE_ADDENDUM_V2.json记录作者在任何新增候选模型推断之前授权D01074（v6/1026）、D01132（v6/1999）作EXCLUDE_UNRESOLVED_PROVENANCE，视觉关系仍UNRESOLVED，不称已确认重复。原4项排除不变，无替代样本。新资格表共2968身份：ID INCLUDE1994、原排除4、保守排除2；OOD INCLUDE968。待批准清单EVALUATION_MANIFEST_PENDING_V2.csv共2962行，所有行PENDING_SOURCE_AND_PROTOCOL_APPROVAL。资格处置完成不关闭来源使用许可或整份协议批准。

同一最终ID/OOD清单用于所有方法、seed及骨干。ID按v6原数组索引升序，OOD按val清单原位置升序；original_index和过滤后的evaluation_order严格分列，禁止用后者回读原数组。运行前核对数据、清单hash、顺序及真实标签。

Tiny20类逐类说明见data/class_qualification.csv。依据标注主体与CIFAR-10十类的同义/子类关系；本候选无先定重叠/边界wnid命中。该结论是类别层级，不声称对968图完成全面场景重标注。灰度、低对比度、复杂构图、背景ID物体不能自动剔除。CIFAR-10.1按作者标签，不挑“典型/容易”图。

## 3. 冻结资产和方法

原准备目录data/frozen_assets_verified.csv与旧收尾目录ASSET_MAPPING.csv仍为冻结资产导航；V2的六模型精确路径、SHA256、c_global及活动参考见本目录TECHNICAL_CONTRACT_V2.json。六模型与资产字段和旧合同逐项相同，P11 is_active=False/PROVENANCE_ONLY，VGG只用P13。本轮只对锁定旧样本严格加载模型；实际输入完整性见INPUT_HASH_RECHECK.json，结果见TECHNICAL_RESULTS_V2.json。

方法authority：FINAL_beta040_formal_rebuild_20260824_165921的FORMAL_ASSET_AUTHORITY_MANIFEST.json和FINAL_BETA040_PROTOCOL_FREEZE.json；VGG数值authority：VGG_FP32_TABLE3_CLOSURE/PROTOCOL.json及run_closure.py。不按mtime选版。

- Orbit为Function-only，beta=.40，ResNet Layer3，VGG Block4；不乘q_evi。视图original、brightness×.9、右下1px reflect-shift，原图预测类别固定供三视图梯度。
- 梯度目标为固定类raw logit减其余logit的logsumexp；feature-gradient平方空间均值给channel energy；signature=(energy+1e-12)/(sum energy+C×1e-12)，ResNet C=256、VGG C=512；等权三签名JSD/ln(3)，沿正式函数。
- q_func用对应模型/seed现有类条件reference，按原图预测类别选择，midrank=(N_less+.5 N_equal)/N，float64 searchsorted左右边界；无新数据拟合。
- Global alpha=1+c_global×e；Orbit alpha=1+c_global×exp(-.40 q_func)×e；score=10/sum(alpha)，越大越OOD。不直接缩放alpha，不改变先验1。
- ResNet c_global S0–S2=16.661915817233986、18.482690632268103、24.498207644808492；reference=Stage_P3/protocol/calibration/seed_s_calibration.npz。
- VGG c_global S0–S2=7.3263600911805815、8.23366795313237、8.241928951246619；沿FP32协议复用P13_block4_manuscript_figure_repair/final_repaired/references/Vs_block4_qfunc_reference.npz，不以其他同名reference替代。
- C-EDL C009：T=5、conflict_beta=1、lambda=.75、delta=.5。Softplus(beta=1,threshold=20)的非负evidence输入[N,5,10]，沿cedl_parametric_offline.py的float64实现。五视图均值、intra population std(ddof=0)、pair inter定义、EPSILON=1e-8和clip[0,1]均保留；adjusted_alpha=adjusted_evidence+1。正式C009路径不额外乘c_global，不暗换历史另列matched-scale变体。选择文件只读，不执行搜索。

## 4. 预处理与唯一V2数值方案

RGB uint8 HWC；CIFAR-10.1原生32×32无缩放裁剪。Tiny PIL convert RGB→BILINEAR Resize(32)→CenterCrop(32)，64×64旧输入等价resize((32,32))。无EXIF转向、训练增强或随机裁剪。转连续CHW float32/255；所有视图先在[0,1]原图构造，再按mean=(.4914,.4822,.4465)、std=(.2470,.2435,.2616)标准化。

|步骤|ResNet18|VGG16-BN|
|---|---|---|
|原图与Orbit三视图计算小批次|32，尾批保留|128，沿FP32 closure，尾批保留|
|三方法原图基础产物|同一原图forward共享raw、直接e、p32、S32和固定参考类别；不另用128重算C-EDL原图|closure原图forward共享|
|前向/特征梯度/energy/signature/JSD|CUDA FP32，autocast关闭，Layer3|CUDA FP32，autocast关闭，Block4|
|TF32及后端|matmul=False、cuDNN=False；deterministic=True、benchmark=False、matmul_precision=highest；CUBLAS_WORKSPACE_CONFIG=:4096:8|同左|
|Global/Orbit evidence|p32/S32先升float64再e=max(p×S−1,0)；min<-1e-4停止，负舍入数逐项记录|直接Softplus FP32 e升float64|
|C-EDL CPU变换逻辑批次|128，生成完整视图后拆GPU；原尾批规则|128，原尾批规则|
|C-EDL GPU计算小批次|最多32；view0复用共享基础e；view1–4按生成后原序拆分|128；view0复用closure基础e|
|C-EDL证据与评分|直接Softplus FP32存储，float64 C009，不乘c_global|同左|
|reference/CDF/alpha/拟议metrics/bootstrap|CPU float64|同左|
|loader|workers=0、shuffle=False、drop_last=False|同左|

两种e表示不能假定逐位相同，不平均、不更换定义、不修改阈值或拟合c_global/reference。ResNet P3 reference混合B0 prediction/evidence与P0 d_GOP的继承事实保留；VGG只使用P13，P11仅谱系。V2相对原C-EDL128是一项执行调度修订，理由是方法间统一原图路径，不声明比特等同旧正式运行，不以检测效果作选择。

设备与软件绑定TECHNICAL_CONTRACT_V2.json中记录的本机CUDA:0和RTX5060Ti环境。原适用容差逐项不变：同路径atol1e-6/rtol1e-5；float64评分/CDF atol1e-12/rtol1e-12；恢复e atol2e-5/rtol1e-6；签名归一atol1e-6；JSD范围1e-7。旧跨批次p绝对2e-5、e绝对2e-4、raw atol2e-4/rtol1e-5仍保留给历史断言及诊断，旧S0/S1 FAIL不会被V2 PASS覆盖。没有为S、d或最终跨批次分数新拟合容差。

来源实现对照必须同输入、同计算小批次、同后端；共享原图唯一来源、CPU变换与拆分恢复须逐项验证。发现方法语义不保或原适用门槛失败则停止，不自动试第二方案；设备不足不静默改精度或小批次。本轮技术检查通过不授权正式候选推断。

## 5. C-EDL随机状态与恢复

沿原run_inference.py/materialize_formal.py transformed_batch，CPU NumPy RandomState(20260813)，不是default_rng。每个骨干、seed、域开始重新初始化；独立技术组作为独立工程检查也从同一状态起点开始。拟议正式数据ID先、OOD后，同域最终清单同序，**CPU变换逻辑批次128**。跨seed/骨干同位置变换一致；不逐图重置、不加训练seed、不乱序并行。

每逻辑批view0为原图。view1–4依次先rng.choice(['rotate','shift','add_noise'],size=实际逻辑批长)，再按样本原序逐图抽参数：rotate uniform(-15,15)，OpenCV中心(w//2,h//2)、INTER_LINEAR/BORDER_REPLICATE；shift randint(-2,2)，HWC axis0的np.roll；noise normal(0,.01,image.shape)；clip[0,1]、float32、HWC→CHW。torch.manual_seed(20260813)，model.eval，尾批实际长度不补齐。

完整视图张量生成后才能按原序拆成ResNet32或VGG128的GPU计算小批次。不能在每个GPU小批次重置/重新抽CPU随机变换。view0从共享原图基础缓存按身份取值，不执行另一套原图forward；同一行的raw、e、p、S、固定类别及三方法生产者ID必须一致。

保存每逻辑批trace、视图张量hash、完整RandomState前后状态、身份/位置/拆分表；仅保存noise std不足恢复。恢复限完整原子落盘且hash/行数通过的逻辑批边界，恢复完整RNG与下一原始位置；或从域起点仅重放CPU变换到边界。未完成逻辑批不得接续部分默认分数，应从该批开始前状态重做该批；已生成张量按原序拆分和重组，不另抽视图。原129/新增旧技术130的尾批分别1/2，仅为工程验证，不是新增候选。

## 6. 评价和统计（本轮均未执行）

1. OOD=1、ID=0，score大为OOD。AUROC标准tie平均秩/等价ROC积分。FPR95取首个经验ROC点满足TPR_OOD≥.95，完整相同score阈值块纳入，ID score≥阈值算FPR；无插值、不反向、不挑阈值。
2. 每骨干分别报告三方法×S0–S2的AUROC、FPR95和seed等权均值；主要Orbit−Global，次要Orbit−C-EDL，每seed差值也报告。AUROC差为正有利、FPR95差为负有利；指标都保留。
3. 拟5000次同步配对样本bootstrap，**两域均按真实类别分层并固定各类当前纳入数**。这是新协议，不能回写成既有论文Tiny sample-level bootstrap。
4. numpy.random.Generator(numpy.random.PCG64(20260912))；replicate0..4999。每次先ID标签0..9，后OOD wnid字典序；每类以原manifest类内位置为池，有放回抽该类原数量，保存计划/位置/hash。
5. 同一抽样位置同时用于所有方法、seed及骨干；逐骨干先算每seed配对差，再取三个差的算术平均。不能把3×样本量作为独立观测，不能把不同seed分数拼接扩大测试集。
6. 双侧95% percentile CI=np.quantile(replicates,[.025,.975],method='linear')。点估计用完整样本，不以bootstrap均值替代。不在重采样中训练/拟合/重算reference。区间作固定比较描述，不作未规定的多重显著性胜利声明；全部两骨干×两比较×两指标区间都报告。
7. 区间条件于固定模型、配置和本次样本/类构成，不能覆盖未观察类别、全部训练随机性或历史开发选择。不承诺该规模检出显著差异。
8. 正/负/跨零全部报告；不得据此换数据、补图、改beta/层/视图、挑seed或删比较。资格剔除仅来自事先规则，不按性能决定。不得利用新样本训练、调参、校准或自动改论文。

## 7. 已执行的旧样本技术检查与证据界限

TECHNICAL_SAMPLES_V2.csv固定：原OLD_ID_DEVELOPMENT129（真实标签全0；身份/顺序不变，包含1759、45886），OLD_ID_MULTICLASS130（同一旧开发池按真类0–9每类原顺序前13），旧OLD_TINY_TECHNICAL32（原位置0–31）。ID两组重叠13，因此246唯一ID、259次组内呈现；加Tiny共278唯一身份、291次呈现/模型。真实类覆盖和预测类覆盖分别记录，禁止用模型输出替换技术样本。本轮六模型在130组实测预测类均覆盖十类；CDF合成ties/端点检查另列。

两骨干各S0–S2在三组上共18条链全部通过；S0/S1此前未执行的旧Tiny链已完成。V2使用原适用门槛，没有删种子、训练、调参、重新校准或改变层/视图/参考。CPU变换trace/hash/完整状态恢复与原logical128相同，split/rejoin保持身份顺序。

旧原图直接Softplus32/128 evidence断言S0/S1仍FAIL，最大差分别0.00021457672119140625和0.00021791458129882812，超过原绝对2e-4；这不涉及p×S−1表示因素。原129新鲜诊断的S1有1张CDF位置变化（源索引[49656]），不能声称“误差小即排序不变”。P3恢复e与直接e差异单独保存，不以共享forward消除其语义区别。

Global/Orbit继承原图类别，raw/alpha argmax仅诊断；C009预测独立按其原评分实现输出。原P0混合来源衔接继续DIAGNOSTIC_ONLY_NOT_REPLAY；没有对应正式FP32/相同后端历史完整载荷时维持NOT_REPLAYED，VGG当前源码重执行通过不当作正式历史逐位回放。旧TF32-on Tiny载荷不是当前TF32-off比特标准。

本轮不计算真实图像的AUROC/FPR95或检测效果；第6节仍只是待批准统计方案。所有新增CIFAR-10.1及Tiny968的模型前向、特征、梯度、分数、性能均NOT_RUN。

## 8. 统一批准闸门与停止

1. D01074/D01132本次保守资格排除已明确授权并完成，视觉关系UNRESOLVED；不改写为确认重复，不补样本。
2. CIFAR-10.1图像/标签使用依据保持NEEDS_AUTHOR_OR_INSTITUTION_CONFIRMATION。代码MIT不等于图像/标签许可；原Tiny Images历史声明、派生v6条件和本地研究使用依据分别处理。旧网页目录索引或历史文本均不单独关闭v6使用事项，本轮不循环网页搜索。
3. 需要一次由数据作者或学校适当负责人确认：本次学术评估适用依据及文件/日期、原Tiny Images声明对v6的范围，以及引用、结果发表、图像展示、保留、再分发限制。未获答复不得填“已获许可”。
4. 单一V2工程方案已经完成旧样本检查，旧FAIL与历史回放缺口仍按第7节保留。技术通过不代表正式新增样本评估获准。
5. 最终清单EVALUATION_MANIFEST_PENDING_V2.csv及本整份协议仍需作者统一批准；来源依据未确认前保持待决，不先跑其中一部分。

不回填每类200、不补齐2000、不扩展Tiny，不改名为未筛选原全集。数据名称为“CIFAR-10.1 v6资格筛选子集”。完成两份交付后停止，不修改论文或旧正式结果。
