# Final view-configuration publication authority

**最终结论：APPROVED FOR MANUSCRIPT**

Authority ID：`FINAL_PUBLICATION_VIEW_ABLATION_AUTHORITY`  
记录时间（UTC）：2026-09-07T13:34:06.819005+00:00  
实际保存目录：`PROJECT\outputs\FINAL_VIEW_ABLATION_PUBLICATION`

## 1. 正式决定与适用范围

本记录正式指定下列统一 FP32 fresh-forward development 三配置结果为 `FINAL_PUBLICATION_VIEW_ABLATION_AUTHORITY`。本记录是新的 publication-authority transition，解除这三行用于稿件的历史来源冲突；不把旧 historical reproduction FAIL 改写为 PASS，也不放宽或追溯修改旧协议的 1e-5 门槛。

| View configuration | AUROC | FPR95 |
|---|---:|---:|
| Brightness-only | 0.87738508 | 0.4225 |
| Reflect-shift-only | 0.87766472 | 0.4155 |
| Full Orbit | 0.87842696 | 0.4110 |

三行正式稿件数据源为 [FINAL_TABLE4_VIEW_CONFIGURATION_SOURCE.csv](FINAL_TABLE4_VIEW_CONFIGURATION_SOURCE.csv)，SHA-256：`d79476c93201a6b581967cb9f510527a3ba1a196ac09ef13c2e2f004bf321bf8`。CSV 仅含表头与上述三行，值为 0–1 单位，不包含 feature-layer 或其他结果。

共同定义：ResNet18-EDL Seed-0；Layer3；beta=0.40；risk=q_func only；formal decision margin（固定 original predicted class，z_c − logsumexp(z_others)）；formal normalized generalized JSD；每种配置使用自身 configuration-specific calibration reference，均由冻结 CIFAR-10 calibration 3000 个样本构建。c_global=16.661915817233986，未重新拟合。Fresh-forward batch=32，FP32，AMP/TF32 关闭；风险/uncertainty 后处理保留冻结协议规定的 float64 路径。“统一 FP32”不表示所有后处理都改为 FP32。

Performance 仅为 CIFAR-10 ID development（train validation，n=2000）vs CIFAR-100 development（train，n=12500），OOD 为 positive。它们是固定 Seed-0 development 描述性结果，不构成跨 seed、跨设备、显著性或泛化最优结论。

## 2. 历史 FAIL 保留及根因

原 `PROJECT\outputs\FINAL_VIEW_ABLATION_AUTHORITY_REPAIR_20260907_204813\VIEW_ABLATION_REPAIR_REPORT.md`、旧协议、旧 Status/Publication_authority_status、诊断记录及所有历史资产均保留原始字节。旧报告中的 STOP、historical reproduction FAIL 和当时的 publication BLOCKED 是历史任务的有效结论，保持不变。当前稿件三配置 authority 以本新记录为准。

旧 FAIL 仅表示：统一 fresh-forward Full Orbit AUROC=0.87842696 未能在预设绝对容差 1e-5 内复现历史混合数值路径的 0.87840948；既有诊断记录的差为 0.00001748。它不表示新三配置的数据完整性、数学定义或重复执行核验失败。本次未重新计算 AUROC/FPR95 或该历史诊断。

根因已由既有证据定位：历史 ID/calibration evidence 继承 B0 CUDA FP16/autocast validation forward；历史 d_GOP/reference 来自较早的冻结提取批次，与本次统一 FP32 fresh-forward 及各配置新建 reference 的数值路径不同。FP16/autocast 指 evidence 的来源，不将全部历史 reference 运算笼统称为 FP16。主要差异来自 evidence，其次是 calibration/ID d_GOP 与 empirical reference；历史 CIFAR-100 development Full Orbit d_GOP 与 fresh-forward 逐样本一致。checkpoint、样本索引、标签与 base prediction 身份一致。根因记录为 `ROOT_CAUSE_IDENTIFIED_AUTHORITY_CONFLICT`；历史来源拆分只用于归因，没有将替换组合当作候选进行择优。

## 3. 两次独立执行与完整性依据

新三配置均已进行两次独立执行（准确范围：同一运行环境、相同 batch/order 的两次完整 forward/gradient 提取）。V1/V2/V3 的 calibration、ID development、OOD development 共 9 项 rerun_exact 全部为 true；逐样本 prediction/evidence/d_GOP 精确一致，支持固定确定性后处理下的一致结果。此处不声称两次独立进程、不同设备或不同环境的外部复现，也没有在本次重新执行。

既有 `INDEPENDENT_QA.json` 为 PASS 71/71，包含 52,500 行、样本唯一性、索引/标签/预测身份、配置自身 reference、q_func 与独立指标核验。本次只读取既有结论并检查其 SHA-256、协议及数值来源一致性，没有运行既有核验脚本或解码逐样本数组重新计分。相关证据均与历史 ARTIFACT_MANIFEST 和本次前置基线一致。

## 4. 数据边界、无选择与不利变化

本次 authority transition 的模型执行、AUROC/FPR95 重算、效率重跑次数均为 0；无训练、调参、checkpoint 选择、配置选择或新的 reference 拟合。上述三项 performance 的生成及本次 publication 指定均没有 official-test、holdout 或 Tiny ImageNet 性能评价；没有使用这些数据选择 beta、layer、views 或其他配置。

既有 Figure 8(b) closure 的 latency 曾按当时授权使用 CIFAR-10 official test 0:2048 作为计时输入池，未计算 official-test performance。该历史事实必须保留，因此“没有 official test”专指本次无实验使用及三项 development performance 的数据边界，不能扩展成整个历史 timing 流程从未接触 official-test 数据。本次全项目 no-mutation 审计会读取文件原始字节用于哈希，包括这些数据资产；不解码数据、不运行模型、不计算性能。

新三配置整体迁移，全部保留，没有配置选择，也没有挑选改善的指标。Brightness-only FPR95 从旧稿 0.4220 变为 0.4225（恶化 0.0005，即 0.05 个百分点）；旧 Full Orbit 历史复现 FAIL 亦保留。因此本次不是择优迁移。Full Orbit 在这三项固定 development 点估计中 AUROC 最高、FPR95 最低只能作为描述，不能作为配置晋升、显著性或普遍优越性依据。

## 5. Table 4 与 Figure 8(b) 的边界

**Table 4 的 feature-layer 三行（Layer2 / Layer3 / Layer4）不属于本次 authority transition，不修改，也不重新认证其 authority。** 本次不生成混合新旧的六行表，不修改论文、旧 Table 4、Table 7 或任何历史图件。

Figure 8(b) performance source：`PROJECT\outputs\FINAL_FIG8B_MATCHED_CLOSURE\FINAL_FIG8B_PERFORMANCE_COST_SOURCE.csv`。该文件三项 AUROC 和三项 FPR95 均已与本次正式 CSV 和 fresh-forward development CSV 使用 Decimal 零容差逐项比较，全部完全一致；0.411 与 0.4110 仅为文本尾零差异。配置名称、reference ID/SHA-256 与 performance 来源哈希也一致。该 Figure 8(b) 文件保持原始字节，包括其 Full Orbit 历史 `Performance_source_status=FAIL`；该字段继续表示旧复现门槛，不否定本记录的新 publication 指定。

既有 configuration-matched closure 为 PASS 27/27。本次未重跑效率，未重新制图，也未修改任何 latency/SD。Performance batch=32、latency batch=64 的原有边界保留；不声称跨 batch 数值逐位一致。本记录仅将图中已有三项 performance 与最终 publication authority 对齐。

## 6. 本次 no-mutation audit

审计结果：**PASS**。全项目 `PROJECT` 所有文件实际读取原始字节计算前后 SHA-256，唯一排除本次新建输出目录 `PROJECT\outputs\FINAL_VIEW_ABLATION_PUBLICATION`。

- 基线文件数：273092；结束文件数：273092。
- 历史文件内容修改：0；删除：0；本次目录外新增：0。
- Protected baseline integrity：PASS。
- 逐文件前后证据：`_audit/FILES_BEFORE.json`、`_audit/FILES_AFTER.json`；结果：`_audit/NO_MUTATION_AUDIT.json`。
- 本次来源核验：PASS 77/77，见 `_audit/AUTHORITY_QA.json`。审批记录与最终 CSV 的交付哈希另存 `_audit/DELIVERY_SHA256.json`，避免文档自引用哈希。

全项目审计覆盖旧 repair FAIL 报告、Figure 8(b)、feature-layer 来源、checkpoint、数据、稿件与历史效率资产。审计结束后只在本次排除的新目录内定稿本记录和交付哈希，没有修改历史资产。

## 7. 来源 SHA-256

以下均为本次实际核对的既有文件，路径前缀为 `PROJECT\outputs`。

| 既有来源 | SHA-256 |
|---|---|
| `FINAL_VIEW_ABLATION_AUTHORITY_REPAIR_20260907_204813/VIEW_ABLATION_REPAIR_REPORT.md` | `a12361910e284efb5229d7d3dfa3cafedbcb93e2460f3c3837f098d343af3be0` |
| `FINAL_VIEW_ABLATION_AUTHORITY_REPAIR_20260907_204813/VIEW_ABLATION_REPAIR_PROTOCOL.json` | `8a58e505471182466984bba8ecdadd248eacf4888f14f55d45015dd70e38e972` |
| `FINAL_VIEW_ABLATION_AUTHORITY_REPAIR_20260907_204813/PROTOCOL_HASH_LOCK.json` | `a5913aba02493dcfbb6b653de9cd7a786728bc442321369500da1932e5573ec9` |
| `FINAL_VIEW_ABLATION_AUTHORITY_REPAIR_20260907_204813/EXECUTION_QA.json` | `fc84b524aac70b58c24d4729739c46b1195108fd119c802bab47752b7126d7b7` |
| `FINAL_VIEW_ABLATION_AUTHORITY_REPAIR_20260907_204813/INDEPENDENT_QA.json` | `b464ca70673f864fc989f6cd594d6c17a1a9eacc5195e4ae94511e536b26e53f` |
| `FINAL_VIEW_ABLATION_AUTHORITY_REPAIR_20260907_204813/FULL_ORBIT_REPRODUCTION_DIAGNOSIS.json` | `d1b9321a00853a7755537432ba70630e3e5781da2f2682de6b20afa22a0b1457` |
| `FINAL_VIEW_ABLATION_AUTHORITY_REPAIR_20260907_204813/FULL_ORBIT_LINEAGE_DIAGNOSTICS.csv` | `a9c6f11f27edcbace240e3ad0cc024c0f380f9c7dc9c449962b47ef394a9bdc0` |
| `FINAL_VIEW_ABLATION_AUTHORITY_REPAIR_20260907_204813/FINAL_VIEW_ABLATION_DEVELOPMENT_RESULTS.csv` | `eb957f28466e0deb407b1cca8c75afaf57ece310c89738be242a071352192133` |
| `FINAL_VIEW_ABLATION_AUTHORITY_REPAIR_20260907_204813/FINAL_TABLE4_VIEW_CONFIGURATION_AUTHORITY.csv` | `172e90d8407d1ff0df4d74efbc7b3cf0f2c2f2a861986fd3b3c467932f3e8430` |
| `FINAL_VIEW_ABLATION_AUTHORITY_REPAIR_20260907_204813/FINAL_VIEW_ABLATION_SAMPLE_SCORES.parquet` | `a4a825ce8a9c979b5a6240d644e510914b9df8a5c2c46023b9ea78ba5e501865` |
| `FINAL_VIEW_ABLATION_AUTHORITY_REPAIR_20260907_204813/DATA_ACCESS_LOG.json` | `f4e7ad4b27b212232fa8111baa8c6a364c63563af0ef3ca0a67d5331231a044d` |
| `FINAL_VIEW_ABLATION_AUTHORITY_REPAIR_20260907_204813/run_repair.py` | `1a47c297d39461b912fa7a4f12fd29d7d39dc0aec62391f47591a0c90050a9f9` |
| `FINAL_VIEW_ABLATION_AUTHORITY_REPAIR_20260907_204813/V1_calibration_reference.npz` | `fe0df1c38ab7be97eff45f6eed44a2a1026fe0134f283444a41475ebc2d24d1d` |
| `FINAL_VIEW_ABLATION_AUTHORITY_REPAIR_20260907_204813/V2_calibration_reference.npz` | `55349af5fc71244f3994ea0de104c6fc0d84c0659586786c55acf314728a2b5a` |
| `FINAL_VIEW_ABLATION_AUTHORITY_REPAIR_20260907_204813/V3_calibration_reference.npz` | `adc3ead98d9bdc36aeb2731ae662d4cc915ccba3b31b7063b961caadb7ded776` |
| `FINAL_FIG8B_MATCHED_CLOSURE/FINAL_FIG8B_PERFORMANCE_COST_SOURCE.csv` | `5d04c45e0ba23f2d097de194bf190c7cb5c071998b8c967a87b3c0e31a5a3308` |
| `FINAL_FIG8B_MATCHED_CLOSURE/FINAL_FIG8B_CLOSURE_REPORT.md` | `243fa34920a19d6d780e82fff3811c879befce5caceb29d804eef930d9a42213` |
| `FINAL_FIG8B_MATCHED_CLOSURE/FINAL_FIG8B_QA.md` | `728ae0250741ece35ea3e27ed9782792b35af1a9390a0343a0c3ff66569e704a` |
| `FINAL_FIG8B_MATCHED_CLOSURE/FINAL_QA.json` | `a1ffc4d9c1abd54cb33b04f6ea5ecf3a4b48e07cf6b7dad4900f3bdcf7d1469f` |
| `FINAL_FIG8B_MATCHED_CLOSURE/CONFIGURATION_IDENTITY.json` | `faaa4fc6c24351ca746eebdcbe02636d5d6f3abdf2688d959954abf60655761f` |

**最终结论：APPROVED FOR MANUSCRIPT。** 此结论仅授权上述三项 fresh-forward development view-configuration performance 作为稿件 publication authority；historical reproduction FAIL 保留，feature-layer authority 不在本次范围。
