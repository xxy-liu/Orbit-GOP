# Feature-layer publication authority

**APPROVED FOR MANUSCRIPT**

Authority ID：`FINAL_PUBLICATION_LAYER_ABLATION_AUTHORITY`  
实际保存目录：`PROJECT\outputs\FINAL_LAYER_ABLATION_AUTHORITY_20260907_214436`

本次指定 Layer2/Layer3/Layer4 的统一 fresh-forward development 点估计为稿件来源，仅在协议身份、Layer3 当前 Full Orbit 逐样本身份、全部两次重复、独立 QA 与全项目 no-mutation 全部 PASS 后生效。

| Layer | AUROC | FPR95 | AUROC new-old | FPR95 new-old |
|---|---:|---:|---:|---:|
| Layer2 | 0.86731460 | 0.5235 | +0.00001460 | -0.0005 |
| Layer3 | 0.87842696 | 0.4110 | +0.00002696 | -0.0005 |
| Layer4 | 0.86995556 | 0.4725 | +0.00005556 | -0.0005 |

- protocol_identity: PASS
- execution: PASS
- independent_QA: PASS
- no_mutation: PASS
- Layer2_repeat: PASS
- Layer3_repeat: PASS
- Layer4_repeat: PASS
- Layer3_identity: PASS

固定 ResNet18-EDL Seed-0、beta=0.40、c_global=16.661915817233986、q_func only、三视图、formal decision margin、normalized generalized JSD、每层自身 calibration reference。CUDA FP32，AMP/TF32 off，batch=32，后处理保留正式 float64 路径。Calibration=3000；CIFAR-10 ID development=2000；CIFAR-100 development=12500；OOD positive=True。

正式三行 source：`FINAL_TABLE4_FEATURE_LAYER_SOURCE.csv`；SHA-256：`11f24e5c160766351d1762176a48a06dfa1d5682568ac1a75d5ef377c23946e3`。六行组合：`FINAL_TABLE4_COMBINED_SOURCE.csv`。

正式层仍为历史冻结 Layer3，无新配置选择。保留所有不利结果和两次 run；无官方测试、holdout、Tiny ImageNet 实验使用，无训练或调参，无统计推断。稿件及历史资产均未修改。完整证据与边界见 `FINAL_LAYER_ABLATION_REPORT.md`。
