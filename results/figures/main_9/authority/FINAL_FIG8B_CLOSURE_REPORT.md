# Figure 8(b) 最终 configuration-matched efficiency closure

**本次 closure：PASS。** 三种配置均已重新完成 Batch-64 计时，并按自身正式 calibration reference 与协议冻结 development performance 组合。

实际输出目录：`PROJECT\outputs\FINAL_FIG8B_MATCHED_CLOSURE`。

| View configuration | Batch-64 latency (ms), mean ± SD | Frozen AUROC | AUROC (%) | Frozen FPR95 |
|---|---:|---:|---:|---:|
| Brightness-only | 32.444087 ± 2.902984 | 0.87738508 | 87.738508 | 0.4225 |
| Reflect-shift-only | 28.803781 ± 0.011484 | 0.87766472 | 87.766472 | 0.4155 |
| Full Orbit | 44.241128 ± 1.284110 | 0.87842696 | 87.842696 | 0.4110 |

Brightness-only 五轮为 35.047641、34.594329、33.948926、29.793028、28.836513 ms；Full Orbit 五轮为 46.235769、44.821458、43.279097、43.599110、43.270206 ms。两者存在轮间下降和波动，均完整保留并体现在 SD 中。固定软件/硬件身份不能证明运行负载、时钟或热状态逐时相同；本次未同步记录这些运行状态，因此不将成本差异因果归结为 reference，也不据这些观测重新选择配置或轮次。

## 配置和计时协议

ResNet18-EDL Seed-0，冻结 checkpoint，Layer3，beta=0.40，q_func only，formal decision margin 和 formal normalized generalized JSD，c_global=16.661915817233986。Brightness-only 为 Original + Brightness 0.90；Reflect-shift-only 为 Original + Reflect-shift 1px；Full Orbit 包含三视图。
GPU NVIDIA GeForce RTX 5060 Ti，CPU AMD Ryzen 5 9600X；Python 3.10.10、PyTorch 2.11.0+cu128、torchvision 0.26.0+cu128、NumPy 1.26.4、CUDA runtime 12.8、cuDNN 91900、driver 591.86。完整环境及所有冻结 flag 与父 campaign 一致；CPU/interop threads=6，float32 GPU / float64 CPU risk，AMP/TF32 关闭。
复用 FINAL_EFFICIENCY_CONFIRMATION_20260907_202800 的 scorer 原始字节。每配置 50 warmup，5 repeats，200 batches/repeat，Batch=64；累计 150 warmup、3000 timed calls。每轮 perf_counter 前后 CUDA synchronize；warmup 每 batch 同步。SD 为五轮 mean 的样本标准差，ddof=1。无轮次删除、择优或追加重测。
输入池为 CIFAR-10 official test 0:2048，CPU pinned preload 与父 campaign 的输入哈希完全一致。确定性选择：absolute=repeat_id*200+iteration_id；start=(absolute*64)%(2048-64+1)。只用于计时，不评估 official-test performance，不读取 OOD 来计算指标。
计时包含 CPU slicing、H2D、视图构造、normalize、forward、gradient、GOP/JSD、CPU 返回、midrank、exp、c_global、alpha/uncertainty、形状和有限性检查；排除加载、decode/preload、模型/reference 初始化与绑定。warmup 和计时边界以 AST 与父脚本核对。
方法顺序为 Brightness-only、Reflect-shift-only、Full Orbit。Full Orbit 是本次同时期 control，仅进入 Figure 8(b)，未修改 Table 7 当前正式结果。

## Reference 与 performance 溯源

- Brightness-only: `V1_sha256_fe0df1c38ab7be97eff45f6eed44a2a1026fe0134f283444a41475ebc2d24d1d`；SHA-256 `fe0df1c38ab7be97eff45f6eed44a2a1026fe0134f283444a41475ebc2d24d1d`；reference n=3000。
- Reflect-shift-only: `V2_sha256_55349af5fc71244f3994ea0de104c6fc0d84c0659586786c55acf314728a2b5a`；SHA-256 `55349af5fc71244f3994ea0de104c6fc0d84c0659586786c55acf314728a2b5a`；reference n=3000。
- Full Orbit: `V3_sha256_adc3ead98d9bdc36aeb2731ae662d4cc915ccba3b31b7063b961caadb7ded776`；SHA-256 `adc3ead98d9bdc36aeb2731ae662d4cc915ccba3b31b7063b961caadb7ded776`；reference n=3000。

冻结 performance 读取自 `PROJECT\outputs\FINAL_VIEW_ABLATION_AUTHORITY_REPAIR_20260907_204813\FINAL_VIEW_ABLATION_DEVELOPMENT_RESULTS.csv`，SHA-256 `eb957f28466e0deb407b1cca8c75afaf57ece310c89738be242a071352192133`。本次只解析三行 CSV 并与既定值逐项匹配；未读取样本分数重新计算 AUROC/FPR95。

源记录仍保留 Full Orbit 历史复现门限 FAIL：旧 AUROC=0.87840948，新冻结值=0.87842696，绝对差=0.00001748，超过旧门限 0.00001。本任务按正式指定的新冻结值完成成本闭合；没有放宽旧门限、改写源 Status 或解除旧 Table4 authority 状态。
performance 提取 batch=32，latency 按任务要求 batch=64；模型、数学定义、视图与 configuration-specific reference 一致，不声称跨 batch 浮点值逐位相同。

## 图与证据边界

x=Batch-64 latency (ms)，y=AUROC (%)；横向误差条为五轮 latency SD，没有 AUROC CI，不进行显著性推断或配置选择。图内小字为 ResNet18-EDL · Seed-0 development；110×82 mm，SVG/PDF 可编辑文本，PNG 600 dpi。图仅报告本次性能—成本坐标，不将小幅 AUROC 差异解释为显著改进。

## No-mutation

PASS：基线 273060 个文件；修改 0，删除 0，当前目录外新增 0。全项目两次实际 SHA-256，唯一排除本输出目录；含 checkpoint、datasets、performance assets、Table7 和所有历史效率结果。模型参数及 buffers 前后哈希一致。

未训练、未调参、未修改 checkpoint、未重算 AUROC/FPR95、未运行 official-test 性能评估、未选择配置、未覆盖历史效率结果。完整检查见 FINAL_FIG8B_QA.md；15 轮原始计时见 TIMING_REPEATS.csv。
