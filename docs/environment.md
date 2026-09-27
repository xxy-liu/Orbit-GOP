# Recorded efficiency environment

Source: original Supplementary Note S1.1, now Repository Note R1.1. These are recorded historical versions, not versions queried from the current computer. Scope is the efficiency benchmarks only.

| Component | Recorded value |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Ti |
| CPU | AMD Ryzen 5 9600X 6-Core Processor |
| Operating system | Windows-10-10.0.26200-SP0 |
| Python | 3.10.10 |
| PyTorch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| NumPy | 1.26.4 |
| CUDA runtime, torch.version.cuda | 12.8 |
| cuDNN version code | 91900 |
| NVIDIA driver | 591.86 |
| Efficiency precision | float32; AMP and TF32 disabled |
| Efficiency checkpoint | Frozen ResNet18-EDL Seed-0 |

The efficiency Orbit-GOP configuration uses Layer3, beta = 0.40, function-only risk, the decision-margin target and normalized generalized Jensen–Shannon divergence. No unrecorded RAM or GPU memory capacity is asserted.

Confirmatory training is different: float16 CUDA convolutions under AMP with FP32 Softplus and EDL loss. Confirmatory inference disables autocast and TF32. Separate recorded confirmatory environment files are under `provenance/confirmatory/environment/`; they do not replace this efficiency record.
