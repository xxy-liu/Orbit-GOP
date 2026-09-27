# Final scientific freeze check

**Result: PASS for this release-policy edit.** The check compares the candidate with the 257-file pre-edit candidate archive and verifies the separate 352-file `GITHUB_RELEASE_FINAL` source tree against its recorded SHA256 baseline. Every retained file under `results/`, `splits/`, `configs/`, `orbit_gop/`, `models/`, `training/`, `scripts/`, `baselines/`, and `tests/` is byte-identical to the pre-edit candidate. The original tree has 0 missing, changed, or extra files. No model training, checkpoint evaluation, original Bootstrap, or scientific-result generation was performed.

| Frozen item | Direct candidate evidence | Check |
|---|---|
| beta = 0.40 | `configs/confirmatory/confirmatory_locked.yaml`, `orbit_gop/_scores.py` | PASS |
| Full Orbit | `orbit_gop/views.py`: Original, Brightness ×0.9, Reflect-shift | PASS |
| Feature hooks | Locked config and `orbit_gop/functional_signature.py`: ResNet18 Layer3; VGG16-BN Block4 before pool4; WRN Block3 before final BN/ReLU | PASS |
| Calibration n = 3000 | Locked config and confirmatory split | PASS |
| Confirmatory versus historical | S3/S4/S5 remain confirmatory; S0/S1/S2 remain historical in separate result, split, and provenance paths | PASS |
| CIFAR-10 roles | Confirmatory CSV has 50,000 unique sample IDs: train 44,000; checkpoint validation 3,000; calibration 3,000 | PASS |
| q_func and ties | `orbit_gop/functional_risk.py`: predicted-class empirical mid-rank, `(N_< + 0.5*N_=)/N` | PASS |
| dGOP | `orbit_gop/functional_risk.py`: equal-weight generalized Jensen–Shannon divergence normalized by `log(V)` | PASS |
| Evidence retention | Locked config and `orbit_gop/_scores.py`: `exp(-beta*q_func)` with beta 0.40 | PASS |
| Prediction | Locked config and `orbit_gop/inference.py`: operational prediction inherits the original-view base prediction | PASS |
| Across-seed SD | Locked config specifies sample SD, `ddof=1`, across S3/S4/S5 | PASS |
| Bootstrap | Locked configuration specifies 5,000 synchronized paired replicates | SPECIFICATION PRESERVED; NOT RERUN |

Independent anchor SHA256 values: `configs/confirmatory/confirmatory_locked.yaml` = `4929fb9853af2dd35f9a979ca1c7619f0158e52910551686ef91b526aabe1009`; `orbit_gop/views.py` = `ccd4e0781d5d78ccdbfa7792e11e947a2014e54478a38298ddf6dc37a28e3fd5`; `orbit_gop/functional_risk.py` = `6dbe913e348f533fc64c2da9a580e5714d4f0f01aff26d2aaf408ad825a8536b`; `orbit_gop/inference.py` = `7eeecc2409824f7ccf120c831f07dfdfa9cbdf4dbc3b016e4d3f87869e084f13`; `scripts/confirmatory/analyze_confirmatory.py` = `1e76af8a298c89c4a054c04d0949c8e66773b1eb7996aafe4186231d2aaa0bf3`; `splits/confirmatory/cifar10_confirmatory_split_v1.csv` = `3f4b54f85e66aa6b694c020012faa30171a360d83f979da70762cc3afda90e5e`.

The freeze check establishes unchanged released content, not R3 end-to-end regeneration. Original nine weights, original per-sample scores, source datasets, and complete historical figure executors are absent. Existing historical `FAIL` and pending experiment-protocol records remain in their original scope.
