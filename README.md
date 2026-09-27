# Orbit-GOP

Orbit-GOP detects near out-of-distribution inputs in evidential classifiers by measuring how gradient-derived functional signatures change across three fixed views. A predicted-class calibration rank controls a positive, shared evidence scale. The operational prediction remains the base classifier's prediction.

This candidate accompanies *Orbit-GOP: Near-Out-of-Distribution Detection for Evidential Deep Learning via Cross-View Functional Stability* by Xue Liu, Shangzhu Jin, Mou Zhou and Jun Peng. Public disclosure of this exact candidate has been approved by the project rights holder(s); see [release authorization status](RELEASE_AUTHORIZATION_STATUS.md). No full-code tag or DOI is assigned here.

## Code availability and usage

The Orbit-GOP implementation is publicly available for academic review, research verification, reproducibility, and non-commercial academic research. This repository is source-available but is not released under a permissive open-source license. Commercial use, redistribution, relicensing, or public distribution of modified versions requires prior written permission from the copyright holders. Third-party components remain subject to their respective licenses. See [USAGE_NOTICE.md](USAGE_NOTICE.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## What is included?

| Scope | Included material | Boundary |
|---|---|---|
| Confirmatory S3–S5 | ResNet18, VGG16-BN and WRN-28-10 definitions; training configuration; Orbit-GOP implementation; view construction; functional signature; normalized dGOP; class-conditioned q_func; evidence calibration; metrics; and confirmatory analysis code | Nine recorded model identities and frozen aggregate results are retained. Original model weights and per-sample scores are absent. |
| Historical S0–S2 | Selected frozen result artefacts, figure coordinates and provenance | Historical archive executors are not distributed and are not part of the portable pipeline. See [historical archive scope](docs/HISTORICAL_ARCHIVE_SCOPE.md). |
| Attribution | Adapted third-party portions, source records and required license texts | See [third-party notices](THIRD_PARTY_NOTICES.md) and [source provenance](docs/THIRD_PARTY_PROVENANCE.md). Third-party licenses do not license original Orbit-GOP contributions. |

The confirmatory code is under [`orbit_gop/`](orbit_gop), [`models/`](models), [`baselines/`](baselines), [`training/`](training) and [`scripts/confirmatory/`](scripts/confirmatory). The 44k/3k/3k split is under [`splits/confirmatory/`](splits/confirmatory); per-seed and across-seed results are under [`results/confirmatory/`](results/confirmatory). The nine checkpoint identity records are in [`provenance/confirmatory/manifests/`](provenance/confirmatory/manifests). The WRN S5 AMP overflow and subsequent engineering repair are disclosed in the retained [bugfix record](provenance/confirmatory/v2/BUGFIX_LOG.md); the S3/S4 and S5 checkpoint identities remain distinct.

## Minimal example: synthetic CPU check

From the repository root, with the dependencies in [requirements.txt](requirements.txt) available:

```sh
python -m unittest discover -s tests -p "test_core_cpu.py" -v
```

This uses synthetic images and checks all three backbones, signature normalization, dGOP bounds, mid-rank ties, prediction inheritance and metric behavior. It does not access datasets, checkpoints or frozen results. To print saved tables without recomputation, run `python -m scripts.reproduce_tables.show_tables`.

## Confirmatory analysis logic

The actual S3–S5 study used seeds 3, 4 and 5 for each of the three backbones. [`prepare_run`](scripts/confirmatory/prepare_run.py) creates a **new** locked run with independently obtained datasets. [`train_confirmatory`](scripts/confirmatory/train_confirmatory.py) and [`evaluate_confirmatory`](scripts/confirmatory/evaluate_confirmatory.py) are the genuine training and evaluation entries. Evaluation fits 3,000-sample calibration references, uses the 44,000-sample feature bank, and writes paired score arrays. [`analyze_confirmatory`](scripts/confirmatory/analyze_confirmatory.py) implements the fixed 5,000-replicate confirmatory analysis after a complete new set of scores exists.

```sh
python scripts/confirmatory/analyze_confirmatory.py --help
python -m scripts.confirmatory.prepare_run --help
python -m scripts.confirmatory.evaluate_confirmatory --help
```

The help commands only describe the interfaces. Actual training/evaluation/analysis needs authorized datasets, all nine appropriate checkpoints or a complete new training run, and a separate `ORBIT_RUN_DIR`; it has not been executed as part of this release repair. The included [`configs/confirmatory/`](configs/confirmatory) and [method documentation](docs/reproducibility.md) identify the frozen design and what a new run can verify. S0–S2 references must not be mixed with S3–S5 models.

## Reproducibility levels

| Level | Release status | Meaning |
|---|---|---|
| R0 — repository/data structure | **VERIFIED** | Files, identities, splits, hashes and package structure checked. |
| R1 — syntax/import/static checks | **VERIFIED** | Python syntax, selected safe imports and help entries checked. |
| R2 — minimal CPU-safe functional smoke | **VERIFIED** | Synthetic core checks; no dataset or checkpoint replay. |
| R3 — complete paper reproduction | **NOT PROVIDED BY THIS RELEASE** | Exact regeneration of every frozen numerical result is outside this package. |

The public release focuses on the portable implementation and confirmatory analysis logic. Exact regeneration of all frozen numerical results additionally requires the original trained checkpoints and per-sample score artefacts, which are not distributed in this repository. The complete original 5,000-replicate bootstrap outputs cannot be regenerated from this package alone; historical figure executors are not fully distributed. Datasets must be obtained independently under their own terms. Selected historical numerical evidence remains frozen and is not regenerated by this release. See [reproducibility boundary](docs/reproducibility.md), [dataset instructions](docs/datasets.md), [paper-to-source mapping](docs/FIGURE_SOURCE_MAPPING.md), and [paper coverage](provenance/PAPER_COVERAGE.csv).

## Environment and integrity

The paper execution record lists Python 3.10.10, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128 and CUDA 12.8. Release checks used CPU execution in a Python 3.10.10 environment with the recorded PyTorch and torchvision package versions; they did not reinstall those wheels, exercise CUDA training, or establish R3. The separate efficiency environment and scope are recorded in [environment.md](docs/environment.md). [`requirements-optional.txt`](requirements-optional.txt) lists `polars` only for the optional offline C-EDL utility; the Orbit-GOP core does not import it.

[`MANIFEST.csv`](MANIFEST.csv) and [`SHA256SUMS.txt`](SHA256SUMS.txt) identify the candidate payload. [`CITATION.cff`](CITATION.cff) gives the manuscript citation without claiming publication acceptance. No GitHub push or Release is performed by this package preparation.
