# Data acquisition and layout

Download each dataset yourself from its official source under its applicable terms. No download or data upload was performed during release preparation. No original images, derived image arrays or third-party dataset copies are included.

| Dataset | Official source | Acquisition and expected role |
|---|---|---|
| CIFAR-10 / CIFAR-100 | [Toronto CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html) | Obtain Python archives and extract into `<data-root>/CIFAR-10/cifar-10-batches-py` and `<data-root>/CIFAR-100/cifar-100-python`. torchvision supports `CIFAR10` / `CIFAR100(..., download=True)` for your own acquisition; study loaders use `download=False`. |
| Tiny ImageNet | [Stanford CS231n](http://cs231n.stanford.edu/tiny-imagenet-200.zip), [OpenOOD data preparation](https://github.com/Jingkang50/OpenOOD/blob/main/docs/data.md) | The original Stanford archive and the OpenOOD `images_classic` layout are not interchangeable. Obtain authorized files and use the exact relative paths and hashes in `provenance/confirmatory/manifests/EVALUATION_MANIFEST.json`. Pass its matching `images_classic` directory as `--tiny-root`. A generic Tiny download alone does not establish the frozen 7,793-row sample match. |
| SVHN | [Stanford SVHN](http://ufldl.stanford.edu/housenumbers/) | Obtain the cropped test `.mat` file under `<data-root>/SVHN`. Used only for historical diagnostics, not a new confirmatory endpoint. |
| CIFAR-10-C | [Official robustness repository](https://github.com/hendrycks/robustness), [dataset archive](https://zenodo.org/records/2535967) | Follow the author's download link. Historical corruption study only; no corruption arrays packaged. |
| CIFAR-10.1 v6 | [Author repository](https://github.com/modestyachts/CIFAR-10.1/tree/d9982abb0bfc4846b8d13a11e66b887d946205d0) | Obtain v6 data/labels from the pinned revision. The historical additional-sample manifest admits 1,994 of 2,000 v6 identities and 968 Tiny identities; do not replace it by the complete unfiltered dataset. |

Split archives in `splits/` contain only indices, labels and metadata, never images. Confirmatory training/validation/calibration roles are 44,000/3,000/3,000 and source-qualified identities matter. Historical roles and additional-sample filtering remain separate. Exact hashes and scientific eligibility rules are retained; path prefixes in public provenance are aliases, not active filesystem locations.
