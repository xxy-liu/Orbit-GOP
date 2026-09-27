from __future__ import annotations
import csv,json,logging,os,sys,traceback
from pathlib import Path
import numpy as np
import torch
from .artifacts import canonical_hash,set_determinism,sha256_file,utc_now,write_json
from .b0.data import IndexedSubset,load_cifar10_train,make_loader
from .b0.engine import infer
from .b0.loss import edl_loss
from .b0.schedule import learning_rate_for_epoch
from models.resnet18 import ResNet18EDLSoftplus
ROOT=None
CONFIG=None
ARGS=None
MANIFESTS=Path(__file__).resolve().parents[1]/'splits/historical'
def parse_args():return ARGS
def build_manifest(root):
    public_root=Path(__file__).resolve().parents[1]
    sources={p.relative_to(public_root).as_posix():sha256_file(p)
             for folder in ('models','training','scripts')
             for p in sorted((public_root/folder).rglob('*.py'))}
    write_json(root/'source_manifest.json',{'files':sources,'manifest_sha256':canonical_hash(sources)})
    return {'manifest_sha256':canonical_hash(sources)}


def atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)



def logger_for(path: Path) -> logging.Logger:
    logger = logging.getLogger(f"b0.{path.parent.name}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger



def validate_config(config: dict) -> None:
    expected = {
        "experiment_name": "ResNet18-EDL-Softplus",
        "epochs": 200,
        "batch_size": 128,
        "optimizer": "SGD",
        "learning_rate": 0.1,
        "momentum": 0.9,
        "nesterov": True,
        "weight_decay": 0.0005,
        "warmup_epochs": 5,
        "kl_annealing_epochs": 10,
        "pretrained": False,
        "evidence_activation": "Softplus",
    }
    mismatches = {key: (config.get(key), value) for key, value in expected.items() if config.get(key) != value}
    if mismatches:
        raise ValueError(f"Frozen Stage B0 configuration mismatch: {mismatches}")
    if config.get("training_seeds") != [0, 1, 2]:
        raise ValueError("Exactly three registered training seeds 0,1,2 are required")



def better(candidate: dict, best: dict | None) -> bool:
    if best is None:
        return True
    keys = (("accuracy", 1), ("ece", -1), ("categorical_nll", -1), ("epoch", -1))
    for key, direction in keys:
        if candidate[key] != best[key]:
            return direction * candidate[key] > direction * best[key]
    return False



def gradient_check(model: torch.nn.Module) -> tuple[bool, int, int]:
    total = 0
    nonfinite = 0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        total += parameter.grad.numel()
        nonfinite += int((~torch.isfinite(parameter.grad)).sum().item())
    return nonfinite == 0, total, nonfinite



def main() -> None:
    args = parse_args()
    config = CONFIG
    validate_config(config)
    # Historical approval artifacts are not fabricated for new public runs.
    # The CLI verifies packaged split identities before calling this loop.
    seed_dir = ROOT / "runs" / f"seed_{args.seed}"
    if seed_dir.exists() and any(seed_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite or resume non-empty formal seed directory: {seed_dir}")
    seed_dir.mkdir(parents=True, exist_ok=True)
    (seed_dir / "failures").mkdir(exist_ok=True)
    log = logger_for(seed_dir / "train.log")
    determinism = set_determinism(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(config["mixed_precision_backbone"] and device.type == "cuda")
    train_indices = np.load(MANIFESTS / "train_indices.npy")
    val_indices = np.load(MANIFESTS / "validation_indices.npy")
    augmented, clean = load_cifar10_train(Path(config["data_root"]))
    train_loader = make_loader(
        IndexedSubset(augmented, train_indices), config["batch_size"], config["num_workers"], True, args.seed
    )
    val_loader = make_loader(
        IndexedSubset(clean, val_indices), config["batch_size"], config["num_workers"], False, args.seed
    )
    set_determinism(args.seed)
    model = ResNet18EDLSoftplus(
        config["num_classes"], config["softplus_beta"], config["softplus_threshold"]
    ).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["learning_rate"],
        momentum=config["momentum"],
        nesterov=config["nesterov"],
        weight_decay=config["weight_decay"],
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
        init_scale=float(config["grad_scaler_init_scale"]),
        growth_interval=int(config["grad_scaler_growth_interval"]),
    )
    config_hash = canonical_hash(config)
    split_hash = canonical_hash([sha256_file(MANIFESTS / "train_indices.npy"),sha256_file(MANIFESTS / "validation_indices.npy")])
    source_manifest_hash = build_manifest(ROOT)["manifest_sha256"]
    write_json(seed_dir / "run_manifest.json", {
        "status": "IN_PROGRESS",
        "experiment_name": config["experiment_name"],
        "training_seed": args.seed,
        "config_sha256": config_hash,
        "split_sha256": split_hash,
        "source_manifest_sha256_at_start": source_manifest_hash,
        "test_set_accessed": False,
        "ood_accessed": False,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "determinism": determinism,
        "started_at_utc": utc_now(),
    })
    history: list[dict] = []
    best_metrics: dict | None = None
    overflow_count = 0
    log.info("START experiment=%s seed=%d device=%s", config["experiment_name"], args.seed, device)
    try:
        for epoch in range(1, config["epochs"] + 1):
            lr = learning_rate_for_epoch(
                epoch, config["epochs"], config["learning_rate"], config["warmup_epochs"], config["minimum_learning_rate"]
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            model.train()
            sums = {name: 0.0 for name in ("total", "bayes_risk", "error", "variance", "kl")}
            correct = 0
            seen = 0
            uncertainty_sum = 0.0
            evidence_sum = 0.0
            for images, targets, _ in train_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                    output = model(images)
                losses = edl_loss(output["alpha"], targets, epoch, config["kl_annealing_epochs"])
                for name in ("raw_output", "evidence", "alpha", "probability", "uncertainty"):
                    if not torch.isfinite(output[name]).all():
                        raise FloatingPointError(f"non-finite forward output: {name}")
                if not all(torch.isfinite(losses[name]) for name in ("total", "bayes_risk", "error", "variance", "kl")):
                    raise FloatingPointError("non-finite EDL loss")
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimizer)
                gradients_ok, gradient_count, nonfinite_gradients = gradient_check(model)
                if not gradients_ok:
                    raise FloatingPointError(
                        f"non-finite gradients {nonfinite_gradients}/{gradient_count}"
                    )
                old_scale = float(scaler.get_scale())
                scaler.step(optimizer)
                scaler.update()
                if float(scaler.get_scale()) < old_scale:
                    overflow_count += 1
                batch = int(targets.size(0))
                seen += batch
                prediction = output["probability"].argmax(dim=1)
                correct += int((prediction == targets).sum().item())
                uncertainty_sum += float(output["uncertainty"].sum().item())
                evidence_sum += float(output["evidence"].sum().item())
                for name in sums:
                    sums[name] += float(losses[name].item()) * batch
            val_metrics, _ = infer(
                model, val_loader, device, epoch, config["kl_annealing_epochs"], config["ece_bins"]
            )
            candidate = {
                "epoch": epoch,
                "accuracy": float(val_metrics["accuracy"]),
                "ece": float(val_metrics["ece"]),
                "categorical_nll": float(val_metrics["categorical_nll"]),
            }
            row = {
                "epoch": epoch,
                "learning_rate": lr,
                "annealing_coefficient": min(1.0, epoch / config["kl_annealing_epochs"]),
                "train_online_accuracy": correct / seen,
                "train_mean_uncertainty": uncertainty_sum / seen,
                "train_mean_total_evidence": evidence_sum / seen,
                "grad_scaler_scale": float(scaler.get_scale()),
                "grad_scaler_overflow_count": overflow_count,
                **{f"train_loss_{name}": value / seen for name, value in sums.items()},
                **{f"val_{name}": value for name, value in val_metrics.items()},
            }
            history.append(row)
            with (seed_dir / "train_history.csv").open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(history[0].keys()))
                writer.writeheader()
                writer.writerows(history)
            if better(candidate, best_metrics):
                best_metrics = candidate
                atomic_torch_save({
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "best_validation_metrics": best_metrics,
                    "training_seed": args.seed,
                    "config_sha256": config_hash,
                    "split_sha256": split_hash,
                    "source_manifest_sha256": source_manifest_hash,
                }, seed_dir / "best.pt")
            log.info(
                "epoch=%03d lr=%.8f train_acc=%.4f val_acc=%.4f val_ece=%.4f val_nll=%.4f best_epoch=%d",
                epoch, lr, correct / seen, val_metrics["accuracy"], val_metrics["ece"],
                val_metrics["categorical_nll"], best_metrics["epoch"],
            )
        atomic_torch_save({
            "model_state": model.state_dict(),
            "epoch": config["epochs"],
            "best_validation_metrics": best_metrics,
            "training_seed": args.seed,
            "config_sha256": config_hash,
            "split_sha256": split_hash,
            "source_manifest_sha256": source_manifest_hash,
        }, seed_dir / "last.pt")
        completion = {
            "status": "COMPLETED",
            "experiment_name": config["experiment_name"],
            "training_seed": args.seed,
            "epochs": config["epochs"],
            "best_validation_metrics": best_metrics,
            "grad_scaler_overflow_count": overflow_count,
            "best_checkpoint_sha256": sha256_file(seed_dir / "best.pt"),
            "last_checkpoint_sha256": sha256_file(seed_dir / "last.pt"),
            "config_sha256": config_hash,
            "split_sha256": split_hash,
            "test_set_accessed": False,
            "ood_accessed": False,
            "completed_at_utc": utc_now(),
        }
        write_json(seed_dir / "training_complete.json", completion)
        write_json(seed_dir / "run_manifest.json", completion)
        log.info("COMPLETED seed=%d best=%s", args.seed, best_metrics)
    except Exception as exc:
        failure = {
            "status": "FAILED",
            "seed": args.seed,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "test_set_accessed": False,
            "ood_accessed": False,
            "failed_at_utc": utc_now(),
        }
        write_json(seed_dir / "failures" / "failure.json", failure)
        write_json(seed_dir / "run_manifest.json", failure)
        log.exception("FAILED seed=%d", args.seed)
        raise

