from __future__ import annotations
import csv,hashlib,io,json,math,os,random,traceback
from datetime import datetime,timezone
from pathlib import Path
from typing import Any,Iterable
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets,transforms
from models.vgg16_bn import VGG16BNEDL
from .b0.loss import edl_loss
from .b0.schedule import learning_rate_for_epoch
ROOT=DATA_ROOT=INITIAL_CHECKPOINT=None
MANIFESTS=Path(__file__).resolve().parents[1]/'splits/historical'
TRAIN_INDICES=MANIFESTS/'train_indices.npy'
VALIDATION_INDICES=MANIFESTS/'validation_indices.npy'
CANDIDATES={}
EPOCHS=200
BATCH_SIZE=128
NUM_WORKERS=4
MOMENTUM=0.9
NESTEROV=True
WEIGHT_DECAY=5e-4
WARMUP_EPOCHS=5
MINIMUM_LR=1e-6
KL_ANNEALING_EPOCHS=10
ECE_BINS=15
MEAN=(0.4914,0.4822,0.4465)
STD=(0.2470,0.2435,0.2616)
MODULE_PREFIXES={k:k+'.' for k in ('block1','block2','block3','block4','block5','classifier')}
DYNAMICS_FIELDS=None


class IndexedSubset(Dataset):
    def __init__(self, dataset: Dataset, indices: np.ndarray):
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, position: int):
        source_index = int(self.indices[position])
        image, target = self.dataset[source_index]
        return image, int(target), source_index



def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()



def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()



def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()



def state_dict_hash(state: dict[str, torch.Tensor]) -> str:
    memory = io.BytesIO()
    torch.save(state, memory)
    return hashlib.sha256(memory.getvalue()).hexdigest()



def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    os.replace(temporary, path)



def write_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")



def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))



def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if not rows:
        raise ValueError("refusing to write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)



def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)



def set_determinism(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)



def worker_init_fn(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)



def make_loader(dataset: Dataset, seed: int, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=NUM_WORKERS > 0,
        worker_init_fn=worker_init_fn,
        generator=generator,
        drop_last=False,
    )



def id_train_validation_datasets() -> tuple[Dataset, Dataset]:
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    clean_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    dataset_root = DATA_ROOT / "CIFAR-10"
    augmented = datasets.CIFAR10(root=str(dataset_root), train=True, transform=train_transform, download=False)
    clean = datasets.CIFAR10(root=str(dataset_root), train=True, transform=clean_transform, download=False)
    return augmented, clean



def norm_of(values: Iterable[torch.Tensor]) -> float:
    tensors = [value.detach().float() for value in values]
    if not tensors:
        return 0.0
    total = sum((value.square().sum() for value in tensors), torch.zeros((), device=tensors[0].device))
    return float(torch.sqrt(total).item())



def parameter_norms(model: nn.Module) -> dict[str, float]:
    named = dict(model.named_parameters())
    result = {
        name: norm_of(value for key, value in named.items() if key.startswith(prefix))
        for name, prefix in MODULE_PREFIXES.items()
    }
    result["global"] = norm_of(named.values())
    return result



def required_gradient_norms(model: nn.Module) -> tuple[float, float]:
    named = dict(model.named_parameters())
    global_norm = norm_of(value.grad for value in named.values() if value.grad is not None)
    classifier_norm = norm_of(
        value.grad for key, value in named.items() if key.startswith("classifier.") and value.grad is not None
    )
    return global_norm, classifier_norm



def ece(probability: np.ndarray, labels: np.ndarray, bins: int = ECE_BINS) -> float:
    confidence = probability.max(axis=1)
    prediction = probability.argmax(axis=1)
    correct = prediction == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        upper = confidence <= edges[index + 1] if index == bins - 1 else confidence < edges[index + 1]
        mask = (confidence >= edges[index]) & upper
        if np.any(mask):
            result += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(result)



@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    evidences: list[np.ndarray] = []
    uncertainties: list[np.ndarray] = []
    nan_count = inf_count = 0
    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            output = model(images)
        monitored = [output[name] for name in ("raw_output", "evidence", "alpha", "probability", "uncertainty")]
        nan_count += sum(int(torch.isnan(value).sum().item()) for value in monitored)
        inf_count += sum(int(torch.isinf(value).sum().item()) for value in monitored)
        probabilities.append(output["probability"].cpu().numpy().astype(np.float64))
        labels.append(targets.numpy().astype(np.int64))
        evidences.append(output["evidence"].cpu().numpy().astype(np.float64))
        uncertainties.append(output["uncertainty"].cpu().numpy().astype(np.float64).reshape(-1))
    probability = np.concatenate(probabilities)
    target = np.concatenate(labels)
    evidence_matrix = np.concatenate(evidences)
    uncertainty = np.concatenate(uncertainties)
    prediction = probability.argmax(1)
    counts = np.bincount(prediction, minlength=10)
    true_probability = probability[np.arange(target.size), target]
    return {
        "validation_accuracy": float((prediction == target).mean()),
        "validation_NLL": float(-np.log(np.clip(true_probability, 1e-12, 1.0)).mean()),
        "validation_ECE": ece(probability, target),
        "mean_validation_evidence": float(evidence_matrix.mean()),
        "mean_validation_total_evidence": float(evidence_matrix.sum(axis=1).mean()),
        "mean_validation_uncertainty": float(uncertainty.mean()),
        "unique_predicted_classes": int(np.count_nonzero(counts)),
        "max_class_prediction_fraction": float(counts.max() / counts.sum()),
        "prediction_counts": counts.tolist(),
        "NaN_count": nan_count,
        "Inf_count": inf_count,
        "evidence_values": evidence_matrix.sum(axis=1),
        "uncertainty_values": uncertainty,
    }



def checkpoint_better(candidate: dict[str, Any], current: dict[str, Any] | None) -> bool:
    if current is None:
        return True
    candidate_key = (
        float(candidate["validation_accuracy"]),
        -float(candidate["validation_ECE"]),
        -float(candidate["validation_NLL"]),
        -int(candidate["epoch"]),
    )
    current_key = (
        float(current["validation_accuracy"]),
        -float(current["validation_ECE"]),
        -float(current["validation_NLL"]),
        -int(current["epoch"]),
    )
    return candidate_key > current_key



def collapse_reason(rows: list[dict[str, Any]]) -> str | None:
    if rows and (int(rows[-1]["NaN_count"]) > 0 or int(rows[-1]["Inf_count"]) > 0):
        return "A_NONFINITE"
    if len(rows) >= 10 and all(int(row["unique_predicted_classes"]) < 8 for row in rows[-10:]):
        return "D_PREDICTED_CLASSES_LT8_10_EPOCHS"
    return None



def run_paths(run_name: str) -> tuple[Path, Path, Path]:
    destination = ROOT / "runs" / run_name
    checkpoint = destination / "best.pt"
    dynamics = ROOT / ("P8_R1_01_LR005_DYNAMICS.csv" if run_name == "LR005" else "P8_R1_02_LR001_DYNAMICS.csv" if run_name == "LR001" else f"runs/{run_name}/dynamics.csv")
    return destination, checkpoint, dynamics



def training_initial_state(seed: int, protocol_sha: str) -> tuple[dict[str, torch.Tensor], str]:
    if seed == 0:
        payload = torch.load(INITIAL_CHECKPOINT, map_location="cpu", weights_only=True)
        # CLI verifies the original frozen initialization file SHA256.
        return payload["model_state"], payload["state_dict_sha256"]
    set_determinism(seed)
    model = VGG16BNEDL()
    return model.state_dict(), state_dict_hash(model.state_dict())



def reproduce_checkpoint(checkpoint: Path, seed: int, device: torch.device) -> dict[str, Any]:
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = VGG16BNEDL()
    model.load_state_dict(saved["model_state"], strict=True)
    model.to(device)
    _, clean = id_train_validation_datasets()
    validation_indices = np.load(VALIDATION_INDICES).astype(np.int64)
    loader = make_loader(IndexedSubset(clean, validation_indices), seed, False)
    actual = evaluate(model, loader, device)
    expected = saved["validation_metrics"]
    float_keys = ["validation_accuracy", "validation_NLL", "validation_ECE", "mean_validation_evidence", "mean_validation_total_evidence", "mean_validation_uncertainty", "max_class_prediction_fraction"]
    differences = {key: abs(float(actual[key]) - float(expected[key])) for key in float_keys}
    exact_keys = {
        "accuracy_exact": actual["validation_accuracy"] == expected["validation_accuracy"],
        "unique_classes_exact": actual["unique_predicted_classes"] == expected["unique_predicted_classes"],
        "prediction_histogram_exact": actual["prediction_counts"] == expected["prediction_counts"],
    }
    passed = exact_keys["accuracy_exact"] and exact_keys["unique_classes_exact"] and exact_keys["prediction_histogram_exact"] and all(value <= 1e-8 for value in differences.values()) and actual["NaN_count"] == 0 and actual["Inf_count"] == 0
    return {"status": "PASS" if passed else "FAIL", "differences": differences, "exact_checks": exact_keys, "actual": {key: value for key, value in actual.items() if key not in {"evidence_values", "uncertainty_values"}}, "expected": expected}



def assess_health(rows: list[dict[str, Any]], best: dict[str, Any] | None, reproduction: dict[str, Any], initial_norm: float, collapse: str | None, total_nan: int, total_inf: int) -> tuple[str, list[str], dict[str, Any]]:
    reasons: list[str] = []
    if len(rows) != EPOCHS:
        reasons.append("DID_NOT_COMPLETE_200_EPOCHS")
    if collapse:
        reasons.append(collapse)
    if total_nan or total_inf:
        reasons.append("NONFINITE")
    if best is None:
        reasons.append("NO_BEST_CHECKPOINT")
    else:
        if int(best["unique_predicted_classes"]) < 8:
            reasons.append("BEST_CHECKPOINT_CLASS_COVERAGE_LT8")
        if float(best["max_class_prediction_fraction"]) > 0.50:
            reasons.append("BEST_CHECKPOINT_MAX_CLASS_FRACTION_GT0.50")
        if not math.isfinite(float(best["mean_validation_total_evidence"])) or not math.isfinite(float(best["mean_validation_uncertainty"])):
            reasons.append("BEST_CHECKPOINT_EVIDENCE_OR_UNCERTAINTY_NONFINITE")
    if reproduction.get("status") != "PASS":
        reasons.append("BEST_CHECKPOINT_REPRODUCTION_FAIL")
    norms = np.asarray([float(row["global_parameter_norm"]) for row in rows], dtype=np.float64) if rows else np.asarray([])
    last10_slope = float(np.polyfit(np.arange(10), norms[-10:], 1)[0]) if norms.size >= 10 else float("nan")
    near_zero_shrinkage = bool(norms.size and norms[-1] <= 0.15 * initial_norm and last10_slope < 0)
    if near_zero_shrinkage:
        reasons.append("R0_LIKE_NEAR_ZERO_PARAMETER_SHRINKAGE")
    values = np.asarray([float(row["validation_accuracy"]) for row in rows], dtype=np.float64) if rows else np.asarray([])
    late_collapse = bool(values.size >= 10 and best is not None and values[-1] < float(best["validation_accuracy"]) - 0.20 and values[-10:].mean() < float(best["validation_accuracy"]) - 0.15)
    if late_collapse:
        reasons.append("LATE_CATASTROPHIC_VALIDATION_COLLAPSE")
    diagnostics = {
        "initial_global_parameter_norm": initial_norm,
        "final_global_parameter_norm": float(norms[-1]) if norms.size else None,
        "final_to_initial_norm_ratio": float(norms[-1] / initial_norm) if norms.size else None,
        "last10_parameter_norm_slope": last10_slope if math.isfinite(last10_slope) else None,
        "r0_like_near_zero_shrinkage": near_zero_shrinkage,
        "late_last10_validation_mean": float(values[-10:].mean()) if values.size >= 10 else None,
        "late_catastrophic_validation_collapse": late_collapse,
    }
    return ("HEALTHY" if not reasons else "UNHEALTHY"), reasons, diagnostics



def train_run(run_name: str, lr: float, seed: int) -> None:
    if run_name in CANDIDATES and CANDIDATES[run_name] != lr:
        raise RuntimeError("candidate LR mapping mismatch")
    protocol = {"protocol_sha256": sha256_file(ROOT / "run_config.json")}
    destination, checkpoint, dynamics_path = run_paths(run_name)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"refusing to overwrite or retry {run_name}")
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "run_state.json", {"status": "IN_PROGRESS", "run": run_name, "seed": seed, "initial_lr": lr, "official_test_used": False, "ood_used": False, "started_at_utc": utc_now()})
    # New run metadata is written above; no historical access log is replayed.
    set_determinism(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for formal P8-R1 training")
    initial_state, initial_state_sha = training_initial_state(seed, protocol["protocol_sha256"])
    model = VGG16BNEDL()
    model.load_state_dict(initial_state, strict=True)
    model.to(device)
    initial_norm = parameter_norms(model)["global"]
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=MOMENTUM, nesterov=NESTEROV, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=65536.0, growth_interval=1_000_000_000)
    train_indices = np.load(TRAIN_INDICES).astype(np.int64)
    validation_indices = np.load(VALIDATION_INDICES).astype(np.int64)
    augmented, clean = id_train_validation_datasets()
    train_loader = make_loader(IndexedSubset(augmented, train_indices), seed, True)
    validation_loader = make_loader(IndexedSubset(clean, validation_indices), seed, False)
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    total_nan = total_inf = overflow_count = 0
    collapse: str | None = None
    try:
        for epoch in range(1, EPOCHS + 1):
            current_lr = learning_rate_for_epoch(epoch, EPOCHS, lr, WARMUP_EPOCHS, MINIMUM_LR)
            for group in optimizer.param_groups:
                group["lr"] = current_lr
            model.train()
            sums = {"total": 0.0, "bayes_risk": 0.0, "error": 0.0, "variance": 0.0, "kl": 0.0}
            correct = seen = batches = epoch_nan = epoch_inf = 0
            global_grad_sum = classifier_grad_sum = 0.0
            for images, targets, _ in train_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                    output = model(images)
                losses = edl_loss(output["alpha"], targets, epoch, KL_ANNEALING_EPOCHS)
                monitored = [output[name] for name in ("raw_output", "evidence", "alpha", "probability", "uncertainty")] + list(losses.values())
                epoch_nan += sum(int(torch.isnan(value).sum().item()) for value in monitored)
                epoch_inf += sum(int(torch.isinf(value).sum().item()) for value in monitored)
                if epoch_nan or epoch_inf:
                    break
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimizer)
                gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
                epoch_nan += sum(int(torch.isnan(value).sum().item()) for value in gradients)
                epoch_inf += sum(int(torch.isinf(value).sum().item()) for value in gradients)
                if epoch_nan or epoch_inf:
                    break
                global_grad, classifier_grad = required_gradient_norms(model)
                global_grad_sum += global_grad
                classifier_grad_sum += classifier_grad
                old_scale = float(scaler.get_scale())
                scaler.step(optimizer)
                scaler.update()
                overflow_count += int(float(scaler.get_scale()) < old_scale)
                batch_n = int(targets.size(0))
                seen += batch_n
                batches += 1
                correct += int((output["probability"].argmax(1) == targets).sum().item())
                for key in sums:
                    sums[key] += float(losses[key].item()) * batch_n
            total_nan += epoch_nan
            total_inf += epoch_inf
            if seen == 0 or epoch_nan or epoch_inf:
                zero_counts = {f"pred_{index}": 0 for index in range(10)}
                norms = parameter_norms(model)
                row = {
                    "epoch": epoch, "actual_learning_rate": current_lr,
                    "train_total_loss": 0.0, "train_data_loss": 0.0, "train_error_loss": 0.0,
                    "train_variance_loss": 0.0, "train_kl_loss_raw": 0.0, "train_kl_loss_weighted": 0.0,
                    "kl_coefficient": min(1.0, epoch / KL_ANNEALING_EPOCHS), "train_accuracy": 0.0,
                    "validation_accuracy": 0.0, "validation_NLL": 0.0, "validation_ECE": 0.0,
                    "mean_validation_evidence": 0.0, "mean_validation_total_evidence": 0.0,
                    "mean_validation_uncertainty": 1.0, "unique_predicted_classes": 0,
                    "max_class_prediction_fraction": 1.0, **zero_counts,
                    "global_parameter_norm": norms["global"], "global_gradient_norm": 0.0,
                    "classifier_gradient_norm": 0.0,
                    **{f"{name}_parameter_norm": norms[name] for name in MODULE_PREFIXES},
                    "NaN_count": epoch_nan, "Inf_count": epoch_inf, "GradScaler_overflow_count": overflow_count,
                }
                rows.append(row)
                write_csv(dynamics_path, rows, DYNAMICS_FIELDS)
                collapse = "A_NONFINITE"
                break
            validation = evaluate(model, validation_loader, device)
            total_nan += int(validation["NaN_count"])
            total_inf += int(validation["Inf_count"])
            if validation["NaN_count"] or validation["Inf_count"]:
                collapse = "A_NONFINITE"
            counts = validation.pop("prediction_counts")
            validation.pop("evidence_values")
            validation.pop("uncertainty_values")
            norms = parameter_norms(model)
            coefficient = min(1.0, epoch / KL_ANNEALING_EPOCHS)
            row = {
                "epoch": epoch, "actual_learning_rate": current_lr,
                "train_total_loss": sums["total"] / seen,
                "train_data_loss": sums["bayes_risk"] / seen,
                "train_error_loss": sums["error"] / seen,
                "train_variance_loss": sums["variance"] / seen,
                "train_kl_loss_raw": sums["kl"] / seen,
                "train_kl_loss_weighted": coefficient * sums["kl"] / seen,
                "kl_coefficient": coefficient, "train_accuracy": correct / seen,
                **{key: validation[key] for key in ("validation_accuracy", "validation_NLL", "validation_ECE", "mean_validation_evidence", "mean_validation_total_evidence", "mean_validation_uncertainty", "unique_predicted_classes", "max_class_prediction_fraction")},
                **{f"pred_{index}": int(counts[index]) for index in range(10)},
                "global_parameter_norm": norms["global"], "global_gradient_norm": global_grad_sum / batches,
                "classifier_gradient_norm": classifier_grad_sum / batches,
                **{f"{name}_parameter_norm": norms[name] for name in MODULE_PREFIXES},
                "NaN_count": int(validation["NaN_count"]), "Inf_count": int(validation["Inf_count"]),
                "GradScaler_overflow_count": overflow_count,
            }
            rows.append(row)
            write_csv(dynamics_path, rows, DYNAMICS_FIELDS)
            candidate = {**row, "prediction_counts": counts}
            if checkpoint_better(candidate, best):
                best = candidate
                atomic_torch_save(checkpoint, {
                    "model_state": model.state_dict(), "epoch": epoch, "validation_metrics": best,
                    "initial_lr": lr, "seed": seed, "protocol_sha256": protocol["protocol_sha256"],
                    "initial_state_sha256": initial_state_sha,
                })
            collapse = collapse or collapse_reason(rows)
            print(json.dumps({"event": "epoch", "run": run_name, "seed": seed, "epoch": epoch, "lr": current_lr, "train_accuracy": row["train_accuracy"], "validation_accuracy": row["validation_accuracy"], "global_parameter_norm": row["global_parameter_norm"], "unique_classes": row["unique_predicted_classes"], "best_epoch": best["epoch"] if best else None, "collapse": collapse}), flush=True)
            if collapse:
                break
        reproduction = reproduce_checkpoint(checkpoint, seed, device) if checkpoint.exists() else {"status": "FAIL", "reason": "NO_CHECKPOINT"}
        write_json(destination / "checkpoint_reproduction.json", reproduction)
        health, reasons, diagnostics = assess_health(rows, best, reproduction, initial_norm, collapse, total_nan, total_inf)
        summary = {
            "status": health, "run": run_name, "seed": seed, "initial_lr": lr,
            "epochs_completed": len(rows), "expected_epochs": EPOCHS,
            "collapse_reason": collapse, "health_reasons": reasons,
            "best_epoch": int(best["epoch"]) if best else None,
            "best_validation_accuracy": float(best["validation_accuracy"]) if best else None,
            "best_validation_metrics": best,
            "final_epoch": rows[-1] if rows else None,
            "total_nan_count": total_nan, "total_inf_count": total_inf,
            "grad_scaler_overflow_count": overflow_count,
            "checkpoint_reproduction": reproduction["status"],
            "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint) if checkpoint.exists() else None,
            "initial_state_sha256": initial_state_sha,
            "health_diagnostics": diagnostics,
            "official_test_used": False, "ood_used": False,
            "completed_at_utc": utc_now(),
        }
        write_json(destination / "run_state.json", summary)
        print(json.dumps({"event": "training_complete", **summary}, ensure_ascii=False), flush=True)
    except Exception as exc:
        write_json(destination / "run_state.json", {"status": "UNHEALTHY", "run": run_name, "seed": seed, "initial_lr": lr, "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(), "official_test_used": False, "ood_used": False, "failed_at_utc": utc_now()})
        raise

