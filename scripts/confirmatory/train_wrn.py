from __future__ import annotations
import json,os,time
from datetime import datetime,timezone
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
from torchvision import datasets
from models.wrn28_10 import WideResNet28x10EDL
from training.wrn_core import *
OUT=DATA=RUNTIME_LOG=None
MANIFESTS=Path(__file__).resolve().parents[1]/'manifests'
def split_paths():return {'train':MANIFESTS/'train_indices.npy'}


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()



def log(event: str, **fields: Any) -> None:
    RUNTIME_LOG.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"time": now(), "event": event, **fields}, ensure_ascii=False, allow_nan=False)
    with RUNTIME_LOG.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    print(line, flush=True)



def atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)



def save_npz(path: Path, **fields: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **fields)
    temporary.replace(path)



def load_checkpoint(seed: int, device: torch.device) -> WideResNet28x10EDL:
    checkpoint = torch.load(OUT / "checkpoints" / f"seed_{seed}_final.pt", map_location="cpu", weights_only=True)
    model = WideResNet28x10EDL()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model



def cmd_train(seed: int) -> int:
    # Public CLI verifies split hashes and inputs before training.
    checkpoint_path = OUT / "checkpoints" / f"seed_{seed}_final.pt"
    if checkpoint_path.exists():
        raise FileExistsError(f"refusing to overwrite {checkpoint_path}")
    config = configure(seed)
    device = torch.device("cuda")
    train_indices = np.load(split_paths()["train"], allow_pickle=False)
    train_aug = datasets.CIFAR10(str(DATA / "CIFAR-10"), train=True, transform=train_transform(), download=False)
    train_clean = datasets.CIFAR10(str(DATA / "CIFAR-10"), train=True, transform=eval_transform(), download=False)
    diagnostic_val = IndexedSubset(train_clean, np.load(split_paths()["val"], allow_pickle=False))
    train_loader = loader(IndexedSubset(train_aug, train_indices), 128, True, seed, workers=4)
    model = WideResNet28x10EDL().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=65536.0, growth_interval=1_000_000_000)
    history = []
    log("training_start", seed=seed, parameters=sum(value.numel() for value in model.parameters()), determinism=config)
    started = time.time()
    for epoch in range(1, 201):
        lr = learning_rate(epoch)
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        seen = correct = 0
        sums = {key: 0.0 for key in ("total", "error", "variance", "kl")}
        evidence_sum = uncertainty_sum = 0.0
        for batch_index, (images, labels, _) in enumerate(train_loader, 1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                output = model(images)
            if not all(torch.isfinite(output[key]).all() for key in ("raw_output", "evidence", "alpha")):
                raise FloatingPointError("non-finite training forward")
            loss = edl_loss(output["alpha"], labels, epoch)
            if not all(torch.isfinite(value) for value in loss.values()):
                raise FloatingPointError("non-finite training loss")
            scaler.scale(loss["total"]).backward()
            scaler.unscale_(optimizer)
            overflow = any(not torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)
            if overflow:
                offending = [name for name, parameter in model.named_parameters()
                             if parameter.grad is not None and not torch.isfinite(parameter.grad).all()]
                old_scale = float(scaler.get_scale())
                step_calls = []
                step_hook = optimizer.register_step_post_hook(lambda *args, **kwargs: step_calls.append(True))
            try:
                scaler.step(optimizer)
                scaler.update()
            finally:
                if overflow:
                    step_hook.remove()
            if overflow:
                new_scale = float(scaler.get_scale())
                log("grad_scaler_overflow", seed=seed, epoch=epoch, batch=batch_index,
                    old_scale=old_scale, new_scale=new_scale, optimizer_step_skipped=not step_calls,
                    nonfinite_parameter_count=len(offending), first_offending_parameter=offending[0])
                if step_calls or not new_scale < old_scale:
                    raise RuntimeError("GradScaler overflow recovery invariant failed")
            batch = labels.size(0)
            seen += batch
            correct += int((output["probability"].argmax(1) == labels).sum())
            evidence_sum += float(output["evidence"].sum())
            uncertainty_sum += float(output["uncertainty"].sum())
            for key in sums:
                sums[key] += float(loss[key]) * batch
        row = {"epoch": epoch, "learning_rate": lr, "train_online_accuracy": correct / seen,
               "train_mean_total_evidence": evidence_sum / seen, "train_mean_uncertainty": uncertainty_sum / seen,
               **{f"loss_{key}": value / seen for key, value in sums.items()}, "elapsed_seconds": time.time() - started}
        history.append(row)
        pd.DataFrame(history).to_csv(OUT / "training_logs" / f"seed_{seed}_history.csv", index=False, encoding="utf-8-sig")
        log("epoch", seed=seed, **row)
    atomic_torch_save({"model_state": model.state_dict(), "seed": seed, "epoch": 200,
                       "protocol_sha256": sha256_file(OUT / "run_config.json"),
                       "training_seconds": time.time() - started}, checkpoint_path)
    final_model = load_checkpoint(seed, device)
    clean_arrays = classification_arrays(final_model, loader(IndexedSubset(train_clean, train_indices), 256, False, seed, workers=4), device)
    val_arrays = classification_arrays(final_model, loader(diagnostic_val, 256, False, seed, workers=4), device)
    clean_metrics = classification_metrics(clean_arrays)
    val_metrics = classification_metrics(val_arrays)
    save_npz(OUT / "raw" / f"seed_{seed}_training_qa_arrays.npz", **{f"train_{k}": v for k, v in clean_arrays.items()}, **{f"val_{k}": v for k, v in val_arrays.items()})
    result = {"seed": seed, "status": "COMPLETED", "final_training": clean_metrics, "diagnostic_val": val_metrics,
              "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256_file(checkpoint_path),
              "training_seconds": time.time() - started, "ood_accessed": False}
    write_json(OUT / "training_logs" / f"seed_{seed}_complete.json", result)
    log("training_complete", **result)
    return 0

