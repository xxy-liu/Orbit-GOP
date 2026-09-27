from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np


CODE_VERSION = "P5_B0_CEDL_PARAMETRIC_OFFLINE_V1"
K = 10
T = 5
EPSILON = 1e-8
DEFAULT_CONFLICT_BETA = 1.5
DEFAULT_CEDL_LAMBDA = 0.5
DEFAULT_DELTA = 1.0

TUNING_ALLOWED_ROLES = frozenset({"cifar10_id_safety", "cifar100_development"})
TUNING_FORBIDDEN_ROLES = frozenset(
    {
        "cifar10_official_test",
        "cifar100_official_test",
        "tiny_imagenet",
        "cifar100_holdout",
    }
)


class DatasetRoleError(ValueError):
    """Raised before any scoring when a tuning request violates the frozen role contract."""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def assert_dataset_role_allowed(dataset_role: str, mode: str) -> None:
    role = str(dataset_role).strip().lower().replace("-", "_").replace(" ", "_")
    mode_normalized = str(mode).strip().lower()
    aliases = {
        "cifar_10_id_safety": "cifar10_id_safety",
        "cifar_100_development": "cifar100_development",
        "cifar_10_official_test": "cifar10_official_test",
        "cifar_100_official_test": "cifar100_official_test",
        "tiny_imagenet_formal_test_endpoint": "tiny_imagenet",
        "cifar_100_holdout": "cifar100_holdout",
    }
    role = aliases.get(role, role)
    if mode_normalized == "tuning":
        if role in TUNING_FORBIDDEN_ROLES:
            raise DatasetRoleError(f"dataset role is sealed and forbidden for tuning: {role}")
        if role not in TUNING_ALLOWED_ROLES:
            raise DatasetRoleError(
                f"dataset role is not in the frozen tuning allow-list: {role}; "
                f"allowed={sorted(TUNING_ALLOWED_ROLES)}"
            )


def _positive_finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and > 0")
    return result


def _validate_view_evidence(view_evidence: np.ndarray) -> np.ndarray:
    evidence = np.asarray(view_evidence, dtype=np.float64)
    if evidence.ndim != 3 or evidence.shape[1:] != (T, K):
        raise ValueError(f"view_evidence must have shape [N,{T},{K}], got {evidence.shape}")
    if not np.isfinite(evidence).all():
        raise ValueError("view_evidence contains non-finite values")
    if np.any(evidence < 0):
        raise ValueError("view_evidence must be non-negative")
    return evidence


def _validate_aggregated_evidence(aggregated_evidence: np.ndarray) -> np.ndarray:
    evidence = np.asarray(aggregated_evidence, dtype=np.float64)
    if evidence.ndim != 2 or evidence.shape[1] != K:
        raise ValueError(f"aggregated_evidence must have shape [N,{K}], got {evidence.shape}")
    if not np.isfinite(evidence).all() or np.any(evidence < 0):
        raise ValueError("aggregated_evidence must be finite and non-negative")
    return evidence


def _inter_root_per_view(evidence_nkt: np.ndarray) -> np.ndarray:
    i_idx, j_idx = np.triu_indices(K, k=1)
    ei = evidence_nkt[:, i_idx, :]
    ej = evidence_nkt[:, j_idx, :]
    minimum = np.minimum(ei, ej)
    maximum = np.maximum(ei, ej)
    ratio = np.divide(minimum, maximum, out=np.zeros_like(minimum), where=maximum > 0)
    strength = evidence_nkt.sum(axis=1)[:, None, :]
    normalized = np.divide(minimum, strength, out=np.zeros_like(minimum), where=strength != 0)
    pair_conflict = ratio * normalized * 2.0
    return np.sqrt(np.square(pair_conflict).sum(axis=1))


def sufficient_primitives_from_view_evidence(view_evidence: np.ndarray) -> dict[str, np.ndarray]:
    """Derive the beta/lambda/delta-independent sufficient statistics.

    Input is [sample, view, class], with the original view at view index 0.
    """
    evidence_ntk = _validate_view_evidence(view_evidence)
    evidence_nkt = np.transpose(evidence_ntk, (0, 2, 1))
    mean = evidence_nkt.mean(axis=2)
    intra = (evidence_nkt.std(axis=2) / (mean + EPSILON)).mean(axis=1)
    return {
        "aggregated_evidence": mean,
        "c_intra": intra,
        "inter_root_per_view": _inter_root_per_view(evidence_nkt),
        "original_prediction": evidence_ntk[:, 0, :].argmax(axis=1).astype(np.int64),
    }


def _finish_score(
    aggregated_evidence: np.ndarray,
    c_intra: np.ndarray,
    c_inter: np.ndarray,
    cedl_lambda: float,
    delta: float,
) -> dict[str, np.ndarray]:
    aggregated = _validate_aggregated_evidence(aggregated_evidence)
    intra = np.asarray(c_intra, dtype=np.float64)
    inter = np.asarray(c_inter, dtype=np.float64)
    if intra.shape != (len(aggregated),) or inter.shape != (len(aggregated),):
        raise ValueError("c_intra and c_inter must each have shape [N]")
    if not np.isfinite(intra).all() or not np.isfinite(inter).all():
        raise ValueError("conflict primitives contain non-finite values")
    penalty = _positive_finite("cedl_lambda", cedl_lambda)
    decay = _positive_finite("delta", delta)
    total = np.clip(inter + intra - inter * intra - penalty * np.square(inter - intra), 0.0, 1.0)
    adjusted_evidence = aggregated * np.exp(-decay * total)[:, None]
    adjusted_alpha = adjusted_evidence + 1.0
    strength = adjusted_alpha.sum(axis=1)
    probability = adjusted_alpha / strength[:, None]
    return {
        "c_intra": intra,
        "c_inter": inter,
        "c_total": total,
        "aggregated_evidence": aggregated,
        "adjusted_alpha": adjusted_alpha,
        "uncertainty": K / strength,
        "prediction": probability.argmax(axis=1).astype(np.int64),
    }


def score_from_sufficient_primitives(
    aggregated_evidence: np.ndarray,
    c_intra: np.ndarray,
    inter_root_per_view: np.ndarray,
    *,
    conflict_beta: float,
    cedl_lambda: float,
    delta: float,
) -> dict[str, np.ndarray]:
    beta = _positive_finite("conflict_beta", conflict_beta)
    roots = np.asarray(inter_root_per_view, dtype=np.float64)
    if roots.ndim != 2 or roots.shape[1] != T:
        raise ValueError(f"inter_root_per_view must have shape [N,{T}], got {roots.shape}")
    if not np.isfinite(roots).all() or np.any(roots < 0):
        raise ValueError("inter_root_per_view must be finite and non-negative")
    c_inter = (1.0 - np.exp(-beta * roots)).mean(axis=1)
    return _finish_score(aggregated_evidence, c_intra, c_inter, cedl_lambda, delta)


def score_tuning_candidate_from_sufficient_primitives(
    dataset_role: str,
    aggregated_evidence: np.ndarray,
    c_intra: np.ndarray,
    inter_root_per_view: np.ndarray,
    *,
    conflict_beta: float,
    cedl_lambda: float,
    delta: float,
) -> dict[str, np.ndarray]:
    """Guarded public entry point for development-only candidate scoring."""
    assert_dataset_role_allowed(dataset_role, "tuning")
    return score_from_sufficient_primitives(
        aggregated_evidence,
        c_intra,
        inter_root_per_view,
        conflict_beta=conflict_beta,
        cedl_lambda=cedl_lambda,
        delta=delta,
    )


def score_from_view_evidence(
    view_evidence: np.ndarray,
    *,
    conflict_beta: float,
    cedl_lambda: float,
    delta: float,
) -> dict[str, np.ndarray]:
    primitives = sufficient_primitives_from_view_evidence(view_evidence)
    return score_from_sufficient_primitives(
        primitives["aggregated_evidence"],
        primitives["c_intra"],
        primitives["inter_root_per_view"],
        conflict_beta=conflict_beta,
        cedl_lambda=cedl_lambda,
        delta=delta,
    )


def score_from_view_alpha(
    view_alpha: np.ndarray,
    *,
    conflict_beta: float,
    cedl_lambda: float,
    delta: float,
) -> dict[str, np.ndarray]:
    alpha = np.asarray(view_alpha, dtype=np.float64)
    if alpha.ndim != 3 or alpha.shape[1:] != (T, K):
        raise ValueError(f"view_alpha must have shape [N,{T},{K}], got {alpha.shape}")
    return score_from_view_evidence(
        alpha - 1.0,
        conflict_beta=conflict_beta,
        cedl_lambda=cedl_lambda,
        delta=delta,
    )


def score_from_frozen_default_primitives(
    aggregated_alpha: np.ndarray,
    c_intra: np.ndarray,
    c_inter_at_default_beta: np.ndarray,
    *,
    conflict_beta: float,
    cedl_lambda: float,
    delta: float,
) -> dict[str, np.ndarray]:
    """Reproduce the frozen default using its saved high-precision conflicts.

    A scalar c_inter computed at beta=1.5 is not sufficient to change beta.  The
    guard prevents accidental use of this path for another beta. Lambda and
    delta remain explicit and can be changed from these primitives.
    """
    beta = _positive_finite("conflict_beta", conflict_beta)
    if beta != DEFAULT_CONFLICT_BETA:
        raise ValueError(
            "saved c_inter_at_default_beta is valid only for conflict_beta=1.5; "
            "use view evidence or inter_root_per_view to vary conflict_beta"
        )
    alpha = np.asarray(aggregated_alpha, dtype=np.float64)
    if alpha.ndim != 2 or alpha.shape[1] != K:
        raise ValueError(f"aggregated_alpha must have shape [N,{K}], got {alpha.shape}")
    return _finish_score(alpha - 1.0, c_intra, c_inter_at_default_beta, cedl_lambda, delta)


def _status(max_abs_score_diff: float) -> str:
    if max_abs_score_diff <= 1e-10:
        return "EXACT_NUMERICAL_REPRODUCTION"
    if max_abs_score_diff <= 1e-7:
        return "FLOAT_PRECISION_EQUIVALENT"
    return "FAIL"


def _write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)


def reproduce_resnet_default(input_path: Path, output_csv: Path, provenance_path: Path) -> int:
    import polars as pl

    frame = pl.read_parquet(input_path)
    required = {
        "dataset",
        "model_seed",
        "cedl_uncertainty",
        "cedl_prediction",
        "cedl_c_intra",
        "cedl_c_inter",
        "cedl_aggregated_alpha",
        "cedl_view_alpha",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"input is missing required columns: {missing}")

    aggregated_alpha = np.asarray(frame["cedl_aggregated_alpha"].to_list(), dtype=np.float64)
    frozen_score = frame["cedl_uncertainty"].to_numpy().astype(np.float64)
    frozen_prediction = frame["cedl_prediction"].to_numpy().astype(np.int64)
    result = score_from_frozen_default_primitives(
        aggregated_alpha,
        frame["cedl_c_intra"].to_numpy(),
        frame["cedl_c_inter"].to_numpy(),
        conflict_beta=DEFAULT_CONFLICT_BETA,
        cedl_lambda=DEFAULT_CEDL_LAMBDA,
        delta=DEFAULT_DELTA,
    )
    score = result["uncertainty"]
    prediction = result["prediction"]
    abs_diff = np.abs(score - frozen_score)
    squared_diff = np.square(score - frozen_score)
    mismatch = prediction != frozen_prediction
    nonfinite = ~(np.isfinite(score) & np.isfinite(frozen_score))

    diagnostic_alpha = np.asarray(frame["cedl_view_alpha"].to_list(), dtype=np.float64)
    diagnostic = score_from_view_alpha(
        diagnostic_alpha,
        conflict_beta=DEFAULT_CONFLICT_BETA,
        cedl_lambda=DEFAULT_CEDL_LAMBDA,
        delta=DEFAULT_DELTA,
    )
    diagnostic_abs_diff = np.abs(diagnostic["uncertainty"] - frozen_score)

    work = frame.select("model_seed", "dataset").with_columns(
        pl.Series("abs_diff", abs_diff),
        pl.Series("squared_diff", squared_diff),
        pl.Series("prediction_mismatch", mismatch),
        pl.Series("nonfinite", nonfinite),
    )
    grouped = work.group_by(["model_seed", "dataset"]).agg(
        pl.len().alias("n"),
        pl.col("abs_diff").max().alias("max_abs_score_diff"),
        pl.col("abs_diff").mean().alias("mean_abs_score_diff"),
        pl.col("squared_diff").mean().sqrt().alias("rmse"),
        pl.col("prediction_mismatch").sum().alias("prediction_mismatch_count"),
        pl.col("nonfinite").sum().alias("nonfinite_count"),
    ).sort(["model_seed", "dataset"])

    rows: list[dict[str, object]] = []
    for record in grouped.iter_rows(named=True):
        status = _status(float(record["max_abs_score_diff"]))
        passed = (
            status != "FAIL"
            and int(record["prediction_mismatch_count"]) == 0
            and int(record["nonfinite_count"]) == 0
        )
        rows.append(
            {
                "seed": f"S{int(record['model_seed'])}",
                "dataset_role": record["dataset"],
                "n": int(record["n"]),
                "max_abs_score_diff": f"{float(record['max_abs_score_diff']):.17g}",
                "mean_abs_score_diff": f"{float(record['mean_abs_score_diff']):.17g}",
                "rmse": f"{float(record['rmse']):.17g}",
                "prediction_mismatch_count": int(record["prediction_mismatch_count"]),
                "nonfinite_count": int(record["nonfinite_count"]),
                "reproduction_status": status,
                "pass": "TRUE" if passed else "FALSE",
            }
        )
    _write_csv(output_csv, rows)

    overall_max = float(abs_diff.max())
    overall_status = _status(overall_max)
    overall_mismatch = int(mismatch.sum())
    overall_nonfinite = int(nonfinite.sum())
    overall_pass = overall_status != "FAIL" and overall_mismatch == 0 and overall_nonfinite == 0
    provenance = {
        "protocol_id": "P5_B0_C_EDL_FAIR_TUNING_V1_20260905",
        "code_version": CODE_VERSION,
        "code_path": str(Path(__file__).resolve()),
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "input_path": str(input_path.resolve()),
        "input_sha256": sha256_file(input_path.resolve()),
        "output_csv": str(output_csv.resolve()),
        "parameters": {
            "T": T,
            "conflict_beta": DEFAULT_CONFLICT_BETA,
            "cedl_lambda": DEFAULT_CEDL_LAMBDA,
            "delta": DEFAULT_DELTA,
        },
        "execution": {
            "model_forward": False,
            "training": False,
            "candidate_grid_executed": False,
            "primary_reproduction_path": "saved high-precision c_intra/c_inter plus aggregated_alpha",
            "reason": "saved view_alpha is float32 and alpha-to-evidence inversion is lossy near alpha=1",
        },
        "rows": int(frame.height),
        "units": len(rows),
        "overall": {
            "max_abs_score_diff": overall_max,
            "mean_abs_score_diff": float(abs_diff.mean()),
            "rmse": float(np.sqrt(squared_diff.mean())),
            "prediction_mismatch_count": overall_mismatch,
            "nonfinite_count": overall_nonfinite,
            "status": overall_status,
            "pass": overall_pass,
        },
        "lossy_view_alpha_diagnostic_not_used_for_gate": {
            "max_abs_score_diff": float(diagnostic_abs_diff.max()),
            "mean_abs_score_diff": float(diagnostic_abs_diff.mean()),
            "prediction_mismatch_count": int(
                np.count_nonzero(diagnostic["prediction"] != frozen_prediction)
            ),
        },
    }
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps(provenance["overall"], indent=2))
    return 0 if overall_pass else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parameterised, model-forward-free C-EDL scorer with frozen dataset-role guards."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    reproduce = subparsers.add_parser("reproduce-default")
    reproduce.add_argument("--input", type=Path, required=True)
    reproduce.add_argument("--output-csv", type=Path, required=True)
    reproduce.add_argument("--provenance", type=Path, required=True)
    guard = subparsers.add_parser("guard-check")
    guard.add_argument("--dataset-role", required=True)
    guard.add_argument("--mode", choices=("tuning", "reproduction"), required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "guard-check":
        assert_dataset_role_allowed(args.dataset_role, args.mode)
        print(f"ALLOW mode={args.mode} dataset_role={args.dataset_role}")
        return 0
    if args.command == "reproduce-default":
        return reproduce_resnet_default(args.input, args.output_csv, args.provenance)
    raise AssertionError(args.command)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DatasetRoleError as exc:
        print(f"DATASET_ROLE_GUARD_ERROR: {exc}", file=sys.stderr)
        raise SystemExit(3)
