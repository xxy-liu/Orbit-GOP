import numpy as np

def stratified_indices(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return np.concatenate([rng.choice(pos, size=len(pos), replace=True) for class_id in np.unique(labels) for pos in [np.flatnonzero(labels == class_id)]])

def bootstrap_weights(labels: np.ndarray, rng: np.random.Generator, batch: int, stratified: bool) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    weights = np.zeros((batch, labels.size), dtype=np.int16)
    groups = [np.flatnonzero(labels == value) for value in np.unique(labels)] if stratified else [np.arange(labels.size)]
    for positions in groups:
        weights[:, positions] = rng.multinomial(
            positions.size, np.full(positions.size, 1.0 / positions.size), size=batch
        ).astype(np.int16, copy=False)
    return weights

class WeightedLayout:
    """Tie-aware vectorized weighted AUROC/FPR95 for fixed sample scores."""
    def __init__(self, id_score: np.ndarray, ood_score: np.ndarray):
        self.n_id = len(id_score)
        self.n_ood = len(ood_score)
        self.score = np.r_[id_score, ood_score].astype(np.float64)
        self.is_ood = np.r_[np.zeros(self.n_id, dtype=np.int8), np.ones(self.n_ood, dtype=np.int8)]
        self.asc = np.argsort(self.score, kind="mergesort")
        ordered = self.score[self.asc]
        self.starts_asc = np.r_[0, 1 + np.flatnonzero(ordered[1:] != ordered[:-1])]
        self.desc = self.asc[::-1]
        ordered = self.score[self.desc]
        self.starts_desc = np.r_[0, 1 + np.flatnonzero(ordered[1:] != ordered[:-1])]

    def _groups(self, weight: np.ndarray, order: np.ndarray, starts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ordered = weight[:, order]
        positive = np.add.reduceat(ordered * self.is_ood[order][None, :], starts, axis=1, dtype=np.int32)
        total = np.add.reduceat(ordered, starts, axis=1, dtype=np.int32)
        return positive, total - positive

    def evaluate(self, id_weight: np.ndarray, ood_weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        weight = np.concatenate([id_weight, ood_weight], axis=1)
        ood_asc, id_asc = self._groups(weight, self.asc, self.starts_asc)
        id_before = np.cumsum(id_asc, axis=1, dtype=np.int64) - id_asc
        auroc = (ood_asc * (id_before + 0.5 * id_asc)).sum(1) / float(self.n_id * self.n_ood)
        ood_desc, id_desc = self._groups(weight, self.desc, self.starts_desc)
        cum_ood = np.cumsum(ood_desc, axis=1, dtype=np.int64)
        cum_id = np.cumsum(id_desc, axis=1, dtype=np.int64)
        point = np.argmax(cum_ood >= 0.95 * self.n_ood, axis=1)
        fpr95 = cum_id[np.arange(weight.shape[0]), point] / float(self.n_id)
        return auroc, fpr95
