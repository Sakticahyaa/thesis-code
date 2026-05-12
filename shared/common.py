"""Shared constants and utilities for all experiments."""
import random, sys
import numpy as np, torch
from sklearn.metrics import cohen_kappa_score

TRAITS = [
    'argument_clarity', 'justifying_persuasiveness', 'organizational_structure',
    'coherence', 'essay_length', 'grammatical_accuracy', 'grammatical_diversity',
    'lexical_accuracy', 'lexical_diversity', 'punctuation_accuracy',
]
SCORE_COLUMNS = [f'{t}(ground_truth)' for t in TRAITS]
NUM_TRAITS    = 10
NUM_CLASSES   = 11  # score 0.0–5.0 in 0.5 steps → class index 0..10


class Tee:
    """Mirror stdout to terminal and a log file simultaneously."""
    def __init__(self, path):
        self._file = open(path, 'w')
        self._orig = sys.__stdout__
        sys.stdout = self
    def write(self, text):
        self._orig.write(text); self._orig.flush()
        self._file.write(text); self._file.flush()
    def flush(self):
        self._orig.flush(); self._file.flush()


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def compute_class_weights(train_df):
    """Per-trait inverse-frequency class weights, mean-normalized. Returns [10, 11]."""
    weights = []
    for col in SCORE_COLUMNS:
        idx    = np.round(train_df[col].values * 2).astype(int)
        counts = np.bincount(idx, minlength=NUM_CLASSES).astype(float)
        counts[counts == 0] = 1.0
        w = 1.0 / counts
        weights.append(w / w.mean())
    return np.array(weights)


def compute_qwk_cls(preds_idx, labels_idx):
    """QWK from integer class indices [N,10]. Returns (avg_qwk, per_trait_list)."""
    qwks = []
    for t in range(preds_idx.shape[1]):
        try:
            q = cohen_kappa_score(labels_idx[:, t], preds_idx[:, t],
                                  weights='quadratic', labels=list(range(NUM_CLASSES)))
        except Exception:
            q = 0.0
        qwks.append(q)
    return float(np.mean(qwks)), qwks


def compute_qwk_reg(preds, labels):
    """QWK from continuous predictions in [0,5]. Rounds to nearest 0.5 step."""
    preds_idx  = np.round(np.clip(preds,  0, 5) * 2).astype(int)
    labels_idx = np.round(np.clip(labels, 0, 5) * 2).astype(int)
    return compute_qwk_cls(preds_idx, labels_idx)
