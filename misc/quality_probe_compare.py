#!/usr/bin/env python3
"""Compare existing quality signals against box/DFL-conditioned probe features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


BASE_FEATURES = ("cls_score", "q_score", "final_cls_mul_q", "final_sqrt_cls_mul_q", "final_cls_mul_q2")


def rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.shape[0], dtype=np.float64)
    sx = x[order]
    i = 0
    while i < x.shape[0]:
        j = i + 1
        while j < x.shape[0] and sx[j] == sx[i]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0
        i = j
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2 or np.isclose(x.std(), 0) or np.isclose(y.std(), 0):
        return None
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def average_precision(y: np.ndarray, score: np.ndarray) -> float | None:
    positives = y.sum()
    if positives == 0:
        return None
    order = np.argsort(-score)
    y = y[order]
    precision = np.cumsum(y) / (np.arange(y.size) + 1)
    return float((precision * y).sum() / positives)


def roc_auc(y: np.ndarray, score: np.ndarray) -> float | None:
    pos = y == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = rankdata(score)
    return float((ranks[pos].sum() - n_pos * (n_pos - 1) / 2.0) / (n_pos * n_neg))


def standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = train.mean(0, keepdims=True)
    sigma = train.std(0, keepdims=True)
    sigma[sigma < 1e-6] = 1.0
    return (train - mu) / sigma, (test - mu) / sigma


def fit_ridge(X: np.ndarray, y: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    X = np.c_[np.ones(X.shape[0]), X]
    eye = np.eye(X.shape[1])
    eye[0, 0] = 0.0
    return np.linalg.solve(X.T @ X + lam * eye, X.T @ y)


def predict_linear(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.c_[np.ones(X.shape[0]), X] @ w


def fit_logistic(X: np.ndarray, y: np.ndarray, steps: int = 600, lr: float = 0.05, lam: float = 1e-3) -> np.ndarray:
    X = np.c_[np.ones(X.shape[0]), X]
    w = np.zeros(X.shape[1], dtype=np.float64)
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-(X @ w).clip(-30, 30)))
        grad = X.T @ (p - y) / y.size
        grad[1:] += lam * w[1:]
        w -= lr * grad
    return w


def predict_logistic(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(np.c_[np.ones(X.shape[0]), X] @ w).clip(-30, 30)))


def bin_gap(y_iou: np.ndarray, score: np.ndarray) -> float | None:
    mid = (0.5 <= y_iou) & (y_iou < 0.75)
    high = y_iou >= 0.75
    if not mid.any() or not high.any():
        return None
    return float(score[high].mean() - score[mid].mean())


def evaluate(name: str, X: np.ndarray, y_iou: np.ndarray, y_ap75: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray):
    X_train, X_test = standardize(X[train_idx], X[test_idx])
    iou_w = fit_ridge(X_train, y_iou[train_idx])
    ap75_w = fit_logistic(X_train, y_ap75[train_idx])
    iou_pred = predict_linear(X_test, iou_w)
    ap75_pred = predict_logistic(X_test, ap75_w)
    return {
        "name": name,
        "rows_train": int(train_idx.size),
        "rows_test": int(test_idx.size),
        "iou_spearman": spearman(iou_pred, y_iou[test_idx]),
        "ap75_auc": roc_auc(y_ap75[test_idx], ap75_pred),
        "ap75_ap": average_precision(y_ap75[test_idx], ap75_pred),
        "bin_gap_0.75_plus_minus_0.5_0.75": bin_gap(y_iou[test_idx], ap75_pred),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-frac", type=float, default=0.3)
    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    names = [str(x) for x in data["feature_names"]]
    X = np.asarray(data["X"], dtype=np.float64)
    y_iou = np.asarray(data["y_iou"], dtype=np.float64)
    y_ap75 = np.asarray(data["y_ap75"], dtype=np.float64)
    if X.shape[0] < 10:
        raise SystemExit(f"Need at least 10 rows, got {X.shape[0]}")
    if not np.isfinite(X).all():
        raise SystemExit("Feature matrix contains NaN or inf")

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(X.shape[0])
    split = max(1, min(X.shape[0] - 1, int(X.shape[0] * (1.0 - args.test_frac))))
    train_idx, test_idx = order[:split], order[split:]
    base_idx = np.array([names.index(n) for n in BASE_FEATURES if n in names], dtype=int)
    results = [
        evaluate("A_existing_quality", X[:, base_idx], y_iou, y_ap75, train_idx, test_idx),
        evaluate("B_quality_box_dfl", X, y_iou, y_ap75, train_idx, test_idx),
    ]
    print(json.dumps({"rows": int(X.shape[0]), "features": names, "results": results}, indent=2))


if __name__ == "__main__":
    main()
