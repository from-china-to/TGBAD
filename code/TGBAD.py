"""TGBAD: Tensor-based Granular-Ball Anomaly Detection.

Standalone implementation of TGBAD, "A Tensor-based Granular Ball Model for
Anomaly Detection".  The method constructs granular balls in the joint space
of normalized feature-block views, runs the inherited EMCTGL tensor-graph
ADMM learner on the per-view granular-ball centers, decodes the ball-level
tensor/graph evidence (reconstruction residual and ball density) back to the
original samples through the preserved membership assignment, and fuses the
resulting multi-tensor granular-ball score (GBS) with the sample-level
feature-tail rarity and Isolation Forest scores.  All components are
converted to average-rank percentiles and combined label-free with the
paper's fixed Top-2 rule.

The EMCTGL tensor-graph ADMM solver is embedded, so this single file runs
standalone.  With no arguments it runs the built-in Example.mat demo:

    python TGBAD.py                                # runs data/Example.mat
    python TGBAD.py data.mat                       # any trandata / X-Y .mat
    python TGBAD.py data.mat --evaluate            # also AUC/PR-AUC/F1-Max
    python TGBAD.py data.mat --out scores.npy      # save anomaly scores

Ground-truth labels are used only for offline evaluation, never for fitting.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import scipy.io as sio
from scipy.optimize import minimize
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.stats import rankdata
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.preprocessing import normalize

EPS = 1e-12

ArrayOrViews = Union[np.ndarray, Sequence[np.ndarray]]


# ---------------------------------------------------------------------------
# Configuration and result containers
# ---------------------------------------------------------------------------


@dataclass
class TGBADConfig:
    """Configuration for the TGBAD anomaly detector."""

    mode: str = "granular"  # "granular" | "full"
    embedding_dim: int = 2

    # Single-matrix to multi-view construction.
    n_views: int = 3
    view_strategy: str = "feature_blocks"
    preprocess: str = "minmax_l2"
    max_features: int = 1200

    # Granular-ball generation.
    max_balls: int = 256
    min_ball_size: int = 8
    max_ball_features: int = 128
    split_criterion: str = "radius_size"  # "sse" | "radius" | "radius_size"
    split_method: str = "kmeans"  # "kmeans" | "pca"
    ball_kmeans_restarts: int = 5

    # EMCTGL tensor-graph objective and ADMM settings.
    lambda_reconstruction: float = 10.0
    beta: float = 1.0
    alpha_l2: float = 1e-2
    lambda2: float = 0.1
    sigma: float = 10.0
    admm_iterations: int = 50
    mu0: float = 0.1
    rho: float = 1.5
    mu_max: float = 1e3
    convergence_tol: float = 1e-3
    tlog_epsilon: float = 1e-3
    graph_topk: int = 15
    kmeans_restarts: int = 20
    graph_objective: str = "paper_left"
    relative_convergence: bool = True
    stability_limit: float = 1e6

    # Multi-tensor granular-ball anomaly score (GBS) construction.
    gb_mode: str = "combined_mean"
    # "recon_only" | "ball_struct" | "intra_only" |
    # "combined_max" | "combined_mean" | "combined_product"
    gb_ball_components: Tuple[str, ...] = ("reconstruction", "ball_density")
    gb_ball_fusion: str = "max"  # "max" | "mean" | "entropy"
    intra_mode: str = "absolute"  # "relative" | "absolute"

    # Final label-free fusion of GBS with support components.
    final_fusion: str = "top2"
    # "top2" | "gb_support" | "gb_weighted" | "mean3" | "max3"
    gb_weight: float = 0.6

    # Support components.
    feature_tail_fraction: float = 1.0
    isolation_estimators: int = 100
    isolation_max_samples: int = 256

    # Optional conversion of continuous scores to binary predictions.
    threshold: Optional[float] = None
    contamination: Optional[float] = None
    random_state: int = 0


@dataclass
class GranularBallSet:
    """Granular representation of the normalized input views."""

    assignments: np.ndarray  # N, ball index per sample
    centers: List[np.ndarray]  # per-view ball centers
    radii: np.ndarray  # M
    sizes: np.ndarray  # M


@dataclass
class TGBADResult:
    """Output of a TGBAD fit."""

    scores: np.ndarray
    labels: Optional[np.ndarray]
    components: Dict[str, np.ndarray]
    percentile_components: Dict[str, np.ndarray]
    ball_assignments: np.ndarray
    ball_radii: np.ndarray
    ball_sizes: np.ndarray
    runtime_seconds: float
    diagnostics: Dict[str, Any]


# ---------------------------------------------------------------------------
# View construction and preprocessing
# ---------------------------------------------------------------------------


def construct_views(
    x: np.ndarray,
    *,
    n_views: int = 3,
    strategy: str = "feature_blocks",
    random_state: int = 0,
) -> List[np.ndarray]:
    """Construct multiple unlabeled views from a single feature matrix."""

    matrix = np.asarray(x, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"x must be two-dimensional, got {matrix.shape}")
    n_features = matrix.shape[1]
    if n_features < 1:
        raise ValueError("x has no features")

    strategy = strategy.lower()
    if strategy in {"single", "none", "identity"} or n_views == 1:
        return [np.ascontiguousarray(matrix)]

    actual_views = min(max(1, int(n_views)), n_features)
    if strategy in {"feature_blocks", "blocks", "block"}:
        blocks = np.array_split(np.arange(n_features), actual_views)
        return [np.ascontiguousarray(matrix[:, block]) for block in blocks if block.size]

    if strategy in {"random_subspaces", "random_subspace", "random"}:
        rng = np.random.default_rng(random_state)
        width = max(1, min(n_features, int(math.ceil(n_features * 0.6))))
        return [
            np.ascontiguousarray(
                matrix[:, np.sort(rng.choice(n_features, width, replace=False))]
            )
            for _ in range(actual_views)
        ]

    raise ValueError(f"unknown view construction strategy: {strategy}")


def randomized_svd_project(
    x: np.ndarray, n_components: int, rng: np.random.Generator
) -> np.ndarray:
    """Project a wide view to n_components by a lightweight randomized SVD."""

    n_components = min(n_components, x.shape[0] - 1, x.shape[1])
    if n_components <= 0 or n_components >= x.shape[1]:
        return x

    oversample = min(20, max(4, x.shape[1] - n_components))
    q = n_components + oversample
    omega = rng.standard_normal((x.shape[1], q))
    y = x @ omega
    q_mat, _ = np.linalg.qr(y, mode="reduced")
    b = q_mat.T @ x
    _, _, vt = np.linalg.svd(b, full_matrices=False)
    components = vt[:n_components].T
    return x @ components


def preprocess_views(
    views: Sequence[np.ndarray],
    *,
    max_features: int,
    random_state: int,
    mode: str = "minmax_l2",
) -> List[np.ndarray]:
    """Per-view normalization (and optional SVD reduction of wide views)."""

    rng = np.random.default_rng(random_state)
    processed: List[np.ndarray] = []
    for view in views:
        x = np.asarray(view, dtype=np.float64)
        x = np.nan_to_num(x, copy=False)
        if mode in {"standard", "standard_l2", "std", "std_l2"}:
            mean = x.mean(axis=0, keepdims=True)
            std = x.std(axis=0, keepdims=True)
            std[std < EPS] = 1.0
            x = (x - mean) / std
        elif mode in {"center", "center_l2"}:
            x = x - x.mean(axis=0, keepdims=True)
        elif mode in {"minmax", "minmax_l2"}:
            min_value = x.min(axis=0, keepdims=True)
            span = x.max(axis=0, keepdims=True) - min_value
            span[span < EPS] = 1.0
            x = (x - min_value) / span
        elif mode in {"raw", "raw_l2"}:
            x = x.copy()
        else:
            raise ValueError(f"Unknown preprocess mode: {mode}")

        if x.shape[1] > max_features:
            x = randomized_svd_project(x, max_features, rng)

        if mode.endswith("_l2"):
            x = normalize(x, norm="l2", axis=1)
        processed.append(np.ascontiguousarray(x))
    return processed


# ---------------------------------------------------------------------------
# Granular-ball generation (improved)
# ---------------------------------------------------------------------------


def _ball_sse(joint: np.ndarray, members: np.ndarray) -> float:
    values = joint[members]
    centered = values - values.mean(axis=0, keepdims=True)
    return float(np.sum(centered * centered))


def _ball_radius(joint: np.ndarray, members: np.ndarray, n_views: int) -> float:
    """Radius of a ball: sqrt of mean within-ball squared distance."""

    values = joint[members]
    centered = values - values.mean(axis=0, keepdims=True)
    return math.sqrt(float(np.sum(centered * centered)) / (members.size * n_views))


def _ball_split_priority(
    joint: np.ndarray, members: np.ndarray, *, criterion: str, n_views: int
) -> float:
    """Priority of a ball for splitting (larger is split first)."""

    criterion = criterion.lower()
    if criterion == "sse":
        return _ball_sse(joint, members)
    if criterion in {"radius", "radii"}:
        return _ball_radius(joint, members, n_views)
    if criterion in {"radius_size", "radius*size"}:
        radius = _ball_radius(joint, members, n_views)
        return radius * math.sqrt(members.size)
    raise ValueError(f"unknown split_criterion: {criterion}")


def _split_ball(
    joint: np.ndarray,
    members: np.ndarray,
    *,
    min_size: int,
    restarts: int,
    method: str,
    random_state: int,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Split a ball into two child balls satisfying the minimum size."""

    values = joint[members]
    if members.size < 2 * min_size:
        return None

    method = method.lower()
    if method == "kmeans":
        if members.size > 5000:
            clusterer = MiniBatchKMeans(
                n_clusters=2,
                batch_size=min(4096, members.size),
                n_init=restarts,
                max_iter=100,
                random_state=random_state,
                reassignment_ratio=0.0,
            )
        else:
            clusterer = KMeans(
                n_clusters=2,
                n_init=restarts,
                max_iter=100,
                random_state=random_state,
                algorithm="lloyd",
            )
        local_labels = clusterer.fit_predict(values)
        left = members[local_labels == 0]
        right = members[local_labels == 1]
        if left.size >= min_size and right.size >= min_size:
            return left, right

    # Deterministic principal-component split (fallback or primary method):
    # cut along the first principal axis at the median projection.
    centered = values - values.mean(axis=0, keepdims=True)
    if method == "pca" and centered.shape[1] > 1:
        try:
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            projections = centered @ vt[0]
        except np.linalg.LinAlgError:
            projections = centered[:, 0]
    else:
        # Balanced maximum-variance split.
        split_axis = int(np.argmax(np.var(values, axis=0)))
        projections = values[:, split_axis]

    order = np.argsort(projections, kind="mergesort")
    cut = members.size // 2
    left = members[order[:cut]]
    right = members[order[cut:]]
    if left.size < min_size or right.size < min_size:
        return None
    return left, right


def construct_granular_balls(
    views: Sequence[np.ndarray], config: TGBADConfig
) -> GranularBallSet:
    """Recursively split balls in the joint-view space.

    The ball with the largest split priority (SSE, radius, or radius*size)
    is split until the target number of balls is reached or no ball remains
    eligible.
    """

    n_samples = views[0].shape[0]
    n_views = len(views)
    joint = np.concatenate(views, axis=1)
    if joint.shape[1] > config.max_ball_features:
        joint = randomized_svd_project(
            joint, config.max_ball_features, np.random.default_rng(config.random_state)
        )

    effective_min_size = min(config.min_ball_size, max(1, n_samples // 2))
    target_balls = min(
        n_samples, config.max_balls, max(2, n_samples // effective_min_size)
    )

    balls = [np.arange(n_samples, dtype=int)]
    priorities = [
        _ball_split_priority(
            joint, balls[0], criterion=config.split_criterion, n_views=n_views
        )
    ]
    split_number = 0
    while len(balls) < target_balls:
        candidates = [
            idx for idx, members in enumerate(balls) if members.size >= 2 * effective_min_size
        ]
        if not candidates:
            break
        source_idx = max(candidates, key=lambda idx: priorities[idx])
        split = _split_ball(
            joint,
            balls[source_idx],
            min_size=effective_min_size,
            restarts=config.ball_kmeans_restarts,
            method=config.split_method,
            random_state=config.random_state + split_number,
        )
        if split is None:
            priorities[source_idx] = -np.inf
            if all(not np.isfinite(priorities[idx]) for idx in candidates):
                break
            continue
        left, right = split
        balls[source_idx] = left
        priorities[source_idx] = _ball_split_priority(
            joint, left, criterion=config.split_criterion, n_views=n_views
        )
        balls.append(right)
        priorities.append(
            _ball_split_priority(
                joint, right, criterion=config.split_criterion, n_views=n_views
            )
        )
        split_number += 1

    assignments = np.empty(n_samples, dtype=int)
    for ball_idx, members in enumerate(balls):
        assignments[members] = ball_idx
    n_balls = len(balls)
    sizes = np.bincount(assignments, minlength=n_balls).astype(int)

    centers: List[np.ndarray] = []
    for view in views:
        center_view = np.empty((n_balls, view.shape[1]), dtype=np.float64)
        for ball_idx in range(n_balls):
            center_view[ball_idx] = view[assignments == ball_idx].mean(axis=0)
        centers.append(center_view)

    squared_local = np.zeros(n_samples, dtype=np.float64)
    for view, center_view in zip(views, centers):
        delta = view - center_view[assignments]
        squared_local += np.sum(delta * delta, axis=1)
    squared_local /= n_views
    radii = np.zeros(n_balls, dtype=np.float64)
    for ball_idx in range(n_balls):
        radii[ball_idx] = math.sqrt(float(squared_local[assignments == ball_idx].mean()))

    return GranularBallSet(
        assignments=assignments.astype(int),
        centers=centers,
        radii=radii,
        sizes=sizes,
    )


# ---------------------------------------------------------------------------
# EMCTGL tensor-graph ADMM solver (embedded reproduction)
# ---------------------------------------------------------------------------


def ridge_self_representation(x: np.ndarray, ridge: float) -> np.ndarray:
    """Solve min_B ||X - B X||_F^2 + ridge ||B||_F^2 in sample space."""

    x = np.asarray(x, dtype=np.float64)
    n_samples, n_features = x.shape
    ridge = float(ridge)
    if n_features <= n_samples:
        gram = x.T @ x
        gram.flat[:: n_features + 1] += ridge
        solved = np.linalg.solve(gram, x.T)
        b = x @ solved
    else:
        kernel = x @ x.T
        lhs = kernel.copy()
        lhs.flat[:: n_samples + 1] += ridge
        b = np.linalg.solve(lhs, kernel)

    b = np.asarray(b, dtype=np.float64)
    b[~np.isfinite(b)] = 0.0
    np.fill_diagonal(b, 0.0)
    return b


class StepTensorFFT:
    """Step-structured multi-view data tensor used by EMCTGL."""

    def __init__(self, views: Sequence[np.ndarray]):
        self.views = [np.asarray(view, dtype=np.float64) for view in views]
        self.n_samples = self.views[0].shape[0]
        self.n_views = len(self.views)
        self.view_dims = [view.shape[1] for view in self.views]
        self.total_dim = int(sum(self.view_dims))

        tensor = np.zeros(
            (self.total_dim, self.n_samples, self.n_views), dtype=np.float64
        )
        row_start = 0
        for view_idx, view in enumerate(self.views):
            row_end = row_start + view.shape[1]
            tensor[row_start:row_end, :, view_idx] = view.T
            row_start = row_end

        self.tensor = tensor
        self.tensor_hat = np.fft.fft(tensor, axis=2)
        self.gram_hat = [
            self.tensor_hat[:, :, idx].conj().T @ self.tensor_hat[:, :, idx]
            for idx in range(self.n_views)
        ]

    def solve_frequency(
        self, idx: int, rhs: np.ndarray, *, a: float, c: float
    ) -> np.ndarray:
        """Solve (c I + a X_hat^H X_hat) Z = rhs for one Fourier slice."""

        x_hat = self.tensor_hat[:, :, idx]
        n = self.n_samples
        d = self.total_dim
        if d < n:
            small = np.eye(d, dtype=np.complex128) + (a / c) * (x_hat @ x_hat.conj().T)
            projected = x_hat @ rhs
            correction = x_hat.conj().T @ np.linalg.solve(small, projected)
            return rhs / c - (a / (c * c)) * correction

        lhs = c * np.eye(n, dtype=np.complex128) + a * self.gram_hat[idx]
        return np.linalg.solve(lhs, rhs)

    def reconstruction_error(self, b: np.ndarray) -> float:
        b_hat = np.fft.fft(b, axis=2)
        identity = np.eye(self.n_samples, dtype=np.complex128)
        value = 0.0
        for idx in range(self.n_views):
            residual = self.tensor_hat[:, :, idx] @ (identity - b_hat[:, :, idx])
            value += float(np.linalg.norm(residual, ord="fro") ** 2)
        return value / self.n_views

    def point_reconstruction_residual(self, b: np.ndarray) -> np.ndarray:
        """Return sample-wise ||X - X*B|| residuals in the original domain."""

        b_hat = np.fft.fft(b, axis=2)
        reconstructed_hat = np.empty_like(self.tensor_hat)
        for idx in range(self.n_views):
            reconstructed_hat[:, :, idx] = self.tensor_hat[:, :, idx] @ b_hat[:, :, idx]
        residual = self.tensor - np.real(np.fft.ifft(reconstructed_hat, axis=2))
        return np.sqrt(np.sum(residual * residual, axis=(0, 2)))


def update_b_tensor(
    x_tensor: StepTensorFFT,
    b: np.ndarray,
    c_tensor: np.ndarray,
    q: np.ndarray,
    t: np.ndarray,
    g1: np.ndarray,
    g2: np.ndarray,
    g3: np.ndarray,
    lambda_reconstruction: float,
    mu: float,
) -> np.ndarray:
    n_samples, n_views = b.shape[0], b.shape[2]
    c_hat = np.fft.fft(c_tensor, axis=2)
    q_hat = np.fft.fft(q, axis=2)
    t_hat = np.fft.fft(t, axis=2)
    g1_hat = np.fft.fft(g1, axis=2)
    g2_hat = np.fft.fft(g2, axis=2)
    g3_hat = np.fft.fft(g3, axis=2)
    b_hat = np.empty((n_samples, n_samples, n_views), dtype=np.complex128)
    for view_idx in range(n_views):
        gram = x_tensor.gram_hat[view_idx]
        rhs = (
            2.0 * lambda_reconstruction * gram
            + mu * c_hat[:, :, view_idx]
            - g1_hat[:, :, view_idx]
            + mu * q_hat[:, :, view_idx]
            + g2_hat[:, :, view_idx]
            + mu * t_hat[:, :, view_idx]
            + g3_hat[:, :, view_idx]
        )
        b_hat[:, :, view_idx] = x_tensor.solve_frequency(
            view_idx, rhs, a=2.0 * lambda_reconstruction, c=3.0 * mu
        )
    updated = np.real(np.fft.ifft(b_hat, axis=2))
    for view_idx in range(n_views):
        np.fill_diagonal(updated[:, :, view_idx], 0.0)
    return updated


def update_q_tensor(
    b: np.ndarray,
    q: np.ndarray,
    t: np.ndarray,
    w: np.ndarray,
    g2: np.ndarray,
    lambda2: float,
    mu: float,
    sigma: float,
) -> np.ndarray:
    updated = np.empty_like(q)
    for view_idx in range(b.shape[2]):
        d = sigma_weights(
            w[:, :, view_idx] - 0.5 * (t[:, :, view_idx] + q[:, :, view_idx].T), sigma
        )
        rhs = (
            lambda2 * (w[:, :, view_idx].T * d[None, :])
            - 0.5 * lambda2 * (t[:, :, view_idx].T * d[None, :])
            + mu * b[:, :, view_idx]
            - g2[:, :, view_idx]
        )
        updated[:, :, view_idx] = rhs / (0.5 * lambda2 * d[None, :] + mu)
    return updated


def update_t_tensor(
    b: np.ndarray,
    q: np.ndarray,
    t: np.ndarray,
    w: np.ndarray,
    g3: np.ndarray,
    lambda2: float,
    mu: float,
    sigma: float,
) -> np.ndarray:
    updated = np.empty_like(t)
    for view_idx in range(b.shape[2]):
        d = sigma_weights(
            w[:, :, view_idx] - 0.5 * (t[:, :, view_idx] + q[:, :, view_idx].T), sigma
        )
        rhs = (
            lambda2 * (d[:, None] * w[:, :, view_idx])
            - 0.5 * lambda2 * (d[:, None] * q[:, :, view_idx].T)
            + mu * b[:, :, view_idx]
            - g3[:, :, view_idx]
        )
        updated[:, :, view_idx] = rhs / (0.5 * lambda2 * d[:, None] + mu)
    return updated


def update_w_tensor_paper_left(
    b: np.ndarray,
    q: np.ndarray,
    t: np.ndarray,
    w: np.ndarray,
    e: np.ndarray,
    g: np.ndarray,
    alpha: np.ndarray,
    o: np.ndarray,
    lambda2: float,
    beta: float,
    sigma: float,
) -> np.ndarray:
    """Update W consistently with the paper's left-normalized objective."""

    n_views = w.shape[2]
    p = graph_degree_from_b(b, alpha)
    p_inv = 1.0 / p
    p_inv_sqrt = 1.0 / np.sqrt(p)
    p_sqrt = np.sqrt(p)
    sum_w = sum(alpha[v] * w[:, :, v] for v in range(n_views))
    graph_target = (e @ g.T) * p_sqrt[None, :] + o
    updated = np.empty_like(w)
    for view_idx in range(n_views):
        d = sigma_weights(
            w[:, :, view_idx] - 0.5 * (t[:, :, view_idx] + q[:, :, view_idx].T), sigma
        )
        other = sum_w - alpha[view_idx] * w[:, :, view_idx]
        sym_target = 0.5 * (t[:, :, view_idx] + q[:, :, view_idx].T)
        rhs = (
            alpha[view_idx] * beta * (p_inv_sqrt[:, None] * graph_target)
            - alpha[view_idx] * beta * (p_inv[:, None] * other)
            + lambda2 * (d[:, None] * sym_target)
        )
        denom = (alpha[view_idx] ** 2) * beta * p_inv + lambda2 * d
        updated[:, :, view_idx] = rhs / (denom[:, None] + EPS)
        np.fill_diagonal(updated[:, :, view_idx], 0.0)
    return updated


def update_o_matrix(residual: np.ndarray, lambda_anomaly: float, beta: float) -> np.ndarray:
    """Row-wise l2 shrinkage for beta ||R - O||_F^2 + lambda ||O||_{2,1}."""

    if lambda_anomaly <= 0 or beta <= EPS:
        return np.zeros_like(residual)
    row_norm = np.linalg.norm(residual, axis=1, keepdims=True)
    shrink = np.maximum(1.0 - lambda_anomaly / (2.0 * beta * (row_norm + EPS)), 0.0)
    return residual * shrink


def sigma_weights(matrix: np.ndarray, sigma: float) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=0)
    return (1.0 + sigma) * (norms + 2.0 * sigma) / (2.0 * (norms + sigma) ** 2 + EPS)


def rotate_tensor(tensor: np.ndarray) -> np.ndarray:
    return np.transpose(tensor, (0, 2, 1))


def inverse_rotate_tensor(tensor: np.ndarray) -> np.ndarray:
    return np.transpose(tensor, (0, 2, 1))


def tlog_rank_shrink(tensor: np.ndarray, *, tau: float, epsilon: float) -> np.ndarray:
    tensor_hat = np.fft.fft(tensor, axis=2)
    result_hat = np.empty_like(tensor_hat)
    for idx in range(tensor_hat.shape[2]):
        u, singular, vt = np.linalg.svd(tensor_hat[:, :, idx], full_matrices=False)
        shrunk = tlog_shrink_singular_values(singular, tau=tau, epsilon=epsilon)
        result_hat[:, :, idx] = (u * shrunk) @ vt
    return np.real(np.fft.ifft(result_hat, axis=2))


def tlog_shrink_singular_values(
    singular: np.ndarray, *, tau: float, epsilon: float
) -> np.ndarray:
    discriminant = ((epsilon + singular) ** 2) / 4.0 - tau
    shrunk = np.zeros_like(singular)
    mask = discriminant > 0
    if not np.any(mask):
        return shrunk
    xi = (singular[mask] - epsilon) / 2.0 + np.sqrt(discriminant[mask])
    f_xi = 0.5 * (xi - singular[mask]) ** 2 + tau * np.log(epsilon + xi)
    f_zero = 0.5 * singular[mask] ** 2 + tau * math.log(epsilon)
    keep = (xi > 0) & (f_xi <= f_zero)
    values = np.zeros_like(xi)
    values[keep] = xi[keep]
    shrunk[mask] = values
    return shrunk


def graph_degree_from_b(b: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    p = np.zeros(b.shape[0], dtype=np.float64)
    for view_idx in range(b.shape[2]):
        p += alpha[view_idx] * np.linalg.norm(b[:, :, view_idx], axis=0)
        p += alpha[view_idx] * np.linalg.norm(b[:, :, view_idx], axis=1)
    p[p < EPS] = EPS
    return p


def normalized_view_graphs(w: np.ndarray, p: np.ndarray) -> List[np.ndarray]:
    scale = 1.0 / np.sqrt(p)
    return [w[:, :, idx] * scale[:, None] * scale[None, :] for idx in range(w.shape[2])]


def left_normalized_view_graphs(w: np.ndarray, p: np.ndarray) -> List[np.ndarray]:
    scale = 1.0 / np.sqrt(p)
    return [w[:, :, idx] * scale[:, None] for idx in range(w.shape[2])]


def fuse_normalized_graphs(w: np.ndarray, alpha: np.ndarray, b: np.ndarray) -> np.ndarray:
    p = graph_degree_from_b(b, alpha)
    graphs = normalized_view_graphs(w, p)
    return sum(alpha[idx] * graphs[idx] for idx in range(len(graphs)))


def update_e(fused: np.ndarray, g: np.ndarray, beta: float) -> np.ndarray:
    target = beta * fused @ g
    u, _, vt = np.linalg.svd(target, full_matrices=False)
    return u[:, : g.shape[1]] @ vt[: g.shape[1], :]


def update_e_paper_left(
    left_target: np.ndarray, g: np.ndarray, p: np.ndarray, beta: float
) -> np.ndarray:
    target = beta * (left_target * np.sqrt(p)[None, :]) @ g
    u, _, vt = np.linalg.svd(target, full_matrices=False)
    return u[:, : g.shape[1]] @ vt[: g.shape[1], :]


def update_g_paper_left(
    left_target: np.ndarray, e: np.ndarray, p: np.ndarray
) -> np.ndarray:
    return np.maximum((1.0 / np.sqrt(p))[:, None] * (left_target.T @ e), 0.0)


def update_alpha_qp(
    graphs: Sequence[np.ndarray],
    target: np.ndarray,
    beta: float,
    lambda1: float,
    current: np.ndarray,
) -> np.ndarray:
    n_views = len(graphs)
    q_mat = np.empty((n_views, n_views), dtype=np.float64)
    g_vec = np.empty(n_views, dtype=np.float64)
    for i in range(n_views):
        g_vec[i] = 2.0 * beta * float(np.sum(graphs[i] * target))
        for j in range(n_views):
            q_mat[i, j] = beta * float(np.sum(graphs[i] * graphs[j]))
    q_mat += lambda1 * np.eye(n_views)

    def objective(alpha: np.ndarray) -> float:
        return float(alpha @ q_mat @ alpha - alpha @ g_vec)

    def gradient(alpha: np.ndarray) -> np.ndarray:
        return 2.0 * q_mat @ alpha - g_vec

    constraints = {
        "type": "eq",
        "fun": lambda a: np.sum(a) - 1.0,
        "jac": lambda a: np.ones_like(a),
    }
    bounds = [(0.0, 1.0)] * n_views
    result = minimize(
        objective,
        current,
        jac=gradient,
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
        options={"maxiter": 100, "ftol": 1e-10, "disp": False},
    )
    if result.success and np.all(np.isfinite(result.x)):
        return project_simplex(result.x)
    return project_simplex(current)


def project_simplex(values: np.ndarray) -> np.ndarray:
    """Project a vector onto the probability simplex."""

    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if v.size == 1:
        return np.ones_like(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho_candidates = u * np.arange(1, v.size + 1) > (cssv - 1)
    if not np.any(rho_candidates):
        return np.full_like(v, 1.0 / v.size)
    rho = np.nonzero(rho_candidates)[0][-1]
    theta = (cssv[rho] - 1.0) / (rho + 1)
    w = np.maximum(v - theta, 0.0)
    total = w.sum()
    if total <= EPS:
        return np.full_like(v, 1.0 / v.size)
    return w / total


def tlog_norm(rotated_tensor: np.ndarray, epsilon: float) -> float:
    value = 0.0
    tensor_hat = np.fft.fft(rotated_tensor, axis=2)
    for idx in range(tensor_hat.shape[2]):
        singular = np.linalg.svd(tensor_hat[:, :, idx], compute_uv=False)
        value += float(np.sum(np.log(epsilon + singular)))
    return value / max(1, rotated_tensor.shape[2])


def sigma_surrogate_error(
    w: np.ndarray, q: np.ndarray, t: np.ndarray, sigma: float
) -> float:
    value = 0.0
    for idx in range(w.shape[2]):
        m = w[:, :, idx] - 0.5 * (t[:, :, idx] + q[:, :, idx].T)
        norms = np.linalg.norm(m, axis=0)
        value += float(np.sum((1.0 + sigma) * norms * norms / (norms + sigma + EPS)))
    return value


def keep_topk_symmetric(matrix: np.ndarray, topk: int) -> np.ndarray:
    """Keep the strongest top-k neighbors per row and symmetrize."""

    n = matrix.shape[0]
    topk = int(max(1, min(topk, n - 1)))
    if topk >= n - 1:
        return 0.5 * (matrix + matrix.T)

    kept = np.zeros_like(matrix)
    idx = np.argpartition(matrix, -topk, axis=1)[:, -topk:]
    rows = np.arange(n)[:, None]
    kept[rows, idx] = matrix[rows, idx]
    kept = np.maximum(kept, kept.T)
    np.fill_diagonal(kept, 0.0)
    return kept


def spectral_embedding_from_affinity(affinity: np.ndarray, n_clusters: int) -> np.ndarray:
    """Compute normalized spectral embedding from an affinity graph."""

    affinity = np.asarray(affinity, dtype=np.float64)
    affinity = 0.5 * (affinity + affinity.T)
    np.fill_diagonal(affinity, 0.0)
    degree = affinity.sum(axis=1)
    isolated = degree <= EPS
    if np.any(isolated):
        affinity = affinity.copy()
        affinity[isolated, isolated] = EPS
        degree = affinity.sum(axis=1)

    inv_sqrt_degree = 1.0 / np.sqrt(degree + EPS)
    lap = affinity * inv_sqrt_degree[:, None] * inv_sqrt_degree[None, :]

    n = lap.shape[0]
    k = min(max(2, n_clusters), n - 1)
    try:
        sparse_lap = sparse.csr_matrix(lap)
        values, vectors = eigsh(sparse_lap, k=k, which="LA", tol=1e-4)
        order = np.argsort(values)[::-1]
        embedding = vectors[:, order[:n_clusters]]
    except Exception:
        values, vectors = np.linalg.eigh(lap)
        order = np.argsort(values)[::-1]
        embedding = vectors[:, order[:n_clusters]]

    return normalize(np.asarray(embedding), norm="l2", axis=1)


def run_kmeans(
    embedding: np.ndarray, n_clusters: int, random_state: int, n_init: int
) -> np.ndarray:
    model = KMeans(
        n_clusters=n_clusters,
        n_init=n_init,
        random_state=random_state,
        algorithm="lloyd",
    )
    return model.fit_predict(embedding)


def relative_inf_change(current: np.ndarray, previous: np.ndarray) -> float:
    numerator = float(np.max(np.abs(current - previous)))
    denominator = max(
        1.0,
        float(np.max(np.abs(current))),
        float(np.max(np.abs(previous))),
    )
    return numerator / denominator


def robust_zscore(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    median = np.median(arr)
    mad = np.median(np.abs(arr - median))
    if mad < EPS:
        std = arr.std()
        if std < EPS:
            return np.zeros_like(arr)
        return (arr - arr.mean()) / std
    return (arr - median) / (1.4826 * mad + EPS)


def view_disagreement_score(w: np.ndarray) -> np.ndarray:
    n_samples, _, n_views = w.shape
    row_mean = np.zeros((n_samples, n_samples), dtype=np.float64)
    row_sq_mean = np.zeros_like(row_mean)
    col_mean = np.zeros_like(row_mean)
    col_sq_mean = np.zeros_like(row_mean)
    for view_idx in range(n_views):
        graph = w[:, :, view_idx]
        row_normed = graph / (np.linalg.norm(graph, axis=1, keepdims=True) + EPS)
        col_normed = graph / (np.linalg.norm(graph, axis=0, keepdims=True) + EPS)
        row_mean += row_normed
        row_sq_mean += row_normed * row_normed
        col_mean += col_normed
        col_sq_mean += col_normed * col_normed
    row_mean /= n_views
    row_sq_mean /= n_views
    col_mean /= n_views
    col_sq_mean /= n_views
    row_var = np.maximum(row_sq_mean - row_mean * row_mean, 0.0)
    col_var = np.maximum(col_sq_mean - col_mean * col_mean, 0.0)
    return np.sqrt(row_var.sum(axis=1)) + np.sqrt(col_var.sum(axis=0))


def compute_anomaly_scores(
    x_tensor: StepTensorFFT,
    b: np.ndarray,
    w: np.ndarray,
    q: np.ndarray,
    t: np.ndarray,
    e: np.ndarray,
    g: np.ndarray,
    alpha: np.ndarray,
    o: Optional[np.ndarray] = None,
    *,
    graph_objective: str = "paper_left",
) -> Dict[str, np.ndarray]:
    """Compute EMCTGL-derived point-wise anomaly signals at ball level."""

    reconstruction = x_tensor.point_reconstruction_residual(b)

    graph_consistency = np.zeros(b.shape[0], dtype=np.float64)
    for view_idx in range(w.shape[2]):
        m = w[:, :, view_idx] - 0.5 * (t[:, :, view_idx] + q[:, :, view_idx].T)
        graph_consistency += np.linalg.norm(m, axis=1) + np.linalg.norm(m, axis=0)

    p = graph_degree_from_b(b, alpha)
    if o is None:
        o = np.zeros((b.shape[0], b.shape[0]), dtype=np.float64)
    if graph_objective == "paper_left":
        graph_views = left_normalized_view_graphs(w, p)
        fused = sum(alpha[idx] * graph_views[idx] for idx in range(w.shape[2]))
        factor_residual = fused - (e @ g.T) * np.sqrt(p)[None, :] - o
    else:
        graph_views = normalized_view_graphs(w, p)
        fused = sum(alpha[idx] * graph_views[idx] for idx in range(w.shape[2]))
        factor_residual = fused - e @ g.T - o
    decomposition = np.linalg.norm(factor_residual, axis=1) + np.linalg.norm(
        factor_residual, axis=0
    )
    explicit_anomaly = np.linalg.norm(o, axis=1)

    view_disagreement = view_disagreement_score(w)
    hub = np.abs(robust_zscore(np.log(p + EPS)))

    return {
        "reconstruction": reconstruction,
        "graph_consistency": graph_consistency,
        "decomposition": decomposition,
        "explicit_anomaly": explicit_anomaly,
        "view_disagreement": view_disagreement,
        "hub": hub,
    }


def emctgl_admm(
    views: Sequence[np.ndarray],
    n_clusters: int,
    *,
    lambda_reconstruction: float,
    beta: float,
    lambda1: float,
    lambda2: float,
    sigma: float,
    max_iter: int,
    mu0: float,
    rho: float,
    mu_max: float,
    tol: float,
    tlog_epsilon: float,
    graph_topk: int,
    random_state: int,
    kmeans_restarts: int,
    graph_objective: str = "paper_left",
    relative_convergence: bool = True,
    stability_limit: float = 1e6,
) -> Dict[str, Any]:
    """ADMM solver following Algorithm 1 of the EMCTGL paper."""

    n_samples = views[0].shape[0]
    n_views = len(views)
    x_tensor = StepTensorFFT(views)

    b = np.stack(
        [
            ridge_self_representation(view, ridge=1.0 / max(lambda_reconstruction, EPS))
            for view in views
        ],
        axis=2,
    )
    q = b.copy()
    t = b.copy()
    c_tensor = b.copy()
    g1 = np.zeros_like(b)
    g2 = np.zeros_like(b)
    g3 = np.zeros_like(b)

    w = np.empty_like(b)
    for view_idx in range(n_views):
        affinity = 0.5 * (np.abs(b[:, :, view_idx]) + np.abs(b[:, :, view_idx].T))
        affinity = keep_topk_symmetric(affinity, graph_topk)
        w[:, :, view_idx] = affinity

    alpha = np.full(n_views, 1.0 / n_views, dtype=np.float64)
    o = np.zeros((n_samples, n_samples), dtype=np.float64)
    fused0 = fuse_normalized_graphs(w, alpha, b)
    e = spectral_embedding_from_affinity(np.maximum(fused0, 0.0), n_clusters)
    init_labels = run_kmeans(e, n_clusters, random_state, kmeans_restarts)
    g = np.zeros((n_samples, n_clusters), dtype=np.float64)
    g[np.arange(n_samples), init_labels] = 1.0

    mu = float(mu0)
    objectives: List[float] = []
    residuals: List[float] = []
    converged = False
    safeguard_triggered = False
    stop_reason = "max_iterations"
    last_residual_components: Dict[str, float] = {}

    for iteration in range(int(max_iter)):
        old = {
            "b": b.copy(),
            "q": q.copy(),
            "c": c_tensor.copy(),
            "w": w.copy(),
            "t": t.copy(),
            "e": e.copy(),
            "g": g.copy(),
            "alpha": alpha.copy(),
            "o": o.copy(),
        }

        b = update_b_tensor(
            x_tensor, b, c_tensor, q, t, g1, g2, g3, lambda_reconstruction, mu
        )
        q = update_q_tensor(b, q, t, w, g2, lambda2, mu, sigma)
        c_tensor = inverse_rotate_tensor(
            tlog_rank_shrink(rotate_tensor(b + g1 / mu), tau=1.0 / mu, epsilon=tlog_epsilon)
        )
        w = update_w_tensor_paper_left(b, q, t, w, e, g, alpha, o, lambda2, beta, sigma)
        t = update_t_tensor(b, q, t, w, g3, lambda2, mu, sigma)

        p = graph_degree_from_b(b, alpha)
        a_views = left_normalized_view_graphs(w, p)
        fused = sum(alpha[v] * a_views[v] for v in range(n_views))
        p_sqrt = np.sqrt(p)
        target_without_anomaly = fused - o
        e = update_e_paper_left(target_without_anomaly, g, p, beta)
        g = update_g_paper_left(target_without_anomaly, e, p)
        graph_target = (e @ g.T) * p_sqrt[None, :]
        o = update_o_matrix(fused - graph_target, 0.0, beta)
        alpha = update_alpha_qp(a_views, graph_target + o, beta, lambda1, alpha)
        graph_residual = fused - graph_target - o

        state_max = max(
            float(np.max(np.abs(value)))
            for value in (b, q, c_tensor, w, t, e, g, alpha, o)
        )
        if not np.isfinite(state_max) or state_max > stability_limit:
            b = old["b"]
            q = old["q"]
            c_tensor = old["c"]
            w = old["w"]
            t = old["t"]
            e = old["e"]
            g = old["g"]
            alpha = old["alpha"]
            o = old["o"]
            safeguard_triggered = True
            stop_reason = "stability_safeguard"
            break

        g1 = g1 + mu * (b - c_tensor)
        g2 = g2 + mu * (q - b)
        g3 = g3 + mu * (t - b)
        mu = min(rho * mu, mu_max)

        objective = (
            tlog_norm(rotate_tensor(c_tensor), tlog_epsilon)
            + lambda_reconstruction * x_tensor.reconstruction_error(b)
            + lambda1 * float(np.dot(alpha, alpha))
            + lambda2 * sigma_surrogate_error(w, q, t, sigma)
            + beta * float(np.linalg.norm(graph_residual, ord="fro") ** 2)
        )
        objectives.append(float(objective))

        if relative_convergence:
            current_factor = e @ g.T
            previous_factor = old["e"] @ old["g"].T
            named_pairs = {
                "w": (w, old["w"]),
                "b": (b, old["b"]),
                "c": (c_tensor, old["c"]),
                "q": (q, old["q"]),
                "alpha": (alpha, old["alpha"]),
                "factor": (current_factor, previous_factor),
                "t": (t, old["t"]),
                "o": (o, old["o"]),
            }
            last_residual_components = {
                name: relative_inf_change(new, previous)
                for name, (new, previous) in named_pairs.items()
            }
            update_residual = max(last_residual_components.values())
            constraint_scale = max(1.0, float(np.max(np.abs(b))))
            last_residual_components.update(
                {
                    "constraint_b_c": float(np.max(np.abs(b - c_tensor)))
                    / constraint_scale,
                    "constraint_q_b": float(np.max(np.abs(q - b)))
                    / constraint_scale,
                    "constraint_t_b": float(np.max(np.abs(t - b)))
                    / constraint_scale,
                }
            )
            primal_residual = max(
                last_residual_components["constraint_b_c"],
                last_residual_components["constraint_q_b"],
                last_residual_components["constraint_t_b"],
            )
            residual = max(update_residual, primal_residual)
        else:
            named_pairs = {
                "w": (w, old["w"]),
                "b": (b, old["b"]),
                "c": (c_tensor, old["c"]),
                "q": (q, old["q"]),
                "alpha": (alpha, old["alpha"]),
                "e": (e, old["e"]),
                "g": (g, old["g"]),
                "t": (t, old["t"]),
                "o": (o, old["o"]),
            }
            last_residual_components = {
                name: float(np.max(np.abs(new - previous)))
                for name, (new, previous) in named_pairs.items()
            }
            residual = max(last_residual_components.values())
        residuals.append(residual)
        if residual < tol:
            converged = True
            stop_reason = "converged"
            break

    scores = compute_anomaly_scores(
        x_tensor,
        b,
        w,
        q,
        t,
        e,
        g,
        alpha,
        o,
        graph_objective=graph_objective,
    )
    return {
        "B": b,
        "Q": q,
        "T": t,
        "W": w,
        "E": e,
        "G": g,
        "O": o,
        "alpha": alpha,
        "scores": scores,
        "objectives": objectives,
        "residuals": residuals,
        "iterations": len(residuals),
        "converged": converged,
        "safeguard_triggered": safeguard_triggered,
        "stop_reason": stop_reason,
        "last_residual_components": last_residual_components,
    }


# ---------------------------------------------------------------------------
# Score construction
# ---------------------------------------------------------------------------


def average_percentile_rank(values: np.ndarray) -> np.ndarray:
    """Map a score vector to average-rank percentiles in [0, 1]."""

    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size <= 1:
        return np.zeros_like(arr)
    return (rankdata(arr, method="average") - 1.0) / (arr.size - 1.0)


def bridge_score(w: np.ndarray, b: np.ndarray, alpha: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Measure the fraction of graph mass crossing learned structure groups."""

    fused = fuse_normalized_graphs(w, alpha, b)
    affinity = np.maximum(0.5 * (fused + fused.T), 0.0)
    groups = np.argmax(g, axis=1)
    cross = groups[:, None] != groups[None, :]
    cross_mass = np.sum(affinity * cross, axis=1)
    total_mass = np.sum(affinity, axis=1)
    return cross_mass / (total_mass + EPS)


def robust_standardize(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    if mad >= EPS:
        return (arr - median) / (1.4826 * mad + EPS)
    std = float(arr.std())
    if std < EPS:
        return np.zeros_like(arr)
    return (arr - float(arr.mean())) / std


def positive_tail_information(values: np.ndarray) -> float:
    """Return one minus normalized entropy of the positive robust-z tail."""

    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size <= 1:
        return 0.0
    tail = np.maximum(robust_standardize(arr), 0.0)
    total = float(tail.sum())
    if total <= EPS:
        return 0.0
    probabilities = tail / total
    positive = probabilities > 0
    entropy = -float(np.sum(probabilities[positive] * np.log(probabilities[positive])))
    entropy /= math.log(arr.size)
    return float(np.clip(1.0 - entropy, 0.0, 1.0))


def ball_density_score(radii: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """Size-normalized radius prior: dispersed small balls are suspicious."""

    radii = np.asarray(radii, dtype=np.float64)
    sizes = np.asarray(sizes, dtype=np.float64)
    median_size = float(np.median(sizes))
    return radii * np.sqrt(median_size / (sizes + EPS))


def intra_ball_deviation(
    views: Sequence[np.ndarray],
    balls: GranularBallSet,
    *,
    mode: str = "relative",
) -> np.ndarray:
    """Sample-resolving intra-ball deviation from the ball center.

    ``relative`` normalizes by the ball radius (local anomaly factor);
    ``absolute`` keeps the raw joint-space distance (global dispersion).
    """

    assignments = balls.assignments
    squared = np.zeros(assignments.size, dtype=np.float64)
    for view, center_view in zip(views, balls.centers):
        delta = view - center_view[assignments]
        squared += np.sum(delta * delta, axis=1)
    distance = np.sqrt(squared / len(views))
    if mode == "absolute":
        return distance
    return distance / (balls.radii[assignments] + EPS)


def fuse_ball_components(
    ball_components: Mapping[str, np.ndarray],
    *,
    fusion: str = "max",
) -> np.ndarray:
    """Fuse ball-level signals into one ball structure score (rank space)."""

    names = list(ball_components)
    if not names:
        raise ValueError("no ball components given")
    ranks = {
        name: average_percentile_rank(np.asarray(values, dtype=np.float64))
        for name, values in ball_components.items()
    }
    matrix = np.column_stack([ranks[name] for name in names])
    if fusion == "max":
        return np.max(matrix, axis=1)
    if fusion == "mean":
        return np.mean(matrix, axis=1)
    if fusion == "entropy":
        information = np.array(
            [positive_tail_information(ball_components[name]) for name in names],
            dtype=np.float64,
        )
        if float(information.sum()) <= EPS:
            weights = np.full(len(names), 1.0 / len(names))
        else:
            weights = information / information.sum()
        return matrix @ weights
    raise ValueError(f"unknown ball fusion: {fusion}")


def build_granular_ball_score(
    views: Sequence[np.ndarray],
    balls: GranularBallSet,
    state: Mapping[str, Any],
    config: TGBADConfig,
) -> Dict[str, np.ndarray]:
    """Build ball-level, intra-ball, and composite granular-ball scores.

    Returns a dict with sample-level arrays for:
      * every ball-level component decoded to samples (e.g. "reconstruction")
      * "ball_density", "local_residual"
      * "ball_struct" (fused ball structure score)
      * "gbs" (the composite multi-tensor granular-ball anomaly score)
    """

    assignments = balls.assignments
    unit_scores = dict(state["scores"])
    unit_scores["bridge"] = bridge_score(
        state["W"], state["B"], state["alpha"], state["G"]
    )
    density = ball_density_score(balls.radii, balls.sizes)
    unit_scores["ball_density"] = density
    if config.intra_mode == "absolute":
        unit_scores["local_residual"] = intra_ball_deviation(
            views, balls, mode="absolute"
        )
    else:
        unit_scores["local_residual"] = intra_ball_deviation(
            views, balls, mode="relative"
        )

    components: Dict[str, np.ndarray] = {}
    for name, values in unit_scores.items():
        arr = np.asarray(values, dtype=np.float64)
        if arr.shape[0] == assignments.size:
            # Sample-level signal (e.g. local_residual): keep as is.
            components[name] = arr.copy()
        else:
            # Ball-level signal: decode through the membership assignment.
            components[name] = arr[assignments]

    # Ball structure score: fuse selected ball-level signals in rank space.
    ball_names = [
        name for name in config.gb_ball_components if name in unit_scores
    ]
    ball_struct = fuse_ball_components(
        {name: unit_scores[name] for name in ball_names},
        fusion=config.gb_ball_fusion,
    )
    components["ball_struct"] = ball_struct[assignments]

    # Composite multi-tensor granular-ball anomaly score.
    gb_mode = config.gb_mode.lower()
    p_ball = ball_struct[assignments]
    p_local = average_percentile_rank(unit_scores["local_residual"])
    p_recon = average_percentile_rank(unit_scores["reconstruction"])[assignments]
    if gb_mode == "recon_only":
        gbs = p_recon
    elif gb_mode == "ball_struct":
        gbs = p_ball
    elif gb_mode == "intra_only":
        gbs = p_local
    elif gb_mode == "combined_max":
        gbs = np.maximum(p_ball, p_local)
    elif gb_mode == "combined_mean":
        gbs = 0.5 * (p_ball + p_local)
    elif gb_mode == "combined_product":
        gbs = np.sqrt(np.maximum(p_ball, 0.0) * np.maximum(p_local, 0.0))
    else:
        raise ValueError(f"unknown gb_mode: {config.gb_mode}")
    components["gbs"] = gbs

    return components


def feature_tail_score(
    views: Sequence[np.ndarray], *, top_fraction: float = 1.0
) -> np.ndarray:
    """Two-sided empirical feature-tail rarity score (label-free)."""

    view_scores: List[np.ndarray] = []
    for view in views:
        matrix = np.asarray(view, dtype=np.float64)
        n_samples, n_features = matrix.shape
        if n_samples <= 1:
            view_scores.append(np.zeros(n_samples, dtype=np.float64))
            continue
        tails = np.empty_like(matrix, dtype=np.float64)
        for feature_idx in range(n_features):
            values = matrix[:, feature_idx]
            unique, inverse, counts = np.unique(
                values, return_inverse=True, return_counts=True
            )
            categorical_limit = min(32, max(2, int(math.sqrt(n_samples))))
            if unique.size <= categorical_limit:
                probability = counts[inverse] / n_samples
            else:
                percentile = average_percentile_rank(values)
                probability = np.maximum(
                    2.0 * np.minimum(percentile, 1.0 - percentile), 1.0 / n_samples
                )
            tails[:, feature_idx] = -np.log(np.maximum(probability, 1.0 / n_samples))
        keep = max(1, min(n_features, int(math.ceil(top_fraction * n_features))))
        if keep == n_features:
            score = tails.mean(axis=1)
        else:
            strongest = np.partition(tails, n_features - keep, axis=1)[:, -keep:]
            score = strongest.mean(axis=1)
        view_scores.append(score)
    return np.mean(np.column_stack(view_scores), axis=1)


def isolation_forest_score(
    views: Sequence[np.ndarray],
    *,
    n_estimators: int,
    max_samples: int,
    max_features: int,
    random_state: int,
) -> np.ndarray:
    """Scalable label-free isolation score on balanced multi-view features."""

    scaled_views: List[np.ndarray] = []
    for view in views:
        matrix = np.nan_to_num(np.asarray(view, dtype=np.float64), copy=True)
        minimum = matrix.min(axis=0, keepdims=True)
        span = matrix.max(axis=0, keepdims=True) - minimum
        span[span < EPS] = 1.0
        matrix = (matrix - minimum) / span
        matrix /= math.sqrt(max(1, matrix.shape[1]))
        scaled_views.append(matrix)
    joint = np.ascontiguousarray(np.concatenate(scaled_views, axis=1))
    if joint.shape[1] > max_features:
        rng = np.random.default_rng(random_state)
        joint = randomized_svd_project(joint, max_features, rng)
    detector = IsolationForest(
        n_estimators=n_estimators,
        max_samples=min(max_samples, joint.shape[0]),
        contamination="auto",
        max_features=1.0,
        bootstrap=False,
        n_jobs=1,
        random_state=random_state,
    )
    detector.fit(joint)
    return -detector.score_samples(joint)


def fuse_final_score(
    components: Mapping[str, np.ndarray],
    *,
    final_fusion: str,
    gb_weight: float = 0.6,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Combine the granular-ball score with the support components.

    All inputs are converted to average-rank percentiles first.  Supported
    protocols:
      * ``top2``       -- paper Top-2 of {gbs, tail, iso}
      * ``gb_support`` -- 0.5*gbs + 0.5*max(tail, iso)  (GB dominant)
      * ``gb_weighted``-- w*gbs + (1-w)/2*(tail + iso)  (GB dominant)
      * ``mean3``      -- mean of the three percentiles
      * ``max3``       -- max of the three percentiles
    """

    percentiles = {
        name: average_percentile_rank(np.asarray(values, dtype=np.float64))
        for name, values in components.items()
    }
    p_gbs = percentiles["gbs"]
    p_tail = percentiles["feature_tail"]
    p_iso = percentiles["isolation"]
    mode = final_fusion.lower()
    if mode == "top2":
        matrix = np.column_stack([p_gbs, p_tail, p_iso])
        total = np.partition(matrix, 1, axis=1)[:, 1:].mean(axis=1)
    elif mode == "gb_support":
        total = 0.5 * p_gbs + 0.5 * np.maximum(p_tail, p_iso)
    elif mode == "gb_weighted":
        weight = float(np.clip(gb_weight, 0.0, 1.0))
        total = weight * p_gbs + (1.0 - weight) * 0.5 * (p_tail + p_iso)
    elif mode == "mean3":
        total = (p_gbs + p_tail + p_iso) / 3.0
    elif mode == "max3":
        total = np.maximum(np.maximum(p_gbs, p_tail), p_iso)
    else:
        raise ValueError(f"unknown final_fusion: {final_fusion}")
    percentiles["total"] = total
    return total, percentiles


def scores_to_labels(
    scores: np.ndarray,
    *,
    threshold: Optional[float] = None,
    contamination: Optional[float] = None,
) -> np.ndarray:
    """Convert continuous anomaly scores to 0/1 predictions."""

    arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    if threshold is not None and contamination is not None:
        raise ValueError("set either threshold or contamination, not both")
    if threshold is not None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must lie in [0, 1]")
        return (arr > threshold).astype(int)
    if contamination is None:
        raise ValueError("threshold or contamination is required")
    if not 0.0 < contamination < 1.0:
        raise ValueError("contamination must lie in (0, 1)")

    n_anomalies = max(1, min(arr.size, int(math.ceil(contamination * arr.size))))
    order = np.argsort(arr, kind="mergesort")
    labels = np.zeros(arr.size, dtype=int)
    labels[order[-n_anomalies:]] = 1
    return labels


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------


class TGBAD:
    """Tensor-based granular-ball anomaly detector."""

    def __init__(self, config: Optional[TGBADConfig] = None):
        self.config = config or TGBADConfig()
        self.result_: Optional[TGBADResult] = None
        self.decision_scores_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None

    def fit(self, data: ArrayOrViews) -> "TGBAD":
        self.result_ = self._fit(data)
        self.decision_scores_ = self.result_.scores
        self.labels_ = self.result_.labels
        return self

    def fit_predict(self, data: ArrayOrViews) -> TGBADResult:
        self.fit(data)
        assert self.result_ is not None
        return self.result_

    def decision_function(self) -> np.ndarray:
        if self.decision_scores_ is None:
            raise RuntimeError("TGBAD is not fitted")
        return self.decision_scores_.copy()

    def predict(
        self,
        *,
        threshold: Optional[float] = None,
        contamination: Optional[float] = None,
    ) -> np.ndarray:
        if self.decision_scores_ is None:
            raise RuntimeError("TGBAD is not fitted")
        return scores_to_labels(
            self.decision_scores_,
            threshold=self.config.threshold if threshold is None else threshold,
            contamination=(
                self.config.contamination if contamination is None else contamination
            ),
        )

    def fit_signals(self, data: ArrayOrViews) -> Dict[str, Any]:
        """Fit the model once and dump all raw signals for offline analysis.

        The returned dict contains the ball-level tensor/graph signals, the
        intra-ball deviations, the support scores, and the ball assignments,
        so that any score-construction variant can be evaluated afterwards
        without re-running the EMCTGL solver.
        """

        start = time.perf_counter()
        cfg = self.config

        if isinstance(data, np.ndarray):
            raw_views = construct_views(
                data,
                n_views=cfg.n_views,
                strategy=cfg.view_strategy,
                random_state=cfg.random_state,
            )
        else:
            raw_views = [np.asarray(view, dtype=np.float64) for view in data]
        if not raw_views:
            raise ValueError("at least one view is required")
        n_samples = raw_views[0].shape[0]
        if n_samples < 2:
            raise ValueError("TGBAD requires at least two samples")

        views = preprocess_views(
            raw_views,
            max_features=cfg.max_features,
            random_state=cfg.random_state,
            mode=cfg.preprocess,
        )

        if cfg.mode.lower() == "granular":
            balls = construct_granular_balls(views, cfg)
            model_views = balls.centers
        else:
            assignments = np.arange(n_samples, dtype=int)
            balls = GranularBallSet(
                assignments=assignments,
                centers=[view.copy() for view in views],
                radii=np.zeros(n_samples, dtype=np.float64),
                sizes=np.ones(n_samples, dtype=int),
            )
            model_views = balls.centers

        n_units = model_views[0].shape[0]
        latent_dim = min(max(1, int(cfg.embedding_dim)), max(1, n_units - 1))
        state = emctgl_admm(
            model_views,
            latent_dim,
            lambda_reconstruction=cfg.lambda_reconstruction,
            beta=cfg.beta,
            lambda1=cfg.alpha_l2,
            lambda2=cfg.lambda2,
            sigma=cfg.sigma,
            max_iter=cfg.admm_iterations,
            mu0=cfg.mu0,
            rho=cfg.rho,
            mu_max=cfg.mu_max,
            tol=cfg.convergence_tol,
            tlog_epsilon=cfg.tlog_epsilon,
            graph_topk=cfg.graph_topk,
            random_state=cfg.random_state,
            kmeans_restarts=cfg.kmeans_restarts,
            graph_objective=cfg.graph_objective,
            relative_convergence=cfg.relative_convergence,
            stability_limit=cfg.stability_limit,
        )

        unit_scores = dict(state["scores"])
        unit_scores["bridge"] = bridge_score(
            state["W"], state["B"], state["alpha"], state["G"]
        )
        unit_scores["ball_density"] = ball_density_score(balls.radii, balls.sizes)

        signals: Dict[str, Any] = {
            "n": int(n_samples),
            "n_balls": int(len(balls.radii)),
            "assignments": balls.assignments.astype(int),
            "ball_raw": {name: np.asarray(values, dtype=np.float64).copy()
                         for name, values in unit_scores.items()},
            "ball_radii": balls.radii.copy(),
            "ball_sizes": balls.sizes.copy(),
            "local_raw": intra_ball_deviation(views, balls, mode="relative"),
            "local_abs_raw": intra_ball_deviation(views, balls, mode="absolute"),
            "views_concat": np.concatenate(views, axis=1),
            "centers_concat": np.concatenate(balls.centers, axis=1),
            "tail_raw": feature_tail_score(
                raw_views, top_fraction=cfg.feature_tail_fraction
            ),
            "iso_raw": isolation_forest_score(
                raw_views,
                n_estimators=cfg.isolation_estimators,
                max_samples=cfg.isolation_max_samples,
                max_features=cfg.max_features,
                random_state=cfg.random_state,
            ),
            "diagnostics": {
                "n_samples": int(n_samples),
                "n_balls": int(len(balls.radii)),
                "iterations": int(state.get("iterations", 0)),
                "converged": bool(state.get("converged", False)),
                "stop_reason": str(state.get("stop_reason", "unknown")),
                "safeguard_triggered": bool(
                    state.get("safeguard_triggered", False)
                ),
                "runtime_seconds": time.perf_counter() - start,
            },
        }
        return signals

    def _fit(self, data: ArrayOrViews) -> TGBADResult:
        start = time.perf_counter()
        cfg = self.config

        if isinstance(data, np.ndarray):
            raw_views = construct_views(
                data,
                n_views=cfg.n_views,
                strategy=cfg.view_strategy,
                random_state=cfg.random_state,
            )
        else:
            raw_views = [np.asarray(view, dtype=np.float64) for view in data]
        if not raw_views:
            raise ValueError("at least one view is required")
        n_samples = raw_views[0].shape[0]
        if n_samples < 2:
            raise ValueError("TGBAD requires at least two samples")

        views = preprocess_views(
            raw_views,
            max_features=cfg.max_features,
            random_state=cfg.random_state,
            mode=cfg.preprocess,
        )

        if cfg.mode.lower() == "granular":
            balls = construct_granular_balls(views, cfg)
            model_views = balls.centers
        else:
            assignments = np.arange(n_samples, dtype=int)
            balls = GranularBallSet(
                assignments=assignments,
                centers=[view.copy() for view in views],
                radii=np.zeros(n_samples, dtype=np.float64),
                sizes=np.ones(n_samples, dtype=int),
            )
            model_views = balls.centers

        n_units = model_views[0].shape[0]
        latent_dim = min(max(1, int(cfg.embedding_dim)), max(1, n_units - 1))
        state = emctgl_admm(
            model_views,
            latent_dim,
            lambda_reconstruction=cfg.lambda_reconstruction,
            beta=cfg.beta,
            lambda1=cfg.alpha_l2,
            lambda2=cfg.lambda2,
            sigma=cfg.sigma,
            max_iter=cfg.admm_iterations,
            mu0=cfg.mu0,
            rho=cfg.rho,
            mu_max=cfg.mu_max,
            tol=cfg.convergence_tol,
            tlog_epsilon=cfg.tlog_epsilon,
            graph_topk=cfg.graph_topk,
            random_state=cfg.random_state,
            kmeans_restarts=cfg.kmeans_restarts,
            graph_objective=cfg.graph_objective,
            relative_convergence=cfg.relative_convergence,
            stability_limit=cfg.stability_limit,
        )

        components = build_granular_ball_score(views, balls, state, cfg)
        components["feature_tail"] = feature_tail_score(
            raw_views, top_fraction=cfg.feature_tail_fraction
        )
        components["isolation"] = isolation_forest_score(
            raw_views,
            n_estimators=cfg.isolation_estimators,
            max_samples=cfg.isolation_max_samples,
            max_features=cfg.max_features,
            random_state=cfg.random_state,
        )

        total, percentiles = fuse_final_score(
            components,
            final_fusion=cfg.final_fusion,
            gb_weight=cfg.gb_weight,
        )
        components["total"] = total

        labels: Optional[np.ndarray]
        if cfg.threshold is None and cfg.contamination is None:
            labels = None
        else:
            labels = scores_to_labels(
                total, threshold=cfg.threshold, contamination=cfg.contamination
            )

        runtime = time.perf_counter() - start
        diagnostics = {
            "mode": cfg.mode.lower(),
            "n_samples": int(n_samples),
            "n_model_units": int(n_units),
            "compression_ratio": float(n_units / n_samples),
            "n_views": int(len(views)),
            "n_balls": int(len(balls.radii)),
            "split_criterion": cfg.split_criterion,
            "split_method": cfg.split_method,
            "gb_mode": cfg.gb_mode,
            "final_fusion": cfg.final_fusion,
            "iterations": int(state.get("iterations", 0)),
            "converged": bool(state.get("converged", False)),
            "stop_reason": str(state.get("stop_reason", "unknown")),
            "safeguard_triggered": bool(state.get("safeguard_triggered", False)),
        }
        return TGBADResult(
            scores=total,
            labels=labels,
            components=components,
            percentile_components=percentiles,
            ball_assignments=balls.assignments,
            ball_radii=balls.radii,
            ball_sizes=balls.sizes,
            runtime_seconds=runtime,
            diagnostics=diagnostics,
        )


# ---------------------------------------------------------------------------
# Data loading and evaluation helpers
# ---------------------------------------------------------------------------


def normalize_anomaly_labels(
    labels: np.ndarray, *, positive_label: Optional[float] = None
) -> np.ndarray:
    """Normalize a binary label vector so that 1 denotes anomaly."""

    y = np.asarray(labels).reshape(-1)
    unique, counts = np.unique(y, return_counts=True)
    if unique.size != 2:
        raise ValueError(
            f"anomaly evaluation requires two labels, got {unique.tolist()}"
        )
    if positive_label is not None:
        if positive_label not in unique:
            raise ValueError(f"positive_label {positive_label} is absent from labels")
        anomaly_value = positive_label
    elif set(unique.tolist()) == {0, 1}:
        anomaly_value = 1
    else:
        anomaly_value = unique[int(np.argmin(counts))]
    return (y == anomaly_value).astype(int)


def load_mat_dataset(
    path: Union[str, Path], *, positive_label: Optional[float] = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load a ``trandata`` (or X/Y) anomaly MAT file as (X, y).

    ``y`` is the normalized binary label vector when the file contains
    ground-truth labels (``trandata`` last column, or ``Y``/``y`` alongside
    ``X``) and ``None`` when the file holds unlabeled data.
    """

    path = Path(path)
    y: Optional[np.ndarray] = None
    try:
        data = sio.loadmat(path)
        if "trandata" in data:
            table = np.asarray(data["trandata"])
            if table.ndim != 2 or table.shape[1] < 2:
                raise ValueError(f"invalid trandata in {path}")
            x, y = table[:, :-1], np.asarray(table[:, -1]).reshape(-1)
        elif "X" in data:
            raw_x = data["X"]
            if np.asarray(raw_x).dtype == object:
                x = np.asarray(raw_x).ravel()[0]
            else:
                x = np.asarray(raw_x)
            raw_y = data.get("Y", data.get("y"))
            if raw_y is not None:
                y = np.asarray(raw_y).reshape(-1)
        else:
            raise KeyError(f"{path} contains neither trandata nor X")
    except NotImplementedError:
        with h5py.File(path, "r") as handle:
            if "trandata" in handle:
                table = np.asarray(handle["trandata"])
                if table.shape[0] < table.shape[1]:
                    table = table.T
                x, y = table[:, :-1], np.asarray(table[:, -1]).reshape(-1)
            else:
                x = np.asarray(handle["X"]).T
                raw_y = handle.get("Y", handle.get("y"))
                if raw_y is not None:
                    y = np.asarray(raw_y).reshape(-1)
    x = np.asarray(x, dtype=np.float64)
    if y is None:
        return x, None
    return x, normalize_anomaly_labels(y, positive_label=positive_label)


def evaluate_scores(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    """Report AUC, PR-AUC and F1-Max (labels used offline only)."""

    y = np.asarray(y_true).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    precision, recall, _ = precision_recall_curve(y, s)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    from sklearn.metrics import auc

    return {
        "auc": float(roc_auc_score(y, s)),
        "pr_auc": float(auc(recall, precision)),
        "f1_max": float(np.max(f1)),
    }


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TGBAD: tensor-based granular-ball anomaly detection. "
        "Run without arguments for the built-in Example.mat demo."
    )
    parser.add_argument(
        "data",
        nargs="?",
        default=None,
        help="path to a trandata/X-Y .mat or .csv file (default: Example.mat)",
    )
    parser.add_argument("--out", help="path to save the anomaly scores (.npy)")
    parser.add_argument(
        "--evaluate", action="store_true", help="report AUC/PR-AUC/F1-Max"
    )
    parser.add_argument("--max-balls", type=int, default=256)
    parser.add_argument("--min-ball-size", type=int, default=8)
    parser.add_argument(
        "--split-criterion",
        choices=["sse", "radius", "radius_size"],
        default="radius_size",
    )
    parser.add_argument("--split-method", choices=["kmeans", "pca"], default="kmeans")
    parser.add_argument(
        "--gb-mode",
        choices=[
            "recon_only",
            "ball_struct",
            "intra_only",
            "combined_max",
            "combined_mean",
            "combined_product",
        ],
        default="combined_mean",
    )
    parser.add_argument(
        "--final-fusion",
        choices=["top2", "gb_support", "gb_weighted", "mean3", "max3"],
        default="top2",
    )
    parser.add_argument("--contamination", type=float)
    parser.add_argument("--random-state", type=int, default=0)
    return parser


def find_example_mat() -> Path:
    """Locate Example.mat next to the script, in ./data, or in ../data."""

    here = Path(__file__).resolve().parent
    candidates = [
        Path.cwd() / "Example.mat",
        Path.cwd() / "data" / "Example.mat",
        here / "Example.mat",
        here / "data" / "Example.mat",
        here.parent / "data" / "Example.mat",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Example.mat not found; place it in ./data/Example.mat or pass a "
        "data file explicitly"
    )


def print_anomaly_report(
    path: Path,
    x: np.ndarray,
    y: Optional[np.ndarray],
    result: TGBADResult,
    evaluate_requested: bool,
) -> None:
    """Print a human-readable anomaly detection report."""

    diagnostics = result.diagnostics
    print("=" * 66)
    print("TGBAD: Tensor-based Granular-Ball Anomaly Detection")
    print("=" * 66)
    print(f"Data file : {path}")
    print(f"Data      : n={x.shape[0]} samples, d={x.shape[1]} features")
    print(
        f"Granules  : {diagnostics['n_balls']} granular balls, "
        f"{diagnostics['n_model_units']} tensor units, "
        f"ADMM iterations={diagnostics['iterations']} "
        f"({diagnostics['stop_reason']})"
    )
    print(f"Runtime   : {result.runtime_seconds:.2f}s")
    print("-" * 66)

    scores = result.scores
    percentiles = result.percentile_components
    order = np.argsort(-scores, kind="mergesort")
    top = int(order[0])
    print("Anomaly scores (in [0, 1]; higher = more anomalous):")
    print("  components (average-rank percentiles): gbs / tail / isolation")
    if x.shape[0] <= 50:
        for i in range(x.shape[0]):
            tag = "  <-- anomaly" if i == top else ""
            print(
                f"  sample {i + 1:>4}: gbs={percentiles['gbs'][i]:.3f}  "
                f"tail={percentiles['feature_tail'][i]:.3f}  "
                f"iso={percentiles['isolation'][i]:.3f}  ->  "
                f"score = {scores[i]:.4f}{tag}"
            )
    else:
        print(
            f"  score range: [{float(scores.min()):.4f}, "
            f"{float(scores.max()):.4f}]  (mean {float(scores.mean()):.4f})"
        )
        for i in order[:10]:
            print(f"  sample {i + 1:>4}: score = {scores[i]:.4f}  <-- top-ranked")
    print(
        "Ranking   : most anomalous -> "
        + " > ".join(f"sample {i + 1}" for i in order[: max(1, min(3, x.shape[0]))])
        + ("" if x.shape[0] <= 3 else " > ...")
    )

    if y is not None:
        metrics = evaluate_scores(y, scores)
        print("-" * 66)
        print(
            f"AUC={metrics['auc']:.4f}  PR-AUC={metrics['pr_auc']:.4f}  "
            f"F1-Max={metrics['f1_max']:.4f}   (ground-truth labels used for "
            f"evaluation only)"
        )
    elif evaluate_requested:
        print("No ground-truth labels found in the file; evaluation skipped.")
    print("=" * 66)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_cli_parser().parse_args(argv)
    if args.data is None:
        path = find_example_mat()
        print(f"[demo] no data file given; using {path}\n")
    else:
        path = Path(args.data)

    if path.suffix.lower() == ".csv":
        table = np.loadtxt(path, delimiter=",", dtype=np.float64)
        x = table[:, :-1] if args.evaluate else table
        y = table[:, -1] if args.evaluate else None
    else:
        x, y = load_mat_dataset(path)

    config = TGBADConfig(
        max_balls=args.max_balls,
        min_ball_size=args.min_ball_size,
        split_criterion=args.split_criterion,
        split_method=args.split_method,
        gb_mode=args.gb_mode,
        final_fusion=args.final_fusion,
        contamination=args.contamination,
        random_state=args.random_state,
    )
    result = TGBAD(config).fit_predict(x)

    print_anomaly_report(path, x, y, result, evaluate_requested=args.evaluate)

    if args.out:
        np.save(args.out, result.scores)
        print(f"scores saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
