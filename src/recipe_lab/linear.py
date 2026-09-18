"""Cached-output, small-space fitting. All objectives here are local surrogates.

Rows are samples. A DirectionPatch is y' = y - (y D) diag(a) D.T.
It is exactly compilable into W' = W - D diag(a) D.T W for y = x W.T.
No claim about behavioral efficacy follows from this algebra.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def finite_matrix(x: np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2 or min(a.shape) < 1 or not np.isfinite(a).all():
        raise ValueError(f"{name}: expected a nonempty finite matrix")
    return a


@dataclass(frozen=True)
class DirectionPatch:
    directions: np.ndarray
    strengths: np.ndarray

    def __post_init__(self) -> None:
        d = np.array(self.directions, dtype=np.float64, copy=True)
        a = np.array(self.strengths, dtype=np.float64, copy=True)
        if d.ndim != 2 or d.shape[0] < 1 or a.shape != (d.shape[1],):
            raise ValueError("directions must be [output_dim,rank], strengths [rank]")
        if not np.isfinite(d).all() or not np.isfinite(a).all():
            raise ValueError("non-finite patch")
        if d.shape[1] > d.shape[0] or not np.allclose(
            d.T @ d, np.eye(d.shape[1]), atol=2e-5, rtol=2e-5
        ):
            raise ValueError("directions must be orthonormal; do not silently QR a recipe")
        d.flags.writeable = a.flags.writeable = False
        object.__setattr__(self, "directions", d)
        object.__setattr__(self, "strengths", a)

    @property
    def rank(self) -> int:
        return self.directions.shape[1]

    @property
    def output_dim(self) -> int:
        return self.directions.shape[0]

    @property
    def is_identity(self) -> bool:
        return self.rank == 0 or not np.any(self.strengths)

    def outputs(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y)
        if y.shape[-1] != self.output_dim or not np.isfinite(y).all():
            raise ValueError("output shape/non-finite mismatch")
        return y - ((y @ self.directions) * self.strengths) @ self.directions.T

    def weights(self, w: np.ndarray) -> np.ndarray:
        w = finite_matrix(w, "weight")
        if w.shape[0] != self.output_dim:
            raise ValueError("weight output dimension mismatch")
        return w - self.directions @ (
            self.strengths[:, None] * (self.directions.T @ w)
        )

    def compress(self, rank: int) -> "DirectionPatch":
        if isinstance(rank, bool) or rank < 0 or rank > self.rank:
            raise ValueError("invalid retained rank")
        order = np.argsort(-np.abs(self.strengths), kind="stable")[:rank]
        return DirectionPatch(self.directions[:, order], self.strengths[order])


def fit_symmetric_patch(
    good_outputs: np.ndarray,
    bad_outputs: np.ndarray,
    target_outputs: np.ndarray,
    bank: np.ndarray,
    retain_weight: float = 1.0,
    ridge: float = 1e-4,
) -> DirectionPatch:
    """Solve a cached-output surrogate in an orthonormal bank.

    Minimize ||Z_b H-E||_F^2/N_b + retain_weight*||Z_g H||_F^2/N_g
             + ridge*||H||_F^2, subject to H=H.T.
    E=(bad-target)@bank. The component orthogonal to bank is constant.
    Solve A H + H A = C+C.T by diagonalizing the small SPD matrix A.
    No positivity, strength bounds, or global behavioral guarantees are implied.
    The targets must be supplied by a separately reviewed, compatible procedure.
    """
    g, b, t, q = [finite_matrix(x, n) for x, n in [
        (good_outputs, "good"), (bad_outputs, "bad"),
        (target_outputs, "target"), (bank, "bank")
    ]]
    if b.shape != t.shape or g.shape[1] != b.shape[1] or q.shape[0] != b.shape[1]:
        raise ValueError("incompatible cached-output dimensions")
    if not np.allclose(q.T @ q, np.eye(q.shape[1]), atol=2e-5, rtol=2e-5):
        raise ValueError("bank must be orthonormal")
    if not np.isfinite(retain_weight) or retain_weight < 0 or not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("retain_weight >= 0 and ridge > 0 are required")
    zg, zb, e = g @ q, b @ q, (b - t) @ q
    A = zb.T @ zb / len(b) + retain_weight * (zg.T @ zg) / len(g) + ridge * np.eye(q.shape[1])
    C = zb.T @ e / len(b)
    av, au = np.linalg.eigh(A)
    if av.min() <= 0:
        raise ValueError("ill-conditioned fit; increase ridge")
    h = au @ ((au.T @ (C + C.T) @ au) / (av[:, None] + av[None, :])) @ au.T
    h = (h + h.T) / 2
    strengths, vectors = np.linalg.eigh(h)
    order = np.argsort(-np.abs(strengths), kind="stable")
    return DirectionPatch(q @ vectors[:, order], strengths[order])


def local_metrics(patch: DirectionPatch, good: np.ndarray, bad: np.ndarray, target: np.ndarray) -> dict:
    return {
        "retain_mse": float(np.mean((patch.outputs(good) - good) ** 2)),
        "target_mse": float(np.mean((patch.outputs(bad) - target) ** 2)),
        "rank": patch.rank,
        "max_abs_strength": float(np.max(np.abs(patch.strengths), initial=0)),
        "scope": "local_cached_outputs_only",
    }
