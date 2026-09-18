"""Bounded-memory reference writer; never edits a source in place."""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import os
import shutil
import tempfile
import numpy as np
from .linear import DirectionPatch


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for data in iter(lambda: f.read(8 << 20), b""):
            h.update(data)
    return h.hexdigest()


def check_source(source: Path, destination: Path, expected_sha256: str) -> None:
    if not source.is_file() or source.resolve() == destination.resolve():
        raise ValueError("source must exist and destination must be different")
    if destination.exists():
        raise FileExistsError(destination)
    if len(expected_sha256) != 64 or sha256_file(source) != expected_sha256:
        raise ValueError("source SHA-256 mismatch")
    if not destination.parent.is_dir():
        raise ValueError("destination parent must already exist")


@contextmanager
def atomic_copy(source: Path, destination: Path):
    """Copy to a sibling temp; publish with an exclusive hard link, never replace.

    Source and destination must be on a local filesystem supporting hard links.
    The final link refers to the copied TEMP, never to the source.
    """
    fd, name = tempfile.mkstemp(prefix=".recipe-", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    temp = Path(name)
    try:
        shutil.copyfile(source, temp)
        yield temp
        with open(temp, "rb") as f:
            os.fsync(f.fileno())
        os.link(temp, destination)  # raises if destination appeared concurrently
        try:
            dfd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        except OSError:
            pass
    finally:
        temp.unlink(missing_ok=True)


def compile_npy(source: str | Path, destination: str | Path, patch: DirectionPatch,
                expected_sha256: str, tile_rows: int = 64) -> dict:
    source, destination = Path(source), Path(destination)
    if tile_rows <= 0:
        raise ValueError("tile_rows must be positive")
    check_source(source, destination, expected_sha256)
    w = np.load(source, mmap_mode="r", allow_pickle=False)
    if w.ndim != 2 or w.shape[0] != patch.output_dim or w.dtype not in (np.dtype("float32"), np.dtype("float16")):
        raise ValueError("NPY writer accepts a 2D F32/F16 writer matrix")
    if not w.flags.c_contiguous:
        raise ValueError("only C-contiguous matrices are supported")
    s = np.zeros((patch.rank, w.shape[1]), dtype=np.float64)
    if not patch.is_identity:
        for start in range(0, len(w), tile_rows):
            stop = min(start + tile_rows, len(w))
            rows = np.asarray(w[start:stop], dtype=np.float64)
            if not np.isfinite(rows).all():
                raise ValueError("source contains non-finite weights")
            s += patch.directions[start:stop].T @ rows
    with atomic_copy(source, destination) as temp:
        if not patch.is_identity:
            out = np.load(temp, mmap_mode="r+", allow_pickle=False)
            for start in range(0, len(w), tile_rows):
                stop = min(start + tile_rows, len(w))
                value = np.asarray(w[start:stop], dtype=np.float64) - patch.directions[start:stop] @ (patch.strengths[:, None] * s)
                with np.errstate(over="ignore"):
                    cast = value.astype(w.dtype)
                if not np.isfinite(cast).all():
                    raise ValueError("output overflow")
                out[start:stop] = cast
            out.flush()
            del out
        if sha256_file(source) != expected_sha256:
            raise ValueError("source changed during compilation")
    return {"format": "npy", "source_sha256": expected_sha256,
            "output_sha256": sha256_file(destination), "rank": patch.rank,
            "target_shape": list(w.shape), "target_read_passes": 0 if patch.is_identity else 2,
            "target_final_write_passes": 0 if patch.is_identity else 1}


def load_patch(path: str | Path) -> DirectionPatch:
    with np.load(path, allow_pickle=False) as f:
        if set(f.files) != {"directions", "strengths"}:
            raise ValueError("NPZ requires exactly directions and strengths")
        return DirectionPatch(f["directions"], f["strengths"])


def save_patch(path: str | Path, patch: DirectionPatch) -> None:
    path = Path(path)
    with open(path, "xb") as f:
        np.savez(f, directions=patch.directions.astype(np.float32),
                 strengths=patch.strengths.astype(np.float32))
