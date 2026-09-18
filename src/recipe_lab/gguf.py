"""Narrow GGUF v3 reference implementation, NOT a universal GGUF editor.

Supported target tensors: 2D little-endian F32/F16/BF16/Q4_0/Q8_0.
Q4_0 and Q8_0 use frozen original block scales. Unknown TARGET types, sharded
models, rank>2 targets, known companion biases and malformed headers fail before a copy is created.
Untouched tensors and metadata are copied byte-for-byte, regardless of type.
No real-model llama.cpp load qualification has been performed for this prototype.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import math
import struct
import numpy as np
from .linear import DirectionPatch
from .storage import atomic_copy, check_source, sha256_file

QTYPES = {0: (1, 4, "F32"), 1: (1, 2, "F16"), 2: (32, 18, "Q4_0"),
          8: (32, 34, "Q8_0"), 30: (1, 2, "BF16")}
META_FORMAT = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "B", 10: "Q", 11: "q", 12: "d"}
MAX_HEADER = 128 << 20
MAX_ITEMS = 1_000_000


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: tuple[int, ...]  # numpy order; GGUF stores dimensions reversed
    qtype: int
    offset: int            # absolute file offset
    span: int              # until next tensor/file end, including optional padding


@dataclass(frozen=True)
class GGUFInfo:
    tensors: dict[str, TensorInfo]
    data_start: int
    metadata: dict


def inspect_gguf(path: str | Path) -> GGUFInfo:
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as f:
        def read(n: int) -> bytes:
            if n < 0 or f.tell() + n > min(size, MAX_HEADER):
                raise ValueError("invalid or oversized GGUF header")
            data = f.read(n)
            if len(data) != n:
                raise ValueError("truncated header")
            return data
        def number(fmt: str):
            return struct.unpack("<" + fmt, read(struct.calcsize("<" + fmt)))[0]
        def text():
            n = number("Q")
            if n > MAX_HEADER:
                raise ValueError("oversized string")
            return read(n).decode("utf-8")
        def value(t: int, depth: int = 0):
            if depth > 4:
                raise ValueError("metadata nesting too deep for reference reader")
            if t in META_FORMAT:
                result = number(META_FORMAT[t])
                if t == 7 and result not in (0, 1):
                    raise ValueError("invalid boolean")
                return result
            if t == 8:
                return text()
            if t == 9:
                sub, count = number("I"), number("Q")
                if count > MAX_ITEMS:
                    raise ValueError("oversized metadata array")
                # Arrays are skipped without building a giant Python list.
                if sub in META_FORMAT and sub != 7:
                    read(struct.calcsize("<" + META_FORMAT[sub]) * count)
                else:
                    for _ in range(count):
                        value(sub, depth + 1)
                return None
            raise ValueError(f"unknown metadata type {t}")
        if read(4) != b"GGUF" or number("I") != 3:
            raise ValueError("only little-endian GGUF v3 is supported")
        n_tensors, n_meta = number("Q"), number("Q")
        if not 0 < n_tensors <= MAX_ITEMS or n_meta > MAX_ITEMS:
            raise ValueError("invalid item count")
        meta = {}
        for _ in range(n_meta):
            key, t = text(), number("I")
            if len(key) > 65535 or key in meta:
                raise ValueError("invalid/duplicate metadata key")
            meta[key] = value(t)
        alignment = meta.get("general.alignment", 32)
        if not isinstance(alignment, int) or alignment < 1 or alignment > 4096 or alignment & (alignment - 1):
            raise ValueError("unsupported alignment")
        raw = []
        names = set()
        for _ in range(n_tensors):
            name, nd = text(), number("I")
            if name in names or not 1 <= nd <= 4:
                raise ValueError("duplicate tensor or unsupported rank")
            names.add(name)
            dims = tuple(number("Q") for _ in range(nd))
            if any(d < 1 or d > 2**32 for d in dims):
                raise ValueError("invalid tensor dimensions")
            qtype, rel = number("I"), number("Q")
            if rel % alignment:
                raise ValueError("misaligned tensor")
            raw.append((name, dims[::-1], qtype, rel))
        start = (f.tell() + alignment - 1) // alignment * alignment
        ordered = sorted(raw, key=lambda r: r[3])
        tensors = {}
        for i, (name, shape, qtype, rel) in enumerate(ordered):
            end = start + ordered[i + 1][3] if i + 1 < len(ordered) else size
            offset = start + rel
            if not start <= offset < end <= size:
                raise ValueError("tensor offsets overlap/out of bounds")
            tensors[name] = TensorInfo(name, shape, qtype, offset, end - offset)
        return GGUFInfo(tensors, start, meta)


def row_bytes(qtype: int, columns: int) -> int:
    if qtype not in QTYPES:
        raise ValueError(f"unsupported target qtype {qtype}; no silent fallback")
    block, byte_count, _ = QTYPES[qtype]
    if columns % block:
        raise ValueError("row is not block-aligned")
    return columns // block * byte_count


def decode_rows(raw: bytes, qtype: int, columns: int) -> np.ndarray:
    stride = row_bytes(qtype, columns)
    if not raw or len(raw) % stride:
        raise ValueError("incorrect row byte count")
    rows = len(raw) // stride
    if qtype in (0, 1):
        return np.frombuffer(raw, dtype="<f4" if qtype == 0 else "<f2").astype(np.float32).reshape(rows, columns)
    if qtype == 30:
        u = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
        return u.view(np.float32).reshape(rows, columns)
    _, size, _ = QTYPES[qtype]
    b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, size)
    d = b[:, :2].copy().view("<f2").astype(np.float32)
    if qtype == 8:
        v = b[:, 2:].copy().view(np.int8).astype(np.float32)
    else:
        codes = np.concatenate((b[:, 2:] & 15, b[:, 2:] >> 4), axis=1)
        v = codes.astype(np.float32) - 8
    return (d * v).reshape(rows, columns)


def encode_rows(values: np.ndarray, original: bytes, qtype: int) -> tuple[bytes, int, int]:
    """Return encoded bytes, clipped element count, unrepresentable zero-scale count.
    The quantized modes preserve the exact original two-byte scale representation.
    Tie behavior is explicit and need not reproduce a fresh native calibration.
    """
    x = np.asarray(values, dtype=np.float32)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("finite 2D values required")
    if qtype in (0, 1):
        with np.errstate(over="ignore"):
            y = x.astype("<f4" if qtype == 0 else "<f2")
        if not np.isfinite(y).all():
            raise ValueError("float output overflow")
        return y.tobytes(), 0, 0
    if qtype == 30:
        u = x.copy().view(np.uint32)
        rounded = u + np.uint32(0x7fff) + ((u >> 16) & 1)
        encoded = (rounded >> 16).astype("<u2").tobytes()
        if not np.isfinite(decode_rows(encoded, 30, x.shape[1])).all():
            raise ValueError("BF16 output overflow")
        return encoded, 0, 0
    _, size, _ = QTYPES[qtype]
    b = np.frombuffer(original, dtype=np.uint8).reshape(-1, size).copy()
    d = b[:, :2].copy().view("<f2").astype(np.float32)
    if not np.isfinite(d).all():
        raise ValueError("non-finite quantization scale")
    z = x.reshape(-1, 32)
    zero_bad = int(np.count_nonzero((d == 0) & (z != 0)))
    q = np.divide(z, d, out=np.zeros_like(z), where=d != 0)
    if qtype == 8:
        code = np.sign(q) * np.floor(np.abs(q) + 0.5)
        clipped = int(np.count_nonzero((code < -127) | (code > 127)))
        payload = np.clip(code, -127, 127).astype(np.int8).view(np.uint8)
    else:
        code = np.floor(q + 8.5)
        clipped = int(np.count_nonzero((code < 0) | (code > 15)))
        code = np.clip(code, 0, 15).astype(np.uint8)
        payload = code[:, :16] | (code[:, 16:] << 4)
    nonzero = d[:, 0] != 0
    b[nonzero, 2:] = payload[nonzero]
    return b.tobytes(), clipped, zero_bad


def compile_gguf(source: str | Path, destination: str | Path,
                 patches: dict[str, DirectionPatch], expected_sha256: str,
                 tile_rows: int = 64, max_clip_fraction: float = 0.01) -> dict:
    source, destination = Path(source), Path(destination)
    if not patches or tile_rows <= 0 or not 0 <= max_clip_fraction <= 1:
        raise ValueError("invalid patch set/tile size/clipping budget")
    check_source(source, destination, expected_sha256)
    info = inspect_gguf(source)
    if info.metadata.get("split.count", 1) != 1:
        raise ValueError("sharded GGUF requires a transactional multi-file coordinator; not implemented")
    for name, p in patches.items():
        if name not in info.tensors:
            raise ValueError(f"missing target {name}")
        t = info.tensors[name]
        # The output hook transforms Wx+b, whereas this reference backend only
        # edits W. Reject a conventional companion bias before any write.
        # Nonstandard naming still requires an independently verified profile.
        companion = name[:-len(".weight")] + ".bias" if name.endswith(".weight") else None
        if not p.is_identity and companion in info.tensors:
            raise ValueError(f"{name}: companion bias requires a paired bias compiler; not implemented")
        if len(t.shape) != 2 or t.shape[0] != p.output_dim:
            raise ValueError(f"{name}: only explicit 2D writer matrices supported")
        rb = row_bytes(t.qtype, t.shape[1])
        if rb * t.shape[0] > t.span:
            raise ValueError("target extends beyond its byte span")
    report = []
    with atomic_copy(source, destination) as temp:
        with source.open("rb") as src, temp.open("r+b") as out:
            for name, p in patches.items():
                t = info.tensors[name]
                m, n = t.shape
                rb = row_bytes(t.qtype, n)
                rec = {"tensor": name, "qtype": QTYPES[t.qtype][2], "shape": list(t.shape),
                       "rank": p.rank, "clip_fraction": 0.0, "write_passes": 0}
                if p.is_identity:
                    report.append(rec)
                    continue
                s = np.zeros((p.rank, n), dtype=np.float64)
                for start in range(0, m, tile_rows):
                    stop = min(m, start + tile_rows)
                    src.seek(t.offset + start * rb)
                    w = decode_rows(src.read((stop - start) * rb), t.qtype, n)
                    if not np.isfinite(w).all():
                        raise ValueError("source contains non-finite values")
                    s += p.directions[start:stop].T @ w
                clip_count, changed = 0, 0
                squared_error, squared_delta, achieved_delta = 0.0, 0.0, 0.0
                for start in range(0, m, tile_rows):
                    stop = min(m, start + tile_rows)
                    src.seek(t.offset + start * rb)
                    raw = src.read((stop - start) * rb)
                    w = decode_rows(raw, t.qtype, n)
                    desired = w - p.directions[start:stop] @ (p.strengths[:, None] * s)
                    encoded, clipped, zero_bad = encode_rows(desired, raw, t.qtype)
                    if zero_bad:
                        raise ValueError("frozen zero scales cannot represent the proposed update")
                    actual = decode_rows(encoded, t.qtype, n)
                    clip_count += clipped
                    changed += sum(a != b for a, b in zip(raw, encoded))
                    squared_error += float(np.sum((actual - desired) ** 2))
                    squared_delta += float(np.sum((desired - w) ** 2))
                    achieved_delta += float(np.sum((actual - w) ** 2))
                    out.seek(t.offset + start * rb)
                    out.write(encoded)
                fraction = clip_count / (m * n)
                if fraction > max_clip_fraction:
                    raise ValueError(f"clipping {fraction:.5f} exceeds budget {max_clip_fraction}")
                rec.update(clip_fraction=fraction, changed_bytes=changed,
                           relative_update_encoding_error=math.sqrt(squared_error / max(squared_delta, 1e-30)),
                           realized_update_norm_ratio=math.sqrt(achieved_delta / max(squared_delta, 1e-30)),
                           read_passes=2, write_passes=1)
                report.append(rec)
        if sha256_file(source) != expected_sha256:
            raise ValueError("source changed during compilation")
    return {"format": "gguf-v3-reference", "source_sha256": expected_sha256,
            "output_sha256": sha256_file(destination), "tensors": report,
            "qualified_for_model_runtime": False}
