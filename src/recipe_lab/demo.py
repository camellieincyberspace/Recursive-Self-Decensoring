"""Synthetic algebra demo. No language model and no decensor result is produced."""
from pathlib import Path
import json
import numpy as np
from .linear import DirectionPatch, fit_symmetric_patch, local_metrics
from .storage import compile_npy, save_patch, sha256_file
from .budget import memory_estimate


def run_demo(out_dir: str | Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(42)
    m, n, p = 64, 96, 8
    q, _ = np.linalg.qr(rng.normal(size=(m, p)))
    teacher = DirectionPatch(q[:, :2], np.array([0.65, 0.25]))
    good = rng.normal(size=(256, m))
    # Synthetic retain samples avoid the two target directions, by construction.
    good -= (good @ q[:, :2]) @ q[:, :2].T
    bad = rng.normal(size=(256, m)) + 3 * q[:, 0]
    target = teacher.outputs(bad)
    fitted = fit_symmetric_patch(good, bad, target, q, retain_weight=5, ridge=1e-6)
    patch = fitted.compress(2)
    original = (rng.normal(size=(m, n)) * 0.08).astype(np.float32)
    source, destination = out_dir / "toy-writer.npy", out_dir / "toy-writer-patched.npy"
    np.save(source, original)
    save_patch(out_dir / "recipe.npz", patch)
    result = compile_npy(source, destination, patch, sha256_file(source), tile_rows=7)
    x = rng.normal(size=(32, n))
    compiled_y = x @ np.load(destination).T
    runtime_y = patch.outputs(x @ original.T)
    report = {"synthetic_only": True, "language_model_tested": False,
              "local_metrics": local_metrics(patch, good, bad, target),
              "max_compile_error": float(np.max(np.abs(compiled_y - runtime_y))),
              "compile_report": result, "illustrative_memory": memory_estimate(parameters=m*n, bits=32, layers=1, width=m, samples=512)}
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
