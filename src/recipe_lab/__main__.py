from __future__ import annotations
import argparse
import json
from pathlib import Path
from .budget import memory_estimate
from .demo import run_demo
from .storage import load_patch, compile_npy
from .gguf import compile_gguf, inspect_gguf


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline reference kernels; NOT a turnkey model editing pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("demo"); d.add_argument("--out", required=True)
    b = sub.add_parser("budget")
    for field in ("parameters", "bits", "layers", "width", "samples"):
        b.add_argument("--" + field, type=int, required=True)
    i = sub.add_parser("inspect-gguf"); i.add_argument("source")
    for name in ["compile-npy", "compile-gguf"]:
        p = sub.add_parser(name)
        p.add_argument("source"); p.add_argument("destination")
        p.add_argument("--recipe", required=True)
        p.add_argument("--source-sha256", required=True)
        p.add_argument("--tile-rows", type=int, default=64)
        if name == "compile-gguf":
            p.add_argument("--tensor", required=True)
            p.add_argument("--max-clip-fraction", type=float, default=0.01)
    args = parser.parse_args()
    try:
        if args.command == "demo":
            result = run_demo(args.out)
        elif args.command == "budget":
            result = memory_estimate(args.parameters, args.bits, args.layers, args.width, args.samples)
        elif args.command == "inspect-gguf":
            info = inspect_gguf(args.source)
            result = {"data_start": info.data_start,
                      "tensors": {n: {"shape": list(t.shape), "qtype": t.qtype, "offset": t.offset, "span": t.span}
                                  for n, t in info.tensors.items()}}
        else:
            patch = load_patch(args.recipe)
            if args.command == "compile-npy":
                result = compile_npy(args.source, args.destination, patch, args.source_sha256, args.tile_rows)
            else:
                result = compile_gguf(args.source, args.destination, {args.tensor: patch}, args.source_sha256,
                                      args.tile_rows, args.max_clip_fraction)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
