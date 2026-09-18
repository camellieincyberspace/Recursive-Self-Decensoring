import argparse
import importlib
import json
from pathlib import Path
from .demo import run_demo
from .core import Harness


def main():
    p = argparse.ArgumentParser(description="RSD reference harness; demo does not run an LLM")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo")
    d.add_argument("--out", required=True)
    d.add_argument("--trials", type=int, default=4)
    r = sub.add_parser("run")
    r.add_argument("--factory", required=True, help="trusted installed module:function returning Harness")
    r.add_argument("--config", required=True)
    r.add_argument("--trials", type=int, default=4)
    args = p.parse_args()
    if args.cmd == "demo":
        result = run_demo(args.out, args.trials)
    else:
        # Configuration is supplied by the operator, never by a candidate output.
        module, name = args.factory.split(":", 1)
        factory = getattr(importlib.import_module(module), name)
        h = factory(json.loads(Path(args.config).read_text(encoding="utf-8")))
        if not isinstance(h, Harness):
            raise TypeError("factory must return Harness")
        try:
            result = h.run(args.trials)
        finally:
            h.store.close()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
