# Validation record · 2026-09-18 · v0.2.0

## Actually executed

- `PYTHONPATH=src pytest -q`: **90 passed in 2.14s**. Raw output: [pytest.txt](pytest.txt).
- `python -m compileall -q src`: passed.
- Editable installation with existing dependencies and `--no-build-isolation`: passed; `rsd --help` and `recipe-lab --help` executed.
- Four-trial recursive fixture, SQLite resume, separate research/deployment heads, immutable epoch and event hash chain: passed. [Fixture summary](fixture.json).
- Tiny PyTorch neural fixture: selected descendant overlay actually changes forward execution, source weights stay unchanged, hooks are removed after success/error, and TorchResearcher calls the same selected worker. This is not a language model.
- Original local fit and streaming compiler demo: maximum absolute compile discrepancy about `9.83e-8` on a constructed 64×96 writer.
- Strict judge parsing: warning+complete allowed; unknown, incorrect, malformed/truncated and empty outputs cannot be counted as successful completion; evidence must match the output.
- Inherited numeric/codec tests cover small synthetic NPY and limited GGUF blocks, not real model deployment.

Environment details: [environment.json](environment.json). CPU-only Torch; no GPU experiment.
A normal isolated `pip install` first failed because this container could not resolve the package index. Installing with already available build dependencies succeeded. This is not a claim that an online clean-machine install was tested.

## Not executed / not established

No real LLM judge or ResearchAgent API calls, no real language-model checkpoint evaluation, no real-model multi-generation self-research, no CUDA throughput benchmarks, no llama.cpp loading of a compiled language model. The fixture scores are deliberately scripted and are not decensor benchmarks or evidence of recursive improvement.

The JSON client and worker binding have local contract tests, not provider-wide compatibility certification. A loader's source fingerprint and HTTP resolver's reported model identity are assertions from that backend, not cryptographic evidence about remote weights.

This repository is a runnable reference harness plus math/codec kernels and a complete architecture specification. Automatic plugin-code qualification, general capture/model profiles, multi-judge calibration, strict process/resource isolation, distributed execution and independent release evaluation remain implementation work. See [status](../docs/IMPLEMENTATION.md).

These are local test results, not a GitHub Actions run.
