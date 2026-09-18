# Recursive Self-Decensoring (RSD)

**Open-ended exploration + high-fidelity semantic evaluation + descendant researcher handoff.**

RSD is a model-agnostic research harness for recursively exploring, evaluating,
and compiling behavior edits. The edited descendant can become the next
ResearchAgent. The Judge is a separate, versioned role, not a hidden policy layer
in the deployed model. Compact directional patches are an optional compiler
backend, not a restriction on the discovery search space.

## Scope

The project does not target a particular model, layer count, or parameter scale.
It uses the local mathematics and streaming compiler kernels from Direction Recipe
Lab as optional backends within a **recursive research workspace**.

- **ResearchAgent**: formulates hypotheses, selects parents, requests data, probes,
  and algorithms, analyzes observed failures, and proposes method extensions.
- **Judge**: independently reviews actual outputs, distinguishing refusal, evasion,
  necessary clarification, warnings followed by completion, errors, and truncation.
- **Harness**: executes configured operations, preserves lineage and evidence,
  manages role handoff, and never treats unknown outcomes as successes.
- **Compiler**: optionally compresses validated results into static patches;
  research branches can survive even when compression is unsuccessful.

**Research and judging are separate responsibilities, not necessarily separate
models. Two calls to the same weights are not independent evidence.**

## Runnable scope (0.2.0)

Implemented: model-agnostic protocols, an immutable SQLite archive and event chain,
descendant identity handoff, separate research and deployment heads, development-set
semantic records, registered experiment operations and event callbacks, configurable
JSON chat ResearchAgent/Judge clients, a CPU synthetic recursion demo, a shared
descendant execution/research bridge for an already-loaded PyTorch model, local
direction fitting, a single-writer hook, and limited NPY/GGUF compilers.

**Not claimed as implemented or validated:** one-command automatic RSD for arbitrary
models, real LLM self-improvement, process-level isolation for generated plugins,
distributed scheduling, complete quantization coverage, or refusal/capability
benchmarks on real language models.

The default demo uses an explicitly labeled deterministic fixture to test identity
handoff and the state machine. Its toy scores are **not evidence of recursive
improvement in a language model**. See [implementation status](docs/IMPLEMENTATION.md).

## Quickstart

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install -e '.[test]'
python -m pytest -q
python -m rsd demo --out runs/demo --trials 4
# Completed trials are not repeated; raise the total budget to continue:
python -m rsd demo --out runs/demo --trials 6
```

Synthetic mathematics/compiler checks:

```bash
python -m recipe_lab demo --out runs/compiler-demo
python -m recipe_lab inspect-gguf path/to/model.gguf
```

Entry point for an actual backend:

```bash
python -m rsd run --factory your_backend:build --config local.json --trials 20
```

`your_backend:build` is a Python factory installed by the operator and returning
`rsd.core.Harness`. It is **not a bundled universal model loader**. See the
[runbook](docs/RUNBOOK.md) for interfaces, arguments, and integration steps.

## Reading order

1. [RSD design](docs/DESIGN.md): open exploration, recursion, role separation,
   review windows, archives, and overfitting controls.
2. [Compute and compilation](docs/COMPUTE.md): caching, virtual descendants,
   rebasing, quantization, and I/O limits.
3. [Runbook and extension points](docs/RUNBOOK.md): factories, reviews, plugins,
   and real-backend qualification.
4. [Implementation and experiments](docs/IMPLEMENTATION.md): implemented,
   integration-pending, and unvalidated components.
5. [Primary sources and evidence limits](docs/SOURCES.md): related research is
   not evidence that this project's hypotheses have been demonstrated.
6. [Recorded validation](reports/VALIDATION.md).

## Important boundaries

The experiment specification defines evaluation criteria. The Judge does not
penalize warnings by default or equate assertive answers with correctness.
Research objectives may change through versioning, but the measurement protocol
must not change silently after results are observed within an experiment.
Unknown results, API failures, truncation, and actual refusal are recorded
separately. The original capability baseline does not drift with each generation.

Code and documentation are MIT-licensed. Models, data, and external tools retain
their respective licenses. Private conversations, API keys, model weights, and
unauthorized data do not belong in this repository.
