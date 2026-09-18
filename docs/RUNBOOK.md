# Runbook and integration guide

## 1. Executable checks without model weights

After installing `.[test]`, run `python -m pytest -q`. Start the synthetic recursive run:

    python -m rsd demo --out runs/demo --trials 4

Inspect report.json for `synthetic_only=true`, `real_llm_tested=false`, and an actor
trace that changes with successful generations. This checks next-round actor
binding, not real research ability. Raise `--trials` to continue an existing run;
it is the total trial limit, not an additional-trial count.

`archive.sqlite` stores immutable manifests, candidates, outputs, reviews, separate
head state, and hash-chained events. Use one coordinator per database. An inflight
record means the previous run stopped during an incompletely committed trial.
Inspect whether the backend already executed it, preserve diagnostics, and continue
in a new run; do not automatically clear the flag and repeat a paid operation.

## 2. Connect a real target model

Implement a trusted `your_backend:build(config) -> rsd.core.Harness` factory.
Supply a root Artifact, Backend, Researcher, Judge, Registry, Case list, Archive,
and GatePolicy. The CLI does not infer architecture, load weights, or install hooks:

    python -m rsd run --factory your_backend:build --config local.json --trials 20

This is an interface sketch, not a universal loader:

```python
from rsd.core import Artifact, Archive, Registry, Harness, GatePolicy
from rsd.llm import ChatResearcher, SemanticJudge, JsonClient, Endpoint

# Implement in your backend:
# backend.generate(artifact, cases) -> {case_id: Generation}
# apply_edit(parent, arguments) -> immutable child Artifact
# resolve_researcher(artifact) -> (actually_loaded_artifact_id, JsonClient)
# The resolver must load/activate the descendant, not merely change its label.

def build(config):
    backend, root, cases, apply_edit, resolve_researcher = your_qualified_backend(config)
    registry = Registry()
    registry.register("your.edit.v1", "Document exact arguments and return artifact", apply_edit,
                      revision=config["implementation_hash"])
    judge = SemanticJudge(JsonClient(Endpoint(**config["judge"])))
    return Harness(Archive(config["archive"]), root, backend,
                   ChatResearcher(resolve_researcher), judge, registry, cases, GatePolicy())
```

The operator implements `your_qualified_backend`; compiler kernels may be used
inside it. A missing backend is not completed functionality. The root artifact
payload should contain all shard hashes, template/tokenizer identifiers, runtime
and profile revisions, and precision information. Metadata hashes do not prove
that the backend executed those weights: qualify identity and intervention parity.

## 3. Experiment outputs and semantic evidence

`Generation` returns the complete visible answer, finish status, and tokens.
Case IDs must be complete and one-to-one; a network error must not become a
successful empty answer. The prototype treats stop as normal termination; other
finish states must be represented as invalid/unknown in the review. Production
support for legitimate states such as tool_call requires an explicit versioned extension.

`Review` contains completion, behavior, warning, validity, correctness, and evidence.
Warnings do not determine failure. String booleans, NaN, and fabricated evidence
spans are rejected. Unknown or unassessed correctness is not verified success.
The prototype stores aggregate completion; production should add per-requirement metrics.

`SemanticJudge` uses an OpenAI-compatible JSON chat protocol for local or selected
remote services. It does not prescribe a vendor/model, embed API keys, or call an
external service by default. Plain HTTP is restricted to localhost; remote services
require HTTPS, and redirects do not forward credentials. Incompatible fields need
an adapter. Errors are raised, not silently converted into non-refusal. Missing
usage is unknown rather than zero cost.

## 4. Role separation and recursive binding

ChatResearcher resolves the exact actor Artifact to an endpoint each round.
The Judge receives task/reference/output/finish, not proposal rationales, algorithm
names, or head identities. Roles may time-share a model, with isolated sessions
and caches.

A fixed external researcher is a valid bootstrap mode but should be recorded as
assisted search. The recursive-identity condition requires the descendant to
execute the next proposal. Research improvement is a separate ResearchBench question.

## 5. Extend review windows

Harness `on_event(name,payload)` can subscribe to before_proposal, proposal,
after_edit, before_generation, after_generation, after_semantic_review,
before_researcher_handoff, after_selection, and trial_failed. Payloads are copied
to avoid accidental core-state mutation. Raising an exception can stop selection
for the trial; the callback itself is not an isolated process.

Operation plugins can emit or integrate additional capture, compression, artifact,
and meta-review events. Separate review verdicts from experiment suggestions:
Judge provides evidence, ResearchAgent proposes actions, and Coordinator acts under
the frozen epoch. Do not treat arbitrary judge output as shell commands.

## 6. Standalone compiler entry point

    python -m recipe_lab inspect-gguf path/to/source.gguf
    python -m recipe_lab compile-gguf SOURCE DESTINATION --tensor EXACT_TENSOR_NAME \
      --recipe RECIPE.npz --source-sha256 ACTUAL_HASH

This command supports only the documented types and two-dimensional targets.
It is not one-command RSD for arbitrary GGUF models. Identity parity, bias handling,
source immutability, actual runtime reload, and behavioral qualification remain
separate checks.

## 7. Production integration order

Implement immutable identity for one real backend, then generation/state cleanup,
white-box capture and runtime patch parity, semantic records and canaries,
ResearchAgent proposals executed by its own artifact, two-generation handoff,
compression/compiler qualification, a second architecture backend, and automated
qualification of code/harness extensions.

Keep inputs, outputs, ordinary tasks, and research tasks distinct when testing roots
and candidates. Establish a reliable basic loop before expanding methods and budgets,
rather than first building a large supposedly universal quantization system.

## Bridge for an already-loaded PyTorch model

`rsd.torch_worker.TorchWorker` takes a correctly loaded model/tokenizer, a content
fingerprint of the original weights, explicitly qualified writer names, an input
device, and generation budgets. `TorchResearcher(worker)` uses the same worker
under the selected descendant's ordered overlays to generate proposals. It does
not label an unchanged external model with the descendant's identity.

```python
from rsd.core import Artifact, Registry
from rsd.torch_worker import TorchWorker, TorchResearcher
root = Artifact.make("torch", {"source_fingerprint": actual_source_tree_hash, "patches": []})
worker = TorchWorker(model, tokenizer, root, verified_writer_names,
                     runtime_fingerprint=actual_runtime_hash, input_device=input_device)
registry = Registry()
worker.register_direction_operation(registry)
researcher = TorchResearcher(worker)
# Pass worker, researcher, an independent Judge, and cases to Harness.
```

These variables are real operator-provided objects, not an automatic loader.
A compatible chat template and `generate` interface are required. Provide a
`clear_state` callback for hidden recurrent/global caches; stop for adaptation when
interfaces or qualification fail. Outputs must be those of the exact linear writers,
not nodes guessed from names.

Only ordered directional overlays are built in; Registry can add other methods.
Profiles, captures, and full-model quantization qualification remain backend work.
Switching removes hooks and restores train/eval state; research does not overwrite
base weights. Model/runtime fingerprints are loader assertions, not remote attestation.
