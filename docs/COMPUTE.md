# Compute strategy: reduce repeated work without restricting research freedom

## 1. Separate available methods from default spending

Open exploration does not require full ARA, a complete benchmark battery, and a
full checkpoint rewrite on every trial. High-rank edits, nonlinear interventions,
local post-training, and tool self-improvement belong to the method space.
Using inexpensive evidence to decide when to invoke costly methods belongs to
scheduling. Limited hardware should not become a methodological commitment to rank 1.

The ResearchAgent proposes hypotheses and experiments; a small sampler can sweep
continuous parameters; execution workers batch real runs; the Judge focuses on
semantic ambiguity. An LLM does not need to issue a tool call for each matrix product.

## 2. Three artifact levels: research state, executable descendant, release file

Research state can contain captured activations, arbitrary weight updates, method
branches, and tool code. An executable descendant can be represented by a pristine
source plus ordered runtime overlays, without first writing a full checkpoint.
A release file requires actual bytes at a specified precision and reload qualification.

A descendant acting as researcher must actually execute those weights or overlays.
Writing "you are M2" in a prompt is not a handoff. On a memory-constrained GPU,
researcher inference, target generation, and judging can run sequentially instead
of keeping three model copies resident. Candidate and role switches must clear
relevant KV and recurrent state rather than presenting reused state as new inference.

## 3. Cache four distinct objects

- **Original-anchor outputs/logits**: reuse only for identical artifacts, inputs,
  templates, modes, and runtimes.
- **Captures/local statistics**: bind to the exact parent, token trajectory, and
  nodes; do not reuse across descendants without checking validity.
- **Judge results**: reuse only when the complete task, answer, reference, rubric,
  and judge configuration hashes match.
- **Small patches**: archive directions, low-rank factors, and implementation
  versions; quantized qualification does not transfer between different bytes.

When qualified descendants share a base, retain one weight copy and hot-swap small
interventions during experiments instead of repeatedly downloading or reloading.
Remote white-box execution shifts local capacity requirements; it does not remove
system-wide computation costs.

## 4. Retain the inexpensive kernel without letting it define the frontend

`recipe_lab.linear` implements symmetric small-subspace fitting on cached outputs:

    Y' = Y - (YQ) H Q^T, H=H^T
    A H + H A = C + C^T

This optional surrogate cheaply screens candidates on CPU. It is not equivalent
to full ARA and does not promise a global optimum. A small local loss does not
establish better generation. Teacher hidden outputs must come from actual model
execution, not from an LLM's correctness labels.

The operation catalog may also include RDO, T2 general output operators, T3 general
BA updates, arbitrary local matrices, sparse edits, router/state interventions,
and repair. An unadapted method must report the runtime capabilities it requires.

## 5. Compression is an objective, not a mandatory generational gate

The T1 family, DeltaW=-D diag(a) D^T W, does not contain every low-rank update.
For W=I this update is symmetric; a general rank-1 uv^T need not be.

Allow compact direction recipes, general low-rank/sparse static patches, and
research-time runtime interventions. Mark nonlinear interventions that cannot be
compiled statically as runtime-required; do not claim a file edit implements them.

Compression can follow substantial progress across several generations, avoiding
premature loss of useful high-freedom intermediate states. The compression window
compares teacher and child on target behavior and always compares ordinary
capabilities against the pristine base. Geometric rank and functional precision
are different; refusal-geometry research does not establish a universal
one-dimensional mechanism. [S6]

## 6. Composition across generations and quantization error

    (I-P2)(I-P1) = I-P1-P2+P2P1

Do not simply add successive directional updates when operators do not commute.
Preserve an ordered recipe, or compose, recompress, and retest. Prefer rebuilding
from pristine bytes and accumulated ordered edits rather than requantizing the
previous generation repeatedly. This reduces unrelated rounding accumulation.
After a quantized merge, further research must bind to the new artifact hash.

When needed, raise the precision of selected tensors, recalibrate locally, or
retain an external adapter. Do not force a larger edit merely to cross fixed-scale
quantization bins at the expense of behavior. Check each choice against actual
outputs in the compilation window.

## 7. Account accurately for streaming compilation

For W of shape (m,n) and D of shape (m,k), accumulate S=D^T W, then write row blocks
of W'=W-D diag(a)S. Common layouts require two reads of targeted data and one write,
not a single read; source copying and hashing are additional I/O.
The local working set is approximately O(km+kn+tile), not O(total model parameters).

The independent reference writer supports NPY F32/F16 and two-dimensional targets
in single-file, little-endian GGUF v3: F32/F16/BF16/Q4_0/Q8_0. Q4_0/Q8_0 are
re-encoded using the original scale bytes. K-quants, IQ, FP8, NVFP4, MXFP4,
multi-file atomic transactions, and three-dimensional fused-expert write-back
are not supported. Unsupported targets fail preflight; unknown untouched tensors
may be copied byte-for-byte. Known companion biases must not be silently omitted.

Compiler byte/parity tests are not language-model quality tests. A parseable
quantized container is not runtime qualification. The project does not depend on
private DWM implementation details or claim to reproduce its exact formula.

## 8. Adaptive budgets should consider marginal information, not only score

A production scheduler can compare estimated information value per estimated cost
for recapture, new algorithms, broader validation, hook diagnosis, compression,
or stopping. These estimates are uncertain and require an exploration budget;
always choosing the cheapest-looking option is not sufficient.

Development can use multiple fidelities, but each ranking must compare compatible
data and decoding budgets. Optuna's report/should_prune interface does not support
multi-objective trials; use an external promotion scheduler or separate studies. [S7]

Cache inexpensive CPU surrogate results and send only selected candidates to real
generation. Baseline outputs cannot replace candidate execution. Truncation caused
by a short token limit is neither demonstrated refusal nor demonstrated completion;
rerun within budget or record an unknown result.

## 9. Budget formulas use the actual profile

    Weight payload lower bound = parameter count * bits/8
    Capture cache = samples * captured positions * sum(writer widths) * dtype bytes
    Direction table = sum(writer width * rank * dtype bytes)
    Generation token budget = sum_stage(candidates * samples * max generated length)

Add quantization metadata, mixed precision, KV/recurrent state, allocator overhead,
temporary operations, and judge costs. Measure speed on the actual hardware,
execution graph, batch size, and context rather than inferring minutes from
parameter count. The budget CLI requires explicit parameter count, layer count,
width, and sample count; it has no model-specific defaults.

The reference harness enforces a total trial limit and HTTP timeouts and records
generation tokens. Unified GPU/cash/token hard limits, job cancellation, and
asynchronous accounting remain unimplemented; real backends must supply the
necessary constraints.
