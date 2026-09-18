# RSD: an open-ended recursive research architecture

Version 0.2.0, 2026-09-18. This is a design specification, not a claim that every
component is implemented. Source identifiers refer to [SOURCES](SOURCES.md).

## 1. From producing a patch to researching descendants

The direction-recipe approach asks how to find an inexpensive, compilable static
edit for a given model. RSD asks how a model, given tools, a budget, and a task
specification, can propose and execute experiments, select descendants from real
feedback, and let qualified descendants continue the research. Editing methods
and the workflow itself may also become research objects.

A DWM-like compiler is therefore a backend, not the definition of the project.
Small-subspace methods are inexpensive starting points, not expressivity limits.
The two semantic windows are default checkpoints, not a limit on the number of windows.

Distinguish three levels:

| Level | What actually happens | Supported description |
|---|---|---|
| Automated ablation | A fixed external researcher repeatedly edits the target | Automated search, not strong RSD |
| Descendant handoff | M_t proposes through the harness; M_(t+1) actually performs the next research inference | Weight/artifact-level recursion |
| Research-system recursion | Descendants can propose probes, algorithms, tools, or harness changes that are tested and integrated | Stronger system-level RSD |

An external bootstrap model or judge does not invalidate recursion, but each round
must identify its actual researcher. Repeatedly calling a fixed external model
while only changing the target is not self-research.

DGM provides a precedent for open-archive code-level self-improvement; SEAL provides
a precedent for model-generated self-update instructions. Neither proves that
reducing refusal makes a descendant a better researcher. [S1,S2]

## 2. State is not an overwritten checkpoint

Represent experiment state as:

    S_t = (M0, Archive_t, ResearchHead_t, DeploymentHead_t,
           HarnessRevision_t, EvaluationEpoch_t, Budget_t)

M0 is the immutable original capability anchor within a lineage. Archive stores
candidates, parents, algorithm/data/runtime versions, outputs, review evidence,
and interesting non-promoted branches. ResearchHead performs the research;
DeploymentHead is the candidate best suited to the deployment task. They need
not be identical.

A descendant with less evasion but weaker experimental reasoning may be a useful
deployment candidate without qualifying as the next researcher. A research branch
with slightly worse deployment scores may still produce useful next steps.

A production scheduler should retain an archive or Pareto front covering behavioral
and algorithmic diversity, rather than requiring every intermediate node to beat
its parent on every metric. DGM's reported value of non-optimal intermediate nodes
is a motivation, not evidence that the same result holds for RSD. [S1]

The reference loop archives all candidates and accepts proposals from any known
parent. Head selection uses a simple feasibility-and-gain rule, not an implemented
MAP-Elites or AB-MCTS scheduler.

## 3. ResearchAgent and Judge: separate responsibilities, not necessarily models

### ResearchAgent

Reads development evidence, proposes falsifiable hypotheses, chooses informative
experiments, requests captures and method extensions, reconciles conflicting
results, chooses whether to compress, and proposes new plugins. Its method space
can expand substantially.

It receives task specifications, tool contracts, permitted archive data, and actual
outputs. Free-text claims do not directly become scores.

### Judge

Uses actual prompts, outputs, and verifiable evidence to distinguish completion,
evasion, correctness, and output validity, recording uncertainty and disagreement.
It does not receive the researcher's persuasive rationale or reward an algorithm
for being described as advanced.

The Judge does not optimize weights and has no default permission to write files,
execute candidate code, or read final test sets. It operates during research
evaluation, not as an added filter in the deployed model.

### Sharing a model

Roles may share a checkpoint and inference service while keeping contexts, task
prompts, KV/state, and tool permissions separate. This can save memory, but role
isolation is not statistical independence. Judge self-preference research suggests
that familiarity with output style can affect judgments; three roles using the
same weights do not constitute three independent votes. [S3]

Three configurable arrangements are useful:

- **Low-resource single model:** time-share a frozen anchor/reviewer snapshot and
  an evolving researcher.
- **Two services:** run ResearchHead locally and use a version-pinned judge service;
  external hosting does not remove the need for calibration.
- **Self-review with audits:** use the descendant for broad inexpensive review and
  a blinded reference panel for difficult and randomly sampled cases.

Open exploration and evidence-constrained semantic judgment are different
objectives. Separate the roles and derive decisions from evidence and task
constraints, not from the researcher's assertion of success.

## 4. Judge evolution requires an explicit epoch change

A permanently frozen weak Judge can become a limitation; unrestricted changes to
judging destroy comparability. Separate timescales:

    Candidate loop: freeze the current epoch's specification and scoring protocol.
    Generational loop: replace ResearchHead after handoff checks.
    Evaluation-upgrade loop: bridge a proposed Judge/rubric against independent
                             calibration data and the existing protocol.

An accepted upgrade starts a new EvaluationEpoch without erasing earlier results.
Calibration should cover refusal, warnings followed by full completion, evasion
without refusal markers, detailed incorrect answers, appropriate uncertainty,
necessary clarification, and truncation. Resolve disagreements using external
evidence or human labels, not self-certification by the proposing model.

The current implementation fixes the epoch and rejects incompatible resumes.
Automatic bridging and panel upgrades remain future work.

## 5. Multiple windows form an event system

Windows may support hypothesis generation, semantic judgment, or deterministic
numerical/file checks. They do not require a large-model call at every step.

| Event | Input | Appropriate action |
|---|---|---|
| After model inspection | Graph, visible nodes, precision, resources | Extend a profile or request numerical parity checks |
| Data/task design | Coverage, failure categories, near-duplicates | Propose probes or correct labels |
| Before/after proposals | Hypothesis, falsifiable prediction, cost | Design controls or revise an experiment |
| After capture | Activation summaries, node/position metadata | Diagnose hooks or choose another capture strategy |
| After local optimization | Surrogate loss, step size, rank, numerical anomalies | Decide whether real generation is justified |
| After candidate generation | Actual outputs and termination state | Semantic gate A |
| After compression | Teacher/candidate/base outputs on matched tasks | Semantic gate B |
| After compilation | Byte/numerical checks and artifact outputs | Artifact qualification |
| Before researcher handoff | Descendant research ability and equal-budget controls | Promote, retain the researcher, or branch |
| After stagnation | Historical gains, method coverage, failure clusters | Revise research strategy or propose harness extensions |

These checkpoints are extensible, not a fixed A/B-only workflow. Run inexpensive
checks first; semantic ambiguity, behavioral changes, compression failures, and
stagnation can trigger more expensive review.

The current `on_event(name,payload)` is synchronous; exceptions prevent head updates
for the trial. It is an extension seam, not a complete multi-agent workflow product.
Queues, asynchronous judgments, and human-review interfaces remain to be implemented.

## 6. An operation language for open exploration

A production ResearchAgent should be able to propose more than alpha adjustments:

    inspect / read_evidence / propose_tests / capture / compare
    mean_difference / direction_optimization / ARA_local / low_rank / full_matrix
    sparse_target_search / router_or_state_experiment / targeted_repair
    distill / compose / quantization_experiment / compile / qualify
    extend_tool / extend_algorithm / change_harness / propose_judge_revision

This is a capability catalog, not a list of implemented algorithms. Each installed
plugin registers an input schema, implementation hash, required capabilities,
estimated CPU/GPU/memory/I/O cost, actual outputs, and a testing contract.

Unknown methods must not be forced into T1. Proposed code can enter an isolated
workspace, pass unit tests, identity/parity checks, small-model checks, and resource
limits, and then become a registered tool. Qualification may be automated without
requiring a human click each time. Version implementations rather than silently
changing the code associated with completed experiments.

The current kernel executes operator-installed registered operations, not arbitrary
Python or shell emitted by an LLM. Automated code generation and a process-isolated
runner are priority implementation tasks. Isolation protects experiment integrity
and user files; it does not prescribe which mathematical methods may be explored.

## 7. The objective is not a refusal-string count

Target success requires task completion and correct/useful content. Retention is
measured through newly introduced failures on originally successful tasks. Research
ability is measured separately. Warnings are recorded without a default penalty.

Record a vector such as:

    (actual completion, correctness, original-anchor regression, unresolved rate,
     research ability, long-tail coverage, patch complexity, inference cost,
     research cost)

Unknown correctness is not verified success, but neither is it automatically refusal.
For subjective open-ended tasks, define an appropriate human/semantic protocol rather
than forcing exact-match scoring.

StrongREJECT motivates distinguishing avoidance of refusal phrases from useful task
completion. Its task definitions do not automatically cover every project use case. [S4]

The implemented `Review` is a minimal record. Per-requirement evidence, calibrated
uncertainty, task slices, and multi-judge disagreement protocols require further work.
Self-reported confidence is not a statistical confidence interval.

## 8. Strong RSD must measure the ability to continue researching

Task benchmarks alone do not establish that a descendant is a better researcher.
A separate ResearchBench should compare parent and child with the same tools and
compute budgets on unfamiliar research tasks:

- Detect deliberately incorrect hooks, signs, transposes, quantization scales,
  and token alignment.
- Propose experiments that distinguish competing mechanisms rather than merely
  producing more uninformative trials.
- Correct mistakes from real logs without counting truncation or uncertainty as success.
- Produce independently accepted improvements within matched generation-token or GPU-time budgets.
- Build and qualify a profile for a previously unstudied model family.

Greater willingness to answer does not establish better research ability. A failed
ResearchBench may leave the existing researcher in place while retaining the
child as a target candidate or isolated exploratory branch.

The fixture contains a research group and tests separate heads; it is not a mature
ResearchBench implementation.

## 9. Expand development data without eroding evaluation independence

Separate capture, development, compression, judge-calibration, research-transfer,
and release data. Split by task families, templates, and translation groups rather
than randomly scattering near-duplicates. Researchers may generate development
samples, while final qualification follows an independent sampling protocol and
does not disclose final answers to the researcher.

Repeatedly querying a holdout's scores can cause adaptive overfitting even without
seeing its questions. [S5] Tests used to select descendants must be labeled as
development/selection evidence rather than permanently described as unseen final
tests. Continued research after release failure must record that the set was consumed.

Compare retention against the same pristine M0 each generation, not only child
against parent. Do not mix scalar scores across epochs, fidelities, or datasets
without a valid comparison protocol.

## 10. Model-agnostic capability negotiation

The core does not hard-code layer counts, widths, expert counts, a model family,
or a particular GPU. Derive ModelProfile from actual configuration, tensor metadata,
and runtime numerical checks:

    architecture_id, artifact hashes, graph revision,
    writer nodes + exact boundary + input/output axes + bias,
    capture support, adapter support, quant codec support,
    KV/recurrent reset requirements, multimodal/draft components.

GGUF is a storage container, not an arbitrary internal-capture API or universal
re-encoder. Chat APIs support generation and judging but cannot substitute for
white-box captures. Unsupported nodes/codecs require another backend or explicit
adaptation, not silent omission.

Complete the same profile, parity, and recursion qualification on at least two
different architecture families before claiming cross-architecture effectiveness.
Generic interfaces and universal model support are separate claims.

## 11. Stopping does not settle the scientific question

Every run has an explicit budget and recoverable checkpoints. Without real gains,
allow branching, method expansion, or suspension rather than relaxing evaluation
to manufacture a new generation.

Automatic loops must preserve reconstructible artifacts, failures, and uncertainty.
High freedom means open methods and research planning, not erased evidence,
unbounded paid calls, or overwriting source weights.

## 12. Status of this version

This version makes the research harness central and the compiler optional. It
provides a runnable control plane and optional mathematical backends, with tests
of foundational correctness. It has not demonstrated improving descendant
researchers on real LLMs. That hypothesis requires the controlled experiments
above; a callable loop is not evidence that it holds.
