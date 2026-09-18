# Implementation status, qualification, and controls

## 1. Implemented does not mean validated on every model

| Component | Current status | Validation scope |
|---|---|---|
| RSD control plane | Runnable | Immutable candidate identities, original anchor, multiple parents, separate heads, handoff, resume, failed trials do not promote |
| SQLite archive | Runnable, single coordinator | Rejects epoch mismatches, chains event hashes, does not silently replay inflight work |
| ResearchAgent | JSON chat client and descendant resolver interface | Argument/binding tests; no real LLM calls |
| Judge | Separate context and strict structure/evidence validation | Warnings are not penalized; unknown/truncated outputs cannot become verified successes; not a universally calibrated judge |
| Event windows | Extensible synchronous callbacks | Additional review failures can block promotion; not a process permission boundary |
| Experiment methods | Registration interface and synthetic fixture | Not a completed integration of all ARA/RDO/repair algorithms |
| PyTorch execution bridge | Already-loaded model/tokenizer and explicit writer profile | Tiny neural fixture verifies actual overlay effects and restoration; no real language-model run |
| Mathematics backend | Independent local T1 fitting, direction compression, single-writer hook | Numerical/algebraic tests, not behavioral effectiveness |
| NPY/GGUF compilation | Independent limited codecs, fixed scales, source protection | Synthetic files; no real-model llama.cpp run |
| Resource controls | Trial limits, HTTP timeouts, generation-token records | No unified GPU/cash/token hard limits or distributed scheduler |
| Strong RSD | Goals and contracts defined | Synthetic identity handoff tested; real descendant self-research untested |

`recipe_lab` is the existing general-purpose kernel; `rsd` is the main control plane.
The inherited schemas belong to recipe_lab and are not a complete production
specification for the new core Review. The Python parser is the current execution contract.

## 2. Implementation sequence

M1: Integrate a real backend, pin hashes/templates/execution graph, and qualify
root-model and identity numerical behavior.

M2: Implement exact-node capture and runtime overlays, using the same descendant
for target generation and researcher generation.

M3: Add development-set semantic and deterministic checks, human calibration,
judge error/unknown accounting, and paired regressions.

M4: Run at least two real RSD generations, recording researcher weights, proposals,
execution, costs, and independent outcomes.

M5: Add an automated qualification workspace for data, algorithm, and tool extensions;
archive failed artifacts without overwriting the active experiment.

M6: Add optional compression, quantized artifact compilation and reload qualification,
and a second architecture profile.

M7: Add adaptive multi-fidelity scheduling, asynchronous review, ResearchBench,
and Judge epoch bridging.

Every stage should allow continued research without mandatory compression. The
deployment backend must not define the boundaries of the research space.

## 3. Required equal-budget controls

A. Fixed researcher with the same method library.
B. Descendant handoff with the same method library.
C. Descendant handoff with an extensible method library.
D. C with a high-fidelity semantic gate, compared against a keyword scorer.
E. C with compression/quantization, compared against the same uncompiled candidate.

Match and report model-execution tokens, judge costs, wall time, and/or GPU time.
Separate recursive gains from simply running more trials or using a stronger
external researcher. Use multiple seeds, task groups, and model families rather
than generalizing from one successful demonstration.

## 4. Minimum evidence for a real strong-RSD claim

1. Record the actual M_t artifact/overlay used to generate the next proposal,
   rather than only changing an identifier.
2. Derive M_(t+1) from that proposal's executed tools and preserve independent
   development and qualification evidence.
3. Demonstrate at least one actual M_(t+1) handoff, with a traceable original anchor
   and Judge epoch.
4. Compare against a fixed researcher at equal budget; report research gains and
   failures, including runs with no improvement.
5. Qualify the compressed/quantized artifact separately, without inheriting the
   floating-point teacher's scores.

This version has not met these criteria on real language models and does not
report them as completed.

## 5. Known engineering limits

The core is a single-process coordinator, not an isolation boundary for hostile
Python plugins. Trusted plugins can execute arbitrary Python. Production code
extensions require separate OS/container workers, no production credentials,
read-only inputs, and explicit output directories. The SQLite hash chain detects
ordinary tampering; it is not unforgeable against an administrator.

The current reviewer uses one judge and one result, without complete calibration,
a panel, per-requirement scoring, or statistical gates. Head selection is a
development heuristic, not research-grade multiple-testing control or automatic
release qualification. Token costs depend on backend reports; missing usage does
not establish zero cost.

Version changes to code, data, rubrics, and model revisions. The default
`.gitignore` excludes run data and weights.
