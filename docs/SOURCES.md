# Primary sources and evidence limits

Reviewed 2026-09-18. These sources motivate components; they do not demonstrate
that this project has implemented strong RSD.

| ID | Primary source | Used ideas and limits |
|---|---|---|
| S1 | [Darwin Godel Machine](https://arxiv.org/abs/2505.22954), [authors' introduction](https://sakana.ai/dgm/) | Open archives, code-level self-improvement, non-monotonic intermediate candidates; not weight-level decensoring recursion |
| S2 | [Self-Adapting Language Models / SEAL](https://arxiv.org/abs/2506.10943) | Model-generated update data/instructions and downstream feedback; not evidence of lossless edits by this method |
| S3 | [Self-Preference Bias in LLM-as-a-Judge](https://arxiv.org/abs/2410.21819) | Familiarity and same-model preference motivate role separation and calibration; not a claim that all judges fail |
| S4 | [A StrongREJECT for Empty Jailbreaks](https://arxiv.org/abs/2402.10260) | Distinguishes avoiding refusal phrases from actual completion; its task specification does not cover every use case |
| S5 | [Generalization in Adaptive Data Analysis and Holdout Reuse](https://arxiv.org/abs/1506.02629) | Adaptive holdout reuse risks overfitting; this prototype does not implement the paper's guarantee-bearing algorithms |
| S6 | [The Geometry of Refusal in Large Language Models](https://arxiv.org/abs/2502.17420) | Multiple directions and the distinction between functional independence and orthogonality; rank 1 should not be a universal requirement |
| S7 | [Optuna Trial API](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html) | Multi-objective limits of report/should_prune motivate external promotion scheduling |
| S8 | [Heretic ARA source](https://github.com/p-e-w/heretic/blob/edc3b123456c7f86f24d409b838ab3a7226e285e/src/heretic/model.py) | Local matrix optimization on cached module I/O; the small-subspace kernel here is not equivalent |
| S9 | [llama.cpp quant codecs](https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/gguf/quants.py), [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) | Codecs require type-specific implementation and qualification; decoding does not establish fixed-scale write-back support |

This version does not reuse DGM, SEAL, or Heretic source code and includes no model
weights or research datasets. recipe_lab is independently written reference code
whose mathematics and format semantics derive from public material. Directly
integrating third-party source later requires retaining licenses and checking compatibility.

The layered architecture, separate heads, semantic-window protocol, and optional
compression strategy are project design choices. Claims that freer exploration
improves results, lower refusal improves research ability, or strong RSD accelerates
itself remain hypotheses. The cited papers cannot replace project-specific experiments.
