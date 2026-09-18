"""Optional bridge for an already loaded torch model and chat tokenizer.

No model download, architecture guessing, or quantized file rewrite. Callers
qualify exact linear-writer targets and provide an immutable source fingerprint.
Both task generation and researcher proposals execute the selected descendant.
"""
from __future__ import annotations
from contextlib import ExitStack, contextmanager
import numpy as np
from recipe_lab.linear import DirectionPatch
from recipe_lab.hooks import output_patch
from .core import Artifact, Case, Generation, Registry, canonical, digest
from .llm import RESEARCH_PROMPT, strict_json


class TorchWorker:
    def __init__(self, model, tokenizer, root: Artifact, targets: list[str],
                 runtime_fingerprint: str, input_device="cpu", max_new_tokens=256,
                 system_prompt="", clear_state=None):
        root.verify()
        if root.parent_id is not None or root.payload.get("patches") != []:
            raise ValueError("worker requires a pristine root with patches=[]")
        if not root.payload.get("source_fingerprint") or not runtime_fingerprint:
            raise ValueError("source/runtime fingerprints required")
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("invalid generation limit")
        self.model, self.tokenizer, self.root = model, tokenizer, root
        self.modules = dict(model.named_modules())
        if len(set(targets)) != len(targets) or any(t not in self.modules for t in targets):
            raise ValueError("duplicate/missing qualified writer target")
        self.targets = set(targets)
        self.device, self.max_tokens = input_device, max_new_tokens
        self.system, self.clear_state = system_prompt, clear_state or (lambda: None)
        self.version = "torch-worker-v1:" + digest({"root": root.id,
            "targets": targets, "runtime": runtime_fingerprint,
            "max_new_tokens": max_new_tokens, "system": system_prompt})

    def _patches(self, artifact):
        artifact.verify()
        if artifact.root_id != self.root.id or artifact.backend != self.root.backend:
            raise ValueError("wrong worker lineage")
        payload = artifact.payload
        if payload.get("source_fingerprint") != self.root.payload["source_fingerprint"]:
            raise ValueError("source fingerprint mismatch")
        patches = payload.get("patches")
        if not isinstance(patches, list):
            raise ValueError("missing ordered patch list")
        result = []
        for p in patches:
            if set(p) != {"target", "directions", "strengths"} or p["target"] not in self.targets:
                raise ValueError("patch target not qualified")
            patch = DirectionPatch(np.asarray(p["directions"]), np.asarray(p["strengths"]))
            result.append((self.modules[p["target"]], patch))
        return result

    @contextmanager
    def activate(self, artifact):
        patches = self._patches(artifact)
        self.clear_state()
        training = self.model.training
        self.model.eval()
        try:
            with ExitStack() as stack:
                for module, patch in patches:
                    stack.enter_context(output_patch(module, patch))
                yield
        finally:
            self.model.train(training)
            self.clear_state()

    def text(self, artifact: Artifact, system: str, user: str) -> Generation:
        import torch
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        # No universal template fallback: a missing/incompatible template is an error.
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False,
                                                      add_generation_prompt=True)
        inputs = self.tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self.activate(artifact), torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=self.max_tokens, do_sample=False)
        sequences = out.sequences if hasattr(out, "sequences") else out
        ids = sequences[0, inputs["input_ids"].shape[1]:].detach().cpu().tolist()
        config = getattr(self.model, "generation_config", None)
        eos = getattr(config, "eos_token_id", None)
        if eos is None:
            eos = getattr(self.tokenizer, "eos_token_id", None)
        eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
        stopped = bool(ids and ids[-1] in eos) or len(ids) < self.max_tokens
        return Generation(self.tokenizer.decode(ids, skip_special_tokens=True),
                          "stop" if stopped else "length", len(ids))

    def generate(self, artifact: Artifact, cases: list[Case]) -> dict[str, Generation]:
        return {c.id: self.text(artifact, self.system, c.prompt) for c in cases}

    def register_direction_operation(self, registry: Registry):
        def apply(parent, arguments):
            if set(arguments) != {"target", "directions", "strengths"}:
                raise ValueError("direction operation arguments do not match")
            payload = parent.payload
            child = Artifact.make(parent.backend, {**payload,
                "patches": payload["patches"] + [arguments]}, parent)
            self._patches(child)  # validate BEFORE installing any hooks
            return child
        registry.register("writer.direction", "Append an ordered output-direction patch. "
            "Arguments: target (qualified writer name), directions [out_dim,rank] "
            "with orthonormal columns, strengths [rank]. Targets: " + ", ".join(sorted(self.targets)),
            apply, self.version + ":direction-v1")


class TorchResearcher:
    """Research generated by the same virtual descendant, not an external fixed endpoint."""
    def __init__(self, worker: TorchWorker):
        self.worker = worker
        self.version = "torch-researcher-v1:" + digest({"worker": worker.version,
                                                       "prompt": RESEARCH_PROMPT})

    def propose(self, actor: Artifact, observation: dict, catalog: dict) -> dict:
        out = self.worker.text(actor, RESEARCH_PROMPT,
            canonical({"actor_id": actor.id, "observation": observation, "operations": catalog}))
        if out.finish != "stop":
            raise ValueError("research proposal truncated; no implicit budget increase")
        return strict_json(out.text)
