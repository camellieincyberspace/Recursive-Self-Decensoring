"""Tiny neural fixture: tests actual hook routing, NOT language-model efficacy."""
import json
from types import SimpleNamespace
import pytest
from rsd.core import Artifact, Registry, Case
from rsd.torch_worker import TorchWorker, TorchResearcher

torch = pytest.importorskip("torch")

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.writer = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad(): self.writer.weight.copy_(torch.eye(2))
        self.generation_config = SimpleNamespace(eos_token_id=9)
    def generate(self, input_ids, **kwargs):
        value = self.writer(torch.tensor([[1., 0.]])).argmax(-1).item()
        return torch.cat([input_ids, torch.tensor([[value + 5, 9]])], dim=1)

class Tokenizer:
    eos_token_id = 9
    def apply_chat_template(self, messages, **kwargs): return json.dumps(messages)
    def __call__(self, text, **kwargs): return {"input_ids": torch.tensor([[1, 2]])}
    def decode(self, ids, **kwargs): return str(ids[0])

def fixture():
    model = Model()
    root = Artifact.make("torch", {"source_fingerprint": "fixture-source-v1", "patches": []})
    worker = TorchWorker(model, Tokenizer(), root, ["writer"], "fixture-runtime")
    registry = Registry(); worker.register_direction_operation(registry)
    proposal = {"actor_id": root.id, "parent_id": root.id, "operation": "writer.direction",
                "arguments": {"target": "writer", "directions": [[1.], [0.]], "strengths": [2.]},
                "hypothesis": "test fixture sign change"}
    return worker, root, registry.apply(proposal, root)

def test_actual_descendant_overlay_changes_execution_and_restores():
    w, root, child = fixture()
    assert w.generate(root, [Case("x", "target", "test")])["x"].text == "5"
    assert w.generate(child, [Case("x", "target", "test")])["x"].text == "6"
    assert w.generate(root, [Case("x", "target", "test")])["x"].text == "5"
    assert torch.equal(w.model.writer.weight, torch.eye(2))
    assert not w.model.writer._forward_hooks

def test_hooks_removed_on_exception():
    w, _, child = fixture()
    with pytest.raises(RuntimeError):
        with w.activate(child): raise RuntimeError("fixture")
    assert not w.model.writer._forward_hooks

def test_wrong_lineage_rejected():
    w, _, _ = fixture()
    with pytest.raises(ValueError):
        w._patches(Artifact.make("other", {}))

def test_researcher_calls_same_worker_with_actor():
    w, _, child = fixture(); trace = []
    original = w.text
    def traced(actor, system, user):
        trace.append(actor.id)
        return original(actor, system, user)
    w.text = traced
    # Fixture text is a number, not a research JSON object. It must fail parsing.
    with pytest.raises(ValueError): TorchResearcher(w).propose(child, {}, {})
    assert trace == [child.id]
