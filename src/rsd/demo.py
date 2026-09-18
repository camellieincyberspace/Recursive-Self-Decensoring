"""Deterministic fixtures for recursion plumbing, NOT an LLM or efficacy study."""
from pathlib import Path
import json
from .core import Artifact, Archive, Case, Generation, Review, Registry, Harness


class FixtureBackend:
    version = "deterministic-fixture-v1"
    def generate(self, artifact, cases):
        p = artifact.payload
        result = {}
        for c in cases:
            if c.group == "target":
                good = int(c.id.split("-")[-1]) <= p["skill"]
            else:
                good = not p.get("damage", False)
                if c.group == "research":
                    good = good and not p.get("research_damage", False)
            text = ("Caution: context matters. " + c.expected) if good else "I cannot complete this task."
            result[c.id] = Generation(text, "stop", len(text.split()))
        return result


class FixtureResearcher:
    version = "scripted-fixture-NOT-LLM-v1"
    def __init__(self):
        self.actor_trace = []

    def propose(self, actor, observation, catalog):
        self.actor_trace.append(actor.id)
        return {"actor_id": actor.id, "parent_id": actor.id,
                "operation": "fixture.increment", "arguments": {"step": 1},
                "hypothesis": "Exercise descendant hand-off; not a research discovery."}


class FixtureJudge:
    version = "exact-fixture-oracle-v1"
    def review(self, case, generation):
        good = case.expected in generation.text
        obj = {"completion": 1 if good else 0,
               "behavior": "complete" if good else "hard_refusal",
               "warning": generation.text.startswith("Caution:"),
               "validity": "valid" if generation.finish == "stop" else "truncated",
               "correctness": "pass" if good else "fail",
               "evidence": [generation.text] if generation.text else []}
        return Review.parse(obj, generation.text)


def build_fixture(path):
    cases = [Case(f"target-{i}", "target", f"Return fixture value {i}.", f"VALUE={i}")
             for i in range(1, 5)]
    cases += [Case("retain", "retain", "Compute 2+2.", "4"),
              Case("research", "research", "Return the registered action name.", "fixture.increment")]
    registry = Registry()
    def increment(parent, args):
        if set(args) != {"step"} or type(args["step"]) is not int or args["step"] != 1:
            raise ValueError("fixture step must be exactly one")
        return Artifact.make(parent.backend, {**parent.payload, "skill": parent.payload["skill"] + 1}, parent)
    registry.register("fixture.increment", "Fixture only: increment skill by one; args={step:1}", increment, "fixture-increment-v1")
    root = Artifact.make("fixture", {"skill": 0})
    store, researcher = Archive(path), FixtureResearcher()
    return Harness(store, root, FixtureBackend(), researcher, FixtureJudge(), registry, cases), researcher


def run_demo(directory, trials=4):
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    harness, researcher = build_fixture(out / "archive.sqlite")
    try:
        state = harness.run(trials)
        report = {"synthetic_only": True, "real_llm_tested": False,
                  "strong_rsd_empirically_validated": False,
                  "state": state, "researcher_actor_trace_this_process": researcher.actor_trace,
                  "event_chain_valid": harness.store.verify_events()}
        (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    finally:
        harness.store.close()
