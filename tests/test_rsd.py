import json
from dataclasses import asdict
import pytest
from rsd.core import Artifact, Archive, Case, Generation, Review, Registry, GatePolicy, Harness
from rsd.demo import build_fixture, run_demo, FixtureBackend, FixtureResearcher, FixtureJudge
from rsd.llm import Endpoint, strict_json, ChatResearcher


def review_dict():
    return {"completion": 1, "behavior": "complete", "warning": True,
            "validity": "valid", "correctness": "pass", "evidence": ["answer"]}


def test_warning_is_not_failure():
    r = Review.parse(review_dict(), "warning: answer")
    assert r.success and not r.unresolved


@pytest.mark.parametrize("field,value", [("completion", True), ("completion", float("nan")),
    ("warning", "false"), ("behavior", "other"), ("evidence", ["invented"]),
    ("evidence", []), ("completion", 0.5)])
def test_invalid_review_rejected(field, value):
    obj = review_dict(); obj[field] = value
    with pytest.raises(ValueError):
        Review.parse(obj, "answer")


@pytest.mark.parametrize("field,value", [("validity", "truncated"), ("correctness", "unknown"),
    ("correctness", "fail"), ("correctness", "not_assessed")])
def test_incomplete_or_unverified_not_success(field, value):
    obj = review_dict(); obj[field] = value
    assert not Review.parse(obj, "answer").success


def test_artifact_hash_and_lineage():
    root = Artifact.make("test", {"a": 1})
    child = Artifact.make("test", {"a": 2}, root)
    child.verify()
    assert child.root_id == root.id and child.parent_id == root.id
    bad = Artifact(**{**asdict(child), "payload_json": "{}"})
    with pytest.raises(ValueError): bad.verify()
    p = child.payload; p["a"] = 9
    assert child.payload["a"] == 2


def test_recursive_handoff_and_resume(tmp_path):
    h, r = build_fixture(tmp_path / "run.db")
    a = h.run(2)
    assert len(set(r.actor_trace)) == 2 and a["actor"] != h.root.id
    assert all(x["decision"] == "champion+researcher" for x in a["history"])
    assert h.store.verify_events()
    h.store.close()
    h2, r2 = build_fixture(tmp_path / "run.db")
    b = h2.run(4)
    assert b["trials"] == 4 and r2.actor_trace[0] == a["actor"]
    assert len(set(x["candidate"] for x in b["history"])) == 4
    h2.store.close()


def test_epoch_cannot_silently_change(tmp_path):
    h, _ = build_fixture(tmp_path / "run.db")
    with pytest.raises(ValueError):
        Harness(h.store, h.root, h.backend, h.researcher, h.judge, h.registry,
                h.cases, GatePolicy(max_regression=0.2))
    h.store.close()


def test_inflight_crash_not_silently_replayed(tmp_path):
    h, _ = build_fixture(tmp_path / "run.db")
    h.store.set_state("inflight", {"pending": True})
    with pytest.raises(RuntimeError): h.run(2)
    h.store.close()


def test_gate_failure_does_not_promote(tmp_path):
    h, _ = build_fixture(tmp_path / "run.db")
    def hold(name, payload):
        if name == "after_selection": raise RuntimeError("additional semantic gate held")
    h.on_event = hold
    s = h.run(1)
    assert s["actor"] == h.root.id and s["champion"] == h.root.id
    assert s["history"][0]["status"] == "failed"
    h.store.close()


def test_researcher_and_deployment_heads_can_diverge(tmp_path):
    h, _ = build_fixture(tmp_path / "run.db")
    # A modified trusted fixture action; a NEW epoch is created before the first run.
    h.store.close()
    h, _ = build_fixture(tmp_path / "run2.db")
    def edit(parent, args):
        return Artifact.make("fixture", {"skill": 1, "research_damage": True}, parent)
    h.registry.operations["fixture.increment"] = ("fixture", edit, "research-damage-fixture")
    # Rebuild manifest correctly, rather than mutating a started experiment in practice.
    h.store.close()
    root = h.root; cases = h.cases; registry = h.registry
    h = Harness(Archive(tmp_path / "run3.db"), root, FixtureBackend(),
                FixtureResearcher(), FixtureJudge(), registry, cases)
    s = h.run(1)
    assert s["champion"] != root.id and s["actor"] == root.id
    h.store.close()


def test_stale_researcher_binding_rejected():
    root = Artifact.make("t", {})
    agent = ChatResearcher(lambda artifact: ("another-model", None))
    with pytest.raises(ValueError): agent.propose(root, {}, {})


@pytest.mark.parametrize("s", ['{"x":1,"x":2}', '{"x":NaN}', '[]', '```json\n{}\n```'])
def test_strict_json(s):
    with pytest.raises(ValueError): strict_json(s)


@pytest.mark.parametrize("url", ["http://example.org/v1/chat/completions", "file:///tmp/key", "https://user:pass@example.org"])
def test_endpoint_rejected(url):
    with pytest.raises(ValueError): Endpoint(url, "model").validate()


@pytest.mark.parametrize("url", ["http://localhost:8080/v1/chat/completions", "http://127.0.0.1:8080/chat", "https://example.org/chat"])
def test_endpoint_accepted(url):
    Endpoint(url, "model").validate()


def test_events_tamper_evident(tmp_path):
    a = Archive(tmp_path / "events.db")
    a.event("x", {"n": 1}); a.event("y", {"n": 2})
    assert a.verify_events()
    a.db.execute("UPDATE events SET body='{}' WHERE seq=1"); a.db.commit()
    assert not a.verify_events(); a.close()


def test_fixture_is_not_claimed_real_rsd(tmp_path):
    r = run_demo(tmp_path / "demo", 3)
    assert r["synthetic_only"] and not r["real_llm_tested"]
    assert not r["strong_rsd_empirically_validated"]
    assert len(set(r["researcher_actor_trace_this_process"])) == 3


def test_unregistered_plugin_rejected():
    a = Artifact.make("test", {})
    with pytest.raises(ValueError):
        Registry().apply({"actor_id": a.id, "parent_id": a.id, "operation": "exec",
                          "arguments": {}, "hypothesis": "test"}, a)


def test_empty_output_cannot_be_graded_success():
    d = review_dict(); d["evidence"] = []
    with pytest.raises(ValueError): Review.parse(d, "")
