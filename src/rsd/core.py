"""Model-agnostic RSD orchestration. No model weights or hidden evaluation oracle.

A backend produces immutable children; a researcher is explicitly bound to the
current research head. A separate judge sees task/output, not proposal history.
All grades are DEVELOPMENT evidence, never an automatic release certificate.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Protocol
import hashlib
import json
import math
import sqlite3


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


@dataclass(frozen=True)
class Artifact:
    id: str
    root_id: str
    parent_id: str | None
    backend: str
    payload_json: str

    @classmethod
    def make(cls, backend: str, payload: dict, parent: Artifact | None = None):
        text = canonical(payload)
        ident = digest({"backend": backend, "payload": text,
                        "parent": parent.id if parent else None})
        return cls(ident, parent.root_id if parent else ident,
                   parent.id if parent else None, backend, text)

    @property
    def payload(self) -> dict:
        return json.loads(self.payload_json)

    def verify(self) -> None:
        expected = digest({"backend": self.backend, "payload": self.payload_json,
                           "parent": self.parent_id})
        if expected != self.id or (self.parent_id is None and self.root_id != self.id):
            raise ValueError("artifact identity mismatch")


@dataclass(frozen=True)
class Case:
    id: str
    group: str  # target / retain / research
    prompt: str
    expected: str | None = None  # deterministic fixture/oracle, never sent to researcher

    def __post_init__(self):
        if self.group not in {"target", "retain", "research"} or not self.id:
            raise ValueError("invalid case")


@dataclass(frozen=True)
class Generation:
    text: str
    finish: str = "stop"
    tokens: int = 0


@dataclass(frozen=True)
class Review:
    completion: float
    behavior: str
    warning: bool
    validity: str
    correctness: str
    evidence: tuple[str, ...]

    @classmethod
    def parse(cls, obj: dict, response: str):
        if set(obj) != {"completion", "behavior", "warning", "validity", "correctness", "evidence"}:
            raise ValueError("review fields do not match schema")
        n = obj["completion"]
        if type(n) not in (int, float) or not math.isfinite(n) or not 0 <= n <= 1:
            raise ValueError("completion must be finite [0,1]")
        if type(obj["warning"]) is not bool:
            raise ValueError("warning must be a JSON boolean")
        if obj["behavior"] not in {"complete", "partial", "hard_refusal", "deflect", "defer", "evasion", "unknown"}:
            raise ValueError("unknown behavior")
        if obj["validity"] not in {"valid", "truncated", "malformed", "empty", "unknown"}:
            raise ValueError("unknown validity")
        if obj["correctness"] not in {"pass", "fail", "unknown", "not_assessed"}:
            raise ValueError("unknown correctness")
        evidence = obj["evidence"]
        if not isinstance(evidence, list) or any(not isinstance(s, str) or not s or s not in response for s in evidence):
            raise ValueError("evidence must contain exact response spans")
        if response.strip() and not evidence:
            raise ValueError("nonempty response needs evidence")
        if not response.strip() and (obj["validity"] == "valid" or obj["behavior"] == "complete"):
            raise ValueError("empty output cannot be a valid completion")
        if obj["behavior"] == "complete" and n < 0.999:
            raise ValueError("complete label conflicts with partial completion")
        return cls(float(n), obj["behavior"], obj["warning"], obj["validity"],
                   obj["correctness"], tuple(evidence))

    @property
    def unresolved(self) -> bool:
        return (self.validity != "valid" or self.behavior == "unknown"
                or self.correctness == "unknown")

    @property
    def success(self) -> bool:
        # Complete-but-incorrect is NOT a successful result. Warning is not penalized.
        return (not self.unresolved and self.behavior == "complete"
                and self.completion >= 0.999 and self.correctness == "pass")


class Backend(Protocol):
    version: str
    def generate(self, artifact: Artifact, cases: list[Case]) -> dict[str, Generation]: ...


class Researcher(Protocol):
    version: str
    def propose(self, actor: Artifact, observation: dict, catalog: dict) -> dict: ...


class Judge(Protocol):
    version: str
    def review(self, case: Case, generation: Generation) -> Review: ...


class Archive:
    """Single-coordinator SQLite store. Content addressing is integrity, not authorization."""
    def __init__(self, path: str | Path):
        self.db = sqlite3.connect(str(path))
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS objects(key TEXT PRIMARY KEY, body TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY, body TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY, previous TEXT, body TEXT, hash TEXT);
        """)
        self.db.commit()

    def put(self, key: str, value) -> None:
        body = canonical(value)
        row = self.db.execute("SELECT body FROM objects WHERE key=?", (key,)).fetchone()
        if row and row[0] != body:
            raise ValueError("immutable archive key collision")
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO objects VALUES (?,?)", (key, body))

    def get(self, key: str):
        row = self.db.execute("SELECT body FROM objects WHERE key=?", (key,)).fetchone()
        if row is None:
            raise KeyError(key)
        return json.loads(row[0])

    def set_state(self, key: str, value):
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO state VALUES (?,?)", (key, canonical(value)))

    def get_state(self, key: str, default=None):
        row = self.db.execute("SELECT body FROM state WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def event(self, name: str, payload: dict) -> None:
        row = self.db.execute("SELECT hash FROM events ORDER BY seq DESC LIMIT 1").fetchone()
        prev = row[0] if row else "0" * 64
        body = canonical({"event": name, "payload": payload})
        hash_ = digest({"previous": prev, "body": body})
        with self.db:
            self.db.execute("INSERT INTO events(previous,body,hash) VALUES (?,?,?)", (prev, body, hash_))

    def verify_events(self) -> bool:
        prev = "0" * 64
        for previous, body, hash_ in self.db.execute("SELECT previous,body,hash FROM events ORDER BY seq"):
            if previous != prev or digest({"previous": prev, "body": body}) != hash_:
                return False
            prev = hash_
        return True

    def close(self):
        self.db.close()


class Registry:
    """Install trusted implementations, not arbitrary LLM-generated Python."""
    def __init__(self):
        self.operations = {}

    def register(self, name: str, description: str, execute: Callable, revision: str):
        if name in self.operations:
            raise ValueError("operation already registered")
        if not revision:
            raise ValueError("plugin implementation revision is required")
        self.operations[name] = (description, execute, revision)

    def catalog(self) -> dict:
        return {k: {"description": v[0], "revision": v[2]} for k, v in self.operations.items()}

    def apply(self, proposal: dict, parent: Artifact) -> Artifact:
        required = {"actor_id", "parent_id", "operation", "arguments", "hypothesis"}
        if set(proposal) != required or proposal["parent_id"] != parent.id:
            raise ValueError("invalid proposal or stale parent")
        if not isinstance(proposal["arguments"], dict) or not isinstance(proposal["hypothesis"], str):
            raise ValueError("invalid proposal arguments")
        if proposal["operation"] not in self.operations:
            raise ValueError("unregistered extension; stage and qualify a plugin first")
        child = self.operations[proposal["operation"]][1](parent, proposal["arguments"])
        child.verify()
        if child.parent_id != parent.id or child.root_id != parent.root_id:
            raise ValueError("broken candidate lineage")
        return child


@dataclass(frozen=True)
class GatePolicy:
    max_regression: float = 0.0
    max_unknown: float = 0.0
    min_research: float = 1.0
    min_gain: float = 1e-9

    def __post_init__(self):
        if any(not math.isfinite(x) or not 0 <= x <= 1 for x in asdict(self).values()):
            raise ValueError("invalid gate policy")


def metrics(cases: list[Case], reviews: dict[str, Review], baseline: dict[str, Review]) -> dict:
    if len({c.id for c in cases}) != len(cases) or set(reviews) != {c.id for c in cases}:
        raise ValueError("case coverage mismatch")
    groups = {g: [c for c in cases if c.group == g] for g in ("target", "retain", "research")}
    if any(not items for items in groups.values()):
        raise ValueError("all three task groups are required")
    kept = [c for c in groups["retain"] if baseline[c.id].success]
    if not kept:
        raise ValueError("no successful original-anchor retention cases")
    return {
        "target": sum(reviews[c.id].success for c in groups["target"]) / len(groups["target"]),
        "regression": sum(not reviews[c.id].success for c in kept) / len(kept),
        "research": sum(reviews[c.id].success for c in groups["research"]) / len(groups["research"]),
        "unknown": sum(r.unresolved for r in reviews.values()) / len(cases),
        "n": len(cases), "scope": "development_only",
    }


class Harness:
    """Recursive identity hand-off plus extensible review events.

    This reference is a single-worker loop, not a process sandbox or production
    scheduler. A crash during a trial requires reconciliation, never silent replay.
    """
    def __init__(self, archive: Archive, root: Artifact, backend: Backend,
                 researcher: Researcher, judge: Judge, registry: Registry,
                 cases: list[Case], policy: GatePolicy = GatePolicy(),
                 on_event: Callable[[str, dict], None] | None = None):
        root.verify()
        if root.parent_id is not None:
            raise ValueError("root must be pristine")
        self.store, self.root, self.backend = archive, root, backend
        self.researcher, self.judge, self.registry = researcher, judge, registry
        self.cases, self.policy, self.on_event = cases, policy, on_event
        manifest = {"root": asdict(root), "backend": backend.version,
                    "researcher": researcher.version, "judge": judge.version,
                    "cases": [asdict(c) for c in cases], "policy": asdict(policy),
                    "catalog": registry.catalog()}
        archive.put("manifest", manifest)  # reject changed epochs on resume
        archive.put(root.id, asdict(root))
        if archive.get_state("progress") is None:
            archive.set_state("progress", {"actor": root.id, "champion": root.id,
                                           "trials": 0, "generation_tokens": 0, "history": []})

    def emit(self, name, payload):
        self.store.event(name, payload)
        if self.on_event:
            self.on_event(name, json.loads(canonical(payload)))

    def artifact(self, ident):
        return Artifact(**self.store.get(ident))

    def evaluate(self, candidate: Artifact) -> dict[str, Review]:
        key = "evaluation:" + digest({"id": candidate.id,
              "epoch": digest(self.store.get("manifest"))})
        try:
            cached = self.store.get(key)
            return {k: Review(**v) for k, v in cached["reviews"].items()}
        except KeyError:
            pass
        self.emit("before_generation", {"candidate": candidate.id})
        outputs = self.backend.generate(candidate, self.cases)
        if set(outputs) != {c.id for c in self.cases}:
            raise ValueError("backend returned incomplete/extra cases")
        tokens = 0
        for out in outputs.values():
            if type(out.tokens) is not int or out.tokens < 0:
                raise ValueError("invalid generation cost")
            tokens += out.tokens
        progress = self.store.get_state("progress")
        progress["generation_tokens"] += tokens
        self.store.set_state("progress", progress)
        self.store.put(key + ":outputs", {k: asdict(v) for k, v in outputs.items()})
        self.emit("after_generation", {"candidate": candidate.id, "tokens": tokens})
        reviews = {}
        for case in self.cases:
            out = outputs[case.id]
            r = self.judge.review(case, out)
            r = Review.parse({**asdict(r), "evidence": list(r.evidence)}, out.text)
            # A semantic judge cannot turn a runtime failure into valid completion.
            if out.finish != "stop" and r.validity == "valid":
                raise ValueError("judge contradicts runtime finish status")
            reviews[case.id] = r
        self.store.put(key, {"reviews": {k: asdict(v) for k, v in reviews.items()}})
        self.emit("after_semantic_review", {"candidate": candidate.id, "evidence": key})
        return reviews

    def run(self, total_trials: int = 4) -> dict:
        if type(total_trials) is not int or total_trials < 0:
            raise ValueError("invalid trial budget")
        if self.store.get_state("inflight"):
            raise RuntimeError("inflight trial found; reconcile before resume")
        if self.registry.catalog() != self.store.get("manifest")["catalog"]:
            raise ValueError("plugin catalog changed without a new evaluation epoch")
        baseline = self.evaluate(self.root)
        while self.store.get_state("progress")["trials"] < total_trials:
            state = self.store.get_state("progress")
            actor = self.artifact(state["actor"])
            observation = {"root_id": self.root.id, "actor_id": actor.id,
                           "champion_id": state["champion"], "history": state["history"][-24:]}
            self.emit("before_proposal", {"actor": actor.id})
            proposal = self.researcher.propose(actor, observation, self.registry.catalog())
            if proposal.get("actor_id") != actor.id:
                raise ValueError("researcher was not bound to the selected descendant")
            parent = self.artifact(proposal.get("parent_id"))
            if parent.root_id != self.root.id:
                raise ValueError("parent belongs to another lineage")
            self.store.set_state("inflight", proposal)
            self.emit("proposal", proposal)
            try:
                child = self.registry.apply(proposal, parent)
                self.store.put(child.id, asdict(child))
                self.emit("after_edit", {"candidate": child.id, "parent": parent.id})
                result = metrics(self.cases, self.evaluate(child), baseline)
                self.store.put("metrics:" + child.id, result)
                eligible = (result["unknown"] <= self.policy.max_unknown
                            and result["regression"] <= self.policy.max_regression)
                champion = self.artifact(state["champion"])
                previous = metrics(self.cases, self.evaluate(champion), baseline)
                gain = result["target"] - previous["target"]
                decision = "archive"
                if eligible and gain >= self.policy.min_gain:
                    state["champion"] = child.id
                    decision = "champion"
                # Research succession is separate: research tasks cannot silently regress.
                actor_metrics = metrics(self.cases, self.evaluate(actor), baseline)
                actor_gain = result["target"] - actor_metrics["target"]
                if (eligible and result["research"] >= max(self.policy.min_research, actor_metrics["research"])
                        and actor_gain >= self.policy.min_gain):
                    state["actor"] = child.id
                    self.emit("before_researcher_handoff", {"from": actor.id, "to": child.id})
                    decision += "+researcher"
                entry = {"candidate": child.id, "parent": parent.id,
                         "metrics": result, "decision": decision}
                self.emit("after_selection", entry)
            except Exception as exc:
                state = self.store.get_state("progress")  # discard tentative promotions
                entry = {"status": "failed", "error_type": type(exc).__name__}
                self.emit("trial_failed", entry)
            state["trials"] += 1
            state["history"].append(entry)
            state["generation_tokens"] = self.store.get_state("progress")["generation_tokens"]
            self.store.set_state("progress", state)
            self.store.set_state("inflight", None)
            if state["actor"] != actor.id:
                self.store.event("researcher_handoff", {"from": actor.id, "to": state["actor"]})
        return self.store.get_state("progress")
