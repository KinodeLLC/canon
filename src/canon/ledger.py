"""
The Ledger: effect journal, capability broker, and audit chain.

Three things live here, and they are separate on purpose.

  Journal    every effect performed, in order, with its arguments and result.
             Hash-chained, so a missing or altered entry is detectable. This is
             what makes replay exact and what shadow deployment diffs against.

  Broker     decides whether an effect may be performed at all. A capability
             grant names operations, an actor, an expiry, a call ceiling and
             optional argument constraints. Nothing is permitted by default.

  Audit      an append-only, hash-chained record of governance events: grants
             issued and denied, definitions added, verifications run,
             promotions approved. Distinct from the journal because it answers
             a different question -- not "what did the program do" but "who
             authorised it, and on what basis".

Three execution modes:

  live     perform effects against real handlers and record them
  replay   perform nothing; return recorded results, asserting each call
           matches what was recorded
  shadow   run new code against a recorded journal: effect results come from
           the recording, and any divergence is collected rather than raised

Shadow mode is the mechanism that lets a change be evaluated against real
production traffic before it is promoted, without that change being able to
touch anything.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time as _host_time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Callable, Optional

from . import values as V
from .diagnostics import Span
from .interp import CapabilityDenied, Fault, Runtime


# --------------------------------------------------------------------------
# Hashing
# --------------------------------------------------------------------------

def _hash(*parts: bytes) -> str:
    h = hashlib.blake2b(digest_size=32)
    for p in parts:
        h.update(p)
        h.update(b"\x1e")
    return base64.b32encode(h.digest()).decode("ascii").rstrip("=").lower()[:32]


def _canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str).encode("utf-8")


GENESIS = "0" * 32


# --------------------------------------------------------------------------
# Journal
# --------------------------------------------------------------------------

@dataclass
class Entry:
    seq: int
    op: str
    args: list
    result: Any
    def_hash: str = ""
    logical: int = 0
    wall_millis: int = 0
    actor: str = ""
    meta: dict = field(default_factory=dict)
    prev: str = GENESIS
    hash: str = ""

    def payload(self) -> dict:
        return {
            "seq": self.seq,
            "op": self.op,
            "args": [V.to_json(a) if not isinstance(a, (dict, list, str, int, bool))
                     and a is not None else a for a in self.args],
            "result": V.to_json(self.result)
            if not isinstance(self.result, (dict, list, str, int, bool, type(None)))
            else self.result,
            "def_hash": self.def_hash,
            "logical": self.logical,
            "wall_millis": self.wall_millis,
            "actor": self.actor,
            "meta": self.meta,
            "prev": self.prev,
        }

    def compute_hash(self) -> str:
        return _hash(_canon(self.payload()))

    def to_json(self) -> dict:
        d = self.payload()
        d["hash"] = self.hash
        return d

    @staticmethod
    def from_json(d: dict) -> "Entry":
        e = Entry(
            seq=d["seq"], op=d["op"],
            args=[V.from_json(a) for a in d.get("args", [])],
            result=V.from_json(d.get("result")),
            def_hash=d.get("def_hash", ""),
            logical=d.get("logical", 0),
            wall_millis=d.get("wall_millis", 0),
            actor=d.get("actor", ""),
            meta=d.get("meta", {}),
            prev=d.get("prev", GENESIS),
        )
        e.hash = d.get("hash", "")
        return e

    def args_digest(self) -> str:
        return _hash(_canon([V.to_json(a) for a in self.args]))


class Journal:
    """An ordered, hash-chained record of every effect performed in a run."""

    def __init__(self, entries=None, run_id: str = ""):
        self.entries: list = list(entries or [])
        self.run_id = run_id or _hash(_canon({"t": 0, "n": len(self.entries)}))
        self._logical = max((e.logical for e in self.entries), default=0)

    # -- writing ---------------------------------------------------------

    def append(self, op: str, args: list, result: Any, def_hash: str = "",
               actor: str = "", meta: Optional[dict] = None,
               wall_millis: Optional[int] = None) -> Entry:
        self._logical += 1
        prev = self.entries[-1].hash if self.entries else GENESIS
        e = Entry(
            seq=len(self.entries), op=op, args=list(args), result=result,
            def_hash=def_hash, logical=self._logical,
            wall_millis=wall_millis if wall_millis is not None else 0,
            actor=actor, meta=dict(meta or {}), prev=prev)
        e.hash = e.compute_hash()
        self.entries.append(e)
        return e

    def next_logical(self) -> int:
        self._logical += 1
        return self._logical

    # -- reading ---------------------------------------------------------

    def __len__(self):
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    def ops(self) -> list:
        return [e.op for e in self.entries]

    def by_op(self, op: str) -> list:
        return [e for e in self.entries if e.op == op]

    def by_def(self, def_hash: str) -> list:
        return [e for e in self.entries if e.def_hash == def_hash]

    # -- integrity -------------------------------------------------------

    def verify(self) -> list:
        """
        Re-derive the chain. Returns a list of problems; empty means intact.

        This is what makes the journal usable as evidence rather than as a log:
        a modified argument, a deleted entry or a reordering all break the
        chain at a specific sequence number.
        """
        problems = []
        prev = GENESIS
        for i, e in enumerate(self.entries):
            if e.seq != i:
                problems.append({"seq": i, "problem": "sequence number mismatch",
                                 "found": e.seq})
            if e.prev != prev:
                problems.append({"seq": i, "problem": "broken chain",
                                 "expected_prev": prev, "found_prev": e.prev})
            recomputed = e.compute_hash()
            if e.hash != recomputed:
                problems.append({"seq": i, "problem": "entry hash mismatch",
                                 "recorded": e.hash, "recomputed": recomputed,
                                 "op": e.op})
            prev = e.hash
        return problems

    def head(self) -> str:
        return self.entries[-1].hash if self.entries else GENESIS

    # -- persistence -----------------------------------------------------

    def to_jsonl(self) -> str:
        return "\n".join(json.dumps(e.to_json(), sort_keys=True,
                                    separators=(",", ":"))
                         for e in self.entries)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(self.to_jsonl())
            if self.entries:
                f.write("\n")

    @staticmethod
    def load(path: str) -> "Journal":
        entries = []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(Entry.from_json(json.loads(line)))
        return Journal(entries)

    @staticmethod
    def from_jsonl(text: str) -> "Journal":
        entries = [Entry.from_json(json.loads(l))
                   for l in text.splitlines() if l.strip()]
        return Journal(entries)


# --------------------------------------------------------------------------
# Capability broker
# --------------------------------------------------------------------------

@dataclass
class Grant:
    """
    Permission to perform a set of effect operations, under constraints.

    `operations` accepts exact keys (`ledger.append`) and effect wildcards
    (`ledger.*`). `constraints` maps an operation key to a predicate over its
    arguments, which is how a grant can be narrowed to, say, one table or one
    spend ceiling rather than a whole effect.
    """
    id: str
    actor: str
    operations: list
    reason: str = ""
    max_calls: Optional[int] = None
    expires_logical: Optional[int] = None
    max_classification: str = "restricted"
    constraints: dict = field(default_factory=dict)
    calls_used: int = 0

    def covers(self, key: str) -> bool:
        for pattern in self.operations:
            if pattern == key:
                return True
            if pattern.endswith(".*") and key.split(".", 1)[0] == pattern[:-2]:
                return True
            if pattern == "*":
                return True
        return False

    def to_json(self) -> dict:
        return {"id": self.id, "actor": self.actor,
                "operations": list(self.operations), "reason": self.reason,
                "max_calls": self.max_calls,
                "expires_logical": self.expires_logical,
                "max_classification": self.max_classification,
                "calls_used": self.calls_used}


@dataclass
class Decision:
    allowed: bool
    grant_id: str = ""
    reason: str = ""
    detail: dict = field(default_factory=dict)


class CapabilityBroker:
    """
    Decides whether an effect may be performed. Denies by default.

    Every decision, allow or deny, is recorded in the audit log. A denied call
    is as much a governance fact as an approved one -- an agent repeatedly
    attempting an operation it was never granted is exactly the signal an
    operator wants surfaced.
    """

    def __init__(self, grants=None, audit: Optional["AuditLog"] = None):
        self.grants: list = list(grants or [])
        self.audit = audit
        self.denials: list = []
        self.calls: dict = {}

    def grant(self, actor: str, operations, reason: str = "", **kw) -> Grant:
        g = Grant(id=_hash(_canon({"a": actor, "o": sorted(operations),
                                   "n": len(self.grants)}))[:16],
                  actor=actor, operations=list(operations), reason=reason, **kw)
        self.grants.append(g)
        # `is not None`, not truthiness: AuditLog defines __len__, so an empty
        # log is falsy and the first record would be silently dropped.
        if self.audit is not None:
            self.audit.record("capability.granted", subject=actor,
                              detail=g.to_json())
        return g

    def revoke(self, grant_id: str) -> bool:
        for i, g in enumerate(self.grants):
            if g.id == grant_id:
                self.grants.pop(i)
                if self.audit:
                    self.audit.record("capability.revoked", subject=g.actor,
                                      detail={"grant": grant_id})
                return True
        return False

    def granted_operations(self) -> list:
        out = set()
        for g in self.grants:
            out.update(g.operations)
        return sorted(out)

    def check(self, key: str, args: list, logical: int = 0,
              classification: str = "public") -> Decision:
        from . import types as TY

        candidates = [g for g in self.grants if g.covers(key)]
        if not candidates:
            return self._deny(key, "no grant covers this operation",
                              {"operation": key,
                               "granted": self.granted_operations()})

        for g in candidates:
            if g.expires_logical is not None and logical > g.expires_logical:
                continue
            if g.max_calls is not None and g.calls_used >= g.max_calls:
                continue
            if TY.class_rank(classification) > TY.class_rank(g.max_classification):
                continue
            pred = g.constraints.get(key) or g.constraints.get("*")
            if pred is not None:
                try:
                    ok = pred(args)
                except Exception as ex:
                    ok = False
                if not ok:
                    continue
            g.calls_used += 1
            self.calls[key] = self.calls.get(key, 0) + 1
            return Decision(True, g.id)

        # Something covered the operation but every candidate refused; say why.
        reasons = []
        for g in candidates:
            if g.expires_logical is not None and logical > g.expires_logical:
                reasons.append(f"grant {g.id} expired")
            elif g.max_calls is not None and g.calls_used >= g.max_calls:
                reasons.append(f"grant {g.id} exhausted "
                               f"({g.calls_used}/{g.max_calls} calls)")
            elif TY.class_rank(classification) > TY.class_rank(g.max_classification):
                reasons.append(f"grant {g.id} is limited to "
                               f"{g.max_classification} data, this call "
                               f"touches {classification}")
            else:
                reasons.append(f"grant {g.id} constraint rejected the arguments")
        return self._deny(key, "; ".join(reasons),
                          {"operation": key, "candidates": len(candidates)})

    def _deny(self, key, reason, detail) -> Decision:
        d = Decision(False, "", reason, detail)
        self.denials.append({"operation": key, "reason": reason, **detail})
        # `is not None`, not truthiness: AuditLog defines __len__, so an empty
        # log is falsy and the first record would be silently dropped.
        if self.audit is not None:
            self.audit.record("capability.denied", subject=key,
                              detail={"reason": reason, **detail})
        return d


# --------------------------------------------------------------------------
# Audit log
# --------------------------------------------------------------------------

@dataclass
class AuditRecord:
    seq: int
    action: str
    subject: str
    actor: str
    detail: dict
    wall_millis: int
    prev: str = GENESIS
    hash: str = ""

    def payload(self) -> dict:
        return {"seq": self.seq, "action": self.action, "subject": self.subject,
                "actor": self.actor, "detail": self.detail,
                "wall_millis": self.wall_millis, "prev": self.prev}

    def compute_hash(self) -> str:
        return _hash(_canon(self.payload()))

    def to_json(self) -> dict:
        d = self.payload()
        d["hash"] = self.hash
        return d

    @staticmethod
    def from_json(d) -> "AuditRecord":
        r = AuditRecord(d["seq"], d["action"], d["subject"], d.get("actor", ""),
                        d.get("detail", {}), d.get("wall_millis", 0),
                        d.get("prev", GENESIS))
        r.hash = d.get("hash", "")
        return r


class AuditLog:
    """
    Append-only, hash-chained governance record.

    Nothing here is ever rewritten. A correction is a new record referring to
    the earlier one, which is what an auditor needs: the fact that a decision
    was changed is itself part of the record.
    """

    ACTIONS = [
        "definition.added", "definition.superseded",
        "verification.passed", "verification.failed",
        "capability.granted", "capability.denied", "capability.revoked",
        "promotion.requested", "promotion.approved", "promotion.blocked",
        "shadow.completed", "policy.evaluated", "intent.linked",
        "model.invoked", "run.started", "run.completed",
    ]

    def __init__(self, records=None, actor: str = "system",
                 clock: Optional[Callable[[], int]] = None):
        self.records: list = list(records or [])
        self.actor = actor
        self.clock = clock or (lambda: int(_host_time.time() * 1000))

    def record(self, action: str, subject: str = "", detail=None,
               actor: Optional[str] = None) -> AuditRecord:
        prev = self.records[-1].hash if self.records else GENESIS
        r = AuditRecord(
            seq=len(self.records), action=action, subject=subject,
            actor=actor or self.actor, detail=dict(detail or {}),
            wall_millis=self.clock(), prev=prev)
        r.hash = r.compute_hash()
        self.records.append(r)
        return r

    def verify(self) -> list:
        problems = []
        prev = GENESIS
        for i, r in enumerate(self.records):
            if r.prev != prev:
                problems.append({"seq": i, "problem": "broken chain"})
            if r.hash != r.compute_hash():
                problems.append({"seq": i, "problem": "record hash mismatch",
                                 "action": r.action})
            prev = r.hash
        return problems

    def head(self) -> str:
        return self.records[-1].hash if self.records else GENESIS

    def by_action(self, action: str) -> list:
        return [r for r in self.records if r.action == action]

    def by_subject(self, subject: str) -> list:
        return [r for r in self.records if r.subject == subject]

    def to_jsonl(self) -> str:
        return "\n".join(json.dumps(r.to_json(), sort_keys=True,
                                    separators=(",", ":"))
                         for r in self.records)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(self.to_jsonl())
            if self.records:
                f.write("\n")

    @staticmethod
    def load(path: str) -> "AuditLog":
        records = []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        records.append(AuditRecord.from_json(json.loads(line)))
        return AuditLog(records)

    def __len__(self):
        return len(self.records)

    def __iter__(self):
        return iter(self.records)


# --------------------------------------------------------------------------
# Divergence
# --------------------------------------------------------------------------

@dataclass
class Divergence:
    seq: int
    kind: str          # operation | arguments | extra | missing
    expected: Any = None
    actual: Any = None
    detail: str = ""

    def to_json(self) -> dict:
        return {"seq": self.seq, "kind": self.kind, "detail": self.detail,
                "expected": V.to_json(self.expected)
                if self.expected is not None else None,
                "actual": V.to_json(self.actual)
                if self.actual is not None else None}


# --------------------------------------------------------------------------
# The Ledger runtime
# --------------------------------------------------------------------------

LIVE = "live"
REPLAY = "replay"
SHADOW = "shadow"


class Ledger(Runtime):
    """
    The Runtime the interpreter performs effects against.

    Handlers are supplied per effect operation. Anything without a handler in
    live mode is a fault rather than a silent no-op: a program that thinks it
    wrote to a ledger and did not is worse than one that stops.
    """

    def __init__(self, broker: Optional[CapabilityBroker] = None,
                 journal: Optional[Journal] = None,
                 audit: Optional[AuditLog] = None,
                 mode: str = LIVE,
                 handlers: Optional[dict] = None,
                 actor: str = "agent",
                 model_runtime=None,
                 start_millis: int = 1_700_000_000_000):
        self.audit = audit if audit is not None else AuditLog(actor=actor)
        self.broker = broker if broker is not None else CapabilityBroker(
            audit=self.audit)
        if self.broker.audit is None:
            self.broker.audit = self.audit
        self.journal = journal if journal is not None else Journal()
        self.mode = mode
        self.handlers = dict(handlers or {})
        self.actor = actor
        self.model_runtime = model_runtime
        self.divergences: list = []
        self.classification_of: dict = {}
        self._cursor = 0
        self._start_millis = start_millis
        self._source = None        # journal being replayed against
        self.current_def_hash = ""

    # ------------------------------------------------------------------

    @staticmethod
    def replaying(source: Journal, **kw) -> "Ledger":
        led = Ledger(mode=REPLAY, **kw)
        led._source = source
        return led

    @staticmethod
    def shadowing(source: Journal, **kw) -> "Ledger":
        led = Ledger(mode=SHADOW, **kw)
        led._source = source
        return led

    def handle(self, key: str, fn: Callable):
        """Register a handler for one effect operation."""
        self.handlers[key] = fn
        return self

    def granted(self) -> list:
        return self.broker.granted_operations()

    # ------------------------------------------------------------------

    def perform(self, key: str, args: list, span=None,
                idempotent: bool = False):
        logical = self.journal.next_logical()
        classification = self.classification_of.get(key, "public")

        decision = self.broker.check(key, args, logical, classification)
        if not decision.allowed:
            raise CapabilityDenied(
                "CANON-E0403",
                f"capability {key!r} is not granted",
                span,
                facts={"operation": key,
                       "reason": decision.reason,
                       "granted": self.granted(),
                       "actor": self.actor,
                       **decision.detail})

        if self.mode in (REPLAY, SHADOW):
            return self._from_recording(key, args, span, logical)

        result = self._execute(key, args, span)
        self.journal.append(key, args, result, def_hash=self.current_def_hash,
                            actor=self.actor,
                            wall_millis=self._start_millis + logical,
                            meta={"grant": decision.grant_id,
                                  "idempotent": idempotent})
        return result

    def _execute(self, key: str, args: list, span):
        fn = self.handlers.get(key)
        if fn is None:
            # Built-in nondeterministic sources are handled here so they are
            # journaled and therefore reproducible.
            if key == "time.now":
                return V.Instant(self._start_millis + self.journal._logical,
                                 self.journal._logical)
            if key == "random.int":
                lo, hi = (args + [0, 0])[:2]
                if hi <= lo:
                    return lo
                seed = _hash(_canon([self.journal.run_id,
                                     self.journal._logical]))
                return lo + (int(seed[:8], 32) % (hi - lo))
            if key == "random.uuid":
                return _hash(_canon([self.journal.run_id,
                                     self.journal._logical]))[:26]
            if key == "random.bytes":
                n = args[0] if args else 0
                out = b""
                i = 0
                while len(out) < n:
                    out += hashlib.blake2b(
                        _canon([self.journal.run_id, self.journal._logical, i]),
                        digest_size=32).digest()
                    i += 1
                return out[:max(0, n)]
            if key.startswith("log."):
                return V.UNIT
            if key == "audit.entry":
                action, subject, detail = (list(args) + ["", "", ""])[:3]
                self.audit.record(f"program.{action}", subject=str(subject),
                                  detail={"detail": str(detail)},
                                  actor=self.actor)
                return V.UNIT
            raise Fault(
                "CANON-E0403",
                f"no handler registered for {key!r}", span,
                facts={"operation": key,
                       "registered": sorted(self.handlers)},
                )
        return fn(*args)

    def _from_recording(self, key: str, args: list, span, logical: int):
        src = self._source
        if src is None or self._cursor >= len(src.entries):
            if self.mode == SHADOW:
                self.divergences.append(Divergence(
                    self._cursor, "extra", expected=None, actual=key,
                    detail="the new version performed an effect the recording "
                           "does not contain"))
                return V.UNIT
            raise Fault(
                "CANON-E0803",
                "the journal ran out during replay", span,
                facts={"operation": key, "cursor": self._cursor,
                       "recorded": len(src.entries) if src else 0})

        entry = src.entries[self._cursor]
        self._cursor += 1

        if entry.op != key:
            div = Divergence(entry.seq, "operation", entry.op, key,
                             "a different effect was performed at this point")
            if self.mode == SHADOW:
                self.divergences.append(div)
                return V.UNIT
            raise Fault(
                "CANON-E0801",
                f"journal divergence: expected {entry.op!r}, performed {key!r}",
                span, facts=div.to_json())

        recorded_args = [V.to_json(a) for a in entry.args]
        actual_args = [V.to_json(a) for a in args]
        if recorded_args != actual_args:
            div = Divergence(entry.seq, "arguments", entry.args, args,
                             f"{key} was called with different arguments")
            if self.mode == SHADOW:
                self.divergences.append(div)
            else:
                raise Fault(
                    "CANON-E0802",
                    f"journal divergence: {key} arguments differ",
                    span, facts=div.to_json())

        if self.mode == SHADOW:
            self.journal.append(key, args, entry.result,
                                def_hash=self.current_def_hash,
                                actor=self.actor,
                                wall_millis=entry.wall_millis,
                                meta={"shadow": True})
        return entry.result

    # ------------------------------------------------------------------

    def ask(self, req):
        """Route an `ask` through the capability broker and the model runtime."""
        logical = self.journal.next_logical()
        decision = self.broker.check(
            "model.infer", [req.model, req.system], logical,
            self.classification_of.get("model.infer", "public"))
        if not decision.allowed:
            raise CapabilityDenied(
                "CANON-E0403",
                "capability 'model.infer' is not granted",
                req.span,
                facts={"operation": "model.infer", "model": req.model,
                       "reason": decision.reason,
                       "granted": self.granted()})

        if self.model_runtime is None:
            raise Fault(
                "CANON-E0403",
                "no model runtime is configured", req.span,
                facts={"model": req.model,
                       "hint": "construct the Ledger with model_runtime="})

        if self.mode in (REPLAY, SHADOW):
            recorded = self._recorded_ask(req)
            if recorded is not None:
                return recorded

        outcome = self.model_runtime.run(req, req.result_type,
                                         self.current_def_hash)
        self.journal.append(
            "model.infer",
            [req.model, req.system, [lbl for lbl, _ in req.inputs]],
            outcome.value,
            def_hash=self.current_def_hash, actor=self.actor,
            wall_millis=self._start_millis + logical,
            meta={"request_key": outcome.request_key,
                  "attempts": outcome.attempts,
                  "cached": outcome.cached,
                  "cost": str(outcome.cost),
                  "input_tokens": outcome.input_tokens,
                  "output_tokens": outcome.output_tokens})
        self.audit.record("model.invoked", subject=req.model,
                          detail={"attempts": outcome.attempts,
                                  "cost": str(outcome.cost),
                                  "def": self.current_def_hash},
                          actor=self.actor)
        return outcome.value

    def _recorded_ask(self, req):
        src = self._source
        if src is None or self._cursor >= len(src.entries):
            return None
        entry = src.entries[self._cursor]
        if entry.op != "model.infer":
            return None
        self._cursor += 1
        if self.mode == SHADOW:
            self.journal.append("model.infer", entry.args, entry.result,
                                def_hash=self.current_def_hash,
                                actor=self.actor,
                                wall_millis=entry.wall_millis,
                                meta={"shadow": True})
        return entry.result

    # ------------------------------------------------------------------

    def unconsumed(self) -> list:
        """Recorded effects the new version never performed."""
        if self._source is None:
            return []
        return [Divergence(e.seq, "missing", expected=e.op, actual=None,
                           detail="the recording contains an effect the new "
                                  "version did not perform")
                for e in self._source.entries[self._cursor:]]

    def report(self) -> dict:
        divs = list(self.divergences) + self.unconsumed()
        return {
            "mode": self.mode,
            "actor": self.actor,
            "effects": len(self.journal),
            "journal_head": self.journal.head(),
            "journal_intact": not self.journal.verify(),
            "audit_head": self.audit.head(),
            "audit_intact": not self.audit.verify(),
            "denials": list(self.broker.denials),
            "divergences": [d.to_json() for d in divs],
            "diverged": bool(divs),
            "calls": dict(self.broker.calls),
        }


# --------------------------------------------------------------------------
# Convenience constructors
# --------------------------------------------------------------------------

def open_ledger(grants, handlers=None, actor: str = "agent",
                model_provider=None, env=None, mode: str = LIVE,
                source: Optional[Journal] = None) -> Ledger:
    """
    Build a Ledger with a broker, a fresh journal and audit chain, and a model
    runtime wired to the given provider.
    """
    from .model import ModelRuntime, default_provider

    audit = AuditLog(actor=actor)
    broker = CapabilityBroker(audit=audit)
    if isinstance(grants, dict):
        for who, ops in grants.items():
            broker.grant(who, ops, reason="configured at startup")
    elif grants:
        broker.grant(actor, list(grants), reason="configured at startup")

    journal = Journal()
    mr = None
    if env is not None:
        mr = ModelRuntime(model_provider or default_provider(), env,
                          journal=None)

    led = Ledger(broker=broker, journal=journal, audit=audit, mode=mode,
                 handlers=handlers, actor=actor, model_runtime=mr)
    led._source = source
    audit.record("run.started", subject=journal.run_id,
                 detail={"mode": mode, "grants": broker.granted_operations()})
    return led
