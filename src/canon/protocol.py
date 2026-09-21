"""
The agent-computer interface.

A JSON-RPC surface over Atlas, the verifier and the gate, spoken over stdin and
stdout. It exists because the file-and-shell interface an agent normally works
through is a poor fit for the thing it actually needs to do:

  * Reading code by file wastes most of a context budget on locating things.
    `atlas.view` returns a projection sized to a stated token budget and says
    what it left out.

  * Editing by writing files leaves broken intermediate states behind when a
    change does not work out. `edit.propose` type-checks and diffs a change
    without applying it; `edit.commit` applies it atomically or not at all.

  * "What will this break" is normally answered by reading. `atlas.blast` and
    `atlas.capabilities` answer it from the graph.

Every response is a JSON object with `ok`. Errors are the same structured
diagnostics the compiler produces, not strings, so an agent handles a failed
edit the same way it handles a failed check.

The interface is read-only until `edit.commit`, and `edit.commit` refuses
anything that does not type-check. There is no method that writes unchecked
code.
"""

from __future__ import annotations

import json
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any, Optional

from . import values as V
from .atlas import Atlas, estimate_tokens
from .canonical import Hasher, short
from .diagnostics import Bag
from .interp import Budget, Fault, Interpreter
from .ledger import AuditLog, CapabilityBroker, Journal, Ledger
from .shadow import Authorization, evaluate_promotion, structural_diff
from .sources import Workspace, load, load_text
from .verifier import Verifier

PROTOCOL_VERSION = "0.1"


# --------------------------------------------------------------------------
# Session
# --------------------------------------------------------------------------

class Session:
    """
    One agent's working state: a loaded workspace, an atlas, and open
    proposals.

    Proposals are held by id rather than applied, so an agent can open several,
    compare their impact, and commit at most one.
    """

    def __init__(self, paths=None, actor: str = "agent"):
        self.actor = actor
        self.audit = AuditLog(actor=actor)
        self.workspace: Optional[Workspace] = None
        self.atlas: Optional[Atlas] = None
        self.hashes: dict = {}
        self.proposals: dict = {}
        self._next_id = 1
        if paths:
            self.open(paths)

    # ------------------------------------------------------------------

    def open(self, paths) -> dict:
        ws = load(list(paths))
        self.workspace = ws
        if ws.bag.has_errors:
            self.atlas = None
            return {"ok": False, "stage": "parse",
                    "diagnostics": ws.bag.to_json(),
                    "summary": ws.summary()}

        from .checker import check
        cr = check(ws.modules, ws.bag)
        if cr.bag.has_errors:
            self.atlas = None
            return {"ok": False, "stage": "check",
                    "diagnostics": cr.bag.to_json(),
                    "summary": ws.summary()}

        defs = Hasher().add_modules(cr.modules)
        self.hashes = {qn: di.hash for qn, di in defs.items()}
        self.atlas = Atlas(cr, defs, sources=ws.sources)
        self.audit.record("run.started", subject="session",
                          detail={"files": sorted(ws.sources)})
        return {"ok": True, "summary": ws.summary(),
                "stats": self.atlas.stats(),
                "warnings": [d.to_json() for d in cr.bag
                             if d.severity.value != "error"]}

    def require_atlas(self):
        if self.atlas is None:
            raise ProtocolError("no workspace is loaded, or it does not check",
                                {"hint": "call workspace.open first"})
        return self.atlas

    def new_id(self) -> str:
        i = self._next_id
        self._next_id += 1
        return f"prop-{i}"


class ProtocolError(Exception):
    def __init__(self, message, detail=None):
        self.message = message
        self.detail = detail or {}
        super().__init__(message)


# --------------------------------------------------------------------------
# Methods
# --------------------------------------------------------------------------

def _need(params: dict, key: str):
    if key not in params:
        raise ProtocolError(f"missing required parameter {key!r}",
                            {"parameter": key, "given": sorted(params)})
    return params[key]


class Methods:
    """Every method the interface exposes. One method per public operation."""

    def __init__(self, session: Session):
        self.s = session

    # -- workspace -------------------------------------------------------

    def workspace_open(self, p) -> dict:
        return self.s.open(_need(p, "paths"))

    def workspace_stats(self, p) -> dict:
        return {"ok": True, **self.s.require_atlas().stats()}

    def workspace_describe(self, p) -> dict:
        """What this interface can do, for an agent that has not seen it."""
        return {"ok": True, "protocol": PROTOCOL_VERSION,
                "methods": sorted(DISPATCH),
                "notes": [
                    "Editing is transactional: propose, inspect impact, then "
                    "commit or abort. Nothing is written until commit.",
                    "atlas.view takes a token budget and reports what it "
                    "omitted rather than truncating silently.",
                    "Every error is a structured diagnostic with a code, "
                    "facts and, where possible, concrete repairs."]}

    # -- atlas -----------------------------------------------------------

    def atlas_search(self, p) -> dict:
        hits = self.s.require_atlas().search(_need(p, "query"),
                                             p.get("limit", 20))
        return {"ok": True, "results": [h.summary() for h in hits]}

    def atlas_show(self, p) -> dict:
        atlas = self.s.require_atlas()
        node = atlas.resolve(_need(p, "target"))
        if node is None:
            return _unknown(p["target"], atlas)
        detail = p.get("detail", "full")
        return {"ok": True, "definition": node.to_json(),
                "detail": detail, "text": atlas.render(node, detail)}

    def atlas_view(self, p) -> dict:
        atlas = self.s.require_atlas()
        focus = _need(p, "focus")
        return {"ok": True,
                **atlas.view(focus, p.get("budget", 4000),
                             p.get("include_dependents", True))}

    def atlas_callers(self, p) -> dict:
        atlas = self.s.require_atlas()
        return {"ok": True,
                "callers": atlas.callers(_need(p, "target"),
                                         p.get("transitive", False))}

    def atlas_calls(self, p) -> dict:
        atlas = self.s.require_atlas()
        return {"ok": True,
                "calls": atlas.calls(_need(p, "target"),
                                     p.get("transitive", False))}

    def atlas_blast(self, p) -> dict:
        r = self.s.require_atlas().blast_radius(_need(p, "target"))
        if "error" in r:
            return {"ok": False, "error": r["error"]}
        return {"ok": True, **r}

    def atlas_capabilities(self, p) -> dict:
        r = self.s.require_atlas().capabilities(_need(p, "target"))
        if "error" in r:
            return {"ok": False, "error": r["error"]}
        return {"ok": True, **r}

    def atlas_effect_users(self, p) -> dict:
        return {"ok": True,
                "definitions": self.s.require_atlas()
                .effect_users(_need(p, "effect"))}

    def atlas_unverified(self, p) -> dict:
        return {"ok": True, **self.s.require_atlas().unverified()}

    # -- editing ---------------------------------------------------------

    def edit_propose(self, p) -> dict:
        atlas = self.s.require_atlas()
        changes = _need(p, "changes")
        if not isinstance(changes, dict) or not changes:
            raise ProtocolError("changes must be a non-empty map of "
                                "filename to complete new text",
                                {"given": type(changes).__name__})
        proposal = atlas.propose(changes, self.s.actor)
        pid = self.s.new_id()
        self.s.proposals[pid] = proposal
        impact = proposal.impact()
        self.s.audit.record(
            "definition.added" if proposal.ok else "verification.failed",
            subject=pid,
            detail={"files": sorted(changes), "ok": proposal.ok})
        return {"ok": proposal.ok, "proposal": pid, "impact": impact,
                "diagnostics": proposal.diagnostics() if not proposal.ok else []}

    def edit_impact(self, p) -> dict:
        proposal = self._proposal(_need(p, "proposal"))
        return {"ok": proposal.ok, "impact": proposal.impact()}

    def edit_canonical(self, p) -> dict:
        proposal = self._proposal(_need(p, "proposal"))
        if not proposal.ok:
            return {"ok": False, "diagnostics": proposal.diagnostics()}
        return {"ok": True, "sources": proposal.canonical()}

    def edit_commit(self, p) -> dict:
        pid = _need(p, "proposal")
        proposal = self._proposal(pid)
        if not proposal.ok:
            return {"ok": False,
                    "error": "this proposal does not check and cannot be "
                             "committed",
                    "diagnostics": proposal.diagnostics()}
        new_atlas = proposal.commit()
        self.s.atlas = new_atlas
        self.s.hashes = {qn: di.hash for qn, di in new_atlas.defs.items()}
        self.s.workspace.sources.update(proposal.changes)
        written = {}
        if p.get("write", False):
            for name, text in proposal.canonical().items():
                with open(name, "w", encoding="utf-8", newline="\n") as f:
                    f.write(text)
                written[name] = len(text)
        self.s.proposals.pop(pid, None)
        self.s.audit.record("promotion.approved", subject=pid,
                            detail={"files": sorted(proposal.changes),
                                    "written": sorted(written)})
        return {"ok": True, "committed": pid, "written": written,
                "stats": new_atlas.stats()}

    def edit_abort(self, p) -> dict:
        pid = _need(p, "proposal")
        proposal = self._proposal(pid)
        proposal.abort()
        self.s.proposals.pop(pid, None)
        return {"ok": True, "aborted": pid}

    def _proposal(self, pid):
        if pid not in self.s.proposals:
            raise ProtocolError(f"no open proposal {pid!r}",
                                {"open": sorted(self.s.proposals)})
        return self.s.proposals[pid]

    # -- verification ----------------------------------------------------

    def verify_run(self, p) -> dict:
        atlas = self.s.require_atlas()
        v = Verifier(atlas.cr, seed=p.get("seed", "canon"),
                     runs=p.get("runs", 40), hashes=self.s.hashes)
        report = v.verify_all(p.get("only"))
        self.s.audit.record(
            "verification.passed" if report.ok else "verification.failed",
            subject=p.get("only") and ",".join(p["only"]) or "workspace",
            detail={"seed": report.seed,
                    "failed": [f.qualname for f in report.failures()]})
        return {"ok": report.ok, **report.to_json()}

    def verify_tests(self, p) -> dict:
        from .interp import run_tests
        atlas = self.s.require_atlas()
        audit = AuditLog(actor=self.s.actor)
        broker = CapabilityBroker(audit=audit)
        broker.grant(self.s.actor, p.get("grants", ["*"]),
                     reason="declared test run")
        led = Ledger(broker=broker, audit=audit, actor=self.s.actor)
        results = run_tests(atlas.cr, led)
        passed = [r for r in results if r.get("passed")]
        return {"ok": len(passed) == len(results),
                "total": len(results), "passed": len(passed),
                "results": results}

    # -- promotion -------------------------------------------------------

    def gate_evaluate(self, p) -> dict:
        atlas = self.s.require_atlas()
        pid = _need(p, "proposal")
        proposal = self._proposal(pid)
        if not proposal.ok:
            return {"ok": False, "decision": "block",
                    "diagnostics": proposal.diagnostics()}

        diff = structural_diff(atlas.cr, proposal.result,
                               atlas.defs, proposal.new_atlas.defs)
        auth = Authorization(
            actor=self.s.actor,
            intent_id=p.get("intent", ""),
            allowed_definitions=p.get("allowed_definitions", []),
            allowed_capabilities=p.get("allowed_capabilities", []),
            max_blast_radius=p.get("max_blast_radius", 10),
            allow_new_capabilities=p.get("allow_new_capabilities", False),
            allow_contract_changes=p.get("allow_contract_changes", False),
            allow_signature_changes=p.get("allow_signature_changes", False),
            allow_behaviour_change=p.get("allow_behaviour_change", True),
            require_verification=p.get("require_verification", True),
            max_classification=p.get("max_classification", "internal"))

        verification = None
        if auth.require_verification:
            v = Verifier(proposal.result, seed=p.get("seed", "canon"),
                         runs=p.get("runs", 30),
                         hashes={qn: di.hash
                                 for qn, di in proposal.new_atlas.defs.items()})
            targets = [c.qualname for c in diff.edited()]
            verification = v.verify_all(targets or None).to_json()

        from .shadow import differential
        targets = [c.qualname for c in diff.changed()
                   if c.kind == "changed" and c.qualname in proposal.result.env.fns]
        diff_report = differential(atlas.cr, proposal.result, targets,
                                   p.get("seed", "canon"),
                                   p.get("runs", 30)).to_json() \
            if targets else None

        decision = evaluate_promotion(diff, auth, verification, diff_report,
                                      None, self.s.audit)
        return {"ok": decision.decision != "block", **decision.to_json()}

    # -- execution -------------------------------------------------------

    def run_call(self, p) -> dict:
        atlas = self.s.require_atlas()
        fn = _need(p, "function")
        args = [V.from_json(a) for a in p.get("arguments", [])]

        audit = AuditLog(actor=self.s.actor)
        broker = CapabilityBroker(audit=audit)
        grants = p.get("grants")
        if grants:
            broker.grant(self.s.actor, grants, reason="explicit run grant")
        led = Ledger(broker=broker, audit=audit, actor=self.s.actor)

        # Effects are satisfied with generated values unless the caller turns
        # simulation off. An agent exploring what a function does has no real
        # environment to run against, and a run that faults on the first
        # effect tells it nothing. The response says plainly that the effects
        # were simulated, so a result is never mistaken for a real one.
        simulate = p.get("simulate", True)
        if simulate:
            _install_simulated_effects(led, atlas.cr,
                                       seed=p.get("seed", "canon"))

        budget = Budget(steps=p.get("steps", 1_000_000),
                        io=p.get("io", 100))
        it = Interpreter(atlas.cr, led, budget, self.s.hashes)
        try:
            value = it.call(fn, args)
        except Fault as f:
            return {"ok": False, "fault": f.to_json(),
                    "simulated": simulate,
                    "journal": [e.to_json() for e in led.journal],
                    "cost": budget.snapshot(),
                    "denials": led.broker.denials}
        return {"ok": True, "result": V.to_json(value),
                "rendered": V.show(value),
                "simulated": simulate,
                "journal": [e.to_json() for e in led.journal],
                "cost": budget.snapshot(),
                "denials": led.broker.denials}

    # -- audit -----------------------------------------------------------

    def audit_tail(self, p) -> dict:
        n = p.get("limit", 40)
        records = [r.to_json() for r in self.s.audit.records[-n:]]
        return {"ok": True, "records": records,
                "head": self.s.audit.head(),
                "intact": not self.s.audit.verify()}


DISPATCH = {
    "workspace.open": "workspace_open",
    "workspace.stats": "workspace_stats",
    "workspace.describe": "workspace_describe",
    "atlas.search": "atlas_search",
    "atlas.show": "atlas_show",
    "atlas.view": "atlas_view",
    "atlas.callers": "atlas_callers",
    "atlas.calls": "atlas_calls",
    "atlas.blast": "atlas_blast",
    "atlas.capabilities": "atlas_capabilities",
    "atlas.effectUsers": "atlas_effect_users",
    "atlas.unverified": "atlas_unverified",
    "edit.propose": "edit_propose",
    "edit.impact": "edit_impact",
    "edit.canonical": "edit_canonical",
    "edit.commit": "edit_commit",
    "edit.abort": "edit_abort",
    "verify.run": "verify_run",
    "verify.tests": "verify_tests",
    "gate.evaluate": "gate_evaluate",
    "run.call": "run_call",
    "audit.tail": "audit_tail",
}


def _install_simulated_effects(led: Ledger, cr, seed: str = "canon"):
    """
    Register a handler for every declared effect operation that answers with
    a generated value of the operation's result type.

    Values are derived from the operation and its arguments, so the same call
    gives the same answer within a run and across runs. A simulated run is
    therefore reproducible, which is what makes it worth showing an agent.
    """
    from .checker import Checker
    from .verifier import Generator, Source

    checker = Checker()
    checker.env = cr.env

    for eff_name, ei in cr.env.effects.items():
        for op_name, op in ei.ops.items():
            key = f"{eff_name}.{op_name}"
            result_type = checker.resolve_type(op.result, None)

            def handler(*args, _t=result_type, _k=key):
                gen = Generator(cr.env,
                                Source(f"{seed}/{_k}/{V.value_hash(args)}"))
                gen._cr = cr
                return gen.generate(_t, 6)

            led.handle(key, handler)


def _unknown(target, atlas) -> dict:
    """
    Report an unknown definition with the nearest real ones.

    Edit distance rather than substring search: an agent's misses are
    misspellings and half-remembered names far more often than they are
    prefixes, and `cart_totl` shares no substring with `cart_total`.
    """
    from .checker import _closest

    near = _closest(target, atlas.nodes.keys(), limit=5)
    if not near:
        near = [n.qualname for n in atlas.search(target, 5)]
    return {"ok": False, "error": f"unknown definition {target!r}",
            "suggestions": near}


# --------------------------------------------------------------------------
# Transport
# --------------------------------------------------------------------------

def handle(session: Session, request: dict) -> dict:
    """Dispatch one request. Never raises; failures come back as responses."""
    rid = request.get("id")
    method = request.get("method", "")
    params = request.get("params") or {}

    if method not in DISPATCH:
        return {"id": rid, "ok": False,
                "error": f"unknown method {method!r}",
                "methods": sorted(DISPATCH)}

    methods = Methods(session)
    fn = getattr(methods, DISPATCH[method])
    try:
        result = fn(params)
    except ProtocolError as pe:
        return {"id": rid, "ok": False, "error": pe.message,
                "detail": pe.detail}
    except Fault as f:
        return {"id": rid, "ok": False, "fault": f.to_json()}
    except Exception as e:  # a bug here must not kill the session
        return {"id": rid, "ok": False,
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc().splitlines()[-6:]}
    out = {"id": rid}
    out.update(result)
    return out


def serve(stdin=None, stdout=None, paths=None, actor: str = "agent") -> int:
    """
    Run the interface over a stream of newline-delimited JSON requests.

    One request per line, one response per line. Line-delimited rather than
    framed so a session can be driven by hand from a terminal when something
    needs debugging.
    """
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    session = Session(paths, actor)

    banner = {"protocol": PROTOCOL_VERSION, "methods": sorted(DISPATCH),
              "ok": True}
    if paths:
        banner["workspace"] = session.workspace.summary() \
            if session.workspace else {}
    stdout.write(json.dumps(banner) + "\n")
    stdout.flush()

    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as e:
            stdout.write(json.dumps(
                {"ok": False, "error": f"invalid JSON: {e}"}) + "\n")
            stdout.flush()
            continue
        if request.get("method") == "session.close":
            stdout.write(json.dumps({"id": request.get("id"), "ok": True,
                                     "closed": True}) + "\n")
            stdout.flush()
            return 0
        response = handle(session, request)
        stdout.write(json.dumps(response, default=str) + "\n")
        stdout.flush()
    return 0
