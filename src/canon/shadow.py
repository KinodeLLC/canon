"""
behavioural diffing, shadow execution, the promotion gate.

this answers the thing a promotion decision actually turns on, which is what
would change if this shipped. not whether it compiles and not whether the tests
pass, the specific difference in behaviour between what is running now and what
somebody is proposing.

three layers, cheapest first.

structural diff is which definitions changed by content hash and what the blast
radius of that is across the call graph. it costs nothing, it is a graph walk
over hashes.

differential run takes each changed function, generates inputs, and runs both
versions side by side under the same conditions, reporting any input where they
disagree along with both results. cheap, and it finds most real regressions.

shadow replay runs the new version against a recorded production journal.
effect results come out of the recording so nothing gets touched, and anything
the new version would have done differently gets collected.

then the gate takes all of that and compares it against whatever the change was
authorised to do. a change that only touches definitions inside its authorised
blast radius, picks up no new capabilities and diverges on nothing can go out
with nobody reading it. anything else gets escalated with the reason on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from . import values as V
from .canonical import Hasher, short
from .checker import CheckResult
from .interp import Budget, Fault, Interpreter
from .ledger import AuditLog, Journal, Ledger, SHADOW
from .verifier import Generator, Source, VerificationRuntime, shrink_args


# --------------------------------------------------------------------------
# Structural diff
# --------------------------------------------------------------------------

@dataclass
class DefChange:
    qualname: str
    kind: str                 # added | removed | changed | unchanged
    old_hash: str = ""
    new_hash: str = ""
    old_local: str = ""
    new_local: str = ""
    effects_added: list = field(default_factory=list)
    effects_removed: list = field(default_factory=list)
    intent_changed: bool = False
    contracts_changed: bool = False
    signature_changed: bool = False

    @property
    def body_changed(self) -> bool:
        """True when the definition itself was edited, not just a dependency."""
        return self.old_local != self.new_local

    def to_json(self) -> dict:
        return {"definition": self.qualname, "kind": self.kind,
                "old": self.old_hash, "new": self.new_hash,
                "body_changed": self.body_changed,
                "effects_added": self.effects_added,
                "effects_removed": self.effects_removed,
                "intent_changed": self.intent_changed,
                "contracts_changed": self.contracts_changed,
                "signature_changed": self.signature_changed}


@dataclass
class StructuralDiff:
    changes: list = field(default_factory=list)
    blast_radius: list = field(default_factory=list)
    capabilities_added: list = field(default_factory=list)
    capabilities_removed: list = field(default_factory=list)
    classifications_added: list = field(default_factory=list)

    def changed(self) -> list:
        return [c for c in self.changes if c.kind != "unchanged"]

    def edited(self) -> list:
        """Definitions whose own body changed, ignoring dependency churn."""
        return [c for c in self.changes
                if c.kind == "changed" and c.body_changed]

    def to_json(self) -> dict:
        return {"changes": [c.to_json() for c in self.changed()],
                "blast_radius": sorted(self.blast_radius),
                "capabilities_added": sorted(self.capabilities_added),
                "capabilities_removed": sorted(self.capabilities_removed),
                "classifications_added": sorted(self.classifications_added)}


def structural_diff(old: CheckResult, new: CheckResult,
                    old_hashes: Optional[dict] = None,
                    new_hashes: Optional[dict] = None) -> StructuralDiff:
    old_defs = old_hashes or {qn: di for qn, di
                              in Hasher().add_modules(old.modules).items()}
    new_defs = new_hashes or {qn: di for qn, di
                              in Hasher().add_modules(new.modules).items()}

    diff = StructuralDiff()
    names = sorted(set(old_defs) | set(new_defs))

    for qn in names:
        o = old_defs.get(qn)
        n = new_defs.get(qn)

        if o is None:
            diff.changes.append(DefChange(qn, "added", new_hash=n.hash,
                                          new_local=n.local_hash,
                                          effects_added=sorted(n.effects)))
            continue
        if n is None:
            diff.changes.append(DefChange(qn, "removed", old_hash=o.hash,
                                          old_local=o.local_hash,
                                          effects_removed=sorted(o.effects)))
            continue
        if o.hash == n.hash:
            diff.changes.append(DefChange(qn, "unchanged", o.hash, n.hash,
                                          o.local_hash, n.local_hash))
            continue

        change = DefChange(qn, "changed", o.hash, n.hash,
                           o.local_hash, n.local_hash,
                           effects_added=sorted(set(n.effects) - set(o.effects)),
                           effects_removed=sorted(set(o.effects) - set(n.effects)))
        of, nf = old.env.fns.get(qn), new.env.fns.get(qn)
        if of and nf:
            change.intent_changed = (of.decl.intent or "") != (nf.decl.intent or "")
            change.signature_changed = str(of.type) != str(nf.type)
            change.contracts_changed = (
                [_r(x) for x in of.decl.requires] != [_r(x) for x in nf.decl.requires]
                or [_r(x) for x in of.decl.ensures] != [_r(x) for x in nf.decl.ensures]
                or sorted(l.name for l in of.decl.laws)
                != sorted(l.name for l in nf.decl.laws))
        diff.changes.append(change)

    # Blast radius: everything that transitively depends on an edited
    # definition. This is what a reviewer would have to look at if they were
    # checking the change by hand.
    edited = {c.qualname for c in diff.changes
              if c.kind in ("changed", "removed", "added") and
              (c.kind != "changed" or c.body_changed)}
    reverse = {}
    for qn, di in new_defs.items():
        for dep in di.deps:
            reverse.setdefault(dep, set()).add(qn)
    radius = set(edited)
    frontier = list(edited)
    while frontier:
        cur = frontier.pop()
        for dependent in reverse.get(cur, ()):
            if dependent not in radius:
                radius.add(dependent)
                frontier.append(dependent)
    diff.blast_radius = sorted(radius)

    # Capability and classification deltas are measured over the definitions
    # this change actually touched, not over the whole program. A program-wide
    # union hides the case that matters: a function newly reaching a capability
    # that something *else* in the codebase already had. That is still a
    # privilege increase for this change, and the gate has to see it.
    # The delta is computed per definition and then unioned, not as one set
    # difference across all of them. Unioning first lets a definition that
    # already held a capability mask another definition newly gaining it, which
    # is exactly the privilege increase the gate exists to catch.
    touched = {c.qualname for c in diff.changes if c.kind != "unchanged"}
    caps_added, caps_removed, cls_added = set(), set(), set()
    for qn in touched:
        ofi, nfi = old.env.fns.get(qn), new.env.fns.get(qn)
        before = set(ofi.transitive) if ofi is not None else set()
        after = set(nfi.transitive) if nfi is not None else set()
        caps_added |= after - before
        caps_removed |= before - after
        cls_before = set(ofi.touches) if ofi is not None else set()
        cls_after = set(nfi.touches) if nfi is not None else set()
        cls_added |= cls_after - cls_before

    diff.capabilities_added = sorted(caps_added)
    diff.capabilities_removed = sorted(caps_removed)
    diff.classifications_added = sorted(cls_added)

    return diff


def _r(e) -> str:
    from .canonical import Printer
    try:
        return Printer().expr(e)
    except Exception:
        return "?"


# --------------------------------------------------------------------------
# Differential execution
# --------------------------------------------------------------------------

@dataclass
class Disagreement:
    qualname: str
    args: list
    old_result: Any = None
    new_result: Any = None
    old_fault: Optional[str] = None
    new_fault: Optional[str] = None
    old_effects: list = field(default_factory=list)
    new_effects: list = field(default_factory=list)

    def summary(self) -> str:
        if self.old_fault or self.new_fault:
            return (f"{self.old_fault or V.show(self.old_result)} -> "
                    f"{self.new_fault or V.show(self.new_result)}")
        if self.old_effects != self.new_effects:
            return (f"effects {self.old_effects} -> {self.new_effects}")
        return f"{V.show(self.old_result)} -> {V.show(self.new_result)}"

    def to_json(self) -> dict:
        return {"function": self.qualname,
                "arguments": [V.to_json(a) for a in self.args],
                "rendered": [V.show(a) for a in self.args],
                "old_result": V.to_json(self.old_result)
                if self.old_result is not None else None,
                "new_result": V.to_json(self.new_result)
                if self.new_result is not None else None,
                "old_fault": self.old_fault, "new_fault": self.new_fault,
                "old_effects": self.old_effects,
                "new_effects": self.new_effects,
                "summary": self.summary()}


@dataclass
class DifferentialReport:
    checked: list = field(default_factory=list)
    skipped: dict = field(default_factory=dict)
    disagreements: list = field(default_factory=list)
    runs: int = 0
    seed: str = ""

    @property
    def identical(self) -> bool:
        return not self.disagreements

    def to_json(self) -> dict:
        return {"identical": self.identical, "runs": self.runs,
                "seed": self.seed, "checked": sorted(self.checked),
                "skipped": self.skipped,
                "disagreements": [d.to_json() for d in self.disagreements]}


def differential(old: CheckResult, new: CheckResult, targets,
                 seed: str = "canon", runs: int = 40,
                 shrink: bool = True) -> DifferentialReport:
    """
    Run both versions on identical generated inputs and report disagreement.

    Both sides get an effect runtime seeded from the arguments alone, so the
    two runs see the same world. A difference in outcome is therefore a
    difference in the code, not in the environment.
    """
    rep = DifferentialReport(seed=seed)

    for qn in sorted(set(targets)):
        ofi = old.env.fns.get(qn)
        nfi = new.env.fns.get(qn)
        if ofi is None or nfi is None:
            rep.skipped[qn] = "not present in both versions"
            continue
        if ofi.tparams or nfi.tparams:
            rep.skipped[qn] = "generic"
            continue
        if str(ofi.type.params) != str(nfi.type.params):
            rep.skipped[qn] = "parameter types differ; inputs are not comparable"
            continue

        rep.checked.append(qn)
        gen = Generator(old.env, Source(f"{seed}/{qn}"))
        gen._cr = old

        for run in range(runs):
            args = [gen.generate(pt, 2 + (run % 10)) for pt in ofi.type.params]
            if not _in_domain(old, ofi, args):
                continue
            rep.runs += 1

            o = _run_one(old, ofi, args, seed)
            n = _run_one(new, nfi, args, seed)
            if _agree(o, n):
                continue

            if shrink:
                def still_differs(trial):
                    if not _in_domain(old, ofi, trial):
                        return False
                    return not _agree(_run_one(old, ofi, trial, seed),
                                      _run_one(new, nfi, trial, seed))
                args = shrink_args(args, still_differs)
                o = _run_one(old, ofi, args, seed)
                n = _run_one(new, nfi, args, seed)

            rep.disagreements.append(Disagreement(
                qualname=qn, args=args,
                old_result=o["result"], new_result=n["result"],
                old_fault=o["fault"], new_fault=n["fault"],
                old_effects=o["effects"], new_effects=n["effects"]))
            break

    return rep


def _run_one(cr: CheckResult, fi, args, seed) -> dict:
    src = Source(f"{seed}/world/{V.value_hash(tuple(args))}")
    rt = VerificationRuntime(cr.env, src)
    budget = Budget(steps=2_000_000, io=5000)
    it = Interpreter(cr, rt, budget, enforce_contracts=False)
    try:
        result = it.call_fn(fi, list(args))
        return {"result": result, "fault": None,
                "effects": [k for k, _ in rt.performed]}
    except Fault as f:
        return {"result": None, "fault": f.code,
                "effects": [k for k, _ in rt.performed]}


def _agree(a, b) -> bool:
    if a["fault"] != b["fault"]:
        return False
    if a["fault"] is None and V.compare(a["result"], b["result"]) != 0:
        return False
    return a["effects"] == b["effects"]


def _in_domain(cr: CheckResult, fi, args) -> bool:
    """Whether the inputs satisfy the function's preconditions."""
    if not fi.decl.requires:
        return True
    it = Interpreter(cr, VerificationRuntime(cr.env, Source("domain")),
                     Budget(steps=200_000), enforce_contracts=False)
    frame = it.globals.child(
        {p.name: a for p, a in zip(fi.decl.params, args)})
    try:
        return all(it.eval(r, frame) is True for r in fi.decl.requires)
    except Fault:
        return False


# --------------------------------------------------------------------------
# Shadow replay
# --------------------------------------------------------------------------

@dataclass
class ShadowReport:
    entry_point: str = ""
    replayed: int = 0
    divergences: list = field(default_factory=list)
    fault: Optional[dict] = None
    old_result: Any = None
    new_result: Any = None

    @property
    def clean(self) -> bool:
        return not self.divergences and self.fault is None

    def to_json(self) -> dict:
        return {"entry_point": self.entry_point, "clean": self.clean,
                "replayed": self.replayed, "divergences": self.divergences,
                "fault": self.fault,
                "old_result": V.to_json(self.old_result)
                if self.old_result is not None else None,
                "new_result": V.to_json(self.new_result)
                if self.new_result is not None else None}


def shadow_replay(new: CheckResult, journal: Journal, entry: str,
                  args: list, hashes=None,
                  expected_result: Any = None) -> ShadowReport:
    """
    Run the new version against a recorded journal.

    Effects are answered from the recording rather than performed, so this is
    safe to run against production traffic. What it produces is the set of
    points where the new version would have behaved differently.
    """
    rep = ShadowReport(entry_point=entry)
    audit = AuditLog(actor="shadow")
    led = Ledger(mode=SHADOW, audit=audit, actor="shadow")
    led.broker.grant("shadow", ["*"], reason="shadow evaluation is inert")
    led._source = journal

    it = Interpreter(new, led, Budget(steps=5_000_000), hashes)
    try:
        rep.new_result = it.call(entry, list(args))
    except Fault as f:
        rep.fault = f.to_json()

    report = led.report()
    rep.divergences = report["divergences"]
    rep.replayed = led._cursor
    rep.old_result = expected_result

    if expected_result is not None and rep.new_result is not None:
        if V.compare(expected_result, rep.new_result) != 0:
            rep.divergences.append({
                "seq": -1, "kind": "result",
                "detail": "the entry point returned a different value",
                "expected": V.to_json(expected_result),
                "actual": V.to_json(rep.new_result)})
    return rep


# --------------------------------------------------------------------------
# Promotion gate
# --------------------------------------------------------------------------

@dataclass
class Authorization:
    """
    What a change was permitted to do, set before the agent started work.

    The gate compares the change against this rather than against a reviewer's
    judgement, which is what makes the decision reproducible and auditable.
    """
    actor: str = "agent"
    intent_id: str = ""
    allowed_definitions: list = field(default_factory=list)
    allowed_capabilities: list = field(default_factory=list)
    max_blast_radius: int = 10
    allow_new_capabilities: bool = False
    allow_contract_changes: bool = False
    allow_signature_changes: bool = False
    allow_behaviour_change: bool = True
    require_verification: bool = True
    max_classification: str = "internal"

    def covers_definition(self, qn: str) -> bool:
        if not self.allowed_definitions:
            return True
        for pat in self.allowed_definitions:
            if pat == qn or pat == "*":
                return True
            if pat.endswith(".*") and qn.startswith(pat[:-1]):
                return True
        return False

    def to_json(self) -> dict:
        return {"actor": self.actor, "intent": self.intent_id,
                "allowed_definitions": self.allowed_definitions,
                "allowed_capabilities": self.allowed_capabilities,
                "max_blast_radius": self.max_blast_radius,
                "allow_new_capabilities": self.allow_new_capabilities,
                "allow_contract_changes": self.allow_contract_changes,
                "allow_signature_changes": self.allow_signature_changes,
                "allow_behaviour_change": self.allow_behaviour_change,
                "require_verification": self.require_verification,
                "max_classification": self.max_classification}


@dataclass
class Finding:
    code: str
    severity: str             # block | escalate | note
    message: str
    detail: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"code": self.code, "severity": self.severity,
                "message": self.message, "detail": self.detail}


@dataclass
class PromotionDecision:
    decision: str             # promote | escalate | block
    findings: list = field(default_factory=list)
    diff: Optional[dict] = None
    verification: Optional[dict] = None
    differential: Optional[dict] = None
    shadow: Optional[dict] = None

    @property
    def blocked(self) -> bool:
        return self.decision == "block"

    def to_json(self) -> dict:
        return {"decision": self.decision,
                "findings": [f.to_json() for f in self.findings],
                "diff": self.diff, "verification": self.verification,
                "differential": self.differential, "shadow": self.shadow}

    def render(self) -> str:
        lines = [f"decision: {self.decision.upper()}"]
        for f in self.findings:
            lines.append(f"  [{f.severity}] {f.code}: {f.message}")
        return "\n".join(lines)


def evaluate_promotion(diff: StructuralDiff, auth: Authorization,
                       verification=None, differential_report=None,
                       shadow_reports=None, audit: Optional[AuditLog] = None
                       ) -> PromotionDecision:
    """
    Decide whether a change may be promoted, and say exactly why.

    Every finding names the rule it came from. An escalation that cannot
    explain itself is indistinguishable from an outage, so the reasons matter
    as much as the verdict.
    """
    findings = []

    # Scope is checked against definitions whose own body was edited, plus
    # additions and removals. A definition whose deep hash moved only because
    # a dependency changed was not edited by anyone, and treating that as an
    # out-of-scope modification would make any authorisation narrower than the
    # whole call graph unusable.
    authored = [c for c in diff.changes
                if c.kind in ("added", "removed")
                or (c.kind == "changed" and c.body_changed)]
    for change in authored:
        if not auth.covers_definition(change.qualname):
            findings.append(Finding(
                "CANON-E0905", "block",
                f"{change.qualname} was modified but is outside the "
                f"authorised scope",
                {"definition": change.qualname,
                 "kind": change.kind,
                 "authorised": auth.allowed_definitions}))

    if len(diff.blast_radius) > auth.max_blast_radius:
        findings.append(Finding(
            "CANON-E0905", "escalate",
            f"blast radius of {len(diff.blast_radius)} definitions exceeds "
            f"the authorised {auth.max_blast_radius}",
            {"blast_radius": diff.blast_radius[:40],
             "limit": auth.max_blast_radius}))

    if diff.capabilities_added:
        unauthorised = [c for c in diff.capabilities_added
                        if c not in auth.allowed_capabilities]
        if unauthorised and not auth.allow_new_capabilities:
            findings.append(Finding(
                "CANON-E0903", "block",
                "the change introduces capabilities it was not authorised for",
                {"capabilities": unauthorised,
                 "authorised": auth.allowed_capabilities}))
        elif unauthorised:
            findings.append(Finding(
                "CANON-E0903", "escalate",
                "the change introduces new capabilities",
                {"capabilities": unauthorised}))

    from . import types as TY
    for cls in diff.classifications_added:
        if TY.class_rank(cls) > TY.class_rank(auth.max_classification):
            findings.append(Finding(
                "CANON-E0903", "block",
                f"the change reaches {cls} data, above the authorised "
                f"{auth.max_classification}",
                {"classification": cls,
                 "authorised": auth.max_classification}))

    for change in diff.changed():
        if change.signature_changed and not auth.allow_signature_changes:
            findings.append(Finding(
                "CANON-E0903", "escalate",
                f"{change.qualname} changed signature",
                {"definition": change.qualname}))
        if change.contracts_changed and not auth.allow_contract_changes:
            findings.append(Finding(
                "CANON-E0903", "escalate",
                f"{change.qualname} changed its contracts, which weakens the "
                f"guarantee anything downstream relies on",
                {"definition": change.qualname}))
        if change.intent_changed:
            findings.append(Finding(
                "CANON-W0005", "note",
                f"{change.qualname} changed its stated intent",
                {"definition": change.qualname}))

    if auth.require_verification:
        if verification is None:
            findings.append(Finding(
                "CANON-E0903", "block",
                "verification is required but was not run", {}))
        elif not verification.get("ok", False):
            failed = [f["function"] for f in verification.get("functions", [])
                      if not f.get("ok")]
            findings.append(Finding(
                "CANON-E0904", "block",
                f"verification failed for {len(failed)} function(s)",
                {"functions": failed[:20]}))

    if differential_report is not None and not differential_report.get("identical", True):
        sev = "escalate" if auth.allow_behaviour_change else "block"
        ds = differential_report.get("disagreements", [])
        findings.append(Finding(
            "CANON-E0904", sev,
            f"behaviour differs on {len(ds)} generated input(s)",
            {"disagreements": ds[:10]}))

    for sr in (shadow_reports or []):
        if not sr.get("clean", True):
            sev = "escalate" if auth.allow_behaviour_change else "block"
            findings.append(Finding(
                "CANON-E0904", sev,
                f"shadow replay of {sr.get('entry_point')} diverged from "
                f"recorded production behaviour",
                {"divergences": sr.get("divergences", [])[:10]}))

    if any(f.severity == "block" for f in findings):
        decision = "block"
    elif any(f.severity == "escalate" for f in findings):
        decision = "escalate"
    else:
        decision = "promote"

    result = PromotionDecision(
        decision=decision, findings=findings, diff=diff.to_json(),
        verification=verification, differential=differential_report,
        shadow=shadow_reports)

    if audit is not None:
        action = {"promote": "promotion.approved",
                  "escalate": "promotion.requested",
                  "block": "promotion.blocked"}[decision]
        audit.record(action, subject=auth.intent_id or auth.actor,
                     detail={"decision": decision,
                             "findings": [f.to_json() for f in findings],
                             "changed": [c.qualname for c in diff.changed()],
                             "blast_radius": len(diff.blast_radius)},
                     actor=auth.actor)

    return result


# --------------------------------------------------------------------------
# End-to-end
# --------------------------------------------------------------------------

def gate(old: CheckResult, new: CheckResult, auth: Authorization,
         seed: str = "canon", runs: int = 40,
         journals=None, hashes_old=None, hashes_new=None,
         audit: Optional[AuditLog] = None,
         verification=None) -> PromotionDecision:
    """Run the full pipeline and return a decision."""
    diff = structural_diff(old, new, hashes_old, hashes_new)
    targets = [c.qualname for c in diff.changed()
               if c.kind == "changed" and c.qualname in new.env.fns]
    diff_report = differential(old, new, targets, seed, runs).to_json() \
        if targets else None

    shadow_reports = []
    for j in (journals or []):
        r = shadow_replay(new, j["journal"], j["entry"], j["args"],
                          hashes_new, j.get("expected"))
        shadow_reports.append(r.to_json())

    return evaluate_promotion(diff, auth, verification, diff_report,
                              shadow_reports, audit)
