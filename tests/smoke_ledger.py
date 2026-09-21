"""
Ledger and model-primitive smoke test.

Exercises the properties the whole design rests on: model output is typed and
contract-checked, effects are capability-gated, the journal is tamper-evident,
a recorded run replays exactly, and a changed version can be shadow-diffed
against real recorded traffic without touching anything.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from canon import Hasher, parse  # noqa: E402
from canon import values as V  # noqa: E402
from canon.checker import check  # noqa: E402
from canon.interp import Budget, Fault, Interpreter  # noqa: E402
from canon.ledger import (  # noqa: E402
    AuditLog, CapabilityBroker, Journal, Ledger, open_ledger,
)
from canon.model import DeterministicProvider, ModelRuntime  # noqa: E402

SRC = r'''
module triage

enum Severity {
  | Low
  | Medium
  | High
  | Critical
}

record Ticket {
  id: Text
  body: Text
  classify body personal
}

record Assessment {
  severity: Severity
  summary: Text
  needs_human: Bool
}

effect tickets {
  store(id: Text, severity: Text) -> Unit
  idempotent load(id: Text) -> Option<Text>
}

fn assess(t: Ticket) -> Assessment
  intent "Judge how urgent a support ticket is."
  uses model.infer, model.judge
  ensures result.summary != ""
  cost tokens 4000, io 2
{
  ask Assessment from claude.opus {
    system "Assess the urgency of this support ticket."
    input body: t.body
    grounded_in t.body
    retries 3 on contract_violation, type_error, grounding_failure
    max_tokens 512
  }
}

fn record_assessment(t: Ticket, a: Assessment) -> Text
  intent "Persist an assessment and return the stored severity label."
  uses tickets.store, audit.entry
  ensures result != ""
{
  let label = match a.severity {
    case Low => "low"
    case Medium => "medium"
    case High => "high"
    case Critical => "critical"
  }
  do tickets.store(t.id, label)
  do audit.entry("ticket.assessed", t.id, label)
  label
}
'''

# A changed version of record_assessment: the label for High differs. This is
# the kind of edit a shadow deployment has to catch.
CHANGED = SRC.replace('case High => "high"', 'case High => "urgent"')


def build(src):
    mod, bag = parse(src, "triage.canon")
    if bag.has_errors:
        print(bag.render(src))
        raise SystemExit(1)
    res = check([mod], bag)
    errs = [d for d in res.bag if d.severity.value == "error"]
    if errs:
        for d in errs:
            print(d.render(src))
        raise SystemExit(1)
    hashes = {qn: di.hash for qn, di in Hasher().add_modules([mod]).items()}
    return res, hashes


def ticket(tid="t-1", body="The checkout page returns a 500 for every card payment."):
    return V.Record("Ticket", (("id", tid), ("body", body)))


def main():
    res, hashes = build(SRC)
    failures = []

    def case(name, fn):
        try:
            print(f"  ok    {name}: {fn()}")
        except AssertionError as ae:
            failures.append(name)
            print(f"  FAIL  {name}: {ae}")

    stored = {}

    def make_ledger(mode="live", source=None, grants=None, actor="agent"):
        led = open_ledger(
            grants if grants is not None
            else ["model.infer", "model.judge", "tickets.*", "audit.entry"],
            actor=actor, env=res.env,
            model_provider=DeterministicProvider("kinode"),
            mode=mode, source=source)
        led.handle("tickets.store",
                   lambda i, s: (stored.__setitem__(i, s), V.UNIT)[1])
        led.handle("tickets.load",
                   lambda i: V.some(stored[i]) if i in stored else V.NONE)
        led.classification_of["model.infer"] = "personal"
        return led

    # ---------------------------------------------------------------
    print("the ask primitive")

    def t_typed():
        led = make_ledger()
        it = Interpreter(res, led, Budget(), hashes)
        a = it.call("assess", [ticket()])
        assert isinstance(a, V.Record) and a.type_name == "Assessment", V.show(a)
        sev = a.get("severity")
        assert isinstance(sev, V.Variant) and sev.type_name == "Severity", V.show(sev)
        assert isinstance(a.get("needs_human"), bool)
        return f"{V.show(sev)} / needs_human={a.get('needs_human')}"
    case("a model answer arrives as a typed value", t_typed)

    def t_deterministic():
        a = Interpreter(res, make_ledger(), Budget(), hashes).call("assess", [ticket()])
        b = Interpreter(res, make_ledger(), Budget(), hashes).call("assess", [ticket()])
        assert V.compare(a, b) == 0, f"{V.show(a)} != {V.show(b)}"
        return "identical across runs"
    case("the same ask with the same inputs is reproducible", t_deterministic)

    def t_contract_retry():
        # An obligation the deterministic provider cannot satisfy: the summary
        # must equal a fixed string. The runtime should exhaust its retries and
        # fail with the contract as the reason, not with a parse error.
        bad = SRC.replace('ensures result.summary != ""',
                          'ensures result.summary == "impossible-sentinel"')
        r2, h2 = build(bad)
        led = make_ledger()
        it = Interpreter(r2, led, Budget(), h2)
        try:
            it.call("assess", [ticket()])
        except Fault as f:
            assert f.code == "CANON-E0502", f.code
            assert f.facts.get("reason") == "contract_violation", f.facts
            assert f.facts.get("attempts") == 4, f.facts.get("attempts")
            return (f"{f.code} after {f.facts['attempts']} attempts, "
                    f"reason={f.facts['reason']}")
        raise AssertionError("expected the contract to fail the ask")
    case("a model answer that breaks the contract is retried then refused",
         t_contract_retry)

    # ---------------------------------------------------------------
    print("\ncapabilities")

    def t_model_denied():
        led = make_ledger(grants=["tickets.*", "audit.entry"])
        it = Interpreter(res, led, Budget(), hashes)
        try:
            it.call("assess", [ticket()])
        except Fault as f:
            assert f.code == "CANON-E0403", f.code
            return f"{f.code}: {f.facts.get('operation')}"
        raise AssertionError("expected model.infer to be denied")
    case("an ungranted model call is denied", t_model_denied)

    def t_classification():
        # A grant limited to internal data must not cover a call the runtime
        # has classified as touching personal data.
        audit = AuditLog()
        broker = CapabilityBroker(audit=audit)
        broker.grant("agent", ["model.infer"], max_classification="internal")
        led = Ledger(broker=broker, audit=audit,
                     model_runtime=ModelRuntime(
                         DeterministicProvider(), res.env))
        led.classification_of["model.infer"] = "personal"
        it = Interpreter(res, led, Budget(), hashes)
        try:
            it.call("assess", [ticket()])
        except Fault as f:
            assert f.code == "CANON-E0403", f.code
            assert "personal" in f.facts.get("reason", ""), f.facts
            return "grant limited to internal data rejected a personal-data call"
        raise AssertionError("expected the classification limit to deny")
    case("a grant's data classification limit is enforced", t_classification)

    def t_call_ceiling():
        audit = AuditLog()
        broker = CapabilityBroker(audit=audit)
        broker.grant("agent", ["tickets.store", "audit.entry"], max_calls=1)
        led = Ledger(broker=broker, audit=audit)
        led.handle("tickets.store", lambda i, s: V.UNIT)
        it = Interpreter(res, led, Budget(), hashes)
        a = V.Record("Assessment", (("severity", V.Variant("High", (), "Severity")),
                                    ("summary", "s"), ("needs_human", True)))
        try:
            it.call("record_assessment", [ticket(), a])
        except Fault as f:
            assert f.code == "CANON-E0403", f.code
            return f"exhausted after {broker.calls.get('tickets.store', 0)} call(s)"
        raise AssertionError("expected the call ceiling to deny the second call")
    case("a grant's call ceiling is enforced", t_call_ceiling)

    # ---------------------------------------------------------------
    print("\njournal")

    def t_journal():
        led = make_ledger()
        it = Interpreter(res, led, Budget(), hashes)
        a = it.call("assess", [ticket()])
        it.call("record_assessment", [ticket(), a])
        problems = led.journal.verify()
        assert not problems, problems
        ops = led.journal.ops()
        assert ops == ["model.infer", "tickets.store", "audit.entry"], ops
        assert all(e.def_hash for e in led.journal), "entries lack a def hash"
        return f"{len(led.journal)} entries, chain intact, head={led.journal.head()[:10]}"
    case("every effect is journaled and attributed to a definition", t_journal)

    def t_tamper():
        led = make_ledger()
        it = Interpreter(res, led, Budget(), hashes)
        a = it.call("assess", [ticket()])
        it.call("record_assessment", [ticket(), a])
        # Alter a recorded argument, as a tamperer would.
        led.journal.entries[1].args = ["t-1", "low"]
        problems = led.journal.verify()
        assert problems, "tampering went undetected"
        assert problems[0]["problem"] == "entry hash mismatch", problems[0]
        return f"detected at seq {problems[0]['seq']} ({problems[0]['problem']})"
    case("an altered journal entry is detected", t_tamper)

    # ---------------------------------------------------------------
    print("\nreplay and shadow")

    def record_production():
        led = make_ledger()
        it = Interpreter(res, led, Budget(), hashes)
        a = it.call("assess", [ticket()])
        label = it.call("record_assessment", [ticket(), a])
        return led.journal, a, label

    def t_replay():
        source, a, label = record_production()
        led = make_ledger(mode="replay", source=source)
        it = Interpreter(res, led, Budget(), hashes)
        a2 = it.call("assess", [ticket()])
        label2 = it.call("record_assessment", [ticket(), a2])
        assert V.compare(a, a2) == 0, "replayed assessment differs"
        assert label == label2, f"{label} != {label2}"
        assert not led.report()["diverged"], led.report()["divergences"]
        return f"replayed {len(source)} effects with no divergence"
    case("a recorded run replays exactly", t_replay)

    high = V.Record("Assessment",
                    (("severity", V.Variant("High", (), "Severity")),
                     ("summary", "payment failures"), ("needs_human", True)))

    def record_known():
        """Record a run that definitely exercises the branch CHANGED alters."""
        led = make_ledger()
        it = Interpreter(res, led, Budget(), hashes)
        label = it.call("record_assessment", [ticket(), high])
        return led.journal, label

    def t_shadow():
        source, label = record_known()
        r2, h2 = build(CHANGED)
        led = make_ledger(mode="shadow", source=source)
        it = Interpreter(r2, led, Budget(), h2)
        label2 = it.call("record_assessment", [ticket(), high])
        rep = led.report()
        assert label == "high" and label2 == "urgent", f"{label} -> {label2}"
        assert rep["diverged"], "the shadow run reported no divergence"
        return (f"return value {label!r} -> {label2!r}, "
                f"{len(rep['divergences'])} effect divergence(s)")
    case("a behavioural change is visible in a shadow run", t_shadow)

    def t_shadow_effects():
        source, label = record_known()
        r2, h2 = build(CHANGED)
        led = make_ledger(mode="shadow", source=source)
        it = Interpreter(r2, led, Budget(), h2)
        it.call("record_assessment", [ticket(), high])
        rep = led.report()
        arg_divs = [d for d in rep["divergences"] if d["kind"] == "arguments"]
        assert arg_divs, f"expected an argument divergence: {rep['divergences']}"
        return (f"{len(arg_divs)} argument divergence(s): "
                f"{arg_divs[0]['expected']} -> {arg_divs[0]['actual']}")
    case("shadow diffing catches changed effect arguments", t_shadow_effects)

    def t_shadow_isolated():
        # A shadow run must not reach any real handler.
        source, _ = record_known()
        before = dict(stored)
        stored["t-1"] = "sentinel"
        r2, h2 = build(CHANGED)
        led = make_ledger(mode="shadow", source=source)
        it = Interpreter(r2, led, Budget(), h2)
        it.call("record_assessment", [ticket(), high])
        assert stored["t-1"] == "sentinel", \
            f"the shadow run mutated real state: {stored['t-1']}"
        stored.clear()
        stored.update(before)
        return "no handler was invoked"
    case("a shadow run cannot touch real state", t_shadow_isolated)

    # ---------------------------------------------------------------
    print("\naudit")

    def t_audit():
        led = make_ledger()
        it = Interpreter(res, led, Budget(), hashes)
        a = it.call("assess", [ticket()])
        it.call("record_assessment", [ticket(), a])
        problems = led.audit.verify()
        assert not problems, problems
        actions = [r.action for r in led.audit]
        for expected in ("run.started", "capability.granted", "model.invoked",
                         "program.ticket.assessed"):
            assert expected in actions, f"missing {expected!r} in {actions}"
        return f"{len(led.audit)} records, chain intact: {', '.join(sorted(set(actions)))}"
    case("governance events are recorded in a verifiable chain", t_audit)

    def t_denials_audited():
        led = make_ledger(grants=["tickets.*"])
        it = Interpreter(res, led, Budget(), hashes)
        try:
            it.call("assess", [ticket()])
        except Fault:
            pass
        denied = led.audit.by_action("capability.denied")
        assert denied, "a denial was not written to the audit log"
        return f"{len(denied)} denial(s) recorded"
    case("denied capability requests are audited", t_denials_audited)

    print("\nRESULT:", "pass" if not failures else f"FAIL ({failures})")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
