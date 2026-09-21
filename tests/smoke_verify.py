"""
Verification, differential execution, and the promotion gate.

Checks that the verifier finds genuine contract and law violations, shrinks
counterexamples to something readable, and that the promotion gate reaches the
right decision for safe, behaviour-changing, and out-of-scope edits.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from canon import Hasher, parse  # noqa: E402
from canon import values as V  # noqa: E402
from canon.checker import check  # noqa: E402
from canon.shadow import (  # noqa: E402
    Authorization, differential, evaluate_promotion, gate, structural_diff,
)
from canon.verifier import Verifier  # noqa: E402

SRC = r'''
module billing

record Money {
  amount: Int
  currency: Text
  invariant amount >= 0
}

record LineItem {
  sku: Text
  qty: Int
  unit: Money
  invariant qty >= 0
}

effect store {
  write(key: Text, value: Text) -> Unit
  read(key: Text) -> Option<Text>
}

fn line_total(l: LineItem) -> Int
  intent "The extended amount of a line item."
  requires l.qty >= 0
  ensures result >= 0
  law never_negative
{
  l.qty * l.unit.amount
}

fn subtotal(items: List<LineItem>) -> Int
  intent "The sum of all line totals."
  ensures result >= 0
  law never_negative
  law order_independent
{
  List.fold(items, 0, fn(acc: Int, l: LineItem) => acc + line_total(l))
}

fn apply_discount(total: Int, percent: Int) -> Int
  intent "Reduce a total by a whole-number percentage."
  requires percent >= 0
  requires percent <= 100
  requires total >= 0
  ensures result >= 0
  ensures result <= total
  law never_negative
{
  let cut = Int.div(total * percent, 100)
  total - Option.unwrap_or(cut, 0)
}

fn add(a: Int, b: Int) -> Int
  intent "Integer addition."
  ensures result == a + b
  law commutative
{
  a + b
}

fn overcharge(total: Int, fee: Int) -> Int
  intent "Deliberately claims the result never exceeds the total."
  requires total >= 0
  requires fee >= 0
  ensures result <= total
{
  total + fee
}

fn not_commutative(a: Int, b: Int) -> Int
  intent "Deliberately claims a law it does not satisfy."
  law commutative
{
  a - b
}

fn leaky(key: Text) -> Unit
  intent "Deliberately claims purity while performing an effect."
  uses store.write
  law pure
{
  store.write(key, "x")
}
'''

# A behaviour-preserving edit: same results, different implementation.
SAFE = SRC.replace(
    "  l.qty * l.unit.amount\n",
    "  let n = l.qty\n  n * l.unit.amount\n")

# A behaviour-changing edit: the discount is now applied with a floor.
BREAKING = SRC.replace(
    "  total - Option.unwrap_or(cut, 0)\n",
    "  Int.max(0, total - Option.unwrap_or(cut, 0) - 1)\n")

# An edit that reaches for a capability the change was not authorised to use.
ESCALATING = SRC.replace(
    "fn add(a: Int, b: Int) -> Int\n  intent \"Integer addition.\"\n"
    "  ensures result == a + b\n  law commutative\n{\n  a + b\n}",
    "fn add(a: Int, b: Int) -> Int\n  intent \"Integer addition.\"\n"
    "  uses store.write\n  ensures result == a + b\n  law commutative\n{\n"
    "  do store.write(\"sum\", Int.to_text(a + b))\n  a + b\n}")


def build(src, label):
    mod, bag = parse(src, f"{label}.canon")
    if bag.has_errors:
        print(bag.render(src))
        raise SystemExit(1)
    res = check([mod], bag)
    errs = [d for d in res.bag if d.severity.value == "error"]
    if errs:
        for d in errs:
            print(d.render(src))
        raise SystemExit(1)
    return res, Hasher().add_modules([mod])


def main():
    base, base_hashes = build(SRC, "billing")
    hashes = {qn: di.hash for qn, di in base_hashes.items()}
    failures = []

    def case(name, fn):
        try:
            print(f"  ok    {name}: {fn()}")
        except AssertionError as ae:
            failures.append(name)
            print(f"  FAIL  {name}: {ae}")

    # ---------------------------------------------------------------
    print("verification")
    v = Verifier(base, seed="kinode", runs=60, hashes=hashes)
    report = v.verify_all()
    by_name = {f.qualname.rsplit(".", 1)[-1]: f for f in report.functions}

    def t_sound():
        good = ["line_total", "subtotal", "apply_discount", "add"]
        bad = [n for n in good if not by_name[n].ok]
        assert not bad, f"correct functions reported as failing: " \
                        f"{[(n, [c.problem for c in by_name[n].counterexamples]) for n in bad]}"
        total_runs = sum(by_name[n].runs for n in good)
        return f"{len(good)} correct functions passed over {total_runs} runs"
    case("correct functions pass", t_sound)

    def t_postcondition():
        f = by_name["overcharge"]
        assert not f.ok, "a false postcondition was not caught"
        ce = f.counterexamples[0]
        assert ce.kind == "postcondition", ce.kind
        return f"counterexample ({', '.join(V.show(a) for a in ce.args)})"
    case("a false postcondition is found", t_postcondition)

    def t_shrunk():
        ce = by_name["overcharge"].counterexamples[0]
        # total=0, fee=1 is the minimal witness for result <= total.
        assert all(isinstance(a, int) and abs(a) <= 2 for a in ce.args), \
            f"counterexample was not shrunk: {[V.show(a) for a in ce.args]}"
        return f"shrunk to ({', '.join(V.show(a) for a in ce.args)})"
    case("counterexamples are shrunk to a readable size", t_shrunk)

    def t_law():
        f = by_name["not_commutative"]
        assert not f.ok, "a violated law was not caught"
        ce = f.counterexamples[0]
        assert ce.kind == "law" and ce.law == "commutative", (ce.kind, ce.law)
        return f"{ce.law}: {ce.problem[:60]}"
    case("a violated law is found", t_law)

    def t_purity():
        f = by_name["leaky"]
        assert not f.ok, "a purity violation was not caught"
        ce = f.counterexamples[0]
        assert ce.law == "pure", ce.law
        return ce.problem
    case("a purity claim contradicted by an effect is found", t_purity)

    def t_reproducible():
        r2 = Verifier(base, seed="kinode", runs=60, hashes=hashes).verify_all()
        a = {f.qualname: [c.to_json()["rendered"] for c in f.counterexamples]
             for f in report.functions}
        b = {f.qualname: [c.to_json()["rendered"] for c in r2.functions
                          for c in f.counterexamples]
             for f in r2.functions}
        same = all(a[k] == [c.to_json()["rendered"] for c in
                            next(f for f in r2.functions if f.qualname == k)
                            .counterexamples] for k in a)
        assert same, "the same seed produced different counterexamples"
        return "same seed, same counterexamples"
    case("verification is reproducible from its seed", t_reproducible)

    # ---------------------------------------------------------------
    print("\nstructural diff")

    def t_safe_diff():
        new, new_hashes = build(SAFE, "billing")
        d = structural_diff(base, new, base_hashes, new_hashes)
        edited = [c.qualname.rsplit(".", 1)[-1] for c in d.edited()]
        assert edited == ["line_total"], edited
        # subtotal depends on line_total, so it is in the blast radius even
        # though its own body is untouched.
        radius = {q.rsplit(".", 1)[-1] for q in d.blast_radius}
        assert "subtotal" in radius, radius
        return f"edited={edited}, blast radius={sorted(radius)}"
    case("an edit's blast radius includes its dependents", t_safe_diff)

    def t_cap_diff():
        new, new_hashes = build(ESCALATING, "billing")
        d = structural_diff(base, new, base_hashes, new_hashes)
        assert "store.write" in d.capabilities_added, d.capabilities_added
        return f"capabilities added: {d.capabilities_added}"
    case("a newly reached capability is detected", t_cap_diff)

    # ---------------------------------------------------------------
    print("\ndifferential execution")

    def t_safe_identical():
        new, new_hashes = build(SAFE, "billing")
        d = structural_diff(base, new, base_hashes, new_hashes)
        targets = [c.qualname for c in d.changed() if c.kind == "changed"]
        r = differential(base, new, targets, seed="kinode", runs=40)
        assert r.identical, [x.summary() for x in r.disagreements]
        return f"{r.runs} runs across {len(r.checked)} functions, no difference"
    case("a behaviour-preserving edit shows no difference", t_safe_identical)

    def t_breaking_found():
        new, new_hashes = build(BREAKING, "billing")
        d = structural_diff(base, new, base_hashes, new_hashes)
        targets = [c.qualname for c in d.changed() if c.kind == "changed"]
        r = differential(base, new, targets, seed="kinode", runs=40)
        assert not r.identical, "a behaviour change went undetected"
        dis = r.disagreements[0]
        return f"{dis.qualname.rsplit('.', 1)[-1]}({', '.join(V.show(a) for a in dis.args)}): {dis.summary()}"
    case("a behaviour-changing edit is detected", t_breaking_found)

    # ---------------------------------------------------------------
    print("\npromotion gate")

    good_verification = {"ok": True, "functions": []}

    def t_promote():
        new, new_hashes = build(SAFE, "billing")
        auth = Authorization(
            actor="agent-1", intent_id="INT-100",
            allowed_definitions=["billing.line_total"],
            max_blast_radius=10)
        dec = gate(base, new, auth, seed="kinode", runs=30,
                   hashes_old=base_hashes, hashes_new=new_hashes,
                   verification=good_verification)
        assert dec.decision == "promote", dec.render()
        return "safe edit promoted without escalation"
    case("a safe, in-scope, verified edit is promoted", t_promote)

    def t_escalate_behaviour():
        new, new_hashes = build(BREAKING, "billing")
        auth = Authorization(
            actor="agent-1", intent_id="INT-101",
            allowed_definitions=["billing.apply_discount"],
            max_blast_radius=10)
        dec = gate(base, new, auth, seed="kinode", runs=30,
                   hashes_old=base_hashes, hashes_new=new_hashes,
                   verification=good_verification)
        assert dec.decision == "escalate", dec.render()
        codes = {f.code for f in dec.findings}
        return f"escalated with {sorted(codes)}"
    case("a behaviour-changing edit is escalated", t_escalate_behaviour)

    def t_block_scope():
        new, new_hashes = build(BREAKING, "billing")
        auth = Authorization(
            actor="agent-1", intent_id="INT-102",
            allowed_definitions=["billing.line_total"],
            max_blast_radius=10)
        dec = gate(base, new, auth, seed="kinode", runs=20,
                   hashes_old=base_hashes, hashes_new=new_hashes,
                   verification=good_verification)
        assert dec.decision == "block", dec.render()
        f = next(x for x in dec.findings if x.code == "CANON-E0905")
        return f.message
    case("an out-of-scope edit is blocked", t_block_scope)

    def t_block_capability():
        new, new_hashes = build(ESCALATING, "billing")
        auth = Authorization(
            actor="agent-1", intent_id="INT-103",
            allowed_definitions=["billing.*"],
            allowed_capabilities=[], allow_new_capabilities=False,
            max_blast_radius=20)
        dec = gate(base, new, auth, seed="kinode", runs=20,
                   hashes_old=base_hashes, hashes_new=new_hashes,
                   verification=good_verification)
        assert dec.decision == "block", dec.render()
        f = next(x for x in dec.findings if "capabilit" in x.message)
        return f"{f.message}: {f.detail['capabilities']}"
    case("an unauthorised new capability is blocked", t_block_capability)

    def t_block_unverified():
        new, new_hashes = build(SAFE, "billing")
        auth = Authorization(actor="agent-1", intent_id="INT-104",
                             allowed_definitions=["billing.*"])
        dec = gate(base, new, auth, seed="kinode", runs=10,
                   hashes_old=base_hashes, hashes_new=new_hashes,
                   verification={"ok": False, "functions": [
                       {"function": "billing.overcharge", "ok": False}]})
        assert dec.decision == "block", dec.render()
        return "failed verification blocks promotion"
    case("a change that fails verification cannot be promoted",
         t_block_unverified)

    print("\nRESULT:", "pass" if not failures else f"FAIL ({failures})")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
