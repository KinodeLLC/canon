"""Checker smoke test: types, effects, exhaustiveness, contracts, blast radius."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from canon import parse  # noqa: E402
from canon.checker import check  # noqa: E402

GOOD = r'''
module payments

record Money {
  amount: Int
  currency: Text
  invariant amount >= 0
}

enum ChargeStatus {
  | Pending
  | Captured
  | Refunded
}

enum RefundError {
  | NotCaptured
  | AmountExceeded(Int)
  | AlreadyRefunded
}

record Charge {
  id: Text
  amount: Money
  status: ChargeStatus
  classify id pseudonymous
}

record Refund {
  charge_id: Text
  amount: Money
}

effect ledger {
  append(entry: Text, amount: Int) -> Unit
  idempotent read(id: Text) -> Option<Text>
}

fn ceiling_of(c: Charge) -> Int
  intent "The most that can be refunded against a charge."
  ensures result >= 0
{
  c.amount.amount
}

fn refund(c: Charge, amt: Int) -> Result<Refund, RefundError>
  intent "Refund up to the captured amount of a charge."
  uses ledger.append
  requires amt > 0
  ensures true
  law idempotent_by(c.id)
  cost steps 5000, io 2
{
  match c.status {
    case Pending => Err(NotCaptured)
    case Refunded => Err(AlreadyRefunded)
    case Captured => {
      let ceiling = ceiling_of(c)
      if amt > ceiling then Err(AmountExceeded(ceiling)) else {
        do ledger.append(c.id, amt)
        Ok(Refund { charge_id: c.id, amount: Money { amount: amt, currency: c.amount.currency } })
      }
    }
  }
}

fn total_refunded(rs: List<Refund>) -> Int
  intent "Sum the amounts of a list of refunds."
  ensures result >= 0
{
  List.fold(rs, 0, fn(acc: Int, r: Refund) => acc + r.amount.amount)
}

fn triage(body: Text) -> Text
  intent "Classify the urgency of a support ticket."
  uses model.infer, model.judge
  cost tokens 2000, io 1
{
  ask Text from claude.opus {
    system "Classify the urgency of this ticket as low, medium or high."
    input body
    grounded_in body
    temperature 0
    retries 3 on contract_violation, grounding_failure
    max_tokens 32
  }
}

fn describe(c: Charge) -> Text
  intent "A short human description of a charge."
{
  let status_text = match c.status {
    case Pending => "pending"
    case Captured => "captured"
    case Refunded => "refunded"
  }
  c.id ++ " is " ++ status_text
}
'''

BAD = r'''
module broken

enum Colour {
  | Red
  | Green
  | Blue
}

effect db {
  write(k: Text, v: Text) -> Unit
  read(k: Text) -> Option<Text>
}

fn missing_arm(c: Colour) -> Int
  intent "Does not handle Blue."
{
  match c {
    case Red => 1
    case Green => 2
  }
}

fn undeclared_effect(k: Text) -> Unit
  intent "Writes without declaring the capability."
{
  db.write(k, "v")
}

fn wrong_type(a: Int) -> Text
  intent "Returns the wrong type."
{
  a
}

fn mixed_numbers(a: Int, b: Dec) -> Dec
  intent "Mixes Int and Dec without conversion."
{
  a + b
}

fn uses_division(a: Int, b: Int) -> Int
  intent "Uses an operator that does not exist."
{
  a / b
}

fn infinite(n: Int) -> Int
  intent "Recursion with no measure."
{
  infinite(n)
}

fn bad_try(a: Int) -> Int
  intent "Uses ? outside a Result function."
{
  let r = Ok(a)
  r?
}

fn over_broad(k: Text) -> Unit
  intent "Requests every db operation."
  uses db.*
{
  db.write(k, "v")
}

fn unknown_thing(a: Int) -> Int
  intent "Calls something that does not exist."
{
  compute_totl(a)
}

fn compute_total(a: Int) -> Int
  intent "The function the previous one meant to call."
{
  a
}
'''


def main():
    print("=" * 70)
    print("WELL-FORMED MODULE")
    print("=" * 70)
    mod, bag = parse(GOOD, "payments.canon")
    if bag.has_errors:
        print(bag.render(GOOD))
        return 1
    res = check([mod], bag)
    errs = [d for d in res.bag if d.severity.value == "error"]
    warns = [d for d in res.bag if d.severity.value == "warning"]
    print(f"errors: {len(errs)}  warnings: {len(warns)}")
    for d in errs:
        print(d.render(GOOD))
    for d in warns:
        print("  " + d.message + f"  ({d.span})")

    print("\n--- capability footprint ---")
    for qn in sorted(res.env.fns):
        fi = res.env.fns[qn]
        direct = ", ".join(sorted(fi.performed)) or "(pure)"
        trans = ", ".join(sorted(fi.transitive)) or "(pure)"
        print(f"  {qn}")
        print(f"      declared:   {', '.join(fi.declared.keys()) or '(pure)'}")
        print(f"      performed:  {direct}")
        print(f"      transitive: {trans}")
        if fi.calls:
            print(f"      calls:      {', '.join(sorted(fi.calls))}")
        if fi.models:
            print(f"      models:     {', '.join(sorted(fi.models))}")
        if fi.touches:
            print(f"      data class: {', '.join(sorted(fi.touches))}")

    print("\n--- match exhaustiveness ---")
    for m in res.matches:
        print(f"  {m.scrutinee_type:16} covered={m.covered} "
              f"exhaustive={m.exhaustive}")

    print()
    print("=" * 70)
    print("MODULE WITH DELIBERATE ERRORS")
    print("=" * 70)
    mod2, bag2 = parse(BAD, "broken.canon")
    res2 = check([mod2], bag2)
    errs2 = [d for d in res2.bag if d.severity.value == "error"]
    print(f"errors: {len(errs2)}\n")
    for d in errs2:
        print(d.render(BAD))
        print()

    expected_codes = {
        "CANON-E0305",   # non-exhaustive match
        "CANON-E0401",   # undeclared effect
        "CANON-E0301",   # type mismatch / bad operator
        "CANON-E0311",   # recursion without decreases
        "CANON-E0310",   # ? outside Result
        "CANON-E0201",   # unknown name
    }
    found = {d.code for d in errs2}
    missing = expected_codes - found
    print("codes found:", ", ".join(sorted(found)))
    if missing:
        print("MISSING expected codes:", ", ".join(sorted(missing)))

    good_ok = len(errs) == 0
    bad_ok = not missing
    print("\nRESULT:", "pass" if (good_ok and bad_ok) else "FAIL")
    return 0 if (good_ok and bad_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
