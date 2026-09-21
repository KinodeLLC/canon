"""Front-end smoke test: lex, parse, print, re-parse, hash."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from canon import Hasher, format_module, parse, short  # noqa: E402

SRC = r'''
--- Payment capture and refund.
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
  | Gone
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
  --- Append an immutable entry.
  append(entry: Text, amount: Int) -> Unit
  idempotent read(id: Text) -> Option<Text>
}

fn max_refundable(c: Charge) -> Int
  intent "The refundable ceiling for a charge is its captured amount."
  requires c.amount.amount >= 0
  ensures result >= 0
{
  c.amount.amount
}

fn refund(c: Charge, amt: Int) -> Result<Refund, RefundError>
  intent "Refund up to the captured amount of a charge, once."
  uses ledger.append
  requires amt > 0
  ensures true
  law idempotent_by(c.id)
  cost steps 5000, io 2
{
  match c.status {
    case Captured => {
      let ceiling = max_refundable(c)
      if amt > ceiling then Err(AmountExceeded(ceiling)) else {
        do ledger.append(c.id, amt)
        Ok(Refund { charge_id: c.id, amount: Money { amount: amt, currency: "USD" } })
      }
    }
    case Refunded => Err(Gone)
    case Pending => Err(NotCaptured)
  }
}

fn triage(body: Text) -> Text
  intent "Classify a support ticket by urgency."
  uses model.infer
  cost tokens 2000, io 1
{
  ask Text from claude.opus {
    system "Classify the urgency of this support ticket."
    input body
    grounded_in body
    temperature 0
    retries 3 on contract_violation, type_error
    max_tokens 64
  }
}

test "refund of a pending charge is rejected" {
  let c = Charge { id: "c1", amount: Money { amount: 100, currency: "USD" }, status: Pending }
  refund(c, 50) == Err(NotCaptured)
}
'''


def main():
    mod, bag = parse(SRC, "payments.canon")
    print("=== diagnostics ===")
    if len(bag):
        print(bag.render(SRC))
    else:
        print("(none)")

    print(f"\nmodule: {mod.name}")
    print(f"decls: {len(mod.decls)} "
          f"({len(mod.functions())} fn, {len(mod.types())} types, "
          f"{len(mod.effects())} effects, {len(mod.tests())} tests)")

    print("\n=== canonical form ===")
    text = format_module(mod)
    print(text)

    # Round-trip: printing canonical output and reparsing must be stable.
    mod2, bag2 = parse(text, "payments.canon")
    if bag2.has_errors:
        print("!! reparse of canonical output failed")
        print(bag2.render(text))
        return 1
    text2 = format_module(mod2)
    if text != text2:
        print("!! canonical form is not idempotent")
        import difflib
        for l in difflib.unified_diff(text.splitlines(), text2.splitlines(),
                                      "first", "second", lineterm=""):
            print(l)
        return 1
    print("canonical form is idempotent: yes")

    print("\n=== content hashes ===")
    h = Hasher()
    defs = h.add_modules([mod])
    for qn in sorted(defs):
        d = defs[qn]
        print(f"  {short(d.hash)}  {d.kind:7} {qn}")
        if d.deps:
            print(f"             deps: {', '.join(sorted(d.deps))}")
        if d.effects:
            print(f"             effects: {', '.join(sorted(d.effects))}")
        if d.models:
            print(f"             models: {', '.join(sorted(d.models))}")

    # Renaming a local must not change the hash.
    renamed = SRC.replace("let ceiling = max_refundable(c)",
                          "let cap = max_refundable(c)") \
                 .replace("if amt > ceiling then Err(AmountExceeded(ceiling))",
                          "if amt > cap then Err(AmountExceeded(cap))")
    mod3, bag3 = parse(renamed, "payments.canon")
    if bag3.has_errors:
        print("!! renamed variant failed to parse")
        print(bag3.render(renamed))
        return 1
    defs3 = Hasher().add_modules([mod3])
    same = defs["payments.refund"].hash == defs3["payments.refund"].hash
    print(f"\nlocal rename preserves hash: {'yes' if same else 'NO'}")

    # Reordering independent clauses must not change the hash.
    reordered = SRC.replace(
        "  uses ledger.append\n  requires amt > 0\n  ensures true\n",
        "  ensures true\n  requires amt > 0\n  uses ledger.append\n")
    mod4, bag4 = parse(reordered, "payments.canon")
    defs4 = Hasher().add_modules([mod4])
    same2 = defs["payments.refund"].hash == defs4["payments.refund"].hash
    print(f"clause reorder preserves hash: {'yes' if same2 else 'NO'}")

    # Changing a dependency must change the dependent's deep hash but not its
    # local hash.
    changed = SRC.replace("  c.amount.amount\n}", "  c.amount.amount + 0\n}")
    mod5, _ = parse(changed, "payments.canon")
    defs5 = Hasher().add_modules([mod5])
    deep_changed = defs["payments.refund"].hash != defs5["payments.refund"].hash
    local_same = (defs["payments.refund"].local_hash
                  == defs5["payments.refund"].local_hash)
    print(f"dependency edit changes deep hash: {'yes' if deep_changed else 'NO'}")
    print(f"dependency edit preserves local hash: {'yes' if local_same else 'NO'}")

    ok = (not bag.has_errors and same and same2 and deep_changed and local_same)
    print("\nRESULT:", "pass" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
