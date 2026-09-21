"""Interpreter smoke test: evaluation, contracts, budgets, effects, totality."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from canon import Hasher, parse  # noqa: E402
from canon import values as V  # noqa: E402
from canon.checker import check  # noqa: E402
from canon.interp import (  # noqa: E402
    Budget, Fault, Interpreter, Runtime, run_tests,
)

SRC = r'''
module shop

record Money {
  amount: Int
  currency: Text
  invariant amount >= 0
}

enum OrderStatus {
  | Draft
  | Placed
  | Shipped
  | Cancelled
}

enum OrderError {
  | NotDraft
  | EmptyOrder
  | OverLimit(Int)
}

record Line {
  sku: Text
  qty: Int
  unit_price: Money
}

record Order {
  id: Text
  status: OrderStatus
  lines: List<Line>
}

effect store {
  save(id: Text, blob: Text) -> Unit
  idempotent load(id: Text) -> Option<Text>
}

const credit_limit: Int = 100000

fn line_total(l: Line) -> Int
  intent "The extended price of one order line."
  requires l.qty >= 0
  ensures result >= 0
{
  l.qty * l.unit_price.amount
}

fn order_total(o: Order) -> Int
  intent "The sum of all line totals on an order."
  ensures result >= 0
  law never_negative
{
  List.fold(o.lines, 0, fn(acc: Int, l: Line) => acc + line_total(l))
}

fn place(o: Order) -> Result<Order, OrderError>
  intent "Move a draft order to placed, if it is within the credit limit."
  uses store.save, audit.entry
  ensures true
  law idempotent_by(o.id)
  cost steps 100000, io 4
{
  match o.status {
    case Placed => Ok(o)
    case Shipped => Err(NotDraft)
    case Cancelled => Err(NotDraft)
    case Draft => {
      let total = order_total(o)
      if List.is_empty(o.lines) then Err(EmptyOrder) else {
        if total > credit_limit then Err(OverLimit(credit_limit)) else {
          do store.save(o.id, "placed")
          do audit.entry("order.place", o.id, Int.to_text(total))
          Ok(Order { ..o, status: Placed })
        }
      }
    }
  }
}

fn countdown(n: Int) -> Int
  intent "A terminating recursive function."
  requires n >= 0
  ensures result == 0
  decreases n
{
  if n <= 0 then 0 else countdown(n - 1)
}

fn bad_contract(n: Int) -> Int
  intent "Deliberately claims something false."
  ensures result > n
{
  n
}

fn spin(n: Int) -> Int
  intent "Allocates work proportional to n."
  requires n >= 0
{
  List.fold(List.range(0, n), 0, fn(acc: Int, i: Int) => acc + i)
}

test "line total multiplies quantity by unit price" {
  let l = Line { sku: "a", qty: 3, unit_price: Money { amount: 250, currency: "USD" } }
  line_total(l) == 750
}

test "an empty draft order cannot be placed" {
  let o = Order { id: "o1", status: Draft, lines: [] }
  place(o) == Err(EmptyOrder)
}
'''


class RecordingRuntime(Runtime):
    """Grants a fixed set of operations and records everything performed."""

    def __init__(self, allow):
        self.allow = set(allow)
        self.performed = []

    def granted(self):
        return sorted(self.allow)

    def perform(self, key, args, span=None, idempotent=False):
        if key not in self.allow:
            return super().perform(key, args, span, idempotent)
        self.performed.append((key, tuple(args)))
        if key == "store.load":
            return V.NONE
        return V.UNIT


def main():
    mod, bag = parse(SRC, "shop.canon")
    if bag.has_errors:
        print(bag.render(SRC))
        return 1
    res = check([mod], bag)
    errs = [d for d in res.bag if d.severity.value == "error"]
    if errs:
        for d in errs:
            print(d.render(SRC))
        return 1
    hashes = {qn: di.hash for qn, di in Hasher().add_modules([mod]).items()}
    failures = []

    def case(name, fn):
        try:
            outcome = fn()
            print(f"  ok    {name}: {outcome}")
        except AssertionError as ae:
            failures.append(name)
            print(f"  FAIL  {name}: {ae}")

    # ---------------------------------------------------------------
    print("evaluation")

    def mk_order(status="Draft", lines=1, qty=2, price=500):
        ls = tuple(
            V.Record("Line", (("sku", f"s{i}"), ("qty", qty),
                              ("unit_price", V.Record(
                                  "Money", (("amount", price),
                                            ("currency", "USD"))))))
            for i in range(lines))
        return V.Record("Order", (("id", "o1"),
                                  ("status", V.Variant(status, (), "OrderStatus")),
                                  ("lines", ls)))

    def t_total():
        rt = RecordingRuntime([])
        it = Interpreter(res, rt, Budget(), hashes)
        total = it.call("order_total", [mk_order(lines=3, qty=2, price=500)])
        assert total == 3000, f"expected 3000, got {total}"
        return total
    case("order_total sums lines", t_total)

    def t_place():
        rt = RecordingRuntime(["store.save", "audit.entry"])
        it = Interpreter(res, rt, Budget(), hashes)
        out = it.call("place", [mk_order()])
        assert V.is_ok(out), f"expected Ok, got {V.show(out)}"
        placed = out.args[0]
        assert placed.get("status").ctor == "Placed"
        assert [k for k, _ in rt.performed] == ["store.save", "audit.entry"], \
            rt.performed
        return f"{V.show(out.args[0].get('status'))}, effects={[k for k,_ in rt.performed]}"
    case("place performs exactly its declared effects", t_place)

    def t_recursion():
        it = Interpreter(res, RecordingRuntime([]), Budget(), hashes)
        return it.call("countdown", [200])
    case("terminating recursion returns", t_recursion)

    # ---------------------------------------------------------------
    print("\ncapability enforcement")

    def t_denied():
        rt = RecordingRuntime(["store.save"])       # audit.entry withheld
        it = Interpreter(res, rt, Budget(), hashes)
        try:
            it.call("place", [mk_order()])
        except Fault as f:
            assert f.code == "CANON-E0403", f.code
            return f"{f.code} for {f.facts.get('operation')}"
        raise AssertionError("expected the call to be denied")
    case("an ungranted capability is denied at the boundary", t_denied)

    # ---------------------------------------------------------------
    print("\ncontract enforcement")

    def t_invariant():
        it = Interpreter(res, RecordingRuntime([]), Budget(), hashes)
        bad = V.Record("Line", (("sku", "x"), ("qty", -1),
                                ("unit_price", V.Record(
                                    "Money", (("amount", 100), ("currency", "USD"))))))
        try:
            it.call("line_total", [bad])
        except Fault as f:
            assert f.code == "CANON-E0507", f.code
            return f"{f.code}: {f.facts.get('clause')}"
        raise AssertionError("expected a precondition failure")
    case("a violated precondition is a structured fault", t_invariant)

    def t_post():
        it = Interpreter(res, RecordingRuntime([]), Budget(), hashes)
        try:
            it.call("bad_contract", [5])
        except Fault as f:
            assert f.code == "CANON-E0508", f.code
            return f"{f.code}: {f.facts.get('clause')} with result={f.facts.get('result')}"
        raise AssertionError("expected a postcondition failure")
    case("a false postcondition is caught with its arguments", t_post)

    def t_record_invariant():
        it = Interpreter(res, RecordingRuntime([]), Budget(), hashes)
        src = 'module t\nfn f() -> Money { Money { amount: -5, currency: "USD" } }'
        m2, b2 = parse(SRC + "\n\nfn make_bad() -> Money\n  intent \"builds an "
                       "invalid Money\"\n{\n  Money { amount: -5, currency: \"USD\" }\n}\n",
                       "shop.canon")
        r2 = check([m2], b2)
        it2 = Interpreter(r2, RecordingRuntime([]), Budget())
        try:
            it2.call("make_bad", [])
        except Fault as f:
            assert f.code == "CANON-E0508", f.code
            return f"{f.code}: {f.facts.get('clause')}"
        raise AssertionError("expected a record invariant failure")
    case("a record invariant is enforced at construction", t_record_invariant)

    # ---------------------------------------------------------------
    print("\nbudget enforcement")

    def t_budget():
        it = Interpreter(res, RecordingRuntime([]), Budget(steps=5000), hashes)
        try:
            it.call("spin", [100000])
        except Fault as f:
            assert f.code == "CANON-E0601", f.code
            return f"{f.code} after {f.facts.get('used')} steps"
        raise AssertionError("expected the step budget to stop it")
    case("an over-budget computation is stopped", t_budget)

    def t_io_budget():
        rt = RecordingRuntime(["store.save", "audit.entry"])
        it = Interpreter(res, rt, Budget(io=1), hashes)
        try:
            it.call("place", [mk_order()])
        except Fault as f:
            assert f.code == "CANON-E0602", f.code
            return f"{f.code}: {f.facts.get('resource')}"
        raise AssertionError("expected the io budget to stop it")
    case("the io budget bounds effect count", t_io_budget)

    # ---------------------------------------------------------------
    print("\ndeclared tests")
    rt = RecordingRuntime(["store.save", "audit.entry"])
    for t in run_tests(res, rt):
        status = "ok   " if t["passed"] else "FAIL "
        if not t["passed"]:
            failures.append(t["name"])
        detail = ""
        if not t["passed"]:
            detail = f"  {t.get('fault', {}).get('message', t.get('value', ''))}"
        print(f"  {status} {t['name']}{detail}  cost={t['cost']['steps']} steps")

    print("\nRESULT:", "pass" if not failures else f"FAIL ({failures})")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
