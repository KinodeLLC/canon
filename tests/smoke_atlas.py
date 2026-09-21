"""Atlas: graph queries, token-budgeted projections, and edit transactions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from canon.atlas import Atlas, estimate_tokens  # noqa: E402

SRC = r'''
module orders

record Money {
  amount: Int
  currency: Text
  invariant amount >= 0
}

record Customer {
  id: Text
  email: Text
  classify email personal
}

record Order {
  id: Text
  customer: Customer
  total: Money
}

effect store {
  write(key: Text, value: Text) -> Unit
  read(key: Text) -> Option<Text>
}

effect mailer {
  send(to: Text, subject: Text, body: Text) -> Unit
}

fn order_value(o: Order) -> Int
  intent "The monetary value of an order in minor units."
  ensures result >= 0
  law never_negative
{
  o.total.amount
}

fn is_large(o: Order) -> Bool
  intent "Whether an order needs manual review."
  ensures true
{
  order_value(o) > 100000
}

fn persist(o: Order) -> Unit
  intent "Write an order to durable storage."
  uses store.write
{
  store.write(o.id, o.customer.id)
}

fn notify(o: Order) -> Unit
  intent "Email the customer about their order."
  uses mailer.send
{
  mailer.send(o.customer.email, "Your order", o.id)
}

fn submit(o: Order) -> Bool
  intent "Persist an order, notify the customer, and flag large ones."
  uses store.write, mailer.send
{
  do persist(o)
  do notify(o)
  is_large(o)
}

fn undocumented(n: Int) -> Int
{
  n
}
'''


def main():
    atlas = Atlas.from_sources({"orders.canon": SRC})
    errs = [d for d in atlas.cr.bag if d.severity.value == "error"]
    if errs:
        for d in errs:
            print(d.render(SRC))
        return 1

    failures = []

    def case(name, fn):
        try:
            print(f"  ok    {name}: {fn()}")
        except AssertionError as ae:
            failures.append(name)
            print(f"  FAIL  {name}: {ae}")

    print("graph queries")

    def t_resolve():
        n = atlas.resolve("submit")
        assert n is not None and n.qualname == "orders.submit"
        by_hash = atlas.resolve(n.hash)
        assert by_hash is n, "hash lookup returned a different node"
        return f"{n.qualname} {n.hash[:10]} :: {n.signature}"
    case("definitions resolve by name and by hash", t_resolve)

    def t_callers():
        direct = atlas.callers("order_value")
        trans = atlas.callers("order_value", transitive=True)
        assert "orders.is_large" in direct, direct
        assert "orders.submit" in trans, trans
        assert "orders.submit" not in direct, direct
        return f"direct={direct}, transitive={sorted(trans)}"
    case("direct and transitive callers are distinguished", t_callers)

    def t_blast():
        b = atlas.blast_radius("order_value")
        assert set(b["capabilities"]) >= {"store.write", "mailer.send"}, b
        assert "personal" in b["data_classifications"], b
        return (f"{b['count']} dependents, capabilities="
                f"{b['capabilities']}, data={b['data_classifications']}")
    case("blast radius reports reachable capabilities and data classes",
         t_blast)

    def t_caps():
        c = atlas.capabilities("submit")
        assert set(c["transitive"]) == {"store.write", "mailer.send"}, c
        assert "orders.persist" in c["granted_by"], c["granted_by"]
        return f"{c['transitive']} granted by {sorted(c['granted_by'])}"
    case("capability footprint attributes each effect to its source", t_caps)

    def t_effect_users():
        users = atlas.effect_users("mailer.send")
        assert set(users) == {"orders.notify", "orders.submit"}, users
        return f"mailer.send reachable from {users}"
    case("every definition reaching an effect can be listed", t_effect_users)

    def t_unverified():
        u = atlas.unverified()
        assert "orders.undocumented" in u["without_contracts"], u
        assert "orders.undocumented" in u["without_intent"], u
        assert "orders.order_value" not in u["without_contracts"], u
        return (f"{len(u['without_contracts'])} of {u['total_functions']} "
                f"functions have no contracts")
    case("definitions without contracts or intent are listed", t_unverified)

    def t_search():
        hits = atlas.search("customer")
        names = [h.qualname for h in hits]
        assert "orders.notify" in names, names
        return f"'customer' -> {names[:4]}"
    case("search matches names, intent and documentation", t_search)

    print("\nprojections")

    def t_view_levels():
        n = atlas.resolve("order_value")
        sig = atlas.render(n, "signature")
        con = atlas.render(n, "contract")
        full = atlas.render(n, "full")
        assert len(sig) < len(con) < len(full), (len(sig), len(con), len(full))
        assert "never_negative" in con and "{" not in con.split("law")[0]
        return (f"signature {estimate_tokens(sig)}t < contract "
                f"{estimate_tokens(con)}t < full {estimate_tokens(full)}t")
    case("detail levels cost progressively more tokens", t_view_levels)

    def t_view_complete():
        v = atlas.view("submit", budget=4000)
        assert v["complete"], v["omitted"]
        assert v["tokens_used"] <= 4000, v["tokens_used"]
        kinds = {s["kind"] for s in v["sections"]}
        assert kinds == {"definition", "dependency"}, kinds
        deps = {s["name"] for s in v["sections"] if s["kind"] == "dependency"}
        assert "orders.persist" in deps and "orders.notify" in deps, deps
        return (f"{v['tokens_used']} tokens, {len(v['sections'])} sections, "
                f"dependencies as contracts")
    case("a view includes callees as contracts, not full bodies",
         t_view_complete)

    def t_view_budget():
        v = atlas.view("submit", budget=40)
        assert v["tokens_used"] <= 60, v["tokens_used"]
        assert not v["complete"], "a tight budget should report omissions"
        assert v["omitted"], v
        return f"{v['tokens_used']} tokens used, omitted {v['omitted']}"
    case("a tight budget reports what was left out", t_view_budget)

    print("\nedit transactions")

    def t_reject():
        bad = SRC.replace("  o.total.amount\n", "  o.total.nonexistent\n")
        p = atlas.propose({"orders.canon": bad})
        assert not p.ok, "a broken proposal was accepted"
        codes = {d["code"] for d in p.diagnostics()}
        assert "CANON-E0206" in codes, codes
        assert atlas.resolve("order_value").hash, "original atlas was mutated"
        return f"rejected with {sorted(codes)}"
    case("a proposal that does not check is rejected", t_reject)

    def t_impact():
        edited = SRC.replace("  o.total.amount\n",
                             "  Int.max(0, o.total.amount)\n")
        p = atlas.propose({"orders.canon": edited})
        assert p.ok, p.render_diagnostics()
        imp = p.impact()
        assert imp["edited"] == ["orders.order_value"], imp["edited"]
        radius = set(imp["blast_radius"])
        assert {"orders.is_large", "orders.submit"} <= radius, radius
        assert not imp["capabilities_added"], imp["capabilities_added"]
        return (f"edited={imp['edited']}, blast radius={len(radius)}, "
                f"no new capabilities")
    case("a valid proposal reports its blast radius before commit", t_impact)

    def t_capability_propagates():
        # Granting `order_value` an effect must force every caller to declare
        # it too. A proposal that adds the effect without updating the callers
        # is rejected, naming each caller that has to change.
        partial = SRC.replace(
            "  ensures result >= 0\n  law never_negative\n{\n"
            "  o.total.amount\n}",
            "  uses store.write\n  ensures result >= 0\n  law never_negative\n{\n"
            "  do store.write(o.id, \"seen\")\n  o.total.amount\n}")
        p = atlas.propose({"orders.canon": partial})
        assert not p.ok, "an effect was allowed to propagate silently"
        undeclared = [d for d in p.diagnostics() if d["code"] == "CANON-E0401"]
        named = {d["message"].split("'")[1] for d in undeclared}
        assert "store.write" in named, named
        return (f"{len(undeclared)} caller(s) must declare the effect before "
                f"this can check")
    case("an effect cannot reach a caller without being declared there",
         t_capability_propagates)

    def t_capability_growth():
        escalated = (SRC
                     .replace("  ensures result >= 0\n  law never_negative\n{\n"
                              "  o.total.amount\n}",
                              "  uses store.write\n  ensures result >= 0\n"
                              "  law never_negative\n{\n"
                              "  do store.write(o.id, \"seen\")\n"
                              "  o.total.amount\n}")
                     .replace("fn is_large(o: Order) -> Bool\n"
                              "  intent \"Whether an order needs manual review.\"\n"
                              "  ensures true\n",
                              "fn is_large(o: Order) -> Bool\n"
                              "  intent \"Whether an order needs manual review.\"\n"
                              "  uses store.write\n  ensures true\n"))
        p = atlas.propose({"orders.canon": escalated})
        assert p.ok, p.render_diagnostics()
        imp = p.impact()
        assert "store.write" in imp["capabilities_added"], \
            f"capabilities_added={imp['capabilities_added']}"
        assert "orders.order_value" in imp["edited"], imp["edited"]
        return (f"new capabilities: {imp['capabilities_added']}, "
                f"edited {len(imp['edited'])} definitions")
    case("a proposal that widens the capability footprint says so",
         t_capability_growth)

    def t_commit():
        edited = SRC.replace("  o.total.amount\n",
                             "  Int.max(0, o.total.amount)\n")
        p = atlas.propose({"orders.canon": edited})
        before = atlas.resolve("order_value").hash
        after_atlas = p.commit()
        after = after_atlas.resolve("order_value").hash
        assert before != after, "committing did not change the hash"
        assert atlas.resolve("order_value").hash == before, \
            "the original atlas was mutated by a commit"
        return f"{before[:10]} -> {after[:10]}, original unchanged"
    case("commit produces a new atlas and leaves the old one intact",
         t_commit)

    def t_canonical():
        messy = SRC.replace("fn undocumented(n: Int) -> Int\n{\n  n\n}",
                            "fn undocumented(n:Int)->Int\n{\n      n\n}")
        p = atlas.propose({"orders.canon": messy})
        assert p.ok, p.render_diagnostics()
        out = p.canonical()["orders.canon"]
        assert "fn undocumented(n: Int) -> Int" in out, "not canonicalised"
        imp = p.impact()
        assert not imp["edited"], f"formatting changed a hash: {imp['edited']}"
        return "reformatting alone changes no definition hash"
    case("formatting differences do not change any hash", t_canonical)

    print("\nRESULT:", "pass" if not failures else f"FAIL ({failures})")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
