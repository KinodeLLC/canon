"""The agent interface: a full session driven the way an agent would drive it."""

import io
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from canon.protocol import DISPATCH, Session, handle, serve  # noqa: E402

SRC = r'''
module shop

record Money {
  amount: Int
  currency: Text = "USD"
  invariant amount >= 0
}

record Item {
  sku: Text
  qty: Int
  price: Money
  invariant qty >= 0
}

effect store {
  write(key: Text, value: Text) -> Unit
  read(key: Text) -> Option<Text>
}

effect mail {
  send(to: Text, subject: Text) -> Unit
}

fn line_total(i: Item) -> Int
  intent "The extended price of one line."
  requires i.qty >= 0
  ensures result >= 0
  law never_negative
{
  i.qty * i.price.amount
}

fn cart_total(items: List<Item>) -> Int
  intent "The total price of a cart."
  ensures result >= 0
  law never_negative
{
  List.fold(items, 0, fn(acc: Int, i: Item) => acc + line_total(i))
}

fn persist_cart(key: Text, items: List<Item>) -> Unit
  intent "Store a cart."
  uses store.write
{
  store.write(key, Int.to_text(cart_total(items)))
}

fn undocumented(n: Int) -> Int
{
  n
}
'''


def main():
    failures = []

    def case(name, fn):
        try:
            print(f"  ok    {name}: {fn()}")
        except AssertionError as ae:
            failures.append(name)
            print(f"  FAIL  {name}: {ae}")

    tmp = tempfile.mkdtemp(prefix="canon-protocol-")
    path = os.path.join(tmp, "shop.canon")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(SRC)

    session = Session()
    counter = [0]

    def call(method, **params):
        counter[0] += 1
        return handle(session, {"id": counter[0], "method": method,
                                "params": params})

    print("session")

    def t_describe():
        r = call("workspace.describe")
        assert r["ok"] and "atlas.view" in r["methods"], r
        return f"{len(r['methods'])} methods, protocol {r['protocol']}"
    case("the interface describes itself", t_describe)

    def t_open():
        r = call("workspace.open", paths=[path])
        assert r["ok"], r
        assert r["summary"]["files"] == 1, r["summary"]
        return (f"{r['stats']['definitions']} definitions, "
                f"{len(r['warnings'])} warnings")
    case("a workspace opens and reports its warnings", t_open)

    def t_unopened_guard():
        fresh = Session()
        r = handle(fresh, {"id": 1, "method": "atlas.blast",
                           "params": {"target": "x"}})
        assert not r["ok"] and "no workspace" in r["error"], r
        return r["error"]
    case("queries before a workspace is open fail cleanly", t_unopened_guard)

    print("\nreading")

    def t_search():
        r = call("atlas.search", query="cart")
        names = [x["definition"] for x in r["results"]]
        assert "shop.cart_total" in names, names
        return f"'cart' -> {names[:3]}"
    case("search finds definitions by name and intent", t_search)

    def t_budgeted_view():
        wide = call("atlas.view", focus="persist_cart", budget=4000)
        tight = call("atlas.view", focus="persist_cart", budget=30)
        assert wide["complete"], wide["omitted"]
        assert not tight["complete"], tight
        assert tight["tokens_used"] <= 60, tight["tokens_used"]
        return (f"budget 4000 -> {wide['tokens_used']} tokens complete; "
                f"budget 30 -> {tight['tokens_used']} tokens, omitted "
                f"{sum(len(v) for v in tight['omitted'].values())}")
    case("a view respects its token budget and reports omissions",
         t_budgeted_view)

    def t_blast():
        r = call("atlas.blast", target="line_total")
        assert r["ok"], r
        assert "shop.persist_cart" in r["transitive_dependents"], r
        assert "store.write" in r["capabilities"], r
        return (f"{r['count']} dependents, capabilities {r['capabilities']}")
    case("blast radius answers what a change could reach", t_blast)

    def t_unknown_suggests():
        r = call("atlas.show", target="cart_totl")
        assert not r["ok"], r
        assert any(s.endswith("cart_total") for s in r["suggestions"]), r
        return f"unknown name suggested {r['suggestions']}"
    case("an unknown definition comes back with suggestions",
         t_unknown_suggests)

    print("\nediting")

    def t_reject_broken():
        broken = SRC.replace("  i.qty * i.price.amount",
                             "  i.qty * i.price.amountt")
        r = call("edit.propose", changes={path: broken})
        assert not r["ok"], r
        codes = {d["code"] for d in r["diagnostics"]}
        assert "CANON-E0206" in codes, codes
        # The rejected proposal must not have changed anything.
        after = call("atlas.show", target="line_total")
        assert after["ok"], after
        return f"rejected with {sorted(codes)}, workspace untouched"
    case("a proposal that does not check is rejected", t_reject_broken)

    def t_impact_before_commit():
        edited = SRC.replace("  i.qty * i.price.amount",
                             "  Int.max(0, i.qty * i.price.amount)")
        r = call("edit.propose", changes={path: edited})
        assert r["ok"], r
        imp = r["impact"]
        assert imp["edited"] == ["shop.line_total"], imp["edited"]
        assert "shop.cart_total" in imp["blast_radius"], imp["blast_radius"]
        assert not imp["capabilities_added"], imp
        call("edit.abort", proposal=r["proposal"])
        return (f"edited {imp['edited']}, blast radius "
                f"{len(imp['blast_radius'])}, aborted")
    case("impact is available before deciding to commit",
         t_impact_before_commit)

    def t_capability_growth():
        escalated = SRC.replace(
            "fn undocumented(n: Int) -> Int\n{\n  n\n}",
            "fn undocumented(n: Int) -> Int\n  uses mail.send\n{\n"
            "  do mail.send(\"ops@example.com\", \"n\")\n  n\n}")
        r = call("edit.propose", changes={path: escalated})
        assert r["ok"], r
        assert "mail.send" in r["impact"]["capabilities_added"], r["impact"]
        gate = call("gate.evaluate", proposal=r["proposal"],
                    allowed_definitions=["shop.*"],
                    allow_new_capabilities=False,
                    require_verification=False)
        assert gate["decision"] == "block", gate
        call("edit.abort", proposal=r["proposal"])
        return f"capability growth detected and blocked by the gate"
    case("a proposal that widens capabilities is blocked by the gate",
         t_capability_growth)

    def t_gate_promote():
        edited = SRC.replace("  i.qty * i.price.amount",
                             "  let q = i.qty\n  q * i.price.amount")
        r = call("edit.propose", changes={path: edited})
        assert r["ok"], r
        gate = call("gate.evaluate", proposal=r["proposal"],
                    intent="INT-1", allowed_definitions=["shop.line_total"],
                    require_verification=True, runs=20)
        assert gate["decision"] == "promote", gate
        return "behaviour-preserving, in-scope, verified edit promoted"
    case("the gate promotes a safe verified edit", t_gate_promote)

    def t_commit():
        before = call("atlas.show", target="line_total")["definition"]["hash"]
        edited = SRC.replace("  i.qty * i.price.amount",
                             "  Int.max(0, i.qty * i.price.amount)")
        r = call("edit.propose", changes={path: edited})
        assert r["ok"], r
        c = call("edit.commit", proposal=r["proposal"], write=True)
        assert c["ok"], c
        after = call("atlas.show", target="line_total")["definition"]["hash"]
        assert before != after, "commit did not change the hash"
        with open(path, "r", encoding="utf-8") as f:
            on_disk = f.read()
        assert "Int.max(0, i.qty * i.price.amount)" in on_disk, on_disk[:200]
        return f"{before[:9]} -> {after[:9]}, written to disk"
    case("commit applies the change and writes canonical source", t_commit)

    def t_commit_requires_check():
        broken = SRC.replace("fn undocumented(n: Int) -> Int",
                             "fn undocumented(n: Int) -> Text")
        r = call("edit.propose", changes={path: broken})
        assert not r["ok"], r
        c = handle(session, {"id": 99, "method": "edit.commit",
                             "params": {"proposal": r.get("proposal", "none")}})
        assert not c["ok"], c
        return c.get("error", "")[:70]
    case("a failing proposal cannot be committed", t_commit_requires_check)

    print("\nrunning and verifying")

    def t_run_denied():
        # Arguments are JSON values in the request, not JSON-encoded strings.
        r = call("run.call", function="persist_cart", arguments=["k", []])
        assert not r["ok"], r
        assert r["fault"]["code"] == "CANON-E0403", r["fault"]
        return f"{r['fault']['code']}: ungranted effect refused"

    def t_run_with_grant():
        r = call("run.call", function="persist_cart", arguments=["k", []],
                 grants=["store.write"])
        assert r["ok"], r
        assert r["simulated"], "the response did not say it was simulated"
        assert r["journal"] and r["journal"][0]["op"] == "store.write", \
            r["journal"]
        return (f"granted run performed {len(r['journal'])} journaled "
                f"effect(s), marked simulated")

    def t_run_reproducible():
        a = call("run.call", function="persist_cart", arguments=["k", []],
                 grants=["store.write"])
        b = call("run.call", function="persist_cart", arguments=["k", []],
                 grants=["store.write"])
        assert a["journal"][0]["args"] == b["journal"][0]["args"], \
            (a["journal"], b["journal"])
        return "two simulated runs produced identical journals"
    case("running without a grant is refused", t_run_denied)
    case("a granted run performs and journals its effects", t_run_with_grant)
    case("simulated runs are reproducible", t_run_reproducible)

    def t_run_granted():
        r = call("run.call", function="cart_total", arguments=[[]])
        assert r["ok"], r
        assert r["result"] == 0, r
        return f"cart_total([]) = {r['rendered']}, cost {r['cost']['steps']} steps"
    case("a pure function runs without any grant", t_run_granted)

    def t_verify():
        r = call("verify.run", only=["shop.cart_total", "shop.line_total"],
                 runs=25)
        assert r["ok"], r
        return (f"{r['summary']['passed']}/{r['summary']['total']} verified "
                f"over {r['summary']['runs']} cases")
    case("verification runs through the interface", t_verify)

    def t_unverified():
        r = call("atlas.unverified")
        assert "shop.undocumented" in r["without_contracts"], r
        return (f"{len(r['without_contracts'])}/{r['total_functions']} "
                f"functions have no contracts")
    case("the interface reports what cannot be verified", t_unverified)

    print("\naudit and errors")

    def t_audit():
        r = call("audit.tail")
        actions = {x["action"] for x in r["records"]}
        assert r["intact"], r
        assert "promotion.approved" in actions, actions
        return f"{len(r['records'])} records, chain intact"
    case("the session keeps a verifiable audit trail", t_audit)

    def t_unknown_method():
        r = call("nonsense.method")
        assert not r["ok"] and "unknown method" in r["error"], r
        return r["error"]
    case("an unknown method fails without killing the session",
         t_unknown_method)

    def t_missing_param():
        r = call("atlas.blast")
        assert not r["ok"], r
        assert r["detail"]["parameter"] == "target", r
        return r["error"]
    case("a missing parameter is reported structurally", t_missing_param)

    print("\ntransport")

    def t_serve():
        requests = [
            {"id": 1, "method": "workspace.open", "params": {"paths": [path]}},
            {"id": 2, "method": "atlas.blast", "params": {"target": "line_total"}},
            {"id": 3, "method": "session.close"},
        ]
        stdin = io.StringIO("\n".join(json.dumps(r) for r in requests) + "\n")
        stdout = io.StringIO()
        code = serve(stdin, stdout)
        lines = [json.loads(l) for l in stdout.getvalue().splitlines()]
        assert code == 0, code
        assert lines[0]["protocol"], lines[0]
        assert lines[1]["id"] == 1 and lines[1]["ok"], lines[1]
        assert lines[2]["id"] == 2 and lines[2]["ok"], lines[2]
        assert lines[3]["closed"], lines[3]
        return f"{len(lines)} responses over a line-delimited stream"
    case("a full session runs over stdin and stdout", t_serve)

    def t_bad_json():
        stdin = io.StringIO("{not json\n" + json.dumps(
            {"id": 1, "method": "session.close"}) + "\n")
        stdout = io.StringIO()
        serve(stdin, stdout)
        lines = [json.loads(l) for l in stdout.getvalue().splitlines()]
        assert not lines[1]["ok"] and "invalid JSON" in lines[1]["error"]
        return "malformed input answered without dropping the session"
    case("malformed input does not end the session", t_bad_json)

    print("\nRESULT:", "pass" if not failures else f"FAIL ({failures})")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
