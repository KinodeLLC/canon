"""
The `canon` command.

One entry point for the whole family: the loader dispatches on file extension,
so `canon check src/` type-checks a directory holding Canon, Verdict, Loom,
Weft, Tract, Rune and Intent side by side and reports against one set of
diagnostics.

Every subcommand takes `--json`, because the primary caller is a program. The
human rendering is generated from the same structure, not the other way round.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

from . import LANGUAGE_VERSION, __version__
from . import values as V
from .canonical import IR_VERSION, Hasher, format_module, short
from .diagnostics import Bag
from .sources import Workspace, known_extensions, load


EXIT_OK = 0
EXIT_DIAGNOSTICS = 1
EXIT_USAGE = 2


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _emit(args, payload: dict, text: Optional[str] = None) -> int:
    if args.json:
        print(json.dumps(payload, indent=2, default=str, sort_keys=True))
    elif text is not None:
        print(text)
    return EXIT_OK if payload.get("ok", True) else EXIT_DIAGNOSTICS


def _report(ws: Workspace, bag: Bag, args) -> Optional[int]:
    """Print diagnostics. Returns an exit code if there were errors."""
    errors = [d for d in bag if d.severity.value == "error"]
    warnings = [d for d in bag if d.severity.value == "warning"]
    shown = errors + (warnings if getattr(args, "warnings", False) else [])

    if args.json:
        if errors or shown:
            print(json.dumps({"ok": not errors,
                              "diagnostics": [d.to_json() for d in shown]},
                             indent=2, default=str))
    else:
        for d in shown:
            src = ws.sources.get(d.span.start.file)
            print(d.render(src, color=_color()))
    return EXIT_DIAGNOSTICS if errors else None


def _color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


def _load(args):
    ws = load(args.paths)
    return ws


def _check(args):
    """Load and type-check. Returns (workspace, check result or None, code)."""
    ws = _load(args)
    code = _report(ws, ws.bag, args)
    if code is not None:
        return ws, None, code
    from .checker import check
    cr = check(ws.modules, ws.bag)
    code = _report(ws, cr.bag, args)
    return ws, cr, code


# --------------------------------------------------------------------------
# Subcommands
# --------------------------------------------------------------------------

def cmd_check(args) -> int:
    ws, cr, code = _check(args)
    if code is not None:
        return code
    counts = ws.summary()
    warnings = [d for d in cr.bag if d.severity.value == "warning"]
    payload = {"ok": True, **counts, "warnings": len(warnings),
               "definitions": len(Hasher().add_modules(cr.modules))}
    text = (f"ok: {counts['files']} files, {payload['definitions']} "
            f"definitions, {len(warnings)} warnings")
    return _emit(args, payload, text)


def cmd_fmt(args) -> int:
    ws = _load(args)
    code = _report(ws, ws.bag, args)
    if code is not None:
        return code

    changed, unchanged = [], []
    for mod in ws.modules:
        name = mod.source_file
        original = ws.sources.get(name, "")
        canonical = format_module(mod)
        if canonical == original:
            unchanged.append(name)
            continue
        changed.append(name)
        if args.write:
            with open(name, "w", encoding="utf-8", newline="\n") as f:
                f.write(canonical)
        elif not args.json:
            print(canonical, end="")

    payload = {"ok": True, "changed": changed, "unchanged": unchanged,
               "written": bool(args.write)}
    text = (f"{len(changed)} file(s) reformatted, {len(unchanged)} already "
            f"canonical" + (" (written)" if args.write else ""))
    if args.check and changed:
        payload["ok"] = False
        return _emit(args, payload,
                     f"not canonical: {', '.join(changed)}") or EXIT_DIAGNOSTICS
    return _emit(args, payload, text if args.write or args.json else None)


def cmd_hash(args) -> int:
    ws, cr, code = _check(args)
    if code is not None:
        return code
    defs = Hasher().add_modules(cr.modules)
    rows = []
    for qn in sorted(defs):
        d = defs[qn]
        rows.append({"definition": qn, "kind": d.kind, "hash": d.hash,
                     "local_hash": d.local_hash, "language": d.origin,
                     "effects": sorted(d.effects),
                     "deps": sorted(d.deps)})
    lines = [f"{short(r['hash'])}  {r['kind']:7} {r['definition']}"
             for r in rows]
    return _emit(args, {"ok": True, "definitions": rows}, "\n".join(lines))


def cmd_test(args) -> int:
    from .interp import run_tests
    from .ledger import AuditLog, CapabilityBroker, Ledger

    ws, cr, code = _check(args)
    if code is not None:
        return code

    audit = AuditLog(actor="cli")
    broker = CapabilityBroker(audit=audit)
    broker.grant("cli", args.grant or ["*"], reason="declared test run")
    led = Ledger(broker=broker, audit=audit, actor="cli")

    results = run_tests(cr, led)
    passed = [r for r in results if r.get("passed")]
    lines = []
    for r in results:
        mark = "ok  " if r["passed"] else "FAIL"
        detail = ""
        if not r["passed"]:
            detail = "  " + (r.get("fault", {}).get("message")
                             or f"returned {r.get('value', 'false')}")
        lines.append(f"  {mark} {r['module']}: {r['name']}{detail}")
    lines.append(f"{len(passed)}/{len(results)} tests passed")

    payload = {"ok": len(passed) == len(results), "total": len(results),
               "passed": len(passed), "results": results}
    return _emit(args, payload, "\n".join(lines))


def cmd_verify(args) -> int:
    from .verifier import Verifier

    ws, cr, code = _check(args)
    if code is not None:
        return code
    hashes = {qn: di.hash for qn, di in Hasher().add_modules(cr.modules).items()}
    v = Verifier(cr, seed=args.seed, runs=args.runs, hashes=hashes)
    report = v.verify_all(args.only or None)
    return _emit(args, {"ok": report.ok, **report.to_json()}, report.render())


def cmd_run(args) -> int:
    from .interp import Budget, Fault, Interpreter
    from .ledger import AuditLog, CapabilityBroker, Ledger
    from .model import DeterministicProvider, ModelRuntime

    ws, cr, code = _check(args)
    if code is not None:
        return code

    try:
        arguments = [V.from_json(json.loads(a)) for a in args.arguments]
    except json.JSONDecodeError as e:
        return _emit(args, {"ok": False,
                            "error": f"arguments must be JSON values: {e}"},
                     f"invalid argument: {e}") or EXIT_USAGE

    audit = AuditLog(actor="cli")
    broker = CapabilityBroker(audit=audit)
    if args.grant:
        broker.grant("cli", args.grant, reason="explicit --grant")
    led = Ledger(broker=broker, audit=audit, actor="cli",
                 model_runtime=ModelRuntime(DeterministicProvider(), cr.env))
    budget = Budget(steps=args.steps, io=args.io)
    it = Interpreter(cr, led, budget, hashes={
        qn: di.hash for qn, di in Hasher().add_modules(cr.modules).items()})

    try:
        value = it.call(args.function, arguments)
    except Fault as f:
        payload = {"ok": False, "fault": f.to_json(),
                   "journal": [e.to_json() for e in led.journal],
                   "cost": budget.snapshot()}
        return _emit(args, payload, f"{f.code}: {f.message}") \
            or EXIT_DIAGNOSTICS

    if args.journal:
        led.journal.save(args.journal)

    payload = {"ok": True, "result": V.to_json(value),
               "rendered": V.show(value), "cost": budget.snapshot(),
               "effects": led.journal.ops(),
               "journal_head": led.journal.head()}
    return _emit(args, payload, V.show(value))


def cmd_atlas(args) -> int:
    from .atlas import Atlas

    ws, cr, code = _check(args)
    if code is not None:
        return code
    defs = Hasher().add_modules(cr.modules)
    atlas = Atlas(cr, defs, sources=ws.sources)

    q = args.query
    if q == "stats":
        s = atlas.stats()
        text = (f"{s['definitions']} definitions across "
                f"{len(s['modules'])} modules\n"
                f"  by kind: {s['by_kind']}\n"
                f"  by language: {s['by_language']}\n"
                f"  effects in use: {', '.join(s['effects_in_use']) or '(none)'}")
        return _emit(args, {"ok": True, **s}, text)

    if q == "unverified":
        u = atlas.unverified()
        text = (f"{len(u['without_contracts'])}/{u['total_functions']} "
                f"functions have no contracts\n"
                f"{len(u['without_intent'])} have no stated intent")
        return _emit(args, {"ok": True, **u}, text)

    if q == "search":
        hits = atlas.search(args.target or "", args.limit)
        text = "\n".join(f"{short(h.hash)}  {h.qualname}: {h.signature}"
                         for h in hits)
        return _emit(args, {"ok": True,
                            "results": [h.summary() for h in hits]}, text)

    if args.target is None:
        return _emit(args, {"ok": False,
                            "error": f"query {q!r} needs a target"},
                     f"query {q!r} needs a target") or EXIT_USAGE

    if q == "show":
        node = atlas.resolve(args.target)
        if node is None:
            return _emit(args, {"ok": False, "error": "unknown definition"},
                         f"unknown definition {args.target!r}") \
                or EXIT_DIAGNOSTICS
        return _emit(args, {"ok": True, "definition": node.to_json(),
                            "text": atlas.render(node, args.detail)},
                     atlas.render(node, args.detail))

    if q == "view":
        v = atlas.view(args.target, args.budget)
        text = v["text"]
        if v["omitted"]:
            text += f"\n\n-- omitted for budget: {v['omitted']}"
        return _emit(args, {"ok": True, **v}, text)

    if q in ("callers", "calls"):
        fn = atlas.callers if q == "callers" else atlas.calls
        items = fn(args.target, args.transitive)
        return _emit(args, {"ok": True, q: items}, "\n".join(items))

    if q == "blast":
        r = atlas.blast_radius(args.target)
        if "error" in r:
            return _emit(args, {"ok": False, **r}, r["error"]) \
                or EXIT_DIAGNOSTICS
        text = (f"{r['definition']} {short(r['hash'])}\n"
                f"  {r['count']} transitive dependents\n"
                f"  capabilities: {', '.join(r['capabilities']) or '(none)'}\n"
                f"  data: {', '.join(r['data_classifications']) or '(none)'}")
        return _emit(args, {"ok": True, **r}, text)

    if q == "caps":
        r = atlas.capabilities(args.target)
        if "error" in r:
            return _emit(args, {"ok": False, **r}, r["error"]) \
                or EXIT_DIAGNOSTICS
        lines = [f"{r['definition']}",
                 f"  declared:   {', '.join(r['declared']) or '(pure)'}",
                 f"  transitive: {', '.join(r['transitive']) or '(pure)'}"]
        for src, effects in sorted(r["granted_by"].items()):
            lines.append(f"    via {src}: {', '.join(effects)}")
        return _emit(args, {"ok": True, **r}, "\n".join(lines))

    if q == "effect-users":
        items = atlas.effect_users(args.target)
        return _emit(args, {"ok": True, "definitions": items},
                     "\n".join(items))

    return _emit(args, {"ok": False, "error": f"unknown query {q!r}"},
                 f"unknown query {q!r}") or EXIT_USAGE


def cmd_diff(args) -> int:
    from .checker import check
    from .shadow import differential, structural_diff

    old_ws = load([args.old])
    new_ws = load([args.new])
    for ws in (old_ws, new_ws):
        if ws.bag.has_errors:
            _report(ws, ws.bag, args)
            return EXIT_DIAGNOSTICS
    old_cr = check(old_ws.modules, old_ws.bag)
    new_cr = check(new_ws.modules, new_ws.bag)
    for ws, cr in ((old_ws, old_cr), (new_ws, new_cr)):
        if cr.bag.has_errors:
            _report(ws, cr.bag, args)
            return EXIT_DIAGNOSTICS

    diff = structural_diff(old_cr, new_cr)
    targets = [c.qualname for c in diff.changed()
               if c.kind == "changed" and c.qualname in new_cr.env.fns]
    behaviour = differential(old_cr, new_cr, targets, args.seed,
                             args.runs).to_json() if targets else None

    lines = []
    for c in diff.changed():
        mark = {"added": "+", "removed": "-", "changed": "~"}[c.kind]
        note = "" if c.body_changed or c.kind != "changed" \
            else "  (dependency only)"
        lines.append(f" {mark} {c.qualname}{note}")
    if diff.capabilities_added:
        lines.append(f" ! capabilities added: "
                     f"{', '.join(diff.capabilities_added)}")
    if behaviour and not behaviour["identical"]:
        for d in behaviour["disagreements"]:
            lines.append(f" ! {d['function']}"
                         f"({', '.join(d['rendered'])}): {d['summary']}")
    lines.append(f"{len(diff.changed())} definitions changed, blast radius "
                 f"{len(diff.blast_radius)}")

    payload = {"ok": True, "diff": diff.to_json(), "behaviour": behaviour}
    return _emit(args, payload, "\n".join(lines))


def cmd_serve(args) -> int:
    from .protocol import serve
    return serve(paths=args.paths or None, actor=args.actor)


def cmd_journal(args) -> int:
    from .ledger import Journal
    j = Journal.load(args.path)
    problems = j.verify()
    lines = [f"{len(j)} entries, head {short(j.head())}"]
    for e in j:
        lines.append(f"  {e.seq:4}  {e.op:24} {short(e.def_hash)}")
    if problems:
        lines.append(f"INTEGRITY FAILURE: {len(problems)} problem(s)")
        for p in problems:
            lines.append(f"  seq {p['seq']}: {p['problem']}")
    else:
        lines.append("hash chain intact")
    payload = {"ok": not problems, "entries": len(j), "head": j.head(),
               "problems": problems,
               "operations": j.ops() if args.verbose else None}
    return _emit(args, payload, "\n".join(lines))


# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="canon",
        description="The Canon language family: check, verify and evolve "
                    "programs written for agent authorship.",
        epilog="source extensions: " + ", ".join(known_extensions()))
    p.add_argument("--version", action="version",
                   version=f"canon {__version__} "
                           f"(language {LANGUAGE_VERSION}, IR {IR_VERSION})")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp, paths=True):
        if paths:
            sp.add_argument("paths", nargs="+",
                            help="source files or directories")
        sp.add_argument("--json", action="store_true",
                        help="emit structured output")
        sp.add_argument("-w", "--warnings", action="store_true",
                        help="show warnings as well as errors")
        return sp

    c = common(sub.add_parser("check", help="parse and type-check"))
    c.set_defaults(fn=cmd_check)

    f = common(sub.add_parser("fmt", help="render canonical source"))
    f.add_argument("--write", action="store_true", help="rewrite files in place")
    f.add_argument("--check", action="store_true",
                   help="fail if any file is not already canonical")
    f.set_defaults(fn=cmd_fmt)

    h = common(sub.add_parser("hash", help="show content hashes"))
    h.set_defaults(fn=cmd_hash)

    t = common(sub.add_parser("test", help="run declared tests"))
    t.add_argument("--grant", action="append",
                   help="capability to grant (repeatable)")
    t.set_defaults(fn=cmd_test)

    v = common(sub.add_parser("verify", help="check contracts and laws"))
    v.add_argument("--seed", default="canon", help="generation seed")
    v.add_argument("--runs", type=int, default=60, help="cases per function")
    v.add_argument("--only", action="append", help="limit to a definition")
    v.set_defaults(fn=cmd_verify)

    r = common(sub.add_parser("run", help="call a function"))
    r.add_argument("--function", required=True, help="function to call")
    r.add_argument("--arguments", nargs="*", default=[],
                   help="JSON-encoded arguments")
    r.add_argument("--grant", action="append",
                   help="capability to grant (repeatable)")
    r.add_argument("--steps", type=int, default=1_000_000)
    r.add_argument("--io", type=int, default=100)
    r.add_argument("--journal", help="write the effect journal to this path")
    r.set_defaults(fn=cmd_run)

    # The query comes before the paths so the command reads as a question:
    # `canon atlas blast lending.refund src/`.
    a = sub.add_parser("atlas", help="query the definition graph")
    a.add_argument("query",
                   choices=["stats", "search", "show", "view", "callers",
                            "calls", "blast", "caps", "effect-users",
                            "unverified"])
    a.add_argument("target", nargs="?", help="definition, hash or search text")
    a.add_argument("paths", nargs="+", help="source files or directories")
    a.add_argument("--json", action="store_true",
                   help="emit structured output")
    a.add_argument("-w", "--warnings", action="store_true",
                   help="show warnings as well as errors")
    a.add_argument("--detail", default="full",
                   choices=["name", "signature", "contract", "full"])
    a.add_argument("--budget", type=int, default=4000,
                   help="token budget for view")
    a.add_argument("--transitive", action="store_true")
    a.add_argument("--limit", type=int, default=20)
    a.set_defaults(fn=cmd_atlas)

    d = sub.add_parser("diff", help="compare two versions")
    d.add_argument("old", help="path to the current version")
    d.add_argument("new", help="path to the proposed version")
    d.add_argument("--seed", default="canon")
    d.add_argument("--runs", type=int, default=40)
    d.add_argument("--json", action="store_true")
    d.add_argument("-w", "--warnings", action="store_true")
    d.set_defaults(fn=cmd_diff)

    s = sub.add_parser("serve", help="run the agent interface on stdin/stdout")
    s.add_argument("paths", nargs="*", help="sources to open at startup")
    s.add_argument("--actor", default="agent")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_serve)

    j = sub.add_parser("journal", help="inspect an effect journal")
    j.add_argument("path")
    j.add_argument("--verbose", action="store_true")
    j.add_argument("--json", action="store_true")
    j.set_defaults(fn=cmd_journal)

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.fn(args)
    except KeyboardInterrupt:
        return 130
    except BrokenPipeError:
        return 0


if __name__ == "__main__":
    sys.exit(main())
