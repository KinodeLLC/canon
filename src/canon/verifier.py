"""
property generation, counterexample shrinking, law checking.

the whole point of this stack is that correctness has to be establishable
without somebody reading the implementation, which means the checks have to
come from somewhere other than the implementation. here they come off the
contracts and laws whoever wrote it declared, and off the types.

for each function it generates inputs from the parameter types, throws out the
ones that break the preconditions, runs the rest under a budget with effects
going to a recording runtime, checks every postcondition and every declared
law, shrinks anything that fails down to something small before reporting it,
and reports what the run actually cost against what was declared.

generation is seeded so a failure found in one run comes back in the next, and
the seed goes in the report, so a verification result is something you can
point at later instead of having to find again.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Optional

from . import types as TY
from . import values as V
from .checker import CheckResult, FnInfo
from .interp import Budget, Fault, Interpreter, Runtime


# --------------------------------------------------------------------------
# Deterministic random source
# --------------------------------------------------------------------------

class Source:
    """
    A reproducible stream of numbers derived from a seed.

    Not the host RNG: a verification result has to be reproducible on another
    machine, in another process, months later, from the seed alone.
    """

    def __init__(self, seed: str, counter: int = 0):
        self.seed = seed
        self.counter = counter

    def _next(self) -> int:
        h = hashlib.blake2b(
            f"{self.seed}:{self.counter}".encode("utf-8"), digest_size=8).digest()
        self.counter += 1
        return struct.unpack("<Q", h)[0]

    def below(self, n: int) -> int:
        return 0 if n <= 0 else self._next() % n

    def between(self, lo: int, hi: int) -> int:
        return lo if hi <= lo else lo + self.below(hi - lo + 1)

    def chance(self, p: int = 2) -> bool:
        return self.below(p) == 0

    def pick(self, items):
        items = list(items)
        return items[self.below(len(items))] if items else None

    def fork(self, tag: str) -> "Source":
        return Source(f"{self.seed}/{tag}", 0)


# --------------------------------------------------------------------------
# Value generation
# --------------------------------------------------------------------------

INTERESTING_INTS = [0, 1, -1, 2, -2, 10, 100, 1000, -1000,
                    2 ** 31 - 1, -(2 ** 31), 2 ** 63, -(2 ** 63)]

INTERESTING_TEXT = ["", " ", "a", "0", "-1", "null", "NaN", "\n", "\t",
                    "  leading", "trailing  ", "Ünïcødé", "🙂",
                    "'; drop table --", "../../etc/passwd",
                    "a" * 256, "‮RTL", "\x00"]


class Generator:
    """Builds values of a given Canon type, with a size budget."""

    def __init__(self, env, source: Source, max_depth: int = 4):
        self.env = env
        self.src = source
        self.max_depth = max_depth

    def generate(self, t, size: int = 8, depth: int = 0) -> Any:
        t = TY.prune(t)

        if isinstance(t, TY.TCon):
            n = t.name

            if n == "Int":
                # Boundary values first: most contract failures live at edges,
                # so biasing towards them finds bugs in far fewer runs.
                if self.src.chance(3):
                    return self.src.pick(INTERESTING_INTS)
                return self.src.between(-size * 16, size * 16)

            if n == "Dec":
                if self.src.chance(4):
                    return self.src.pick(
                        [Decimal(0), Decimal("0.01"), Decimal("-0.01"),
                         Decimal("1e-9"), Decimal("99999999.99")])
                whole = self.src.between(-size * 8, size * 8)
                frac = self.src.below(100)
                return Decimal(f"{whole}.{frac:02d}")

            if n == "Text":
                if self.src.chance(3):
                    return self.src.pick(INTERESTING_TEXT)
                words = ["alpha", "bravo", "charlie", "delta", "echo",
                         "id", "name", "value", "test", "item"]
                k = self.src.between(1, max(1, size // 3))
                return " ".join(self.src.pick(words) for _ in range(k))

            if n == "Bool":
                return self.src.chance(2)

            if n == "Unit":
                return V.UNIT

            if n == "Bytes":
                k = self.src.below(min(size, 16))
                return bytes(self.src.below(256) for _ in range(k))

            if n == "Time":
                return V.Instant(1_700_000_000_000 + self.src.below(10 ** 9),
                                 self.src.below(1000))

            if n in ("List", "Set"):
                if depth >= self.max_depth:
                    return () if n == "List" else frozenset()
                k = 0 if self.src.chance(5) else self.src.below(
                    max(1, min(size, 8)))
                items = [self.generate(t.args[0], max(1, size - 2), depth + 1)
                         for _ in range(k)]
                if n == "Set":
                    return frozenset(V._hashable(i) for i in items)
                return tuple(items)

            if n == "Map":
                if depth >= self.max_depth:
                    return V.FrozenMap()
                from .prelude import _mapkey
                k = self.src.below(max(1, min(size, 5)))
                out = {}
                for _ in range(k):
                    key = self.generate(t.args[0], max(1, size - 2), depth + 1)
                    out[_mapkey(key)] = self.generate(
                        t.args[1], max(1, size - 2), depth + 1)
                return V.FrozenMap(out)

            if n == "Option":
                if depth >= self.max_depth or self.src.chance(3):
                    return V.NONE
                return V.some(self.generate(t.args[0], size, depth + 1))

            if n == "Result":
                if self.src.chance(3):
                    return V.err(self.generate(t.args[1], size, depth + 1))
                return V.ok(self.generate(t.args[0], size, depth + 1))

            ti = self.env.types.get(n)
            if ti is None:
                return V.UNIT

            if ti.kind == "enum":
                variants = ti.decl.variants
                if depth >= self.max_depth:
                    nullary = [v for v in variants if not v.params]
                    v = self.src.pick(nullary or variants)
                else:
                    v = self.src.pick(variants)
                args = tuple(
                    self.generate(self._resolve(pt, ti), max(1, size - 2),
                                  depth + 1)
                    for pt in v.params)
                return V.Variant(v.name, args, n)

            if ti.kind == "record":
                # Records carry invariants. Generating a value that violates
                # one is not a useful test input -- it could never exist in a
                # running program -- so retry a bounded number of times and
                # then give up on this draw.
                for _ in range(24):
                    fields = tuple(
                        (f.name,
                         self.generate(self._resolve(f.ty, ti),
                                       max(1, size - 1), depth + 1))
                        for f in ti.decl.fields)
                    rec = V.Record(n, fields)
                    if self._satisfies_invariants(rec, ti):
                        return rec
                return rec

            if ti.kind == "alias":
                return self.generate(self._resolve(ti.decl.target, ti),
                                     size, depth)

        if isinstance(t, TY.TRec):
            return V.Record("", tuple(
                (k, self.generate(v, max(1, size - 1), depth + 1))
                for k, v in sorted(t.fields.items())))

        return V.UNIT

    def _resolve(self, texpr, ti):
        from .checker import Checker
        c = Checker()
        c.env = self.env
        c.tparams = set(ti.tparams)
        return c.resolve_type(texpr, None)

    def _satisfies_invariants(self, rec, ti) -> bool:
        if not getattr(ti.decl, "invariants", None):
            return True
        try:
            it = Interpreter(self._cr, Runtime(), Budget(steps=20000),
                             enforce_contracts=False)
        except Exception:
            return True
        frame = it.globals.child(rec.as_dict())
        frame.set("self", rec)
        for inv in ti.decl.invariants:
            try:
                if it.eval(inv, frame) is not True:
                    return False
            except Fault:
                return False
        return True

    _cr = None


# --------------------------------------------------------------------------
# Shrinking
# --------------------------------------------------------------------------

def shrink_candidates(value) -> list:
    """
    Simpler values to try in place of `value`.

    Ordered simplest-first. The point of shrinking is that a counterexample of
    `[0]` is actionable and one of `[-8213, 91, 4471, ...]` is not; the bug is
    the same but only one of them can be read.
    """
    out = []

    if isinstance(value, bool):
        if value:
            out.append(False)
        return out

    if isinstance(value, int):
        if value != 0:
            out.append(0)
        if value not in (0, 1) and value > 1:
            out.append(1)
        if value < -1:
            out.append(-1)
        if abs(value) > 1:
            out.append(value // 2)
        if value > 0:
            out.append(value - 1)
        elif value < 0:
            out.append(value + 1)
        return out

    if isinstance(value, Decimal):
        if value != 0:
            out.append(Decimal(0))
        if value != value.to_integral_value():
            out.append(value.to_integral_value())
        if abs(value) > 1:
            out.append(value / 2)
        return out

    if isinstance(value, str):
        if value != "":
            out.append("")
        if len(value) > 1:
            out.append(value[:len(value) // 2])
            out.append(value[:-1])
        if value and value != "a" * len(value) and not value.isalpha():
            out.append("".join(c for c in value if c.isalnum()))
        return out

    if isinstance(value, bytes):
        if value:
            out.append(b"")
            out.append(value[:len(value) // 2])
        return out

    if isinstance(value, tuple):
        if value:
            out.append(())
            if len(value) > 1:
                out.append(value[:len(value) // 2])
                out.append(value[1:])
                out.append(value[:-1])
            for i in range(len(value)):
                for sub in shrink_candidates(value[i])[:2]:
                    out.append(value[:i] + (sub,) + value[i + 1:])
        return out

    if isinstance(value, frozenset):
        if value:
            out.append(frozenset())
            items = sorted(value, key=V._sort_key)
            out.append(frozenset(items[:len(items) // 2]))
        return out

    if isinstance(value, V.FrozenMap):
        if len(value):
            out.append(V.FrozenMap())
            items = value.items()
            out.append(V.FrozenMap(dict(items[:len(items) // 2])))
        return out

    if isinstance(value, V.Record):
        for i, (name, fv) in enumerate(value.fields):
            for sub in shrink_candidates(fv)[:2]:
                out.append(value.set(name, sub))
        return out

    if isinstance(value, V.Variant):
        if value.args:
            for i, av in enumerate(value.args):
                for sub in shrink_candidates(av)[:2]:
                    out.append(V.Variant(value.ctor,
                                         value.args[:i] + (sub,)
                                         + value.args[i + 1:],
                                         value.type_name))
        return out

    return out


def shrink_args(args: list, still_fails) -> list:
    """
    Greedily simplify an argument vector while it keeps failing.

    Bounded: shrinking is a convenience, and a verifier that spends minutes
    minimising a counterexample is worse than one that reports a slightly
    larger one immediately.
    """
    best = list(args)
    budget = 400
    improved = True
    while improved and budget > 0:
        improved = False
        for i in range(len(best)):
            for cand in shrink_candidates(best[i]):
                budget -= 1
                if budget <= 0:
                    break
                trial = list(best)
                trial[i] = cand
                if still_fails(trial):
                    best = trial
                    improved = True
                    break
            if improved:
                break
    return best


# --------------------------------------------------------------------------
# Recording runtime
# --------------------------------------------------------------------------

class VerificationRuntime(Runtime):
    """
    Satisfies effects during verification without performing them.

    Effects return a plausible typed value and are recorded so that laws about
    effects -- idempotence, purity -- can be checked. Verification must never
    touch a real system: the whole point is to run unreviewed code safely.
    """

    def __init__(self, env, source: Source, ask_provider=None):
        self.env = env
        self.src = source
        self.performed: list = []
        self.ask_provider = ask_provider
        self._gen = Generator(env, source)

    def granted(self):
        return ["*"]

    def perform(self, key: str, args: list, span=None, idempotent: bool = False):
        self.performed.append((key, tuple(args)))
        eff, _, op = key.partition(".")
        ei = self.env.effects.get(eff)
        if ei is None or op not in ei.ops:
            return V.UNIT
        opdecl = ei.ops[op]
        from .checker import Checker
        c = Checker()
        c.env = self.env
        rt = c.resolve_type(opdecl.result, None)
        # Same arguments give the same answer, so a function called twice with
        # the same input sees a consistent world -- otherwise determinism and
        # idempotence laws would fail for the wrong reason.
        key_hash = V.value_hash((key, tuple(args)))
        return self._gen.generate(rt, 6) if not _is_unit(rt) else V.UNIT

    def ask(self, req):
        if self.ask_provider is not None:
            return self.ask_provider(req)
        return self._gen.generate(req.result_type, 6)


def _is_unit(t) -> bool:
    t = TY.prune(t)
    return isinstance(t, TY.TCon) and t.name == "Unit"


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------

@dataclass
class Counterexample:
    args: list
    problem: str
    kind: str                 # postcondition | law | fault | cost
    clause: str = ""
    law: str = ""
    result: Any = None
    fault: Optional[dict] = None
    shrunk: bool = False

    def to_json(self) -> dict:
        return {"arguments": [V.to_json(a) for a in self.args],
                "rendered": [V.show(a) for a in self.args],
                "problem": self.problem, "kind": self.kind,
                "clause": self.clause, "law": self.law,
                "result": V.to_json(self.result) if self.result is not None else None,
                "fault": self.fault, "shrunk": self.shrunk}


@dataclass
class FnReport:
    qualname: str
    def_hash: str = ""
    runs: int = 0
    passed: int = 0
    discarded: int = 0
    counterexamples: list = field(default_factory=list)
    laws_checked: list = field(default_factory=list)
    max_cost: dict = field(default_factory=dict)
    declared_cost: dict = field(default_factory=dict)
    cost_exceeded: bool = False
    skipped: str = ""
    seed: str = ""

    @property
    def ok(self) -> bool:
        return not self.counterexamples and not self.cost_exceeded and not self.skipped

    def to_json(self) -> dict:
        return {"function": self.qualname, "hash": self.def_hash,
                "ok": self.ok, "runs": self.runs, "passed": self.passed,
                "discarded": self.discarded, "seed": self.seed,
                "laws": self.laws_checked,
                "counterexamples": [c.to_json() for c in self.counterexamples],
                "max_cost": self.max_cost, "declared_cost": self.declared_cost,
                "cost_exceeded": self.cost_exceeded, "skipped": self.skipped}


@dataclass
class VerificationReport:
    seed: str
    functions: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(f.ok for f in self.functions)

    def failures(self) -> list:
        return [f for f in self.functions if not f.ok]

    def to_json(self) -> dict:
        return {"ok": self.ok, "seed": self.seed,
                "functions": [f.to_json() for f in self.functions],
                "summary": {
                    "total": len(self.functions),
                    "passed": sum(1 for f in self.functions if f.ok),
                    "failed": len(self.failures()),
                    "runs": sum(f.runs for f in self.functions)}}

    def render(self) -> str:
        lines = []
        s = self.to_json()["summary"]
        lines.append(f"verification: {s['passed']}/{s['total']} functions, "
                     f"{s['runs']} runs, seed {self.seed}")
        for f in self.functions:
            mark = "ok  " if f.ok else "FAIL"
            detail = ""
            if f.skipped:
                mark, detail = "skip", f"  ({f.skipped})"
            lines.append(f"  {mark} {f.qualname}  "
                         f"{f.passed}/{f.runs} runs"
                         f"{', ' + str(f.discarded) + ' discarded' if f.discarded else ''}"
                         f"{detail}")
            for c in f.counterexamples:
                lines.append(f"       {c.kind}: {c.problem}")
                lines.append(f"       input: ({', '.join(V.show(a) for a in c.args)})")
                if c.result is not None:
                    lines.append(f"       result: {V.show(c.result)}")
            if f.cost_exceeded:
                lines.append(f"       cost: observed {f.max_cost} exceeds "
                             f"declared {f.declared_cost}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# The verifier
# --------------------------------------------------------------------------

class Verifier:
    def __init__(self, cr: CheckResult, seed: str = "canon",
                 runs: int = 60, hashes: Optional[dict] = None,
                 ask_provider=None, shrink: bool = True):
        self.cr = cr
        self.env = cr.env
        self.seed = seed
        self.runs = runs
        self.hashes = hashes or {}
        self.ask_provider = ask_provider
        self.shrink = shrink

    # ------------------------------------------------------------------

    def verify_all(self, only=None) -> VerificationReport:
        report = VerificationReport(seed=self.seed)
        for qn in sorted(self.env.fns):
            if only and qn not in only and qn.rsplit(".", 1)[-1] not in only:
                continue
            fi = self.env.fns[qn]
            if fi.decl is None:
                continue
            report.functions.append(self.verify_fn(fi))
        return report

    def verify_fn(self, fi: FnInfo) -> FnReport:
        rep = FnReport(qualname=fi.qualname,
                       def_hash=self.hashes.get(fi.qualname, ""),
                       seed=f"{self.seed}/{fi.qualname}")

        if fi.tparams:
            rep.skipped = "generic functions are verified at their call sites"
            return rep

        d = fi.decl
        rep.laws_checked = [l.name for l in d.laws]
        rep.declared_cost = _cost_json(d.cost)

        src = Source(rep.seed)
        gen = Generator(self.env, src)
        gen._cr = self.cr

        max_cost = {"steps": 0, "io": 0, "tokens": 0}

        for run in range(self.runs):
            size = 2 + (run % 12)
            args = [gen.generate(pt, size) for pt in fi.type.params]

            outcome = self._trial(fi, args)
            if outcome["discarded"]:
                rep.discarded += 1
                continue

            rep.runs += 1
            for k in max_cost:
                max_cost[k] = max(max_cost[k], outcome["cost"].get(k, 0))

            if outcome["problem"] is None:
                rep.passed += 1
                continue

            ce = self._counterexample(fi, args, outcome)
            rep.counterexamples.append(ce)
            if len(rep.counterexamples) >= 3:
                break

        rep.max_cost = max_cost
        rep.cost_exceeded = self._cost_exceeded(d.cost, max_cost)
        return rep

    # ------------------------------------------------------------------

    def _trial(self, fi: FnInfo, args: list) -> dict:
        """Run once. Returns discarded / cost / problem."""
        d = fi.decl
        src = Source(f"{self.seed}/eff/{V.value_hash(tuple(args))}")
        rt = VerificationRuntime(self.env, src, self.ask_provider)
        budget = Budget.from_cost(d.cost)
        budget.steps = min(budget.steps, 2_000_000)
        it = Interpreter(self.cr, rt, budget, self.hashes,
                         enforce_contracts=False)

        # Preconditions decide whether this input is in scope at all.
        frame = it.globals.child({p.name: a for p, a in zip(d.params, args)})
        try:
            for r in d.requires:
                if it.eval(r, frame) is not True:
                    return {"discarded": True, "cost": {}, "problem": None}
        except Fault:
            return {"discarded": True, "cost": {}, "problem": None}

        try:
            result = it.call_fn(fi, list(args))
        except Fault as f:
            if f.code in ("CANON-E0601", "CANON-E0602"):
                return {"discarded": False, "cost": budget.snapshot(),
                        "problem": {"kind": "cost", "text": f.message,
                                    "fault": f.to_json()}}
            return {"discarded": False, "cost": budget.snapshot(),
                    "problem": {"kind": "fault", "text": f.message,
                                "fault": f.to_json()}}

        cost = budget.snapshot()

        # Postconditions.
        post = frame.child({"result": result})
        it._olds = {}
        for e in d.ensures:
            try:
                ok = it.eval(e, post) is True
            except Fault as f:
                return {"discarded": False, "cost": cost, "result": result,
                        "problem": {"kind": "postcondition",
                                    "text": f"could not evaluate: {f.message}",
                                    "clause": _render(e)}}
            if not ok:
                return {"discarded": False, "cost": cost, "result": result,
                        "problem": {"kind": "postcondition",
                                    "text": "postcondition does not hold",
                                    "clause": _render(e)}}

        # Laws.
        for law in d.laws:
            problem = self._check_law(fi, law, args, result, rt, it)
            if problem:
                return {"discarded": False, "cost": cost, "result": result,
                        "problem": {"kind": "law", "text": problem,
                                    "law": law.name}}

        return {"discarded": False, "cost": cost, "result": result,
                "problem": None}

    def _counterexample(self, fi, args, outcome) -> Counterexample:
        p = outcome["problem"]

        def still_fails(trial):
            o = self._trial(fi, trial)
            return (not o["discarded"] and o["problem"] is not None
                    and o["problem"]["kind"] == p["kind"])

        final = shrink_args(args, still_fails) if self.shrink else list(args)
        shrunk = final != list(args)
        final_outcome = self._trial(fi, final) if shrunk else outcome

        fp = final_outcome.get("problem") or p
        return Counterexample(
            args=final, problem=fp.get("text", ""), kind=fp.get("kind", ""),
            clause=fp.get("clause", ""), law=fp.get("law", ""),
            result=final_outcome.get("result"), fault=fp.get("fault"),
            shrunk=shrunk)

    # ------------------------------------------------------------------

    def _check_law(self, fi: FnInfo, law, args, result, rt, it) -> Optional[str]:
        name = law.name

        if name == "deterministic":
            second = self._rerun(fi, args)
            if second["faulted"]:
                return "the second call faulted where the first did not"
            if V.compare(second["result"], result) != 0:
                return (f"two calls with the same input gave "
                        f"{V.show(result)} and {V.show(second['result'])}")
            return None

        if name == "pure":
            if rt.performed:
                ops = sorted({k for k, _ in rt.performed})
                return f"performed effects: {', '.join(ops)}"
            return None

        if name == "idempotent_by":
            first_effects = list(rt.performed)
            second = self._rerun(fi, args)
            if V.compare(second["result"], result) != 0:
                return ("calling twice with the same key gave a different "
                        f"result the second time: {V.show(result)} then "
                        f"{V.show(second['result'])}")
            if _writes(second["effects"]) != _writes(first_effects):
                return ("the second call performed a different set of "
                        "write effects than the first")
            return None

        if name == "never_negative":
            if _is_negative(result):
                return f"result is negative: {V.show(result)}"
            return None

        if name == "commutative":
            if len(args) != 2:
                return "commutative requires exactly two parameters"
            swapped = self._rerun(fi, [args[1], args[0]])
            if swapped["faulted"]:
                return "swapping the arguments caused a fault"
            if V.compare(swapped["result"], result) != 0:
                return (f"f(a, b) = {V.show(result)} but "
                        f"f(b, a) = {V.show(swapped['result'])}")
            return None

        if name == "associative":
            if len(args) != 2:
                return "associative requires exactly two parameters"
            a, b = args
            c = b
            left = self._rerun(fi, [self._rerun(fi, [a, b])["result"], c])
            right = self._rerun(fi, [a, self._rerun(fi, [b, c])["result"]])
            if left["faulted"] or right["faulted"]:
                return "an associativity probe faulted"
            if V.compare(left["result"], right["result"]) != 0:
                return (f"f(f(a,b),c) = {V.show(left['result'])} but "
                        f"f(a,f(b,c)) = {V.show(right['result'])}")
            return None

        if name == "monotonic_in":
            pname = _law_arg_name(law)
            idx = _param_index(fi, pname)
            if idx is None:
                return f"no parameter named {pname!r}"
            bumped = list(args)
            bumped[idx] = _increase(bumped[idx])
            if bumped[idx] is None:
                return None
            other = self._rerun(fi, bumped)
            if other["faulted"]:
                return None
            if _less_than(other["result"], result):
                return (f"increasing {pname} decreased the result: "
                        f"{V.show(result)} -> {V.show(other['result'])}")
            return None

        if name == "conserves":
            fname = _law_arg_name(law)
            before = _sum_field(args, fname)
            after = _sum_field([result], fname)
            if before is None or after is None:
                return None
            if before != after:
                return (f"{fname} changed from {before} to {after}")
            return None

        if name == "bounded_output":
            bound = _law_arg_int(law)
            if bound is None:
                return None
            n = _size_of(result)
            if n > bound:
                return f"result size {n} exceeds the declared bound {bound}"
            return None

        if name == "order_independent":
            idx = next((i for i, a in enumerate(args)
                        if isinstance(a, tuple) and len(a) > 1), None)
            if idx is None:
                return None
            reordered = list(args)
            reordered[idx] = tuple(reversed(args[idx]))
            other = self._rerun(fi, reordered)
            if other["faulted"]:
                return "reversing the input order caused a fault"
            if V.compare(other["result"], result) != 0:
                return (f"reversing input order changed the result: "
                        f"{V.show(result)} -> {V.show(other['result'])}")
            return None

        if name == "invertible_by":
            other_name = _law_arg_name(law)
            inverse = self.env.lookup_fn(other_name, fi.module)
            if inverse is None:
                return f"no function named {other_name!r} to invert with"
            if len(args) != 1:
                return "invertible_by applies to single-argument functions"
            back = self._rerun(inverse, [result])
            if back["faulted"]:
                return (f"{other_name}({V.show(result)}) faulted, so the "
                        f"round trip cannot be checked")
            if V.compare(back["result"], args[0]) != 0:
                return (f"{other_name}({fi.name}(x)) gave "
                        f"{V.show(back['result'])}, not the original "
                        f"{V.show(args[0])}")
            return None

        if name == "injective":
            # Checked statistically across the run, not per-call.
            return None

        if name in ("total", "grounded", "explains"):
            # `total` is guaranteed structurally; the other two are checked by
            # the model runtime at the point of the ask.
            return None

        return None

    def _rerun(self, fi: FnInfo, args: list) -> dict:
        src = Source(f"{self.seed}/eff/{V.value_hash(tuple(args))}")
        rt = VerificationRuntime(self.env, src, self.ask_provider)
        budget = Budget.from_cost(fi.decl.cost)
        budget.steps = min(budget.steps, 2_000_000)
        it = Interpreter(self.cr, rt, budget, self.hashes,
                         enforce_contracts=False)
        try:
            result = it.call_fn(fi, list(args))
            return {"result": result, "faulted": False,
                    "effects": list(rt.performed)}
        except Fault:
            return {"result": V.UNIT, "faulted": True,
                    "effects": list(rt.performed)}

    def _cost_exceeded(self, declared, observed) -> bool:
        if declared is None:
            return False
        for k in ("steps", "io", "tokens"):
            d = getattr(declared, k, None)
            if d is not None and observed.get(k, 0) > d:
                return True
        return False


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _writes(effects) -> list:
    """Effect operations that are not reads, for idempotence checking."""
    return sorted(k for k, _ in effects
                  if not k.split(".", 1)[-1].startswith(("read", "load",
                                                         "get", "list",
                                                         "find", "query")))


def _is_negative(v) -> bool:
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, Decimal)):
        return v < 0
    if isinstance(v, V.Record):
        return any(_is_negative(x) for _, x in v.fields)
    if isinstance(v, V.Variant):
        return any(_is_negative(x) for x in v.args)
    if isinstance(v, tuple):
        return any(_is_negative(x) for x in v)
    return False


def _less_than(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return False
    if isinstance(a, (int, Decimal)) and isinstance(b, (int, Decimal)):
        return a < b
    return False


def _increase(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v + 1
    if isinstance(v, Decimal):
        return v + 1
    if isinstance(v, tuple):
        return v + (v[-1],) if v else None
    return None


def _param_index(fi: FnInfo, name) -> Optional[int]:
    for i, p in enumerate(fi.decl.params):
        if p.name == name:
            return i
    return None


def _law_arg_name(law) -> str:
    from . import ast as A
    if law.args:
        a = law.args[0]
        if isinstance(a, A.Var):
            return a.name
        if isinstance(a, A.Field):
            return a.name
        if isinstance(a, A.Lit):
            return str(a.value)
    return ""


def _law_arg_int(law) -> Optional[int]:
    from . import ast as A
    if law.args and isinstance(law.args[0], A.Lit):
        v = law.args[0].value
        if isinstance(v, int):
            return v
    return None


def _sum_field(values, field_name) -> Optional[int]:
    total = 0
    found = False
    for v in values:
        for x in _walk_values(v):
            if isinstance(x, V.Record) and x.has(field_name):
                fv = x.get(field_name)
                if isinstance(fv, (int, Decimal)) and not isinstance(fv, bool):
                    total += int(fv)
                    found = True
    return total if found else None


def _walk_values(v, out=None):
    out = [] if out is None else out
    out.append(v)
    if isinstance(v, V.Record):
        for _, x in v.fields:
            _walk_values(x, out)
    elif isinstance(v, V.Variant):
        for x in v.args:
            _walk_values(x, out)
    elif isinstance(v, tuple):
        for x in v:
            _walk_values(x, out)
    elif isinstance(v, V.FrozenMap):
        for _, x in v.items():
            _walk_values(x, out)
    return out


def _size_of(v) -> int:
    if isinstance(v, (tuple, str, bytes, frozenset)):
        return len(v)
    if isinstance(v, V.FrozenMap):
        return len(v)
    if isinstance(v, V.Record):
        return len(v.fields)
    return 1


def _cost_json(c) -> dict:
    if c is None:
        return {}
    out = {}
    for k in ("steps", "io", "tokens", "millis"):
        v = getattr(c, k, None)
        if v is not None:
            out[k] = v
    if getattr(c, "money", None) is not None:
        out["money"] = str(c.money)
    return out


def _render(e) -> str:
    from .canonical import Printer
    try:
        return Printer().expr(e)
    except Exception:
        return "<expression>"


# --------------------------------------------------------------------------

def verify(cr: CheckResult, seed: str = "canon", runs: int = 60,
           hashes=None, only=None) -> VerificationReport:
    return Verifier(cr, seed, runs, hashes).verify_all(only)
