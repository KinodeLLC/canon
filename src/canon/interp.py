"""
The Canon evaluator.

Properties that matter more than speed here:

  Deterministic. Nothing in the evaluator reads the clock, the host random
  source, the filesystem or the network. Everything nondeterministic is an
  effect, and effects go through a Runtime the caller supplies. Give it a
  replaying Runtime and the same program takes the same path it took in
  production, instruction for instruction.

  Bounded. Every evaluation runs under a Budget covering steps, io operations,
  model tokens and wall time. A program that exceeds its declared cost stops
  with a structured fault rather than consuming the machine. This is what makes
  it safe to execute agent-authored code that has not yet been reviewed.

  Contract-enforcing. Preconditions are checked on entry, postconditions on
  exit, record invariants on construction, and `old(...)` is captured before
  the body runs. A contract failure is a structured fault carrying the
  arguments that produced it, which is exactly the shape the verifier needs to
  turn a failure into a shrinkable counterexample.

Note on naming: `Interpreter.eval` walks typed Canon AST nodes by dispatching
on node class. It is unrelated to the host language's `eval` builtin, which is
never called anywhere in this package. No host code is ever constructed from,
or derived from, Canon source.
"""

from __future__ import annotations

import sys
import time as _host_time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Optional

from . import ast as A
from . import prelude as P
from . import values as V
from .checker import CheckResult, Env, FnInfo
from .diagnostics import Span


# --------------------------------------------------------------------------
# Faults
# --------------------------------------------------------------------------

class Fault(Exception):
    """
    A structured runtime failure.

    Faults are not catchable inside Canon. A Canon program handles expected
    failure with `Result`; a Fault means a contract, a budget or a capability
    boundary was breached, which is a fact about the program rather than about
    its input, and must reach the caller intact.
    """

    def __init__(self, code: str, message: str, span: Optional[Span] = None,
                 facts: Optional[dict] = None):
        self.code = code
        self.message = message
        self.span = span
        self.facts = facts or {}
        super().__init__(message)

    def to_json(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "span": self.span.to_json() if self.span else None,
            "facts": {k: (V.to_json(v) if _is_value(v) else v)
                      for k, v in self.facts.items()},
        }

    def __str__(self):
        return f"{self.code}: {self.message}"


class BudgetExceeded(Fault):
    pass


class ContractViolation(Fault):
    pass


class CapabilityDenied(Fault):
    pass


def _is_value(v) -> bool:
    return isinstance(v, (int, str, bool, Decimal, bytes, tuple, frozenset,
                          V.Record, V.Variant, V.FrozenMap, V.Instant,
                          V._Unit))


class _Propagate(Exception):
    """Internal: carries an Err value out to the enclosing function via `?`."""

    def __init__(self, value):
        self.value = value


# --------------------------------------------------------------------------
# Budget
# --------------------------------------------------------------------------

@dataclass
class Budget:
    """
    Resource ceilings for one evaluation.

    `steps` is the only one that is always present. The others default to
    unlimited so that a pure computation is not forced to declare io or token
    budgets it cannot use.
    """
    steps: int = 5_000_000
    io: int = 10_000
    tokens: int = 1_000_000
    millis: int = 60_000
    money: Optional[Decimal] = None

    used_steps: int = 0
    used_io: int = 0
    used_tokens: int = 0
    used_money: Decimal = field(default_factory=lambda: Decimal(0))
    started: float = field(default_factory=_host_time.monotonic)

    def charge_step(self, n: int = 1):
        self.used_steps += n
        if self.used_steps > self.steps:
            raise BudgetExceeded(
                "CANON-E0601",
                f"step budget exceeded ({self.steps} steps)",
                facts={"limit": self.steps, "used": self.used_steps,
                       "resource": "steps"})

    def charge_io(self, n: int = 1):
        self.used_io += n
        if self.used_io > self.io:
            raise BudgetExceeded(
                "CANON-E0602",
                f"io budget exceeded ({self.io} operations)",
                facts={"limit": self.io, "used": self.used_io,
                       "resource": "io"})

    def charge_tokens(self, n: int):
        self.used_tokens += n
        if self.used_tokens > self.tokens:
            raise BudgetExceeded(
                "CANON-E0602",
                f"token budget exceeded ({self.tokens} tokens)",
                facts={"limit": self.tokens, "used": self.used_tokens,
                       "resource": "tokens"})

    def charge_money(self, amount: Decimal):
        self.used_money += amount
        if self.money is not None and self.used_money > self.money:
            raise BudgetExceeded(
                "CANON-E0602",
                f"spend budget exceeded ({self.money})",
                facts={"limit": str(self.money), "used": str(self.used_money),
                       "resource": "money"})

    def check_time(self):
        elapsed = (_host_time.monotonic() - self.started) * 1000
        if elapsed > self.millis:
            raise BudgetExceeded(
                "CANON-E0602",
                f"time budget exceeded ({self.millis} ms)",
                facts={"limit": self.millis, "used": int(elapsed),
                       "resource": "millis"})

    def snapshot(self) -> dict:
        return {"steps": self.used_steps, "io": self.used_io,
                "tokens": self.used_tokens, "money": str(self.used_money),
                "millis": int((_host_time.monotonic() - self.started) * 1000)}

    @staticmethod
    def from_cost(cost: Optional[A.Cost], base: Optional["Budget"] = None
                  ) -> "Budget":
        b = Budget()
        if base is not None:
            b.steps, b.io, b.tokens, b.millis = (base.steps, base.io,
                                                 base.tokens, base.millis)
            b.money = base.money
        if cost is not None:
            if cost.steps is not None:
                b.steps = cost.steps
            if cost.io is not None:
                b.io = cost.io
            if cost.tokens is not None:
                b.tokens = cost.tokens
            if cost.millis is not None:
                b.millis = cost.millis
            if cost.money is not None:
                b.money = cost.money
        return b


# --------------------------------------------------------------------------
# Runtime interface
# --------------------------------------------------------------------------

class Runtime:
    """
    Where every effect goes.

    The evaluator never performs an effect itself. This base class denies
    everything, which is the correct default: a program gets exactly the
    capabilities it was handed and nothing else.
    """

    def perform(self, key: str, args: list, span=None, idempotent: bool = False):
        raise CapabilityDenied(
            "CANON-E0403",
            f"capability {key!r} is not granted",
            span, facts={"operation": key,
                         "granted": sorted(self.granted())})

    def ask(self, req: "AskRequest"):
        raise CapabilityDenied(
            "CANON-E0403",
            "capability 'model.infer' is not granted",
            req.span, facts={"operation": "model.infer", "model": req.model})

    def granted(self):
        return []


@dataclass
class AskRequest:
    """Everything the runtime needs to answer an `ask` expression."""
    model: str
    result_type: Any               # checker Type
    system: str
    inputs: list                   # list[(label, value)]
    grounded_in: list              # list[str]
    examples: list                 # list[(value, value)]
    temperature: Optional[Decimal]
    retries: int
    retry_on: list
    max_tokens: Optional[int]
    judge: Any                     # callable(value) -> bool, or None
    span: Optional[Span] = None
    fn_qualname: str = ""
    def_hash: str = ""
    # Contract obligations the model's output must satisfy. The runtime calls
    # these and retries on failure.
    obligations: list = field(default_factory=list)


# --------------------------------------------------------------------------
# Frames
# --------------------------------------------------------------------------

class Frame:
    __slots__ = ("vars", "parent")

    def __init__(self, parent=None, vars=None):
        self.vars = dict(vars) if vars else {}
        self.parent = parent

    def get(self, name):
        f = self
        while f is not None:
            if name in f.vars:
                return f.vars[name]
            f = f.parent
        raise Fault("CANON-E0201", f"unbound name {name!r} at runtime",
                    facts={"name": name})

    def has(self, name) -> bool:
        f = self
        while f is not None:
            if name in f.vars:
                return True
            f = f.parent
        return False

    def set(self, name, value):
        self.vars[name] = value

    def child(self, vars=None) -> "Frame":
        return Frame(self, vars)


# --------------------------------------------------------------------------
# Call records, for tracing and verification
# --------------------------------------------------------------------------

@dataclass
class CallRecord:
    qualname: str
    def_hash: str
    args: tuple
    result: Any = None
    fault: Optional[Fault] = None
    effects: list = field(default_factory=list)
    cost: dict = field(default_factory=dict)
    depth: int = 0


# --------------------------------------------------------------------------
# Interpreter
# --------------------------------------------------------------------------

# The deepest Canon call chain permitted. This is a language-level limit and
# has to be enforced before the host stack runs out, otherwise a deep call
# surfaces as a host crash instead of a Canon fault.
MAX_DEPTH = 512

# Evaluating one Canon call consumes several host frames (call_fn -> eval ->
# node handler -> nested expression handlers). This multiplier is deliberately
# generous; the cost of a high host recursion limit is nothing, while the cost
# of getting it wrong is an uncatchable crash.
_HOST_FRAMES_PER_CALL = 24
_HOST_HEADROOM = 2000


def _ensure_host_stack(depth_limit: int = MAX_DEPTH):
    need = depth_limit * _HOST_FRAMES_PER_CALL + _HOST_HEADROOM
    if sys.getrecursionlimit() < need:
        sys.setrecursionlimit(need)


class Interpreter:
    def __init__(self, check_result: CheckResult, runtime: Optional[Runtime] = None,
                 budget: Optional[Budget] = None, hashes: Optional[dict] = None,
                 enforce_contracts: bool = True, trace: bool = False):
        self.cr = check_result
        self.env: Env = check_result.env
        self.runtime = runtime or Runtime()
        self.budget = budget or Budget()
        self.hashes = hashes or {}
        self.enforce = enforce_contracts
        self.trace = trace
        self.calls: list = []
        self.depth = 0
        # Stack of (qualname, decreases measure) for termination checking.
        self.measures: list = []
        self.globals = Frame()
        self._cur_fn: Optional[FnInfo] = None
        self._type_cache: dict = {}
        self._tail_cache: dict = {}
        _ensure_host_stack(MAX_DEPTH)
        self._init_globals()

    def _init_globals(self):
        for qn, ci in self.env.consts.items():
            pass  # constants are evaluated lazily on first use

        self._const_cache: dict = {}

    # ------------------------------------------------------- public entry

    def call(self, name: str, args: list, module: str = ""):
        fi = self.env.lookup_fn(name, module)
        if fi is None:
            raise Fault("CANON-E0201", f"no function named {name!r}",
                        facts={"name": name,
                               "known": sorted(self.env.fns)})
        return self.call_fn(fi, list(args))

    def call_fn(self, fi: FnInfo, args: list, span=None):
        d: A.FnDecl = fi.decl
        if d is None:
            raise Fault("CANON-E0201", f"{fi.qualname} has no body")

        if len(args) != len(d.params):
            raise Fault(
                "CANON-E0302",
                f"{fi.qualname} takes {len(d.params)} arguments, "
                f"got {len(args)}", span,
                facts={"function": fi.qualname, "expected": len(d.params),
                       "found": len(args)})

        self.depth += 1
        if self.depth > MAX_DEPTH:
            self.depth -= 1
            raise Fault(
                "CANON-E0604",
                f"call depth exceeded at {fi.qualname}", span or d.span,
                facts={"function": fi.qualname, "limit": MAX_DEPTH})

        frame = self.globals.child(
            {p.name: a for p, a in zip(d.params, args)})

        rec = CallRecord(qualname=fi.qualname,
                         def_hash=self.hashes.get(fi.qualname, ""),
                         args=tuple(args), depth=self.depth)
        if self.trace:
            self.calls.append(rec)

        measure = None
        prev_fn = self._cur_fn
        self._cur_fn = fi
        # The runtime tags every effect with the definition that performed it,
        # so a journal entry can always be attributed to an exact version.
        prev_hash = getattr(self.runtime, "current_def_hash", "")
        if hasattr(self.runtime, "current_def_hash"):
            self.runtime.current_def_hash = rec.def_hash

        try:
            # Preconditions, in the parameter scope.
            if self.enforce:
                for r in d.requires:
                    if self.eval(r, frame) is not True:
                        raise ContractViolation(
                            "CANON-E0507",
                            f"precondition failed in {fi.qualname}",
                            r.span,
                            facts={"function": fi.qualname,
                                   "clause": _render(r),
                                   "arguments": _arg_map(d.params, args)})

            # `old(...)` must be captured before the body runs.
            olds = self._capture_olds(d, frame)

            # Termination measure for recursive functions.
            if d.decreases is not None:
                measure = self.eval(d.decreases, frame)
                for qn, prev in reversed(self.measures):
                    if qn == fi.qualname:
                        if not (isinstance(measure, int)
                                and isinstance(prev, int) and measure < prev):
                            raise Fault(
                                "CANON-E0605",
                                f"the decreases measure of {fi.qualname} did "
                                f"not decrease",
                                d.decreases.span,
                                facts={"function": fi.qualname,
                                       "previous": prev, "current": measure,
                                       "clause": _render(d.decreases)})
                        break
                self.measures.append((fi.qualname, measure))

            try:
                result = self.eval(d.body, frame)
            except _Propagate as prop:
                result = prop.value

            # Postconditions, with `result` and captured `old` values bound.
            if self.enforce and d.ensures:
                post = frame.child({"result": result})
                self._olds = olds
                for e in d.ensures:
                    if self.eval(e, post) is not True:
                        raise ContractViolation(
                            "CANON-E0508",
                            f"postcondition failed in {fi.qualname}",
                            e.span,
                            facts={"function": fi.qualname,
                                   "clause": _render(e),
                                   "arguments": _arg_map(d.params, args),
                                   "result": result})
                self._olds = {}

            rec.result = result
            return result

        except Fault as f:
            rec.fault = f
            raise
        except RecursionError:
            # Backstop: the host stack ran out before MAX_DEPTH was reached.
            # Report it as a Canon fault rather than letting a host error
            # escape, so the caller sees a structured, attributable failure.
            f = Fault(
                "CANON-E0604",
                f"host stack exhausted while evaluating {fi.qualname}",
                span or d.span,
                facts={"function": fi.qualname, "depth": self.depth,
                       "limit": MAX_DEPTH,
                       "host_limit": sys.getrecursionlimit()})
            rec.fault = f
            raise f from None
        finally:
            if measure is not None and self.measures \
                    and self.measures[-1][0] == fi.qualname:
                self.measures.pop()
            self._cur_fn = prev_fn
            if hasattr(self.runtime, "current_def_hash"):
                self.runtime.current_def_hash = prev_hash
            self.depth -= 1

    def _capture_olds(self, d: A.FnDecl, frame) -> dict:
        olds = {}
        if not self.enforce:
            return olds
        for e in d.ensures:
            for node in e.walk():
                if isinstance(node, A.Call) and isinstance(node.fn, A.Var) \
                        and node.fn.name == "old" and node.args:
                    olds[id(node)] = self.eval(node.args[0], frame)
        return olds

    # ------------------------------------------------------- evaluation

    _olds: dict = {}

    def eval(self, e, frame: Frame):
        self.budget.charge_step()
        if self.budget.used_steps % 4096 == 0:
            self.budget.check_time()

        m = getattr(self, "_ev_" + type(e).__name__, None)
        if m is None:
            raise Fault("CANON-E0705",
                        f"cannot evaluate {type(e).__name__}",
                        getattr(e, "span", None))
        return m(e, frame)

    # -- literals and names ---------------------------------------------

    def _ev_Lit(self, e: A.Lit, frame):
        if e.lit_kind == "unit":
            return V.UNIT
        return e.value

    def _ev_Var(self, e: A.Var, frame):
        if frame.has(e.name):
            return frame.get(e.name)

        ci = self.env.lookup_const(e.name, "")
        if ci is not None:
            return self._const_value(ci)

        fi = self.env.lookup_fn(e.name, "")
        if fi is not None:
            return V.Closure(tuple(p.name for p in fi.decl.params),
                             fi.decl.body, self.globals, fi.qualname)

        b = P.lookup(e.name)
        if b is not None:
            return V.Native(b.name, b.arity, b)

        raise Fault("CANON-E0201", f"unbound name {e.name!r}", e.span,
                    facts={"name": e.name})

    def _ev_QualVar(self, e: A.QualVar, frame):
        full = e.full()
        b = P.lookup(full)
        if b is not None:
            return V.Native(b.name, b.arity, b)
        fi = self.env.fns.get(full) or self.env.lookup_fn(e.name, e.module)
        if fi is not None:
            return V.Closure(tuple(p.name for p in fi.decl.params),
                             fi.decl.body, self.globals, fi.qualname)
        ci = self.env.consts.get(full)
        if ci is not None:
            return self._const_value(ci)
        raise Fault("CANON-E0201", f"unbound name {full!r}", e.span,
                    facts={"name": full})

    def _const_value(self, ci):
        if ci.qualname not in self._const_cache:
            self._const_cache[ci.qualname] = self.eval(ci.decl.value,
                                                       self.globals)
        return self._const_cache[ci.qualname]

    # -- structure -------------------------------------------------------

    def _ev_Field(self, e: A.Field, frame):
        if isinstance(e.target, A.Var) and not frame.has(e.target.name) \
                and e.target.name in self.env.modules:
            return self._ev_QualVar(
                A.QualVar(module=e.target.name, name=e.name, span=e.span), frame)
        target = self.eval(e.target, frame)
        if isinstance(target, V.Record):
            if target.has(e.name):
                return target.get(e.name)
            raise Fault("CANON-E0206",
                        f"{target.type_name} has no field {e.name!r}", e.span,
                        facts={"record": target.type_name, "field": e.name})
        raise Fault("CANON-E0304",
                    f"field access on a {V.type_name_of(target)}", e.span,
                    facts={"type": V.type_name_of(target), "field": e.name})

    def _ev_RecordLit(self, e: A.RecordLit, frame):
        ti = self.env.types.get(e.type_name)
        order = [f.name for f in ti.decl.fields] if ti else \
            [n for n, _ in e.fields]

        base_fields = {}
        if e.base is not None:
            base = self.eval(e.base, frame)
            if isinstance(base, V.Record):
                base_fields = base.as_dict()

        given = {n: self.eval(v, frame) for n, v in e.fields}
        merged = dict(base_fields)
        merged.update(given)

        fields = tuple((n, merged[n]) for n in order if n in merged)
        rec = V.Record(e.type_name, fields)
        self._check_invariants(rec, ti, e.span)
        return rec

    def _check_invariants(self, rec: V.Record, ti, span):
        """Record invariants are checked wherever a record value is built."""
        if not self.enforce or ti is None or not getattr(ti.decl, "invariants", None):
            return
        inner = self.globals.child(rec.as_dict())
        inner.set("self", rec)
        for inv in ti.decl.invariants:
            if self.eval(inv, inner) is not True:
                raise ContractViolation(
                    "CANON-E0508",
                    f"invariant of {rec.type_name} violated", span or inv.span,
                    facts={"record": rec.type_name,
                           "clause": _render(inv),
                           "value": rec})

    def _ev_CtorCall(self, e: A.CtorCall, frame):
        ci = self.env.ctors.get(e.name)
        args = tuple(self.eval(a, frame) for a in e.args)
        return V.Variant(e.name, args, ci.type_name if ci else "")

    def _ev_ListLit(self, e: A.ListLit, frame):
        return tuple(self.eval(i, frame) for i in e.items)

    def _ev_MapLit(self, e: A.MapLit, frame):
        return V.FrozenMap({P._mapkey(self.eval(k, frame)): self.eval(v, frame)
                            for k, v in e.entries})

    # -- application -----------------------------------------------------

    def _ev_Call(self, e: A.Call, frame):
        if isinstance(e.fn, A.Var):
            if e.fn.name == "abort":
                msg = self.eval(e.args[0], frame) if e.args else "aborted"
                raise Fault("CANON-E0706", str(msg), e.span,
                            facts={"message": str(msg)})
            if e.fn.name == "old":
                if id(e) in self._olds:
                    return self._olds[id(e)]
                return self.eval(e.args[0], frame) if e.args else V.UNIT

            # Direct call to a known function avoids building a closure.
            if not frame.has(e.fn.name):
                fi = self.env.lookup_fn(e.fn.name, "")
                if fi is not None:
                    args = [self.eval(a, frame) for a in e.args]
                    return self.call_fn(fi, args, e.span)

        if isinstance(e.fn, A.QualVar):
            b = P.lookup(e.fn.full())
            if b is not None:
                args = [self.eval(a, frame) for a in e.args]
                return self._call_builtin(b, args, e.span)
            fi = self.env.fns.get(e.fn.full()) or \
                self.env.lookup_fn(e.fn.name, e.fn.module)
            if fi is not None:
                args = [self.eval(a, frame) for a in e.args]
                return self.call_fn(fi, args, e.span)

        callee = self.eval(e.fn, frame)
        args = [self.eval(a, frame) for a in e.args]
        return self.apply(callee, args, e.span)

    def apply(self, callee, args: list, span=None):
        if isinstance(callee, V.Native):
            b = callee.impl if isinstance(callee.impl, P.Builtin) \
                else P.lookup(callee.name)
            return self._call_builtin(b, args, span)
        if isinstance(callee, V.Closure):
            fi = self.env.fns.get(callee.name)
            if fi is not None:
                return self.call_fn(fi, args, span)
            inner = callee.env.child(dict(zip(callee.params, args)))
            return self.eval(callee.body, inner)
        raise Fault("CANON-E0303",
                    f"{V.type_name_of(callee)} is not callable", span,
                    facts={"type": V.type_name_of(callee)})

    def _call_builtin(self, b: P.Builtin, args: list, span=None):
        if b is None:
            raise Fault("CANON-E0201", "unknown builtin", span)
        try:
            if b.needs_apply:
                return b.impl(lambda f, a: self.apply(f, a, span), *args)
            return b.impl(*args)
        except P.Abort as ab:
            raise Fault("CANON-E0706", ab.message, span,
                        facts={"message": ab.message})
        except Fault:
            raise
        except TypeError as te:
            raise Fault(
                "CANON-E0302",
                f"{b.name} was called with {len(args)} arguments", span,
                facts={"builtin": b.name, "expected": b.arity,
                       "found": len(args), "detail": str(te)})
        except (ZeroDivisionError, ArithmeticError) as ae:
            raise Fault("CANON-E0701", f"{b.name}: {ae}", span,
                        facts={"builtin": b.name})

    def _ev_Lambda(self, e: A.Lambda, frame):
        return V.Closure(tuple(p.name for p in e.params), e.body, frame)

    def _ev_Perform(self, e: A.Perform, frame):
        # A local of function type shadows an effect name.
        if frame.has(e.effect):
            target = frame.get(e.effect)
            if isinstance(target, V.Record):
                fn = target.get(e.op)
                args = [self.eval(a, frame) for a in e.args]
                return self.apply(fn, args, e.span)

        ei = self.env.effects.get(e.effect)
        if ei is None:
            fi = self.env.fns.get(f"{e.effect}.{e.op}")
            if fi is not None:
                args = [self.eval(a, frame) for a in e.args]
                return self.call_fn(fi, args, e.span)
            b = P.lookup(f"{e.effect}.{e.op}")
            if b is not None:
                args = [self.eval(a, frame) for a in e.args]
                return self._call_builtin(b, args, e.span)
            raise Fault("CANON-E0204", f"unknown effect {e.effect!r}", e.span,
                        facts={"effect": e.effect})

        op = ei.ops.get(e.op)
        args = [self.eval(a, frame) for a in e.args]
        key = f"{e.effect}.{e.op}"
        self.budget.charge_io()
        return self.runtime.perform(key, args, e.span,
                                    idempotent=bool(op and op.idempotent))

    # -- control flow ----------------------------------------------------

    def _ev_Let(self, e: A.Let, frame):
        value = self.eval(e.value, frame)
        return self.eval(e.body, frame.child({e.name: value}))

    def _ev_If(self, e: A.If, frame):
        c = self.eval(e.cond, frame)
        if c is True:
            return self.eval(e.then, frame)
        if c is False:
            return self.eval(e.otherwise, frame)
        raise Fault("CANON-E0301",
                    f"`if` condition evaluated to {V.type_name_of(c)}",
                    e.cond.span, facts={"type": V.type_name_of(c)})

    def _ev_Block(self, e: A.Block, frame):
        inner = frame.child()
        for s in e.stmts:
            if isinstance(s, A.SLet):
                inner.set(s.name, self.eval(s.value, inner))
            elif isinstance(s, A.SExpr):
                self.eval(s.value, inner)
            elif isinstance(s, A.SAssert):
                if self.enforce and self.eval(s.cond, inner) is not True:
                    raise ContractViolation(
                        "CANON-E0508",
                        s.message or "assertion failed", s.span,
                        facts={"clause": _render(s.cond),
                               "message": s.message})
        return self.eval(e.result, inner)

    def _ev_Match(self, e: A.Match, frame):
        subject = self.eval(e.scrutinee, frame)
        for arm in e.arms:
            binds = {}
            if self.match_pattern(arm.pattern, subject, binds):
                inner = frame.child(binds)
                if arm.guard is not None:
                    if self.eval(arm.guard, inner) is not True:
                        continue
                return self.eval(arm.body, inner)
        raise Fault(
            "CANON-E0705",
            "no match arm applied", e.span,
            facts={"value": subject, "type": V.type_name_of(subject)},
            )

    def _ev_Try(self, e: A.Try, frame):
        v = self.eval(e.operand, frame)
        if isinstance(v, V.Variant) and v.ctor == "Ok":
            return v.args[0] if v.args else V.UNIT
        if isinstance(v, V.Variant) and v.ctor == "Err":
            raise _Propagate(v)
        raise Fault("CANON-E0301",
                    f"`?` applied to a {V.type_name_of(v)}", e.span,
                    facts={"type": V.type_name_of(v)})

    def _ev_Binary(self, e: A.Binary, frame):
        op = e.op
        # Short-circuit before evaluating the right side.
        if op == "and":
            return False if self.eval(e.left, frame) is not True \
                else self.eval(e.right, frame) is True
        if op == "or":
            return True if self.eval(e.left, frame) is True \
                else self.eval(e.right, frame) is True
        if op == "|>":
            if isinstance(e.right, A.Call):
                call = A.Call(fn=e.right.fn,
                              args=list(e.right.args) + [e.left], span=e.span)
            else:
                call = A.Call(fn=e.right, args=[e.left], span=e.span)
            return self._ev_Call(call, frame)

        a = self.eval(e.left, frame)
        b = self.eval(e.right, frame)

        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op == "*":
            self.budget.charge_step(_mul_cost(a, b))
            return a * b
        if op == "++":
            return a + b
        if op == "==":
            return V.compare(a, b) == 0
        if op == "!=":
            return V.compare(a, b) != 0
        if op == "<":
            return V.compare(a, b) < 0
        if op == ">":
            return V.compare(a, b) > 0
        if op == "<=":
            return V.compare(a, b) <= 0
        if op == ">=":
            return V.compare(a, b) >= 0
        raise Fault("CANON-E0301", f"unknown operator {op!r}", e.span,
                    facts={"operator": op})

    def _ev_Unary(self, e: A.Unary, frame):
        v = self.eval(e.operand, frame)
        if e.op == "not":
            return v is not True
        if e.op == "-":
            return -v
        raise Fault("CANON-E0301", f"unknown operator {e.op!r}", e.span)

    # -- patterns --------------------------------------------------------

    def match_pattern(self, p, value, binds: dict) -> bool:
        if isinstance(p, A.PWild):
            return True
        if isinstance(p, A.PVar):
            binds[p.name] = value
            return True
        if isinstance(p, A.PLit):
            if p.lit_kind == "unit":
                return isinstance(value, V._Unit)
            return V.compare(value, p.value) == 0
        if isinstance(p, A.PCtor):
            if not isinstance(value, V.Variant) or value.ctor != p.name:
                return False
            if len(p.args) != len(value.args):
                return False
            return all(self.match_pattern(sub, v, binds)
                       for sub, v in zip(p.args, value.args))
        if isinstance(p, A.PRecord):
            if not isinstance(value, V.Record) or value.type_name != p.type_name:
                return False
            for fname, sub in p.fields:
                if not value.has(fname):
                    return False
                if not self.match_pattern(sub, value.get(fname), binds):
                    return False
            return True
        if isinstance(p, A.PList):
            if not isinstance(value, tuple):
                return False
            if p.rest is None:
                if len(value) != len(p.items):
                    return False
            elif len(value) < len(p.items):
                return False
            for sub, v in zip(p.items, value):
                if not self.match_pattern(sub, v, binds):
                    return False
            if p.rest:
                binds[p.rest] = value[len(p.items):]
            return True
        return False

    # -- the model primitive ---------------------------------------------

    def _ev_Ask(self, e: A.Ask, frame):
        s = e.spec or A.AskSpec()
        system = self.eval(s.system, frame) if s.system is not None else ""
        inputs = [(label, self.eval(v, frame)) for label, v in s.inputs]
        grounded = [self.eval(g, frame) for g in s.grounded_in]
        examples = [(self.eval(a, frame), self.eval(b, frame))
                    for a, b in s.examples]
        judge = self.eval(s.judge, frame) if s.judge is not None else None
        judge_fn = None
        if judge is not None:
            judge_fn = lambda v: self.apply(judge, [v], e.span) is True

        req = AskRequest(
            model=e.model,
            result_type=self.resolve_type(e.result_type),
            system=system if isinstance(system, str) else str(system),
            inputs=inputs,
            grounded_in=[g for g in grounded if isinstance(g, str)],
            examples=examples,
            temperature=s.temperature,
            retries=s.retries,
            retry_on=list(s.retry_on),
            max_tokens=s.max_tokens,
            judge=judge_fn,
            span=e.span,
            fn_qualname=self._cur_fn.qualname if self._cur_fn else "",
            obligations=self._obligations_for(e, frame),
        )
        self.budget.charge_io()
        return self.runtime.ask(req)

    def resolve_type(self, texpr):
        """Resolve an AST type expression to a checker type, with caching."""
        key = id(texpr)
        if key not in self._type_cache:
            from .checker import Checker
            c = Checker()
            c.env = self.env
            c.cur_module = self._cur_fn.module if self._cur_fn else ""
            c.tparams = set(self._cur_fn.tparams) if self._cur_fn else set()
            self._type_cache[key] = c.resolve_type(texpr, None)
        return self._type_cache[key]

    def _obligations_for(self, e: A.Ask, frame) -> list:
        """
        The contract obligations a model's answer must satisfy.

        Only the enclosing function's `ensures` clauses are attached, and only
        when this `ask` is in tail position -- that is, when its value *is* the
        function's result. Attaching them anywhere else would check a clause
        against a value it was never written about, and reject correct answers.
        """
        fi = self._cur_fn
        if fi is None or fi.decl is None or not fi.decl.ensures:
            return []
        if id(e) not in self._tail_asks(fi.decl):
            return []

        obligations = []
        for clause in fi.decl.ensures:
            def check(value, _clause=clause, _frame=frame):
                post = _frame.child({"result": value})
                try:
                    if self.eval(_clause, post) is True:
                        return None
                except Fault as f:
                    return f"{_render(_clause)} could not be evaluated: {f.message}"
                return f"the result must satisfy: {_render(_clause)}"
            obligations.append(check)
        return obligations

    def _tail_asks(self, decl: A.FnDecl) -> set:
        """Ask nodes whose value is the function's return value."""
        key = id(decl)
        if key in self._tail_cache:
            return self._tail_cache[key]

        found = set()

        def walk(node):
            if node is None:
                return
            if isinstance(node, A.Ask):
                found.add(id(node))
            elif isinstance(node, A.Block):
                walk(node.result)
            elif isinstance(node, A.If):
                walk(node.then)
                walk(node.otherwise)
            elif isinstance(node, A.Match):
                for arm in node.arms:
                    walk(arm.body)
            elif isinstance(node, A.Let):
                walk(node.body)

        walk(decl.body)
        self._tail_cache[key] = found
        return found


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _mul_cost(a, b) -> int:
    """Large integer multiplication is charged by operand size."""
    try:
        if isinstance(a, int) and isinstance(b, int):
            bits = max(a.bit_length(), b.bit_length())
            return max(0, bits // 64)
    except Exception:
        pass
    return 0


def _render(e) -> str:
    from .canonical import Printer
    try:
        return Printer().expr(e)
    except Exception:
        return "<expression>"


def _arg_map(params, args) -> dict:
    return {p.name: a for p, a in zip(params, args)}


# --------------------------------------------------------------------------

def run(check_result: CheckResult, fn_name: str, args: list,
        runtime: Optional[Runtime] = None, budget: Optional[Budget] = None,
        module: str = ""):
    """Evaluate one call. Returns (value, interpreter)."""
    it = Interpreter(check_result, runtime, budget)
    return it.call(fn_name, args, module), it


def run_tests(check_result: CheckResult, runtime: Optional[Runtime] = None,
              budget: Optional[Budget] = None) -> list:
    """
    Execute every `test` declaration. Returns a list of result dicts.

    Tests are ordinary Canon expressions that must evaluate to `true`, so they
    run under the same budgets and contract enforcement as production code.
    """
    out = []
    for m in check_result.modules:
        for t in m.tests():
            it = Interpreter(check_result, runtime,
                             budget or Budget())
            entry = {"module": m.name, "name": t.name}
            try:
                value = it.eval(t.body, it.globals)
                entry["passed"] = value is True
                if value is not True:
                    entry["value"] = V.show(value)
            except Fault as f:
                entry["passed"] = False
                entry["fault"] = f.to_json()
            entry["cost"] = it.budget.snapshot()
            out.append(entry)
    return out
