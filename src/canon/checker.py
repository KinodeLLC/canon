"""
The Canon checker.

One pass over a set of modules produces, for every definition:

  * a fully resolved type
  * the exact set of effect operations its body performs
  * the transitive capability footprint of its whole call graph
  * exhaustiveness results for every match
  * type-checked contract clauses
  * a call graph

The transitive capability footprint is the part that matters operationally. It
answers "what can this change possibly touch" statically, without running
anything, which is what a promotion gate needs in order to approve an
agent-authored change without a human reading the body.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional

from . import ast as A
from . import prelude as P
from . import types as TY
from .diagnostics import Bag, Repair, Span


# --------------------------------------------------------------------------
# Environment entries
# --------------------------------------------------------------------------

@dataclass
class TypeInfo:
    name: str
    kind: str                 # record | enum | alias
    tparams: list
    decl: object
    module: str = ""

    @property
    def arity(self) -> int:
        return len(self.tparams)


@dataclass
class CtorInfo:
    name: str
    type_name: str
    tparams: list
    payload: list             # list[A.TypeExpr]
    module: str = ""


@dataclass
class FnInfo:
    qualname: str
    name: str
    module: str
    decl: A.FnDecl
    type: TY.TFun
    tparams: list
    declared: TY.EffectSet
    # Filled in during body checking.
    performed: set = field(default_factory=set)
    calls: set = field(default_factory=set)
    transitive: set = field(default_factory=set)
    models: set = field(default_factory=set)
    touches: set = field(default_factory=set)   # data classifications reached


@dataclass
class EffectInfo:
    name: str
    decl: A.EffectDecl
    ops: dict                 # op name -> A.EffectOp
    module: str = ""
    builtin: bool = False


@dataclass
class ConstInfo:
    qualname: str
    name: str
    module: str
    decl: A.ConstDecl
    type: TY.Type


class Env:
    """The global environment: everything visible across all checked modules."""

    def __init__(self):
        self.types: dict = {}
        self.ctors: dict = {}
        self.fns: dict = {}
        self.effects: dict = {}
        self.consts: dict = {}
        self.modules: dict = {}
        self._install_prelude()

    def _install_prelude(self):
        for name, decl in P.BUILTIN_TYPES.items():
            self.types[name] = TypeInfo(name, "record", decl.type_params,
                                        decl, "prelude")
        for name, decl in P.BUILTIN_EFFECTS.items():
            self.effects[name] = EffectInfo(
                name, decl, {o.name: o for o in decl.ops}, "prelude", True)
        # Option and Result constructors behave as ordinary enum constructors.
        for cname, (tname, tps, idxs) in TY.PRELUDE_CTORS.items():
            payload = [A.TName(name=tps[i]) for i in idxs]
            self.ctors[cname] = CtorInfo(cname, tname, tps, payload, "prelude")

    def lookup_fn(self, name: str, module: str = "") -> Optional[FnInfo]:
        if module and f"{module}.{name}" in self.fns:
            return self.fns[f"{module}.{name}"]
        if name in self.fns:
            return self.fns[name]
        for qn, fi in self.fns.items():
            if qn.endswith("." + name):
                return fi
        return None

    def lookup_const(self, name: str, module: str = "") -> Optional[ConstInfo]:
        if module and f"{module}.{name}" in self.consts:
            return self.consts[f"{module}.{name}"]
        if name in self.consts:
            return self.consts[name]
        for qn, ci in self.consts.items():
            if qn.endswith("." + name):
                return ci
        return None


@dataclass
class MatchInfo:
    """Recorded per match so the verifier knows which arms exist."""
    span: Span
    scrutinee_type: str
    covered: list
    exhaustive: bool


@dataclass
class CheckResult:
    env: Env
    bag: Bag
    modules: list
    matches: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.bag.has_errors

    def fn(self, name: str) -> Optional[FnInfo]:
        return self.env.lookup_fn(name)


# --------------------------------------------------------------------------
# Scope
# --------------------------------------------------------------------------

class Scope:
    __slots__ = ("vars", "parent", "used")

    def __init__(self, parent=None):
        self.vars: dict = {}
        self.parent = parent
        self.used: set = set()

    def declare(self, name, ty, span=None):
        self.vars[name] = (ty, span)

    def find(self, name):
        s = self
        while s is not None:
            if name in s.vars:
                s.used.add(name)
                return s.vars[name][0]
            s = s.parent
        return None

    def declared_here(self):
        return self.vars


# --------------------------------------------------------------------------
# Checker
# --------------------------------------------------------------------------

ARITH_OPS = {"+", "-", "*"}
DIV_OPS = {"/", "%"}
CMP_OPS = {"<", ">", "<=", ">="}
EQ_OPS = {"==", "!="}
BOOL_OPS = {"and", "or"}


class Checker:
    def __init__(self, bag: Optional[Bag] = None):
        self.bag = bag if bag is not None else Bag()
        self.env = Env()
        self.matches: list = []
        # per-function state
        self.cur: Optional[FnInfo] = None
        self.cur_module: str = ""
        self.tparams: set = set()
        self.in_ensures: bool = False
        self.result_type: Optional[TY.Type] = None

    # ================================================== entry point

    def check(self, modules) -> CheckResult:
        modules = list(modules)
        for m in modules:
            self.env.modules[m.name] = m

        self._collect_types(modules)
        self._collect_effects(modules)
        self._collect_ctors(modules)
        self._collect_signatures(modules)
        self._collect_consts(modules)

        for m in modules:
            self.cur_module = m.name
            for d in m.decls:
                if isinstance(d, A.FnDecl):
                    self._check_fn(d, m)
                elif isinstance(d, A.RecordDecl):
                    self._check_record(d, m)
                elif isinstance(d, A.ConstDecl):
                    self._check_const(d, m)
                elif isinstance(d, A.TestDecl):
                    self._check_test(d, m)

        self._compute_transitive()
        return CheckResult(self.env, self.bag, modules, self.matches)

    # ================================================== collection passes

    def _collect_types(self, modules):
        for m in modules:
            for d in m.decls:
                if isinstance(d, (A.RecordDecl, A.EnumDecl, A.AliasDecl)):
                    kind = ("record" if isinstance(d, A.RecordDecl)
                            else "enum" if isinstance(d, A.EnumDecl) else "alias")
                    if d.name in self.env.types and \
                            self.env.types[d.name].module not in ("prelude", m.name):
                        self.bag.error(
                            "CANON-E0203", f"type {d.name!r} is already defined",
                            d.span,
                            facts={"name": d.name,
                                   "existing": self.env.types[d.name].module})
                        continue
                    if d.name in TY.PRIMITIVES or d.name in TY.GENERICS:
                        self.bag.error(
                            "CANON-E0203",
                            f"{d.name!r} is a builtin type and cannot be redefined",
                            d.span, facts={"name": d.name})
                        continue
                    self.env.types[d.name] = TypeInfo(
                        d.name, kind, d.type_params, d, m.name)

    def _collect_effects(self, modules):
        for m in modules:
            for d in m.effects():
                if d.name in self.env.effects and \
                        self.env.effects[d.name].module != m.name:
                    self.bag.error(
                        "CANON-E0203", f"effect {d.name!r} is already defined",
                        d.span, facts={"name": d.name})
                    continue
                seen = {}
                for o in d.ops:
                    if o.name in seen:
                        self.bag.error(
                            "CANON-E0203",
                            f"effect operation {d.name}.{o.name!r} is declared twice",
                            o.span, facts={"effect": d.name, "op": o.name})
                    seen[o.name] = o
                self.env.effects[d.name] = EffectInfo(d.name, d, seen, m.name)

    def _collect_ctors(self, modules):
        for m in modules:
            for d in m.decls:
                if not isinstance(d, A.EnumDecl):
                    continue
                for v in d.variants:
                    if v.name in self.env.ctors:
                        prev = self.env.ctors[v.name]
                        if prev.type_name != d.name:
                            self.bag.error(
                                "CANON-E0203",
                                f"constructor {v.name!r} is already used by "
                                f"{prev.type_name}",
                                v.span,
                                facts={"ctor": v.name,
                                       "existing_type": prev.type_name,
                                       "new_type": d.name},
                                notes=["Constructor names are global so that "
                                       "patterns never need a type annotation."])
                            continue
                    self.env.ctors[v.name] = CtorInfo(
                        v.name, d.name, d.type_params, list(v.params), m.name)

    def _collect_signatures(self, modules):
        for m in modules:
            for d in m.functions():
                qn = f"{m.name}.{d.name}" if m.name else d.name
                if qn in self.env.fns:
                    self.bag.error(
                        "CANON-E0203", f"function {qn!r} is already defined",
                        d.span, facts={"name": qn})
                    continue
                self.tparams = set(d.type_params)
                self.cur_module = m.name
                params = [self.resolve_type(p.ty, p.span) for p in d.params]
                result = self.resolve_type(d.result, d.span)
                declared = TY.EffectSet.from_refs(d.uses)
                self._validate_effect_refs(d.uses)
                ftype = TY.TFun(params, result, frozenset(declared.keys()))
                self.env.fns[qn] = FnInfo(
                    qualname=qn, name=d.name, module=m.name, decl=d,
                    type=ftype, tparams=list(d.type_params), declared=declared)
                self.tparams = set()

    def _collect_consts(self, modules):
        for m in modules:
            for d in m.decls:
                if not isinstance(d, A.ConstDecl):
                    continue
                qn = f"{m.name}.{d.name}" if m.name else d.name
                self.cur_module = m.name
                self.env.consts[qn] = ConstInfo(
                    qn, d.name, m.name, d, self.resolve_type(d.ty, d.span))

    def _validate_effect_refs(self, refs):
        for r in refs:
            ei = self.env.effects.get(r.effect)
            if ei is None:
                self.bag.error(
                    "CANON-E0204", f"unknown effect {r.effect!r}", r.span,
                    facts={"effect": r.effect,
                           "known": sorted(self.env.effects)},
                    repairs=self._near_repairs(r.effect, self.env.effects, r.span))
                continue
            if r.op is not None and r.op not in ei.ops:
                self.bag.error(
                    "CANON-E0205",
                    f"effect {r.effect!r} has no operation {r.op!r}", r.span,
                    facts={"effect": r.effect, "op": r.op,
                           "known": sorted(ei.ops)},
                    repairs=self._near_repairs(r.op, ei.ops, r.span))
            if r.op is None:
                self.bag.warn(
                    "CANON-W0006",
                    f"`{r.effect}.*` grants every operation of {r.effect!r}",
                    r.span,
                    facts={"effect": r.effect, "operations": sorted(ei.ops)},
                    repairs=[Repair(
                        "replace-span",
                        "name only the operations actually used",
                        ", ".join(f"{r.effect}.{o}" for o in sorted(ei.ops)),
                        r.span, 0.4)],
                    notes=["A wildcard makes the capability footprint an "
                           "over-approximation, which widens the blast radius "
                           "reported at promotion time."])

    # ================================================== type resolution

    def resolve_type(self, texpr, span=None) -> TY.Type:
        if texpr is None:
            return TY.UNIT

        if isinstance(texpr, A.TName):
            name = texpr.name
            if name in self.tparams and not texpr.args:
                return TY.TRigid(name)
            if name in TY.PRIMITIVES and not texpr.args:
                return TY.PRIMITIVES[name]
            if name in TY.GENERICS:
                arity = TY.GENERICS[name]
                args = [self.resolve_type(a, span) for a in texpr.args]
                if len(args) != arity:
                    self._arity_error(name, arity, len(args), texpr.span)
                    args = (args + [TY.fresh() for _ in range(arity)])[:arity]
                return TY.TCon(name, args)
            ti = self.env.types.get(name)
            if ti is None:
                self.bag.error(
                    "CANON-E0202", f"unknown type {name!r}", texpr.span,
                    facts={"name": name, "known": self._known_type_names()},
                    repairs=self._near_repairs(name, self.env.types, texpr.span))
                return TY.NEVER
            args = [self.resolve_type(a, span) for a in texpr.args]
            if len(args) != ti.arity:
                self._arity_error(name, ti.arity, len(args), texpr.span)
                args = (args + [TY.fresh() for _ in range(ti.arity)])[:ti.arity]
            if ti.kind == "alias":
                saved = self.tparams
                self.tparams = set(ti.tparams)
                target = self.resolve_type(ti.decl.target, span)
                self.tparams = saved
                return TY.subst(target, dict(zip(ti.tparams, args)))
            return TY.TCon(name, args)

        if isinstance(texpr, A.TVar):
            return TY.TRigid(texpr.name)

        if isinstance(texpr, A.TFn):
            self._validate_effect_refs(texpr.effects)
            return TY.TFun([self.resolve_type(p, span) for p in texpr.params],
                           self.resolve_type(texpr.result, span),
                           frozenset(e.key() for e in texpr.effects))

        if isinstance(texpr, A.TRecord):
            return TY.TRec({n: self.resolve_type(t, span)
                            for n, t in texpr.fields})

        return TY.NEVER

    def _arity_error(self, name, expected, got, span):
        self.bag.error(
            "CANON-E0312",
            f"type {name!r} takes {expected} type argument"
            f"{'' if expected == 1 else 's'}, found {got}",
            span, facts={"type": name, "expected": expected, "found": got})

    def _known_type_names(self):
        return sorted(set(TY.PRIMITIVES) | set(TY.GENERICS) | set(self.env.types))

    # ================================================== declaration checks

    def _check_fn(self, d: A.FnDecl, m: A.Module):
        qn = f"{m.name}.{d.name}" if m.name else d.name
        fi = self.env.fns.get(qn)
        if fi is None:
            return
        self.cur = fi
        self.cur_module = m.name
        self.tparams = set(d.type_params)
        self.result_type = fi.type.result

        scope = Scope()
        for p, pt in zip(d.params, fi.type.params):
            scope.declare(p.name, pt, p.span)

        # Preconditions are checked in the parameter scope only.
        for r in d.requires:
            t = self.infer(r, scope)
            self._expect(t, TY.BOOL, r.span, "a `requires` clause must be Bool")

        # Postconditions additionally see `result`.
        post = Scope(scope)
        post.declare("result", fi.type.result, d.span)
        self.in_ensures = True
        for e in d.ensures:
            t = self.infer(e, post)
            self._expect(t, TY.BOOL, e.span, "an `ensures` clause must be Bool")
        self.in_ensures = False

        for law in d.laws:
            self._check_law(law, scope, fi)

        if d.decreases is not None:
            t = self.infer(d.decreases, scope)
            self._expect(t, TY.INT, d.decreases.span,
                         "a `decreases` clause must be Int")

        body_t = self.infer(d.body, scope)
        self._expect(body_t, fi.type.result, self._body_span(d),
                     f"the body of {d.name!r} must produce its declared "
                     f"result type")

        self._check_effects(d, fi)
        self._check_recursion(d, fi)
        self._warn_unused(scope, d)
        self._warn_missing_contracts(d, fi)

        self.cur = None
        self.tparams = set()
        self.result_type = None

    def _body_span(self, d):
        return d.body.span if d.body is not None else d.span

    def _check_law(self, law: A.LawRef, scope, fi):
        spec = LAWS.get(law.name)
        if spec is None:
            self.bag.error(
                "CANON-E0505", f"unknown law {law.name!r}", law.span,
                facts={"law": law.name, "known": sorted(LAWS)},
                repairs=self._near_repairs(law.name, LAWS, law.span))
            return
        arity, _desc = spec
        if len(law.args) != arity:
            self.bag.error(
                "CANON-E0506",
                f"law {law.name!r} takes {arity} argument"
                f"{'' if arity == 1 else 's'}, found {len(law.args)}",
                law.span, facts={"law": law.name, "expected": arity,
                                 "found": len(law.args)})
            return
        for a in law.args:
            self.infer(a, scope)

    def _check_effects(self, d: A.FnDecl, fi: FnInfo):
        missing = fi.declared.missing(fi.performed)
        for key in missing:
            eff, _, op = key.partition(".")
            self.bag.error(
                "CANON-E0401",
                f"effect {key!r} is performed but not declared", d.span,
                facts={"operation": key,
                       "declared": fi.declared.keys(),
                       "performed": sorted(fi.performed)},
                repairs=[Repair(
                    "add-clause",
                    f"declare the effect on {d.name!r}",
                    f"uses {key}", d.span, 0.95)],
                notes=["A function's declared effects are its capability "
                       "footprint. The checker will not infer them, because an "
                       "inferred footprint silently widens when a body "
                       "changes."])
        for key in fi.declared.unused(fi.performed):
            self.bag.warn(
                "CANON-W0002", f"declared effect {key!r} is never used", d.span,
                facts={"operation": key},
                repairs=[Repair("manual",
                                f"remove `{key}` from the uses clause")],
                notes=["Unused declarations widen the capability footprint "
                       "without cause."])

    def _check_recursion(self, d: A.FnDecl, fi: FnInfo):
        qn = fi.qualname
        if qn not in fi.calls:
            if d.decreases is not None:
                self.bag.warn(
                    "CANON-W0002",
                    f"{d.name!r} declares `decreases` but is not recursive",
                    d.span, facts={"function": d.name})
            return
        if d.decreases is None:
            self.bag.error(
                "CANON-E0311",
                f"{d.name!r} is recursive and must declare `decreases`",
                d.span,
                facts={"function": d.name},
                repairs=[Repair(
                    "add-clause",
                    "state the quantity that strictly decreases each call",
                    "decreases <expression>", d.span, 0.5)],
                notes=["Every Canon function is total. Recursion is allowed "
                       "only with a measure the runtime can check, so an agent "
                       "cannot write a program that fails to terminate."])

    def _check_record(self, d: A.RecordDecl, m: A.Module):
        self.tparams = set(d.type_params)
        self.cur_module = m.name
        seen = set()
        for f in d.fields:
            if f.name in seen:
                self.bag.error(
                    "CANON-E0203",
                    f"field {f.name!r} is declared twice in {d.name}",
                    f.span, facts={"record": d.name, "field": f.name})
            seen.add(f.name)
            self.resolve_type(f.ty, f.span)
        for fname in d.classification:
            if fname not in seen:
                self.bag.error(
                    "CANON-E0206",
                    f"{d.name} has no field {fname!r} to classify", d.span,
                    facts={"record": d.name, "field": fname,
                           "known": sorted(seen)})
        for fname, cls in d.classification.items():
            if cls not in TY.CLASS_RANK:
                self.bag.error(
                    "CANON-E0201", f"unknown data classification {cls!r}",
                    d.span, facts={"classification": cls,
                                   "known": TY.CLASSIFICATIONS})
        if d.invariants:
            scope = Scope()
            self_t = TY.TCon(d.name, [TY.TRigid(t) for t in d.type_params])
            scope.declare("self", self_t, d.span)
            for f in d.fields:
                scope.declare(f.name, self.resolve_type(f.ty, f.span), f.span)
            for inv in d.invariants:
                t = self.infer(inv, scope)
                self._expect(t, TY.BOOL, inv.span,
                             "a record invariant must be Bool")
        self.tparams = set()

    def _check_const(self, d: A.ConstDecl, m: A.Module):
        qn = f"{m.name}.{d.name}" if m.name else d.name
        ci = self.env.consts.get(qn)
        if ci is None:
            return
        self.cur_module = m.name
        scope = Scope()
        t = self.infer(d.value, scope)
        self._expect(t, ci.type, d.value.span,
                     f"the value of constant {d.name!r} must match its type")

    def _check_test(self, d: A.TestDecl, m: A.Module):
        self.cur_module = m.name
        fake = FnInfo(qualname=f"{m.name}.test:{d.name}", name=d.name,
                      module=m.name, decl=None, type=TY.TFun([], TY.BOOL),
                      tparams=[], declared=TY.EffectSet())
        saved = self.cur
        self.cur = fake
        scope = Scope()
        t = self.infer(d.body, scope)
        self._expect(t, TY.BOOL, d.body.span,
                     "a test body must evaluate to Bool")
        self.cur = saved

    # ================================================== inference

    def infer(self, e, scope: Scope) -> TY.Type:
        if e is None:
            return TY.UNIT

        m = getattr(self, "_infer_" + type(e).__name__, None)
        if m is None:
            self.bag.error("CANON-E0308",
                           f"cannot infer {type(e).__name__}", e.span)
            return TY.NEVER
        return m(e, scope)

    # -- literals and names ---------------------------------------------

    def _infer_Lit(self, e: A.Lit, scope) -> TY.Type:
        return {"int": TY.INT, "dec": TY.DEC, "text": TY.TEXT,
                "bool": TY.BOOL, "unit": TY.UNIT}[e.lit_kind]

    def _infer_Var(self, e: A.Var, scope) -> TY.Type:
        t = scope.find(e.name)
        if t is not None:
            if e.name == "result" and not self.in_ensures:
                self.bag.error(
                    "CANON-E0201",
                    "`result` is only bound inside an `ensures` clause", e.span,
                    facts={"name": "result"})
            return t

        ci = self.env.lookup_const(e.name, self.cur_module)
        if ci is not None:
            self._note_dep(ci.qualname)
            return ci.type

        fi = self.env.lookup_fn(e.name, self.cur_module)
        if fi is not None:
            self._note_call(fi)
            inst, _ = TY.instantiate(fi.type, fi.tparams)
            return inst

        b = P.lookup(e.name)
        if b is not None:
            inst, _ = TY.instantiate(b.type, b.tparams)
            return inst

        if e.name in self.env.ctors:
            ty, _payload = self._ctor_type(e.name, [], e.span)
            return ty

        candidates = (set(self.env.fns) | set(P.BUILTINS)
                      | set(self.env.consts) | set(self._scope_names(scope)))
        self.bag.error(
            "CANON-E0201", f"unknown name {e.name!r}", e.span,
            facts={"name": e.name, "in_scope": sorted(self._scope_names(scope))},
            repairs=self._near_repairs(e.name, candidates, e.span))
        return TY.NEVER

    def _infer_QualVar(self, e: A.QualVar, scope) -> TY.Type:
        full = e.full()
        b = P.lookup(full)
        if b is not None:
            inst, _ = TY.instantiate(b.type, b.tparams)
            return inst
        fi = self.env.fns.get(full) or self.env.lookup_fn(e.name, e.module)
        if fi is not None:
            self._note_call(fi)
            inst, _ = TY.instantiate(fi.type, fi.tparams)
            return inst
        ci = self.env.consts.get(full)
        if ci is not None:
            self._note_dep(full)
            return ci.type
        self.bag.error(
            "CANON-E0201", f"unknown name {full!r}", e.span,
            facts={"name": full, "module": e.module},
            repairs=self._near_repairs(full, set(P.BUILTINS) | set(self.env.fns),
                                       e.span))
        return TY.NEVER

    def _scope_names(self, scope):
        out = set()
        s = scope
        while s is not None:
            out |= set(s.vars)
            s = s.parent
        return out

    # -- structure -------------------------------------------------------

    def _infer_Field(self, e: A.Field, scope) -> TY.Type:
        # `a.b` where `a` is a module name is a qualified reference.
        if isinstance(e.target, A.Var) and e.target.name in self.env.modules \
                and scope.find(e.target.name) is None:
            return self._infer_QualVar(
                A.QualVar(module=e.target.name, name=e.name, span=e.span), scope)

        t = TY.prune(self.infer(e.target, scope))

        if isinstance(t, TY.TRec):
            if e.name in t.fields:
                return t.fields[e.name]
            self.bag.error(
                "CANON-E0206", f"no field {e.name!r} on {TY.show(t)}", e.span,
                facts={"field": e.name, "available": sorted(t.fields)},
                repairs=self._near_repairs(e.name, t.fields, e.span))
            return TY.NEVER

        if isinstance(t, TY.TCon):
            ti = self.env.types.get(t.name)
            if ti is not None and ti.kind == "record":
                for f in ti.decl.fields:
                    if f.name == e.name:
                        saved = self.tparams
                        self.tparams = set(ti.tparams)
                        ft = self.resolve_type(f.ty, f.span)
                        self.tparams = saved
                        self._note_classification(ti, f.name)
                        return TY.subst(ft, dict(zip(ti.tparams, t.args)))
                self.bag.error(
                    "CANON-E0206", f"{t.name} has no field {e.name!r}", e.span,
                    facts={"record": t.name, "field": e.name,
                           "available": [f.name for f in ti.decl.fields]},
                    repairs=self._near_repairs(
                        e.name, [f.name for f in ti.decl.fields], e.span))
                return TY.NEVER
            if t.name == "Never":
                return TY.NEVER

        self.bag.error(
            "CANON-E0304",
            f"field access on {TY.show(t)}, which is not a record", e.span,
            facts={"type": TY.show(TY.zonk(t)), "field": e.name})
        return TY.NEVER

    def _note_classification(self, ti: TypeInfo, field_name: str):
        if self.cur is None:
            return
        cls = getattr(ti.decl, "classification", {}).get(field_name)
        if cls:
            self.cur.touches.add(cls)

    def _infer_RecordLit(self, e: A.RecordLit, scope) -> TY.Type:
        ti = self.env.types.get(e.type_name)
        if ti is None or ti.kind != "record":
            self.bag.error(
                "CANON-E0202", f"unknown record type {e.type_name!r}", e.span,
                facts={"name": e.type_name,
                       "known": sorted(n for n, i in self.env.types.items()
                                       if i.kind == "record")},
                repairs=self._near_repairs(e.type_name, self.env.types, e.span))
            for _, v in e.fields:
                self.infer(v, scope)
            return TY.NEVER

        self._note_dep(e.type_name)
        args = [TY.fresh(p) for p in ti.tparams]
        mapping = dict(zip(ti.tparams, args))
        declared = {}
        saved = self.tparams
        self.tparams = set(ti.tparams)
        for f in ti.decl.fields:
            declared[f.name] = TY.subst(self.resolve_type(f.ty, f.span), mapping)
        self.tparams = saved

        if e.base is not None:
            bt = self.infer(e.base, scope)
            self._expect(bt, TY.TCon(e.type_name, args), e.base.span,
                         "the base of a record update must have the same type")

        given = set()
        for name, val in e.fields:
            if name not in declared:
                self.bag.error(
                    "CANON-E0206",
                    f"{e.type_name} has no field {name!r}", val.span,
                    facts={"record": e.type_name, "field": name,
                           "available": sorted(declared)},
                    repairs=self._near_repairs(name, declared, val.span))
                self.infer(val, scope)
                continue
            given.add(name)
            vt = self.infer(val, scope)
            self._expect(vt, declared[name], val.span,
                         f"field {e.type_name}.{name}")

        if e.base is None:
            missing = sorted(set(declared) - given)
            if missing:
                fill = ", ".join(f"{n}: <{TY.show(declared[n])}>"
                                 for n in missing)
                self.bag.error(
                    "CANON-E0301",
                    f"{e.type_name} literal is missing "
                    f"{len(missing)} field{'' if len(missing) == 1 else 's'}: "
                    + ", ".join(missing),
                    e.span,
                    facts={"record": e.type_name, "missing": missing,
                           "types": {n: TY.show(declared[n]) for n in missing}},
                    repairs=[Repair("insert-before",
                                    "supply the missing fields", fill,
                                    e.span, 0.6)])
        return TY.TCon(e.type_name, args)

    def _ctor_type(self, name, args, span):
        """Returns (result type, payload types). Payload is None if unknown."""
        ci = self.env.ctors.get(name)
        if ci is None:
            self.bag.error(
                "CANON-E0207", f"unknown constructor {name!r}", span,
                facts={"name": name, "known": sorted(self.env.ctors)},
                repairs=self._near_repairs(name, self.env.ctors, span))
            return TY.NEVER, None
        self._note_dep(ci.type_name)
        metas = [TY.fresh(p) for p in ci.tparams]
        mapping = dict(zip(ci.tparams, metas))
        saved = self.tparams
        self.tparams = set(ci.tparams)
        payload = [TY.subst(self.resolve_type(p, span), mapping)
                   for p in ci.payload]
        self.tparams = saved

        if len(args) != len(payload):
            self.bag.error(
                "CANON-E0309",
                f"constructor {name!r} takes {len(payload)} argument"
                f"{'' if len(payload) == 1 else 's'}, found {len(args)}",
                span,
                facts={"ctor": name, "expected": len(payload),
                       "found": len(args),
                       "types": [TY.show(p) for p in payload]})
        return TY.TCon(ci.type_name, metas), payload

    def _infer_CtorCall(self, e: A.CtorCall, scope) -> TY.Type:
        ty, payload = self._ctor_type(e.name, e.args, e.span)
        if payload is None:
            for a in e.args:
                self.infer(a, scope)
            return TY.NEVER
        for a, want in zip(e.args, payload):
            got = self.infer(a, scope)
            self._expect(got, want, a.span, f"argument to {e.name}")
        for a in e.args[len(payload):]:
            self.infer(a, scope)
        return ty

    def _infer_ListLit(self, e: A.ListLit, scope) -> TY.Type:
        if not e.items:
            return TY.list_of(TY.fresh("item"))
        first = self.infer(e.items[0], scope)
        for it in e.items[1:]:
            t = self.infer(it, scope)
            self._expect(t, first, it.span,
                         "every element of a list literal must have the "
                         "same type")
        return TY.list_of(first)

    def _infer_MapLit(self, e: A.MapLit, scope) -> TY.Type:
        if not e.entries:
            return TY.map_of(TY.fresh("k"), TY.fresh("v"))
        kt = self.infer(e.entries[0][0], scope)
        vt = self.infer(e.entries[0][1], scope)
        for k, v in e.entries[1:]:
            self._expect(self.infer(k, scope), kt, k.span, "map key")
            self._expect(self.infer(v, scope), vt, v.span, "map value")
        return TY.map_of(kt, vt)

    # -- application -----------------------------------------------------

    def _infer_Call(self, e: A.Call, scope) -> TY.Type:
        # `abort(...)` and `old(...)` are handled before general application so
        # their special typing rules apply.
        if isinstance(e.fn, A.Var) and e.fn.name == "abort":
            for a in e.args:
                self.infer(a, scope)
            return TY.NEVER
        if isinstance(e.fn, A.Var) and e.fn.name == "old":
            if not self.in_ensures:
                self.bag.error(
                    "CANON-E0504",
                    "`old` may only be used inside an `ensures` clause",
                    e.span, facts={"name": "old"})
            if len(e.args) != 1:
                self.bag.error("CANON-E0302",
                               "`old` takes exactly one argument", e.span,
                               facts={"expected": 1, "found": len(e.args)})
                return TY.NEVER
            return self.infer(e.args[0], scope)

        ft = TY.prune(self.infer(e.fn, scope))

        if isinstance(ft, TY.TCon) and ft.name == "Never":
            for a in e.args:
                self.infer(a, scope)
            return TY.NEVER

        if not isinstance(ft, TY.TFun):
            self.bag.error(
                "CANON-E0303", f"{TY.show(TY.zonk(ft))} is not callable",
                e.span, facts={"type": TY.show(TY.zonk(ft))})
            for a in e.args:
                self.infer(a, scope)
            return TY.NEVER

        if len(e.args) != len(ft.params):
            name = self._callee_name(e.fn)
            self.bag.error(
                "CANON-E0302",
                f"{name} takes {len(ft.params)} argument"
                f"{'' if len(ft.params) == 1 else 's'}, found {len(e.args)}",
                e.span,
                facts={"callee": name, "expected": len(ft.params),
                       "found": len(e.args),
                       "signature": TY.show(TY.zonk(ft))})
        for a, want in zip(e.args, ft.params):
            got = self.infer(a, scope)
            self._expect(got, want, a.span,
                         f"argument to {self._callee_name(e.fn)}")
        for a in e.args[len(ft.params):]:
            self.infer(a, scope)

        # Calling a value of function type performs that type's effects.
        for key in ft.effects:
            self._perform(key, e.span)
        return ft.result

    def _callee_name(self, fn) -> str:
        if isinstance(fn, A.Var):
            return repr(fn.name)
        if isinstance(fn, A.QualVar):
            return repr(fn.full())
        if isinstance(fn, A.Field):
            return repr(fn.name)
        return "this function"

    def _infer_Lambda(self, e: A.Lambda, scope) -> TY.Type:
        inner = Scope(scope)
        params = []
        for p in e.params:
            pt = self.resolve_type(p.ty, p.span)
            inner.declare(p.name, pt, p.span)
            params.append(pt)
        before = set(self.cur.performed) if self.cur else set()
        rt = self.infer(e.body, inner)
        after = set(self.cur.performed) if self.cur else set()
        self._warn_unused(inner, e, kind="parameter")
        return TY.TFun(params, rt, frozenset(after - before))

    def _infer_Perform(self, e: A.Perform, scope) -> TY.Type:
        # A local of function type shadows an effect of the same name.
        local = scope.find(e.effect)
        if local is not None:
            call = A.Call(fn=A.Field(target=A.Var(name=e.effect, span=e.span),
                                     name=e.op, span=e.span),
                          args=e.args, span=e.span)
            return self._infer_Call(call, scope)

        ei = self.env.effects.get(e.effect)
        if ei is None:
            # Could be a module-qualified function call: payments.refund(x)
            fi = self.env.fns.get(f"{e.effect}.{e.op}")
            if fi is not None:
                call = A.Call(
                    fn=A.QualVar(module=e.effect, name=e.op, span=e.span),
                    args=e.args, span=e.span)
                return self._infer_Call(call, scope)
            b = P.lookup(f"{e.effect}.{e.op}")
            if b is not None:
                call = A.Call(
                    fn=A.QualVar(module=e.effect, name=e.op, span=e.span),
                    args=e.args, span=e.span)
                return self._infer_Call(call, scope)
            self.bag.error(
                "CANON-E0204", f"unknown effect or module {e.effect!r}", e.span,
                facts={"name": e.effect,
                       "known_effects": sorted(self.env.effects),
                       "known_modules": sorted(self.env.modules)},
                repairs=self._near_repairs(
                    e.effect,
                    set(self.env.effects) | set(self.env.modules), e.span))
            for a in e.args:
                self.infer(a, scope)
            return TY.NEVER

        op = ei.ops.get(e.op)
        if op is None:
            self.bag.error(
                "CANON-E0205",
                f"effect {e.effect!r} has no operation {e.op!r}", e.span,
                facts={"effect": e.effect, "op": e.op,
                       "available": sorted(ei.ops)},
                repairs=self._near_repairs(e.op, ei.ops, e.span))
            for a in e.args:
                self.infer(a, scope)
            return TY.NEVER

        saved = self.tparams
        self.tparams = set()
        want = [self.resolve_type(p.ty, p.span) for p in op.params]
        rt = self.resolve_type(op.result, op.span)
        self.tparams = saved

        if len(e.args) != len(want):
            self.bag.error(
                "CANON-E0302",
                f"{e.effect}.{e.op} takes {len(want)} argument"
                f"{'' if len(want) == 1 else 's'}, found {len(e.args)}",
                e.span,
                facts={"operation": f"{e.effect}.{e.op}",
                       "expected": len(want), "found": len(e.args),
                       "parameters": [f"{p.name}: {TY.show(t)}"
                                      for p, t in zip(op.params, want)]})
        for a, w in zip(e.args, want):
            got = self.infer(a, scope)
            self._expect(got, w, a.span, f"argument to {e.effect}.{e.op}")
        for a in e.args[len(want):]:
            self.infer(a, scope)

        self._perform(f"{e.effect}.{e.op}", e.span)
        return rt

    def _perform(self, key: str, span):
        if self.cur is not None:
            self.cur.performed.add(key)

    def _note_call(self, fi: FnInfo):
        if self.cur is not None:
            self.cur.calls.add(fi.qualname)

    def _note_dep(self, qualname: str):
        pass

    # -- control flow ----------------------------------------------------

    def _infer_Let(self, e: A.Let, scope) -> TY.Type:
        vt = self.infer(e.value, scope)
        if e.ty is not None:
            want = self.resolve_type(e.ty, e.span)
            self._expect(vt, want, e.value.span,
                         f"the annotated type of {e.name!r}")
            vt = want
        inner = Scope(scope)
        self._warn_shadow(e.name, scope, e.span)
        inner.declare(e.name, vt, e.span)
        rt = self.infer(e.body, inner)
        self._warn_unused(inner, e)
        return rt

    def _infer_If(self, e: A.If, scope) -> TY.Type:
        ct = self.infer(e.cond, scope)
        self._expect(ct, TY.BOOL, e.cond.span,
                     "an `if` condition must be Bool",
                     note="There is no truthiness in Canon; compare explicitly.")
        tt = self.infer(e.then, scope)
        et = self.infer(e.otherwise, scope)
        try:
            TY.unify(tt, et)
        except TY.Mismatch:
            self.bag.error(
                "CANON-E0307",
                "the branches of an `if` have different types", e.span,
                facts={"then": TY.show(TY.zonk(tt)),
                       "else": TY.show(TY.zonk(et))},
                related=[(e.then.span, f"this branch is {TY.show(TY.zonk(tt))}"),
                         (e.otherwise.span,
                          f"this branch is {TY.show(TY.zonk(et))}")])
            return TY.NEVER
        return tt if not _is_never(tt) else et

    def _infer_Block(self, e: A.Block, scope) -> TY.Type:
        inner = Scope(scope)
        for s in e.stmts:
            if isinstance(s, A.SLet):
                vt = self.infer(s.value, inner)
                if s.ty is not None:
                    want = self.resolve_type(s.ty, s.span)
                    self._expect(vt, want, s.value.span,
                                 f"the annotated type of {s.name!r}")
                    vt = want
                self._warn_shadow(s.name, inner, s.span)
                inner.declare(s.name, vt, s.span)
            elif isinstance(s, A.SExpr):
                t = TY.prune(self.infer(s.value, inner))
                if not _is_unit_like(t):
                    self.bag.warn(
                        "CANON-W0001",
                        f"the result of this `do` statement is discarded "
                        f"({TY.show(TY.zonk(t))})",
                        s.span, facts={"type": TY.show(TY.zonk(t))},
                        repairs=[Repair("manual",
                                        "bind it with `let` if it is needed, "
                                        "or drop the statement")])
            elif isinstance(s, A.SAssert):
                t = self.infer(s.cond, inner)
                self._expect(t, TY.BOOL, s.cond.span,
                             "an `assert` condition must be Bool")
        rt = self.infer(e.result, inner)
        self._warn_unused(inner, e)
        return rt

    def _infer_Match(self, e: A.Match, scope) -> TY.Type:
        st = self.infer(e.scrutinee, scope)
        result = TY.fresh("match")
        covered = []
        for arm in e.arms:
            inner = Scope(scope)
            self.check_pattern(arm.pattern, st, inner)
            covered.append(arm.pattern)
            if arm.guard is not None:
                gt = self.infer(arm.guard, inner)
                self._expect(gt, TY.BOOL, arm.guard.span,
                             "a match guard must be Bool")
            bt = self.infer(arm.body, inner)
            try:
                TY.unify(result, bt)
            except TY.Mismatch:
                self.bag.error(
                    "CANON-E0307",
                    "match arms produce different types", arm.body.span,
                    facts={"expected": TY.show(TY.zonk(result)),
                           "found": TY.show(TY.zonk(bt))})
            self._warn_unused(inner, arm, kind="binding")
        self._check_exhaustive(e, st, covered)
        return result

    def _infer_Try(self, e: A.Try, scope) -> TY.Type:
        t = TY.prune(self.infer(e.operand, scope))
        if isinstance(t, TY.TCon) and t.name == "Never":
            return TY.NEVER
        if not (isinstance(t, TY.TCon) and t.name == "Result"):
            self.bag.error(
                "CANON-E0301",
                f"`?` requires a Result, found {TY.show(TY.zonk(t))}",
                e.span, facts={"found": TY.show(TY.zonk(t))},
                repairs=[Repair("manual",
                                "use `match` if the value is an Option, or "
                                "convert it with Option.to_result")])
            return TY.NEVER
        ok_t, err_t = t.args
        rt = TY.prune(self.result_type) if self.result_type else None
        if not (isinstance(rt, TY.TCon) and rt.name == "Result"):
            self.bag.error(
                "CANON-E0310",
                "`?` may only be used in a function that returns Result",
                e.span,
                facts={"function_result": TY.show(TY.zonk(rt)) if rt else "?"},
                repairs=[Repair("manual",
                                "change the result type to Result<_, _>, or "
                                "handle the error with `match`")])
            return ok_t
        try:
            TY.unify(err_t, rt.args[1])
        except TY.Mismatch:
            self.bag.error(
                "CANON-E0301",
                "the error type propagated by `?` does not match the "
                "function's error type", e.span,
                facts={"propagated": TY.show(TY.zonk(err_t)),
                       "expected": TY.show(TY.zonk(rt.args[1]))},
                repairs=[Repair("manual",
                                "map the error with Result.map_err before `?`")])
        return ok_t

    # -- the model primitive ---------------------------------------------

    def _infer_Ask(self, e: A.Ask, scope) -> TY.Type:
        rt = self.resolve_type(e.result_type, e.span)
        s = e.spec or A.AskSpec()

        if s.system is not None:
            t = self.infer(s.system, scope)
            self._expect(t, TY.TEXT, s.system.span,
                         "an ask `system` instruction must be Text")
        for label, v in s.inputs:
            self.infer(v, scope)
        for g in s.grounded_in:
            t = self.infer(g, scope)
            self._expect(t, TY.TEXT, g.span,
                         "a `grounded_in` source must be Text",
                         note="Grounding compares the model's output against "
                              "this text, so it has to be text.")
        for a, b in s.examples:
            at = self.infer(a, scope)
            bt = self.infer(b, scope)
            self._expect(bt, rt, b.span,
                         "an example output must have the ask's result type")
        if s.judge is not None:
            t = self.infer(s.judge, scope)
            if not (isinstance(TY.prune(t), TY.TFun)):
                self.bag.error(
                    "CANON-E0301",
                    "a `judge` must be a function from the result to Bool",
                    s.judge.span, facts={"found": TY.show(TY.zonk(t))})
            else:
                ft = TY.prune(t)
                if len(ft.params) == 1:
                    self._expect(ft.params[0], rt, s.judge.span,
                                 "the judge's parameter")
                    self._expect(ft.result, TY.BOOL, s.judge.span,
                                 "the judge's result")

        self._check_model(e, s)

        unknown = [r for r in s.retry_on if r not in RETRY_REASONS]
        for r in unknown:
            self.bag.error(
                "CANON-E0201", f"unknown retry reason {r!r}", e.span,
                facts={"reason": r, "known": sorted(RETRY_REASONS)},
                repairs=self._near_repairs(r, RETRY_REASONS, e.span))

        if not _is_coercible(rt, self.env):
            self.bag.error(
                "CANON-E0301",
                f"a model cannot produce {TY.show(TY.zonk(rt))}", e.span,
                facts={"type": TY.show(TY.zonk(rt))},
                notes=["An ask result must be a type the runtime can build "
                       "from a model response: a primitive, a list, a map, an "
                       "option, a record of such, or an enum."])

        if self.cur is not None:
            self.cur.models.add(e.model)
        self._perform("model.infer", e.span)
        if s.grounded_in:
            self._perform("model.judge", e.span)
        return rt

    def _check_model(self, e: A.Ask, s: A.AskSpec):
        """
        Validate the target model and the settings against its capabilities.

        Model capabilities are checked here rather than at runtime because the
        failure is static: a model that rejects `temperature` will reject it on
        every call, and finding that out from a 400 in production is strictly
        worse than finding it out from the checker.
        """
        from .model import lookup_model, known_models

        info = lookup_model(e.model)
        if info is None:
            self.bag.error(
                "CANON-E0201", f"unknown model {e.model!r}", e.span,
                facts={"model": e.model, "known": known_models()},
                repairs=self._near_repairs(e.model, known_models(), e.span))
            return

        if s.temperature is not None:
            if not (0 <= s.temperature <= 2):
                self.bag.error(
                    "CANON-E0301",
                    f"temperature must be between 0 and 2, "
                    f"found {s.temperature}",
                    e.span, facts={"temperature": str(s.temperature)})
            elif not info.accepts_temperature:
                supported = sorted(
                    a for a in known_models()
                    if lookup_model(a).accepts_temperature)
                self.bag.error(
                    "CANON-E0301",
                    f"{e.model} does not accept a temperature setting",
                    e.span,
                    facts={"model": e.model, "model_id": info.id,
                           "models_accepting_temperature": supported},
                    repairs=[Repair(
                        "manual",
                        "remove the temperature line; steer the model with "
                        "the system instruction and contracts instead")],
                    notes=["This model rejects the parameter outright rather "
                           "than ignoring it, so the clause would fail every "
                           "call at runtime."])

        if s.max_tokens is not None and s.max_tokens > info.max_output:
            self.bag.error(
                "CANON-E0301",
                f"max_tokens {s.max_tokens} exceeds the {info.id} limit of "
                f"{info.max_output}",
                e.span,
                facts={"requested": s.max_tokens, "limit": info.max_output,
                       "model": info.id},
                repairs=[Repair("replace-span", "use the model's limit",
                                str(info.max_output), e.span, 0.8)])

        if self.cur is not None and self.cur.decl is not None:
            cost = self.cur.decl.cost
            if s.max_tokens and cost and cost.tokens is not None \
                    and s.max_tokens > cost.tokens:
                self.bag.warn(
                    "CANON-W0006",
                    f"this ask may use {s.max_tokens} tokens but the function "
                    f"declares a budget of {cost.tokens}",
                    e.span,
                    facts={"ask_max_tokens": s.max_tokens,
                           "declared_budget": cost.tokens},
                    repairs=[Repair("manual",
                                    "raise the function's token budget or "
                                    "lower max_tokens")])

    # ================================================== patterns

    def check_pattern(self, p, expected: TY.Type, scope: Scope):
        expected = TY.prune(expected)

        if isinstance(p, A.PWild):
            return
        if isinstance(p, A.PVar):
            self._warn_shadow(p.name, scope, p.span)
            scope.declare(p.name, expected, p.span)
            return
        if isinstance(p, A.PLit):
            lt = {"int": TY.INT, "dec": TY.DEC, "text": TY.TEXT,
                  "bool": TY.BOOL, "unit": TY.UNIT}[p.lit_kind]
            self._expect(lt, expected, p.span, "a literal pattern")
            return

        if isinstance(p, A.PCtor):
            ci = self.env.ctors.get(p.name)
            if ci is None:
                self.bag.error(
                    "CANON-E0207", f"unknown constructor {p.name!r}", p.span,
                    facts={"name": p.name, "known": sorted(self.env.ctors)},
                    repairs=self._near_repairs(p.name, self.env.ctors, p.span))
                return
            metas = [TY.fresh(t) for t in ci.tparams]
            self._expect(TY.TCon(ci.type_name, metas), expected, p.span,
                         f"a {ci.type_name} pattern")
            mapping = dict(zip(ci.tparams, metas))
            saved = self.tparams
            self.tparams = set(ci.tparams)
            payload = [TY.subst(self.resolve_type(t, p.span), mapping)
                       for t in ci.payload]
            self.tparams = saved
            if len(p.args) != len(payload):
                self.bag.error(
                    "CANON-E0309",
                    f"pattern {p.name!r} takes {len(payload)} argument"
                    f"{'' if len(payload) == 1 else 's'}, found {len(p.args)}",
                    p.span,
                    facts={"ctor": p.name, "expected": len(payload),
                           "found": len(p.args)})
            for sub, want in zip(p.args, payload):
                self.check_pattern(sub, want, scope)
            return

        if isinstance(p, A.PRecord):
            ti = self.env.types.get(p.type_name)
            if ti is None or ti.kind != "record":
                self.bag.error(
                    "CANON-E0202",
                    f"unknown record type {p.type_name!r} in pattern", p.span,
                    facts={"name": p.type_name})
                return
            metas = [TY.fresh(t) for t in ti.tparams]
            self._expect(TY.TCon(p.type_name, metas), expected, p.span,
                         f"a {p.type_name} pattern")
            mapping = dict(zip(ti.tparams, metas))
            saved = self.tparams
            self.tparams = set(ti.tparams)
            declared = {f.name: TY.subst(self.resolve_type(f.ty, f.span), mapping)
                        for f in ti.decl.fields}
            self.tparams = saved
            for fname, sub in p.fields:
                if fname not in declared:
                    self.bag.error(
                        "CANON-E0206",
                        f"{p.type_name} has no field {fname!r}", p.span,
                        facts={"record": p.type_name, "field": fname,
                               "available": sorted(declared)},
                        repairs=self._near_repairs(fname, declared, p.span))
                    continue
                self.check_pattern(sub, declared[fname], scope)
            return

        if isinstance(p, A.PList):
            item = TY.fresh("item")
            self._expect(TY.list_of(item), expected, p.span, "a list pattern")
            for sub in p.items:
                self.check_pattern(sub, item, scope)
            if p.rest:
                scope.declare(p.rest, TY.list_of(item), p.span)
            return

    # ================================================== exhaustiveness

    def _check_exhaustive(self, e: A.Match, st: TY.Type, patterns: list):
        st = TY.prune(st)
        has_catch_all = any(
            isinstance(p, (A.PWild, A.PVar)) for p in patterns)

        # An arm with a guard cannot be counted as covering anything, since the
        # guard may fail at runtime.
        unguarded = [arm.pattern for arm in e.arms if arm.guard is None]
        has_catch_all = any(isinstance(p, (A.PWild, A.PVar)) for p in unguarded)

        variants = self._variants_of(st)
        if variants is None:
            info = MatchInfo(e.span, TY.show(TY.zonk(st)), [], has_catch_all)
            self.matches.append(info)
            if not has_catch_all:
                self.bag.error(
                    "CANON-E0305",
                    f"match on {TY.show(TY.zonk(st))} is not exhaustive",
                    e.span,
                    facts={"type": TY.show(TY.zonk(st)),
                           "reason": "type has no finite set of constructors"},
                    repairs=[Repair("manual",
                                    "add a `case _ =>` arm covering the rest")])
            return

        covered = []
        for p in unguarded:
            if isinstance(p, A.PCtor):
                covered.append(p.name)
            elif isinstance(p, A.PRecord):
                covered.append(p.type_name)
            elif isinstance(p, A.PLit) and p.lit_kind == "bool":
                covered.append("true" if p.value else "false")

        info = MatchInfo(e.span, TY.show(TY.zonk(st)), covered, True)
        self.matches.append(info)

        if has_catch_all:
            # A catch-all after full coverage is dead code.
            if set(variants) <= set(covered):
                for arm in e.arms:
                    if isinstance(arm.pattern, (A.PWild, A.PVar)) \
                            and arm.guard is None:
                        self.bag.warn(
                            "CANON-E0306",
                            "this arm is unreachable: every constructor is "
                            "already covered",
                            arm.span, facts={"covered": sorted(set(covered))})
                        break
            return

        missing = [v for v in variants if v not in covered]
        if missing:
            arms = "\n".join(f"case {v} => <expression>" for v in missing)
            info.exhaustive = False
            self.bag.error(
                "CANON-E0305",
                f"match on {TY.show(TY.zonk(st))} does not cover "
                + ", ".join(missing),
                e.span,
                facts={"type": TY.show(TY.zonk(st)), "missing": missing,
                       "covered": sorted(set(covered))},
                repairs=[Repair("manual", "add the missing arms", arms,
                                e.span, 0.7)],
                notes=["Canon has no runtime match failure. Every match must "
                       "be total, so an unhandled case is a compile error "
                       "rather than a production incident."])

        seen = set()
        for arm in e.arms:
            if arm.guard is not None:
                continue
            key = None
            if isinstance(arm.pattern, A.PCtor) and not arm.pattern.args:
                key = arm.pattern.name
            if key is not None:
                if key in seen:
                    self.bag.warn(
                        "CANON-E0306",
                        f"this arm is unreachable: {key} is already covered",
                        arm.span, facts={"constructor": key})
                seen.add(key)

    def _variants_of(self, t):
        t = TY.prune(t)
        if isinstance(t, TY.TCon):
            if t.name in TY.PRELUDE_VARIANTS:
                return list(TY.PRELUDE_VARIANTS[t.name])
            ti = self.env.types.get(t.name)
            if ti is not None and ti.kind == "enum":
                return [v.name for v in ti.decl.variants]
        return None

    # ================================================== operators

    def _infer_Binary(self, e: A.Binary, scope) -> TY.Type:
        if e.op == "|>":
            # x |> f  is  f(x); x |> f(a) is f(a, x)
            if isinstance(e.right, A.Call):
                call = A.Call(fn=e.right.fn,
                              args=list(e.right.args) + [e.left], span=e.span)
            else:
                call = A.Call(fn=e.right, args=[e.left], span=e.span)
            return self._infer_Call(call, scope)

        lt = self.infer(e.left, scope)
        rt = self.infer(e.right, scope)

        if e.op in DIV_OPS:
            name = "Int.div" if _same(lt, TY.INT) else "Dec.div"
            if e.op == "%":
                name = "Int.rem"
            self.bag.error(
                "CANON-E0301",
                f"there is no `{e.op}` operator in Canon", e.span,
                facts={"operator": e.op, "use_instead": name},
                repairs=[Repair(
                    "replace-span",
                    f"use {name}, which returns Option and cannot fault",
                    f"{name}({_src(e.left)}, {_src(e.right)})", e.span, 0.8)],
                notes=["Division is the only partial arithmetic operation. "
                       "Making it return Option keeps every expression total, "
                       "so no generated program can fault on a zero divisor."])
            return TY.NEVER

        if e.op in BOOL_OPS:
            self._expect(lt, TY.BOOL, e.left.span, f"the left side of `{e.op}`")
            self._expect(rt, TY.BOOL, e.right.span, f"the right side of `{e.op}`")
            return TY.BOOL

        if e.op == "++":
            lp = TY.prune(lt)
            if _same(lp, TY.TEXT):
                self._expect(rt, TY.TEXT, e.right.span, "the right side of `++`")
                return TY.TEXT
            if isinstance(lp, TY.TCon) and lp.name == "List":
                self._expect(rt, lp, e.right.span, "the right side of `++`")
                return lp
            self.bag.error(
                "CANON-E0301",
                f"`++` joins Text or List, found {TY.show(TY.zonk(lt))}",
                e.span, facts={"found": TY.show(TY.zonk(lt))})
            return TY.NEVER

        if e.op in ARITH_OPS:
            lp = TY.prune(lt)
            if _same(lp, TY.TEXT) and e.op == "+":
                self.bag.error(
                    "CANON-E0301",
                    "`+` does not concatenate Text", e.span,
                    facts={"operator": "+", "use_instead": "++"},
                    repairs=[Repair("replace-span", "use `++`", "++",
                                    e.span, 0.9)])
                return TY.TEXT
            try:
                TY.unify(lt, rt)
            except TY.Mismatch:
                self.bag.error(
                    "CANON-E0301",
                    f"`{e.op}` needs both sides to have the same type", e.span,
                    facts={"left": TY.show(TY.zonk(lt)),
                           "right": TY.show(TY.zonk(rt)),
                           "operator": e.op},
                    repairs=self._numeric_repairs(lt, rt, e),
                    notes=["Canon does not convert between Int and Dec "
                           "implicitly, because silent numeric widening is a "
                           "common source of monetary error."])
                return TY.NEVER
            lp = TY.prune(lt)
            if not (_same(lp, TY.INT) or _same(lp, TY.DEC)
                    or isinstance(lp, TY.TMeta) or _is_never(lp)):
                self.bag.error(
                    "CANON-E0301",
                    f"`{e.op}` needs Int or Dec, found {TY.show(TY.zonk(lp))}",
                    e.span, facts={"found": TY.show(TY.zonk(lp)),
                                   "operator": e.op})
                return TY.NEVER
            return lt

        if e.op in CMP_OPS:
            try:
                TY.unify(lt, rt)
            except TY.Mismatch:
                self.bag.error(
                    "CANON-E0301",
                    f"`{e.op}` compares values of the same type", e.span,
                    facts={"left": TY.show(TY.zonk(lt)),
                           "right": TY.show(TY.zonk(rt))},
                    repairs=self._numeric_repairs(lt, rt, e))
                return TY.BOOL
            lp = TY.prune(lt)
            if not (_same(lp, TY.INT) or _same(lp, TY.DEC) or _same(lp, TY.TEXT)
                    or _same(lp, TY.TIME) or isinstance(lp, TY.TMeta)
                    or _is_never(lp)):
                self.bag.error(
                    "CANON-E0301",
                    f"`{e.op}` is defined for Int, Dec, Text and Time, "
                    f"found {TY.show(TY.zonk(lp))}",
                    e.span, facts={"found": TY.show(TY.zonk(lp))})
            return TY.BOOL

        if e.op in EQ_OPS:
            try:
                TY.unify(lt, rt)
            except TY.Mismatch:
                self.bag.error(
                    "CANON-E0301",
                    f"`{e.op}` compares values of the same type", e.span,
                    facts={"left": TY.show(TY.zonk(lt)),
                           "right": TY.show(TY.zonk(rt))},
                    repairs=self._numeric_repairs(lt, rt, e),
                    notes=["Comparing values of different types is always "
                           "false, so it is rejected rather than allowed."])
            return TY.BOOL

        self.bag.error("CANON-E0301", f"unknown operator {e.op!r}", e.span,
                       facts={"operator": e.op})
        return TY.NEVER

    def _infer_Unary(self, e: A.Unary, scope) -> TY.Type:
        t = self.infer(e.operand, scope)
        if e.op == "not":
            self._expect(t, TY.BOOL, e.operand.span, "the operand of `not`")
            return TY.BOOL
        if e.op == "-":
            tp = TY.prune(t)
            if not (_same(tp, TY.INT) or _same(tp, TY.DEC)
                    or isinstance(tp, TY.TMeta) or _is_never(tp)):
                self.bag.error(
                    "CANON-E0301",
                    f"unary `-` needs Int or Dec, found {TY.show(TY.zonk(tp))}",
                    e.span, facts={"found": TY.show(TY.zonk(tp))})
                return TY.NEVER
            return t
        self.bag.error("CANON-E0301", f"unknown operator {e.op!r}", e.span)
        return TY.NEVER

    def _numeric_repairs(self, lt, rt, e) -> list:
        lp, rp = TY.prune(lt), TY.prune(rt)
        out = []
        if _same(lp, TY.INT) and _same(rp, TY.DEC):
            out.append(Repair("replace-span", "widen the left side to Dec",
                              f"Int.to_dec({_src(e.left)})", e.left.span, 0.85))
        elif _same(lp, TY.DEC) and _same(rp, TY.INT):
            out.append(Repair("replace-span", "widen the right side to Dec",
                              f"Int.to_dec({_src(e.right)})", e.right.span, 0.85))
        return out

    # ================================================== transitive closure

    def _compute_transitive(self):
        """
        Propagate effects along the call graph until it settles.

        The result is the capability footprint of the whole call graph rooted
        at each function, which is what a promotion gate compares against the
        authorisation the change was granted.
        """
        for fi in self.env.fns.values():
            fi.transitive = set(fi.performed)

        changed = True
        rounds = 0
        limit = len(self.env.fns) + 2
        while changed and rounds < limit:
            changed = False
            rounds += 1
            for fi in self.env.fns.values():
                before = len(fi.transitive)
                for callee in fi.calls:
                    other = self.env.fns.get(callee)
                    if other is not None:
                        fi.transitive |= other.transitive
                        fi.models |= other.models
                        fi.touches |= other.touches
                if len(fi.transitive) != before:
                    changed = True

    # ================================================== diagnostics helpers

    def _expect(self, got, want, span, context: str, note: str = ""):
        try:
            TY.unify(got, want)
            return True
        except TY.Mismatch as mm:
            notes = [note] if note else []
            self.bag.error(
                "CANON-E0301",
                f"{context}: expected {TY.show(TY.zonk(want))}, "
                f"found {TY.show(TY.zonk(got))}",
                span,
                facts={"expected": TY.show(TY.zonk(want)),
                       "found": TY.show(TY.zonk(got)),
                       "context": context,
                       "detail": mm.detail},
                notes=notes)
            return False

    def _warn_shadow(self, name, scope, span):
        if scope.find(name) is not None:
            self.bag.warn(
                "CANON-W0003", f"{name!r} shadows an outer binding", span,
                facts={"name": name},
                notes=["Shadowing is legal but makes a definition harder to "
                       "read in isolation."])

    def _warn_unused(self, scope: Scope, node, kind: str = "binding"):
        for name, (_t, span) in scope.declared_here().items():
            if name in scope.used or name.startswith("_") or name == "result":
                continue
            code = "CANON-W0002" if kind == "parameter" else "CANON-W0001"
            self.bag.warn(
                code, f"unused {kind} {name!r}", span or node.span,
                facts={"name": name},
                repairs=[Repair("manual",
                                f"remove it, or rename to `_{name}` to mark "
                                f"it deliberately unused")])

    def _warn_missing_contracts(self, d: A.FnDecl, fi: FnInfo):
        if not d.requires and not d.ensures and not d.laws:
            if not _is_trivial(d):
                self.bag.warn(
                    "CANON-W0004",
                    f"{d.name!r} has no contracts", d.span,
                    facts={"function": d.name},
                    repairs=[Repair(
                        "add-clause",
                        "state at least one property the result must satisfy",
                        "ensures <property of result>", d.span, 0.3)],
                    notes=["A function with no contract can only be verified "
                           "by executing it. The verifier has nothing to "
                           "check and the promotion gate has nothing to "
                           "compare against."])
        if not d.intent:
            self.bag.warn(
                "CANON-W0005", f"{d.name!r} has no stated intent", d.span,
                facts={"function": d.name},
                repairs=[Repair("add-clause", "state what this is for",
                                'intent "..."', d.span, 0.3)],
                notes=["Intent is part of the definition hash and is what an "
                       "Intent specification is matched against."])

    def _near_repairs(self, name, candidates, span) -> list:
        near = _closest(name, candidates)
        return [Repair("replace-span", f"did you mean {n!r}", n, span, 0.55)
                for n in near]


# --------------------------------------------------------------------------
# Laws known to the verifier
# --------------------------------------------------------------------------

LAWS = {
    "idempotent_by": (1, "Calling twice with the same key has the same effect "
                         "as calling once."),
    "deterministic": (0, "Same inputs always give the same result."),
    "total": (0, "Defined for every input satisfying the preconditions."),
    "pure": (0, "Performs no effects."),
    "commutative": (0, "f(a, b) == f(b, a)."),
    "associative": (0, "f(f(a, b), c) == f(a, f(b, c))."),
    "monotonic_in": (1, "Increasing the named parameter never decreases the "
                        "result."),
    "conserves": (1, "The named quantity is the same before and after."),
    "bounded_output": (1, "The result's size never exceeds the given bound."),
    "never_negative": (0, "The result is never negative."),
    "order_independent": (0, "The result does not depend on input order."),
    "injective": (0, "Distinct inputs give distinct outputs."),
    "grounded": (0, "Every claim in the output is supported by the inputs."),
    "explains": (0, "The result carries a reason for every factor used."),
}

RETRY_REASONS = {"contract_violation", "type_error", "refusal", "timeout",
                 "grounding_failure", "judge_rejected"}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _same(a, b) -> bool:
    a = TY.prune(a)
    return isinstance(a, TY.TCon) and isinstance(b, TY.TCon) \
        and a.name == b.name and not a.args and not b.args


def _is_never(t) -> bool:
    t = TY.prune(t)
    return isinstance(t, TY.TCon) and t.name == "Never"


def _is_unit_like(t) -> bool:
    t = TY.prune(t)
    if isinstance(t, TY.TMeta):
        return True
    return isinstance(t, TY.TCon) and t.name in ("Unit", "Never")


def _is_trivial(d: A.FnDecl) -> bool:
    """A one-expression accessor does not need a contract to be trustworthy."""
    b = d.body
    if isinstance(b, A.Block) and not b.stmts:
        b = b.result
    return isinstance(b, (A.Lit, A.Var, A.Field))


def _is_coercible(t, env: Env) -> bool:
    """Whether the runtime can build this type from a model response."""
    t = TY.prune(t)
    if isinstance(t, TY.TMeta) or isinstance(t, TY.TRigid):
        return True
    if isinstance(t, TY.TCon):
        if t.name in ("Int", "Dec", "Text", "Bool", "Unit", "Never"):
            return True
        if t.name in ("List", "Set", "Option"):
            return _is_coercible(t.args[0], env)
        if t.name == "Map":
            return all(_is_coercible(a, env) for a in t.args)
        if t.name == "Result":
            return all(_is_coercible(a, env) for a in t.args)
        ti = env.types.get(t.name)
        if ti is None:
            return False
        if ti.kind == "enum":
            return True
        if ti.kind == "record":
            return True
    if isinstance(t, TY.TRec):
        return all(_is_coercible(v, env) for v in t.fields.values())
    return False


def _src(e) -> str:
    from .canonical import Printer
    try:
        return Printer().expr(e)
    except Exception:
        return "<expr>"


def _closest(name: str, candidates, limit: int = 2) -> list:
    """
    Small edit-distance suggestions.

    Candidates are compared both fully qualified and by their last segment, so
    a misspelling of `compute_total` still matches `payments.compute_total`.
    The suggestion returned is whichever spelling actually matched, since that
    is the one that can be substituted into the source.
    """
    name_l = name.lower()
    threshold = 1 if len(name) <= 4 else (2 if len(name) <= 8 else 3)
    scored = []
    seen = set()
    for c in candidates:
        c = str(c)
        for form in {c, c.rsplit(".", 1)[-1]}:
            if form in seen:
                continue
            d = _edit(name_l, form.lower())
            if d <= threshold:
                seen.add(form)
                scored.append((d, len(form), form))
    scored.sort()
    return [c for _, _, c in scored[:limit]]


def _edit(a: str, b: str) -> int:
    if abs(len(a) - len(b)) > 3:
        return 99
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


# --------------------------------------------------------------------------

def check(modules, bag: Optional[Bag] = None) -> CheckResult:
    return Checker(bag).check(modules)
