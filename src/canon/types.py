"""
the canon type system.

predictable beats clever. every function and lambda parameter and record field
is annotated, so the checker only ever instantiates polymorphism and never has
to generalise it. no let polymorphism, no bidirectional inference, no
subtyping. what you get for that is a type error landing on one concrete spot
with a concrete expected and actual, which is what makes it fixable by a
machine.

no implicit conversion. Int does not turn into Dec, Option does not turn into
what is inside it, nothing is truthy. silent coercion is the main way a
generated program that looks fine does the wrong thing once it is running.

effects are part of the type. a function type carries the set of effect
operations it can perform, so the capability footprint of a call graph is
something you compute rather than something you trust.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# --------------------------------------------------------------------------
# Representation
# --------------------------------------------------------------------------

class Type:
    def __str__(self):
        return show(self)

    def __repr__(self):
        return show(self)


@dataclass(eq=False)
class TCon(Type):
    """A named type constructor applied to zero or more arguments."""
    name: str
    args: list = field(default_factory=list)

    def __eq__(self, other):
        return (isinstance(other, TCon) and other.name == self.name
                and len(other.args) == len(self.args)
                and all(a == b for a, b in zip(self.args, other.args)))

    def __hash__(self):
        return hash((self.name, tuple(map(id, self.args))))


@dataclass(eq=False)
class TMeta(Type):
    """A unification variable created when instantiating a generic signature."""
    id: int
    ref: Optional[Type] = None
    origin: str = ""

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)


@dataclass(eq=False)
class TRigid(Type):
    """A type parameter bound by a declaration; only unifies with itself."""
    name: str

    def __eq__(self, other):
        return isinstance(other, TRigid) and other.name == self.name

    def __hash__(self):
        return hash(("rigid", self.name))


@dataclass(eq=False)
class TFun(Type):
    params: list = field(default_factory=list)
    result: Optional[Type] = None
    effects: frozenset = frozenset()

    def __eq__(self, other):
        return (isinstance(other, TFun)
                and len(other.params) == len(self.params)
                and all(a == b for a, b in zip(self.params, other.params))
                and other.result == self.result
                and other.effects == self.effects)

    def __hash__(self):
        return hash(("fn", len(self.params)))


@dataclass(eq=False)
class TRec(Type):
    """A structural record type. Nominal records are TCon plus a declaration."""
    fields: dict = field(default_factory=dict)

    def __eq__(self, other):
        return (isinstance(other, TRec)
                and set(other.fields) == set(self.fields)
                and all(other.fields[k] == v for k, v in self.fields.items()))

    def __hash__(self):
        return hash(("rec", tuple(sorted(self.fields))))


# --------------------------------------------------------------------------
# Builtin types
# --------------------------------------------------------------------------

INT = TCon("Int")
DEC = TCon("Dec")
TEXT = TCon("Text")
BOOL = TCon("Bool")
UNIT = TCon("Unit")
TIME = TCon("Time")
BYTES = TCon("Bytes")
NEVER = TCon("Never")      # the type of `abort`; unifies with anything

PRIMITIVES = {
    "Int": INT, "Dec": DEC, "Text": TEXT, "Bool": BOOL,
    "Unit": UNIT, "Time": TIME, "Bytes": BYTES, "Never": NEVER,
}

# Generic builtins and their arities.
GENERICS = {
    "List": 1,
    "Set": 1,
    "Map": 2,
    "Option": 1,
    "Result": 2,
}


def list_of(t) -> TCon:
    return TCon("List", [t])


def set_of(t) -> TCon:
    return TCon("Set", [t])


def map_of(k, v) -> TCon:
    return TCon("Map", [k, v])


def option_of(t) -> TCon:
    return TCon("Option", [t])


def result_of(ok, err) -> TCon:
    return TCon("Result", [ok, err])


# Constructors contributed by the prelude, so `Ok`, `Err`, `Some` and `None`
# behave exactly like user-declared enum constructors everywhere.
PRELUDE_CTORS = {
    #  name  -> (type name, type params, payload types as parameter indices)
    "Ok":   ("Result", ["Ok", "Err"], [0]),
    "Err":  ("Result", ["Ok", "Err"], [1]),
    "Some": ("Option", ["T"], [0]),
    "None": ("Option", ["T"], []),
}

PRELUDE_VARIANTS = {
    "Result": ["Ok", "Err"],
    "Option": ["Some", "None"],
    "Bool": ["true", "false"],
}


# --------------------------------------------------------------------------
# Substitution and resolution
# --------------------------------------------------------------------------

_meta_counter = [0]


def fresh(origin: str = "") -> TMeta:
    _meta_counter[0] += 1
    return TMeta(_meta_counter[0], None, origin)


def prune(t: Type) -> Type:
    """Follow bound unification variables to the type they resolved to."""
    while isinstance(t, TMeta) and t.ref is not None:
        t = t.ref
    return t


def subst(t: Type, mapping: dict) -> Type:
    """Replace rigid type variables using `mapping`."""
    t = prune(t)
    if isinstance(t, TRigid):
        return mapping.get(t.name, t)
    if isinstance(t, TCon):
        if not t.args:
            return t
        return TCon(t.name, [subst(a, mapping) for a in t.args])
    if isinstance(t, TFun):
        return TFun([subst(p, mapping) for p in t.params],
                    subst(t.result, mapping), t.effects)
    if isinstance(t, TRec):
        return TRec({k: subst(v, mapping) for k, v in t.fields.items()})
    return t


def instantiate(t: Type, params: list) -> tuple:
    """
    Replace a declaration's type parameters with fresh unification variables.
    Returns (instantiated type, {param name: meta}).
    """
    mapping = {p: fresh(p) for p in params}
    return subst(t, mapping), mapping


def free_rigids(t: Type, out: Optional[set] = None) -> set:
    out = set() if out is None else out
    t = prune(t)
    if isinstance(t, TRigid):
        out.add(t.name)
    elif isinstance(t, TCon):
        for a in t.args:
            free_rigids(a, out)
    elif isinstance(t, TFun):
        for p in t.params:
            free_rigids(p, out)
        free_rigids(t.result, out)
    elif isinstance(t, TRec):
        for v in t.fields.values():
            free_rigids(v, out)
    return out


def occurs(v: TMeta, t: Type) -> bool:
    t = prune(t)
    if t is v:
        return True
    if isinstance(t, TCon):
        return any(occurs(v, a) for a in t.args)
    if isinstance(t, TFun):
        return any(occurs(v, p) for p in t.params) or occurs(v, t.result)
    if isinstance(t, TRec):
        return any(occurs(v, x) for x in t.fields.values())
    return False


# --------------------------------------------------------------------------
# Unification
# --------------------------------------------------------------------------

class Mismatch(Exception):
    """Raised on unification failure. Carries the innermost conflicting pair."""

    def __init__(self, expected, actual, detail: str = ""):
        self.expected = expected
        self.actual = actual
        self.detail = detail
        super().__init__(detail or f"expected {show(expected)}, found {show(actual)}")


def unify(a: Type, b: Type):
    """
    Make two types equal, binding unification variables as needed.

    `Never` unifies with anything: it is the result type of `abort` and of the
    diverging branch of a `?`, so it must not force those positions to agree
    with each other.
    """
    a, b = prune(a), prune(b)

    if a is b:
        return

    if isinstance(a, TCon) and a.name == "Never":
        return
    if isinstance(b, TCon) and b.name == "Never":
        return

    if isinstance(a, TMeta):
        if occurs(a, b):
            raise Mismatch(a, b, "infinite type")
        a.ref = b
        return
    if isinstance(b, TMeta):
        if occurs(b, a):
            raise Mismatch(b, a, "infinite type")
        b.ref = a
        return

    if isinstance(a, TRigid) and isinstance(b, TRigid):
        if a.name != b.name:
            raise Mismatch(a, b)
        return

    if isinstance(a, TCon) and isinstance(b, TCon):
        if a.name != b.name:
            raise Mismatch(a, b)
        if len(a.args) != len(b.args):
            raise Mismatch(a, b, "different number of type arguments")
        for x, y in zip(a.args, b.args):
            unify(x, y)
        return

    if isinstance(a, TFun) and isinstance(b, TFun):
        if len(a.params) != len(b.params):
            raise Mismatch(a, b, "different number of parameters")
        for x, y in zip(a.params, b.params):
            unify(x, y)
        unify(a.result, b.result)
        return

    if isinstance(a, TRec) and isinstance(b, TRec):
        if set(a.fields) != set(b.fields):
            missing = sorted(set(a.fields) ^ set(b.fields))
            raise Mismatch(a, b, "different fields: " + ", ".join(missing))
        for k in a.fields:
            unify(a.fields[k], b.fields[k])
        return

    raise Mismatch(a, b)


def zonk(t: Type) -> Type:
    """Resolve a type fully, for reporting. Unbound variables become `_`."""
    t = prune(t)
    if isinstance(t, TMeta):
        return TRigid("_")
    if isinstance(t, TCon):
        return TCon(t.name, [zonk(a) for a in t.args]) if t.args else t
    if isinstance(t, TFun):
        return TFun([zonk(p) for p in t.params], zonk(t.result), t.effects)
    if isinstance(t, TRec):
        return TRec({k: zonk(v) for k, v in t.fields.items()})
    return t


# --------------------------------------------------------------------------
# Display
# --------------------------------------------------------------------------

def show(t) -> str:
    t = prune(t) if isinstance(t, Type) else t
    if t is None:
        return "Unit"
    if isinstance(t, TCon):
        if t.args:
            return f"{t.name}<{', '.join(show(a) for a in t.args)}>"
        return t.name
    if isinstance(t, TRigid):
        return t.name
    if isinstance(t, TMeta):
        return f"?{t.origin or t.id}"
    if isinstance(t, TFun):
        s = f"Fn({', '.join(show(p) for p in t.params)}) -> {show(t.result)}"
        if t.effects:
            s += " uses " + ", ".join(sorted(t.effects))
        return s
    if isinstance(t, TRec):
        return "{ " + ", ".join(f"{k}: {show(v)}"
                                for k, v in sorted(t.fields.items())) + " }"
    return str(t)


# --------------------------------------------------------------------------
# Effect sets
# --------------------------------------------------------------------------

class EffectSet:
    """
    A set of effect operation keys, each `effect.op`, plus wildcards `effect.*`.

    Wildcards only appear in declarations and grants, never in inferred sets,
    so an inferred set is always a set of concrete operations. That is what
    makes the capability footprint of a call graph exact rather than an
    over-approximation.
    """

    __slots__ = ("ops", "wild")

    def __init__(self, ops=(), wild=()):
        self.ops = frozenset(ops)
        self.wild = frozenset(wild)

    @staticmethod
    def from_refs(refs) -> "EffectSet":
        ops, wild = set(), set()
        for r in refs:
            if r.op is None:
                wild.add(r.effect)
            else:
                ops.add(f"{r.effect}.{r.op}")
        return EffectSet(ops, wild)

    @staticmethod
    def from_keys(keys) -> "EffectSet":
        ops, wild = set(), set()
        for k in keys:
            if k.endswith(".*"):
                wild.add(k[:-2])
            else:
                ops.add(k)
        return EffectSet(ops, wild)

    def covers(self, key: str) -> bool:
        if key in self.ops:
            return True
        eff = key.split(".", 1)[0]
        return eff in self.wild

    def union(self, other: "EffectSet") -> "EffectSet":
        return EffectSet(self.ops | other.ops, self.wild | other.wild)

    def missing(self, keys) -> list:
        return sorted(k for k in keys if not self.covers(k))

    def unused(self, used) -> list:
        """Declared operations that nothing in the body actually performs."""
        used = set(used)
        out = [o for o in self.ops if o not in used]
        for w in self.wild:
            if not any(u.split(".", 1)[0] == w for u in used):
                out.append(w + ".*")
        return sorted(out)

    def keys(self) -> list:
        return sorted(list(self.ops) + [w + ".*" for w in self.wild])

    def is_empty(self) -> bool:
        return not self.ops and not self.wild

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.ops) + len(self.wild)

    def __contains__(self, key):
        return self.covers(key)

    def __str__(self):
        return ", ".join(self.keys()) if len(self) else "(pure)"

    def __repr__(self):
        return f"EffectSet({self})"

    def __eq__(self, other):
        return (isinstance(other, EffectSet) and other.ops == self.ops
                and other.wild == self.wild)

    def __hash__(self):
        return hash((self.ops, self.wild))


EMPTY_EFFECTS = EffectSet()


# --------------------------------------------------------------------------
# Data classification lattice
#
# Used by Rune policy and by Weft schemas. A capability grant can be
# constrained by classification, so a definition that touches `restricted`
# data cannot be called from a context only cleared for `internal`.
# --------------------------------------------------------------------------

CLASSIFICATIONS = ["public", "internal", "confidential", "pseudonymous",
                   "personal", "sensitive", "restricted"]

CLASS_RANK = {c: i for i, c in enumerate(CLASSIFICATIONS)}


def class_rank(c: str) -> int:
    return CLASS_RANK.get(c, CLASS_RANK["internal"])


def max_class(a: str, b: str) -> str:
    return a if class_rank(a) >= class_rank(b) else b
