"""
The Canon prelude: builtin types, builtin effects, and the standard library.

The builtin effects are written in Canon itself and parsed at import time, so
they are described by exactly the same declarations a user would write. There
is no privileged syntax for them.

Standard library functions are declared with their Canon signature alongside a
host implementation. Every one of them is total: no library call can raise, so
anything that can fail returns `Option` or `Result`. That is what lets the
checker treat `?` and `match` as the only two ways control flow leaves a
function, which in turn is what makes the effect and cost analysis exact.
"""

from __future__ import annotations

import hashlib
import unicodedata
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
from typing import Optional

from . import ast as A
from . import types as TY
from . import values as V
from .lexer import Lexer
from .parser import Parser


# --------------------------------------------------------------------------
# Builtin effects, declared in Canon
# --------------------------------------------------------------------------

PRELUDE_EFFECTS_SOURCE = r'''
module prelude

--- Model invocation. Performed by the `ask` expression rather than called
--- directly, so that every model call carries a result type and contracts.
effect model {
  --- Generate a response. The runtime fills in prompt assembly from the
  --- `ask` body; this is the operation that is journaled and budgeted.
  infer(prompt: Text, model: Text, max_tokens: Int, temperature: Dec) -> Text
  --- Embed text into a vector.
  idempotent embed(text: Text, model: Text) -> List<Dec>
  --- Ask a model to adjudicate a claim against evidence. Used by `judge`
  --- clauses and by grounding checks.
  judge(claim: Text, evidence: Text, model: Text) -> Bool
}

--- Reading the clock. Time is an effect because a function that reads the
--- clock is not reproducible unless the reading is journaled.
effect time {
  now() -> Time
}

--- Nondeterministic sources. Same reasoning as `time`.
effect random {
  int(lo: Int, hi: Int) -> Int
  uuid() -> Text
  bytes(n: Int) -> Bytes
}

--- Structured logging. Separate from `audit`, which is not discardable.
effect log {
  info(message: Text, data: Text) -> Unit
  warn(message: Text, data: Text) -> Unit
  error(message: Text, data: Text) -> Unit
}

--- Append-only audit trail. Distinct from `log` because entries are part of
--- the compliance record and cannot be dropped or sampled.
effect audit {
  entry(action: Text, subject: Text, detail: Text) -> Unit
}
'''


def load_prelude_effects():
    lx = Lexer(PRELUDE_EFFECTS_SOURCE, "<prelude>")
    toks = lx.run()
    p = Parser(toks, PRELUDE_EFFECTS_SOURCE, "<prelude>", lx.bag)
    mod = p.parse_module()
    if p.bag.has_errors:
        raise RuntimeError("prelude failed to parse:\n"
                           + p.bag.render(PRELUDE_EFFECTS_SOURCE))
    return {e.name: e for e in mod.effects()}


BUILTIN_EFFECTS = load_prelude_effects()


# --------------------------------------------------------------------------
# Type expression to Type
# --------------------------------------------------------------------------

def parse_type_string(s: str) -> A.TypeExpr:
    lx = Lexer(s, "<signature>")
    toks = lx.run()
    p = Parser(toks, s, "<signature>", lx.bag)
    t = p.parse_type()
    if p.bag.has_errors:
        raise RuntimeError(f"bad builtin signature {s!r}:\n"
                           + p.bag.render(s))
    return t


def to_type(texpr, rigids=(), known=None, on_unknown=None) -> TY.Type:
    """
    Convert a parsed type expression into a checker Type.

    `rigids` names that should become rigid type variables.
    `known` maps user type names to their arity; unknown names are reported
    through `on_unknown` and become `Never` so checking can continue.
    """
    rigids = set(rigids)
    known = known if known is not None else {}

    def go(t):
        if t is None:
            return TY.UNIT
        if isinstance(t, A.TName):
            name = t.name
            if name in rigids and not t.args:
                return TY.TRigid(name)
            if name in TY.PRIMITIVES and not t.args:
                return TY.PRIMITIVES[name]
            arity = TY.GENERICS.get(name, known.get(name))
            if arity is None:
                if on_unknown:
                    on_unknown(t)
                return TY.NEVER
            args = [go(a) for a in t.args]
            if len(args) != arity:
                if on_unknown:
                    on_unknown(t, arity)
                while len(args) < arity:
                    args.append(TY.fresh())
                args = args[:arity]
            return TY.TCon(name, args)
        if isinstance(t, A.TVar):
            return TY.TRigid(t.name)
        if isinstance(t, A.TFn):
            return TY.TFun([go(p) for p in t.params], go(t.result),
                           frozenset(e.key() for e in t.effects))
        if isinstance(t, A.TRecord):
            return TY.TRec({n: go(ft) for n, ft in t.fields})
        return TY.NEVER

    return go(texpr)


# --------------------------------------------------------------------------
# Standard library
# --------------------------------------------------------------------------

class Builtin:
    __slots__ = ("name", "sig_text", "tparams", "impl", "needs_apply",
                 "effects", "doc", "_type")

    def __init__(self, name, sig_text, impl, tparams=(), needs_apply=False,
                 effects=(), doc=""):
        self.name = name
        self.sig_text = sig_text
        self.tparams = list(tparams)
        self.impl = impl
        self.needs_apply = needs_apply
        self.effects = frozenset(effects)
        self.doc = doc
        self._type = None

    @property
    def type(self) -> TY.TFun:
        if self._type is None:
            te = parse_type_string(self.sig_text)
            # Prelude record types (currently just Pair) must be resolvable
            # from a builtin signature.
            known = {n: len(d.type_params) for n, d in BUILTIN_TYPES.items()}
            self._type = to_type(te, rigids=self.tparams, known=known)
        return self._type

    @property
    def arity(self) -> int:
        return len(self.type.params)

    def __repr__(self):
        return f"<builtin {self.name}: {self.sig_text}>"


BUILTINS: dict = {}


def bi(name, sig, tparams=(), needs_apply=False, effects=(), doc=""):
    def deco(fn):
        BUILTINS[name] = Builtin(name, sig, fn, tparams, needs_apply,
                                 effects, doc)
        return fn
    return deco


# -- control ---------------------------------------------------------------

class Abort(Exception):
    """Raised by `abort`. Caught at the call boundary and turned into a fault."""

    def __init__(self, message):
        self.message = message
        super().__init__(message)


@bi("abort", "Fn(Text) -> Never",
    doc="Terminate with a fault. Unreachable in a verified program.")
def _abort(msg):
    raise Abort(msg)


@bi("old", "Fn(T) -> T", tparams=["T"],
    doc="In an `ensures` clause, the value an expression had on entry.")
def _old(v):
    return v


# -- Int -------------------------------------------------------------------

@bi("Int.abs", "Fn(Int) -> Int")
def _int_abs(a):
    return abs(a)


@bi("Int.min", "Fn(Int, Int) -> Int")
def _int_min(a, b):
    return a if a <= b else b


@bi("Int.max", "Fn(Int, Int) -> Int")
def _int_max(a, b):
    return a if a >= b else b


@bi("Int.clamp", "Fn(Int, Int, Int) -> Int",
    doc="clamp(value, lo, hi). If lo > hi the result is lo.")
def _int_clamp(v, lo, hi):
    if lo > hi:
        return lo
    return lo if v < lo else (hi if v > hi else v)


@bi("Int.to_dec", "Fn(Int) -> Dec")
def _int_to_dec(a):
    return Decimal(a)


@bi("Int.to_text", "Fn(Int) -> Text")
def _int_to_text(a):
    return str(a)


@bi("Int.pow", "Fn(Int, Int) -> Int",
    doc="Negative exponents give 0. The exponent is capped to keep the "
        "operation within the step budget.")
def _int_pow(a, b):
    if b < 0:
        return 0
    if b > 4096:
        b = 4096
    return a ** b


@bi("Int.div", "Fn(Int, Int) -> Option<Int>",
    doc="Truncating division. None when the divisor is zero.")
def _int_div(a, b):
    if b == 0:
        return V.NONE
    q = abs(a) // abs(b)
    return V.some(-q if (a < 0) != (b < 0) else q)


@bi("Int.rem", "Fn(Int, Int) -> Option<Int>")
def _int_rem(a, b):
    if b == 0:
        return V.NONE
    r = abs(a) % abs(b)
    return V.some(-r if a < 0 else r)


@bi("Int.parse", "Fn(Text) -> Option<Int>")
def _int_parse(s):
    t = s.strip()
    try:
        return V.some(int(t, 10))
    except ValueError:
        return V.NONE


# -- Dec -------------------------------------------------------------------

@bi("Dec.abs", "Fn(Dec) -> Dec")
def _dec_abs(a):
    return abs(a)


@bi("Dec.min", "Fn(Dec, Dec) -> Dec")
def _dec_min(a, b):
    return a if a <= b else b


@bi("Dec.max", "Fn(Dec, Dec) -> Dec")
def _dec_max(a, b):
    return a if a >= b else b


@bi("Dec.round", "Fn(Dec, Int) -> Dec",
    doc="Round to `places` using banker's rounding.")
def _dec_round(a, places):
    places = max(-28, min(28, places))
    q = Decimal(1).scaleb(-places)
    return a.quantize(q, rounding=ROUND_HALF_EVEN)


@bi("Dec.floor", "Fn(Dec) -> Int")
def _dec_floor(a):
    return int(a.to_integral_value(rounding="ROUND_FLOOR"))


@bi("Dec.ceil", "Fn(Dec) -> Int")
def _dec_ceil(a):
    return int(a.to_integral_value(rounding="ROUND_CEILING"))


@bi("Dec.div", "Fn(Dec, Dec) -> Option<Dec>")
def _dec_div(a, b):
    if b == 0:
        return V.NONE
    try:
        return V.some(a / b)
    except InvalidOperation:
        return V.NONE


@bi("Dec.to_text", "Fn(Dec) -> Text")
def _dec_to_text(a):
    return format(a.normalize(), "f")


@bi("Dec.parse", "Fn(Text) -> Option<Dec>")
def _dec_parse(s):
    try:
        return V.some(Decimal(s.strip()))
    except InvalidOperation:
        return V.NONE


# -- Text ------------------------------------------------------------------

@bi("Text.length", "Fn(Text) -> Int",
    doc="Length in Unicode scalar values, not bytes.")
def _text_length(s):
    return len(s)


@bi("Text.is_empty", "Fn(Text) -> Bool")
def _text_is_empty(s):
    return len(s) == 0


@bi("Text.concat", "Fn(Text, Text) -> Text")
def _text_concat(a, b):
    return a + b


@bi("Text.join", "Fn(List<Text>, Text) -> Text")
def _text_join(xs, sep):
    return sep.join(xs)


@bi("Text.split", "Fn(Text, Text) -> List<Text>")
def _text_split(s, sep):
    if sep == "":
        return tuple(s)
    return tuple(s.split(sep))


@bi("Text.trim", "Fn(Text) -> Text")
def _text_trim(s):
    return s.strip()


@bi("Text.upper", "Fn(Text) -> Text")
def _text_upper(s):
    return s.upper()


@bi("Text.lower", "Fn(Text) -> Text")
def _text_lower(s):
    return s.lower()


@bi("Text.contains", "Fn(Text, Text) -> Bool")
def _text_contains(s, sub):
    return sub in s


@bi("Text.starts_with", "Fn(Text, Text) -> Bool")
def _text_starts(s, p):
    return s.startswith(p)


@bi("Text.ends_with", "Fn(Text, Text) -> Bool")
def _text_ends(s, p):
    return s.endswith(p)


@bi("Text.replace", "Fn(Text, Text, Text) -> Text")
def _text_replace(s, a, b):
    return s if a == "" else s.replace(a, b)


@bi("Text.slice", "Fn(Text, Int, Int) -> Text",
    doc="Half-open range, clamped to the string. Never fails.")
def _text_slice(s, start, end):
    n = len(s)
    start = max(0, min(n, start))
    end = max(start, min(n, end))
    return s[start:end]


@bi("Text.pad_left", "Fn(Text, Int, Text) -> Text")
def _text_pad_left(s, width, ch):
    if not ch:
        ch = " "
    ch = ch[0]
    return s if len(s) >= width else ch * (width - len(s)) + s


@bi("Text.normalize", "Fn(Text) -> Text",
    doc="Unicode NFC. Applied before comparison in security-sensitive code.")
def _text_normalize(s):
    return unicodedata.normalize("NFC", s)


@bi("Text.hash", "Fn(Text) -> Text",
    doc="A stable, non-cryptographic content hash. Same input, same output, "
        "across runs and machines.")
def _text_hash(s):
    return hashlib.blake2b(s.encode("utf-8"), digest_size=16).hexdigest()


# -- List ------------------------------------------------------------------

@bi("List.length", "Fn(List<T>) -> Int", tparams=["T"])
def _list_length(xs):
    return len(xs)


@bi("List.is_empty", "Fn(List<T>) -> Bool", tparams=["T"])
def _list_is_empty(xs):
    return len(xs) == 0


@bi("List.get", "Fn(List<T>, Int) -> Option<T>", tparams=["T"],
    doc="Bounds-checked. Out of range gives None rather than a fault.")
def _list_get(xs, i):
    if 0 <= i < len(xs):
        return V.some(xs[i])
    return V.NONE


@bi("List.first", "Fn(List<T>) -> Option<T>", tparams=["T"])
def _list_first(xs):
    return V.some(xs[0]) if xs else V.NONE


@bi("List.last", "Fn(List<T>) -> Option<T>", tparams=["T"])
def _list_last(xs):
    return V.some(xs[-1]) if xs else V.NONE


@bi("List.append", "Fn(List<T>, T) -> List<T>", tparams=["T"])
def _list_append(xs, x):
    return xs + (x,)


@bi("List.prepend", "Fn(List<T>, T) -> List<T>", tparams=["T"])
def _list_prepend(xs, x):
    return (x,) + xs


@bi("List.concat", "Fn(List<T>, List<T>) -> List<T>", tparams=["T"])
def _list_concat(a, b):
    return a + b


@bi("List.reverse", "Fn(List<T>) -> List<T>", tparams=["T"])
def _list_reverse(xs):
    return tuple(reversed(xs))


@bi("List.take", "Fn(List<T>, Int) -> List<T>", tparams=["T"])
def _list_take(xs, n):
    return xs[:max(0, n)]


@bi("List.drop", "Fn(List<T>, Int) -> List<T>", tparams=["T"])
def _list_drop(xs, n):
    return xs[max(0, n):]


@bi("List.contains", "Fn(List<T>, T) -> Bool", tparams=["T"])
def _list_contains(xs, x):
    return any(V.compare(i, x) == 0 for i in xs)


@bi("List.index_of", "Fn(List<T>, T) -> Option<Int>", tparams=["T"])
def _list_index_of(xs, x):
    for i, item in enumerate(xs):
        if V.compare(item, x) == 0:
            return V.some(i)
    return V.NONE


@bi("List.unique", "Fn(List<T>) -> List<T>", tparams=["T"],
    doc="Preserves first-occurrence order.")
def _list_unique(xs):
    seen = []
    out = []
    for x in xs:
        k = V._sort_key(x)
        if k not in seen:
            seen.append(k)
            out.append(x)
    return tuple(out)


@bi("List.sort", "Fn(List<T>) -> List<T>", tparams=["T"],
    doc="Sorts by the language's total value order, so the result is stable "
        "across runs and machines.")
def _list_sort(xs):
    return tuple(sorted(xs, key=V._sort_key))


@bi("List.sum", "Fn(List<Int>) -> Int")
def _list_sum(xs):
    return sum(xs)


@bi("List.sum_dec", "Fn(List<Dec>) -> Dec")
def _list_sum_dec(xs):
    total = Decimal(0)
    for x in xs:
        total += x
    return total


@bi("List.range", "Fn(Int, Int) -> List<Int>",
    doc="Half-open [lo, hi). Capped at 1,000,000 elements so a generated "
        "program cannot allocate without bound.")
def _list_range(lo, hi):
    if hi <= lo:
        return ()
    n = min(hi - lo, 1_000_000)
    return tuple(range(lo, lo + n))


@bi("List.map", "Fn(List<A>, Fn(A) -> B) -> List<B>", tparams=["A", "B"],
    needs_apply=True)
def _list_map(apply, xs, f):
    return tuple(apply(f, [x]) for x in xs)


@bi("List.filter", "Fn(List<T>, Fn(T) -> Bool) -> List<T>", tparams=["T"],
    needs_apply=True)
def _list_filter(apply, xs, f):
    return tuple(x for x in xs if apply(f, [x]) is True)


@bi("List.fold", "Fn(List<T>, A, Fn(A, T) -> A) -> A", tparams=["T", "A"],
    needs_apply=True,
    doc="The only general iteration construct. Bounded by the list length, "
        "which is what keeps every Canon function total.")
def _list_fold(apply, xs, init, f):
    acc = init
    for x in xs:
        acc = apply(f, [acc, x])
    return acc


@bi("List.flat_map", "Fn(List<A>, Fn(A) -> List<B>) -> List<B>",
    tparams=["A", "B"], needs_apply=True)
def _list_flat_map(apply, xs, f):
    out = []
    for x in xs:
        out.extend(apply(f, [x]))
    return tuple(out)


@bi("List.all", "Fn(List<T>, Fn(T) -> Bool) -> Bool", tparams=["T"],
    needs_apply=True)
def _list_all(apply, xs, f):
    return all(apply(f, [x]) is True for x in xs)


@bi("List.any", "Fn(List<T>, Fn(T) -> Bool) -> Bool", tparams=["T"],
    needs_apply=True)
def _list_any(apply, xs, f):
    return any(apply(f, [x]) is True for x in xs)


@bi("List.count", "Fn(List<T>, Fn(T) -> Bool) -> Int", tparams=["T"],
    needs_apply=True)
def _list_count(apply, xs, f):
    return sum(1 for x in xs if apply(f, [x]) is True)


@bi("List.find", "Fn(List<T>, Fn(T) -> Bool) -> Option<T>", tparams=["T"],
    needs_apply=True)
def _list_find(apply, xs, f):
    for x in xs:
        if apply(f, [x]) is True:
            return V.some(x)
    return V.NONE


@bi("List.sort_by", "Fn(List<T>, Fn(T) -> K) -> List<T>", tparams=["T", "K"],
    needs_apply=True, doc="Stable sort by a derived key.")
def _list_sort_by(apply, xs, f):
    decorated = [(V._sort_key(apply(f, [x])), i, x) for i, x in enumerate(xs)]
    decorated.sort(key=lambda t: (t[0], t[1]))
    return tuple(t[2] for t in decorated)


@bi("List.zip", "Fn(List<A>, List<B>) -> List<Pair<A, B>>",
    tparams=["A", "B"],
    doc="Truncates to the shorter input.")
def _list_zip(a, b):
    return tuple(V.Record("Pair", (("first", x), ("second", y)))
                 for x, y in zip(a, b))


# -- Map -------------------------------------------------------------------

@bi("Map.empty", "Fn() -> Map<K, V>", tparams=["K", "V"])
def _map_empty():
    return V.FrozenMap()


@bi("Map.get", "Fn(Map<K, V>, K) -> Option<V>", tparams=["K", "V"])
def _map_get(m, k):
    key = _mapkey(k)
    return V.some(m.get(key)) if m.has(key) else V.NONE


@bi("Map.insert", "Fn(Map<K, V>, K, V) -> Map<K, V>", tparams=["K", "V"])
def _map_insert(m, k, v):
    return m.insert(_mapkey(k), v)


@bi("Map.remove", "Fn(Map<K, V>, K) -> Map<K, V>", tparams=["K", "V"])
def _map_remove(m, k):
    return m.remove(_mapkey(k))


@bi("Map.has", "Fn(Map<K, V>, K) -> Bool", tparams=["K", "V"])
def _map_has(m, k):
    return m.has(_mapkey(k))


@bi("Map.size", "Fn(Map<K, V>) -> Int", tparams=["K", "V"])
def _map_size(m):
    return len(m)


@bi("Map.keys", "Fn(Map<K, V>) -> List<K>", tparams=["K", "V"])
def _map_keys(m):
    return m.keys()


@bi("Map.values", "Fn(Map<K, V>) -> List<V>", tparams=["K", "V"])
def _map_values(m):
    return m.values()


def _mapkey(k):
    """Map keys are normalised so structurally equal values collide."""
    if isinstance(k, (str, int, bool)):
        return k
    return V.value_hash(k)


# -- Option / Result -------------------------------------------------------

@bi("Option.is_some", "Fn(Option<T>) -> Bool", tparams=["T"])
def _opt_is_some(o):
    return isinstance(o, V.Variant) and o.ctor == "Some"


@bi("Option.is_none", "Fn(Option<T>) -> Bool", tparams=["T"])
def _opt_is_none(o):
    return isinstance(o, V.Variant) and o.ctor == "None"


@bi("Option.unwrap_or", "Fn(Option<T>, T) -> T", tparams=["T"])
def _opt_unwrap_or(o, d):
    return o.args[0] if isinstance(o, V.Variant) and o.ctor == "Some" else d


@bi("Option.map", "Fn(Option<A>, Fn(A) -> B) -> Option<B>",
    tparams=["A", "B"], needs_apply=True)
def _opt_map(apply, o, f):
    if isinstance(o, V.Variant) and o.ctor == "Some":
        return V.some(apply(f, [o.args[0]]))
    return V.NONE


@bi("Option.to_result", "Fn(Option<T>, E) -> Result<T, E>", tparams=["T", "E"])
def _opt_to_result(o, e):
    if isinstance(o, V.Variant) and o.ctor == "Some":
        return V.ok(o.args[0])
    return V.err(e)


@bi("Result.is_ok", "Fn(Result<T, E>) -> Bool", tparams=["T", "E"])
def _res_is_ok(r):
    return V.is_ok(r)


@bi("Result.is_err", "Fn(Result<T, E>) -> Bool", tparams=["T", "E"])
def _res_is_err(r):
    return V.is_err(r)


@bi("Result.unwrap_or", "Fn(Result<T, E>, T) -> T", tparams=["T", "E"])
def _res_unwrap_or(r, d):
    return r.args[0] if V.is_ok(r) else d


@bi("Result.map", "Fn(Result<A, E>, Fn(A) -> B) -> Result<B, E>",
    tparams=["A", "B", "E"], needs_apply=True)
def _res_map(apply, r, f):
    return V.ok(apply(f, [r.args[0]])) if V.is_ok(r) else r


@bi("Result.map_err", "Fn(Result<T, A>, Fn(A) -> B) -> Result<T, B>",
    tparams=["T", "A", "B"], needs_apply=True)
def _res_map_err(apply, r, f):
    return r if V.is_ok(r) else V.err(apply(f, [r.args[0]]))


@bi("Result.to_option", "Fn(Result<T, E>) -> Option<T>", tparams=["T", "E"])
def _res_to_option(r):
    return V.some(r.args[0]) if V.is_ok(r) else V.NONE


# -- Set -------------------------------------------------------------------

@bi("Set.empty", "Fn() -> Set<T>", tparams=["T"])
def _set_empty():
    return frozenset()


@bi("Set.of", "Fn(List<T>) -> Set<T>", tparams=["T"])
def _set_of(xs):
    return frozenset(V._hashable(x) for x in xs)


@bi("Set.has", "Fn(Set<T>, T) -> Bool", tparams=["T"])
def _set_has(s, x):
    return V._hashable(x) in s


@bi("Set.add", "Fn(Set<T>, T) -> Set<T>", tparams=["T"])
def _set_add(s, x):
    return s | {V._hashable(x)}


@bi("Set.size", "Fn(Set<T>) -> Int", tparams=["T"])
def _set_size(s):
    return len(s)


@bi("Set.to_list", "Fn(Set<T>) -> List<T>", tparams=["T"])
def _set_to_list(s):
    return tuple(sorted(s, key=V._sort_key))


# -- Time ------------------------------------------------------------------

@bi("Time.epoch_millis", "Fn(Time) -> Int")
def _time_epoch(t):
    return t.epoch_millis


@bi("Time.before", "Fn(Time, Time) -> Bool")
def _time_before(a, b):
    return a.epoch_millis < b.epoch_millis


@bi("Time.plus_millis", "Fn(Time, Int) -> Time")
def _time_plus(t, ms):
    return V.Instant(t.epoch_millis + ms, t.logical)


# --------------------------------------------------------------------------
# The Pair record, used by List.zip
# --------------------------------------------------------------------------

PRELUDE_TYPES_SOURCE = r'''
module prelude

record Pair<A, B> {
  first: A
  second: B
}
'''


def load_prelude_types():
    lx = Lexer(PRELUDE_TYPES_SOURCE, "<prelude>")
    toks = lx.run()
    p = Parser(toks, PRELUDE_TYPES_SOURCE, "<prelude>", lx.bag)
    mod = p.parse_module()
    if p.bag.has_errors:
        raise RuntimeError("prelude types failed to parse:\n"
                           + p.bag.render(PRELUDE_TYPES_SOURCE))
    return {d.name: d for d in mod.types()}


BUILTIN_TYPES = load_prelude_types()


# Names the canonical encoder must treat as builtins rather than dependencies.
def _register_builtin_names():
    from . import canonical
    canonical.BUILTIN_NAMES.update(BUILTINS.keys())
    canonical.BUILTIN_NAMES.update(["abort", "old", "result"])


_register_builtin_names()


def builtin_names() -> list:
    return sorted(BUILTINS.keys())


def lookup(name: str) -> Optional[Builtin]:
    return BUILTINS.get(name)
