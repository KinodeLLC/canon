"""
Canonical form and content addressing.

Two jobs:

1. `encode` turns a declaration into a canonical byte string. The encoding is
   invariant under things that do not change meaning: local variable names,
   the order of independent contract clauses, the spelling of operators that
   have synonyms, whitespace, and comments. Two declarations that mean the
   same thing encode identically.

2. `Hasher` turns that encoding into an identifier. Each definition gets two:

     local hash   covers the definition's own body, with dependencies
                  referenced by name. Changes only when this definition is
                  edited.
     deep hash    covers the body with dependencies referenced by *their* deep
                  hashes. Changes when this definition or anything it
                  transitively depends on is edited.

   The deep hash is the definition's identity. It is what verification results,
   cached evaluations, journal entries and audit records are keyed by, so a
   result can never be attributed to the wrong version of the code.

Local variables are encoded by binding depth rather than name, so renaming a
parameter or a let-binding produces the same hash. That makes a rename a
zero-risk edit: nothing downstream is invalidated, no re-verification is
needed, and no reviewer has to look at it.

`Printer` renders a declaration back to source text. Because the AST it prints
from is already canonical, formatting is not a matter of preference and there
is no configuration.
"""

from __future__ import annotations

import base64
import hashlib
from decimal import Decimal
from typing import Optional

from . import ast as A

# Bump when the encoding below changes in any way. Every stored hash becomes
# invalid at that point, so this is always a breaking change.
IR_VERSION = 1

HASH_PREFIX = "#"
HASH_CHARS = 26          # base32 of a 256-bit digest, truncated
BUILTIN_NAMES = set()    # filled in by stdlib at import time


# --------------------------------------------------------------------------
# Hash formatting
# --------------------------------------------------------------------------

def digest(data: bytes) -> str:
    h = hashlib.blake2b(data, digest_size=32).digest()
    b32 = base64.b32encode(h).decode("ascii").rstrip("=").lower()
    return HASH_PREFIX + b32[:HASH_CHARS]


def short(h: str) -> str:
    """An 8-character abbreviation, for display only."""
    return h[:9] if h.startswith(HASH_PREFIX) else h[:8]


# --------------------------------------------------------------------------
# Canonical encoder
# --------------------------------------------------------------------------

class Encoder:
    """
    Produces the canonical S-expression encoding of a declaration.

    `resolve` maps a global name to its deep hash. When it is None, or returns
    None for a name, the name is encoded literally, which yields the local
    hash.
    """

    def __init__(self, resolve=None, module: str = ""):
        self.resolve = resolve
        self.module = module
        self.deps: set = set()          # global names referenced
        self.effects: set = set()       # effect op keys performed
        self.models: set = set()        # model names invoked
        self.scope: list = []           # local binder names, innermost last

    # -- scope helpers ---------------------------------------------------

    def bind(self, name: str):
        self.scope.append(name)

    def unbind(self, n: int = 1):
        for _ in range(n):
            if self.scope:
                self.scope.pop()

    def lookup(self, name: str) -> Optional[int]:
        """De Bruijn index from the innermost binder, or None if not local."""
        for i in range(len(self.scope) - 1, -1, -1):
            if self.scope[i] == name:
                return len(self.scope) - 1 - i
        return None

    def gref(self, name: str) -> str:
        """Encode a reference to a global definition."""
        self.deps.add(name)
        if self.resolve is not None:
            h = self.resolve(name)
            if h:
                return f"(h {h})"
        return f"(g {_atom(name)})"

    # -- entry point -----------------------------------------------------

    def decl(self, d) -> str:
        if isinstance(d, A.FnDecl):
            return self.fn(d)
        if isinstance(d, A.RecordDecl):
            return self.record(d)
        if isinstance(d, A.EnumDecl):
            return self.enum(d)
        if isinstance(d, A.AliasDecl):
            return self.alias(d)
        if isinstance(d, A.ConstDecl):
            return self.const(d)
        if isinstance(d, A.EffectDecl):
            return self.effect(d)
        if isinstance(d, A.TestDecl):
            return f"(test {_atom(d.name)} {self.expr(d.body)})"
        raise TypeError(f"cannot encode {type(d).__name__}")

    # -- declarations ----------------------------------------------------

    def fn(self, d: A.FnDecl) -> str:
        parts = ["fn", _atom(d.name)]
        if d.type_params:
            parts.append("(tp " + " ".join(_atom(t) for t in d.type_params) + ")")

        # Parameters bind in order; their names are erased from the encoding
        # but their types and order are kept.
        parts.append("(p " + " ".join(self.ty(p.ty) for p in d.params) + ")")
        for p in d.params:
            self.bind(p.name)

        parts.append("(r " + self.ty(d.result) + ")")

        # Clause sets are unordered, so they are sorted by their encoding.
        if d.uses:
            keys = sorted({e.key() for e in d.uses})
            parts.append("(u " + " ".join(_atom(k) for k in keys) + ")")

        if d.requires:
            enc = sorted(self.expr(e) for e in d.requires)
            parts.append("(rq " + " ".join(enc) + ")")

        if d.ensures:
            # `result` is in scope inside ensures clauses.
            self.bind("result")
            enc = sorted(self.expr(e) for e in d.ensures)
            self.unbind()
            parts.append("(en " + " ".join(enc) + ")")

        if d.laws:
            enc = sorted(
                "(" + _atom(l.name) + "".join(" " + self.expr(a) for a in l.args) + ")"
                for l in d.laws
            )
            parts.append("(lw " + " ".join(enc) + ")")

        if d.cost:
            parts.append(self.cost(d.cost))

        if d.decreases:
            parts.append("(dec " + self.expr(d.decreases) + ")")

        # `intent` is part of the hash: the stated purpose is part of the
        # definition's identity, so changing the intent forces re-verification
        # and a new audit record even if the body is untouched.
        if d.intent:
            parts.append("(i " + _atom(d.intent) + ")")

        parts.append("(b " + self.expr(d.body) + ")")
        self.unbind(len(d.params))
        return "(" + " ".join(parts) + ")"

    def record(self, d: A.RecordDecl) -> str:
        parts = ["rec", _atom(d.name)]
        if d.type_params:
            parts.append("(tp " + " ".join(_atom(t) for t in d.type_params) + ")")
        # A field's default is part of the record's meaning -- it decides what
        # a literal that omits the field produces -- so it is hashed.
        fields = " ".join(
            "(" + _atom(f.name) + " " + self.ty(f.ty)
            + (" (d " + self.expr(f.default) + ")" if f.default is not None
               else "")
            + ")"
            for f in d.fields
        )
        parts.append("(f " + fields + ")")
        if d.invariants:
            self.bind("self")
            enc = sorted(self.expr(e) for e in d.invariants)
            self.unbind()
            parts.append("(inv " + " ".join(enc) + ")")
        if d.classification:
            cls = " ".join(f"({_atom(k)} {_atom(v)})"
                           for k, v in sorted(d.classification.items()))
            parts.append("(cls " + cls + ")")
        if d.intent:
            parts.append("(i " + _atom(d.intent) + ")")
        return "(" + " ".join(parts) + ")"

    def enum(self, d: A.EnumDecl) -> str:
        parts = ["enum", _atom(d.name)]
        if d.type_params:
            parts.append("(tp " + " ".join(_atom(t) for t in d.type_params) + ")")
        # Variant order is preserved: it is often meaningful (severity ladders,
        # state progressions) and is cheap to keep stable.
        vs = " ".join(
            "(" + _atom(v.name) + "".join(" " + self.ty(t) for t in v.params) + ")"
            for v in d.variants
        )
        parts.append("(v " + vs + ")")
        if d.intent:
            parts.append("(i " + _atom(d.intent) + ")")
        return "(" + " ".join(parts) + ")"

    def alias(self, d: A.AliasDecl) -> str:
        parts = ["alias", _atom(d.name)]
        if d.type_params:
            parts.append("(tp " + " ".join(_atom(t) for t in d.type_params) + ")")
        parts.append(self.ty(d.target))
        return "(" + " ".join(parts) + ")"

    def const(self, d: A.ConstDecl) -> str:
        return f"(const {_atom(d.name)} {self.ty(d.ty)} {self.expr(d.value)})"

    def effect(self, d: A.EffectDecl) -> str:
        ops = " ".join(
            "(" + _atom(o.name)
            + (" idem" if o.idempotent else "")
            + " (p " + " ".join(self.ty(p.ty) for p in o.params) + ")"
            + " " + self.ty(o.result) + ")"
            for o in sorted(d.ops, key=lambda o: o.name)
        )
        return f"(eff {_atom(d.name)} ({ops}))"

    def cost(self, c: A.Cost) -> str:
        items = []
        for k in ("steps", "io", "tokens", "millis"):
            v = getattr(c, k)
            if v is not None:
                items.append(f"({k} {v})")
        if c.money is not None:
            items.append(f"(money {_dec(c.money)})")
        return "(c " + " ".join(items) + ")"

    # -- types -----------------------------------------------------------

    def ty(self, t) -> str:
        if t is None:
            return "(t Unit)"
        if isinstance(t, A.TName):
            if t.args:
                return "(t " + _atom(t.name) + " " + " ".join(
                    self.ty(a) for a in t.args) + ")"
            return "(t " + _atom(t.name) + ")"
        if isinstance(t, A.TVar):
            return "(tv " + _atom(t.name) + ")"
        if isinstance(t, A.TFn):
            eff = sorted({e.key() for e in t.effects})
            s = "(tfn (" + " ".join(self.ty(p) for p in t.params) + ") " \
                + self.ty(t.result)
            if eff:
                s += " (u " + " ".join(_atom(e) for e in eff) + ")"
            return s + ")"
        if isinstance(t, A.TRecord):
            # Structural record types are order-independent, so sort.
            fs = " ".join(f"({_atom(n)} {self.ty(ft)})"
                          for n, ft in sorted(t.fields, key=lambda x: x[0]))
            return "(trec " + fs + ")"
        raise TypeError(f"cannot encode type {type(t).__name__}")

    # -- expressions -----------------------------------------------------

    def expr(self, e) -> str:
        if e is None:
            return "(unit)"

        if isinstance(e, A.Lit):
            return self.lit(e)

        if isinstance(e, A.Var):
            idx = self.lookup(e.name)
            if idx is not None:
                return f"(l {idx})"
            if e.name in BUILTIN_NAMES:
                return f"(bi {_atom(e.name)})"
            return self.gref(e.name)

        if isinstance(e, A.QualVar):
            return self.gref(e.full())

        if isinstance(e, A.Field):
            return f"(. {self.expr(e.target)} {_atom(e.name)})"

        if isinstance(e, A.Call):
            return "(call " + self.expr(e.fn) + "".join(
                " " + self.expr(a) for a in e.args) + ")"

        if isinstance(e, A.Perform):
            key = f"{e.effect}.{e.op}"
            self.effects.add(key)
            return "(perf " + _atom(key) + "".join(
                " " + self.expr(a) for a in e.args) + ")"

        if isinstance(e, A.Lambda):
            for p in e.params:
                self.bind(p.name)
            s = ("(lam (" + " ".join(self.ty(p.ty) for p in e.params) + ") "
                 + self.expr(e.body) + ")")
            self.unbind(len(e.params))
            return s

        if isinstance(e, A.Let):
            v = self.expr(e.value)
            self.bind(e.name)
            b = self.expr(e.body)
            self.unbind()
            ann = " " + self.ty(e.ty) if e.ty else ""
            return f"(let{ann} {v} {b})"

        if isinstance(e, A.If):
            return ("(if " + self.expr(e.cond) + " " + self.expr(e.then)
                    + " " + self.expr(e.otherwise) + ")")

        if isinstance(e, A.Match):
            arms = []
            for arm in e.arms:
                n = self.bind_pattern(arm.pattern)
                g = " " + self.expr(arm.guard) if arm.guard else ""
                arms.append("(" + self.pat(arm.pattern) + g + " "
                            + self.expr(arm.body) + ")")
                self.unbind(n)
            # Arm order is semantic (first match wins), so it is preserved.
            return "(match " + self.expr(e.scrutinee) + " " + " ".join(arms) + ")"

        if isinstance(e, A.ListLit):
            return "(list" + "".join(" " + self.expr(i) for i in e.items) + ")"

        if isinstance(e, A.MapLit):
            entries = sorted(
                "(" + self.expr(k) + " " + self.expr(v) + ")"
                for k, v in e.entries)
            return "(map " + " ".join(entries) + ")"

        if isinstance(e, A.RecordLit):
            # Field order in a literal carries no meaning, so sort it.
            fs = " ".join(
                f"({_atom(n)} {self.expr(v)})"
                for n, v in sorted(e.fields, key=lambda x: x[0]))
            base = " (base " + self.expr(e.base) + ")" if e.base else ""
            self.deps.add(e.type_name)
            return f"(reclit {_atom(e.type_name)}{base} ({fs}))"

        if isinstance(e, A.CtorCall):
            self.deps.add(e.name)
            return "(ctor " + _atom(e.name) + "".join(
                " " + self.expr(a) for a in e.args) + ")"

        if isinstance(e, A.Binary):
            # Commutative operators over encoded operands are sorted so that
            # `a + b` and `b + a` share a hash.
            le, re_ = self.expr(e.left), self.expr(e.right)
            if e.op in COMMUTATIVE and le > re_:
                le, re_ = re_, le
            return f"({e.op} {le} {re_})"

        if isinstance(e, A.Unary):
            return f"(u{e.op} {self.expr(e.operand)})"

        if isinstance(e, A.Try):
            return f"(try {self.expr(e.operand)})"

        if isinstance(e, A.Block):
            return self.block(e)

        if isinstance(e, A.Ask):
            return self.ask(e)

        raise TypeError(f"cannot encode expression {type(e).__name__}")

    def block(self, b: A.Block) -> str:
        parts = []
        bound = 0
        for s in b.stmts:
            if isinstance(s, A.SLet):
                parts.append("(slet " + self.expr(s.value) + ")")
                self.bind(s.name)
                bound += 1
            elif isinstance(s, A.SExpr):
                parts.append("(sdo " + self.expr(s.value) + ")")
            elif isinstance(s, A.SAssert):
                parts.append("(sassert " + self.expr(s.cond) + " "
                             + _atom(s.message) + ")")
        parts.append("(sres " + self.expr(b.result) + ")")
        self.unbind(bound)
        return "(blk " + " ".join(parts) + ")"

    def ask(self, e: A.Ask) -> str:
        self.models.add(e.model)
        self.effects.add("model.infer")
        s = e.spec or A.AskSpec()
        parts = ["ask", self.ty(e.result_type), _atom(e.model)]
        if s.system is not None:
            parts.append("(sys " + self.expr(s.system) + ")")
        if s.inputs:
            # Input labels are meaningful to the model, so they are kept and
            # sorted by label.
            ins = " ".join(f"({_atom(lbl)} {self.expr(v)})"
                           for lbl, v in sorted(s.inputs, key=lambda x: x[0]))
            parts.append("(in " + ins + ")")
        if s.grounded_in:
            parts.append("(gnd " + " ".join(
                sorted(self.expr(g) for g in s.grounded_in)) + ")")
        if s.examples:
            parts.append("(ex " + " ".join(
                "(" + self.expr(a) + " " + self.expr(b) + ")"
                for a, b in s.examples) + ")")
        if s.temperature is not None:
            parts.append("(temp " + _dec(s.temperature) + ")")
        if s.retries:
            on = " ".join(sorted(_atom(r) for r in s.retry_on))
            parts.append(f"(retry {s.retries} ({on}))")
        if s.max_tokens is not None:
            parts.append(f"(maxtok {s.max_tokens})")
        if s.judge is not None:
            parts.append("(judge " + self.expr(s.judge) + ")")
        return "(" + " ".join(parts) + ")"

    def lit(self, e: A.Lit) -> str:
        if e.lit_kind == "int":
            return f"(int {e.value})"
        if e.lit_kind == "dec":
            return f"(dec {_dec(e.value)})"
        if e.lit_kind == "text":
            return f"(txt {_atom(e.value)})"
        if e.lit_kind == "bool":
            return "(bool 1)" if e.value else "(bool 0)"
        return "(unit)"

    # -- patterns --------------------------------------------------------

    def bind_pattern(self, p) -> int:
        """Bind every variable a pattern introduces. Returns how many."""
        names = []
        _pattern_vars(p, names)
        for n in names:
            self.bind(n)
        return len(names)

    def pat(self, p) -> str:
        if isinstance(p, A.PWild):
            return "(pw)"
        if isinstance(p, A.PVar):
            return "(pv)"      # name erased; binding order carries identity
        if isinstance(p, A.PLit):
            return "(pl " + self.lit(A.Lit(value=p.value, lit_kind=p.lit_kind)) + ")"
        if isinstance(p, A.PCtor):
            self.deps.add(p.name)
            return "(pc " + _atom(p.name) + "".join(
                " " + self.pat(a) for a in p.args) + ")"
        if isinstance(p, A.PRecord):
            self.deps.add(p.type_name)
            fs = " ".join(f"({_atom(n)} {self.pat(sub)})"
                          for n, sub in sorted(p.fields, key=lambda x: x[0]))
            return (f"(pr {_atom(p.type_name)} ({fs})"
                    + (" open" if p.open else "") + ")")
        if isinstance(p, A.PList):
            items = "".join(" " + self.pat(i) for i in p.items)
            return "(pls" + items + (" rest" if p.rest else "") + ")"
        raise TypeError(f"cannot encode pattern {type(p).__name__}")


COMMUTATIVE = {"+", "*", "==", "!=", "and", "or"}


def _pattern_vars(p, out: list):
    """Collect pattern-bound names in left-to-right order."""
    if isinstance(p, A.PVar):
        out.append(p.name)
    elif isinstance(p, A.PCtor):
        for a in p.args:
            _pattern_vars(a, out)
    elif isinstance(p, A.PRecord):
        for _, sub in p.fields:
            _pattern_vars(sub, out)
    elif isinstance(p, A.PList):
        for i in p.items:
            _pattern_vars(i, out)
        if p.rest:
            out.append(p.rest)


def _atom(s) -> str:
    """Quote a string into the encoding unambiguously."""
    if s is None:
        return '""'
    s = str(s)
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def _dec(d) -> str:
    """Normalize a decimal so 1.50 and 1.5 encode identically."""
    if not isinstance(d, Decimal):
        d = Decimal(str(d))
    n = d.normalize()
    sign, digits, exp = n.as_tuple()
    if isinstance(exp, int) and exp > 0:
        n = n.quantize(Decimal(1))
    return format(n, "f")


# --------------------------------------------------------------------------
# Hashing
# --------------------------------------------------------------------------

class DefInfo:
    """Everything the rest of the system needs to know about one definition."""

    __slots__ = ("name", "qualname", "module", "kind", "decl", "encoding",
                 "local_hash", "hash", "deps", "effects", "models", "origin")

    def __init__(self, name, qualname, module, kind, decl, encoding,
                 local_hash, deep_hash, deps, effects, models, origin):
        self.name = name
        self.qualname = qualname
        self.module = module
        self.kind = kind
        self.decl = decl
        self.encoding = encoding
        self.local_hash = local_hash
        self.hash = deep_hash
        self.deps = deps
        self.effects = effects
        self.models = models
        self.origin = origin

    def __repr__(self):
        return f"<{self.kind} {self.qualname} {short(self.hash)}>"


class Hasher:
    """
    Computes local and deep hashes for every declaration in a set of modules.

    Deep hashing needs dependencies resolved first, so declarations are
    processed in dependency order. Cycles are permitted: members of a cycle are
    hashed together as a group, and every member takes the group hash as its
    dependency contribution. That keeps mutual recursion from being a special
    case anywhere else in the system.
    """

    def __init__(self, ir_version: int = IR_VERSION):
        self.ir_version = ir_version
        self.defs: dict = {}        # qualname -> DefInfo
        self.by_hash: dict = {}     # deep hash -> DefInfo

    def add_modules(self, modules) -> dict:
        decls = []
        for m in modules:
            for d in m.decls:
                if isinstance(d, (A.Decl, A.EffectDecl)):
                    qn = f"{m.name}.{d.name}" if m.name else d.name
                    decls.append((m, d, qn))

        # Pass 1: local hashes and the raw name-level dependency graph.
        raw_deps = {}
        info = {}
        for m, d, qn in decls:
            enc = Encoder(resolve=None, module=m.name)
            body = enc.decl(d)
            local = digest(f"canon-ir/{self.ir_version}\n{body}".encode("utf-8"))
            info[qn] = (m, d, body, local, enc)
            raw_deps[qn] = enc.deps

        # Resolve dependency names to qualified names where possible.
        resolved_deps = {}
        for qn, deps in raw_deps.items():
            mod = qn.rsplit(".", 1)[0] if "." in qn else ""
            out = set()
            for name in deps:
                cand = f"{mod}.{name}" if mod else name
                if cand in info:
                    out.add(cand)
                elif name in info:
                    out.add(name)
                else:
                    for other in info:
                        if other.endswith("." + name):
                            out.add(other)
                            break
            out.discard(qn)
            resolved_deps[qn] = out

        # Pass 2: deep hashes, in dependency order with cycle grouping.
        order, groups = _topo_groups(resolved_deps)
        deep: dict = {}

        for group in order:
            if len(group) == 1 and group[0] not in groups:
                qn = group[0]
                m, d, _, local, _ = info[qn]
                enc = Encoder(resolve=lambda n, _q=qn: _resolve(n, _q, info, deep),
                              module=m.name)
                body = enc.decl(d)
                deep[qn] = digest(
                    f"canon-ir/{self.ir_version}\n{body}".encode("utf-8"))
                self._record(qn, m, d, body, local, deep[qn], resolved_deps[qn], enc)
            else:
                # Cyclic group: hash the concatenation of all members with
                # intra-group references left as names, then give every member
                # a hash derived from the group hash and its own position.
                members = sorted(group)
                bodies = []
                encs = {}
                for qn in members:
                    m, d, _, local, _ = info[qn]
                    enc = Encoder(
                        resolve=lambda n, _q=qn, _g=set(members):
                            _resolve(n, _q, info, deep, skip=_g),
                        module=m.name)
                    bodies.append(enc.decl(d))
                    encs[qn] = enc
                group_hash = digest(
                    (f"canon-ir/{self.ir_version}\ncycle\n" + "\n".join(bodies))
                    .encode("utf-8"))
                for qn in members:
                    deep[qn] = digest(
                        (group_hash + "/" + qn).encode("utf-8"))
                for qn in members:
                    m, d, _, local, _ = info[qn]
                    self._record(qn, m, d, encs[qn].decl(d), local, deep[qn],
                                 resolved_deps[qn], encs[qn])

        return self.defs

    def _record(self, qn, m, d, body, local, deep_hash, deps, enc):
        kind = {
            A.FnDecl: "fn", A.RecordDecl: "record", A.EnumDecl: "enum",
            A.AliasDecl: "alias", A.ConstDecl: "const",
            A.EffectDecl: "effect", A.TestDecl: "test",
        }.get(type(d), "decl")
        di = DefInfo(
            name=d.name, qualname=qn, module=m.name, kind=kind, decl=d,
            encoding=body, local_hash=local, deep_hash=deep_hash,
            deps=set(deps), effects=set(enc.effects), models=set(enc.models),
            origin=getattr(d, "origin", m.language),
        )
        self.defs[qn] = di
        self.by_hash[deep_hash] = di


def _resolve(name, from_qn, info, deep, skip=frozenset()):
    mod = from_qn.rsplit(".", 1)[0] if "." in from_qn else ""
    for cand in ((f"{mod}.{name}" if mod else None), name):
        if cand and cand in skip:
            return None
        if cand and cand in deep:
            return deep[cand]
    for other in info:
        if other.endswith("." + name) and other not in skip:
            return deep.get(other)
    return None


def _topo_groups(deps: dict):
    """
    Tarjan's strongly connected components, returning groups in dependency
    order. Nodes in a group of size > 1 are mutually recursive.
    """
    index = {}
    low = {}
    on_stack = {}
    stack = []
    result = []
    cyclic = set()
    counter = [0]

    def strongconnect(v):
        # Iterative to avoid Python recursion limits on large codebases.
        work = [(v, iter(sorted(deps.get(v, ()))))]
        index[v] = low[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        on_stack[v] = True
        while work:
            node, it = work[-1]
            advanced = False
            for w in it:
                if w not in deps:
                    continue
                if w not in index:
                    index[w] = low[w] = counter[0]
                    counter[0] += 1
                    stack.append(w)
                    on_stack[w] = True
                    work.append((w, iter(sorted(deps.get(w, ())))))
                    advanced = True
                    break
                elif on_stack.get(w):
                    low[node] = min(low[node], index[w])
            if advanced:
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
            if low[node] == index[node]:
                comp = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    comp.append(w)
                    if w == node:
                        break
                if len(comp) > 1:
                    cyclic.update(comp)
                result.append(sorted(comp))

    for v in sorted(deps):
        if v not in index:
            strongconnect(v)
    return result, cyclic


# --------------------------------------------------------------------------
# Canonical printer
# --------------------------------------------------------------------------

class Printer:
    """
    Renders declarations as canonical source text.

    There are no options. Layout is a function of the AST, so every author and
    every tool produces byte-identical output for the same definition, and a
    textual diff is always a semantic diff.
    """

    INDENT = "  "
    WIDTH = 88

    def __init__(self):
        self.out: list = []
        self.depth = 0

    def line(self, s: str = ""):
        self.out.append(self.INDENT * self.depth + s if s else "")

    def render(self) -> str:
        return "\n".join(self.out).rstrip() + "\n"

    # -- modules ---------------------------------------------------------

    def module(self, m: A.Module) -> str:
        self.out = []
        self.depth = 0
        if m.doc:
            for l in m.doc.split("\n"):
                self.line(f"--- {l}".rstrip())
        self.line(f"module {m.name}")
        if m.imports:
            self.line()
            for imp in sorted(m.imports, key=lambda i: i.module):
                s = f"import {imp.module}"
                if imp.alias:
                    s += f" as {imp.alias}"
                if imp.names:
                    s += " { " + ", ".join(sorted(imp.names)) + " }"
                self.line(s)
        # Declaration order: types, effects, constants, functions, tests.
        rank = {"record": 0, "enum": 0, "alias": 0, "effect": 1,
                "const": 2, "fn": 3, "test": 4}
        def key(d):
            k = {A.RecordDecl: "record", A.EnumDecl: "enum", A.AliasDecl: "alias",
                 A.EffectDecl: "effect", A.ConstDecl: "const", A.FnDecl: "fn",
                 A.TestDecl: "test"}.get(type(d), "fn")
            return (rank.get(k, 5), getattr(d, "name", ""))
        for d in sorted(m.decls, key=key):
            self.line()
            self.decl(d)
        return self.render()

    def decl(self, d):
        if d.doc:
            for l in d.doc.split("\n"):
                self.line(f"--- {l}".rstrip())
        if isinstance(d, A.FnDecl):
            self.fn(d)
        elif isinstance(d, A.RecordDecl):
            self.record(d)
        elif isinstance(d, A.EnumDecl):
            self.enum(d)
        elif isinstance(d, A.AliasDecl):
            self.line(f"alias {d.name}{self.tparams(d.type_params)} = "
                      f"{self.ty(d.target)}")
        elif isinstance(d, A.ConstDecl):
            self.line(f"const {d.name}: {self.ty(d.ty)} = {self.expr(d.value)}")
        elif isinstance(d, A.EffectDecl):
            self.effect(d)
        elif isinstance(d, A.TestDecl):
            self.line(f'test "{_esc(d.name)}" ' + "{")
            self.depth += 1
            self.block_body(d.body)
            self.depth -= 1
            self.line("}")

    def tparams(self, tps) -> str:
        return f"<{', '.join(tps)}>" if tps else ""

    def fn(self, d: A.FnDecl):
        params = ", ".join(f"{p.name}: {self.ty(p.ty)}"
                           + (f" = {self.expr(p.default)}" if p.default else "")
                           for p in d.params)
        self.line(f"fn {d.name}{self.tparams(d.type_params)}({params}) "
                  f"-> {self.ty(d.result)}")
        self.depth += 1
        if d.intent:
            self.line(f'intent "{_esc(d.intent)}"')
        if d.uses:
            keys = sorted({e.key() for e in d.uses})
            self.line("uses " + ", ".join(keys))
        for r in d.requires:
            self.line(f"requires {self.expr(r)}")
        for e in d.ensures:
            self.line(f"ensures {self.expr(e)}")
        for l in sorted(d.laws, key=lambda x: x.name):
            args = ("(" + ", ".join(self.expr(a) for a in l.args) + ")") if l.args else ""
            self.line(f"law {l.name}{args}")
        if d.cost:
            items = []
            for k in ("steps", "io", "tokens", "millis"):
                v = getattr(d.cost, k)
                if v is not None:
                    items.append(f"{k} {v}")
            if d.cost.money is not None:
                items.append(f"money {_dec(d.cost.money)}")
            if items:
                self.line("cost " + ", ".join(items))
        if d.decreases:
            self.line(f"decreases {self.expr(d.decreases)}")
        self.depth -= 1
        self.line("{")
        self.depth += 1
        self.block_body(d.body)
        self.depth -= 1
        self.line("}")

    def record(self, d: A.RecordDecl):
        self.line(f"record {d.name}{self.tparams(d.type_params)} " + "{")
        self.depth += 1
        if d.intent:
            self.line(f'intent "{_esc(d.intent)}"')
        for f in d.fields:
            if f.doc:
                for l in f.doc.split("\n"):
                    self.line(f"--- {l}".rstrip())
            default = f" = {self.expr(f.default)}" if f.default is not None else ""
            self.line(f"{f.name}: {self.ty(f.ty)}{default}")
        for fname, cls in sorted(d.classification.items()):
            self.line(f"classify {fname} {cls}")
        for inv in d.invariants:
            self.line(f"invariant {self.expr(inv)}")
        self.depth -= 1
        self.line("}")

    def enum(self, d: A.EnumDecl):
        self.line(f"enum {d.name}{self.tparams(d.type_params)} " + "{")
        self.depth += 1
        if d.intent:
            self.line(f'intent "{_esc(d.intent)}"')
        for v in d.variants:
            if v.doc:
                for l in v.doc.split("\n"):
                    self.line(f"--- {l}".rstrip())
            payload = ("(" + ", ".join(self.ty(t) for t in v.params) + ")") \
                if v.params else ""
            self.line(f"| {v.name}{payload}")
        self.depth -= 1
        self.line("}")

    def effect(self, d: A.EffectDecl):
        self.line(f"effect {d.name} " + "{")
        self.depth += 1
        for o in sorted(d.ops, key=lambda o: o.name):
            if o.doc:
                for l in o.doc.split("\n"):
                    self.line(f"--- {l}".rstrip())
            params = ", ".join(f"{p.name}: {self.ty(p.ty)}" for p in o.params)
            prefix = "idempotent " if o.idempotent else ""
            self.line(f"{prefix}{o.name}({params}) -> {self.ty(o.result)}")
        self.depth -= 1
        self.line("}")

    def block_body(self, b):
        if not isinstance(b, A.Block):
            self.line(self.expr(b))
            return
        for s in b.stmts:
            if isinstance(s, A.SLet):
                ann = f": {self.ty(s.ty)}" if s.ty else ""
                self.line(f"let {s.name}{ann} = {self.expr(s.value)}")
            elif isinstance(s, A.SExpr):
                self.line(f"do {self.expr(s.value)}")
            elif isinstance(s, A.SAssert):
                msg = f', "{_esc(s.message)}"' if s.message else ""
                self.line(f"assert {self.expr(s.cond)}{msg}")
        self.line(self.expr(b.result))

    # -- types -----------------------------------------------------------

    def ty(self, t) -> str:
        if t is None:
            return "Unit"
        if isinstance(t, A.TName):
            if t.args:
                return f"{t.name}<{', '.join(self.ty(a) for a in t.args)}>"
            return t.name
        if isinstance(t, A.TVar):
            return t.name
        if isinstance(t, A.TFn):
            s = f"Fn({', '.join(self.ty(p) for p in t.params)}) -> {self.ty(t.result)}"
            if t.effects:
                s += " uses " + ", ".join(sorted(e.key() for e in t.effects))
            return s
        if isinstance(t, A.TRecord):
            fs = ", ".join(f"{n}: {self.ty(ft)}"
                           for n, ft in sorted(t.fields, key=lambda x: x[0]))
            return "{ " + fs + " }"
        return "Unknown"

    # -- expressions -----------------------------------------------------

    def expr(self, e, prec: int = 0) -> str:
        if e is None:
            return "()"
        if isinstance(e, A.Lit):
            return self.lit(e)
        if isinstance(e, A.Var):
            return e.name
        if isinstance(e, A.QualVar):
            return e.full()
        if isinstance(e, A.Field):
            return f"{self.expr(e.target, 99)}.{e.name}"
        if isinstance(e, A.Call):
            return (f"{self.expr(e.fn, 99)}("
                    + ", ".join(self.expr(a) for a in e.args) + ")")
        if isinstance(e, A.Perform):
            return (f"{e.effect}.{e.op}("
                    + ", ".join(self.expr(a) for a in e.args) + ")")
        if isinstance(e, A.Lambda):
            ps = ", ".join(f"{p.name}: {self.ty(p.ty)}" for p in e.params)
            return f"fn({ps}) => {self.expr(e.body)}"
        if isinstance(e, A.Let):
            ann = f": {self.ty(e.ty)}" if e.ty else ""
            return (f"let {e.name}{ann} = {self.expr(e.value)} "
                    f"in {self.expr(e.body)}")
        if isinstance(e, A.If):
            return (f"if {self.expr(e.cond)} then {self.expr(e.then)} "
                    f"else {self.expr(e.otherwise)}")
        if isinstance(e, A.Match):
            return self.match_inline(e)
        if isinstance(e, A.ListLit):
            return "[" + ", ".join(self.expr(i) for i in e.items) + "]"
        if isinstance(e, A.MapLit):
            return ("{" + ", ".join(f"{self.expr(k)}: {self.expr(v)}"
                                    for k, v in e.entries) + "}")
        if isinstance(e, A.RecordLit):
            fs = ", ".join(f"{n}: {self.expr(v)}"
                           for n, v in sorted(e.fields, key=lambda x: x[0]))
            base = f"..{self.expr(e.base)}, " if e.base else ""
            return f"{e.type_name} {{ {base}{fs} }}"
        if isinstance(e, A.CtorCall):
            if e.args:
                return f"{e.name}(" + ", ".join(self.expr(a) for a in e.args) + ")"
            return e.name
        if isinstance(e, A.Binary):
            p, assoc = A.BINARY_OPS.get(e.op, (8, "left"))
            l = self.expr(e.left, p if assoc == "left" else p + 1)
            r = self.expr(e.right, p + 1 if assoc == "left" else p)
            s = f"{l} {e.op} {r}"
            return f"({s})" if p < prec else s
        if isinstance(e, A.Unary):
            sep = " " if e.op == "not" else ""
            return f"{e.op}{sep}{self.expr(e.operand, 9)}"
        if isinstance(e, A.Try):
            return f"{self.expr(e.operand, 99)}?"
        if isinstance(e, A.Block):
            return self.block_inline(e)
        if isinstance(e, A.Ask):
            return self.ask_inline(e)
        return "<?>"

    def block_inline(self, b: A.Block) -> str:
        sub = Printer()
        sub.depth = self.depth + 1
        sub.block_body(b)
        inner = "\n".join(sub.out)
        return "{\n" + inner + "\n" + self.INDENT * self.depth + "}"

    def match_inline(self, e: A.Match) -> str:
        sub = Printer()
        sub.depth = self.depth + 1
        for arm in e.arms:
            guard = f" if {sub.expr(arm.guard)}" if arm.guard else ""
            sub.line(f"case {sub.pat(arm.pattern)}{guard} => "
                     f"{sub.expr(arm.body)}")
        inner = "\n".join(sub.out)
        return (f"match {self.expr(e.scrutinee)} " + "{\n" + inner + "\n"
                + self.INDENT * self.depth + "}")

    def ask_inline(self, e: A.Ask) -> str:
        s = e.spec or A.AskSpec()
        sub = Printer()
        sub.depth = self.depth + 1
        if s.system is not None:
            sub.line(f"system {sub.expr(s.system)}")
        for lbl, v in s.inputs:
            if isinstance(v, A.Var) and v.name == lbl:
                sub.line(f"input {lbl}")
            else:
                sub.line(f"input {lbl}: {sub.expr(v)}")
        for g in s.grounded_in:
            sub.line(f"grounded_in {sub.expr(g)}")
        if s.examples:
            pairs = ", ".join(f"({sub.expr(a)}, {sub.expr(b)})"
                              for a, b in s.examples)
            sub.line(f"examples [{pairs}]")
        if s.temperature is not None:
            sub.line(f"temperature {_dec(s.temperature)}")
        if s.retries:
            on = (" on " + ", ".join(s.retry_on)) if s.retry_on else ""
            sub.line(f"retries {s.retries}{on}")
        if s.max_tokens is not None:
            sub.line(f"max_tokens {s.max_tokens}")
        if s.judge is not None:
            sub.line(f"judge {sub.expr(s.judge)}")
        inner = "\n".join(sub.out)
        return (f"ask {self.ty(e.result_type)} from {e.model} " + "{\n"
                + inner + "\n" + self.INDENT * self.depth + "}")

    def lit(self, e: A.Lit) -> str:
        if e.lit_kind == "text":
            return '"' + _esc(e.value) + '"'
        if e.lit_kind == "bool":
            return "true" if e.value else "false"
        if e.lit_kind == "unit":
            return "()"
        if e.lit_kind == "dec":
            return _dec(e.value)
        return str(e.value)

    def pat(self, p) -> str:
        if isinstance(p, A.PWild):
            return "_"
        if isinstance(p, A.PVar):
            return p.name
        if isinstance(p, A.PLit):
            return self.lit(A.Lit(value=p.value, lit_kind=p.lit_kind))
        if isinstance(p, A.PCtor):
            if p.args:
                return f"{p.name}(" + ", ".join(self.pat(a) for a in p.args) + ")"
            return p.name
        if isinstance(p, A.PRecord):
            fs = ", ".join(
                (n if isinstance(sub, A.PVar) and sub.name == n
                 else f"{n}: {self.pat(sub)}")
                for n, sub in p.fields)
            tail = ", .." if p.open else ""
            return f"{p.type_name} {{ {fs}{tail} }}"
        if isinstance(p, A.PList):
            items = ", ".join(self.pat(i) for i in p.items)
            rest = f", ..{p.rest}" if p.rest else ""
            return f"[{items}{rest}]"
        return "_"


def _esc(s) -> str:
    if s is None:
        return ""
    return (str(s).replace("\\", "\\\\").replace('"', '\\"')
            .replace("\n", "\\n").replace("\t", "\\t"))


def format_module(m: A.Module) -> str:
    return Printer().module(m)


def encode_decl(d, resolve=None) -> str:
    return Encoder(resolve=resolve).decl(d)


def hash_decl(d, resolve=None, ir_version: int = IR_VERSION) -> str:
    body = Encoder(resolve=resolve).decl(d)
    return digest(f"canon-ir/{ir_version}\n{body}".encode("utf-8"))
