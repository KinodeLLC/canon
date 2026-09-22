"""
the abstract syntax, which is the core ir.

the other six languages, intent and loom and verdict and weft and tract and
rune, all lower into these node types. that is what lets one type checker and
one verifier and one runtime and one audit trail cover all of them instead of
six of each.

every node carries a span so diagnostics can point at something. nodes are
mutable while lowering runs and then get frozen by the canonicaliser, which is
what produces the hashed form.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Optional

from .diagnostics import Span


# --------------------------------------------------------------------------
# Base
# --------------------------------------------------------------------------

@dataclass
class Node:
    span: Span = field(default_factory=Span.unknown, repr=False, compare=False)

    def children(self):
        """Yield direct child nodes, for generic traversal."""
        for f in self.__dataclass_fields__:
            if f == "span":
                continue
            v = getattr(self, f)
            if isinstance(v, Node):
                yield v
            elif isinstance(v, (list, tuple)):
                for x in v:
                    if isinstance(x, Node):
                        yield x

    def walk(self):
        yield self
        for c in self.children():
            yield from c.walk()


# ==========================================================================
# Types
# ==========================================================================

@dataclass
class TypeExpr(Node):
    pass


@dataclass
class TName(TypeExpr):
    """A named type, possibly generic: Int, List<Text>, Result<A, B>."""
    name: str = ""
    args: list = field(default_factory=list)


@dataclass
class TFn(TypeExpr):
    """Function type with an effect row: Fn(Int, Text) -> Bool uses db.read"""
    params: list = field(default_factory=list)
    result: Optional[TypeExpr] = None
    effects: list = field(default_factory=list)   # list[EffectRef]


@dataclass
class TRecord(TypeExpr):
    """Structural record type, used mostly by lowered languages."""
    fields: list = field(default_factory=list)    # list[(name, TypeExpr)]


@dataclass
class TVar(TypeExpr):
    """A type variable introduced by a generic definition."""
    name: str = ""


# ==========================================================================
# Effects and capabilities
# ==========================================================================

@dataclass
class EffectRef(Node):
    """
    A reference to an effect or one of its operations.

    `db.read`    -> effect "db", op "read"
    `db.*`       -> effect "db", op None (all operations)
    """
    effect: str = ""
    op: Optional[str] = None

    def key(self) -> str:
        return f"{self.effect}.{self.op}" if self.op else f"{self.effect}.*"

    def covers(self, other: "EffectRef") -> bool:
        if self.effect != other.effect:
            return False
        return self.op is None or self.op == other.op

    def __str__(self):
        return self.key()


@dataclass
class EffectOp(Node):
    name: str = ""
    params: list = field(default_factory=list)   # list[Param]
    result: Optional[TypeExpr] = None
    doc: str = ""
    # Whether repeating this op with identical arguments is safe. Used by the
    # workflow engine to decide what can be retried without compensation.
    idempotent: bool = False


@dataclass
class EffectDecl(Node):
    name: str = ""
    ops: list = field(default_factory=list)      # list[EffectOp]
    doc: str = ""


# ==========================================================================
# Patterns
# ==========================================================================

@dataclass
class Pattern(Node):
    pass


@dataclass
class PWild(Pattern):
    pass


@dataclass
class PVar(Pattern):
    name: str = ""


@dataclass
class PLit(Pattern):
    value: Any = None
    lit_kind: str = "int"      # int | dec | text | bool | unit


@dataclass
class PCtor(Pattern):
    """Constructor pattern: Ok(x), None, Captured, Some(Charge { id: i })"""
    name: str = ""
    args: list = field(default_factory=list)


@dataclass
class PRecord(Pattern):
    """Record destructuring: Charge { id: i, amount: a }"""
    type_name: str = ""
    fields: list = field(default_factory=list)   # list[(field_name, Pattern)]
    open: bool = False                            # trailing `..`


@dataclass
class PList(Pattern):
    items: list = field(default_factory=list)
    rest: Optional[str] = None                    # [a, b, ..tail]


# ==========================================================================
# Expressions
# ==========================================================================

@dataclass
class Expr(Node):
    pass


@dataclass
class Lit(Expr):
    value: Any = None
    lit_kind: str = "int"      # int | dec | text | bool | unit


@dataclass
class Var(Expr):
    name: str = ""


@dataclass
class QualVar(Expr):
    """A name qualified by module: payments.refund"""
    module: str = ""
    name: str = ""

    def full(self) -> str:
        return f"{self.module}.{self.name}"


@dataclass
class Field(Expr):
    target: Optional[Expr] = None
    name: str = ""


@dataclass
class Call(Expr):
    fn: Optional[Expr] = None
    args: list = field(default_factory=list)


@dataclass
class Perform(Expr):
    """An effect operation call: ledger.append(entry)."""
    effect: str = ""
    op: str = ""
    args: list = field(default_factory=list)


@dataclass
class Lambda(Expr):
    params: list = field(default_factory=list)   # list[Param]
    body: Optional[Expr] = None


@dataclass
class Let(Expr):
    name: str = ""
    ty: Optional[TypeExpr] = None
    value: Optional[Expr] = None
    body: Optional[Expr] = None


@dataclass
class If(Expr):
    cond: Optional[Expr] = None
    then: Optional[Expr] = None
    otherwise: Optional[Expr] = None


@dataclass
class MatchArm(Node):
    pattern: Optional[Pattern] = None
    guard: Optional[Expr] = None
    body: Optional[Expr] = None


@dataclass
class Match(Expr):
    scrutinee: Optional[Expr] = None
    arms: list = field(default_factory=list)


@dataclass
class ListLit(Expr):
    items: list = field(default_factory=list)


@dataclass
class MapLit(Expr):
    entries: list = field(default_factory=list)  # list[(Expr, Expr)]


@dataclass
class RecordLit(Expr):
    type_name: str = ""
    fields: list = field(default_factory=list)   # list[(name, Expr)]
    base: Optional[Expr] = None                  # Type { ..base, f: v }


@dataclass
class CtorCall(Expr):
    name: str = ""
    args: list = field(default_factory=list)


@dataclass
class Binary(Expr):
    op: str = ""
    left: Optional[Expr] = None
    right: Optional[Expr] = None


@dataclass
class Unary(Expr):
    op: str = ""
    operand: Optional[Expr] = None


@dataclass
class Try(Expr):
    """Postfix `?`: unwrap Ok / return Err from the enclosing function."""
    operand: Optional[Expr] = None


@dataclass
class Block(Expr):
    """A sequence of let bindings and effectful expressions ending in a value."""
    stmts: list = field(default_factory=list)    # list[Stmt]
    result: Optional[Expr] = None


@dataclass
class Stmt(Node):
    pass


@dataclass
class SLet(Stmt):
    name: str = ""
    ty: Optional[TypeExpr] = None
    value: Optional[Expr] = None


@dataclass
class SExpr(Stmt):
    value: Optional[Expr] = None


@dataclass
class SAssert(Stmt):
    """An in-body invariant check. Compiles to a contract obligation."""
    cond: Optional[Expr] = None
    message: str = ""


# --------------------------------------------------------------------------
# The native model-invocation form
# --------------------------------------------------------------------------

@dataclass
class AskSpec(Node):
    """
    Settings on an `ask` expression. Everything here is part of the hashed
    definition, so changing a prompt changes the definition hash and therefore
    invalidates its cached results and its verification record.
    """
    system: Optional[Expr] = None            # system instruction (Text)
    inputs: list = field(default_factory=list)   # list[(label, Expr)]
    grounded_in: list = field(default_factory=list)  # list[Expr] of Text sources
    examples: list = field(default_factory=list)     # list[(Expr, Expr)] in/out
    temperature: Optional[Decimal] = None
    retries: int = 0
    retry_on: list = field(default_factory=list)  # contract_violation | type_error | refusal
    max_tokens: Optional[int] = None
    judge: Optional[Expr] = None             # optional secondary check


@dataclass
class Ask(Expr):
    """
    `ask <Type> from <model> { ... }`

    Evaluates to the declared type. The runtime is responsible for coercing the
    model response into that type, enforcing any contract obligations attached
    to the surrounding function, retrying on failure, and journaling the whole
    exchange.
    """
    result_type: Optional[TypeExpr] = None
    model: str = ""
    spec: Optional[AskSpec] = None


# ==========================================================================
# Declarations
# ==========================================================================

@dataclass
class Param(Node):
    name: str = ""
    ty: Optional[TypeExpr] = None
    default: Optional[Expr] = None
    doc: str = ""


@dataclass
class LawRef(Node):
    """A named property obligation, checked by the verifier."""
    name: str = ""
    args: list = field(default_factory=list)


@dataclass
class Cost(Node):
    """
    Declared resource budget. Enforced at runtime and checked by the verifier,
    which reports the observed maximum against the declared ceiling.
    """
    steps: Optional[int] = None
    io: Optional[int] = None
    tokens: Optional[int] = None
    millis: Optional[int] = None
    money: Optional[Decimal] = None


@dataclass
class Decl(Node):
    name: str = ""
    doc: str = ""
    intent: str = ""
    exported: bool = True


@dataclass
class FnDecl(Decl):
    type_params: list = field(default_factory=list)
    params: list = field(default_factory=list)
    result: Optional[TypeExpr] = None
    uses: list = field(default_factory=list)         # list[EffectRef]
    requires: list = field(default_factory=list)     # list[Expr]
    ensures: list = field(default_factory=list)      # list[Expr]
    laws: list = field(default_factory=list)         # list[LawRef]
    cost: Optional[Cost] = None
    body: Optional[Expr] = None
    recursive: bool = False
    decreases: Optional[Expr] = None
    # Set by lowering: which surface language produced this definition.
    origin: str = "canon"


@dataclass
class RecordDecl(Decl):
    type_params: list = field(default_factory=list)
    fields: list = field(default_factory=list)       # list[Param]
    invariants: list = field(default_factory=list)   # list[Expr] over `self`
    # Data classification, set here or by Weft. Drives policy enforcement.
    classification: dict = field(default_factory=dict)  # field -> class


@dataclass
class EnumVariant(Node):
    name: str = ""
    params: list = field(default_factory=list)       # list[TypeExpr]
    doc: str = ""


@dataclass
class EnumDecl(Decl):
    type_params: list = field(default_factory=list)
    variants: list = field(default_factory=list)


@dataclass
class AliasDecl(Decl):
    type_params: list = field(default_factory=list)
    target: Optional[TypeExpr] = None


@dataclass
class ConstDecl(Decl):
    ty: Optional[TypeExpr] = None
    value: Optional[Expr] = None


@dataclass
class TestDecl(Decl):
    body: Optional[Expr] = None
    expect: Optional[Expr] = None


@dataclass
class ImportDecl(Node):
    module: str = ""
    alias: str = ""
    names: list = field(default_factory=list)


@dataclass
class Module(Node):
    name: str = ""
    doc: str = ""
    imports: list = field(default_factory=list)
    decls: list = field(default_factory=list)
    # Surface language this module was written in.
    language: str = "canon"
    source_file: str = ""

    def by_name(self, name: str):
        for d in self.decls:
            if isinstance(d, (Decl, EffectDecl)) and d.name == name:
                return d
        return None

    def functions(self):
        return [d for d in self.decls if isinstance(d, FnDecl)]

    def types(self):
        return [d for d in self.decls
                if isinstance(d, (RecordDecl, EnumDecl, AliasDecl))]

    def effects(self):
        return [d for d in self.decls if isinstance(d, EffectDecl)]

    def tests(self):
        return [d for d in self.decls if isinstance(d, TestDecl)]


# --------------------------------------------------------------------------
# Operator metadata used by the parser and the canonical printer
# --------------------------------------------------------------------------

# (precedence, associativity). Higher binds tighter.
BINARY_OPS = {
    "or":  (1, "left"),
    "||":  (1, "left"),
    "and": (2, "left"),
    "&&":  (2, "left"),
    "==":  (3, "none"),
    "!=":  (3, "none"),
    "<":   (4, "none"),
    ">":   (4, "none"),
    "<=":  (4, "none"),
    ">=":  (4, "none"),
    "++":  (5, "right"),
    "+":   (6, "left"),
    "-":   (6, "left"),
    "*":   (7, "left"),
    "/":   (7, "left"),
    "%":   (7, "left"),
    "|>":  (0, "left"),
}

UNARY_OPS = {"-", "not", "!"}

# Canonical spelling: the printer always emits the left column, so `&&` and
# `and` collapse to one form and cannot produce two different hashes.
OP_CANONICAL = {
    "&&": "and",
    "||": "or",
    "!": "not",
}
