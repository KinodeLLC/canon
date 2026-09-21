"""
Canon diagnostics.

Diagnostics are structured objects rather than formatted strings, because the
primary consumer is a program that has to act on them. Each one carries:

  code        stable machine identifier (CANON-E0101)
  severity    error | warning | advice
  span        source location
  facts       key/value payload (expected type, actual type, missing
              capability, counterexample input, and so on)
  repairs     zero or more concrete edits that would resolve it

Rendered text is generated from the structured form. Nothing downstream should
parse the rendered output.

Diagnostic codes are treated as a public API: new ones can be added, but the
meaning of an existing code does not change.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional


# --------------------------------------------------------------------------
# Source positions
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Pos:
    """A point in a source file. 1-indexed line, 0-indexed column."""
    file: str
    line: int
    col: int
    offset: int = 0

    def __str__(self) -> str:
        return f"{self.file}:{self.line}:{self.col + 1}"


@dataclass(frozen=True)
class Span:
    """A half-open range in a source file."""
    start: Pos
    end: Pos

    @staticmethod
    def unknown() -> "Span":
        p = Pos("<unknown>", 0, 0, 0)
        return Span(p, p)

    def merge(self, other: "Span") -> "Span":
        if other.start.file != self.start.file:
            return self
        s = self.start if self.start.offset <= other.start.offset else other.start
        e = self.end if self.end.offset >= other.end.offset else other.end
        return Span(s, e)

    def __str__(self) -> str:
        if self.start.line == self.end.line:
            return f"{self.start.file}:{self.start.line}:{self.start.col + 1}-{self.end.col + 1}"
        return f"{self.start}..{self.end.line}:{self.end.col + 1}"

    def to_json(self) -> dict:
        return {
            "file": self.start.file,
            "startLine": self.start.line,
            "startCol": self.start.col,
            "endLine": self.end.line,
            "endCol": self.end.col,
            "startOffset": self.start.offset,
            "endOffset": self.end.offset,
        }


# --------------------------------------------------------------------------
# Severity
# --------------------------------------------------------------------------

class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    ADVICE = "advice"


# --------------------------------------------------------------------------
# Repairs
# --------------------------------------------------------------------------

@dataclass
class Repair:
    """
    A concrete repair an agent (or a human tool) can apply mechanically.

    kind:
      "replace-span"    replace `span` with `text`
      "insert-before"   insert `text` at span.start
      "insert-after"    insert `text` at span.end
      "add-clause"      add a function clause (e.g. `uses payments.write`)
      "add-capability"  request a capability grant from the broker
      "manual"          no mechanical edit; `text` describes the required change
    """
    kind: str
    description: str
    text: str = ""
    span: Optional[Span] = None
    confidence: float = 0.5  # 0..1; >=0.9 means "apply without asking"

    def to_json(self) -> dict:
        return {
            "kind": self.kind,
            "description": self.description,
            "text": self.text,
            "span": self.span.to_json() if self.span else None,
            "confidence": self.confidence,
        }


# --------------------------------------------------------------------------
# Diagnostic
# --------------------------------------------------------------------------

@dataclass
class Diagnostic:
    code: str
    severity: Severity
    message: str
    span: Span
    facts: dict = field(default_factory=dict)
    repairs: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    related: list = field(default_factory=list)  # [(Span, str)]

    def to_json(self) -> dict:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "span": self.span.to_json(),
            "facts": _jsonable(self.facts),
            "repairs": [r.to_json() for r in self.repairs],
            "notes": list(self.notes),
            "related": [{"span": s.to_json(), "message": m} for s, m in self.related],
        }

    def render(self, source: Optional[str] = None, color: bool = False) -> str:
        """Render for display. Generated from the structured fields above."""
        sev = self.severity.value
        if color:
            hue = {"error": "\033[31m", "warning": "\033[33m", "advice": "\033[36m"}[sev]
            head = f"{hue}{sev}\033[0m[\033[1m{self.code}\033[0m]: {self.message}"
        else:
            head = f"{sev}[{self.code}]: {self.message}"
        lines = [head, f"  --> {self.span}"]
        if source is not None:
            lines.extend(_snippet(source, self.span))
        for k, v in self.facts.items():
            lines.append(f"  {k}: {_render_fact(v)}")
        for n in self.notes:
            lines.append(f"  note: {n}")
        for sp, msg in self.related:
            lines.append(f"  related: {msg} at {sp}")
        for r in self.repairs:
            tag = "fix" if r.confidence >= 0.9 else "try"
            body = f" `{r.text}`" if r.text and "\n" not in r.text else ""
            lines.append(f"  {tag}: {r.description}{body}")
        return "\n".join(lines)


def _render_fact(v: Any) -> str:
    if isinstance(v, (list, tuple)):
        return ", ".join(str(x) for x in v) if v else "(none)"
    return str(v)


def _jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    if isinstance(o, Span):
        return o.to_json()
    return str(o)


def _snippet(source: str, span: Span, context: int = 0) -> list:
    lines = source.splitlines()
    out = []
    lo = max(1, span.start.line - context)
    hi = min(len(lines), span.end.line + context)
    width = len(str(hi))
    for ln in range(lo, hi + 1):
        if ln - 1 >= len(lines):
            break
        out.append(f"  {str(ln).rjust(width)} | {lines[ln - 1]}")
        if ln == span.start.line:
            end_col = span.end.col if span.end.line == ln else len(lines[ln - 1])
            n = max(1, end_col - span.start.col)
            out.append("  " + " " * width + " | " + " " * span.start.col + "^" * n)
    return out


# --------------------------------------------------------------------------
# Error bag
# --------------------------------------------------------------------------

class CanonError(Exception):
    """Raised when a phase cannot continue. Carries structured diagnostics."""

    def __init__(self, diagnostics):
        if isinstance(diagnostics, Diagnostic):
            diagnostics = [diagnostics]
        self.diagnostics = list(diagnostics)
        super().__init__(self.diagnostics[0].message if self.diagnostics else "canon error")

    def to_json(self) -> dict:
        return {"ok": False, "diagnostics": [d.to_json() for d in self.diagnostics]}


class Bag:
    """Collects diagnostics so a phase can report many problems in one pass."""

    def __init__(self):
        self.items: list = []

    def add(self, d: Diagnostic) -> Diagnostic:
        self.items.append(d)
        return d

    def error(self, code, message, span, **kw) -> Diagnostic:
        return self.add(Diagnostic(code, Severity.ERROR, message, span, **kw))

    def warn(self, code, message, span, **kw) -> Diagnostic:
        return self.add(Diagnostic(code, Severity.WARNING, message, span, **kw))

    def advise(self, code, message, span, **kw) -> Diagnostic:
        return self.add(Diagnostic(code, Severity.ADVICE, message, span, **kw))

    @property
    def errors(self) -> list:
        return [d for d in self.items if d.severity == Severity.ERROR]

    @property
    def has_errors(self) -> bool:
        return any(d.severity == Severity.ERROR for d in self.items)

    def raise_if_errors(self):
        if self.has_errors:
            raise CanonError(self.items)

    def extend(self, other):
        if isinstance(other, Bag):
            self.items.extend(other.items)
        else:
            self.items.extend(other)

    def to_json(self) -> list:
        return [d.to_json() for d in self.items]

    def render(self, source=None, color=False) -> str:
        return "\n".join(d.render(source, color) for d in self.items)

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items)


# --------------------------------------------------------------------------
# Diagnostic code registry. Codes are stable; add new ones rather than
# repurposing existing ones.
# --------------------------------------------------------------------------

CODES = {
    # --- lexing: 00xx
    "CANON-E0001": "unexpected character",
    "CANON-E0002": "unterminated text literal",
    "CANON-E0003": "invalid numeric literal",
    "CANON-E0004": "unterminated block comment",
    "CANON-E0005": "tab character in source (canonical form uses spaces)",

    # --- parsing: 01xx
    "CANON-E0101": "unexpected token",
    "CANON-E0102": "expected a declaration",
    "CANON-E0103": "expected an expression",
    "CANON-E0104": "expected a type",
    "CANON-E0105": "unclosed delimiter",
    "CANON-E0106": "duplicate clause",
    "CANON-E0107": "missing module header",
    "CANON-E0108": "expected a pattern",

    # --- naming / resolution: 02xx
    "CANON-E0201": "unknown name",
    "CANON-E0202": "unknown type",
    "CANON-E0203": "duplicate definition",
    "CANON-E0204": "unknown effect",
    "CANON-E0205": "unknown effect operation",
    "CANON-E0206": "unknown field",
    "CANON-E0207": "unknown constructor",
    "CANON-E0208": "unknown module",
    "CANON-E0209": "cyclic definition",

    # --- types: 03xx
    "CANON-E0301": "type mismatch",
    "CANON-E0302": "wrong number of arguments",
    "CANON-E0303": "not callable",
    "CANON-E0304": "field access on non-record",
    "CANON-E0305": "non-exhaustive match",
    "CANON-E0306": "unreachable match arm",
    "CANON-E0307": "branches have different types",
    "CANON-E0308": "cannot infer type",
    "CANON-E0309": "constructor arity mismatch",
    "CANON-E0310": "error propagation outside Result-returning function",
    "CANON-E0311": "recursive definition without a decreases clause",
    "CANON-E0312": "type argument mismatch",

    # --- effects / capabilities: 04xx
    "CANON-E0401": "undeclared effect",
    "CANON-E0402": "declared effect is never used",
    "CANON-E0403": "capability not granted",
    "CANON-E0404": "effect performed in a pure context",
    "CANON-E0405": "capability escalation across call boundary",
    "CANON-E0406": "effect operation outside its declared effect",

    # --- contracts: 05xx
    "CANON-E0501": "precondition not satisfiable",
    "CANON-E0502": "postcondition violated by counterexample",
    "CANON-E0503": "law violated by counterexample",
    "CANON-E0504": "contract references an unbound name",
    "CANON-E0505": "unknown law",
    "CANON-E0506": "law arity mismatch",
    "CANON-E0507": "precondition violated at runtime",
    "CANON-E0508": "postcondition violated at runtime",

    # --- totality / cost: 06xx
    "CANON-E0601": "step budget exceeded",
    "CANON-E0602": "io budget exceeded",
    "CANON-E0603": "declared cost exceeded under verification",
    "CANON-E0604": "unbounded recursion detected",
    "CANON-E0605": "decreases clause does not decrease",

    # --- runtime: 07xx
    "CANON-E0701": "division by zero",
    "CANON-E0702": "index out of bounds",
    "CANON-E0703": "key not found",
    "CANON-E0704": "arithmetic overflow",
    "CANON-E0705": "pattern match failure",
    "CANON-E0706": "explicit abort",

    # --- ledger / replay: 08xx
    "CANON-E0801": "journal divergence: operation mismatch",
    "CANON-E0802": "journal divergence: argument mismatch",
    "CANON-E0803": "journal exhausted during replay",
    "CANON-E0804": "nondeterminism detected",
    "CANON-E0805": "journal integrity: hash chain broken",

    # --- atlas / governance: 09xx
    "CANON-E0901": "definition hash not found",
    "CANON-E0902": "edit transaction conflict",
    "CANON-E0903": "promotion blocked by policy",
    "CANON-E0904": "behavioral regression detected in shadow",
    "CANON-E0905": "blast radius exceeds authorization",

    # --- warnings
    "CANON-W0001": "unused binding",
    "CANON-W0002": "unused parameter",
    "CANON-W0003": "shadowed binding",
    "CANON-W0004": "function has no contracts",
    "CANON-W0005": "function has no intent",
    "CANON-W0006": "broad capability requested",
    "CANON-W0007": "non-canonical formatting",
    "CANON-W0008": "law is untested (no generator for parameter type)",
}


def describe(code: str) -> str:
    return CODES.get(code, "unknown diagnostic code")


def dumps(diags, indent=2) -> str:
    return json.dumps([d.to_json() for d in diags], indent=indent)
