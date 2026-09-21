"""
Canon lexer, shared by every language in the family.

The tokenizer is parameterized by a keyword set, so Canon, Loom, Verdict, Weft,
Tract, Rune and Intent all use the same lexical layer: one whitespace policy,
one comment syntax, one escape table, one numeric tower.

Lexical rules, chosen to keep the number of ways to write the same thing as
close to one as possible:

  - Tabs are an error. Indentation is spaces only.
  - One comment syntax (`--`) and one doc-comment syntax (`---`).
  - Numbers are Int (arbitrary precision) or Dec (exact decimal). There is no
    binary floating point type, since most of the target workloads are
    monetary or regulatory and cannot tolerate representation error.
  - A small closed escape table for text literals. No octal or \\x escapes.
  - No user-defined operators, macros, or syntax extensions.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from .diagnostics import Bag, Diagnostic, Pos, Repair, Severity, Span


# --------------------------------------------------------------------------
# Token kinds
# --------------------------------------------------------------------------

class T:
    INT = "int"
    DEC = "dec"
    TEXT = "text"
    NAME = "name"          # lower_snake identifier
    UPPER = "upper"        # UpperCamel identifier (types, constructors)
    KEYWORD = "keyword"
    OP = "op"
    PUNCT = "punct"
    DOC = "doc"            # --- doc comment line
    EOF = "eof"


@dataclass(frozen=True)
class Token:
    kind: str
    value: str
    span: Span
    # parsed payload for literals
    payload: object = None

    def __repr__(self) -> str:
        return f"<{self.kind} {self.value!r} @{self.span.start.line}:{self.span.start.col}>"

    def is_kw(self, *kws) -> bool:
        return self.kind == T.KEYWORD and self.value in kws

    def is_punct(self, *ps) -> bool:
        return self.kind == T.PUNCT and self.value in ps

    def is_op(self, *ops) -> bool:
        return self.kind == T.OP and self.value in ops


# --------------------------------------------------------------------------
# Operator and punctuation tables
# --------------------------------------------------------------------------

# Longest-match-first ordering matters.
OPERATORS = [
    "==", "!=", "<=", ">=", "=>", "->", "|>", "++", "&&", "||", "..",
    "+", "-", "*", "/", "%", "<", ">", "=", "!", "?", "|", "&", "^",
]

PUNCTUATION = ["(", ")", "{", "}", "[", "]", ",", ":", ";", ".", "@", "#", "$"]

# Globally reserved words. This set is deliberately small: every reserved word
# is a name an author cannot use, and a language whose code is mostly generated
# should not litter the identifier space. Anything that only has meaning inside
# a particular block -- `system` and `input` inside an `ask` body, `step` inside
# a workflow, `factor` inside a decision -- is a contextual keyword instead. It
# lexes as an ordinary NAME and the relevant parser recognises it by position,
# so `input` remains usable as a variable or field name everywhere else.
KEYWORDS = {
    # module structure
    "module", "import", "export", "as", "from",
    # definitions
    "fn", "record", "enum", "alias", "effect", "const", "test",
    # expressions
    "let", "in", "if", "then", "else", "match", "case", "do",
    "true", "false", "not", "and", "or", "assert", "abort",
    # contracts
    "requires", "ensures", "law", "uses", "cost", "intent", "doc",
    "decreases", "recursive", "result", "old",
    # the native model-invocation form
    "ask",
}

# Contextual keywords, listed here for tooling and documentation. These are
# NOT reserved; each is recognised only by the parser that owns its block.
CONTEXTUAL = {
    "canon": {"invariant", "classify", "idempotent"},
    "ask": {"system", "input", "grounded_in", "examples", "temperature",
            "retries", "on", "max_tokens", "judge", "schema"},
    "loom": {"workflow", "step", "saga", "compensate", "await", "timer",
             "parallel", "signal", "query", "deadline", "retry", "activity",
             "emit", "on_failure", "heartbeat"},
    "verdict": {"decision", "factor", "rule", "table", "when", "otherwise",
                "because", "prohibited", "weight", "outcome", "explain",
                "band", "scale", "reason", "basis"},
    "weft": {"schema", "field", "migrate", "forward", "backward", "source",
             "sink", "transform", "pipeline", "key", "unique", "index",
             "retention", "derive", "partition"},
    "tract": {"resource", "provision", "region", "scale_to", "expose",
              "secret", "depends", "environment", "quota", "budget"},
    "rune": {"policy", "grant", "deny", "allow", "limit", "require_approval",
             "blast_radius", "promote", "role", "actor", "audit", "sunset",
             "window", "escalate"},
    "intent": {"goal", "given", "expect", "scenario", "nonfunctional",
               "owner", "rationale", "accepts", "rejects", "traces",
               "background", "metric"},
}

ESCAPES = {
    "n": "\n", "t": "\t", "r": "\r", "\\": "\\",
    '"': '"', "0": "\0", "'": "'",
}


# --------------------------------------------------------------------------
# Lexer
# --------------------------------------------------------------------------

class Lexer:
    def __init__(self, source: str, filename: str = "<memory>",
                 keywords: Optional[set] = None, bag: Optional[Bag] = None):
        self.src = source
        self.file = filename
        self.keywords = KEYWORDS if keywords is None else keywords
        self.bag = bag if bag is not None else Bag()
        self.i = 0
        self.line = 1
        self.col = 0
        self.tokens: list = []

    # -- position helpers ---------------------------------------------------

    def pos(self) -> Pos:
        return Pos(self.file, self.line, self.col, self.i)

    def _advance(self, n: int = 1) -> str:
        out = self.src[self.i:self.i + n]
        for ch in out:
            if ch == "\n":
                self.line += 1
                self.col = 0
            else:
                self.col += 1
        self.i += n
        return out

    def _peek(self, n: int = 0) -> str:
        j = self.i + n
        return self.src[j] if j < len(self.src) else ""

    def _at_end(self) -> bool:
        return self.i >= len(self.src)

    def _span(self, start: Pos) -> Span:
        return Span(start, self.pos())

    def _emit(self, kind: str, value: str, start: Pos, payload=None):
        self.tokens.append(Token(kind, value, self._span(start), payload))

    # -- main loop ----------------------------------------------------------

    def run(self) -> list:
        while not self._at_end():
            ch = self._peek()

            if ch == "\t":
                start = self.pos()
                self._advance()
                self.bag.error(
                    "CANON-E0005",
                    "tab character in source",
                    self._span(start),
                    facts={"character": "U+0009"},
                    repairs=[Repair("replace-span", "replace the tab with spaces",
                                    "  ", self._span(start), 0.95)],
                    notes=["Canon has exactly one indentation character so that "
                           "canonical form is byte-identical across all authors."],
                )
                continue

            if ch in " \r\n":
                self._advance()
                continue

            if ch == "-" and self._peek(1) == "-":
                self._comment()
                continue

            if ch == '"':
                self._text()
                continue

            if ch.isdigit():
                self._number()
                continue

            if ch.isalpha() or ch == "_":
                self._word()
                continue

            if self._operator():
                continue

            if ch in PUNCTUATION:
                start = self.pos()
                self._advance()
                self._emit(T.PUNCT, ch, start)
                continue

            start = self.pos()
            bad = self._advance()
            self.bag.error(
                "CANON-E0001",
                f"unexpected character {bad!r}",
                self._span(start),
                facts={"character": bad, "codepoint": f"U+{ord(bad):04X}"},
                repairs=[Repair("replace-span", "remove the character", "",
                                self._span(start), 0.6)],
            )

        self._emit(T.EOF, "", self.pos())
        return self.tokens

    # -- lexemes ------------------------------------------------------------

    def _comment(self):
        start = self.pos()
        self._advance(2)  # --
        is_doc = self._peek() == "-"
        if is_doc:
            self._advance()
        body_start = self.i
        while not self._at_end() and self._peek() != "\n":
            self._advance()
        body = self.src[body_start:self.i].strip()
        if is_doc:
            self._emit(T.DOC, body, start, body)
        # ordinary comments are discarded: they do not survive into canonical
        # form. Documentation that matters goes in `doc` / `intent` clauses,
        # which are part of the hashed definition.

    def _text(self):
        start = self.pos()
        self._advance()  # opening quote
        # triple-quoted block text: """ ... """
        if self._peek() == '"' and self._peek(1) == '"':
            self._advance(2)
            buf = []
            while True:
                if self._at_end():
                    self.bag.error("CANON-E0002", "unterminated block text literal",
                                   self._span(start),
                                   repairs=[Repair("insert-after", 'close it with """',
                                                   '"""', self._span(start), 0.8)])
                    break
                if self._peek() == '"' and self._peek(1) == '"' and self._peek(2) == '"':
                    self._advance(3)
                    break
                buf.append(self._advance())
            raw = "".join(buf)
            self._emit(T.TEXT, raw, start, _dedent_block(raw))
            return

        buf = []
        while True:
            if self._at_end() or self._peek() == "\n":
                self.bag.error(
                    "CANON-E0002", "unterminated text literal", self._span(start),
                    repairs=[Repair("insert-after", "close the literal", '"',
                                    self._span(start), 0.85)],
                    notes=["Single-quoted text may not span lines. "
                           'Use a """block""" literal for multi-line text.'],
                )
                break
            ch = self._peek()
            if ch == "\\":
                self._advance()
                esc = self._advance()
                if esc == "u":
                    hexs = ""
                    while len(hexs) < 4 and self._peek() in "0123456789abcdefABCDEF":
                        hexs += self._advance()
                    if len(hexs) == 4:
                        buf.append(chr(int(hexs, 16)))
                    else:
                        self.bag.error("CANON-E0002", "malformed \\u escape",
                                       self._span(start),
                                       facts={"escape": "\\u" + hexs})
                elif esc in ESCAPES:
                    buf.append(ESCAPES[esc])
                else:
                    self.bag.error(
                        "CANON-E0002", f"unknown escape sequence \\{esc}",
                        self._span(start),
                        facts={"escape": f"\\{esc}",
                               "supported": sorted(ESCAPES.keys()) + ["u"]},
                    )
                continue
            if ch == '"':
                self._advance()
                break
            buf.append(self._advance())
        self._emit(T.TEXT, "".join(buf), start, "".join(buf))

    def _number(self):
        start = self.pos()
        buf = []
        seen_dot = False
        while not self._at_end():
            ch = self._peek()
            if ch.isdigit():
                buf.append(self._advance())
            elif ch == "_" and buf:
                self._advance()  # digit separators are erased in canonical form
            elif ch == "." and not seen_dot and self._peek(1).isdigit():
                seen_dot = True
                buf.append(self._advance())
            else:
                break
        text = "".join(buf)
        span = self._span(start)
        # A trailing alphabetic run is an error, not an implicit suffix.
        if self._peek().isalpha():
            junk = ""
            while self._peek().isalnum() or self._peek() == "_":
                junk += self._advance()
            self.bag.error(
                "CANON-E0003", f"invalid numeric literal {text}{junk!r}", span,
                facts={"literal": text + junk},
                notes=["Canon has no numeric suffixes. Use `Dec` or `Int` "
                       "annotations for the intended type."],
            )
            return
        if seen_dot:
            self._emit(T.DEC, text, start, Decimal(text))
        else:
            self._emit(T.INT, text, start, int(text))

    def _word(self):
        start = self.pos()
        buf = []
        while not self._at_end() and (self._peek().isalnum() or self._peek() == "_"):
            buf.append(self._advance())
        word = "".join(buf)
        if word in self.keywords:
            self._emit(T.KEYWORD, word, start)
        elif word[0].isupper():
            self._emit(T.UPPER, word, start)
        else:
            self._emit(T.NAME, word, start)

    def _operator(self) -> bool:
        for op in OPERATORS:
            if self.src.startswith(op, self.i):
                # `--` was already handled as a comment before we get here
                start = self.pos()
                self._advance(len(op))
                self._emit(T.OP, op, start)
                return True
        return False


def _dedent_block(raw: str) -> str:
    """Strip the common leading indentation from a block literal."""
    lines = raw.split("\n")
    if lines and not lines[0].strip():
        lines = lines[1:]
    if lines and not lines[-1].strip():
        lines = lines[:-1]
    indents = [len(l) - len(l.lstrip()) for l in lines if l.strip()]
    cut = min(indents) if indents else 0
    return "\n".join(l[cut:] if len(l) >= cut else l.lstrip() for l in lines)


def lex(source: str, filename: str = "<memory>", keywords=None):
    """Convenience: returns (tokens, bag)."""
    lx = Lexer(source, filename, keywords)
    toks = lx.run()
    return toks, lx.bag
