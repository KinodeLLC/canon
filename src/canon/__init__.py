"""
canon, the core language of the kinode stack.

what you get from here

    parse(source, filename)         -> (Module, Bag)
    check(modules)                  -> CheckResult
    format_module(module)           -> canonical source text
    Hasher().add_modules(modules)   -> {qualname: DefInfo}

three version numbers tracked separately, since they move for different reasons
and break different things. `__version__` is the package release,
`LANGUAGE_VERSION` is the surface syntax and semantics, and `IR_VERSION` is the
core ir and its canonical encoding.

changing `IR_VERSION` invalidates every content hash anybody has stored, so it
is always a breaking release no matter how small the change looks.
"""

__version__ = "0.1.0"
LANGUAGE_VERSION = "0.1"

from .canonical import (  # noqa: E402
    IR_VERSION,
    DefInfo,
    Hasher,
    Printer,
    digest,
    encode_decl,
    format_module,
    hash_decl,
    short,
)
from .diagnostics import (  # noqa: E402
    Bag,
    CanonError,
    Diagnostic,
    Pos,
    Repair,
    Severity,
    Span,
)
from .parser import Parser, parse  # noqa: E402
from .lexer import Lexer, lex  # noqa: E402

__all__ = [
    "__version__", "LANGUAGE_VERSION", "IR_VERSION",
    "parse", "Parser", "lex", "Lexer",
    "format_module", "Printer", "encode_decl", "hash_decl", "digest", "short",
    "Hasher", "DefInfo",
    "Bag", "CanonError", "Diagnostic", "Pos", "Repair", "Severity", "Span",
]
