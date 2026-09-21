"""
Canon: the core language of the Kinode stack.

Public surface:

    parse(source, filename)         -> (Module, Bag)
    check(modules)                  -> CheckResult
    format_module(module)           -> canonical source text
    Hasher().add_modules(modules)   -> {qualname: DefInfo}

Three version numbers are tracked separately because they change for different
reasons and have different blast radii:

    __version__         the package release
    LANGUAGE_VERSION    surface syntax and semantics
    IR_VERSION          the core IR and its canonical encoding

An IR_VERSION change invalidates every stored content hash, so it is always a
breaking release regardless of how small the change is.
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
