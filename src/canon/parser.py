"""
Canon parser.

Recursive descent with precedence climbing. The grammar is deliberately
unambiguous with at most one token of lookahead, which keeps generated code
predictable and makes partial-parse recovery practical.

Two grammar choices are worth noting because they differ from most languages:

  1. Statements inside a block are always introduced by a keyword
     (`let`, `do`, `assert`, `abort`). The block's final item is the only bare
     expression. This removes any need for statement terminators or
     newline-sensitivity, so whitespace is purely cosmetic and the canonical
     printer has complete freedom over layout.

  2. `if` uses `then`/`else` rather than braces, so a brace after an expression
     always means a record literal or a match body and never a block. That
     removes the usual ambiguity between `match x { ... }` and a record
     literal, without needing a parser mode flag anywhere except match
     scrutinees.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Optional

from . import ast as A
from .diagnostics import Bag, Repair, Span
from .lexer import KEYWORDS, Lexer, T, Token


class Parser:
    def __init__(self, tokens: list, source: str = "", filename: str = "<memory>",
                 bag: Optional[Bag] = None, language: str = "canon"):
        self.toks = [t for t in tokens]
        self.src = source
        self.file = filename
        self.bag = bag if bag is not None else Bag()
        self.k = 0
        self.language = language
        self.pending_doc: list = []

    # ---------------------------------------------------------------- cursor

    @property
    def cur(self) -> Token:
        return self.toks[self.k]

    def at(self, n: int = 0) -> Token:
        j = min(self.k + n, len(self.toks) - 1)
        return self.toks[j]

    def next(self) -> Token:
        t = self.toks[self.k]
        if t.kind != T.EOF:
            self.k += 1
        return t

    def skip_docs(self) -> str:
        docs = []
        while self.cur.kind == T.DOC:
            docs.append(self.next().value)
        return "\n".join(docs)

    def eat_kw(self, *kws) -> Optional[Token]:
        if self.cur.is_kw(*kws):
            return self.next()
        return None

    def eat_punct(self, *ps) -> Optional[Token]:
        if self.cur.is_punct(*ps):
            return self.next()
        return None

    def eat_op(self, *ops) -> Optional[Token]:
        if self.cur.is_op(*ops):
            return self.next()
        return None

    # Contextual keywords lex as ordinary names. They are recognised by
    # position, so they stay usable as identifiers everywhere else.

    def at_ctx(self, *names) -> bool:
        return self.cur.kind == T.NAME and self.cur.value in names

    def eat_ctx(self, *names) -> Optional[Token]:
        if self.at_ctx(*names):
            return self.next()
        return None

    # ------------------------------------------------------------- errors

    def err(self, code: str, msg: str, tok: Optional[Token] = None, **kw):
        tok = tok or self.cur
        return self.bag.error(code, msg, tok.span, **kw)

    def expect_punct(self, p: str, what: str = "") -> Optional[Token]:
        t = self.eat_punct(p)
        if t is None:
            self.err(
                "CANON-E0101",
                f"expected {p!r}" + (f" {what}" if what else ""),
                facts={"expected": p, "found": self.cur.value or self.cur.kind},
                repairs=[Repair("insert-before", f"insert {p!r}", p,
                                self.cur.span, 0.75)],
            )
        return t

    def expect_op(self, op: str, what: str = "") -> Optional[Token]:
        t = self.eat_op(op)
        if t is None:
            self.err(
                "CANON-E0101",
                f"expected {op!r}" + (f" {what}" if what else ""),
                facts={"expected": op, "found": self.cur.value or self.cur.kind},
                repairs=[Repair("insert-before", f"insert {op!r}", op,
                                self.cur.span, 0.75)],
            )
        return t

    def expect_name(self, what: str = "a name") -> str:
        if self.cur.kind == T.NAME:
            return self.next().value
        self.err("CANON-E0101", f"expected {what}",
                 facts={"expected": "lowercase identifier",
                        "found": self.cur.value or self.cur.kind})
        return "_"

    def expect_upper(self, what: str = "a type name") -> str:
        if self.cur.kind == T.UPPER:
            return self.next().value
        self.err("CANON-E0101", f"expected {what}",
                 facts={"expected": "UpperCamelCase identifier",
                        "found": self.cur.value or self.cur.kind})
        return "Unknown"

    def span_from(self, start: Token) -> Span:
        end = self.toks[max(0, self.k - 1)]
        return Span(start.span.start, end.span.end)

    def sync_to_decl(self):
        """Error recovery: skip forward to something that can start a declaration."""
        depth = 0
        while self.cur.kind != T.EOF:
            if self.cur.is_punct("{", "(", "["):
                depth += 1
            elif self.cur.is_punct("}", ")", "]"):
                if depth == 0:
                    return
                depth -= 1
            elif depth == 0 and self.cur.is_kw(
                    "fn", "record", "enum", "alias", "effect", "const", "test",
                    "workflow", "decision", "schema_def", "pipeline", "resource",
                    "policy", "goal", "import"):
                return
            self.next()

    # ======================================================== module

    def parse_module(self) -> A.Module:
        start = self.cur
        doc = self.skip_docs()
        mod = A.Module(language=self.language, source_file=self.file, doc=doc)

        if not self.cur.is_kw("module"):
            self.err("CANON-E0107", "file must begin with a module header",
                     facts={"expected": "module <name>"},
                     repairs=[Repair("insert-before", "add a module header",
                                     "module main\n\n", self.cur.span, 0.7)])
        else:
            self.next()
            mod.name = self.parse_qualname()

        while self.cur.is_kw("import"):
            mod.imports.append(self.parse_import())

        while self.cur.kind != T.EOF:
            before = self.k
            d = self.parse_decl()
            if d is not None:
                mod.decls.append(d)
            if self.k == before:
                self.err("CANON-E0102", "expected a declaration",
                         facts={"found": self.cur.value or self.cur.kind})
                self.next()
                self.sync_to_decl()

        mod.span = self.span_from(start)
        return mod

    def parse_qualname(self) -> str:
        parts = []
        if self.cur.kind in (T.NAME, T.UPPER):
            parts.append(self.next().value)
        else:
            self.err("CANON-E0101", "expected a module name")
            return "unknown"
        while self.cur.is_punct(".") and self.at(1).kind in (T.NAME, T.UPPER):
            self.next()
            parts.append(self.next().value)
        return ".".join(parts)

    def parse_import(self) -> A.ImportDecl:
        start = self.next()  # import
        imp = A.ImportDecl(module=self.parse_qualname())
        if self.eat_kw("as"):
            imp.alias = self.expect_name("an import alias")
        if self.cur.is_punct("{"):
            self.next()
            while not self.cur.is_punct("}") and self.cur.kind != T.EOF:
                if self.cur.kind in (T.NAME, T.UPPER):
                    imp.names.append(self.next().value)
                else:
                    self.err("CANON-E0101", "expected an imported name")
                    self.next()
                self.eat_punct(",")
            self.expect_punct("}", "to close the import list")
        imp.span = self.span_from(start)
        return imp

    # ==================================================== declarations

    def parse_decl(self):
        doc = self.skip_docs()
        if self.cur.is_kw("fn"):
            return self.parse_fn(doc)
        if self.cur.is_kw("record"):
            return self.parse_record(doc)
        if self.cur.is_kw("enum"):
            return self.parse_enum(doc)
        if self.cur.is_kw("alias"):
            return self.parse_alias(doc)
        if self.cur.is_kw("effect"):
            return self.parse_effect(doc)
        if self.cur.is_kw("const"):
            return self.parse_const(doc)
        if self.cur.is_kw("test"):
            return self.parse_test(doc)
        if self.cur.kind == T.EOF:
            return None
        return None

    def parse_type_params(self) -> list:
        out = []
        if self.cur.is_op("<"):
            self.next()
            while not self.cur.is_op(">") and self.cur.kind != T.EOF:
                out.append(self.expect_upper("a type parameter"))
                self.eat_punct(",")
            self.expect_op(">", "to close the type parameter list")
        return out

    def parse_params(self) -> list:
        out = []
        self.expect_punct("(", "to start the parameter list")
        while not self.cur.is_punct(")") and self.cur.kind != T.EOF:
            pdoc = self.skip_docs()
            pstart = self.cur
            pname = self.expect_name("a parameter name")
            self.expect_punct(":", "before the parameter type")
            pty = self.parse_type()
            default = None
            if self.eat_op("="):
                default = self.parse_expr()
            p = A.Param(name=pname, ty=pty, default=default, doc=pdoc)
            p.span = self.span_from(pstart)
            out.append(p)
            if not self.eat_punct(","):
                break
        self.expect_punct(")", "to close the parameter list")
        return out

    def parse_fn(self, doc: str = "") -> A.FnDecl:
        start = self.next()  # fn
        fn = A.FnDecl(doc=doc, origin=self.language)
        fn.name = self.expect_name("a function name")
        fn.type_params = self.parse_type_params()
        fn.params = self.parse_params()
        self.expect_op("->", "before the result type")
        fn.result = self.parse_type()
        self.parse_fn_clauses(fn)
        fn.body = self.parse_block()
        fn.span = self.span_from(start)
        return fn

    def parse_fn_clauses(self, fn: A.FnDecl):
        """Clauses may appear in any order; the canonical printer sorts them."""
        while True:
            if self.cur.is_kw("uses"):
                self.next()
                fn.uses.extend(self.parse_effect_refs())
            elif self.cur.is_kw("requires"):
                self.next()
                fn.requires.append(self.parse_expr())
            elif self.cur.is_kw("ensures"):
                self.next()
                fn.ensures.append(self.parse_expr())
            elif self.cur.is_kw("law"):
                self.next()
                fn.laws.append(self.parse_law_ref())
            elif self.cur.is_kw("cost"):
                self.next()
                fn.cost = self.parse_cost(fn.cost)
            elif self.cur.is_kw("intent"):
                self.next()
                fn.intent = self.parse_text_literal("an intent description")
            elif self.cur.is_kw("doc"):
                self.next()
                extra = self.parse_text_literal("documentation text")
                fn.doc = (fn.doc + "\n" + extra).strip() if fn.doc else extra
            elif self.cur.is_kw("recursive"):
                self.next()
                fn.recursive = True
            elif self.cur.is_kw("decreases"):
                self.next()
                fn.decreases = self.parse_expr()
                fn.recursive = True
            else:
                # Purity is stated as `law pure` rather than as a clause: it is
                # a property the verifier checks against the body, not a
                # declaration that silently discards the `uses` list.
                return

    def parse_text_literal(self, what: str) -> str:
        if self.cur.kind == T.TEXT:
            return self.next().payload
        self.err("CANON-E0101", f"expected {what} as a text literal",
                 facts={"found": self.cur.value or self.cur.kind})
        return ""

    def parse_effect_refs(self) -> list:
        refs = [self.parse_effect_ref()]
        while self.eat_punct(","):
            refs.append(self.parse_effect_ref())
        return refs

    def parse_effect_ref(self) -> A.EffectRef:
        start = self.cur
        eff = self.expect_name("an effect name")
        op = None
        if self.eat_punct("."):
            if self.cur.is_op("*"):
                self.next()
                op = None
            else:
                op = self.expect_name("an effect operation name")
        r = A.EffectRef(effect=eff, op=op)
        r.span = self.span_from(start)
        return r

    def parse_law_ref(self) -> A.LawRef:
        start = self.cur
        name = self.next().value if self.cur.kind in (T.NAME, T.UPPER) else "_"
        args = []
        if self.cur.is_punct("("):
            self.next()
            while not self.cur.is_punct(")") and self.cur.kind != T.EOF:
                args.append(self.parse_expr())
                if not self.eat_punct(","):
                    break
            self.expect_punct(")", "to close the law arguments")
        law = A.LawRef(name=name, args=args)
        law.span = self.span_from(start)
        return law

    def parse_cost(self, existing: Optional[A.Cost]) -> A.Cost:
        c = existing or A.Cost()
        c.span = self.cur.span
        while True:
            if self.cur.kind != T.NAME:
                break
            key = self.cur.value
            if key not in ("steps", "io", "tokens", "millis", "money"):
                break
            self.next()
            if self.cur.kind == T.INT:
                val = self.next().payload
            elif self.cur.kind == T.DEC:
                val = self.next().payload
            else:
                self.err("CANON-E0101", f"expected a number after cost key {key!r}")
                val = 0
            if key == "money":
                c.money = Decimal(val)
            else:
                setattr(c, key, int(val))
            if not self.eat_punct(","):
                break
        return c

    def parse_record(self, doc: str = "") -> A.RecordDecl:
        start = self.next()  # record
        r = A.RecordDecl(doc=doc)
        r.name = self.expect_upper("a record name")
        r.type_params = self.parse_type_params()
        self.expect_punct("{", "to open the record body")
        while not self.cur.is_punct("}") and self.cur.kind != T.EOF:
            fdoc = self.skip_docs()
            # `invariant` and `classify` are contextual: they only take effect
            # when not followed by ':', so a field may still be named either.
            if self.at_ctx("invariant") and not self.at(1).is_punct(":"):
                self.next()
                r.invariants.append(self.parse_expr())
                continue
            if self.cur.is_kw("intent"):
                self.next()
                r.intent = self.parse_text_literal("an intent description")
                continue
            if self.at_ctx("classify") and not self.at(1).is_punct(":"):
                self.next()
                fname = self.expect_name("a field name")
                cls = self.expect_name("a data classification")
                r.classification[fname] = cls
                continue
            fstart = self.cur
            fname = self.expect_name("a field name")
            self.expect_punct(":", "before the field type")
            fty = self.parse_type()
            # A field with a default may be omitted from a literal, which is
            # what makes adding a field to an existing record a non-breaking
            # change rather than an edit to every construction site.
            fdefault = self.parse_expr() if self.eat_op("=") else None
            p = A.Param(name=fname, ty=fty, doc=fdoc, default=fdefault)
            p.span = self.span_from(fstart)
            r.fields.append(p)
            self.eat_punct(",")
        self.expect_punct("}", "to close the record body")
        r.span = self.span_from(start)
        return r

    def parse_enum(self, doc: str = "") -> A.EnumDecl:
        start = self.next()  # enum
        e = A.EnumDecl(doc=doc)
        e.name = self.expect_upper("an enum name")
        e.type_params = self.parse_type_params()
        self.expect_punct("{", "to open the enum body")
        while not self.cur.is_punct("}") and self.cur.kind != T.EOF:
            vdoc = self.skip_docs()
            if self.cur.is_kw("intent"):
                self.next()
                e.intent = self.parse_text_literal("an intent description")
                continue
            if not self.eat_op("|"):
                self.err("CANON-E0101", "enum variants must be introduced by '|'",
                         repairs=[Repair("insert-before", "add '|'", "| ",
                                         self.cur.span, 0.9)])
                if self.cur.kind != T.UPPER:
                    self.next()
                    continue
            vstart = self.cur
            vname = self.expect_upper("a variant name")
            params = []
            if self.cur.is_punct("("):
                self.next()
                while not self.cur.is_punct(")") and self.cur.kind != T.EOF:
                    params.append(self.parse_type())
                    if not self.eat_punct(","):
                        break
                self.expect_punct(")", "to close the variant payload")
            v = A.EnumVariant(name=vname, params=params, doc=vdoc)
            v.span = self.span_from(vstart)
            e.variants.append(v)
        self.expect_punct("}", "to close the enum body")
        e.span = self.span_from(start)
        return e

    def parse_alias(self, doc: str = "") -> A.AliasDecl:
        start = self.next()  # alias
        a = A.AliasDecl(doc=doc)
        a.name = self.expect_upper("an alias name")
        a.type_params = self.parse_type_params()
        self.expect_op("=", "before the aliased type")
        a.target = self.parse_type()
        a.span = self.span_from(start)
        return a

    def parse_effect(self, doc: str = "") -> A.EffectDecl:
        start = self.next()  # effect
        e = A.EffectDecl(doc=doc)
        e.name = self.expect_name("an effect name")
        self.expect_punct("{", "to open the effect body")
        while not self.cur.is_punct("}") and self.cur.kind != T.EOF:
            odoc = self.skip_docs()
            # `idempotent` modifies the next operation; an operation may still
            # be called `idempotent`, in which case '(' follows it directly.
            idem = bool(self.at_ctx("idempotent") and self.at(1).kind == T.NAME
                        and self.next())
            ostart = self.cur
            oname = self.expect_name("an operation name")
            params = self.parse_params()
            self.expect_op("->", "before the operation result type")
            res = self.parse_type()
            op = A.EffectOp(name=oname, params=params, result=res,
                            doc=odoc, idempotent=idem)
            op.span = self.span_from(ostart)
            e.ops.append(op)
        self.expect_punct("}", "to close the effect body")
        e.span = self.span_from(start)
        return e

    def parse_const(self, doc: str = "") -> A.ConstDecl:
        start = self.next()  # const
        c = A.ConstDecl(doc=doc)
        c.name = self.expect_name("a constant name")
        self.expect_punct(":", "before the constant type")
        c.ty = self.parse_type()
        self.expect_op("=", "before the constant value")
        c.value = self.parse_expr()
        c.span = self.span_from(start)
        return c

    def parse_test(self, doc: str = "") -> A.TestDecl:
        start = self.next()  # test
        t = A.TestDecl(doc=doc)
        t.name = self.parse_text_literal("a test description")
        t.body = self.parse_block()
        t.span = self.span_from(start)
        return t

    # =========================================================== types

    def parse_type(self) -> A.TypeExpr:
        start = self.cur

        if self.cur.is_punct("{"):
            self.next()
            fields = []
            while not self.cur.is_punct("}") and self.cur.kind != T.EOF:
                fname = self.expect_name("a field name")
                self.expect_punct(":", "before the field type")
                fields.append((fname, self.parse_type()))
                if not self.eat_punct(","):
                    break
            self.expect_punct("}", "to close the record type")
            t = A.TRecord(fields=fields)
            t.span = self.span_from(start)
            return t

        if self.cur.kind == T.UPPER and self.cur.value == "Fn":
            self.next()
            params = []
            self.expect_punct("(", "to start the function type parameters")
            while not self.cur.is_punct(")") and self.cur.kind != T.EOF:
                params.append(self.parse_type())
                if not self.eat_punct(","):
                    break
            self.expect_punct(")", "to close the function type parameters")
            self.expect_op("->", "before the function result type")
            res = self.parse_type()
            effects = []
            if self.eat_kw("uses"):
                effects = self.parse_effect_refs()
            t = A.TFn(params=params, result=res, effects=effects)
            t.span = self.span_from(start)
            return t

        if self.cur.kind == T.UPPER:
            name = self.next().value
            args = []
            if self.cur.is_op("<"):
                self.next()
                while not self.cur.is_op(">") and self.cur.kind != T.EOF:
                    args.append(self.parse_type())
                    if not self.eat_punct(","):
                        break
                self.expect_op(">", "to close the type arguments")
            t = A.TName(name=name, args=args)
            t.span = self.span_from(start)
            return t

        if self.cur.is_punct("(") and self.at(1).is_punct(")"):
            self.next()
            self.next()
            t = A.TName(name="Unit")
            t.span = self.span_from(start)
            return t

        self.err("CANON-E0104", "expected a type",
                 facts={"found": self.cur.value or self.cur.kind})
        self.next()
        t = A.TName(name="Unknown")
        t.span = self.span_from(start)
        return t

    # ===================================================== expressions

    def parse_block(self) -> A.Block:
        start = self.cur
        blk = A.Block()
        if not self.expect_punct("{", "to open a block"):
            blk.result = A.Lit(value=None, lit_kind="unit")
            blk.span = self.span_from(start)
            return blk

        while self.cur.kind != T.EOF and not self.cur.is_punct("}"):
            self.skip_docs()
            if self.cur.is_kw("let"):
                sstart = self.next()
                name = self.expect_name("a binding name")
                ty = None
                if self.eat_punct(":"):
                    ty = self.parse_type()
                self.expect_op("=", "before the bound value")
                val = self.parse_expr()
                s = A.SLet(name=name, ty=ty, value=val)
                s.span = self.span_from(sstart)
                blk.stmts.append(s)
                continue
            if self.cur.is_kw("do"):
                sstart = self.next()
                s = A.SExpr(value=self.parse_expr())
                s.span = self.span_from(sstart)
                blk.stmts.append(s)
                continue
            if self.cur.is_kw("assert"):
                sstart = self.next()
                cond = self.parse_expr()
                msg = ""
                if self.eat_punct(","):
                    msg = self.parse_text_literal("an assertion message")
                s = A.SAssert(cond=cond, message=msg)
                s.span = self.span_from(sstart)
                blk.stmts.append(s)
                continue
            # anything else is the block's result expression
            blk.result = self.parse_expr()
            break

        self.expect_punct("}", "to close the block")
        if blk.result is None:
            blk.result = A.Lit(value=None, lit_kind="unit")
        blk.span = self.span_from(start)
        return blk

    def parse_expr(self, no_record: bool = False) -> A.Expr:
        start = self.cur

        if self.cur.is_kw("let"):
            self.next()
            name = self.expect_name("a binding name")
            ty = None
            if self.eat_punct(":"):
                ty = self.parse_type()
            self.expect_op("=", "before the bound value")
            value = self.parse_expr()
            if not self.eat_kw("in"):
                self.err("CANON-E0101",
                         "a `let` expression requires `in`; inside a block use a "
                         "`let` statement instead",
                         facts={"found": self.cur.value or self.cur.kind},
                         repairs=[Repair("insert-before", "add `in`", " in ",
                                         self.cur.span, 0.8)])
            body = self.parse_expr(no_record)
            e = A.Let(name=name, ty=ty, value=value, body=body)
            e.span = self.span_from(start)
            return e

        if self.cur.is_kw("if"):
            self.next()
            cond = self.parse_expr(no_record=True)
            if not self.eat_kw("then"):
                self.err("CANON-E0101", "expected `then` after the condition",
                         repairs=[Repair("insert-before", "add `then`", " then ",
                                         self.cur.span, 0.85)])
            then = self.parse_expr(no_record)
            if not self.eat_kw("else"):
                self.err("CANON-E0307",
                         "`if` must have an `else`; every expression has a value",
                         facts={"missing": "else branch"},
                         repairs=[Repair("insert-before", "add an else branch",
                                         " else ", self.cur.span, 0.6)])
                other = A.Lit(value=None, lit_kind="unit")
            else:
                other = self.parse_expr(no_record)
            e = A.If(cond=cond, then=then, otherwise=other)
            e.span = self.span_from(start)
            return e

        if self.cur.is_kw("match"):
            self.next()
            scrut = self.parse_expr(no_record=True)
            e = A.Match(scrutinee=scrut, arms=self.parse_match_arms())
            e.span = self.span_from(start)
            return e

        if self.cur.is_kw("ask"):
            return self.parse_ask()

        if self.cur.is_kw("fn"):
            self.next()
            params = self.parse_params()
            self.expect_op("=>", "before the lambda body")
            body = self.parse_expr(no_record)
            e = A.Lambda(params=params, body=body)
            e.span = self.span_from(start)
            return e

        return self.parse_binary(0, no_record)

    def parse_match_arms(self) -> list:
        arms = []
        self.expect_punct("{", "to open the match body")
        while not self.cur.is_punct("}") and self.cur.kind != T.EOF:
            self.skip_docs()
            astart = self.cur
            if not self.eat_kw("case"):
                self.err("CANON-E0101", "match arms must begin with `case`",
                         repairs=[Repair("insert-before", "add `case`", "case ",
                                         self.cur.span, 0.9)])
                if self.cur.is_punct("}"):
                    break
            pat = self.parse_pattern()
            guard = None
            if self.eat_kw("if"):
                guard = self.parse_expr(no_record=True)
            self.expect_op("=>", "before the arm body")
            body = self.parse_expr()
            arm = A.MatchArm(pattern=pat, guard=guard, body=body)
            arm.span = self.span_from(astart)
            arms.append(arm)
            self.eat_punct(",")
        self.expect_punct("}", "to close the match body")
        if not arms:
            self.err("CANON-E0305", "match has no arms")
        return arms

    def parse_binary(self, min_prec: int, no_record: bool = False) -> A.Expr:
        left = self.parse_unary(no_record)
        while True:
            tok = self.cur
            opname = None
            if tok.kind == T.OP and tok.value in A.BINARY_OPS:
                opname = tok.value
            elif tok.is_kw("and", "or"):
                opname = tok.value
            if opname is None:
                return left
            prec, assoc = A.BINARY_OPS[opname]
            if prec < min_prec:
                return left
            self.next()
            nxt = prec + 1 if assoc == "left" else prec
            if assoc == "none":
                nxt = prec + 1
            right = self.parse_binary(nxt, no_record)
            canon = A.OP_CANONICAL.get(opname, opname)
            node = A.Binary(op=canon, left=left, right=right)
            node.span = left.span.merge(right.span)
            left = node

    def parse_unary(self, no_record: bool = False) -> A.Expr:
        start = self.cur
        if self.cur.is_op("-") or self.cur.is_kw("not") or self.cur.is_op("!"):
            op = self.next().value
            op = A.OP_CANONICAL.get(op, op)
            operand = self.parse_unary(no_record)
            e = A.Unary(op=op, operand=operand)
            e.span = self.span_from(start)
            return e
        return self.parse_postfix(no_record)

    def parse_postfix(self, no_record: bool = False) -> A.Expr:
        start = self.cur
        e = self.parse_primary(no_record)
        while True:
            if self.cur.is_punct("."):
                self.next()
                if self.cur.kind not in (T.NAME, T.UPPER):
                    self.err("CANON-E0101", "expected a field or operation name")
                    break
                name = self.next().value
                # `effect.op(args)` becomes a Perform node when the target is a
                # bare lowercase name; resolution decides whether it really is
                # an effect, a module, or a record field.
                if (isinstance(e, A.Var) and self.cur.is_punct("(")
                        and e.name.islower()):
                    args = self.parse_call_args()
                    node = A.Perform(effect=e.name, op=name, args=args)
                    node.span = self.span_from(start)
                    e = node
                    continue
                node = A.Field(target=e, name=name)
                node.span = self.span_from(start)
                e = node
                continue
            if self.cur.is_punct("("):
                args = self.parse_call_args()
                node = A.Call(fn=e, args=args)
                node.span = self.span_from(start)
                e = node
                continue
            if self.cur.is_op("?"):
                self.next()
                node = A.Try(operand=e)
                node.span = self.span_from(start)
                e = node
                continue
            return e
        return e

    def parse_call_args(self) -> list:
        args = []
        self.expect_punct("(", "to start the argument list")
        while not self.cur.is_punct(")") and self.cur.kind != T.EOF:
            args.append(self.parse_expr())
            if not self.eat_punct(","):
                break
        self.expect_punct(")", "to close the argument list")
        return args

    def parse_primary(self, no_record: bool = False) -> A.Expr:
        start = self.cur
        tok = self.cur

        if tok.kind == T.INT:
            self.next()
            e = A.Lit(value=tok.payload, lit_kind="int")
            e.span = tok.span
            return e
        if tok.kind == T.DEC:
            self.next()
            e = A.Lit(value=tok.payload, lit_kind="dec")
            e.span = tok.span
            return e
        if tok.kind == T.TEXT:
            self.next()
            e = A.Lit(value=tok.payload, lit_kind="text")
            e.span = tok.span
            return e
        if tok.is_kw("true", "false"):
            self.next()
            e = A.Lit(value=(tok.value == "true"), lit_kind="bool")
            e.span = tok.span
            return e
        if tok.is_kw("result"):
            self.next()
            e = A.Var(name="result")
            e.span = tok.span
            return e
        if tok.is_kw("old"):
            self.next()
            args = self.parse_call_args()
            e = A.Call(fn=A.Var(name="old"), args=args)
            e.span = self.span_from(start)
            return e
        if tok.is_kw("abort"):
            self.next()
            args = self.parse_call_args() if self.cur.is_punct("(") else []
            e = A.Call(fn=A.Var(name="abort"), args=args)
            e.span = self.span_from(start)
            return e

        if tok.is_punct("("):
            self.next()
            if self.cur.is_punct(")"):
                self.next()
                e = A.Lit(value=None, lit_kind="unit")
                e.span = self.span_from(start)
                return e
            inner = self.parse_expr()
            self.expect_punct(")", "to close the parenthesised expression")
            return inner

        if tok.is_punct("["):
            self.next()
            items = []
            while not self.cur.is_punct("]") and self.cur.kind != T.EOF:
                items.append(self.parse_expr())
                if not self.eat_punct(","):
                    break
            self.expect_punct("]", "to close the list literal")
            e = A.ListLit(items=items)
            e.span = self.span_from(start)
            return e

        if tok.is_punct("{"):
            return self.parse_block()

        if tok.kind == T.UPPER:
            name = self.next().value
            if self.cur.is_punct("{") and not no_record:
                self.next()
                base = None
                fields = []
                while not self.cur.is_punct("}") and self.cur.kind != T.EOF:
                    if self.cur.is_op(".."):
                        self.next()
                        base = self.parse_expr()
                        self.eat_punct(",")
                        continue
                    fname = self.expect_name("a field name")
                    self.expect_punct(":", "before the field value")
                    fields.append((fname, self.parse_expr()))
                    if not self.eat_punct(","):
                        break
                self.expect_punct("}", "to close the record literal")
                e = A.RecordLit(type_name=name, fields=fields, base=base)
                e.span = self.span_from(start)
                return e
            if self.cur.is_punct("("):
                args = self.parse_call_args()
                e = A.CtorCall(name=name, args=args)
                e.span = self.span_from(start)
                return e
            if self.cur.is_punct(".") and self.at(1).kind in (T.NAME, T.UPPER):
                # qualified module reference: Payments.refund
                self.next()
                member = self.next().value
                e = A.QualVar(module=name, name=member)
                e.span = self.span_from(start)
                return e
            e = A.CtorCall(name=name, args=[])
            e.span = self.span_from(start)
            return e

        if tok.kind == T.NAME:
            name = self.next().value
            e = A.Var(name=name)
            e.span = tok.span
            return e

        self.err("CANON-E0103", "expected an expression",
                 facts={"found": tok.value or tok.kind})
        self.next()
        e = A.Lit(value=None, lit_kind="unit")
        e.span = self.span_from(start)
        return e

    # ------------------------------------------------------------ patterns

    def parse_pattern(self) -> A.Pattern:
        start = self.cur
        tok = self.cur

        if tok.kind == T.NAME and tok.value == "_":
            self.next()
            p = A.PWild()
            p.span = tok.span
            return p

        if tok.kind == T.NAME:
            self.next()
            p = A.PVar(name=tok.value)
            p.span = tok.span
            return p

        if tok.kind == T.INT:
            self.next()
            p = A.PLit(value=tok.payload, lit_kind="int")
            p.span = tok.span
            return p
        if tok.kind == T.DEC:
            self.next()
            p = A.PLit(value=tok.payload, lit_kind="dec")
            p.span = tok.span
            return p
        if tok.kind == T.TEXT:
            self.next()
            p = A.PLit(value=tok.payload, lit_kind="text")
            p.span = tok.span
            return p
        if tok.is_kw("true", "false"):
            self.next()
            p = A.PLit(value=(tok.value == "true"), lit_kind="bool")
            p.span = tok.span
            return p

        if tok.is_punct("("):
            self.next()
            if self.cur.is_punct(")"):
                self.next()
                p = A.PLit(value=None, lit_kind="unit")
                p.span = self.span_from(start)
                return p
            inner = self.parse_pattern()
            self.expect_punct(")", "to close the pattern")
            return inner

        if tok.is_punct("["):
            self.next()
            items = []
            rest = None
            while not self.cur.is_punct("]") and self.cur.kind != T.EOF:
                if self.cur.is_op(".."):
                    self.next()
                    rest = self.expect_name("a rest binding name")
                    break
                items.append(self.parse_pattern())
                if not self.eat_punct(","):
                    break
            self.expect_punct("]", "to close the list pattern")
            p = A.PList(items=items, rest=rest)
            p.span = self.span_from(start)
            return p

        if tok.kind == T.UPPER:
            name = self.next().value
            if self.cur.is_punct("{"):
                self.next()
                fields = []
                openp = False
                while not self.cur.is_punct("}") and self.cur.kind != T.EOF:
                    if self.cur.is_op(".."):
                        self.next()
                        openp = True
                        break
                    fname = self.expect_name("a field name")
                    if self.eat_punct(":"):
                        fields.append((fname, self.parse_pattern()))
                    else:
                        fields.append((fname, A.PVar(name=fname)))
                    if not self.eat_punct(","):
                        break
                self.expect_punct("}", "to close the record pattern")
                p = A.PRecord(type_name=name, fields=fields, open=openp)
                p.span = self.span_from(start)
                return p
            args = []
            if self.cur.is_punct("("):
                self.next()
                while not self.cur.is_punct(")") and self.cur.kind != T.EOF:
                    args.append(self.parse_pattern())
                    if not self.eat_punct(","):
                        break
                self.expect_punct(")", "to close the constructor pattern")
            p = A.PCtor(name=name, args=args)
            p.span = self.span_from(start)
            return p

        self.err("CANON-E0108", "expected a pattern",
                 facts={"found": tok.value or tok.kind})
        self.next()
        p = A.PWild()
        p.span = self.span_from(start)
        return p

    # ------------------------------------------------- the `ask` primitive

    def parse_ask(self) -> A.Ask:
        start = self.next()  # ask
        result_type = self.parse_type()
        if not self.eat_kw("from"):
            self.err("CANON-E0101", "expected `from <model>` after the ask type",
                     facts={"expected": "from"},
                     repairs=[Repair("insert-before", "name a model",
                                     " from claude.opus ", self.cur.span, 0.5)])
            model = "default"
        else:
            model = self.parse_model_name()

        spec = A.AskSpec()
        spec.span = self.cur.span
        self.expect_punct("{", "to open the ask body")
        while not self.cur.is_punct("}") and self.cur.kind != T.EOF:
            self.skip_docs()
            if self.at_ctx("system"):
                self.next()
                spec.system = self.parse_expr()
            elif self.at_ctx("input"):
                self.next()
                label = ""
                if self.cur.kind == T.NAME and self.at(1).is_punct(":"):
                    label = self.next().value
                    self.next()
                e = self.parse_expr()
                if not label:
                    label = e.name if isinstance(e, A.Var) else f"input{len(spec.inputs)}"
                spec.inputs.append((label, e))
            elif self.at_ctx("grounded_in"):
                self.next()
                spec.grounded_in.append(self.parse_expr())
            elif self.at_ctx("examples"):
                self.next()
                self.expect_punct("[", "to open the example list")
                while not self.cur.is_punct("]") and self.cur.kind != T.EOF:
                    self.expect_punct("(", "to open an example pair")
                    a = self.parse_expr()
                    self.expect_punct(",", "between example input and output")
                    b = self.parse_expr()
                    self.expect_punct(")", "to close an example pair")
                    spec.examples.append((a, b))
                    if not self.eat_punct(","):
                        break
                self.expect_punct("]", "to close the example list")
            elif self.at_ctx("temperature"):
                self.next()
                if self.cur.kind in (T.INT, T.DEC):
                    spec.temperature = Decimal(str(self.next().payload))
                else:
                    self.err("CANON-E0101", "temperature must be a number")
            elif self.at_ctx("retries"):
                self.next()
                if self.cur.kind == T.INT:
                    spec.retries = int(self.next().payload)
                else:
                    self.err("CANON-E0101", "retries must be an integer")
                if self.eat_ctx("on"):
                    while self.cur.kind == T.NAME:
                        spec.retry_on.append(self.next().value)
                        if not self.eat_punct(","):
                            break
            elif self.at_ctx("max_tokens"):
                self.next()
                if self.cur.kind == T.INT:
                    spec.max_tokens = int(self.next().payload)
                else:
                    self.err("CANON-E0101", "max_tokens must be an integer")
            elif self.at_ctx("judge"):
                self.next()
                spec.judge = self.parse_expr()
            else:
                self.err("CANON-E0101", "unknown ask setting",
                         facts={"found": self.cur.value or self.cur.kind,
                                "known": ["system", "input", "grounded_in",
                                          "examples", "temperature", "retries",
                                          "max_tokens", "judge"]})
                self.next()
        self.expect_punct("}", "to close the ask body")

        if spec.system is None:
            self.bag.warn("CANON-W0005",
                          "ask has no system instruction",
                          self.span_from(start),
                          facts={"model": model},
                          repairs=[Repair("manual",
                                          "add a `system \"...\"` line describing "
                                          "the task")])

        node = A.Ask(result_type=result_type, model=model, spec=spec)
        node.span = self.span_from(start)
        return node

    def parse_model_name(self) -> str:
        parts = []
        if self.cur.kind in (T.NAME, T.UPPER):
            parts.append(self.next().value)
        else:
            self.err("CANON-E0101", "expected a model name")
            return "default"
        while self.cur.is_punct(".") and self.at(1).kind in (T.NAME, T.UPPER, T.INT):
            self.next()
            parts.append(str(self.next().value))
        return ".".join(parts)


# --------------------------------------------------------------------------

def parse(source: str, filename: str = "<memory>", language: str = "canon"):
    """Lex and parse a Canon source file. Returns (Module, Bag)."""
    lx = Lexer(source, filename)
    toks = lx.run()
    p = Parser(toks, source, filename, lx.bag, language)
    mod = p.parse_module()
    return mod, p.bag
