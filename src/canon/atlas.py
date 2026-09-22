"""
the semantic view of a codebase, and what an agent edits through.

an agent working off files and grep burns most of its context budget finding
things and most of its risk on not knowing what a change will hit. here the
codebase is a graph of definitions identified by hash and the questions that
matter are graph queries rather than searches.

show tells you what something is at whatever level of detail you asked for.
callers and calls tell you what depends on it and what it depends on. blast
gives you everything a change here could reach. caps gives you the capability
footprint of that call graph. unverified lists whatever has no contracts or no
intent. view hands back a projection that fits a token budget you name.

edits go through a transaction. you propose, it checks, you look at the blast
radius and the diff, then you commit or abort. a proposal that does not
typecheck or reaches outside what it was allowed to touch never becomes the
working state, so a failed edit costs nothing and leaves nothing behind.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from . import ast as A
from .canonical import DefInfo, Hasher, Printer, format_module, short
from .checker import CheckResult, check
from .diagnostics import Bag


# --------------------------------------------------------------------------
# Token estimation
# --------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """
    A deliberately rough token estimate.

    Atlas budgets context, and being approximately right is enough for that.
    Calling a real tokenizer per projection would cost more than the budget it
    is protecting.
    """
    return max(1, (len(text) + 3) // 4)


# --------------------------------------------------------------------------
# Node
# --------------------------------------------------------------------------

@dataclass
class Node:
    qualname: str
    name: str
    module: str
    kind: str
    hash: str
    local_hash: str
    origin: str = "canon"
    intent: str = ""
    doc: str = ""
    signature: str = ""
    effects: list = field(default_factory=list)
    transitive_effects: list = field(default_factory=list)
    models: list = field(default_factory=list)
    touches: list = field(default_factory=list)
    deps: list = field(default_factory=list)
    calls: list = field(default_factory=list)
    contracts: int = 0
    laws: list = field(default_factory=list)
    cost: dict = field(default_factory=dict)
    span: Optional[dict] = None
    decl: Any = field(default=None, repr=False)

    def summary(self) -> dict:
        return {"definition": self.qualname, "kind": self.kind,
                "hash": self.hash, "short": short(self.hash),
                "signature": self.signature, "intent": self.intent,
                "effects": self.effects, "language": self.origin}

    def to_json(self) -> dict:
        d = self.summary()
        d.update({"module": self.module, "local_hash": self.local_hash,
                  "transitive_effects": self.transitive_effects,
                  "models": self.models, "touches": self.touches,
                  "deps": self.deps, "calls": self.calls,
                  "contracts": self.contracts, "laws": self.laws,
                  "cost": self.cost, "doc": self.doc, "span": self.span})
        return d


# --------------------------------------------------------------------------
# Atlas
# --------------------------------------------------------------------------

class Atlas:
    def __init__(self, cr: CheckResult, defs: Optional[dict] = None,
                 sources: Optional[dict] = None):
        self.cr = cr
        self.env = cr.env
        self.sources = dict(sources or {})
        self.defs: dict = defs or Hasher().add_modules(cr.modules)
        self.nodes: dict = {}
        self.by_hash: dict = {}
        self.reverse: dict = {}
        self._index()

    # ------------------------------------------------------------------

    @staticmethod
    def from_sources(sources: dict) -> "Atlas":
        """
        Parse and check a {filename: text} map, then index it.

        Each file is parsed with the language its extension names, so a
        workspace holding several of the family's languages indexes into one
        graph.
        """
        from .sources import Workspace, load_text

        ws = Workspace()
        for name, text in sorted(sources.items()):
            load_text(text, name, ws)
        cr = check(ws.modules, ws.bag)
        return Atlas(cr, sources=sources)

    def _index(self):
        for qn, di in self.defs.items():
            fi = self.env.fns.get(qn)
            node = Node(
                qualname=qn, name=di.name, module=di.module, kind=di.kind,
                hash=di.hash, local_hash=di.local_hash, origin=di.origin,
                effects=sorted(di.effects), deps=sorted(di.deps),
                models=sorted(di.models), decl=di.decl)

            decl = di.decl
            node.intent = getattr(decl, "intent", "") or ""
            node.doc = getattr(decl, "doc", "") or ""
            node.span = decl.span.to_json() if getattr(decl, "span", None) else None

            if fi is not None:
                node.signature = _signature(fi)
                node.transitive_effects = sorted(fi.transitive)
                node.calls = sorted(fi.calls)
                node.touches = sorted(fi.touches)
                node.contracts = len(fi.decl.requires) + len(fi.decl.ensures)
                node.laws = [l.name for l in fi.decl.laws]
                node.cost = _cost(fi.decl.cost)
            else:
                node.signature = _decl_signature(decl)

            self.nodes[qn] = node
            self.by_hash[di.hash] = node

        for qn, node in self.nodes.items():
            for dep in node.deps:
                self.reverse.setdefault(dep, set()).add(qn)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def resolve(self, target: str) -> Optional[Node]:
        """Find a node by qualified name, bare name, or hash prefix."""
        if target in self.nodes:
            return self.nodes[target]
        if target in self.by_hash:
            return self.by_hash[target]
        if target.startswith("#"):
            for h, n in self.by_hash.items():
                if h.startswith(target):
                    return n
        matches = [n for qn, n in self.nodes.items()
                   if qn.rsplit(".", 1)[-1] == target]
        if len(matches) == 1:
            return matches[0]
        if matches:
            return matches[0]
        return None

    def search(self, text: str, limit: int = 20) -> list:
        """
        Find definitions by name, intent or documentation.

        Intent is searched because it is the closest thing in the codebase to a
        statement of purpose, and it is what an agent is usually matching
        against when it is looking for "the thing that does X".
        """
        needle = text.lower()
        scored = []
        for qn, n in self.nodes.items():
            score = 0
            if needle in n.name.lower():
                score += 10
            if needle == n.name.lower():
                score += 20
            if needle in qn.lower():
                score += 5
            if needle in (n.intent or "").lower():
                score += 8
            if needle in (n.doc or "").lower():
                score += 3
            if needle in (n.signature or "").lower():
                score += 2
            if score:
                scored.append((-score, qn, n))
        scored.sort(key=lambda x: (x[0], x[1]))
        return [n for _, _, n in scored[:limit]]

    # ------------------------------------------------------------------
    # Graph queries
    # ------------------------------------------------------------------

    def callers(self, target: str, transitive: bool = False) -> list:
        node = self.resolve(target)
        if node is None:
            return []
        if not transitive:
            return sorted(self.reverse.get(node.qualname, set()))
        seen = set()
        frontier = [node.qualname]
        while frontier:
            cur = frontier.pop()
            for dependent in self.reverse.get(cur, ()):
                if dependent not in seen:
                    seen.add(dependent)
                    frontier.append(dependent)
        return sorted(seen)

    def calls(self, target: str, transitive: bool = False) -> list:
        node = self.resolve(target)
        if node is None:
            return []
        if not transitive:
            return sorted(node.deps)
        seen = set()
        frontier = list(node.deps)
        while frontier:
            cur = frontier.pop()
            if cur in seen:
                continue
            seen.add(cur)
            n = self.nodes.get(cur)
            if n:
                frontier.extend(n.deps)
        return sorted(seen)

    def blast_radius(self, target: str) -> dict:
        """
        Everything a change to `target` could reach, and what that grants.

        This is the question a promotion decision turns on, so it is a single
        query rather than something to assemble from three others.
        """
        node = self.resolve(target)
        if node is None:
            return {"error": f"unknown definition {target!r}"}

        dependents = self.callers(node.qualname, transitive=True)
        caps = set(node.transitive_effects)
        touches = set(node.touches)
        models = set(node.models)
        for qn in dependents:
            n = self.nodes.get(qn)
            if n:
                caps |= set(n.transitive_effects)
                touches |= set(n.touches)
                models |= set(n.models)

        return {
            "definition": node.qualname,
            "hash": node.hash,
            "direct_dependents": sorted(self.reverse.get(node.qualname, set())),
            "transitive_dependents": dependents,
            "count": len(dependents),
            "capabilities": sorted(caps),
            "data_classifications": sorted(touches),
            "models": sorted(models),
            "modules": sorted({self.nodes[q].module for q in dependents
                               if q in self.nodes} | {node.module}),
        }

    def capabilities(self, target: str) -> dict:
        node = self.resolve(target)
        if node is None:
            return {"error": f"unknown definition {target!r}"}
        by_source = {}
        for qn in [node.qualname] + self.calls(node.qualname, transitive=True):
            n = self.nodes.get(qn)
            if n and n.effects:
                by_source[qn] = n.effects
        return {"definition": node.qualname,
                "declared": node.effects,
                "transitive": node.transitive_effects,
                "granted_by": by_source,
                "models": node.models,
                "data_classifications": node.touches}

    def unverified(self) -> dict:
        """Definitions the verifier cannot say anything useful about."""
        no_contract, no_intent, no_laws = [], [], []
        for qn, n in self.nodes.items():
            if n.kind != "fn":
                continue
            if not n.contracts:
                no_contract.append(qn)
            if not n.intent:
                no_intent.append(qn)
            if not n.laws:
                no_laws.append(qn)
        return {"without_contracts": sorted(no_contract),
                "without_intent": sorted(no_intent),
                "without_laws": sorted(no_laws),
                "total_functions": sum(1 for n in self.nodes.values()
                                       if n.kind == "fn")}

    def effect_users(self, effect_key: str) -> list:
        """Every definition that can reach a given effect operation."""
        out = []
        for qn, n in self.nodes.items():
            for k in n.transitive_effects:
                if k == effect_key or (effect_key.endswith(".*")
                                       and k.startswith(effect_key[:-1])):
                    out.append(qn)
                    break
        return sorted(out)

    def stats(self) -> dict:
        kinds = {}
        langs = {}
        for n in self.nodes.values():
            kinds[n.kind] = kinds.get(n.kind, 0) + 1
            langs[n.origin] = langs.get(n.origin, 0) + 1
        effects = set()
        for n in self.nodes.values():
            effects |= set(n.transitive_effects)
        return {"definitions": len(self.nodes), "by_kind": kinds,
                "by_language": langs, "modules": sorted(self.env.modules),
                "effects_in_use": sorted(effects),
                "unverified": self.unverified()}

    # ------------------------------------------------------------------
    # Projections
    # ------------------------------------------------------------------

    def render(self, target, detail: str = "full") -> str:
        """
        Render one definition at a chosen level of detail.

        `name`      the qualified name alone
        `signature` name, type and effects
        `contract`  signature plus intent, contracts and laws, no body
        `full`      the canonical source of the definition
        """
        node = target if isinstance(target, Node) else self.resolve(target)
        if node is None:
            return ""

        if detail == "name":
            return node.qualname

        if detail == "signature":
            return f"{node.qualname}: {node.signature}"

        if detail == "contract":
            p = Printer()
            lines = [f"{node.qualname}: {node.signature}"]
            if node.intent:
                lines.append(f'  intent "{node.intent}"')
            decl = node.decl
            if isinstance(decl, A.FnDecl):
                for r in decl.requires:
                    lines.append(f"  requires {p.expr(r)}")
                for e in decl.ensures:
                    lines.append(f"  ensures {p.expr(e)}")
                for l in decl.laws:
                    lines.append(f"  law {l.name}")
            if node.cost:
                lines.append("  cost " + ", ".join(
                    f"{k} {v}" for k, v in sorted(node.cost.items())))
            return "\n".join(lines)

        p = Printer()
        p.decl(node.decl)
        return p.render().rstrip()

    def view(self, focus, budget: int = 4000,
             include_dependents: bool = True) -> dict:
        """
        A projection of the codebase sized to a token budget.

        Allocation is by usefulness per token: the focus definitions in full,
        then the contracts of what they call (enough to call them correctly
        without reading them), then the signatures of what calls them (enough
        to know what a change would affect). Whatever does not fit is reported
        as omitted rather than silently dropped -- an agent that does not know
        its view is partial will reason as though it is complete.
        """
        if isinstance(focus, str):
            focus = [focus]
        focus_nodes = [n for n in (self.resolve(f) for f in focus) if n]

        sections = []
        used = 0
        omitted = {"dependencies": [], "dependents": []}

        for n in focus_nodes:
            text = self.render(n, "full")
            cost = estimate_tokens(text)
            if used + cost > budget and sections:
                omitted.setdefault("focus", []).append(n.qualname)
                continue
            sections.append({"kind": "definition", "detail": "full",
                             "name": n.qualname, "text": text, "tokens": cost})
            used += cost

        deps = []
        for n in focus_nodes:
            for d in n.deps:
                if d not in deps and d not in {f.qualname for f in focus_nodes}:
                    deps.append(d)
        for d in deps:
            node = self.nodes.get(d)
            if node is None:
                continue
            text = self.render(node, "contract")
            cost = estimate_tokens(text)
            if used + cost > budget:
                omitted["dependencies"].append(d)
                continue
            sections.append({"kind": "dependency", "detail": "contract",
                             "name": d, "text": text, "tokens": cost})
            used += cost

        if include_dependents:
            seen = set()
            for n in focus_nodes:
                for c in self.reverse.get(n.qualname, ()):
                    if c in seen or c in {f.qualname for f in focus_nodes}:
                        continue
                    seen.add(c)
                    node = self.nodes.get(c)
                    if node is None:
                        continue
                    text = self.render(node, "signature")
                    cost = estimate_tokens(text)
                    if used + cost > budget:
                        omitted["dependents"].append(c)
                        continue
                    sections.append({"kind": "dependent", "detail": "signature",
                                     "name": c, "text": text, "tokens": cost})
                    used += cost

        return {
            "focus": [n.qualname for n in focus_nodes],
            "budget": budget,
            "tokens_used": used,
            "sections": sections,
            "omitted": {k: v for k, v in omitted.items() if v},
            "complete": not any(omitted.values()),
            "text": "\n\n".join(s["text"] for s in sections),
        }

    # ------------------------------------------------------------------
    # Editing
    # ------------------------------------------------------------------

    def propose(self, changes: dict, actor: str = "agent") -> "Proposal":
        """
        Open an edit transaction. Nothing is applied until it is committed.

        `changes` maps a source filename to its complete new text.
        """
        return Proposal(self, changes, actor)


# --------------------------------------------------------------------------
# Transactions
# --------------------------------------------------------------------------

class Proposal:
    """
    A proposed set of source changes, checked but not applied.

    The point of making this a transaction rather than a write is that the
    expensive part of an agent's mistake is usually not the mistake, it is the
    broken intermediate state left behind. Here there is no intermediate state:
    a proposal either becomes the working set atomically or is discarded.
    """

    def __init__(self, atlas: Atlas, changes: dict, actor: str = "agent"):
        self.atlas = atlas
        self.actor = actor
        self.changes = dict(changes)
        self.sources = dict(atlas.sources)
        self.sources.update(changes)

        # Each file is re-parsed with its own language, so a proposal that
        # touches a Verdict decision in a workspace that also holds Loom and
        # Weft still checks the whole set.
        from .sources import Workspace, load_text

        ws = Workspace()
        for name, text in sorted(self.sources.items()):
            load_text(text, name, ws)
        self.bag = ws.bag
        self.modules = ws.modules
        self.workspace = ws

        self.result: Optional[CheckResult] = None
        self.new_atlas: Optional[Atlas] = None
        if not self.bag.has_errors:
            self.result = check(self.modules, self.bag)
            if not self.result.bag.has_errors:
                self.new_atlas = Atlas(self.result, sources=self.sources)

        self.committed = False
        self.aborted = False

    # ------------------------------------------------------------------

    @property
    def ok(self) -> bool:
        return self.new_atlas is not None

    def diagnostics(self) -> list:
        source = next(iter(self.changes.values()), "")
        return [d.to_json() for d in self.bag]

    def render_diagnostics(self) -> str:
        text = next(iter(self.changes.values()), None)
        return self.bag.render(text)

    def impact(self) -> dict:
        """What this proposal would change, before deciding to apply it."""
        from .shadow import structural_diff

        if not self.ok:
            return {"ok": False,
                    "errors": len([d for d in self.bag
                                   if d.severity.value == "error"]),
                    "diagnostics": self.diagnostics()}

        diff = structural_diff(self.atlas.cr, self.result,
                               self.atlas.defs, self.new_atlas.defs)

        radius = set()
        for c in diff.changes:
            if c.kind == "changed" and not c.body_changed:
                continue
            if c.kind == "unchanged":
                continue
            radius.update(self.atlas.callers(c.qualname, transitive=True))
            radius.add(c.qualname)

        return {
            "ok": True,
            "actor": self.actor,
            "files": sorted(self.changes),
            "diff": diff.to_json(),
            "edited": [c.qualname for c in diff.edited()],
            "added": [c.qualname for c in diff.changes if c.kind == "added"],
            "removed": [c.qualname for c in diff.changes if c.kind == "removed"],
            "blast_radius": sorted(radius),
            "capabilities_added": diff.capabilities_added,
            "classifications_added": diff.classifications_added,
            "warnings": [d.to_json() for d in self.bag
                         if d.severity.value == "warning"],
        }

    def canonical(self) -> dict:
        """
        The proposal's sources in canonical form, ready to write.

        Only Canon files are canonicalised. A surface language lowers *to*
        Canon, so re-emitting a lowered module would replace a Verdict decision
        or a Loom workflow with the Canon it expands into -- losing the
        source. Those files are returned unchanged.
        """
        if not self.ok:
            return {}
        out = {}
        for mod in self.new_atlas.cr.modules:
            name = mod.source_file
            if name not in self.changes:
                continue
            if self.workspace.languages.get(name, "canon") == "canon":
                out[name] = format_module(mod)
            else:
                out[name] = self.changes[name]
        return out

    def canonicalised_files(self) -> list:
        """Which changed files the canonical printer actually rewrote."""
        return sorted(n for n in self.changes
                      if self.workspace.languages.get(n, "canon") == "canon")

    def commit(self) -> Atlas:
        if not self.ok:
            raise ValueError("cannot commit a proposal that does not check")
        if self.aborted:
            raise ValueError("this proposal was aborted")
        self.committed = True
        return self.new_atlas

    def abort(self):
        self.aborted = True


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _signature(fi) -> str:
    p = Printer()
    params = ", ".join(f"{prm.name}: {p.ty(prm.ty)}" for prm in fi.decl.params)
    tp = f"<{', '.join(fi.tparams)}>" if fi.tparams else ""
    sig = f"fn {fi.name}{tp}({params}) -> {p.ty(fi.decl.result)}"
    if fi.decl.uses:
        sig += " uses " + ", ".join(sorted({e.key() for e in fi.decl.uses}))
    return sig


def _decl_signature(decl) -> str:
    p = Printer()
    if isinstance(decl, A.RecordDecl):
        return (f"record {decl.name} {{ "
                + ", ".join(f"{f.name}: {p.ty(f.ty)}" for f in decl.fields)
                + " }")
    if isinstance(decl, A.EnumDecl):
        return f"enum {decl.name} {{ " + " | ".join(
            v.name for v in decl.variants) + " }"
    if isinstance(decl, A.AliasDecl):
        return f"alias {decl.name} = {p.ty(decl.target)}"
    if isinstance(decl, A.EffectDecl):
        return f"effect {decl.name} {{ " + ", ".join(
            o.name for o in decl.ops) + " }"
    if isinstance(decl, A.ConstDecl):
        return f"const {decl.name}: {p.ty(decl.ty)}"
    if isinstance(decl, A.TestDecl):
        return f'test "{decl.name}"'
    return getattr(decl, "name", "?")


def _cost(c) -> dict:
    if c is None:
        return {}
    out = {}
    for k in ("steps", "io", "tokens", "millis"):
        v = getattr(c, k, None)
        if v is not None:
            out[k] = v
    if getattr(c, "money", None) is not None:
        out["money"] = str(c.money)
    return out
