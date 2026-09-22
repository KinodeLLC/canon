"""
loading a workspace with more than one language in it.

every surface language lowers to canon so once a file is loaded nothing
downstream cares where it came from. this is the one place that knows which
parser to reach for, going by file extension and importing each language only
when it needs it, so a workspace using canon and verdict does not need loom or
weft or tract or rune installed.

a file written in a language that is not installed gets a diagnostic naming the
package to install rather than a traceback.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from .diagnostics import Bag, Pos, Repair, Span
from .parser import parse as parse_canon


# extension -> (language, package, module path, function name)
LANGUAGES = {
    ".canon": ("canon", None, None, None),
    ".intent": ("intent", "intent", "intent", "parse_intent"),
    ".loom": ("loom", "loom", "loom", "parse_loom"),
    ".verdict": ("verdict", "verdict", "verdict", "parse_verdict"),
    ".weft": ("weft", "weft", "weft", "parse_weft"),
    ".tract": ("tract", "tract", "tract", "parse_tract"),
    ".rune": ("rune", "rune", "rune", "parse_rune"),
}


@dataclass
class Workspace:
    """A loaded set of source files, lowered to Canon modules."""
    modules: list = field(default_factory=list)
    sources: dict = field(default_factory=dict)
    languages: dict = field(default_factory=dict)
    bag: Bag = field(default_factory=Bag)
    # Surface artifacts the languages produce alongside their modules.
    goals: list = field(default_factory=list)
    policies: list = field(default_factory=list)
    resources: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.bag.has_errors

    def summary(self) -> dict:
        by_lang = {}
        for name, lang in self.languages.items():
            by_lang[lang] = by_lang.get(lang, 0) + 1
        return {"files": len(self.sources), "modules": len(self.modules),
                "by_language": by_lang, "goals": len(self.goals),
                "policies": len(self.policies),
                "resources": len(self.resources)}


def language_of(path: str) -> Optional[str]:
    return LANGUAGES.get(os.path.splitext(path)[1], (None,))[0]


def known_extensions() -> list:
    return sorted(LANGUAGES)


def load_text(text: str, filename: str, ws: Optional[Workspace] = None
              ) -> Workspace:
    """Parse one source string, dispatching on the filename's extension."""
    ws = ws or Workspace()
    ext = os.path.splitext(filename)[1]
    entry = LANGUAGES.get(ext)

    if entry is None:
        ws.bag.error(
            "CANON-E0001", f"unknown source extension {ext!r}",
            _span(filename),
            facts={"file": filename, "known": known_extensions()},
            notes=["Each language in the family has its own extension so a "
                   "loader never has to guess what a file contains."])
        return ws

    lang, package, module_path, fn_name = entry
    ws.sources[filename] = text
    ws.languages[filename] = lang

    if lang == "canon":
        mod, bag = parse_canon(text, filename)
        ws.modules.append(mod)
        ws.bag.extend(bag)
        return ws

    try:
        mod_obj = __import__(module_path, fromlist=[fn_name])
        parser = getattr(mod_obj, fn_name)
    except ImportError:
        ws.bag.error(
            "CANON-E0208",
            f"{filename} is written in {lang}, which is not installed",
            _span(filename),
            facts={"file": filename, "language": lang, "package": package},
            repairs=[Repair("manual", f"install the language",
                            f"pip install {package}", None, 0.9)])
        return ws

    result = parser(text, filename)
    # Tract and Rune return their surface artifacts alongside the module.
    if len(result) == 3:
        mod, extra, bag = result
        if lang == "intent":
            ws.goals.extend(extra)
        elif lang == "rune":
            ws.policies.extend(extra)
        elif lang == "tract":
            ws.resources.extend(extra)
    else:
        mod, bag = result
    ws.modules.append(mod)
    ws.bag.extend(bag)
    return ws


def load(paths, ws: Optional[Workspace] = None) -> Workspace:
    """Load files or directories. Directories are searched recursively."""
    ws = ws or Workspace()
    for path in _expand(paths, ws):
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except OSError as e:
            ws.bag.error("CANON-E0001", f"cannot read {path}: {e}",
                         _span(path), facts={"file": path})
            continue
        load_text(text, path, ws)
    return ws


def _expand(paths, ws: Workspace) -> list:
    out = []
    for path in paths:
        if os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                dirs[:] = [d for d in sorted(dirs)
                           if not d.startswith(".") and d != "__pycache__"]
                for name in sorted(files):
                    if os.path.splitext(name)[1] in LANGUAGES:
                        out.append(os.path.join(root, name))
        elif os.path.exists(path):
            out.append(path)
        else:
            ws.bag.error("CANON-E0001", f"no such file or directory: {path}",
                         _span(path), facts={"path": path})
    return out


def check_workspace(ws: Workspace):
    """Type-check everything loaded. Returns a CheckResult."""
    from .checker import check
    return check(ws.modules, ws.bag)


def _span(filename: str) -> Span:
    p = Pos(filename, 1, 0, 0)
    return Span(p, p)
