"""
Runtime value representation.

Values are immutable. Nothing in Canon mutates a value in place, so a value can
be hashed, cached, journaled and compared structurally without defensive
copying. That is what makes deterministic replay and result caching cheap
rather than a special mode.

Mapping to host types:

    Int      int (arbitrary precision)
    Dec      decimal.Decimal
    Text     str
    Bool     bool
    Unit     UNIT (a singleton, not None, so a missing value is never
             mistaken for a unit value)
    Bytes    bytes
    Time     Instant (logical, supplied by the runtime -- never the host clock)
    List     tuple
    Set      frozenset
    Map      FrozenMap
    record   Record
    enum     Variant
    function Closure or Native
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal
from typing import Any


class _Unit:
    _inst = None

    def __new__(cls):
        if cls._inst is None:
            cls._inst = super().__new__(cls)
        return cls._inst

    def __repr__(self):
        return "()"

    def __eq__(self, other):
        return isinstance(other, _Unit)

    def __hash__(self):
        return hash("()")


UNIT = _Unit()


@dataclass(frozen=True)
class Instant:
    """
    A point in time as supplied by the runtime.

    `epoch_millis` is the wall-clock reading that was journaled; `logical` is a
    monotonically increasing counter within a run. Code compares logical values
    so that replay is exact even though wall-clock readings differ.
    """
    epoch_millis: int
    logical: int = 0

    def __str__(self):
        return f"t{self.logical}@{self.epoch_millis}"


class FrozenMap:
    """An immutable, order-independent mapping with structural equality."""

    __slots__ = ("_d", "_h")

    def __init__(self, items=()):
        if isinstance(items, FrozenMap):
            self._d = dict(items._d)
        elif isinstance(items, dict):
            self._d = dict(items)
        else:
            self._d = dict(items)
        self._h = None

    def get(self, k, default=None):
        return self._d.get(k, default)

    def has(self, k):
        return k in self._d

    def insert(self, k, v) -> "FrozenMap":
        d = dict(self._d)
        d[k] = v
        return FrozenMap(d)

    def remove(self, k) -> "FrozenMap":
        d = dict(self._d)
        d.pop(k, None)
        return FrozenMap(d)

    def keys(self):
        return tuple(sorted(self._d.keys(), key=_sort_key))

    def values(self):
        return tuple(self._d[k] for k in self.keys())

    def items(self):
        return tuple((k, self._d[k]) for k in self.keys())

    def __len__(self):
        return len(self._d)

    def __iter__(self):
        return iter(self.keys())

    def __contains__(self, k):
        return k in self._d

    def __eq__(self, other):
        return isinstance(other, FrozenMap) and other._d == self._d

    def __hash__(self):
        if self._h is None:
            self._h = hash(tuple(sorted(
                ((k, _hashable(v)) for k, v in self._d.items()),
                key=lambda kv: _sort_key(kv[0]))))
        return self._h

    def __repr__(self):
        return "{" + ", ".join(f"{k!r}: {v!r}" for k, v in self.items()) + "}"


@dataclass(frozen=True)
class Record:
    """A nominal record value. `fields` is ordered as declared."""
    type_name: str
    fields: tuple          # tuple[(name, value)]

    def get(self, name):
        for n, v in self.fields:
            if n == name:
                return v
        raise KeyError(name)

    def has(self, name) -> bool:
        return any(n == name for n, _ in self.fields)

    def set(self, name, value) -> "Record":
        return Record(self.type_name,
                      tuple((n, value if n == name else v) for n, v in self.fields))

    def as_dict(self) -> dict:
        return {n: v for n, v in self.fields}

    def __repr__(self):
        inner = ", ".join(f"{n}: {v!r}" for n, v in self.fields)
        return f"{self.type_name} {{ {inner} }}"


@dataclass(frozen=True)
class Variant:
    """An enum value: a constructor name and its payload."""
    ctor: str
    args: tuple = ()
    type_name: str = ""

    def __repr__(self):
        if not self.args:
            return self.ctor
        return f"{self.ctor}(" + ", ".join(repr(a) for a in self.args) + ")"


@dataclass(frozen=True)
class Closure:
    """A user-defined function value together with its captured environment."""
    params: tuple
    body: Any
    env: Any
    name: str = "<lambda>"

    def __repr__(self):
        return f"<fn {self.name}/{len(self.params)}>"


@dataclass(frozen=True)
class Native:
    """A builtin implemented in the host language."""
    name: str
    arity: int
    impl: Any

    def __repr__(self):
        return f"<builtin {self.name}/{self.arity}>"


# Convenience constructors for the prelude enums.

def ok(v):
    return Variant("Ok", (v,), "Result")


def err(v):
    return Variant("Err", (v,), "Result")


def some(v):
    return Variant("Some", (v,), "Option")


NONE = Variant("None", (), "Option")


def is_ok(v) -> bool:
    return isinstance(v, Variant) and v.ctor == "Ok"


def is_err(v) -> bool:
    return isinstance(v, Variant) and v.ctor == "Err"


# --------------------------------------------------------------------------
# Ordering and hashing
# --------------------------------------------------------------------------

def _sort_key(v):
    """A total order across value kinds, so sorts and map keys are stable."""
    if isinstance(v, bool):
        return (0, int(v))
    if isinstance(v, int):
        return (1, v)
    if isinstance(v, Decimal):
        return (1, float(v))
    if isinstance(v, str):
        return (2, v)
    if isinstance(v, bytes):
        return (3, v)
    if isinstance(v, _Unit):
        return (4, 0)
    if isinstance(v, Instant):
        return (5, v.logical)
    if isinstance(v, tuple):
        return (6, tuple(_sort_key(x) for x in v))
    if isinstance(v, frozenset):
        return (7, tuple(sorted((_sort_key(x) for x in v))))
    if isinstance(v, Variant):
        return (8, v.ctor, tuple(_sort_key(a) for a in v.args))
    if isinstance(v, Record):
        return (9, v.type_name, tuple((n, _sort_key(x)) for n, x in v.fields))
    if isinstance(v, FrozenMap):
        return (10, tuple((str(k), _sort_key(x)) for k, x in v.items()))
    return (99, repr(v))


def compare(a, b) -> int:
    ka, kb = _sort_key(a), _sort_key(b)
    return -1 if ka < kb else (1 if ka > kb else 0)


def _hashable(v):
    if isinstance(v, (tuple, list)):
        return tuple(_hashable(x) for x in v)
    if isinstance(v, Record):
        return (v.type_name, tuple((n, _hashable(x)) for n, x in v.fields))
    if isinstance(v, Variant):
        return (v.ctor, tuple(_hashable(a) for a in v.args))
    return v


# --------------------------------------------------------------------------
# Canonical serialisation
#
# Every value that crosses a boundary -- into the journal, into a cache key,
# into an audit record, into a model prompt -- is serialised through here, so
# the same value always produces the same bytes.
# --------------------------------------------------------------------------

def to_json(v) -> Any:
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return v
    if isinstance(v, Decimal):
        return {"$dec": format(v.normalize(), "f")}
    if isinstance(v, str):
        return v
    if isinstance(v, bytes):
        return {"$b64": base64.b64encode(v).decode("ascii")}
    if isinstance(v, _Unit):
        return {"$unit": True}
    if isinstance(v, Instant):
        return {"$time": v.epoch_millis, "logical": v.logical}
    if isinstance(v, tuple):
        return [to_json(x) for x in v]
    if isinstance(v, frozenset):
        return {"$set": [to_json(x) for x in sorted(v, key=_sort_key)]}
    if isinstance(v, FrozenMap):
        return {"$map": [[to_json(k), to_json(x)] for k, x in v.items()]}
    if isinstance(v, Record):
        return {"$rec": v.type_name,
                "f": {n: to_json(x) for n, x in v.fields}}
    if isinstance(v, Variant):
        return {"$var": v.ctor, "a": [to_json(a) for a in v.args],
                "t": v.type_name}
    if isinstance(v, (Closure, Native)):
        return {"$fn": getattr(v, "name", "<fn>")}
    if v is None:
        return None
    return {"$opaque": repr(v)}


def from_json(j) -> Any:
    if isinstance(j, bool) or isinstance(j, int) or isinstance(j, str):
        return j
    if j is None:
        return None
    if isinstance(j, list):
        return tuple(from_json(x) for x in j)
    if isinstance(j, dict):
        if "$dec" in j:
            return Decimal(j["$dec"])
        if "$b64" in j:
            return base64.b64decode(j["$b64"])
        if "$unit" in j:
            return UNIT
        if "$time" in j:
            return Instant(j["$time"], j.get("logical", 0))
        if "$set" in j:
            return frozenset(from_json(x) for x in j["$set"])
        if "$map" in j:
            return FrozenMap({from_json(k): from_json(v) for k, v in j["$map"]})
        if "$rec" in j:
            return Record(j["$rec"],
                          tuple((n, from_json(x)) for n, x in j["f"].items()))
        if "$var" in j:
            return Variant(j["$var"], tuple(from_json(a) for a in j["a"]),
                           j.get("t", ""))
        if "$fn" in j:
            return Native(j["$fn"], 0, None)
    return j


def canonical_bytes(v) -> bytes:
    return json.dumps(to_json(v), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def value_hash(v) -> str:
    h = hashlib.blake2b(canonical_bytes(v), digest_size=16).digest()
    return base64.b32encode(h).decode("ascii").rstrip("=").lower()[:16]


# --------------------------------------------------------------------------
# Display
# --------------------------------------------------------------------------

def show(v, depth: int = 0) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, Decimal):
        return format(v.normalize(), "f")
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, bytes):
        return f"0x{v.hex()}"
    if isinstance(v, _Unit):
        return "()"
    if isinstance(v, Instant):
        return str(v)
    if isinstance(v, tuple):
        return "[" + ", ".join(show(x, depth + 1) for x in v) + "]"
    if isinstance(v, frozenset):
        return ("{" + ", ".join(show(x, depth + 1)
                                for x in sorted(v, key=_sort_key)) + "}")
    if isinstance(v, FrozenMap):
        return ("{" + ", ".join(f"{show(k)}: {show(x, depth + 1)}"
                                for k, x in v.items()) + "}")
    if isinstance(v, Record):
        inner = ", ".join(f"{n}: {show(x, depth + 1)}" for n, x in v.fields)
        return f"{v.type_name} {{ {inner} }}"
    if isinstance(v, Variant):
        if not v.args:
            return v.ctor
        return f"{v.ctor}(" + ", ".join(show(a, depth + 1) for a in v.args) + ")"
    if isinstance(v, (Closure, Native)):
        return repr(v)
    if isinstance(v, int):
        return str(v)
    return repr(v)


def type_name_of(v) -> str:
    """The Canon type name of a runtime value, for diagnostics."""
    if isinstance(v, bool):
        return "Bool"
    if isinstance(v, int):
        return "Int"
    if isinstance(v, Decimal):
        return "Dec"
    if isinstance(v, str):
        return "Text"
    if isinstance(v, bytes):
        return "Bytes"
    if isinstance(v, _Unit):
        return "Unit"
    if isinstance(v, Instant):
        return "Time"
    if isinstance(v, tuple):
        return "List"
    if isinstance(v, frozenset):
        return "Set"
    if isinstance(v, FrozenMap):
        return "Map"
    if isinstance(v, Record):
        return v.type_name
    if isinstance(v, Variant):
        return v.type_name or "enum"
    if isinstance(v, (Closure, Native)):
        return "Fn"
    return "Unknown"
