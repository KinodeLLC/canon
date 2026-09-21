"""
The model runtime behind the `ask` expression.

`ask` is a language primitive rather than a library call, which lets the runtime
do five things an SDK call cannot:

  1. Constrain the response to a schema derived from the declared Canon type,
     so the result is a typed value rather than a string to be parsed.
  2. Enforce the enclosing function's contracts on the model's output and
     re-ask on failure, with the failure fed back as repair context.
  3. Treat the call as a capability-scoped effect, so it is denied unless
     granted, counted against io/token/money budgets, and journaled.
  4. Cache by content hash, so an identical ask with identical inputs is free
     and, more importantly, reproducible.
  5. Record the whole exchange -- prompt, response, contract verdict -- as a
     journal entry, which is what makes replay exact.

Providers are pluggable. `DeterministicProvider` produces schema-valid values
without a network and is what the test suite and the verifier run against;
`AnthropicProvider` calls the real API; `ReplayProvider` reads answers back out
of a journal.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Optional

from . import types as TY
from . import values as V
from .interp import AskRequest, CapabilityDenied, Fault


# --------------------------------------------------------------------------
# Model registry
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelInfo:
    """
    What the runtime needs to know about a model to call it correctly and
    charge for it.

    `accepts_temperature` matters: the current Claude models reject
    `temperature` with a 400 rather than ignoring it, so a Canon `temperature`
    clause targeting one of them has to be rejected at check time. Silently
    dropping the clause would be worse -- the author asked for behaviour the
    model cannot provide, and should be told.
    """
    id: str
    vendor: str
    input_per_mtok: Decimal
    output_per_mtok: Decimal
    max_output: int
    context: int
    accepts_temperature: bool = False
    structured_output: bool = True
    thinking_default_on: bool = False


REGISTRY: dict = {}


def _reg(alias, info: ModelInfo):
    REGISTRY[alias] = info


# Claude models. Aliases are the spellings a Canon author writes after `from`.
_CLAUDE = [
    ("claude.opus", "claude-opus-5", "5", "25", 128_000, 1_000_000, False, True, True),
    ("claude.opus5", "claude-opus-5", "5", "25", 128_000, 1_000_000, False, True, True),
    ("claude.sonnet", "claude-sonnet-5", "3", "15", 128_000, 1_000_000, False, True, False),
    ("claude.sonnet5", "claude-sonnet-5", "3", "15", 128_000, 1_000_000, False, True, False),
    ("claude.haiku", "claude-haiku-4-5", "1", "5", 64_000, 200_000, True, True, False),
    ("claude.fable", "claude-fable-5", "10", "50", 128_000, 1_000_000, False, True, True),
    ("claude.opus48", "claude-opus-4-8", "5", "25", 128_000, 1_000_000, False, True, False),
]

for alias, mid, inp, out, mo, ctx, temp, so, think in _CLAUDE:
    _reg(alias, ModelInfo(mid, "anthropic", Decimal(inp), Decimal(out),
                          mo, ctx, temp, so, think))

# A model used only by the deterministic provider. Free, offline, reproducible.
_reg("stub.deterministic",
     ModelInfo("stub-deterministic", "stub", Decimal(0), Decimal(0),
               8192, 1_000_000, True, True, False))


def lookup_model(alias: str) -> Optional[ModelInfo]:
    if alias in REGISTRY:
        return REGISTRY[alias]
    # Allow a bare vendor name to select that vendor's default.
    for key, info in REGISTRY.items():
        if info.id == alias:
            return info
    return None


def known_models() -> list:
    return sorted(REGISTRY)


# --------------------------------------------------------------------------
# Canon type to JSON Schema
# --------------------------------------------------------------------------

def type_to_schema(t, env, seen=None) -> dict:
    """
    Build a JSON Schema for a Canon type.

    The schema is what constrains the model's response, so it has to be exact:
    every object closed with additionalProperties false, every field required.
    A loose schema is how malformed values reach a contract check that was
    never designed to catch them.
    """
    seen = seen or set()
    t = TY.prune(t)

    if isinstance(t, TY.TCon):
        n = t.name
        if n == "Int":
            return {"type": "integer"}
        if n == "Dec":
            # Sent as a string so no precision is lost in JSON's float.
            return {"type": "string",
                    "description": "an exact decimal number, e.g. \"12.50\""}
        if n == "Text":
            return {"type": "string"}
        if n == "Bool":
            return {"type": "boolean"}
        if n == "Unit":
            return {"type": "null"}
        if n == "Bytes":
            return {"type": "string", "description": "base64"}
        if n == "Time":
            return {"type": "integer", "description": "epoch milliseconds"}
        if n == "List":
            return {"type": "array", "items": type_to_schema(t.args[0], env, seen)}
        if n == "Set":
            return {"type": "array", "items": type_to_schema(t.args[0], env, seen),
                    "uniqueItems": True}
        if n == "Map":
            return {"type": "array",
                    "items": {"type": "object",
                              "properties": {
                                  "key": type_to_schema(t.args[0], env, seen),
                                  "value": type_to_schema(t.args[1], env, seen)},
                              "required": ["key", "value"],
                              "additionalProperties": False}}
        if n == "Option":
            inner = type_to_schema(t.args[0], env, seen)
            return {"anyOf": [inner, {"type": "null"}]}
        if n == "Result":
            return {"type": "object",
                    "properties": {
                        "ok": {"anyOf": [type_to_schema(t.args[0], env, seen),
                                         {"type": "null"}]},
                        "err": {"anyOf": [type_to_schema(t.args[1], env, seen),
                                          {"type": "null"}]}},
                    "required": ["ok", "err"],
                    "additionalProperties": False}

        ti = env.types.get(n)
        if ti is None:
            return {}
        if n in seen:
            # Recursive types cannot be expressed in a strict schema.
            return {"type": "object"}
        seen = seen | {n}

        if ti.kind == "enum":
            variants = ti.decl.variants
            if all(not v.params for v in variants):
                return {"type": "string",
                        "enum": [v.name for v in variants],
                        "description": f"one of the {n} cases"}
            branches = []
            for v in variants:
                props = {"case": {"const": v.name}}
                req = ["case"]
                for i, pt in enumerate(v.params):
                    saved = getattr(env, "_tp", set())
                    props[f"field{i}"] = type_to_schema(
                        _resolve(pt, env, ti.tparams), env, seen)
                    req.append(f"field{i}")
                branches.append({"type": "object", "properties": props,
                                 "required": req, "additionalProperties": False})
            return {"anyOf": branches}

        if ti.kind == "record":
            props = {}
            req = []
            for f in ti.decl.fields:
                props[f.name] = type_to_schema(
                    _resolve(f.ty, env, ti.tparams), env, seen)
                if f.doc:
                    props[f.name] = dict(props[f.name], description=f.doc)
                req.append(f.name)
            schema = {"type": "object", "properties": props,
                      "required": req, "additionalProperties": False}
            if ti.decl.intent:
                schema["description"] = ti.decl.intent
            return schema

        if ti.kind == "alias":
            return type_to_schema(_resolve(ti.decl.target, env, ti.tparams),
                                  env, seen)

    if isinstance(t, TY.TRec):
        return {"type": "object",
                "properties": {k: type_to_schema(v, env, seen)
                               for k, v in t.fields.items()},
                "required": sorted(t.fields),
                "additionalProperties": False}

    return {}


def _resolve(texpr, env, tparams):
    """Resolve a declaration's type expression without a full checker pass."""
    from .checker import Checker
    c = Checker()
    c.env = env
    c.tparams = set(tparams)
    return c.resolve_type(texpr, None)


# --------------------------------------------------------------------------
# JSON to Canon value
# --------------------------------------------------------------------------

class CoercionError(Exception):
    def __init__(self, path: str, expected: str, got):
        self.path = path
        self.expected = expected
        self.got = got
        super().__init__(f"at {path or '<root>'}: expected {expected}, "
                         f"got {json.dumps(got)[:120]}")


def coerce(data, t, env, path: str = "") -> Any:
    """
    Turn a decoded JSON response into a Canon value of type `t`.

    Strict by design. A model that returns "42" where an Int was required is a
    contract violation to be retried, not something to silently paper over --
    quiet coercion is how a wrong value reaches production looking right.
    """
    t = TY.prune(t)

    if isinstance(t, TY.TCon):
        n = t.name

        if n == "Int":
            if isinstance(data, bool) or not isinstance(data, int):
                raise CoercionError(path, "an integer", data)
            return data

        if n == "Dec":
            if isinstance(data, str):
                try:
                    return Decimal(data.strip())
                except Exception:
                    raise CoercionError(path, "a decimal string", data)
            if isinstance(data, int) and not isinstance(data, bool):
                return Decimal(data)
            raise CoercionError(path, "a decimal", data)

        if n == "Text":
            if not isinstance(data, str):
                raise CoercionError(path, "a string", data)
            return data

        if n == "Bool":
            if not isinstance(data, bool):
                raise CoercionError(path, "a boolean", data)
            return data

        if n == "Unit":
            return V.UNIT

        if n == "Bytes":
            import base64
            if not isinstance(data, str):
                raise CoercionError(path, "a base64 string", data)
            try:
                return base64.b64decode(data)
            except Exception:
                raise CoercionError(path, "valid base64", data)

        if n == "Time":
            if not isinstance(data, int) or isinstance(data, bool):
                raise CoercionError(path, "epoch milliseconds", data)
            return V.Instant(data, 0)

        if n in ("List", "Set"):
            if not isinstance(data, list):
                raise CoercionError(path, "an array", data)
            items = [coerce(x, t.args[0], env, f"{path}[{i}]")
                     for i, x in enumerate(data)]
            return frozenset(V._hashable(i) for i in items) if n == "Set" \
                else tuple(items)

        if n == "Map":
            if not isinstance(data, list):
                raise CoercionError(path, "an array of key/value objects", data)
            out = {}
            for i, entry in enumerate(data):
                if not isinstance(entry, dict) or "key" not in entry:
                    raise CoercionError(f"{path}[{i}]",
                                        "an object with key and value", entry)
                k = coerce(entry["key"], t.args[0], env, f"{path}[{i}].key")
                v = coerce(entry.get("value"), t.args[1], env, f"{path}[{i}].value")
                from .prelude import _mapkey
                out[_mapkey(k)] = v
            return V.FrozenMap(out)

        if n == "Option":
            if data is None:
                return V.NONE
            return V.some(coerce(data, t.args[0], env, path))

        if n == "Result":
            if not isinstance(data, dict):
                raise CoercionError(path, "an object with ok and err", data)
            if data.get("err") is not None:
                return V.err(coerce(data["err"], t.args[1], env, f"{path}.err"))
            return V.ok(coerce(data.get("ok"), t.args[0], env, f"{path}.ok"))

        ti = env.types.get(n)
        if ti is None:
            raise CoercionError(path, f"a known type (not {n})", data)

        if ti.kind == "enum":
            variants = {v.name: v for v in ti.decl.variants}
            if isinstance(data, str):
                v = variants.get(data)
                if v is None or v.params:
                    raise CoercionError(
                        path, f"one of {sorted(variants)}", data)
                return V.Variant(data, (), n)
            if isinstance(data, dict) and "case" in data:
                cname = data["case"]
                v = variants.get(cname)
                if v is None:
                    raise CoercionError(path, f"one of {sorted(variants)}", cname)
                args = []
                for i, pt in enumerate(v.params):
                    key = f"field{i}"
                    if key not in data:
                        raise CoercionError(f"{path}.{key}", "a value", data)
                    args.append(coerce(data[key],
                                       _resolve(pt, env, ti.tparams), env,
                                       f"{path}.{key}"))
                return V.Variant(cname, tuple(args), n)
            raise CoercionError(path, f"a {n} case", data)

        if ti.kind == "record":
            if not isinstance(data, dict):
                raise CoercionError(path, f"a {n} object", data)
            fields = []
            for f in ti.decl.fields:
                if f.name not in data:
                    raise CoercionError(f"{path}.{f.name}", "a value",
                                        sorted(data))
                fields.append((f.name,
                               coerce(data[f.name],
                                      _resolve(f.ty, env, ti.tparams), env,
                                      f"{path}.{f.name}")))
            return V.Record(n, tuple(fields))

        if ti.kind == "alias":
            return coerce(data, _resolve(ti.decl.target, env, ti.tparams),
                          env, path)

    if isinstance(t, TY.TRec):
        if not isinstance(data, dict):
            raise CoercionError(path, "an object", data)
        return V.Record("", tuple(
            (k, coerce(data.get(k), v, env, f"{path}.{k}"))
            for k, v in sorted(t.fields.items())))

    raise CoercionError(path, TY.show(t), data)


# --------------------------------------------------------------------------
# Providers
# --------------------------------------------------------------------------

@dataclass
class ProviderRequest:
    model_id: str
    system: str
    prompt: str
    schema: dict
    max_tokens: int
    temperature: Optional[Decimal] = None
    # Grounding sources, carried separately from the prompt so a provider can
    # honour them structurally. Not sent on the wire -- they are already part
    # of `prompt` -- and excluded from the cache key for the same reason.
    grounding: list = field(default_factory=list)


@dataclass
class ProviderResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str = "end_turn"
    refusal_category: Optional[str] = None
    raw: Any = None


class ModelProvider:
    name = "abstract"

    def complete(self, req: ProviderRequest) -> ProviderResponse:
        raise NotImplementedError


class DeterministicProvider(ModelProvider):
    """
    Produces a schema-valid response derived from a hash of the request.

    This is not a mock in the usual sense: it is a real provider whose outputs
    are a pure function of its inputs. That makes the whole system testable and
    the verifier able to exercise model-using code without a network, while
    still passing the same coercion and contract path a live response takes.
    """

    name = "deterministic"

    def __init__(self, seed: str = "canon"):
        self.seed = seed

    def complete(self, req: ProviderRequest) -> ProviderResponse:
        h = hashlib.blake2b(
            (self.seed + "\x00" + req.system + "\x00" + req.prompt).encode("utf-8"),
            digest_size=32).digest()
        # When the ask is grounded, draw string values from the source text so
        # the generated answer passes the same grounding check a real response
        # has to pass. A stand-in that could never satisfy the contract would
        # make every grounded function untestable.
        vocab = _vocabulary(req.grounding) if req.grounding else None
        value = _gen_from_schema(req.schema, h, 0, vocab)[0]
        text = json.dumps(value, separators=(",", ":"))
        return ProviderResponse(
            text=text,
            input_tokens=max(1, (len(req.system) + len(req.prompt)) // 4),
            output_tokens=max(1, len(text) // 4),
            raw={"provider": "deterministic"})


DEFAULT_WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
                 "golf", "hotel", "india", "juliet", "kilo", "lima"]


def _vocabulary(sources: list) -> list:
    """Words available to a grounded generation, drawn from the sources."""
    words = []
    seen = set()
    for src in sources:
        for raw in str(src).split():
            w = raw.strip(".,;:!?\"'()[]{}")
            if len(w) >= 4 and w.lower() not in seen:
                seen.add(w.lower())
                words.append(w)
    return words or DEFAULT_WORDS


def _gen_from_schema(schema: dict, h: bytes, i: int, vocab=None):
    """Generate a value satisfying `schema`, driven by bytes from `h`."""
    def byte(j):
        return h[j % len(h)]

    if not schema:
        return None, i

    if "anyOf" in schema:
        branches = schema["anyOf"]
        pick = byte(i) % len(branches)
        return _gen_from_schema(branches[pick], h, i + 1, vocab)

    if "const" in schema:
        return schema["const"], i

    if "enum" in schema:
        opts = schema["enum"]
        return opts[byte(i) % len(opts)], i + 1

    t = schema.get("type")

    if t == "integer":
        if schema.get("description", "").startswith("epoch"):
            return 1_700_000_000_000 + byte(i) * 1000, i + 1
        return byte(i) % 100, i + 1
    if t == "boolean":
        return byte(i) % 2 == 0, i + 1
    if t == "null":
        return None, i
    if t == "string":
        desc = schema.get("description", "")
        if "decimal" in desc:
            return f"{byte(i) % 1000}.{byte(i + 1) % 100:02d}", i + 2
        if "base64" in desc:
            return "Y2Fub24=", i + 1
        words = vocab or DEFAULT_WORDS
        n = 1 + byte(i) % 4
        start = byte(i + 1) % len(words)
        picked = [words[(start + k) % len(words)] for k in range(n)]
        return " ".join(picked), i + 2
    if t == "array":
        n = byte(i) % 3
        i += 1
        out = []
        for _ in range(n):
            v, i = _gen_from_schema(schema.get("items", {}), h, i, vocab)
            out.append(v)
        return out, i
    if t == "object":
        out = {}
        for k in schema.get("required", list(schema.get("properties", {}))):
            v, i = _gen_from_schema(schema.get("properties", {}).get(k, {}),
                                    h, i, vocab)
            out[k] = v
        return out, i
    return None, i


class AnthropicProvider(ModelProvider):
    """
    Calls the Claude API through the official SDK.

    The SDK is an optional dependency: the core language, the checker, the
    verifier and the deterministic provider all work without it. Install with
    `pip install canon[anthropic]` to enable live model calls.
    """

    name = "anthropic"

    def __init__(self, client=None, api_key: Optional[str] = None,
                 effort: str = "high"):
        self._client = client
        self._api_key = api_key
        self.effort = effort

    @property
    def client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as e:
                raise Fault(
                    "CANON-E0403",
                    "the anthropic SDK is not installed",
                    facts={"install": "pip install anthropic",
                           "alternative": "use the deterministic provider"},
                ) from e
            kwargs = {}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            self._client = anthropic.Anthropic(**kwargs)
        return self._client

    def complete(self, req: ProviderRequest) -> ProviderResponse:
        info = None
        for i in REGISTRY.values():
            if i.id == req.model_id:
                info = i
                break

        kwargs = dict(
            model=req.model_id,
            max_tokens=req.max_tokens,
            system=req.system,
            messages=[{"role": "user", "content": req.prompt}],
        )
        if req.schema:
            kwargs["output_config"] = {
                "format": {"type": "json_schema", "schema": req.schema}}
        # Current Claude models reject `temperature` outright rather than
        # ignoring it, so it is only sent to models that accept it. The
        # checker refuses the clause for the others, so reaching here with a
        # temperature set and an unsupporting model should not happen.
        if req.temperature is not None and info is not None \
                and info.accepts_temperature:
            kwargs["temperature"] = float(req.temperature)

        msg = self.client.messages.create(**kwargs)

        if getattr(msg, "stop_reason", None) == "refusal":
            details = getattr(msg, "stop_details", None)
            return ProviderResponse(
                text="", stop_reason="refusal",
                refusal_category=getattr(details, "category", None),
                input_tokens=getattr(msg.usage, "input_tokens", 0),
                output_tokens=getattr(msg.usage, "output_tokens", 0),
                raw=msg)

        text = ""
        for block in msg.content:
            if getattr(block, "type", None) == "text":
                text += block.text

        return ProviderResponse(
            text=text,
            input_tokens=getattr(msg.usage, "input_tokens", 0),
            output_tokens=getattr(msg.usage, "output_tokens", 0),
            stop_reason=getattr(msg, "stop_reason", "end_turn"),
            raw=msg)


class RecordedProvider(ModelProvider):
    """Serves responses out of a recording, keyed by request hash."""

    name = "recorded"

    def __init__(self, entries: dict, fallback: Optional[ModelProvider] = None):
        self.entries = dict(entries)
        self.fallback = fallback

    def complete(self, req: ProviderRequest) -> ProviderResponse:
        key = request_key(req)
        if key in self.entries:
            e = self.entries[key]
            return ProviderResponse(
                text=e["text"], input_tokens=e.get("input_tokens", 0),
                output_tokens=e.get("output_tokens", 0),
                stop_reason=e.get("stop_reason", "end_turn"),
                raw={"provider": "recorded"})
        if self.fallback is not None:
            return self.fallback.complete(req)
        raise Fault(
            "CANON-E0803",
            "no recorded model response for this request",
            facts={"key": key, "model": req.model_id,
                   "recorded": len(self.entries)})


def request_key(req: ProviderRequest) -> str:
    payload = json.dumps({
        "model": req.model_id, "system": req.system, "prompt": req.prompt,
        "schema": req.schema, "max_tokens": req.max_tokens,
        "temperature": str(req.temperature) if req.temperature is not None else None,
    }, sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(payload.encode("utf-8"),
                           digest_size=16).hexdigest()


# --------------------------------------------------------------------------
# Outcomes
# --------------------------------------------------------------------------

@dataclass
class AskOutcome:
    value: Any = None
    ok: bool = False
    attempts: int = 0
    failures: list = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cost: Decimal = field(default_factory=lambda: Decimal(0))
    cached: bool = False
    model_id: str = ""
    request_key: str = ""

    def to_json(self) -> dict:
        return {
            "ok": self.ok,
            "attempts": self.attempts,
            "failures": self.failures,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost": str(self.cost),
            "cached": self.cached,
            "model": self.model_id,
            "request_key": self.request_key,
            "value": V.to_json(self.value) if self.ok else None,
        }


# --------------------------------------------------------------------------
# The ask engine
# --------------------------------------------------------------------------

class ModelRuntime:
    """
    Executes one `ask`: build prompt, constrain, call, coerce, check, retry.

    `obligations` are callables taking the candidate value and returning either
    None (satisfied) or a string describing what failed. The Ledger supplies
    the enclosing function's `ensures` clauses here, which is what makes a
    contract violation by the model a retryable event rather than a fault
    somewhere downstream.
    """

    def __init__(self, provider: ModelProvider, env, cache: Optional[dict] = None,
                 budget=None, journal=None, strict_grounding: bool = True):
        self.provider = provider
        self.env = env
        self.cache = cache if cache is not None else {}
        self.budget = budget
        self.journal = journal
        self.strict_grounding = strict_grounding

    # ------------------------------------------------------------------

    def run(self, req: AskRequest, result_type, def_hash: str = "") -> AskOutcome:
        info = lookup_model(req.model)
        if info is None:
            raise Fault(
                "CANON-E0201", f"unknown model {req.model!r}",
                req.span,
                facts={"model": req.model, "known": known_models()})

        schema = type_to_schema(result_type, self.env)
        max_tokens = req.max_tokens or min(4096, info.max_output)

        outcome = AskOutcome(model_id=info.id)
        attempts = max(1, req.retries + 1)
        repair_note = ""

        for attempt in range(attempts):
            prompt = self._build_prompt(req, schema, repair_note)
            preq = ProviderRequest(
                model_id=info.id,
                system=self._build_system(req, schema),
                prompt=prompt,
                schema=schema if info.structured_output else {},
                max_tokens=max_tokens,
                temperature=req.temperature,
                grounding=list(req.grounded_in),
            )
            key = request_key(preq)
            outcome.request_key = key
            outcome.attempts = attempt + 1

            if key in self.cache:
                resp = self.cache[key]
                outcome.cached = True
            else:
                resp = self.provider.complete(preq)
                self.cache[key] = resp
                outcome.cached = False
                outcome.input_tokens += resp.input_tokens
                outcome.output_tokens += resp.output_tokens
                cost = (Decimal(resp.input_tokens) * info.input_per_mtok
                        + Decimal(resp.output_tokens) * info.output_per_mtok
                        ) / Decimal(1_000_000)
                outcome.cost += cost
                if self.budget is not None:
                    self.budget.charge_tokens(resp.input_tokens + resp.output_tokens)
                    self.budget.charge_money(cost)

            failure = self._evaluate(resp, req, result_type, outcome)
            if failure is None:
                outcome.ok = True
                self._journal(req, preq, resp, outcome, def_hash)
                return outcome

            outcome.failures.append(failure)
            if failure["reason"] not in req.retry_on:
                break
            repair_note = self._repair_note(failure)

        self._journal(req, preq, resp, outcome, def_hash)
        last = outcome.failures[-1] if outcome.failures else {}
        raise Fault(
            "CANON-E0502",
            f"the model did not produce a valid {TY.show(TY.zonk(result_type))} "
            f"after {outcome.attempts} attempt"
            f"{'' if outcome.attempts == 1 else 's'}",
            req.span,
            facts={"model": info.id,
                   "reason": last.get("reason"),
                   "detail": last.get("detail"),
                   "attempts": outcome.attempts,
                   "retry_on": sorted(req.retry_on),
                   "failures": outcome.failures})

    # ------------------------------------------------------------------

    def _evaluate(self, resp, req: AskRequest, result_type,
                  outcome: AskOutcome) -> Optional[dict]:
        if resp.stop_reason == "refusal":
            return {"reason": "refusal",
                    "detail": f"declined ({resp.refusal_category or 'unspecified'})"}

        try:
            data = json.loads(_strip_fence(resp.text))
        except json.JSONDecodeError as e:
            return {"reason": "type_error",
                    "detail": f"response was not valid JSON: {e}",
                    "response": resp.text[:400]}

        try:
            value = coerce(data, result_type, self.env)
        except CoercionError as ce:
            return {"reason": "type_error", "detail": str(ce),
                    "path": ce.path, "expected": ce.expected}

        if req.grounded_in and self.strict_grounding:
            missing = self._grounding_gaps(value, req.grounded_in)
            if missing:
                return {"reason": "grounding_failure",
                        "detail": "output is not supported by the grounding "
                                  "sources",
                        "unsupported": missing[:5]}

        for ob in req.obligations:
            problem = ob(value)
            if problem:
                return {"reason": "contract_violation", "detail": problem}

        if req.judge is not None:
            try:
                if not req.judge(value):
                    return {"reason": "judge_rejected",
                            "detail": "the judge function rejected the result"}
            except Fault as f:
                return {"reason": "judge_rejected", "detail": str(f)}

        outcome.value = value
        return None

    def _grounding_gaps(self, value, sources: list) -> list:
        """
        Check that text the model produced appears in the grounding sources.

        This is a cheap syntactic check on purpose: it catches invented names,
        identifiers and quantities, which is the failure mode that matters for
        extraction, without pretending to be a general entailment judgement.
        Semantic grounding is what a `judge` clause is for.
        """
        corpus = " ".join(sources).lower()
        gaps = []
        for text in _text_leaves(value):
            t = text.strip()
            if len(t) < 4 or " " in t:
                continue
            if t.lower() not in corpus:
                gaps.append(t)
        return gaps

    def _repair_note(self, failure: dict) -> str:
        reason = failure.get("reason")
        detail = failure.get("detail", "")
        if reason == "type_error":
            return ("Your previous response did not match the required "
                    f"schema: {detail}. Return only JSON matching the schema.")
        if reason == "contract_violation":
            return ("Your previous response violated a stated requirement: "
                    f"{detail}. Produce a result that satisfies it.")
        if reason == "grounding_failure":
            unsupported = ", ".join(failure.get("unsupported", []))
            return ("Your previous response contained values not present in "
                    f"the provided sources: {unsupported}. Use only "
                    "information found in the sources.")
        if reason == "judge_rejected":
            return ("Your previous response was rejected on review: "
                    f"{detail}. Try a different answer.")
        return "Your previous response was rejected. Try again."

    # ------------------------------------------------------------------

    def _build_system(self, req: AskRequest, schema: dict) -> str:
        parts = []
        if req.system:
            parts.append(req.system)
        parts.append(
            "Respond with a single JSON value matching the provided schema. "
            "Do not include explanation, commentary, or code fences.")
        if req.grounded_in:
            parts.append(
                "Every value you produce must be supported by the source "
                "material given below. Do not introduce names, identifiers or "
                "quantities that do not appear there.")
        return "\n\n".join(parts)

    def _build_prompt(self, req: AskRequest, schema: dict,
                      repair_note: str = "") -> str:
        parts = []

        for label, value in req.inputs:
            parts.append(f"<{label}>\n{_render_input(value)}\n</{label}>")

        for i, src in enumerate(req.grounded_in):
            parts.append(f"<source_{i}>\n{src}\n</source_{i}>")

        if req.examples:
            ex = []
            for a, b in req.examples:
                ex.append("input: " + _render_input(a)
                          + "\noutput: "
                          + json.dumps(V.to_json(b), separators=(",", ":")))
            parts.append("<examples>\n" + "\n\n".join(ex) + "\n</examples>")

        parts.append("<schema>\n"
                     + json.dumps(schema, indent=2, sort_keys=True)
                     + "\n</schema>")

        if repair_note:
            parts.append("<correction>\n" + repair_note + "\n</correction>")

        return "\n\n".join(parts)

    def _journal(self, req, preq, resp, outcome, def_hash):
        if self.journal is None:
            return
        self.journal.append(
            op="model.infer",
            args=[preq.model_id, preq.system, preq.prompt],
            result=outcome.to_json(),
            def_hash=def_hash,
            meta={"request_key": outcome.request_key,
                  "cached": outcome.cached,
                  "attempts": outcome.attempts,
                  "cost": str(outcome.cost)})


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _strip_fence(text: str) -> str:
    """Remove a markdown code fence if the model wrapped its JSON in one."""
    t = text.strip()
    if t.startswith("```"):
        nl = t.find("\n")
        if nl != -1:
            t = t[nl + 1:]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


def _render_input(v) -> str:
    if isinstance(v, str):
        return v
    return json.dumps(V.to_json(v), indent=2, sort_keys=True)


def _text_leaves(value, out=None) -> list:
    out = [] if out is None else out
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, tuple):
        for x in value:
            _text_leaves(x, out)
    elif isinstance(value, frozenset):
        for x in value:
            _text_leaves(x, out)
    elif isinstance(value, V.Record):
        for _, x in value.fields:
            _text_leaves(x, out)
    elif isinstance(value, V.Variant):
        for x in value.args:
            _text_leaves(x, out)
    elif isinstance(value, V.FrozenMap):
        for _, x in value.items():
            _text_leaves(x, out)
    return out


def default_provider() -> ModelProvider:
    """
    Pick a provider from the environment.

    Deterministic unless a key is present, so nothing accidentally makes a
    billable call during a test run or a verification pass.
    """
    if os.environ.get("CANON_MODEL_PROVIDER") == "anthropic":
        return AnthropicProvider()
    return DeterministicProvider()
