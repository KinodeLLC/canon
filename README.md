# Canon

The core language of the [Kinode](../kinode-stack) stack: contracts,
capability-typed effects, content-addressed definitions, and the shared IR that
Intent, Loom, Verdict, Weft, Tract and Rune all lower to.

This package also contains the runtime, the verifier, the Ledger, Atlas, the
command line and the agent interface.

## Install

```sh
pip install -e .
```

No dependencies. Python 3.11+. Live model calls need the optional Anthropic
SDK (`pip install anthropic`); everything else — checking, verification,
shadow runs — works without it.

## A definition

```canon
module billing

record Money {
  amount: Int
  currency: Text = "USD"
  invariant amount >= 0
}

effect ledger {
  append(entry: Text, amount: Int) -> Unit
  idempotent read(id: Text) -> Option<Text>
}

fn refund(c: Charge, amt: Int) -> Result<Refund, RefundError>
  intent "Refund up to the captured amount of a charge, once."
  uses ledger.append
  requires amt > 0
  ensures true
  law idempotent_by(c.id)
  cost steps 5000, io 2
{
  match c.status {
    case Pending  => Err(NotCaptured)
    case Refunded => Err(AlreadyRefunded)
    case Captured => {
      let ceiling = c.amount.amount
      if amt > ceiling then Err(AmountExceeded(ceiling)) else {
        do ledger.append(c.id, amt)
        Ok(Refund { charge_id: c.id, amount: amt })
      }
    }
  }
}
```

## What is enforced

- **Effects are declared, never inferred.** A function that performs an effect
  must say so, and so must every caller. An inferred footprint widens silently
  when a body changes.
- **Everything is total.** No division operator (use `Int.div`, which returns
  `Option`); recursion requires a `decreases` measure; matches must be
  exhaustive; every evaluation runs under a step, io, token and spend budget.
- **No implicit conversion, no truthiness, no subtyping.** A type error points
  at one site with a concrete expected and actual type.
- **Contracts are enforced at runtime** — preconditions on entry,
  postconditions on exit, record invariants on construction — and carry the
  arguments that produced the failure.
- **Definitions are content-addressed.** Renaming a local, reordering
  independent clauses or reformatting does not change a hash. Editing a
  dependency changes the dependent's deep hash but not its local hash.

## The model primitive

```canon
ask Assessment from claude.opus {
  system "Assess the urgency of this support ticket."
  input body: t.body
  grounded_in t.body
  retries 3 on contract_violation, type_error, grounding_failure
  max_tokens 512
}
```

A JSON Schema is derived from the declared type, the enclosing `ensures`
clauses become obligations on the answer, the call is capability-scoped,
budgeted and journaled, and model capabilities are checked at compile time.

## Command line

```sh
canon check   src/            # parse and type-check
canon fmt     src/ --write    # canonical source
canon hash    src/            # content hashes
canon test    src/            # declared tests
canon verify  src/ --runs 60  # contracts and laws over generated inputs
canon run     src/ --function refund --arguments '{"$rec": "..."}' '5000'
canon atlas   blast refund src/
canon diff    old/ new/       # structural and behavioural
canon journal run.journal     # inspect and verify an effect journal
canon serve   src/            # agent interface on stdin/stdout
```

The loader dispatches on file extension, so `canon check src/` handles a
directory containing all seven of the family's languages.

## Library

```python
from canon.sources import load, check_workspace
from canon.atlas import Atlas
from canon.verifier import Verifier
from canon import Hasher

ws = load(["src/"])
cr = check_workspace(ws)
defs = Hasher().add_modules(cr.modules)
atlas = Atlas(cr, defs, sources=ws.sources)

print(atlas.blast_radius("refund"))
print(Verifier(cr, seed="prod", runs=100).verify_all().render())
```

## Tests

```sh
for t in tests/smoke_*.py; do python "$t"; done
```

Seven suites: front end, checker, interpreter, Ledger, verifier, Atlas, agent
protocol.

## Documentation

[Architecture](../kinode-stack/docs/architecture.md) ·
[Languages](../kinode-stack/docs/languages.md) ·
[Diagnostics](../kinode-stack/docs/diagnostics.md)

## Licence

Apache-2.0. Copyright Kinode.
