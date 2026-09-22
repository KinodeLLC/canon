# Canon

core language for the [kinode](../kinode-stack) stack. contracts, effects that
work like capabilities, definitions addressed by content hash.

intent, loom, verdict, weft, tract and rune all lower to this. the runtime,
verifier, journal, atlas, cli and agent interface are in here too.

## install

```sh
pip install -e .
```

python 3.11+, no deps. live model calls want the anthropic sdk
(`pip install anthropic`), everything else runs without it.

## example

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

## constraints

every effect has to be declared. if a function writes to storage it says so in
the uses clause, and so does anything that calls it, so when you add an effect
down in a call graph you have to go update every caller above it. the error
lists them for you. canon will not infer it because then the list grows every
time somebody edits a body and you do not find out until it is in production

there is no division operator. dividing by zero is the only arithmetic that can
fail so you use Int.div instead and it hands you back an Option. recursion
needs a decreases measure and the runtime checks it, matches have to cover
every case, and everything runs under a budget for steps, io, tokens and money

no implicit conversion, no truthiness, no subtyping. a type error points at one
spot and tells you what it expected and what it got instead of unwinding
through three layers of inference

contracts run at runtime. requires on the way in, ensures on the way out,
record invariants whenever you build one, and when something fails you get back
the arguments that did it

definitions are addressed by content hash. rename a local variable or reorder
your clauses or reformat the file and the hash does not move. change something
a function depends on and its deep hash moves while its local hash stays where
it was, so you can tell the difference between somebody editing a function and
somebody editing what it calls

## models

```canon
ask Assessment from claude.opus {
  system "Assess the urgency of this support ticket."
  input body: t.body
  grounded_in t.body
  retries 3 on contract_violation, type_error, grounding_failure
  max_tokens 512
}
```

`ask` is an expression, not an sdk call. the json schema comes off the type you
declared, the `ensures` clauses on the function become obligations the answer
has to satisfy, and if it fails one you get a retry with the failure handed
back to the model as context. the call needs a grant like any other effect and
it gets budgeted and written to the journal

model capabilities get checked when you compile. current claude models reject
`temperature` with a 400 instead of ignoring it, so canon will not let you
write it for those models rather than letting you find out in production

## cli

```sh
canon check   src/            # parse and typecheck
canon fmt     src/ --write    # canonical source
canon hash    src/            # content hashes
canon test    src/            # declared tests
canon verify  src/ --runs 60  # contracts and laws over generated inputs
canon run     src/ --function refund --arguments '...' '5000'
canon atlas   blast refund src/
canon diff    old/ new/
canon journal run.journal
canon serve   src/            # agent interface on stdin/stdout
```

the loader goes by file extension so `canon check src/` handles a directory
with all seven languages mixed together in it

## library

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

## tests

```sh
for t in tests/smoke_*.py; do python "$t"; done
```

seven suites. front end, checker, interpreter, journal, verifier, atlas, agent
protocol.

## docs

[architecture](../kinode-stack/docs/architecture.md) ·
[languages](../kinode-stack/docs/languages.md) ·
[diagnostics](../kinode-stack/docs/diagnostics.md)

## licence

Apache-2.0, Kinode.
