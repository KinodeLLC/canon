# Changelog

Keep a Changelog format, SemVer. Pre-1.0: breaking changes bump the minor.

## [Unreleased]

## [0.1.0] - 2026-09-21

### Added
- Lexer, parser and AST for the core language, shared by the whole family.
- Canonical form: one spelling per meaning, invariant under local renames,
  clause reordering and formatting.
- Content addressing with separate local and deep hashes, so a dependency edit
  is distinguishable from a direct edit. Mutually recursive definitions hash as
  a group.
- Type system with no subtyping, no implicit conversion and no truthiness.
  Effect rows are declared rather than inferred; exhaustiveness is required.
- Totality: no division operator, recursion requires a `decreases` measure,
  every evaluation runs under step, io, token, time and spend budgets.
- Structured diagnostics with stable codes, machine-readable facts and
  applicable repairs. Rendered text is generated from the structure.
- Evaluator with runtime contract enforcement for preconditions,
  postconditions and record invariants, carrying the arguments that failed.
- Prelude: builtin effects written in Canon, and a total standard library.
- The `ask` expression: schema-constrained, contract-checked, capability-scoped
  and journaled model invocation, with compile-time model capability checking.
- Ledger: hash-chained effect journal, deny-by-default capability broker with
  classification and call-ceiling limits, append-only audit chain, and live,
  replay and shadow execution modes.
- Verifier: seeded type-directed generation, counterexample shrinking, fifteen
  named laws, and cost observation against declared budgets.
- Change evaluation: structural diff with blast radius, differential execution,
  shadow replay, and a promotion gate returning promote, escalate or block.
- Atlas: definition graph, blast radius, capability footprint attributed to
  source, token-budgeted projections, and transactional edits.
- Mixed-language loader dispatching on file extension with lazy imports.
- The `canon` command and a JSON-RPC agent interface over stdin/stdout.
- Record fields may declare defaults, so adding a field is non-breaking.
