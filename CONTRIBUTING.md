# Contributing

## versioning

semver 2.0.0. before 1.0 a breaking change bumps the minor.

three versions get tracked separately and all three go in release notes

| version | bumps when |
| --- | --- |
| package | any release |
| `language_version` | surface syntax or semantics change |
| `ir_version` | the core ir or canonical encoding changes |

an `ir_version` bump invalidates every hash that exists, so it counts as
breaking no matter how small it looks.

## commits

conventional commits, `<type>(<scope>): <subject>`.

types are `feat`, `fix`, `docs`, `refactor`, `perf`, `test`, `build`, `ci`,
`chore`. a `!` after the type or a `BREAKING CHANGE:` paragraph marks a
breaking change.

## releasing

move the `[Unreleased]` entries into a new version section, bump the version in
`pyproject.toml` and `__init__.py`, commit `chore(release): vX.Y.Z`, tag it and
push the tag.
