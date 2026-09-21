# Contributing

## Versioning

SemVer 2.0.0. Pre-1.0, breaking changes bump the minor.

Three versions are tracked separately and all appear in release notes:

| Version | Bumps when |
| --- | --- |
| Package | Any release. |
| `language_version` | Surface syntax or semantics change. |
| `ir_version` | The core IR or canonical encoding changes. |

An `ir_version` bump invalidates existing content hashes, so it counts as
breaking regardless of size.

## Commits

[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/):
`<type>(<scope>): <subject>`.

Types: `feat`, `fix`, `docs`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`.
A `!` after the type or a `BREAKING CHANGE:` body paragraph marks a breaking
change.

## Releasing

1. Move `[Unreleased]` entries into a new version section in `CHANGELOG.md`.
2. Bump the version in `pyproject.toml` and `__init__.py`.
3. Commit `chore(release): vX.Y.Z`, tag `vX.Y.Z`, push the tag.
