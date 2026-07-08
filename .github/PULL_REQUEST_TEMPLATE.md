<!--
Thanks for contributing to OpenSTEF! Please fill out this template to help us
review your PR efficiently. See the contributing guide for more details:
https://openstef.github.io/openstef/contribute/index.html
-->

## What does this PR do?

<!-- Describe the change and the motivation behind it. -->

Closes #

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change (see checklist below)
- [ ] Documentation
- [ ] Refactor / chore / CI

## Breaking changes checklist

<!--
Only fill out this section if you checked "Breaking change" above, or are
unsure whether your change is breaking. Delete the section otherwise.

A change is breaking if it changes public API behavior, removes/renames
something public, or changes what gets (de)serialized. Check any that apply
and describe the migration path below.
-->

- [ ] **Public API changed**: a class, function, method, or parameter was renamed, removed, or had its signature/default/behavior changed
- [ ] **Serialization / pickle compatibility**: a field was added, renamed, removed, or changed type on a `Stateful`/`Transform` subclass (e.g. a scaler, encoder, or other fitted transform). If so:
  - [ ] Bumped `_VERSION` on the affected class
  - [ ] Added/updated `_migrate_state` to migrate previously-pickled state to the new shape (see [`openstef_core.mixins.stateful.Stateful`](packages/openstef-core/src/openstef_core/mixins/stateful.py))
  - [ ] Added a test that restores an old-version state and asserts it migrates correctly (see [`test_stateful.py`](packages/openstef-core/tests/unit/mixins/test_stateful.py) for the pattern)
- [ ] **Config/settings schema changed** in a way that breaks existing configs (field renamed/removed/required)
- [ ] **Default value or behavior changed** in a way that affects existing users' results
- [ ] **Dependency version bump** with a known breaking change

Migration path for existing users (e.g. "old pickled `XScaler` objects auto-migrate on load", "users must now pass `X` explicitly"):

<!-- describe here -->

## AI disclosure

<!--
See our AI-assisted contributions guidelines:
https://openstef.github.io/openstef/contribute/contributing_guide.html#ai-assisted-contributions
-->

- [ ] No AI assistance was used (beyond grammar/spelling)
- [ ] AI assistance was used — tool(s): <!-- e.g. GitHub Copilot, Claude, ChatGPT -->
  - [ ] I have reviewed, understand, and can explain all AI-generated code in this PR
  - [ ] This is disclosed in a commit message (e.g. `Assisted-by: <tool name>`)

## Checklist

- [ ] `poe all --check` passes locally
- [ ] Tests added/updated for the change
- [ ] Documentation updated (docstrings, user guide, examples) if needed
- [ ] Commits are signed off per our [DCO](https://openstef.github.io/openstef/contribute/contributing_guide.html#signing-the-developer-certificate-of-origin-dco) (`git commit -s`)
- [ ] PR title follows [Conventional Commits](https://www.conventionalcommits.org/) (e.g. `feat: ...`, `fix: ...`, `feat!: ...` for breaking changes)
