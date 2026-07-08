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

<!-- Only relevant if you checked "Breaking change" above, or are unsure. -->

- [ ] Public API, config schema, or serialized/pickled objects changed in a way that affects existing users
- [ ] If yes: see the [breaking changes guide](https://openstef.github.io/openstef/contribute/development_workflow.html#breaking-changes) for the pickle/`_migrate_state` migration pattern

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
