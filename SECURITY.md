<!--
SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
SPDX-License-Identifier: MPL-2.0
-->

# Security Policy

## Reporting a vulnerability

Please report security vulnerabilities privately via GitHub's
[security advisories](https://github.com/OpenSTEF/openstef/security/advisories/new)
or by emailing <openstef@lfenergy.org>. Do not open public issues for
security-sensitive reports. We aim to acknowledge reports within five working
days.

## Verifying releases

Every OpenSTEF release is cryptographically signed with
[Sigstore](https://www.sigstore.dev/) **keyless** signing. There is no
long-lived private key: each signature uses an ephemeral key bound to the
GitHub Actions release workflow's OIDC identity, with the signing certificate
issued by Fulcio and recorded in the public Rekor transparency log. This means
**there is no public key to download** — verification proves the artifact was
published by our release workflow.

Two channels are signed:

- **PyPI** — every distribution carries a [PEP 740](https://peps.python.org/pep-0740/)
  attestation, displayed and verified by PyPI on upload.
- **GitHub Releases** — each artifact ships with a `.sigstore.json` bundle for
  offline verification.

### Verify a PyPI download

`pip` verifies attestations automatically when downloading from PyPI. To check
explicitly:

```bash
pip download openstef --no-deps -d ./dist
uvx pypi-attestations verify pypi --repo OpenSTEF/openstef dist/openstef-*.whl
```

### Verify a GitHub Release artifact

Download the artifact and its `.sigstore.json` bundle from the
[Releases page](https://github.com/OpenSTEF/openstef/releases), then:

```bash
uvx sigstore verify identity \
    --cert-identity-regexp 'release-v4.yaml@refs/tags/v4' \
    --cert-oidc-issuer https://token.actions.githubusercontent.com \
    openstef-*.whl
```

A successful run confirms the artifact was signed by the OpenSTEF release
workflow. The signing private key is ephemeral and never stored, so it is never
present on PyPI, GitHub Releases, or any distribution site.
