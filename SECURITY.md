# Security Policy

## Supported versions

| Version | Status |
| --- | --- |
| `0.7.0` workspace | Staged under `Unreleased`; not yet published or security-supported as a release |
| `0.6.x` | Latest published and security-supported line |
| Earlier than `0.6` | Unsupported |

Security fixes are developed on the staged workspace line and backported to the
published supported line when applicable. See
[`docs/release.md`](docs/release.md) for the publication state.

## Reporting vulnerabilities

When the repository **Security** tab shows **Report a vulnerability**, report
suspected vulnerabilities through the corresponding
[GitHub private reporting form](https://github.com/frames-sg/j2k/security/advisories/new).
If that button is unavailable, do not put vulnerability details in a public
issue. Open a [minimal issue](https://github.com/frames-sg/j2k/issues/new)
asking the maintainers to provide a private contact, without naming the
affected code or including proof-of-concept details. A
verified direct private channel must be published before the staged `0.7.0`
release is approved.

Response expectations:

- Acknowledgment of a private report within **3 business days**.
- Triage decision (accepted / declined / needs more information) within
  **14 days** of acknowledgment.
- Coordinated disclosure: we will agree on a publication date with the
  reporter before any advisory or fix details are made public.

## Baseline expectations

- Unsupported input must fail explicitly.
- Error responses must avoid sensitive internal details.
- Device backends must not silently substitute a different explicit backend.
- Unsafe Rust inventory is tracked in `docs/unsafe-audit.md`.
- Fuzzing and malformed-input tests are part of release hardening.
