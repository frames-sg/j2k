# Security Policy

## Supported versions

| Version | Status |
| --- | --- |
| `0.14.0` | Latest published and security-supported release |
| `0.7.0` to `0.13.1` | Security-supported, except the `j2k-ml` `cuda` and `metal` features of `0.7.5`, which do not build for registry users |
| `0.6.x` | Security fixes only, until 1.0 |
| Earlier than `0.6` | Unsupported |

Security fixes are developed on the current workspace line and backported to
older supported lines when applicable. Published releases are listed in
[`CHANGELOG.md`](CHANGELOG.md).

## Reporting vulnerabilities

Report suspected vulnerabilities through the
[GitHub private reporting form](https://github.com/frames-sg/j2k/security/advisories/new)
(**Security** tab, **Report a vulnerability**). If the form is unavailable, do
not put vulnerability details in a public issue: open a
[minimal issue](https://github.com/frames-sg/j2k/issues/new) asking the
maintainers for a private contact, without naming the affected code or
including proof-of-concept details.

A release cannot be published while private vulnerability reporting is off:
the release-candidate and tag checks query the setting through the GitHub API
and stop unless it is enabled.

Response expectations:

- Acknowledgment of a private report within **3 business days**.
- Triage decision (accepted / declined / needs more information) within
  **14 days** of acknowledgment.
- Coordinated disclosure: we will agree on a publication date with the
  reporter before any advisory or fix details are made public.

## Baseline expectations

- Unsupported input returns an error.
- Error messages do not expose internal details.
- An explicitly requested GPU backend never silently switches to another
  backend.
- Every host-side `unsafe` block in the published crates explains why it is
  sound, and Clippy rejects one that doesn't.
- Fuzzing and malformed-input tests run before each release.
