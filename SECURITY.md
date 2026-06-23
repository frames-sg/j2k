# Security Policy

## Supported versions

The current supported development line is `0.6.x`.

## Reporting vulnerabilities

Report suspected vulnerabilities privately through GitHub private vulnerability
reporting: <https://github.com/frames-sg/j2k/security/advisories/new>
(repository **Security** tab → **Report a vulnerability**). Do not open public
issues or publish proof-of-concept exploit details before triage.

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
