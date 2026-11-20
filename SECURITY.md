# Security policy

J2K is built to decode untrusted files, so we treat a crash, hang, unbounded
allocation, or memory-safety bug on malformed input as a security issue.

## Supported versions

| Version | Supported |
| --- | --- |
| `0.14.0` | Yes, latest release |
| `0.7.0` to `0.13.1` | Yes, except the `j2k-ml` `cuda` and `metal` features of `0.7.5`, which don't build from crates.io |
| `0.6.x` | Security fixes only, until 1.0 |
| Earlier than `0.6` | No |

We fix security issues on the current release line and backport the fixes to
older supported lines where they apply.

## Reporting a vulnerability

Please report vulnerabilities privately through
[GitHub's private reporting form](https://github.com/frames-sg/j2k/security/advisories/new)
(the **Security** tab, then **Report a vulnerability**).

If you can't use the form, don't put details in a public issue. Open a
[short issue](https://github.com/frames-sg/j2k/issues/new) asking for a private
contact, without naming the affected code or including a proof of concept.

What to expect:

- We acknowledge private reports within 3 business days.
- We tell you whether we've accepted the report, declined it, or need more
  information within 14 days of acknowledging it.
- We agree on a disclosure date with you before publishing an advisory or fix
  details.

## What J2K promises

- Malformed or unsupported input returns an error.
- Error messages don't expose internal details.
- An explicitly requested GPU backend never silently switches to another
  backend.
- Every host-side `unsafe` block in the published crates documents why it is
  sound, and Clippy rejects one that doesn't.
- The decoders are fuzzed, and malformed-input tests run before every release.
