# Security Policy

## Supported versions

| Version | Status |
| --- | --- |
| `0.11.3` | Latest published and security-supported release |
| `0.11.2` | Previous published release; security-supported |
| `0.11.1` | Previous published release; security-supported |
| `0.11.0` | Previous published release; security-supported |
| `0.10.0` | Previous published release line; security-supported |
| `0.9.0` | Previous published release line; security-supported |
| `0.8.1` | Previous published release line; security-supported |
| `0.8.0` | Previous published release line; security-supported |
| `0.7.5` | Previous published release line; security-supported, except for the documented `j2k-ml` CUDA and Metal packaging defect |
| `0.7.3` | Previous published release line; security-supported |
| `0.7.2` | Previous published release line; security-supported |
| `0.7.1` | Previous published release line; security-supported |
| `0.7.0` | Previous published release line; security-supported |
| `0.6.x` | Supported for security fixes during the pre-1.0 transition |
| Earlier than `0.6` | Unsupported |

Security fixes are developed on the current workspace line and backported to
older supported lines when applicable. See
[`CHANGELOG.md`](CHANGELOG.md) for published releases.

## Reporting vulnerabilities

When the repository **Security** tab shows **Report a vulnerability**, report
suspected vulnerabilities through the corresponding
[GitHub private reporting form](https://github.com/frames-sg/j2k/security/advisories/new).
If that button is unavailable, do not put vulnerability details in a public
issue. Open a [minimal issue](https://github.com/frames-sg/j2k/issues/new)
asking the maintainers for a private contact, without naming the affected code
or including proof-of-concept details. Future releases require a published,
verified private channel.

The tag-publish preflight checks the repository's private vulnerability
reporting setting through the GitHub API and stops the publish unless it is
`enabled: true`; API authorization failures and malformed responses also stop
it. Before creating a release tag, a repository admin must enable
**Security > Private vulnerability reporting** and confirm that **Report a
vulnerability** is visible. The offline repository checks do not make this
request.

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
