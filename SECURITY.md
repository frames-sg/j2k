# Security Policy

## Reporting a Vulnerability

JPEG decoders ingest adversarial byte streams from the wild. If you find a
crash, memory-safety violation, or undefined behavior in `signinum`, please
report it privately rather than opening a public issue.

Use GitHub's private vulnerability reporting for the repository, or contact the
maintainer through the repository owner profile if private reporting is not yet
enabled.

Please include:
- A minimal reproducer (input bytes + API call).
- Rust version, target triple, and cargo features used.
- Expected vs. observed behavior.

Reports are acknowledged within 7 days. Patches are issued as soon as possible,
generally within 30 days for high-severity issues.

## Supported versions

The supported stable line is the current `0.4.x` facade line. Experimental
adapter and transcode crates receive security fixes in their latest `0.4.x`
release while they remain part of the repository.
