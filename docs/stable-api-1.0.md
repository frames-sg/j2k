# Stable API Policy

The stable API inventory is generated. The human-maintained policy is small:
stable crates must preserve the public codec contracts, while experimental
adapters may evolve until promoted.

## Generated snapshot

The generated item-level companion is:

- `docs/stable-api-1.0.public-api.txt`

Regenerate or check it with:

```bash
cargo xtask stable-api
cargo xtask stable-api --write
```

This task must run on macOS with `cargo-public-api` `0.52.0` installed
(`cargo install cargo-public-api --version 0.52.0 --locked`) so target-gated
Metal APIs are included.

The snapshot records public items and exit-code contract expectations for the
stable public line. Manual prose in this file must not duplicate that inventory.

## Stability tiers

- Primary stable user-facing codec APIs: `j2k`, `j2k-core`, `j2k-jpeg`,
  and `j2k-tilecodec`.
- Semver-gated published libraries: `j2k`, `j2k-core`, `j2k-codec-math`,
  `j2k-jpeg`, `j2k-tilecodec`, `j2k-jpeg-metal`, `j2k-metal`,
  `j2k-jpeg-cuda`, `j2k-cuda`, `j2k-transcode`, `j2k-transcode-cuda`,
  `j2k-metal-support`, `j2k-transcode-metal`, `j2k-native`, `j2k-types`,
  `j2k-cuda-runtime`, and `j2k-profile`.
- Adapter and transcode crates are semver-gated published libraries, but their
  supported runtime shapes remain limited by feature gates, hardware
  availability, and `docs/public-support.md`.
- Unpublished tooling: test support, comparators, and xtask automation helpers.

Breaking changes to stable crates require explicit semver review.

## CLI contract

`j2k-cli` currently supports:

- `j2k inspect <file>`
- `j2k transcode <input.jpg> <output.j2k> --htj2k --lossless-53`

Usage errors and unsupported options return exit code `2`. Runtime failures,
including unreadable files and unsupported codec inputs, return exit code `1`.
Successful commands return exit code `0` and write a single summary line to
stdout.
