# Stable API Policy

The stable API inventory is generated. The human-maintained policy is small:
stable crates must preserve the facade release contracts, while experimental
adapters may evolve until promoted.

## Generated snapshot

The generated item-level companion is:

- `docs/stable-api-1.0.public-api.txt`

Regenerate or check it with:

```bash
cargo xtask stable-api
cargo xtask stable-api --write
```

The snapshot records public items and exit-code contract expectations for the
stable facade line. Manual prose in this file must not duplicate that inventory.

## Stability tiers

- Stable: `signinum`, `signinum-core`, `signinum-jpeg`, `signinum-j2k`,
  `signinum-j2k-native`, `signinum-profile`, and `signinum-tilecodec`.
- Experimental: CUDA adapters, Metal adapters, and transcode crates.
- Unpublished tooling: test support, comparators, and xtask automation helpers.

Breaking changes to stable crates require explicit semver review.
