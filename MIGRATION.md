# Migration Notes

The public codec crates moved from the retired `ashlar-*` package family to
the `signinum-*` package family. The whole-slide reader crate moved from
`ziggurat` to `statumen`.

Yanked retired crates remain available to existing lockfiles, but new
dependency resolution should use the current package names.

## Codec Crates

| Retired package | Current package |
|-----------------|-----------------|
| `ashlar-core` | `signinum-core` |
| `ashlar-jpeg` | `signinum-jpeg` |
| `ashlar-j2k` | `signinum-j2k` |
| `ashlar-tilecodec` | `signinum-tilecodec` |
| `ashlar-cli` | `signinum-cli` |
| `ashlar-j2k-native` | `signinum-j2k-native` |
| `ashlar-jpeg-metal` | `signinum-jpeg-metal` |
| `ashlar-j2k-metal` | `signinum-j2k-metal` |
| `ashlar-jpeg-cuda` | `signinum-jpeg-cuda` |
| `ashlar-j2k-cuda` | `signinum-j2k-cuda` |

`ashlar-j2k-compare` was a local oracle/comparison helper. It does not have a
published replacement for downstream use.

## Reader Crate

| Retired package | Current package |
|-----------------|-----------------|
| `ziggurat` | `statumen` |

Use `statumen` when you need whole-slide reader/container behavior. Use the
`signinum-*` crates when you need codec primitives directly.

## Cargo Examples

```toml
[dependencies]
signinum-jpeg = "1.0"
signinum-j2k = "1.0"
signinum-tilecodec = "1.0"
```

For CUDA device-memory output:

```toml
[dependencies]
signinum-jpeg-cuda = { version = "0.2", features = ["cuda-runtime"] }
```

The CUDA adapters upload CPU-decoded bytes into CUDA device memory. They do
not provide CUDA kernel decode or make NVIDIA performance claims.
