# Fuzzing

Signinum treats compressed image bytes as untrusted input. Fuzzing covers the
parser/decode boundaries that ingest JPEG, JPEG 2000 / HTJ2K, and container
tile-compression payloads.

## Scheduled run

The scheduled fuzz workflow installs `cargo-fuzz` and runs:

```sh
SIGNINUM_FUZZ_RUNS=100000 cargo xtask fuzz-run
```

`SIGNINUM_FUZZ_RUNS` controls deterministic run count per target.
`SIGNINUM_FUZZ_MAX_TOTAL_TIME_SECONDS` may be set for wall-clock-bounded local
or CI runs.

Current fuzz targets:

- `crates/signinum-jpeg/fuzz/fuzz_targets/parse_fuzz.rs`
- `crates/signinum-jpeg/fuzz/fuzz_targets/decode_fuzz.rs`
- `crates/signinum-j2k/fuzz/fuzz_targets/parse_fuzz.rs`
- `crates/signinum-j2k/fuzz/fuzz_targets/decode_fuzz.rs`
- `crates/signinum-tilecodec/fuzz/fuzz_targets/decompress_fuzz.rs`

## Seed corpus

The committed seed corpus policy lives in [`../corpus/fuzz/README.md`](../corpus/fuzz/README.md).
Seeds must be small, license-clear, and reproducible from a documented source.
Large or proprietary WSI tiles stay outside the repository and are referenced
through external corpus manifests.

## Crash handling

When a scheduled run produces an artifact, keep the raw artifact from CI and
minimize it before opening a public issue:

```sh
cargo fuzz tmin --manifest-path crates/signinum-jpeg/fuzz/Cargo.toml parse_fuzz artifact
```

Private security reports should include the minimized crash, the target name,
the exact command, Rust version, target triple, and cargo features. A public
regression test should use the minimized reproducer only after security triage
confirms that disclosure is appropriate.

