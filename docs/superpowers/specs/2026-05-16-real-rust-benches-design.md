# Real Rust Benches Design

## Scope

Add real Criterion benchmark targets for the public CPU-facing crates that do
not currently have them: `signinum-j2k` and the facade crate `signinum`.

This pass does not add timing gates to CI and does not introduce external
decoder comparators. CI should continue compiling benchmark targets only.

## Architecture

The new benchmarks will follow the existing workspace pattern:

- Criterion benchmark targets with `harness = false`.
- Deterministic in-process inputs from `signinum-test-support`.
- Public crate APIs only; no private module access.
- `cargo xtask bench-build` compiles the new benches so drift fails in CI.

`signinum-j2k` will benchmark the native CPU public API directly.
`signinum` will benchmark facade dispatch through public re-exports while
forcing CPU-only encode options where needed for portable results.

## Benchmark Targets

`crates/signinum-j2k/benches/public_api.rs` will cover:

- lossless grayscale 8-bit encode
- lossless RGB 8-bit encode
- inspect of a generated codestream
- full RGB decode of a generated codestream
- ROI decode if generated codestreams exercise that path cleanly

Encode benchmarks will use `J2kEncodeValidation::External` so measurements do
not include the facade round-trip validation decode.

`crates/signinum/benches/facade.rs` will cover:

- facade `signinum::j2k::encode_j2k_lossless` with `CpuOnly`
- direct `signinum_j2k::encode_j2k_lossless` with equivalent options

The facade bench is intended to expose dispatch/re-export overhead, not GPU
performance. GPU benchmark coverage remains in the existing backend crates.

## Data Flow

Each benchmark builds deterministic pixel buffers once during setup, validates
sample descriptors before timing, and reuses output buffers inside Criterion
iterations where decoding is measured. Generated codestreams are created during
setup for inspect/decode benchmarks so those measurements do not include encode
cost.

Criterion `black_box` will be used around benchmark inputs and outputs to avoid
dead-code elimination.

## Error Handling

Benchmark setup should fail loudly with `expect` when deterministic fixtures
cannot be encoded, inspected, or decoded. Runtime benchmark iterations should
also surface errors rather than silently skipping paths. The only conditional
scope is ROI decode: if generated codestreams do not support a representative
ROI path cleanly, omit that group in this pass instead of adding a fragile
benchmark.

## Manifests And Documentation

Add `[[bench]]` entries to:

- `crates/signinum-j2k/Cargo.toml`
- `crates/signinum/Cargo.toml`

Add any missing dev-dependencies needed by the benches, including `criterion`
and `signinum-test-support`.

Update:

- `xtask/src/main.rs` so `bench-build` compiles the new bench targets.
- `docs/bench.md` with commands for the new public API benches.

## Testing

Implementation should verify:

- `cargo bench -p signinum-j2k --bench public_api --no-run`
- `cargo bench -p signinum --bench facade --no-run`
- `cargo xtask bench-build`
- narrow affected tests if manifests or shared helpers change

CI remains compile-only for benches. Dedicated timing comparisons stay a local
signoff workflow.
