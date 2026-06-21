# JPEG Metal Routing and Benchmarks

JPEG Metal decode should stay selective. The current Metal paths are for fast
baseline/checkpointed packets, coalesced batches, and resident outputs. They are
not a claim of full JPEG entropy decode coverage.

## Routing Contract

Explicit `BackendRequest::Metal` is strict:

- Accept candidates: fast baseline 4:2:0, 4:2:2, or 4:4:4 packets produced by
  the `j2k-jpeg` fast packet builders.
- Output formats: `Gray8`, `Rgb8`, and `Rgba8`.
- Rejections: unsupported packet shape, unsupported output format, unsupported
  backend, or unavailable Metal runtime. Unsupported explicit Metal requests
  must return a structured error before launching kernels.

`BackendRequest::Auto` is deliberately narrower:

- Single-image full, region, scaled, and region-scaled requests stay on CPU even
  when a fast packet exists.
- Small restart-coded tile batches stay CPU.
- Existing restart-coded batch threshold tests cover the current macOS Auto path
  that can use Metal for coalesced WSI-style batches.
- Sparse viewport workloads stay CPU for scheduled surface output; contiguous
  restart-coded viewports may use the hybrid path on macOS. Reusable resident
  viewport outputs may use direct contiguous decode or resident composition.

## Benchmark Map

Run:

```sh
cargo bench -p j2k-jpeg-metal --bench compare
```

The harness also accepts extra JPEG corpus roots through `J2K_BENCH_INPUTS`.
Inputs are grouped by dimensions, restart/checkpoint shape, and fast packet
family so batch results make coalescing visible.

Use these groups to decide where Metal makes sense:

- `decode_rgb`: cold single full-frame CPU vs explicit Metal. This is the loss
  check for CPU-preferred single decode.
- `wsi_tile_batch_rgb` and `wsi_tile_batch_scaled_rgb_q4`: repeated tile batches
  comparing CPU, explicit Metal, and Auto.
- `wsi_tile_batch_region_scaled_coalesced_rgb_q4`: coalesced region+scaled batch
  candidate where Metal can amortize setup.
- `wsi_tile_batch_region_scaled_distinct_rgb_q4`: low-coalescing control case.
  Treat Metal wins here as evidence, not an assumption.
- `wsi_tile_batch_rgba_textures`: resident texture batches that avoid host
  downloads.
- `viewer_region_scaled_composite_rgb*`: CPU/hybrid viewport comparisons, with
  warm variants separating setup from repeated viewer work.
- `viewer_resident_viewport_rgb_buffer_warm` and
  `viewer_resident_viewport_rgba_texture_warm`: resident-output viewport cases.
- `jpeg_metal_fast_packet_planning`: route-discovery overhead for accepted and
  rejected fast packet families.

The benchmark surface intentionally includes both likely wins and likely losses.
Do not broaden Auto routing unless the relevant group shows repeatable wins for
the workload class being changed.

## Applied Rust Guidance

- Rust API Guidelines: the routing contract keeps errors meaningful and
  documented, validates request shape before dispatch, and links the benchmark
  evidence to the public crate docs.
  <https://rust-lang.github.io/api-guidelines/checklist.html>
- Clippy docs: this crate keeps the existing targeted `pedantic` setup and does
  not enable broad `restriction` or `nursery` groups for benchmark-only code.
  <https://doc.rust-lang.org/clippy/lints.html>
- Cargo features: no new feature flag is added for this routing change; macOS
  Metal availability remains target-gated so feature unification cannot widen
  JPEG Metal behavior accidentally.
  <https://doc.rust-lang.org/cargo/reference/features.html>
- Unsafe Code Guidelines: no new unsafe code is required for the routing or
  benchmark documentation; the Metal path continues to use the existing runtime
  wrappers and tests around resident surfaces.
  <https://rust-lang.github.io/unsafe-code-guidelines/introduction.html>
