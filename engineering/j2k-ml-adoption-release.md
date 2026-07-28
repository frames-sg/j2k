# `j2k-ml` adoption release

This is the working release record for the staged accelerator API. It is not a
publication announcement. Do not publish or post externally until the user
gives a new explicit instruction naming the destination and action.

## Accelerator design

`CudaUploadBurnDecoder` and `MetalUploadBurnDecoder` use only released public
dependency APIs:

1. The named accelerator codec decodes into codec-owned resident storage.
2. The adapter waits for codec completion and validates group status.
3. The adapter copies each dense decoded group into host staging.
4. `Tensor::from_data` creates an ordinary Burn tensor on the selected GPU
   backend, performing the normal upload.

These APIs do not promise a direct Burn destination, zero-copy behavior, or an
asynchronous handoff between the codec and Burn runtimes. The root
`[patch.crates-io]` entries for CubeCL and wgpu are removed from this build
path.

Do not open, comment on, request review for, or otherwise contact CubeCL, wgpu,
Burn, or any other external project on behalf of this work. Keep all further
work inside `frames-sg/j2k` unless the user explicitly authorizes an exact
external action.

## Current preparation evidence

The following checks passed during local macOS preparation on 2026-07-24.
They prove the staged design is buildable and functional, but they are not
release evidence because they were not run from the final clean, versioned
candidate SHA:

- `cargo xtask j2k-ml-package-smoke`, including clean packaged-source
  consumers for `cpu`, `metal`, and `cpu,metal`, packaged examples, and docs;
- `cargo test -p j2k-ml --features cpu`;
- strict `j2k-ml` Clippy and docs for `cpu,metal`;
- `cargo xtask release-metal`;
- repository lint, formatting, unsafe-audit, and command-orchestration tests.

The compatible lockfile refresh occurred after the package smoke and full
Metal gate. CPU tests, strict Clippy, repository lint, and formatting passed
again afterward. Rerun the package smoke and full Metal gate from the final
clean candidate SHA.

The CUDA host remained on the published `v0.7.5` release commit during that
preparation. Run the full CUDA gate only after the corrected source is
committed and the same candidate SHA is available on that host. Do not treat a
different checkout or an uncommitted source copy as exact-SHA release evidence.

## Release completion checklist

1. Run the package smoke with registry-only third-party dependencies:

   ```bash
   cargo xtask j2k-ml-package-smoke
   ```

   Linux must compile `cpu`, `cuda`, and `cpu,cuda`; macOS must compile `cpu`,
   `metal`, and `cpu,metal`. Temporary path overrides may name only
   unpublished J2K workspace crates.
2. Run formatting, Clippy, unit/integration tests, examples, docs, package,
   semver/stable-API, CUDA, and Metal release gates.
3. Replace historical direct-destination benchmark comparisons with
   content-distinct batches 1/8/32/64 for the staged accelerator adapters
   versus CPU-decode-and-upload. Record uncertainty, memory, and transfer
   counters without reusing old direct-route claims.
4. Keep the staged version under `Unreleased` until clean-consumer and hardware
   gates pass; only then freeze a dated changelog and exact candidate SHA.
5. After publication, create fresh consumers pinned to the exact published
   version and repeat CPU, CUDA, and Metal checks before any community notice.

## Community notice requirements

Any later notice must state that `j2k-ml` is independent of Burn, explain that
the application owns training batches while the codec groups compatible
images, and describe CUDA/Metal accurately as accelerator codec decode followed
by decoded-pixel readback and ordinary Burn upload. It must not claim direct
tensor destinations, zero-copy behavior, or performance beyond newly measured
staged-adapter workloads.
