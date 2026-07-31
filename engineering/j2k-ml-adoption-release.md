# `j2k-ml 0.8.0` adoption follow-up

Version `0.8.0` is published. This file now tracks the remaining
post-publication adoption evidence for the accelerator API; it is not a
publication announcement. Do not post externally until the user gives a new
explicit instruction naming the destination and action.

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

## Historical preparation evidence

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
again afterward. Those preparation results were superseded by the exact-SHA
release validation.

The CUDA host remained on the published `v0.7.5` release commit during that
preparation, so that checkout was not release evidence. The final hosted, CUDA,
and Metal gates later passed for the exact `v0.8.0` commit as part of the
[publish workflow](https://github.com/frames-sg/j2k/actions/runs/30425822681).

## Post-publication adoption checklist

1. Create fresh external consumers pinned to `j2k-ml = "=0.8.0"` and test the
   applicable `cpu`, `cuda`, `cpu,cuda`, `metal`, and `cpu,metal` feature sets
   without workspace path overrides.
2. Run the package smoke with registry-only third-party dependencies:

   ```bash
   cargo xtask j2k-ml-package-smoke
   ```

   Linux must compile `cpu`, `cuda`, and `cpu,cuda`; macOS must compile `cpu`,
   `metal`, and `cpu,metal`. Temporary path overrides may name only
   unpublished J2K workspace crates.
3. Replace historical direct-destination benchmark comparisons with
   content-distinct batches 1/8/32/64 for the staged accelerator adapters
   versus CPU-decode-and-upload. Record uncertainty, memory, and transfer
   counters without reusing old direct-route claims.
4. Confirm the versioned examples, guide, and benchmark evidence links resolve
   at tag `v0.8.0`.
5. Review the final notice only after every send gate in
   `engineering/burn-community-notice-draft.md` is complete.

## Community notice requirements

Any later notice must state that `j2k-ml` is independent of Burn, explain that
the application owns training batches while the codec groups compatible
images, and describe CUDA/Metal accurately as accelerator codec decode followed
by decoded-pixel readback and ordinary Burn upload. It must not claim direct
tensor destinations, zero-copy behavior, or performance beyond newly measured
staged-adapter workloads.
