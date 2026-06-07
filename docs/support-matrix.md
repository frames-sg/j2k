# Support Matrix

This matrix describes what is stable enough for broad use on the `0.5.x` line,
what remains experimental, and what evidence is required for public benchmark
claims.

## Stable APIs

| Surface | Status | Scope |
|---------|--------|-------|
| `signinum` | Stable facade | Recommended import surface for applications that need JPEG, JPEG 2000 / HTJ2K, tile decompression, and shared core contracts. |
| `signinum-core` | Stable library | Pixel formats, backend requests, passthrough rows, scratch/context contracts, and device-submission traits. |
| `signinum-jpeg` | Stable library | JPEG inspect, CPU decode for the current 8-bit Huffman subset, ROI/scaled WSI tile decode for supported sequential inputs, progressive 8-bit full/ROI/scaled/region-scaled CPU decode, initial full-image/ROI/scaled/region-scaled 12-bit extended sequential grayscale decode to `Gray16` or expanded `Rgb16`/`Rgba16` including restart-coded grayscale streams, initial 12-bit progressive grayscale decode to `Gray16` or expanded `Rgb16`/`Rgba16`, initial 12-bit extended/progressive APP14 RGB 4:4:4/4:2:2/4:2:0 and YCbCr 4:4:4/4:2:2/4:2:0 decode to `Rgb16`/`Rgba16` including restart-coded extended color streams, initial 12-bit extended/progressive CMYK/YCCK 4:4:4/4:2:2/4:2:0 decode to `Rgb16`/`Rgba16` including restart-coded extended/progressive four-component streams, initial full-image/ROI/scaled/region-scaled lossless SOF3 8-bit grayscale decode to `Gray8`, 16-bit grayscale decode to `Gray16`, 8-bit APP14 RGB and 8-bit YCbCr 4:4:4 decode to `Rgb8` plus full/ROI/scaled/region-scaled `Rgba8`, 16-bit APP14 RGB plus YCbCr 4:4:4 decode to `Rgb16` plus full/ROI/scaled/region-scaled `Rgba16` for predictors 1-7 including restart-coded grayscale, APP14 RGB, and YCbCr streams, and initial even-width non-restart 16-bit APP14 RGB/YCbCr 4:2:2 full/ROI/scaled/region-scaled plus session-batch `Rgb16`/`Rgba16` decode, 8-bit SOF3 grayscale/RGB/YCbCr row streaming, and 16-bit SOF3 grayscale `Gray16` plus APP14 RGB/YCbCr `Rgb16` row streaming for 4:4:4 streams; batch decode, baseline JPEG fixture/fallback encode, and initial 8-bit sequential CMYK/YCCK CPU conversion for `Rgb8` and `Rgba8` full/ROI/scaled/region-scaled output plus RGB row streaming, including 4:4:4 and initial 4:2:2/4:2:0 fixture coverage. Broader CPU parity for nonstandard 12-bit color sampling layouts, stronger non-constant 12-bit oracle fixtures, SOF3 sampled color restart/other 16-bit color layouts, and broader four-component malformed coverage is tracked in [`docs/jpeg-support-phases`](jpeg-support-phases/README.md); unsupported lossless SOF3 sampled RGB/YCbCr layouts and one malformed non-leading-max CMYK sampling shape currently report explicit not-implemented/capability rejection. |
| `signinum-j2k` | Stable library | JPEG 2000 / HTJ2K inspect, CPU decode, ROI/scaled decode, row-bounded decode, batch decode, and lossless encode. |
| `signinum-tilecodec` | Stable library | Deflate, Zstd, LZW, and uncompressed container-tile decompression primitives. |
| `signinum-cli` | Stable behavior | `signinum inspect <file>` header inspection. CLI behavior is tested, but Rust API semver tooling does not apply to the binary surface. |

## Experimental APIs

| Surface | Status | Promotion notes |
|---------|--------|-----------------|
| `signinum-jpeg-metal`, `signinum-j2k-metal` | Experimental adapters | APIs are hardening around resident device surfaces, backend routing, and benchmark evidence. |
| `signinum-jpeg-cuda`, `signinum-j2k-cuda` | Experimental adapters | CUDA device-memory paths require a CUDA driver and optional runtime feature support. `signinum-jpeg-cuda` reserves strict `BackendRequest::Cuda` JPEG decode for Signinum-owned full-frame RGB8 4:2:0, 4:2:2, and 4:4:4 kernels with session-cached packet state and caller-managed output-buffer support; ROI, scaled, and non-RGB8 strict CUDA JPEG requests fail loudly. `signinum-j2k-cuda` reserves `BackendRequest::Cuda` for strict CUDA-resident HTJ2K codestream decode; the current resident path covers full-frame, ROI, reduced-resolution, and ROI+scaled HTJ2K `Gray8`, `Gray16`, `Rgb8`, `Rgba8`, `Rgb16`, and `Rgba16`, with pinned compressed-payload upload, reusable device HT tables, and separate 5/3 vs 9/7 IDWT entrypoints. CPU-staged J2K uploads use explicit CPU-staged APIs. `encode_j2k_lossless_with_cuda` exposes strict CUDA HTJ2K encode stages and treats every backend preference as device-required, including forward RCT/ICT, forward 5/3 and 9/7 DWT, sub-band quantization, batched HT cleanup code-block encode with cooperative magnitude reduction, first-inclusion packetization with HT refinement pass headers and cooperative packet payload assembly, later-layer packet contributions for code blocks already included in prior packets, and deferred first inclusion after empty or non-empty prior packets through flattened persistent tag-tree state. |
| `signinum-transcode`, `signinum-transcode-metal`, `signinum-transcode-cuda` | Experimental transcode | Promotion requires synthetic and real JPEG sampling coverage, native and external HTJ2K acceptance, documented error histograms, loud unsupported-mode failures, and benchmark evidence. |
| `signinum-j2k-native`, `signinum-profile`, `signinum-cuda-runtime` | Published support crates | Public because stable crates depend on them, but not the primary user-facing API. |

## Non-API Tooling

| Surface | Status | Scope |
|---------|--------|-------|
| `signinum-test-support` | Unpublished dev helper | Workspace-versioned synthetic image and benchmark generators for tests, benches, and examples. |
| `xtask` | Unpublished workspace tool | Repository-local automation for tests, docs, benches, fuzz builds, coverage, and packaging. |

## Supported Workflows

| Workflow | Supported through | Notes |
|----------|-------------------|-------|
| Header inspection | `signinum::jpeg`, `signinum::j2k`, `signinum-cli` | Does not decode pixels. |
| Caller-owned CPU decode | `signinum::jpeg`, `signinum::j2k` | Portable default path for supported codestream classes. |
| ROI and scaled tile decode | `signinum::jpeg`, `signinum::j2k` | Used for WSI tile pipelines that bring their own container, cache, and pyramid logic. JPEG ROI/scaled support covers supported sequential inputs, progressive 8-bit CPU output projection, initial grayscale 12-bit extended/progressive `Gray16`/`Rgb16`/`Rgba16` projection including restart-coded extended grayscale streams, initial 12-bit APP14 RGB 4:4:4/4:2:2/4:2:0 plus YCbCr 4:4:4/4:2:2/4:2:0 `Rgb16`/`Rgba16` projection including restart-coded extended color streams, and initial 12-bit extended/progressive CMYK/YCCK 4:4:4/4:2:2/4:2:0 `Rgb16`/`Rgba16` projection including restart-coded extended/progressive four-component streams; Metal acceleration for progressive shapes remains gated on benchmark-proven resident wins. |
| Row-bounded J2K decode | `signinum::j2k` | Intended for memory-bound decode workflows. |
| Tile decompression | `signinum::tilecodec` | Container compression only; container parsing is out of scope. |
| Whole-slide container parsing | Out of scope | Use `statumen` before invoking signinum codecs. |
| DICOM VL WSI export | Out of scope | Use `wsi-dicom` after codec or passthrough policy decisions. |

## Backend support

| Backend | Status | Limits |
|---------|--------|--------|
| CPU | Supported | Always available and used as the portable fallback for `BackendRequest::Auto`. |
| Metal | Experimental | Apple Silicon macOS only. Device-output APIs are available for selected adapter paths. New JPEG classes require CPU parity and benchmark evidence before resident Metal routing. |
| CUDA | Experimental | Requires a CUDA driver. Runtime allocation/copy helpers are enabled through adapter `cuda-runtime` features. |

## Security and fuzzing

Security posture is documented in [`SECURITY.md`](../SECURITY.md) and
[`docs/unsafe-audit.md`](unsafe-audit.md). Fuzz targets live under the codec
crate `fuzz/` directories and are compile-checked by `cargo xtask fuzz-build`.
CI also gates tests, docs, dependency policy, unsafe inventory drift, and
packaging. Malformed input must return structured errors or deterministic
panics only where a documented invariant is violated; there must be no silent
failure paths for externally supplied image data.

## Benchmark publication

A published benchmark is any benchmark result or comparator claim included in
the root README, crate READMEs, release notes, or documentation. Published
claims must record the command, host, compiler, crate revision, input source,
command environment, skipped rows, and comparator availability, comparator
version, and comparator path when a comparator is part of the claim.

OpenJPEG and Grok comparator claims are publication-eligible only when the J2K
comparator run prints comparator availability, comparator version, comparator
path, input source, and the `SIGNINUM_J2K_COMPARE_THREADS` value used for batch
rows. OpenJPEG and Grok decoder calls must remain single-threaded per tile, and
batch comparator rows must use the same outer worker count as signinum. When
single-threaded OpenJPEG or Grok rows are compared against signinum, the
published output must include explicit signinum serial rows for the same shape.

Classic J2K rows generated by signinum's in-repo encoder are labeled
`signinum-generated`. Those rows are useful for development, but they do not
support public OpenJPEG or Grok performance claims. Public comparator claims
must use OpenJPEG-generated inputs or an external corpus with source and
licensing recorded.

No-silent-skip signoff commands:

```sh
SIGNINUM_REQUIRE_OPENJPEG=1 cargo test -p signinum-j2k-compare --test in_process_parity
SIGNINUM_REQUIRE_OPENJPEG=1 cargo test -p signinum-j2k --test openjpeg_parity
SIGNINUM_REQUIRE_OPENJPEG=1 SIGNINUM_REQUIRE_GROK=1 cargo xtask j2k-bench-signoff
```

If Grok is unavailable on the local host, report that the Grok publication gate
was not run. Do not publish Grok comparator claims from skipped rows.

## MSRV

The current MSRV is Rust 1.88, pinned by [`rust-toolchain.toml`](../rust-toolchain.toml).
Lowering MSRV requires a passing all-features compile audit and the required CI
gate set on the selected lower toolchain before release.

The `0.5.x` audit candidates are Rust 1.85, 1.88, 1.90, 1.92, 1.93, and 1.94.
Rust 1.85 is blocked by `fearless_simd 0.3.0`: that dependency uses
`#[target_feature]` on a safe NEON dispatch function, which Rust 1.85 rejects.
Rust 1.88 is the oldest candidate that passed the workspace all-features
compile audit for this line.
