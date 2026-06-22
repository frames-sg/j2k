# Environment Variables

This is the supported `J2K_*` environment-variable surface for the
workspace. Variables not listed here are internal symbols, generated metadata,
or test-only implementation details and must not be treated as user controls.

Stability values:

- Stable: supported for the v0.5.x release line.
- Experimental: accepted for diagnostics or adapter tuning, but may change
  before 1.0.
- Test/CI: supported only for repository tests, CI, and release validation.
- Benchmark: supported only for benchmark harnesses and benchmark signoff.
- Generated: emitted by a build script for version reporting; do not set by
  hand unless reproducing the build-script contract.

## Library Runtime And Profiling

| Variable | Effect | Default | Stability |
| --- | --- | --- | --- |
| `J2K_GPU_ROUTE_PROFILE` | Emits facade/adapter GPU route decisions. Use `1` for rows or `summary` for aggregate rows. | Disabled | Experimental |
| `J2K_JPEG_PROFILE_STAGES` | Emits JPEG CPU profiling rows. Use `1` for rows or `summary` where supported. | Disabled | Experimental |
| `J2K_PROFILE_STAGES` | Emits native/CUDA J2K profiling rows. Use `1` for rows or `summary` where supported. | Disabled | Experimental |
| `J2K_CUDA_TRACE` | Writes CUDA HTJ2K profile trace JSON to the configured path. | No trace file | Experimental |
| `J2K_CUDA_IDWT_TRACE` | Enables CUDA IDWT trace/profile output. | Disabled | Experimental |
| `J2K_CUDA_DISABLE_STAGE_TIMINGS` | Disables CUDA stage timing collection for benchmark runs. | Timings enabled | Experimental |
| `J2K_CUDA_DISABLE_DWT97_FUSED_COLUMN_QUANTIZE` | Disables the fused CUDA DWT 9/7 column quantize path. | Fused path enabled when supported | Experimental |
| `J2K_CUDA_DISABLE_COMPACT_PREENCODED` | Forces the CUDA transcode adapter to decline compact preencoded resident HT encode support. | Compact resident support enabled when supported | Experimental |
| `J2K_JPEG_METAL_FAST420_BATCH_TIMING` | Emits JPEG Metal fast 4:2:0 batch timing profile rows when set to `1`. | Disabled | Experimental |
| `J2K_METAL_PROFILE_STAGES` | Enables J2K Metal stage profile rows when set to `1`. | Disabled | Experimental |
| `J2K_METAL_PROFILE_SIGNPOSTS` | Enables J2K Metal OS signposts when set to `1` and stage profiling is enabled. | Disabled | Experimental |
| `J2K_METAL_PROFILE_DECODE_LABEL` | Labels J2K Metal decode profile rows. Non-alphanumeric characters are sanitized. | `unlabeled` | Experimental |
| `J2K_METAL_PROFILE_DECODE_SPLIT_COMMANDS` | Adds split-command decode timing rows when J2K Metal stage profiling is enabled. | Disabled | Experimental |
| `J2K_METAL_PROFILE_COEFFICIENT_PREP_SPLIT_COMMANDS` | Adds split-command coefficient-prep timing rows when J2K Metal stage profiling is enabled. | Disabled | Experimental |
| `J2K_METAL_PROFILE_CLASSIC_TIER1_DENSITY` | Emits classic Tier-1 density profiling when J2K Metal stage profiling is enabled. | Disabled | Experimental |
| `J2K_METAL_PROFILE_CLASSIC_TIER1_RAW_PACK` | Emits classic Tier-1 raw-pack profiling when J2K Metal stage profiling is enabled. | Disabled | Experimental |
| `J2K_METAL_PROFILE_CLASSIC_TIER1_ARITHMETIC_PACK` | Emits classic Tier-1 arithmetic-pack profiling when J2K Metal stage profiling is enabled. | Disabled | Experimental |
| `J2K_METAL_PROFILE_CLASSIC_TIER1_SYMBOL_PLAN` | Emits classic Tier-1 symbol-plan profiling when J2K Metal stage profiling is enabled. | Disabled | Experimental |
| `J2K_METAL_PROFILE_CLASSIC_TIER1_PASS_PLAN` | Emits classic Tier-1 pass-plan profiling and also enables symbol-plan profiling. | Disabled | Experimental |
| `J2K_METAL_PROFILE_CLASSIC_TIER1_TOKEN_EMIT` | Emits classic Tier-1 token-emission profiling when J2K Metal stage profiling is enabled. | Disabled | Experimental |
| `J2K_METAL_PROFILE_CLASSIC_TIER1_SPLIT_TOKEN_EMIT` | Emits classic Tier-1 split-token-emission profiling when J2K Metal stage profiling is enabled. | Disabled | Experimental |
| `J2K_METAL_PROFILE_CLASSIC_TIER1_TOKEN_PACK` | Emits classic Tier-1 token-pack profiling and also enables token-emission profiling. | Disabled | Experimental |
| `J2K_TRANSCODE_METAL_PROFILE_STAGES` | Enables transcode Metal profiling in the DCT 9/7 benchmark harness. | Disabled | Benchmark |

## Experimental Backend Routing

| Variable | Effect | Default | Stability |
| --- | --- | --- | --- |
| `J2K_METAL_HT_PACKET_CAPACITY` | Overrides J2K Metal HT packet buffer capacity. Invalid values are ignored by the fallback parser. | Built-in capacity | Experimental |
| `J2K_METAL_CLASSIC_SELECTIVE_BYPASS` | Set to `0` to disable classic selective arithmetic coding bypass in the Metal resident style flags. | Selective bypass enabled | Experimental |
| `J2K_METAL_CLASSIC_TIER1_GPU_TOKEN_PACK` | Requests the classic Tier-1 GPU token-pack route when supported. | Disabled | Experimental |
| `J2K_METAL_CLASSIC_TIER1_SPLIT_GPU_TOKEN_PACK` | Requests the split classic Tier-1 GPU token-pack route. | Disabled | Experimental |
| `J2K_METAL_CLASSIC_TIER1_SPLIT_MQ_BYTE_GPU_TOKEN_PACK` | Set to `1` to request, or `0` to disable, the split MQ-byte GPU token-pack route. | Auto | Experimental |
| `J2K_HYBRID_FLAT_CPU_TIER1` | Forces flat CPU Tier-1 batching for J2K Metal hybrid decode when set to a truthy value accepted by the adapter. | Adapter default | Experimental |
| `J2K_CUDA_OXIDE_ARCH` | Overrides the cuda-oxide build target when a `j2k-cuda-runtime/cuda-oxide-*` feature is enabled. | `sm_80` | Experimental |
| `J2K_CUDA_USE_OXIDE_J2K_DECODE_STORE` | Routes supported J2K decode store and inverse-MCT CUDA kernels through the cuda-oxide PTX when `j2k-cuda-runtime/cuda-oxide-j2k-decode-store` is enabled and PTX was generated. | Built-in CUDA C PTX | Experimental |
| `J2K_CUDA_USE_OXIDE_J2K_DEQUANTIZE` | Routes supported J2K HTJ2K dequantize CUDA kernels through the cuda-oxide PTX when `j2k-cuda-runtime/cuda-oxide-j2k-dequantize` is enabled and PTX was generated. | Built-in CUDA C PTX | Experimental |
| `J2K_CUDA_USE_OXIDE_J2K_ENCODE` | Routes supported J2K encode-stage CUDA kernels and HTJ2K encoded-byte compaction through the cuda-oxide PTX when `j2k-cuda-runtime/cuda-oxide-j2k-encode` is enabled and PTX was generated. | Built-in CUDA C PTX | Experimental |
| `J2K_CUDA_USE_OXIDE_J2K_IDWT` | Routes supported generic J2K inverse-DWT CUDA kernels through the cuda-oxide PTX when `j2k-cuda-runtime/cuda-oxide-j2k-idwt` is enabled and PTX was generated. | Built-in CUDA C PTX | Experimental |
| `J2K_CUDA_USE_OXIDE_TRANSCODE` | Routes supported reversible 5/3 and irreversible 9/7 transcode CUDA kernels through the cuda-oxide PTX when `j2k-cuda-runtime/cuda-oxide-transcode` is enabled and PTX was generated. This includes staged, batched, code-block quantize, and fused i16 9/7 paths. | Built-in CUDA C PTX | Experimental |
| `J2K_REQUIRE_CUDA_OXIDE_COPY_U8` | Requires cuda-oxide CopyU8 PTX generation when `j2k-cuda-runtime/cuda-oxide-copy-u8` is enabled. | Disabled | Experimental |
| `J2K_REQUIRE_CUDA_OXIDE_J2K_DECODE_STORE` | Requires cuda-oxide J2K decode store/inverse-MCT PTX generation when `j2k-cuda-runtime/cuda-oxide-j2k-decode-store` is enabled. | Disabled | Experimental |
| `J2K_REQUIRE_CUDA_OXIDE_J2K_DEQUANTIZE` | Requires cuda-oxide J2K HTJ2K dequantize PTX generation when `j2k-cuda-runtime/cuda-oxide-j2k-dequantize` is enabled. | Disabled | Experimental |
| `J2K_REQUIRE_CUDA_OXIDE_J2K_ENCODE` | Requires cuda-oxide J2K encode-stage and HTJ2K compaction PTX generation when `j2k-cuda-runtime/cuda-oxide-j2k-encode` is enabled. | Disabled | Experimental |
| `J2K_REQUIRE_CUDA_OXIDE_J2K_IDWT` | Requires cuda-oxide J2K generic inverse-DWT PTX generation when `j2k-cuda-runtime/cuda-oxide-j2k-idwt` is enabled. | Disabled | Experimental |
| `J2K_REQUIRE_CUDA_OXIDE_TRANSCODE` | Requires cuda-oxide reversible 5/3 transcode PTX generation when `j2k-cuda-runtime/cuda-oxide-transcode` is enabled. | Disabled | Experimental |

## Test And CI Gates

| Variable | Effect | Default | Stability |
| --- | --- | --- | --- |
| `J2K_REQUIRE_OPENJPEG` | Makes OpenJPEG parity tests fail instead of skip when OpenJPEG tools are unavailable. | Skip unavailable comparator paths | Test/CI |
| `J2K_REQUIRE_GROK` | Makes Grok parity tests fail instead of skip when Grok tools or libraries are unavailable. | Skip unavailable comparator paths | Test/CI |
| `J2K_REQUIRE_LIBJPEG_TURBO` | Makes libjpeg-turbo comparison tests fail instead of skip when the bench feature/tooling is unavailable. | Skip unavailable comparator path | Test/CI |
| `J2K_REQUIRE_CUDA_RUNTIME` | Makes CUDA tests and NVIDIA comparison harnesses require a usable CUDA runtime instead of skipping. | Skip runtime-only CUDA paths | Test/CI |
| `J2K_REQUIRE_CUDA_HTJ2K_STRICT` | Requires CUDA HTJ2K strict validation and makes CUDA kernel build failures fatal in the runtime build script. | Non-strict when runtime unavailable | Test/CI |
| `J2K_REQUIRE_CUDA_KERNEL_BUILD` | Makes CUDA kernel compilation failures fatal in the runtime build script. | Kernel build may be skipped when unsupported | Test/CI |
| `J2K_REQUIRE_CUDA_JPEG_HARDWARE_DECODE` | Requires CUDA JPEG hardware decode coverage in relevant CUDA tests/benches. | Hardware decode may skip | Test/CI |
| `J2K_REQUIRE_CUDA_BENCH` | Makes CUDA benchmark probes fail instead of skip when CUDA is unavailable or does not dispatch. | Skip unavailable CUDA benchmark paths | Benchmark |
| `J2K_REQUIRE_METAL_BENCH` | Makes Metal benchmark probes fail instead of skip when Metal is unavailable or does not dispatch. | Skip unavailable Metal benchmark paths | Benchmark |
| `J2K_REQUIRE_NV_BASELINE_BUILD` | Makes the standalone GPU baseline harness require its CUDA build dependencies. | Baseline build may skip unavailable pieces | Test/CI |
| `J2K_RUN_HOSTED_J2K_METAL_RUNTIME_TESTS` | Allows hosted macOS CI to run J2K Metal runtime tests that are otherwise skipped there. | Hosted runtime tests skipped | Test/CI |
| `J2K_REQUIRE_WSI_ROOT` | Makes external JPEG WSI tests fail if `J2K_WSI_ROOT` is missing or empty. | Skip external WSI tests | Test/CI |
| `J2K_REQUIRE_TRANSCODE_WSI_ROOT` | Makes transcode corpus validation require `J2K_TRANSCODE_WSI_ROOT`. | Skip external transcode WSI corpus | Test/CI |
| `J2K_REQUIRE_NDPI` | Makes NDPI passthrough tests fail if `J2K_NDPI_PATH` is missing. | Skip NDPI passthrough test | Test/CI |

## External Data And Comparator Paths

| Variable | Effect | Default | Stability |
| --- | --- | --- | --- |
| `J2K_WSI_ROOT` | Path list root for external JPEG WSI integration tests. | Not set | Test/CI |
| `J2K_TRANSCODE_WSI_ROOT` | Path list root for external JPEG-to-HTJ2K transcode corpus validation. | Not set | Test/CI |
| `J2K_TRANSCODE_WSI_TILE_LIMIT` | Maximum number of external WSI tiles used by transcode corpus validation. | Built-in validation default | Test/CI |
| `J2K_TRANSCODE_WSI_MAX_PAYLOAD_BYTES` | Maximum external WSI tile payload accepted by transcode corpus validation. | Built-in validation default | Test/CI |
| `J2K_NDPI_PATH` | Path to an NDPI slide for passthrough tests. | Not set | Test/CI |
| `J2K_NDPI_TILE_LIMIT` | Maximum NDPI tiles inspected by the passthrough test; `0` means no explicit tile-count cap. | Test default | Test/CI |
| `J2K_NDPI_MAX_PAYLOAD_BYTES` | Maximum NDPI tile payload accepted by the passthrough test. | Test default | Test/CI |
| `J2K_ISO_CONFORMANCE_DIR` | Path to ISO J2K conformance vectors for ignored/external tests. | Not set | Test/CI |
| `J2K_APERIO_TILE_FIXTURE` | Path to an Aperio J2K tile fixture for the ignored lossless encode test. | Not set | Test/CI |
| `J2K_PARITY_CORPUS_MANIFEST` | Manifest path used by `scripts/parity-corpus-fetch.sh`. | `corpus/wsi-samples/manifest.json` or first script argument | Test/CI |
| `J2K_PARITY_CORPUS_DIR` | Output directory used by `scripts/parity-corpus-fetch.sh`. | `corpus/wsi-samples` or second script argument | Test/CI |
| `J2K_PARITY_CORPUS_MAX_BYTES` | Maximum accepted byte size for each downloaded parity-corpus fixture. | `536870912` | Test/CI |
| `J2K_OPENJPEG_BIN` | Override for OpenJPEG `opj_decompress` in J2K parity tests. | `opj_decompress` on `PATH` | Test/CI |
| `J2K_OPENJPEG_DECOMPRESS_BIN` | Override for OpenJPEG `opj_decompress` in benchmark reports. | `opj_decompress` on `PATH` | Benchmark |
| `J2K_OPENJPEG_COMPRESS_BIN` | Override for OpenJPEG `opj_compress` in J2K parity tests. | `opj_compress` on `PATH` | Test/CI |
| `J2K_GROK_BIN` | Override for Grok `grk_decompress` in J2K parity tests. | `grk_decompress` on `PATH` | Test/CI |
| `J2K_GROK_COMPRESS_BIN` | Override for Grok `grk_compress` in J2K parity tests. | `grk_compress` on `PATH` | Test/CI |
| `J2K_GROK_ROOT` | Path to Grok installation/library root for in-process comparator builds and benchmark reports. | Not set | Test/CI |
| `J2K_GROK_SOURCE` | Path to Grok source used by the in-process comparator build script. | Not set | Test/CI |
| `J2K_GROK_VERSION` | Build-script emitted Grok version metadata consumed by the comparator crate. | `unavailable` if not emitted | Generated |
| `J2K_GROK_LIB_DIR` | Build-script emitted Grok library path metadata consumed by the comparator crate. | `unavailable` if not emitted | Generated |

## Benchmark Harnesses

| Variable | Effect | Default | Stability |
| --- | --- | --- | --- |
| `J2K_BENCH_INPUTS` | Path-list of JPEG inputs for JPEG and Metal benchmark/report harnesses. | Harness-specific generated fixtures | Benchmark |
| `J2K_BENCH_INPUT_SOURCE` | Input-source label recorded by `cargo xtask bench-report`. | Falls back to `J2K_BENCH_INPUTS`, otherwise `not recorded` | Benchmark |
| `J2K_BENCH_COMMAND` | Command label recorded by `cargo xtask bench-report`. | `not recorded` | Benchmark |
| `J2K_BENCH_SKIPPED_ROWS` | Semicolon-separated skipped-row reasons recorded by `cargo xtask bench-report`. | Empty | Benchmark |
| `J2K_BENCH_JPEG_DIR` | Directory of JPEG tiles for NVIDIA baseline and transcode comparison benches. | Harness-specific fixture fallback or required by workflow | Benchmark |
| `J2K_BENCH_SVS` | GPU workflow input SVS path used to extract benchmark JPEG tiles when `J2K_BENCH_JPEG_DIR` is unset. | Not set | Benchmark |
| `J2K_REPORT_ITERS` | Iteration count for custom report-style JPEG benchmarks. | Harness default | Benchmark |
| `J2K_ALLOC_REPORT` | Enables allocation report output in the CPU JPEG encode benchmark. | Disabled | Benchmark |
| `J2K_FORCE_FULL_FRAME` | Forces benchmark classification to full-frame mode. | Auto classification | Benchmark |
| `J2K_JPEG_BATCH_THREADS` | Overrides JPEG benchmark batch thread count. | Rayon/runtime default | Benchmark |
| `J2K_JPEG_TILE_BATCH_SIZE` | Overrides JPEG tile batch size in benchmark comparison runs. | Harness default | Benchmark |
| `J2K_JPEG_ENCODE_BENCH_DIM` | Image dimensions for the JPEG Metal baseline encode benchmark. | Harness default | Benchmark |
| `J2K_JPEG_ENCODE_BENCH_BATCH` | Batch size for the JPEG Metal baseline encode benchmark. | Harness default | Benchmark |
| `J2K_JPEG_ENCODE_BENCH_QUALITY` | Quality for the JPEG Metal baseline encode benchmark. | Harness default | Benchmark |
| `J2K_GPU_BENCH_JPEG` | JPEG fixture path for GPU JPEG upload/decode benchmarks. | Generated fixture unless strict bench requires a file | Benchmark |
| `J2K_CUDA_BENCH_JPEG` | CUDA-specific JPEG fixture path; falls back to `J2K_GPU_BENCH_JPEG` in CUDA JPEG benches. | Not set | Benchmark |
| `J2K_GPU_BENCH_SMALL_FIXTURE` | Allows GPU JPEG benches to use small generated fixtures. | Disabled | Benchmark |
| `J2K_GPU_BENCH_DIM` | Generated GPU benchmark dimensions, usually `N` or `WIDTHxHEIGHT`. | Harness default | Benchmark |
| `J2K_GPU_BENCH_BATCH` | Generated GPU benchmark batch count. | Harness default | Benchmark |
| `J2K_GPU_BENCH_BATCH_DIM` | Generated GPU benchmark batch tile dimensions. | Harness default | Benchmark |
| `J2K_GPU_BENCH_RESTART_INTERVAL` | Restart interval for generated GPU JPEG upload benchmark fixtures. | Harness default | Benchmark |
| `J2K_GPU_BENCH_SUBSAMPLING` | Generated GPU JPEG benchmark subsampling, accepted values depend on the harness. | Harness default | Benchmark |
| `J2K_CUDA_BENCH_SUBSAMPLING` | CUDA-specific generated JPEG benchmark subsampling; falls back with `J2K_GPU_BENCH_SUBSAMPLING`. | Harness default | Benchmark |
| `J2K_JPEG_COMPARE_QUALITY` | Quality for the standalone NVIDIA JPEG comparison fixture. | Harness default | Benchmark |
| `J2K_JPEG_COMPARE_ITERS` | Iteration count for the standalone NVIDIA JPEG comparison fixture. | Harness default | Benchmark |
| `J2K_JPEG_COMPARE_WARMUP` | Warmup iteration count for the standalone NVIDIA JPEG comparison fixture. | Harness default | Benchmark |
| `J2K_JPEG_COMPARE_SUBSAMPLING` | Subsampling for the standalone NVIDIA JPEG comparison fixture. | Falls back to `J2K_CUDA_BENCH_SUBSAMPLING` | Benchmark |
| `J2K_JPEG_COMPARE_PATTERN` | Generated image pattern for the standalone NVIDIA JPEG comparison fixture. | Harness default | Benchmark |
| `J2K_COMPARE_THREADS` | Thread count for J2K comparator signoff and benchmark reports. | Comparator default or `not set` in reports | Benchmark |
| `J2K_BATCH_COMPARE_REPEATS` | Repeat count for `jp2k_batch_compare`. | Tool default | Benchmark |
| `J2K_BATCH_COMPARE_THREADS` | Worker count for `jp2k_batch_compare`. | Tool default | Benchmark |
| `J2K_ROI_COMPARE_REPEATS` | Repeat count for `jp2k_roi_batch_compare`. | Tool default | Benchmark |
| `J2K_ROI_COMPARE_THREADS` | Worker count for `jp2k_roi_batch_compare`. | Tool default | Benchmark |
| `J2K_CUDA_DECODE_FORMATS` | Comma-separated CUDA J2K decode benchmark output formats such as `gray8,rgb8,rgba8`. | Harness default | Benchmark |
| `J2K_CUDA_DECODE_BATCH_SIZES` | Comma-separated CUDA J2K decode benchmark batch sizes. | Harness default | Benchmark |
| `J2K_CUDA_PROFILE_BATCH_SIZE` | Batch size for the CUDA HTJ2K decode profile example. | Example default | Benchmark |
| `J2K_CUDA_PROFILE_ITERATIONS` | Iteration count for the CUDA HTJ2K decode profile example. | Example default | Benchmark |
| `J2K_LEVEL1_CUDA_HT_MIN_MPS` | Level-1 CUDA HT throughput floor used by GPU validation workflow. | Workflow default `350` | Benchmark |
| `J2K_LEVEL1_CUDA_HT_MIN_SPEEDUP_VS_NVIDIA` | Level-1 CUDA HT speedup floor versus NVIDIA baseline in GPU validation workflow. | Workflow default `4.0` | Benchmark |
| `J2K_LEVEL2_CUDA_HT_MIN_MPS` | Level-2 CUDA HT throughput floor used by GPU validation workflow. | Workflow default `60` | Benchmark |
| `J2K_LEVEL2_CUDA_HT_MIN_SPEEDUP_VS_NVIDIA` | Level-2 CUDA HT speedup floor versus NVIDIA baseline in GPU validation workflow. | Workflow default `1.10` | Benchmark |

## Xtask And Release Tooling

| Variable | Effect | Default | Stability |
| --- | --- | --- | --- |
| `J2K_FUZZ_RUNS` | Number of runs passed to each `cargo xtask fuzz-run` target. | `1000` | Test/CI |
| `J2K_FUZZ_MAX_TOTAL_TIME_SECONDS` | Optional libFuzzer max total time for `cargo xtask fuzz-run`. | Not passed | Test/CI |
| `J2K_FUZZ_TARGET` | Target triple passed to `cargo fuzz run --target` by `cargo xtask fuzz-run`. | Nightly host target from `rustc -vV` | Test/CI |
| `J2K_SEMVER_TOOLCHAIN` | Rust toolchain used by `cargo xtask semver`. | `stable` | Test/CI |

CI overrides the fuzz defaults to 512 runs / 60 seconds for pull requests and
20,000 runs / 900 seconds for the scheduled long fuzz job.
