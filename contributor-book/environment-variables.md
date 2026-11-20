# Environment variables

Variables read by tests, CI, benchmarks, and xtask. Variables that change how
the library behaves at run time, such as the profiling switches, are in the
user guide's [Environment variables](../book/environment-variables.md). None
of the variables here are part of the stable API; they change when the tools
do.

Conventions:

- "Any value" means the variable only has to be set.
- Path lists use the OS separator: `:` on Unix, `;` on Windows.
- `*_INCLUDE_GENERATED` variables accept `0`, `false`, `no`, or `off` to drop
  generated fixtures.

## Test and CI gates

Without these, tests skip when a device, tool, or external input is missing.
With them, they fail.

| Variable | Effect |
| --- | --- |
| `J2K_REQUIRE_METAL_RUNTIME` | Runs the Metal runtime tests and fails them if Metal is unavailable. `cargo xtask metal-compile` refuses to run while it is set. |
| `J2K_REQUIRE_CUDA_RUNTIME` | Runs the CUDA runtime tests and fails them if CUDA is unavailable. The `j2k-cuda` encode stages also read it: an unavailable CUDA context becomes an error instead of the stage declining. |
| `J2K_REQUIRE_CUDA_OXIDE_BUILD` | Makes a CUDA-Oxide kernel that fails to build a build error, and makes CUDA tests fail instead of skip when a kernel's PTX wasn't built. Set it on CUDA test and benchmark hosts. |
| `J2K_REQUIRE_CUDA_JPEG_HARDWARE_DECODE` | Requires CUDA JPEG hardware decode in the CUDA tests and benches that cover it. |
| `J2K_REQUIRE_CUDA`, `J2K_REQUIRE_METAL` | Set to `1` to fail the `j2k-python` CUDA or MPS tests when the wheel lacks that backend or there is no device. |
| `J2K_REQUIRE_CUDA_BENCH`, `J2K_REQUIRE_METAL_BENCH` | Fail CUDA or Metal benchmark probes when the device is unavailable or doesn't dispatch. |
| `J2K_REQUIRE_OPENJPEG`, `J2K_REQUIRE_GROK` | Fail parity tests and comparator runs when the OpenJPEG or Grok tools or libraries are missing. |
| `J2K_REQUIRE_LIBJPEG_TURBO` | Fails libjpeg-turbo comparison tests when the tooling is missing, and makes a `bench-libjpeg-turbo` build fail instead of warn when `pkg-config` can't find libturbojpeg. |
| `J2K_REQUIRE_OPENJPH`, `J2K_REQUIRE_KAKADU` | Fail the optional OpenJPH or Kakadu comparator rows when `ojph_expand`, `kdu_expand`, or `kdu_compress` is missing. |
| `J2K_REQUIRE_WSI_ROOT` | Fails the external JPEG WSI tests if `J2K_WSI_ROOT` is unset or empty. |
| `J2K_REQUIRE_TRANSCODE_WSI_ROOT` | Fails transcode corpus validation if `J2K_TRANSCODE_WSI_ROOT` is unset. |
| `J2K_REQUIRE_NDPI` | Fails the NDPI passthrough test if `J2K_NDPI_PATH` is unset. |
| `J2K_GPU_SKIP_LOG` | File that collects one `gate<TAB>test` line per skipped GPU test. `cargo xtask` sets it to count skips. |
| `J2K_GPU_CACHE_ENV` | Fingerprint of the GPU driver, runtime, and compiler, computed by the GPU validation workflow for its cache key. Don't set it. |
| `J2K_CUDA_LOSSY_PARITY_PNM` | External RGB PNM used by the CUDA irreversible color transform, 9/7 DWT, and lossy encode parity tests. Default: a generated fixture. |

## External test data

All unset by default; the tests that need them skip.

| Variable | Effect |
| --- | --- |
| `J2K_WSI_ROOT` | Directory of external JPEG WSI files for integration tests. |
| `J2K_WSI_SVS_PATH` | Aperio SVS file for the Metal IDWT and in-process parity tests. |
| `J2K_TRANSCODE_WSI_ROOT` | Path list of JPEG tiles for JPEG-to-HTJ2K transcode validation and benches. |
| `J2K_TRANSCODE_WSI_TILE_LIMIT`, `J2K_TRANSCODE_WSI_MAX_PAYLOAD_BYTES` | Maximum tile count and tile payload size for transcode validation. |
| `J2K_NDPI_PATH` | NDPI slide for the passthrough test. |
| `J2K_NDPI_TILE_LIMIT`, `J2K_NDPI_MAX_PAYLOAD_BYTES` | Maximum tiles (`0` for no cap) and tile payload size for the passthrough test. |
| `J2K_ISO_CONFORMANCE_DIR` | ISO JPEG 2000 conformance files for ignored external tests. |
| `J2K_APERIO_TILE_FIXTURE` | Aperio J2K tile for the ignored lossless encode test. |
| `J2K_PROGRESSION_MATRIX_DIR` | Where the ignored `write_progression_matrix_codestreams` test in `j2k-native` writes its codestreams, for checking with other decoders. |
| `J2K_PARITY_CORPUS_MANIFEST` | Manifest for `scripts/parity-corpus-fetch.sh`; overrides its first argument. Run the script with `--help` for usage. |
| `J2K_PARITY_CORPUS_DIR` | Output directory for that script. Default: the second argument, or `corpus/wsi-samples`. |
| `J2K_PARITY_CORPUS_MAX_BYTES` | Largest file that script accepts. Default: `536870912`. |

## Comparator tools

| Variable | Effect |
| --- | --- |
| `J2K_OPENJPEG_BIN` | `opj_decompress` for the parity tests. Default: `/opt/homebrew/bin/opj_decompress`; the tests skip if it's missing. |
| `J2K_OPENJPEG_COMPRESS_BIN` | `opj_compress` for the parity tests (default `/opt/homebrew/bin/opj_compress`) and the encoder benchmark (default: `PATH`, then `/opt/homebrew/bin`, then `/usr/local/bin`). |
| `J2K_GROK_BIN` | `grk_decompress` for the parity tests. Default: `/opt/homebrew/bin/grk_decompress`. |
| `J2K_GROK_COMPRESS_BIN` | `grk_compress` for the parity tests and encoder benchmark. Default: next to `J2K_GROK_BIN`, then `/opt/homebrew/bin`, then `/usr/local/bin`. |
| `J2K_GROK_SOURCE` | Grok source tree for the in-process comparator build. Default: `/tmp/grok-j2k`. |
| `J2K_GROK_ROOT` | Grok install or library root. Default: `<J2K_GROK_SOURCE>/build/bin`, then `pkg-config`. |
| `J2K_OPENHTJ2K_DEC_BIN` | Pinned `open_htj2k_dec` for the HT+RGN case of the T.803 CPU encoder matrix. |
| `J2K_OPENHTJ2K_SOURCE_DIR` | Clean OpenHTJ2K checkout at the pinned commit. The runner checks its origin, tag, commit, and status. |
| `J2K_OPENHTJ2K_LIB_DIR` | Pinned OpenHTJ2K static library for the in-process comparator. |
| `J2K_OPENJPH_SOURCE_DIR`, `J2K_OPENJPH_LIB_DIR` | Clean OpenJPH checkout and static library for the in-process comparator. |
| `J2K_OPENJPH_EXPAND_BIN` | `ojph_expand` for the fixture comparator (default: `PATH`, `/opt/homebrew/bin`, `/usr/local/bin`) and the encode matrix (default: the pinned build under `target/reference/`, then `PATH`). |
| `J2K_OPENJPH_COMPRESS_BIN` | `ojph_compress` for the encode matrix. Default: the pinned build, then `PATH`. |
| `J2K_KDU_EXPAND_BIN`, `J2K_KDU_COMPRESS_BIN` | Kakadu `kdu_expand` and `kdu_compress`. Default: `PATH`, `/opt/homebrew/bin`, `/usr/local/bin`. |

`scripts/prepare-openhtj2k-reference.sh` and
`scripts/prepare-openjph-reference.sh` print the OpenHTJ2K and OpenJPH
variables for the builds they make.

Build scripts set `J2K_GROK_VERSION`, `J2K_GROK_LIB_DIR`,
`J2K_OPENHTJ2K_VERSION`, `J2K_OPENJPH_VERSION`, `J2K_LIBJPEG_TURBO_VERSION`,
and `J2K_LIBJPEG_TURBO_LIBDIR` to record what was linked. Don't set them.

## Benchmark harnesses

### JPEG benches

These configure the `j2k-jpeg` `compare` and `encode_cpu` benches and the
JPEG Metal benches.

| Variable | Effect |
| --- | --- |
| `J2K_BENCH_INPUTS` | Path list of extra JPEG inputs. |
| `J2K_BENCH_SKIP_REPO_INPUTS` | Any value: drop the two built-in images, so only `J2K_BENCH_INPUTS` is measured. |
| `J2K_BENCH_VERIFY_OUTPUTS` | Any value: check `compare` outputs against libjpeg-turbo before timing. |
| `J2K_BENCH_FULL_FRAME_ONLY`, `J2K_BENCH_PREPARED_ONLY` | Any value: stop `compare` after the full-frame or the prepared-decode groups. |
| `J2K_BENCH_COLD_CONTEXT` | Any value: add groups that create and drop a context and output inside each iteration. |
| `J2K_BENCH_MATCHED_BATCH_ONLY` | Any value: run only the matched-worker batch sweeps. Add `J2K_BENCH_BATCH_MIXED` for mixed inputs, or `J2K_BENCH_BATCH_DISTINCT_STORAGE` to give each item its own storage. |
| `J2K_BENCH_BATCH_SIZES`, `J2K_BENCH_BATCH_WORKERS` | Batch sizes and worker counts for the matched batch comparison. Default: `16,64` and `1,2,4`. |
| `J2K_ENCODE_BENCH_INPUTS` | Path list of PGM/PPM inputs for `encode_cpu`. |
| `J2K_ENCODE_BENCH_QUALITIES` | JPEG qualities for those inputs. Default: `50,90,100`. |
| `J2K_ENCODE_BENCH_BATCH` | Any value: run only the batch comparison with libjpeg-turbo. `J2K_ENCODE_BENCH_BATCH_MIXED` adds mixed batches of 16 and 64. |
| `J2K_ENCODE_BENCH_V3` | Any value: run only the libjpeg-turbo 3 comparison. `J2K_ENCODE_BENCH_V3_FILTER` selects cases by name. |
| `J2K_REPORT_ITERS` | Iterations for the report-style JPEG benches. |
| `J2K_ALLOC_REPORT` | Print the allocation report in the CPU JPEG encode bench. |
| `J2K_FORCE_FULL_FRAME` | Classify every benchmark input as full-frame. |
| `J2K_JPEG_TILE_BATCH_SIZE` | Tile batch size in the comparison runs. |
| `J2K_JPEG_ENCODE_BENCH_DIM`, `J2K_JPEG_ENCODE_BENCH_BATCH`, `J2K_JPEG_ENCODE_BENCH_QUALITY` | Image size, batch size, and quality for the JPEG Metal baseline encode bench. |
| `J2K_JPEG_HARNESS_BASELINE_SHADERS` | Directory with another copy of the JPEG Metal shader files. The ignored `decode_kernel_stage_profile` test compiles it as a baseline column next to the in-tree kernels; it must match the current host ABI. |
| `J2K_JPEG_TEXTURE_SMALL_BATCHES` | Any value: run the small-batch and tile-size controls in the ignored `texture_scheduling_experiment` test. |

### GPU JPEG benches

| Variable | Effect |
| --- | --- |
| `J2K_GPU_BENCH_JPEG` | JPEG fixture for the GPU upload and decode benches. Default: a generated fixture. |
| `J2K_CUDA_BENCH_JPEG` | CUDA-only override of `J2K_GPU_BENCH_JPEG`. |
| `J2K_GPU_BENCH_SMALL_FIXTURE` | Any value: use a small built-in JPEG instead of a generated one. |
| `J2K_GPU_BENCH_DIM`, `J2K_GPU_BENCH_BATCH`, `J2K_GPU_BENCH_BATCH_DIM` | Generated image size (`N` or `WxH`), batch count, and batch tile size. |
| `J2K_GPU_BENCH_RESTART_INTERVAL` | Restart interval of the generated fixture. |
| `J2K_GPU_BENCH_SUBSAMPLING`, `J2K_CUDA_BENCH_SUBSAMPLING` | `420`, `422`, or `444` for the generated CUDA fixture (default `420`). The Metal bench always uses 4:2:0. |

### Comparator harnesses

`jp2k_fixture_compare`:

| Variable | Effect |
| --- | --- |
| `J2K_FIXTURE_COMPARE_MODE` | `portable-native` (default), `portable-emulated`, or `capability`. See [Benchmarks](benchmarks.md#running-the-comparators). |
| `J2K_FIXTURE_COMPARE_INPUT_DIRS` | Path list of directories scanned for `.j2k`, `.j2c`, `.jp2`, `.jph`, and `.jhc` files. `J2K_FIXTURE_COMPARE_INPUT_DIR` is the older single-directory form. |
| `J2K_FIXTURE_COMPARE_MANIFEST` | TSV manifest. Required columns: `path`, `corpus_category`. Optional: `corpus_name`, `license_status`, `encode_command`, `input_fnv1a64`, `source_fnv1a64`, `codec`, `container` (`raw-codestream`, `j2k`, `j2c`, `jp2`, `jph`, `jhc`). Publication needs the hash, codec, and container. Give raw and wrapped variants of one source the same `source_fnv1a64`. |
| `J2K_FIXTURE_COMPARE_INCLUDE_GENERATED` | Drop the generated smoke fixtures. |
| `J2K_FIXTURE_COMPARE_CASE_BATCH_SIZES`, `J2K_FIXTURE_COMPARE_MIXED_BATCH_SIZES` | Per-fixture batch sizes (default `1`) and mixed-batch sizes (default `1,16,256,1024`). The older `J2K_FIXTURE_COMPARE_BATCH_SIZES` and `J2K_FIXTURE_COMPARE_BATCH_SIZE` set both. |
| `J2K_FIXTURE_COMPARE_REPEATS`, `J2K_FIXTURE_COMPARE_THREADS` | Repeats and workers. |
| `J2K_INCLUDE_OPENJPH`, `J2K_INCLUDE_KAKADU` | Add the OpenJPH or Kakadu command-line rows. |

`jp2k_encode_compare`:

| Variable | Effect |
| --- | --- |
| `J2K_ENCODE_COMPARE_INPUT_DIRS` | Path list of directories of 8-bit gray or RGB `.pgm`, `.ppm`, `.pnm`, `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, or `.bmp` images, staged as PNM outside the timed loop. |
| `J2K_ENCODE_COMPARE_MANIFEST` | TSV manifest. Required: `path`, `corpus_category`. Optional: `corpus_name`, `license_status`, `source_command`, `input_fnv1a64` (the hash of the decoded pixels, required for publication). |
| `J2K_ENCODE_COMPARE_INCLUDE_GENERATED` | Drop the generated smoke images. |
| `J2K_ENCODE_COMPARE_CASE_BATCH_SIZES`, `J2K_ENCODE_COMPARE_MIXED_BATCH_SIZES` | Per-source (default `1`) and mixed-batch (default `1,16,256,1024`) sizes. `J2K_ENCODE_COMPARE_BATCH_SIZES` sets both. |
| `J2K_ENCODE_COMPARE_REPEATS` | Repeats. |
| `J2K_ENCODE_COMPARE_ENCODERS` | Comma-separated filter: `j2k`, `openjpeg`, `grok`, `kakadu`. Any filter blocks publication. |
| `J2K_OPENJPH_MATRIX_REPEATS` | Encode repeats per cell of `--openjph-matrix`. Default: `3`. |

`jp2k_batch_compare` and `jp2k_roi_batch_compare`:

| Variable | Effect |
| --- | --- |
| `J2K_BATCH_COMPARE_THREADS` | Thread budget for every decoder; also sizes Grok's pool and J2K's Rayon pool. Exits if `RAYON_NUM_THREADS` differs. Default: available parallelism. |
| `J2K_BATCH_COMPARE_REPEATS` | Repeats. |
| `J2K_BATCH_COMPARE_MAX_ABS_DIFF` | Largest allowed sample difference from any comparator, checked before timing. Default: `1`. Use `0` for lossless inputs. |
| `J2K_BATCH_COMPARE_KEEP_GOING` | Any value: record a failing comparator as `failed: ...` and leave it out of the timing, instead of exiting. |
| `J2K_BATCH_COMPARE_DECODER` | Time only one decoder (`j2k`, `openjpeg`, `grok`, `openhtj2k`, `openjph`). Parity still runs for all. |
| `J2K_BATCH_COMPARE_EXCLUDE` | Comma-separated comparators to skip entirely, such as OpenJPH on Part 1 inputs. |
| `J2K_BATCH_COMPARE_J2K_COLD` | Any value: drop J2K's retained workspaces before every timed sample. |
| `J2K_BATCH_COMPARE_VIEWPORT` | `<reduce>:<count>:<size>:<seed>`: turn the single input into `count` region requests of `size` x `size` pixels at reduction `reduce` (0-6). OpenJPH is excluded. |
| `J2K_ROI_COMPARE_REPEATS`, `J2K_ROI_COMPARE_THREADS` | Repeats and workers for `jp2k_roi_batch_compare`. |

### DICOM decode and Metal step profiling

These configure `jp2k_dicom_bench`, the Metal `dicom_decode_bench` example, and
the ignored `single_image_step_profile` test.

| Variable | Effect |
| --- | --- |
| `J2K_DICOM_BENCH_MODE` | `serial`, `parallel`, or `series`. Default: all. |
| `J2K_DICOM_BENCH_REPEATS` | Timed repeats per row; series rows use at most five. Default: `15`. |
| `J2K_DICOM_BENCH_RAW_DIR` | Directory of `<stem>.RAW` little-endian reference pixels to check against. |
| `J2K_DICOM_BENCH_THREADS` | Workers for the series row. Default: available parallelism. |
| `J2K_DICOM_BENCH_SERIES_REPEAT` | Copies of each input in the series row. Default: `8`. |
| `J2K_DICOM_BENCH_REUSE_DECODER` | Any value: reuse one prepared decoder for repeated single-image decodes. |
| `J2K_METAL_BENCH_ONLY` | `metal-single`, `auto`, `metal-into`, or `metal-batch`. Default: CPU and Metal single and batch rows. |
| `J2K_METAL_BENCH_REPEATS`, `J2K_METAL_BENCH_BATCH` | Timed repeats (default `11`) and batch size (default `16`). |
| `J2K_METAL_BENCH_REUSE` | Any value: reuse one decoder, plan, and device payloads for `metal-single`. |
| `J2K_METAL_BENCH_DISTINCT` | Any value: `metal-batch` uses distinct frames re-encoded as lossless HTJ2K. |
| `J2K_METAL_BENCH_CAPTURE` | Absolute `.gputrace` path for one warmed-up `metal-into` decode. Needs `MTL_CAPTURE_ENABLED=1`. |
| `J2K_STEP_PROFILE_FILES` | Path list of Gray16 codestreams to profile. Default: generated 512x512, 1024x1024, and 3072x4096 images. |
| `J2K_STEP_PROFILE_INTO` | Any value: decode into one reused destination. |
| `J2K_STEP_PROFILE_LOSSY` | Any value: generate irreversible 9/7 inputs. |
| `J2K_STEP_PROFILE_COMPARE_IDWT` | Any value: also profile the IDWT with single-pass tile kernels off. |

### CUDA JPEG 2000 benches

| Variable | Effect |
| --- | --- |
| `J2K_CUDA_DECODE_INPUT_DIRS`, `J2K_CUDA_DECODE_MANIFEST` | External HTJ2K fixtures and their manifest (same columns as `J2K_FIXTURE_COMPARE_MANIFEST`). |
| `J2K_CUDA_DECODE_INCLUDE_GENERATED` | Drop the generated decode fixtures. |
| `J2K_CUDA_DECODE_FORMATS` | Output formats, such as `gray8,rgb8,rgba8`. |
| `J2K_CUDA_DECODE_CASE_BATCH_SIZES`, `J2K_CUDA_DECODE_BATCH_SIZES` | Per-fixture and mixed-batch sizes. Keep the per-fixture list short on large corpora. |
| `J2K_CUDA_DECODE_SAMPLE_SIZE`, `J2K_CUDA_DECODE_MEASUREMENT_SECONDS` | Criterion samples (at least 10) and seconds per row (default 1). |
| `J2K_CUDA_ENCODE_INPUT_DIRS`, `J2K_CUDA_ENCODE_MANIFEST` | Staged PGM/PPM sources and their manifest (`path` and `input_fnv1a64`). |
| `J2K_CUDA_ENCODE_INCLUDE_GENERATED` | Drop the generated host-input encode row. |
| `J2K_CUDA_ENCODE_SAMPLE_SIZE` | Criterion samples, at least 10. |
| `J2K_CUDA_FINAL_STORE_PROFILE` | Run only the final IDWT and store timing matrix. Needs `J2K_REQUIRE_CUDA_BENCH=1`. |

The CUDA decode profile example measures compressed input to finished
device-resident output, and reports median and p95. Its `stages` report
synchronizes at stage boundaries, so use the main numbers for throughput.

| Variable | Effect |
| --- | --- |
| `J2K_CUDA_PROFILE_INPUT` | One encoded gray or RGB input up to 16 bits. |
| `J2K_CUDA_PROFILE_INPUTS` | Path list of inputs or directories, cycled to fill the batch. All must decode to the same pixel format. |
| `J2K_CUDA_PROFILE_API` | `surface` (default) or `owned` (`CudaBatchDecoder`, supports multi-tile inputs). |
| `J2K_CUDA_PROFILE_LAYOUT` | `nhwc` (default), or `nchw` with the `owned` API. |
| `J2K_CUDA_PROFILE_BATCH_SIZE`, `J2K_CUDA_PROFILE_ITERATIONS` | Batch size (default 128) and measured iterations after three warm-ups (default 100). |
| `J2K_CUDA_PROFILE_DOWNLOAD` | `1` to include host readback in each sample. Default: `0`. |
| `J2K_CUDA_PROFILE_MAX_ABS_DIFF` | Largest sample difference from the CPU allowed during validation. Default `0`; raise it only for irreversible inputs. |
| `J2K_CUDA_PROFILE_WIDTH`, `J2K_CUDA_PROFILE_HEIGHT`, `J2K_CUDA_PROFILE_FORMAT`, `J2K_CUDA_PROFILE_TRANSFORM`, `J2K_CUDA_PROFILE_CODING`, `J2K_CUDA_PROFILE_DECOMPOSITION_LEVELS` | The generated input: size (default 512x512), `rgb8` or `gray16`, `53` or `97`, `ht` or `classic`, and levels (default 1). |
| `J2K_CUDA_PROFILE_WRITE_FIXTURE`, `J2K_CUDA_PROFILE_WRITE_REFERENCE` | Save the input, or the CPU reference pixels, for comparison with another decoder. Needs exactly one input; never overwrites. |

### Metal JPEG 2000 benches

| Variable | Effect |
| --- | --- |
| `J2K_METAL_DECODE_INPUT_DIRS`, `J2K_METAL_DECODE_MANIFEST` | External fixtures and their manifest. JP2/JPH wrappers are reported as skipped. |
| `J2K_METAL_DECODE_INCLUDE_GENERATED` | Drop the generated decode fixtures. Published numbers set this. |
| `J2K_METAL_ENCODE_INPUT_DIRS`, `J2K_METAL_ENCODE_MANIFEST` | Staged PGM/PPM sources for the auto-routing encode bench. |
| `J2K_METAL_ENCODE_INCLUDE_GENERATED` | Drop the generated host-input rows. |
| `J2K_METAL_ENCODE_RESIDENT_MAX_ESTIMATED_OUTPUT_BYTES` | Skip resident encode rows whose output would be larger. Default: `2147483648`. |
| `J2K_METAL_CAPTURE_PATH` | Absolute, new `.gputrace` path for the ignored GPU capture test. Needs `MTL_CAPTURE_ENABLED=1`. |
| `J2K_SAMPLED_CORPUS` | Directory with `level0-tile0.j2k` through `level2-tile7.j2k` for the sampled-color characterization test. Needs an optimized build and Metal. |
| `J2K_SAMPLED_COUNT`, `J2K_SAMPLED_LEVEL_COUNT`, `J2K_SAMPLED_GROUP_COUNT` | Tiles per level, levels, and groups from that corpus. Default: `8`, `3`, `1`. |
| `J2K_LARGE_REGION_INPUT` | Large codestream for the ignored large-region CPU and Metal batch tests. |

### Other benches

| Variable | Effect |
| --- | --- |
| `J2K_PACKET_DIRECTORY_SOURCE` | Input for the `packet_directory` group of the `j2k-native` `direct_cpu` bench. Default: a generated 1536x1024 image. |
| `J2K_PACKET_DIRECTORY_REGION_SIDE`, `J2K_PACKET_DIRECTORY_THREADS` | Region side and workers. Default: `256`, `4`. |
| `J2K_PACKET_DIRECTORY_CASE` | Run only the named case. |
| `J2K_TRANSCODE_METAL_PROFILE_STAGES` | Profile rows from the `j2k-transcode-metal` `dct97` bench: `1` for rows, `summary` for aggregates. Some groups log inside the timed loop, so leave it unset for timing. |
| `J2K_ML_BATCH_INPUT_MODE` | `distinct` (default) or `repeated` inputs for the `j2k-ml` batch benches. |
| `J2K_ML_BATCH_PROCESS_MODE` | `criterion` (default) or `profile` for the CUDA and Metal batch benches. |
| `J2K_MPSGRAPH_BENCH_ITERATIONS` | Samples per cell of the MPSGraph bench. Default: `30`, or `1` with debug assertions. |

### Adoption and routing runs

The `GPU benchmarks` workflow reads the first three rows from repository
variables and sets the rest itself. The runs fail if any is unset.

| Variable | Effect |
| --- | --- |
| `J2K_ADOPTION_FIXTURES`, `J2K_ADOPTION_MANIFEST` | Decode fixtures and manifest for `adoption-benchmark --fixtures` and `--manifest`. |
| `J2K_ADOPTION_ENCODE_FIXTURES`, `J2K_ADOPTION_ENCODE_MANIFEST` | Staged PNM sources and manifest for `--encode-fixtures` and `--encode-manifest`. |
| `J2K_AUTO_ROUTING_MANIFEST`, `J2K_AUTO_ROUTING_ROOT` | The routing workload manifest and the corpus root its paths resolve against. |
| `J2K_AUTO_ROUTING_EVIDENCE` | Where the CUDA and Metal routing benches write their raw results. |
| `J2K_AUTO_ROUTING_CANDIDATE_SHA`, `J2K_AUTO_ROUTING_HARDWARE`, `J2K_AUTO_ROUTING_DRIVER` | Commit, GPU, and driver recorded in the routing results. |

## Xtask

| Variable | Effect |
| --- | --- |
| `J2K_COVERAGE_BASE` | Base revision for `cargo xtask coverage`. Default: `origin/$GITHUB_BASE_REF` if set, else `HEAD^`. Full validation sets it to the commit before the push; GPU validation takes it from its `coverage-base-ref` input. |
| `J2K_FUZZ_RUNS` | Runs per target for `cargo xtask fuzz-run`. Default: `1000`. |
| `J2K_FUZZ_MAX_TOTAL_TIME_SECONDS` | libFuzzer's `-max_total_time` for each target. Default: not passed. |
| `J2K_FUZZ_TARGET` | Target triple for `cargo fuzz run --target`. Default: the nightly host target. |
| `J2K_SEMVER_TOOLCHAIN` | Rejected by `cargo xtask semver`, which pins its toolchain in source. |
