# Environment variables

These environment variables change how J2K builds or runs. Only one is
stable: `OPJ_NUM_THREADS`. The others are for profiling, debugging, and trying
experimental GPU routes, and can change or disappear in any release. Anything
not listed here is internal.

Where a row says "on/off", the variable is on for `1`, `true`, `yes`, or `on`.
Where it says "any value", setting it at all turns it on.

## Command line

| Variable | Effect | Default | Stability |
| --- | --- | --- | --- |
| `OPJ_NUM_THREADS` | Decode threads for `j2k decode` and the `opj_decompress` interface when `-threads` isn't given: a count, or `ALL_CPUS`. | One thread | Stable |

## Profiling

Profile rows go to stderr. Where a variable accepts `summary`, it prints
aggregate rows instead of one row per operation.

| Variable | Effect | Default | Stability |
| --- | --- | --- | --- |
| `J2K_PROFILE_STAGES` | CPU and CUDA JPEG 2000 stage timings: `1` for rows, `summary` for aggregates. CPU rows need `j2k-native`'s `profile` feature, which is on by default. CPU HTJ2K phase timings (`ht_*_us`) cover only code blocks decoded serially, as in a single-threaded decode. | Off | Debugging |
| `J2K_JPEG_PROFILE_STAGES` | JPEG CPU stage timings: `1` for rows, `summary` for aggregates where supported. | Off | Debugging |
| `J2K_GPU_ROUTE_PROFILE` | GPU routing decisions made by `j2k` and the GPU crates: `1` for rows, `summary` for aggregates. | Off | Debugging |
| `J2K_CUDA_TRACE` | Writes a CUDA HTJ2K profile trace as JSON to this path. The file must not exist; its parent directory must. | No trace | Debugging |
| `J2K_CUDA_IDWT_TRACE` | CUDA inverse wavelet transform traces. Any value. | Off | Debugging |
| `J2K_CUDA_DISABLE_STAGE_TIMINGS` | Stops CUDA from collecting stage timings. Any value. | Timings collected | Debugging |
| `J2K_METAL_PROFILE_STAGES` | Metal stage timings: `1` for rows, `summary` for aggregates where supported. | Off | Debugging |
| `J2K_METAL_PROFILE_SIGNPOSTS` | OS signposts around Metal work, for Instruments. On/off. | Off | Debugging |
| `J2K_METAL_PROFILE_DECODE_LABEL` | Label added to Metal decode profile rows. Characters other than ASCII letters, digits, `_`, `-`, and `.` become `_`. | `unlabeled` | Debugging |
| `J2K_JPEG_METAL_FAST420_BATCH_TIMING` | Timings for the Metal JPEG fast 4:2:0 batch path: `1` for rows, `summary` for aggregates. | Off | Debugging |

The following add detail to Metal stage profiling, so they do nothing unless
`J2K_METAL_PROFILE_STAGES` is also set. All are on/off and debugging-only.

| Variable | Adds |
| --- | --- |
| `J2K_METAL_PROFILE_DECODE_SPLIT_COMMANDS` | Decode stages timed in separate command buffers |
| `J2K_METAL_PROFILE_COEFFICIENT_PREP_SPLIT_COMMANDS` | Coefficient preparation timed in separate command buffers |
| `J2K_METAL_PROFILE_CLASSIC_TIER1_DENSITY` | Classic Tier-1 encode density statistics |
| `J2K_METAL_PROFILE_CLASSIC_TIER1_RAW_PACK` | Classic Tier-1 encode raw packing |
| `J2K_METAL_PROFILE_CLASSIC_TIER1_ARITHMETIC_PACK` | Classic Tier-1 encode arithmetic packing |
| `J2K_METAL_PROFILE_CLASSIC_TIER1_SYMBOL_PLAN` | Classic Tier-1 encode symbol planning |
| `J2K_METAL_PROFILE_CLASSIC_TIER1_PASS_PLAN` | Classic Tier-1 encode pass planning, plus symbol planning |
| `J2K_METAL_PROFILE_CLASSIC_TIER1_TOKEN_EMIT` | Classic Tier-1 encode token emission |
| `J2K_METAL_PROFILE_CLASSIC_TIER1_SPLIT_TOKEN_EMIT` | Classic Tier-1 encode split token emission |
| `J2K_METAL_PROFILE_CLASSIC_TIER1_TOKEN_PACK` | Classic Tier-1 encode token packing, plus token emission |

## Experimental GPU routes

These switch GPU code paths that are still being tuned.

| Variable | Effect | Default | Stability |
| --- | --- | --- | --- |
| `J2K_METAL_CLASSIC_SELECTIVE_BYPASS` | `0` turns off selective arithmetic-coding bypass in Metal classic encode. This changes the codestream: its code blocks no longer use the bypass style. | On | Experimental |
| `J2K_METAL_CLASSIC_TIER1_GPU_TOKEN_PACK` | Packs Metal classic Tier-1 encode tokens on the GPU. On/off. | Off | Experimental |
| `J2K_METAL_CLASSIC_TIER1_SPLIT_GPU_TOKEN_PACK` | Uses the split form of that GPU token packing. On/off. | Off | Experimental |
| `J2K_METAL_CLASSIC_TIER1_SPLIT_MQ_BYTE_GPU_TOKEN_PACK` | `1` requests the split MQ-byte GPU token packing, and `0` turns it off. A request the encode job can't use returns an error. | Chosen automatically | Experimental |
| `J2K_METAL_HT_PACKET_CAPACITY` | `conservative` sizes Metal HTJ2K packet output buffers generously. Any other value sizes them tightly. | Tight | Experimental |
| `J2K_HYBRID_FLAT_CPU_TIER1` | Forces flat CPU Tier-1 batching in Metal hybrid decode. Any value except empty, `0`, or `false`. | Off | Experimental |
| `J2K_CUDA_DISABLE_COMPACT_PREENCODED` | Stops the CUDA transcode adapter from using compact pre-encoded resident HTJ2K encode. Any value. | Used where supported | Experimental |

## Building CUDA kernels

The build scripts of `j2k-cuda-runtime` and the CUDA engine crates read these
when one of their `cuda-oxide-*` features is enabled.

| Variable | Effect | Default | Stability |
| --- | --- | --- | --- |
| `J2K_CUDA_OXIDE_ARCH` | GPU architecture to build the CUDA kernels for, passed to `cargo oxide build --arch`. | `sm_80` | Experimental |
| `J2K_REQUIRE_CUDA_OXIDE_BUILD` | Makes a kernel that fails to build a build error. Without it, the build prints a warning and skips that kernel, and the CUDA paths that need it return an error at run time. | Off | Experimental |

Variables used by tests, benchmarks, CI, and `cargo xtask` are listed in the
[contributor book](../contributor-book/environment-variables.md).
