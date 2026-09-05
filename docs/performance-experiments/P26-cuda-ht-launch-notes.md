# P26: CUDA HT encode launch size

On 2026-09-04, reducing the HT encoder from 128 threads per block to 32 or 1
did not improve representative full-encode throughput on an RTX 4070 SUPER.
Both candidates were rejected. Production launch geometry and device code
remain unchanged. The reusable product benchmark and independent parity checks
are retained.

## Results

Times are Criterion mean estimates in milliseconds; smaller is better.
The JSON records contain the 95% confidence intervals and input/output hashes.

| Input, batch | 128 threads | 32 threads | 1 thread |
|---|---:|---:|---:|
| RGB8 128×128, B1 | 34.187 | 34.090 | 33.212 |
| Gray8 512×512, B1 | 30.782 | 30.735 | 30.786 |
| RGB8 512×512, B1 | 84.829 | 85.057 | 85.039 |
| RGB8 640×480, B1 | 82.125 | 82.287 | 82.425 |
| RGB8 1024×1024, B1 | 98.375 | 99.288 | 99.098 |
| RGB8 512×512, B16 | 1358.4 | 1363.6 | 1366.5 |

The one-thread RGB128 result was 2.85% faster, but the representative RGB512
single-image and B16 cells were 0.25% and 0.60% slower. This is insufficient
evidence for a general launch change or a new small-image dispatch rule.
Thirty-two threads also provided no representative improvement.

The existing resident B64 code-block microbenchmarks likewise showed little
change: cleanup was 2.9466/2.9316/2.9315 ms and refinement was
4.0740/4.0871/4.0569 ms for 128/32/1 threads. These are secondary observations;
the promotion decision uses the complete product with verified outputs.

## Work and correctness

The product matrix uses `patterned_gray8` and `patterned_rgb8`, unsigned 8-bit
samples, reversible 5/3, three decomposition levels, HT cleanup, and 64×64
code blocks. B16 means sixteen sequential public facade encodes using one
reused `CudaEncodeStageAccelerator`, matching the current sequential boundary.
Every batch item increments its first input byte by the item index.

Timing includes host input, CUDA work, packetization and host codestream output.
Fixture generation, independent CPU decode and first-use context/module loading
are outside timing. Every cell encodes twice before timing, compares exact
codestream bytes, independently decodes both copies and checks every pixel.
Framed SHA-256 hashes match across all three configurations. Each input or output
item is prefixed by its byte length as a little-endian u64; items are concatenated
in batch order. The environment corpus hash identifies the priority RGB512 B16
input. No external image corpus was configured.

The added hardware test covers 1×1, 32×32, 64×64 and 65×17 blocks, batches 1
and 17, and one/three coding passes. Cleanup bytes and metadata must equal the
scalar encoder; both modes independently decode to exact fixture coefficients.
It passed with 128, 32 and 1 thread.

An initial version of this new test incorrectly demanded byte equality for
three-pass CUDA and scalar encodes. CUDA's documented legacy refinement route
uses cleanup bitplane 2; the scalar default selects bitplane 1. The initial
1×1 mismatch was therefore a comparison of different work. The corrected test
uses zero or signed odd magnitudes at least 5, which both CUDA pass modes
represent exactly, and checks refinement with the independent decoder. Existing
SigProp and stuffed MagRef regression tests remain intact. Production semantics
were not changed to satisfy this test.

## Trace and next experiment

A separate Nsight Systems 2025.6.3 CUDA trace ran the six matrix preflights
(42 image encodes) plus the existing availability probe. Of aggregate GPU
kernel time, `j2k_htj2k_encode_codeblocks` consumed 98.1% (2.678 s across 1,225
launches); packetization consumed 1.6%. All other kernels combined were below
0.3%. Actual GPU memcpy time was approximately 11.25 ms across 5,802 copies.

The HT grid had only 1 block in 477 launches and 4 blocks in 336 launches;
other observed grid sizes were 2, 6, 16, 20 and 64. The trace reports 78
registers per thread and no static shared memory for this kernel. It does not
establish achieved occupancy or absence of spills. No hardware traffic,
occupancy, spill or cache counters were collected.

This supports testing HT job batching across subbands, components and images,
using the existing multi-input engine API to expose more independent blocks
per launch. It also makes the serial HT implementation a better kernel target
than DWT fusion for these workloads. These are hypotheses for the next
experiment, not measured speedups. Any deferred completion must retain device
buffers and preserve synchronization on errors; removing waits alone is unsafe.

The trace includes cold contexts and module loads, CPU preflight decoding, and
profiler overhead. CUDA API times include waiting for kernels and must not be
added to GPU times as independent work. In particular, context creation and
packetization averages from a separate two-call `J2K_PROFILE_STAGES=summary`
probe are not steady-state stage measurements. The per-cell JSON fields remain
null where no matching measurement was collected.

## Environment and reproduction

Measurements used the existing checkout on the CUDA host at commit
`23f17707e86e230718c5691e8ba3ae98a1b2ff30`, branch
`feat/htj2k-encode-candidates`. It was clean before the experiment. Only the
benchmark, tests and temporary host launch selector were added for the run.
No worktree, dependency change, driver setting or GPU clock change was used.

- AMD Ryzen 7 5800X3D, 8 cores / 16 logical CPUs; 50,518,622,208 bytes RAM.
- RTX 4070 SUPER, 12,282 MiB; compute capability 8.9; driver 610.88.
- Ubuntu 24.04.4 on WSL2 Linux 5.15.153.1-microsoft-standard-WSL2.
- Rust 1.96.1 (`31fca3adb`, 2026-06-26), LLVM 22.1.2, cargo-oxide 0.2.1.
- CUDA 13.2.78, real sm_80 PTX running on sm_89; `release-bench` profile.

An initial build could not find libclang in the SSH environment. The repository's
`scripts/configure-cuda-bindgen.sh` resolved the existing LLVM 18 installation;
no package installation was needed. `target/cuda-p26/run.py` loads the generated
`build.env`, prepends the user's Cargo bin directory and sets
`J2K_REQUIRE_CUDA_RUNTIME=1`, `J2K_REQUIRE_CUDA_OXIDE_BUILD=1` and
`J2K_REQUIRE_CUDA_BENCH=1` before running its arguments.

```sh
GITHUB_ENV=target/cuda-p26/build.env bash scripts/configure-cuda-bindgen.sh
python3 target/cuda-p26/run.py cargo bench --profile release-bench \
  -p j2k-cuda --features cuda-runtime --bench htj2k_encode -- \
  'j2k_cuda_ht_encode_launch_product|j2k_cuda_htj2k_codeblock_microkernel/cuda_resident' \
  --sample-size 15 --warm-up-time 1 --measurement-time 3 \
  --save-baseline p26-threads-128
```

The treatment processes ran the same compiled executable with `--bench`, the
same filter/sampling arguments, `--baseline p26-threads-128`, and
`J2K_CUDA_HT_ENCODE_BLOCK_THREADS=32` or `1`. The temporary selector changed only
the host block dimension in `htj2k_encode_codeblock_launch_geometry`; the grid
remained one block per job. Its removed patch is preserved under the ignored
remote `target/cuda-p26/launch-switch.patch`. The historical variable has no
effect in the retained source. Reproducing treatment requires that experimental
patch or an equivalent temporary constant change, not merely setting the variable.

The three arms ran as separate processes in order 128, 32, 1, without randomized
repeated rounds. Each cell used 15 samples, a 1-second warmup and a requested
3-second measurement window. B16 needed about 20.5 seconds for its 15 samples.
The JSON estimates retain the printed precision of Criterion logs. Noise and
order effects limit interpretation of sub-percent changes. Conclusions cover
these synthetic encode workloads on this WSL2 GPU, not CUDA decode, Classic J2K,
lossy 9/7 or other hardware.

Trace command, using the same executable:

```sh
python3 target/cuda-p26/run.py nsys profile --sample=none --cpuctxsw=none \
  --trace=cuda --force-overwrite=false -o target/cuda-p26/encode-trace \
  target/release-bench/deps/htj2k_encode-7403319a80a6ec79 \
  --test profile_preflight_only
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum \
  --format csv --output target/cuda-p26/trace-stats target/cuda-p26/encode-trace.nsys-rep
```

The deliberately unmatched filter executes setup/parity probes without timed
Criterion loops. Raw logs, trace, SQLite export and artifact hashes remain under
`target/cuda-p26/` on the CUDA host. The executable SHA-256 was
`267b92b7e623431867516984dfd31108a7cbfc3995583a32427d56d7dbb6407a`;
HT encode PTX SHA-256 was
`bed507bdc733e00b11ecc659e1156e8d68d5f1e11c76a10a358e4b908928e81f`.

The original production `kernels.rs` was restored and verified against SHA-256
`0c441aac01d2c698e374d00de633f9396cfaec728ebb2dbd2f7b84b8eddbca08`.

Validate the two retained records with:

```sh
cargo xtask gpu-experiment validate docs/performance-experiments/P26-cuda-ht-launch-1.json
cargo xtask gpu-experiment validate docs/performance-experiments/P26-cuda-ht-launch-32.json
```

## Verification

On the CUDA host, with the required-runtime/build environment described above:

```sh
cargo test --profile gpu-quick -p j2k-cuda --features cuda-runtime \
  --lib --tests --examples -- --test-threads=1
cargo clippy --profile gpu-quick -p j2k-cuda --features cuda-runtime \
  --benches --tests -- -D warnings -A clippy::disallowed_methods -A clippy::disallowed_macros
cargo fmt -p j2k-cuda -- --check
```

The test command passed 276 tests with no failures or ignored tests. The expanded
shape test also passed separately under all three experimental launch sizes.
The complete matrix passed its pixel and hash checks in all three benchmark
processes. The two JSON records pass the experiment validator. The Mac Clippy
check passes, but its placeholder PTX build is not hardware validation.
Linux Clippy passed after compiling real PTX, and CUDA package formatting passed
on both hosts. A final hardware rerun passed after extracting the independent
decoder helper. On the CUDA host, `cargo xtask repo-lint` passed all 101 checks.

Local `cargo xtask repo-lint` passed 100 of 101 checks. Its remaining failure
was a concurrent, unrelated Metal source-size change:
`encode_bitstream_classic_core.metal` had 1,648 lines against a reviewed ceiling
of 1,635. A full local `git diff --check` also observed a concurrent trailing
blank line in `mct.metal`; the scoped CUDA/docs diff and the remote diff pass.
Those other edits were preserved. Semantic navigation did not resolve the
requested symbol, so targeted source reads were used; a workspace LSP server
with ambiguous ownership was left running.
