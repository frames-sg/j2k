# Benchmarks

Published results, with the machines they ran on, are in
[Performance](../book/performance.md). This chapter is about producing numbers
you can trust: which inputs to use, how the harnesses keep comparisons fair,
and what a report has to say.

A speed benchmark says nothing about conformance. That comes from the
[T.803 tests](testing.md#t803-conformance).

## What there is

- **Criterion benches** in each crate's `benches/`, for components and the
  public APIs. `cargo xtask bench-build --lane host|cuda|metal|all` checks
  that they compile. Time them with `--profile release-bench`.
- **Comparator harnesses** in `j2k-compare`, which decode or encode the same
  bytes through J2K and other codecs:
  - `jp2k_batch_compare`: in-process batch decode against OpenJPEG, Grok,
    OpenHTJ2K, and OpenJPH. This produced the cross-codec decode numbers in
    [Performance](../book/performance.md).
  - `jp2k_roi_batch_compare`: region batches.
  - `jp2k_fixture_compare`: per-fixture and mixed-batch decode against
    OpenJPEG and Grok, with optional OpenJPH and Kakadu command-line rows.
  - `jp2k_encode_compare`: encode against the OpenJPEG and Grok command-line
    encoders, plus `--openjph-matrix` for HTJ2K interoperability.
  - `jp2k_dicom_bench`: DICOM-style series decode.
- **The adoption bundle** (`cargo xtask adoption-*`), which runs the
  comparators and the GPU benches on pinned external corpora and refuses to
  render a report that isn't fair. Use it for numbers you intend to publish.
- **The performance guard** (`cargo xtask j2k-perf-guard`), which catches
  regressions against a fixed baseline.

Environment variables for every harness are in
[Environment variables](environment-variables.md#benchmark-harnesses).

## Corpora

Generated fixtures are fine for smoke tests and development. Published
numbers need external inputs: conformance files, other projects' test data,
natural images encoded with recorded settings, and domain tiles.

| Corpus | Use | Source |
| --- | --- | --- |
| ISO/IEC 15444-4 (T.803) | Reference outputs for the conformance classes. Copyrighted: fetch with `cargo xtask t803 fetch`, never commit or upload. | Pinned in `corpus/j2k-conformance/t803-v3.toml` |
| OpenJPEG test data | Real JP2/J2K edge cases | <https://github.com/uclouvain/openjpeg-data>, pinned at `39524bd3a601d90ed8e0177559400d23945f96a9` |
| OpenJPH | HTJ2K and JPH variants | <https://github.com/aous72/OpenJPH> |
| jpylyzer test files | JP2 parser robustness, valid and invalid files. Not for throughput. | <https://github.com/openpreserve/jpylyzer-test-files> |
| Kodak | Small RGB natural images | Kodak PhotoCD mirrors |
| ImageCompression.info, Tecnick/TESTIMAGES, CLIC | Larger natural images | <https://imagecompression.info/test_images/>, TESTIMAGES, <https://www.compression.cc/>, or the `clic2025` set of <https://github.com/imazen/codec-corpus> |
| WSI, DICOM, medical tiles | The workloads most users care about | Your own, partner, or public TCIA/IDC exports |

Keep downloads under `target/` or another ignored directory unless the
license lets us commit them. Never commit protected health information or
restricted data.

Natural-image sets are encoded to J2K and HTJ2K before decode benchmarking.
`adoption-materialize` does that for you and records the settings:

```bash
cargo run -p xtask --features adoption -- adoption-materialize \
  --encode-fixtures target/j2k-public-corpora/kodak \
  --source-command "downloaded-from-r0k-us-kodak" \
  --license-status redistributable \
  --corpus-name kodak --corpus-category natural-image \
  --out-dir target/j2k-public-corpora/materialized-kodak
```

It stages each 8-bit gray or RGB image (at least 128x128) as PGM/PPM, encodes
classic and HTJ2K lossless codestreams plus JP2 and JPH wrappers, checks the
CPU round trip, and writes `fixtures.tsv` and `encode-fixtures.tsv`.
`adoption-curate` does the same for native compressed corpora such as
OpenJPEG's: it copies only files J2K, OpenJPEG, and (if present) Grok can all
fully decode, and lists the rest in `skipped.tsv`. `adoption-manifest` writes
manifests for directories you've already prepared.

## Harness rules

The comparators are built so that every codec does the same work. These rules
apply to `jp2k_batch_compare`, and the other harnesses follow the same
pattern:

- **Same input.** Every decoder gets the same in-memory bytes, loaded before
  timing.
- **Check pixels before timing.** Each decoder's output is compared with
  J2K's: exact for lossless, within `J2K_BATCH_COMPARE_MAX_ABS_DIFF` otherwise
  (one sample value by default). A decoder that fails stops the run, or with
  `J2K_BATCH_COMPARE_KEEP_GOING` is recorded as failed and left out of the
  timing.
- **Same output work.** Every decoder writes once, directly into a buffer
  allocated the same way inside the timed region. All the C and C++ shims
  share the row packers in `crates/j2k-compare/src/j2k_shim_common.h`.
- **Same thread budget.** `J2K_BATCH_COMPARE_THREADS` is the budget for every
  decoder. The harness sizes J2K's Rayon pool to it and exits if
  `RAYON_NUM_THREADS` disagrees. A one-image batch gives the whole budget to
  each decoder's own threading; a larger batch splits images across workers
  with the same static chunking for everyone, and each worker decodes one
  image serially.
- **Rotate the order.** Decoder order rotates between repeats, so no decoder
  always runs on a warm cache. `jp2k_fixture_compare` also builds batch inputs
  as separate owned copies outside the timed loop, and forces J2K to decode
  each image serially, since OpenJPEG and Grok run one thread per image there.

For published numbers we run three rounds per cell, each in a fresh process,
with the set order rotated between rounds. A decoder's time is the median over
processes of each process's median, and a ratio is computed within each
process before taking the median.

## Running the comparators

Build the pinned reference libraries, then run the batch harness with the same
budget for every decoder:

```bash
scripts/prepare-openjph-reference.sh
scripts/prepare-openhtj2k-reference.sh
export J2K_OPENJPH_SOURCE_DIR="$PWD/target/reference/openjph-0.33.0"
export J2K_OPENJPH_LIB_DIR="$J2K_OPENJPH_SOURCE_DIR/build-reference/src/core"
export J2K_OPENHTJ2K_SOURCE_DIR="$PWD/target/t803/openhtj2k-v0.23.4"
export J2K_OPENHTJ2K_LIB_DIR="$J2K_OPENHTJ2K_SOURCE_DIR/build-reference"
# Grok: set J2K_GROK_SOURCE and J2K_GROK_ROOT to a Release build.
cargo build --profile release-bench -p j2k-compare --bin jp2k_batch_compare

J2K_BATCH_COMPARE_THREADS=8 RAYON_NUM_THREADS=8 J2K_BATCH_COMPARE_REPEATS=11 \
J2K_BATCH_COMPARE_MAX_ABS_DIFF=0 J2K_BATCH_COMPARE_KEEP_GOING=1 \
  target/release-bench/jp2k_batch_compare <set-dir> <image-count>
```

For region and zoom work, pass one image and set
`J2K_BATCH_COMPARE_VIEWPORT=<reduce>:<count>:<size>:<seed>`.

The in-repo OpenJPEG comparator links `openjpeg-sys`'s vendored OpenJPEG,
built without `NDEBUG`. For a fair OpenJPEG number, link an upstream release
build instead and say so in the report.

`jp2k_fixture_compare` walks directories for `.j2k`, `.j2c`, `.jp2`, `.jph`,
and `.jhc` files and fails if a configured directory has none:

```bash
J2K_REQUIRE_OPENJPEG=1 J2K_REQUIRE_GROK=1 \
J2K_FIXTURE_COMPARE_INCLUDE_GENERATED=0 \
J2K_FIXTURE_COMPARE_MANIFEST=corpus/vendor/fixtures.tsv \
J2K_FIXTURE_COMPARE_INPUT_DIRS="corpus/vendor/openjpeg-data:corpus/vendor/openjph:corpus/vendor/kodak-htj2k" \
cargo run -p j2k-compare --release --bin jp2k_fixture_compare
```

`J2K_FIXTURE_COMPARE_MODE` picks what is compared. Publish only
`portable-native` (the default), which keeps operations that J2K, OpenJPEG,
and Grok all do natively. `portable-emulated` adds task-equivalent rows
labeled with their `decode_method`, such as OpenJPEG region+scaled done as a
scaled decode and a crop; `capability` shows feature coverage with explicit
skips. OpenJPH and Kakadu rows (`J2K_INCLUDE_OPENJPH`, `J2K_INCLUDE_KAKADU`)
run command-line tools that write files, so report them separately from the
in-process rows.

`jp2k_encode_compare` stages each source image as PNM outside the timed loop
and gives the same bytes to every encoder through its command line, J2K's
included. OpenJPEG and Grok get an explicit classic lossless profile
(`-n 3 -b 64,64 -p LRCP`, one thread), and each output is decoded and checked
for that profile. Compare encoder speed with these rows, not with Criterion
microbenchmarks. `--openjph-matrix` encodes Gray8, RGB8, and Gray16 through
both J2K and OpenJPH and cross-decodes them; its timings compare an
in-process encoder with a separate process, so they aren't a fair speed
comparison on their own.

## The adoption bundle

`adoption-benchmark` runs the CPU comparators and, when asked, the CUDA and
Metal benches on the same pinned manifests. `adoption-report` turns the
result into a report.

```bash
cargo run -p xtask --features adoption -- adoption-benchmark \
  --fixtures corpus/vendor/materialized/decode-fixtures \
  --manifest corpus/vendor/materialized/fixtures.tsv \
  --encode-fixtures corpus/vendor/materialized/staged-pnm \
  --encode-manifest corpus/vendor/materialized/encode-fixtures.tsv \
  --cuda-decode-batch-sizes 1,16,256,1024 \
  --out-dir target/j2k-adoption-benchmark/full
cargo run -p xtask --features adoption -- adoption-report \
  --run-dir target/j2k-adoption-benchmark/full
```

The adoption commands need the `adoption` feature, so run them through
`cargo run -p xtask --features adoption --`, not `cargo xtask`.

- Add `--require-cuda` or `--require-metal` on GPU hosts. The report then
  checks that those benches ran on the same manifests with generated inputs
  turned off.
- `--openjph` and `--kakadu` add the command-line context rows;
  `--require-openjph` and `--require-kakadu` fail if the tool is missing.
- `--quick --include-generated` is for local smoke checks. Its output can't be
  published.

The harnesses list everything that makes a run unfit to publish in
`publication_blockers`, and set `publication_eligible=true` only when that
list is `none`. Blockers include debug builds, a dirty tree, case filters,
skipped comparators, low repeat counts, generated fixtures, too few distinct
external inputs, missing manifest hashes, missing corpus categories, and
license statuses such as `unknown` or `restricted`. `adoption-report` won't
render a blocked bundle unless you pass `--allow-nonpublishable`, and then
labels it diagnostic-only.

Manifests are TSV files with a `path` and `corpus_category` column. Publication
runs also need the hash, codec, and container columns; the harness checks
them against each file. The full column list is under
`J2K_FIXTURE_COMPARE_MANIFEST` and `J2K_ENCODE_COMPARE_MANIFEST` in
[Environment variables](environment-variables.md#benchmark-harnesses). Before
sharing a bundle, replace absolute paths in manifests and results with corpus
labels or repository-relative paths.

On a self-hosted GPU runner, dispatch the `GPU benchmarks` workflow with
`suite=adoption`. It reads the corpus paths from the repository variables
`J2K_ADOPTION_FIXTURES`, `J2K_ADOPTION_MANIFEST`,
`J2K_ADOPTION_ENCODE_FIXTURES`, and `J2K_ADOPTION_ENCODE_MANIFEST`, and fails
if any is unset rather than generating a substitute.

## Other benchmarks

- **Repeated region decoding.** The `packet_directory` group of the
  `j2k-native` `direct_cpu` bench measures region decodes that reuse the packet
  directory. It checks planes bit for bit and reports latency with fresh and
  reused contexts and with eight concurrent requests. Leave
  `J2K_PROFILE_STAGES` unset while timing.

  ```bash
  J2K_PACKET_DIRECTORY_SOURCE=/path/to/large.jp2 \
  J2K_PACKET_DIRECTORY_REGION_SIDE=1024 J2K_PACKET_DIRECTORY_THREADS=12 \
  cargo bench -p j2k-native --bench direct_cpu -- packet_directory
  ```

- **HT block encode.** `cargo bench -p j2k-native --bench tier1_bitplane --
  'htj2k_cleanup_encode/encode_64x64_'` times 64x64 HT cleanup and refinement
  encoding. It is a microbenchmark; record the seeds it prints if you cite it.
- **JPEG to HTJ2K on Metal.** The `jpeg_to_htj2k_external_distinct_batch`
  group of the `j2k-transcode-metal` `dct97` bench compares CPU, batched CPU,
  and Metal routes on your JPEG tiles. It checks every output before timing.
  Leave `J2K_TRANSCODE_METAL_PROFILE_STAGES` unset for clean timing.

  ```bash
  J2K_REQUIRE_METAL_BENCH=1 J2K_REQUIRE_TRANSCODE_WSI_ROOT=1 \
  J2K_TRANSCODE_WSI_ROOT=/path/to/jpeg-tiles J2K_TRANSCODE_WSI_TILE_LIMIT=32 \
  cargo bench --profile release-bench -p j2k-transcode-metal \
    --features bench-internals --bench dct97 -- jpeg_to_htj2k_external_distinct_batch
  ```

- **Burn batches.** `cargo bench -p j2k-ml --bench batch_decode --features cpu`
  covers Gray, RGB, and RGBA at 8 to 16 bits, batch sizes 1 to 64, and
  reports codec output and Burn tensors separately. `cargo xtask
  j2k-ml-bench-cuda` and `cargo xtask j2k-ml-bench-metal` run the hardware
  versions, where staged rows include the device-to-host copy and the Burn
  upload. `J2K_ML_BATCH_INPUT_MODE` picks distinct or repeated inputs and
  `J2K_ML_BATCH_PROCESS_MODE=profile` prints telemetry instead of Criterion
  samples.
- **MPSGraph.** `cargo bench -p j2k-mpsgraph --bench direct_handoff` compares
  staged decode, readback, and upload against handing Metal-resident batches
  straight to MPSGraph. It checks every result against a CPU calculation,
  rotates the path order, and reports a mean with a 95% interval. Its
  `speed_claim_qualified` row is `true` only with at least 30 samples, a
  direct path at least 10% faster, and intervals that don't overlap.
- **Large regions and subsampled tiles.** Ignored characterization tests time
  owned batches on your own inputs:

  ```bash
  J2K_LARGE_REGION_INPUT=<codestream> cargo test --release -p j2k \
    --test owned_batch local_large_region_cpu_batch_characterization -- --ignored --nocapture
  J2K_REQUIRE_METAL_RUNTIME=1 J2K_LARGE_REGION_INPUT=<codestream> cargo test --release \
    -p j2k-metal --lib local_large_region_batch_characterization -- --ignored --nocapture
  J2K_REQUIRE_METAL_RUNTIME=1 J2K_SAMPLED_CORPUS=<dir> J2K_SAMPLED_COUNT=32 \
    cargo test --release -p j2k-metal --lib local_sampled_color_batch_characterization \
    -- --ignored --nocapture
  ```

- **Metal JPEG** benchmarks are described in
  [Metal JPEG decoder](metal-jpeg-decoder.md#benchmarks).

## Auto routing tables

`BackendRequest::Auto` sends work to a GPU using routing tables generated from
measurements and checked into the repository; it doesn't calibrate on the
user's machine. A workload goes to the GPU only if the CPU, hybrid, and any
GPU-only route produce identical bytes on the same pinned input, the GPU
route's median is at least 10% faster than every alternative, and its 95%
confidence interval doesn't overlap theirs.

To measure, write a JSON workload manifest: a corpus label, an HTTPS source
URL, and `cases`, each with an `id`, a relative `path`, a `kind` (`decode` or
`encode`), a `pixel_format` (`gray8` or `rgb8`), a lowercase `sha256`, and in
schema 2 a `codec` and `container`. Cover more than one favorable image:
small and large, gray and RGB, lossless and lossy, full, region, scaled, and
batch. The loader rejects absolute or escaping paths, symlinks, duplicates,
and hash mismatches.

Dispatch `GPU benchmarks` with `suite=routing`; the runner needs the
`J2K_AUTO_ROUTING_MANIFEST` and `J2K_AUTO_ROUTING_ROOT` repository variables.
Then verify the results and regenerate the tables:

```bash
cargo xtask auto-routing verify \
  --evidence target/gpu-benchmark/auto-routing/evidence.json \
  --external-manifest "$J2K_AUTO_ROUTING_MANIFEST" \
  --criterion-root target/criterion \
  --out target/gpu-benchmark/auto-routing/verified.json
cargo xtask promotion-codegen
```

The verifier makes every routing decision itself and rejects missing cases,
route or output mismatches, changed Criterion estimates, and commit or
platform mismatches. Accepted results go in
`xtask/routing-promotion-evidence.json`, and `promotion-codegen` turns them
into Rust tables. Full validation runs `cargo xtask promotion-codegen --check`
to catch stale tables. A workload with no accepted result stays on the CPU.

## Performance guard

```bash
cargo xtask j2k-perf-guard --lane host
```

The guard checks out a baseline commit in a worktree under `target/j2k-perf/`,
copies the current benchmark sources into it, builds and runs the same
Criterion benches on both trees, and fails if an enforced benchmark's median
got more than 10% slower beyond the confidence intervals (and by more than
100 ns). `--threshold-percent` changes the limit, `--quick` shortens the run,
and `--record-current NAME` / `--compare-current NAME` compare two states of
your own tree instead.

The default baseline is commit `29143c8e1f00bbbe9cf5ab37b3cea882f6d52139`
(`DEFAULT_BASELINE_REF` in `xtask/src/perf_guard.rs`). It gives every run the
same fixed point to measure against, and its crate layout can still build the
current benchmark sources. The
`GPU benchmarks` workflow uses the same ref by default for its `smoke` and
`criterion` suites with `--lane cuda` or `--lane metal`. Move the ref only
when the bench layout changes, and update the constant, the workflow default,
and this paragraph together. The guard isn't part of PR or release CI; run it
when you change hot code.

## Reporting results

A report someone else can trust includes:

- the exact commands and environment variables;
- host CPU or GPU, OS, architecture, thread count, and driver;
- the git revision, whether the tree was dirty (including untracked files),
  the build profile, and whether debug assertions were on;
- each comparator's version and how it was built;
- the corpus: source, license status, how natural images were encoded, and
  the manifest with its coverage counts;
- batch sizes, repeats, samples, and medians;
- skipped cases with reasons, and the harness's `publication_blockers`, which
  must be `none` for a head-to-head comparison.

The harnesses print most of this; keep their raw output with the report. When
a result replaces an older one in [Performance](../book/performance.md),
remove the old one rather than keeping both.
