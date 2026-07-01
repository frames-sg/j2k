# j2k Workspace Reduction Audit

## 1. Executive summary

- **Headline reducible LOC (canonical `j2k` workspace, src only): ~11,351** across **157 verified findings** (21 refuted and dropped). 19 crates + `xtask`, ~231K src LOC.
- **Biggest wins (one line each):**
  - **Metal JPEG `fast444` fork** — `j2k-jpeg-metal/src/compute.rs` carries a full hand-copied 4:4:4 path parallel to the generic `FastSubsampledMetal<P>` code (~900 LOC ceiling; CONFIRMED).
  - **CUDA feature/PTX boilerplate** — 10-kernel-family accessor/guard boilerplate + 28× hand-repeated 10-arm `any(feature="cuda-oxide-*")` cfg lists in `j2k-cuda-runtime` (~550 LOC; CONFIRMED, macro/aggregate-feature fixable).
  - **Dead DCT/DWT reference modules** — `j2k-transcode/src/dct53_1d.rs` + `dct53_multilevel.rs` are reachable only from tests/benches (~560 LOC; CONFIRMED).
  - **Cross-backend kernel reimplementation** — `jidctint` IDCT, 9/7+5/3 lifting, MCT/color-transform, Huffman, and constant tables re-typed across CPU/CUDA/Metal (~600 LOC of the total; mostly CONFIRMED).
  - **God-file / god-function debt** — `j2k-metal/src/compute.rs` (17,757 LOC) hosts a single ~2,900-line function; `j2k-native/src/j2c/encode.rs` (10,815), `j2k-jpeg-metal/src/lib.rs` (7,352, 61% inline tests), `j2k-jpeg/src/decoder.rs` (7,422).
- **Overall health verdict:** **Structurally sound, tactically bloated.** The crate graph is a clean acyclic DAG with deliberate seams (verified: no cycles, no CUDA↔Metal coupling, unsafe FFI correctly isolated). `cargo-machete` finds zero unused dependencies. The debt is concentrated in (a) per-backend copy-paste of identical math/boilerplate and (b) a handful of god-files that were only half-decomposed. Nearly all reducible LOC is **low-to-medium risk**; the only high-risk items are GPU-kernel merges that should explicitly **not** be attempted. **Note also (repo-level, §10): ~10 near-duplicate copy-directories + 59GB of build/corpus artifacts dwarf every in-tree finding.**

## 2. Top 15 highest-impact reductions

| Rank | Unit | Category | Title | Est LOC | Risk | Conf | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | j2k-jpeg-metal | duplication | `fast444` decode variants duplicate the generic `FastSubsampledMetal<P>` path (`compute.rs`) | 900¹ | med | med | CONFIRMED |
| 2 | j2k-jpeg-metal | duplication | `fast444` encode/decode batch families hand-copied vs generic `<P>` path (`compute.rs`) | 700¹ | med | med | CONFIRMED |
| 3 | j2k-transcode | dead_code | `dct53_1d` module used only by own tests/benches (`src/dct53_1d.rs`) | 332 | low | high | CONFIRMED |
| 4 | j2k-cuda-runtime | duplication | Per-kernel-family PTX accessor/guard boilerplate repeated 10× (`build_flags.rs`/`kernels.rs`/`context.rs`) | 300 | med | high | CONFIRMED |
| 5 | j2k-compare | duplication | Infra helpers copy-pasted verbatim between the two bins (should live in `lib.rs`) | 300 | low | high | CONFIRMED |
| 6 | j2k-metal | poor_code | ~2,900-line `encode_ht_cleanup_code_blocks_with_runtime_and_statuses` (`compute.rs:14200-17144`) | 300 | med | high | CONFIRMED |
| 7 | j2k-cuda-runtime | duplication | 10-arm `any(feature="cuda-oxide-*")` cfg list hand-repeated ~28× | 250 | low | high | CONFIRMED |
| 8 | j2k-jpeg-metal | duplication | RGB-surface vs RGBA-texture batch decoders duplicate shared prefixes (`compute.rs`) | 250 | med | med | CONFIRMED |
| 9 | j2k-transcode | dead_code | `dct53_multilevel` + `dct53_2d` oracles reachable only from tests/benches | 230 | med | high | CONFIRMED |
| 10 | cross:bitstream-parse | duplication | JP2 box-parsing stack reimplemented in facade (`j2k/src/parse/boxes.rs`) and engine (`j2k-native/src/jp2/*`) | 210 | med | high | CONFIRMED |
| 11 | cross:jpeg-backends | duplication | Integer `jidctint` IDCT reimplemented in all three backends + CPU SIMD copies | 190 | med | high | CONFIRMED |
| 12 | cross:transcode-glue | duplication | Parallel Cuda vs Metal `DctToWaveletStageAccelerator` harnesses | 160 | med | high | CONFIRMED |
| 13 | arch:giant-files | duplication | Owned vs `compact_owned` + 12-bit region decode bodies (`native/encode.rs`, `jpeg/decoder.rs`) | 160 | med | med | CONFIRMED |
| 14 | j2k-transcode-cuda | duplication | Compact vs non-compact resident-encode paths are near-identical twins (`cuda.rs`) | 150 | med | high | CONFIRMED |
| 15 | j2k-core | dead_code | `DecodeRequest` + four `*_request` dispatch defaults unused (superseded by per-crate enums) | 145 | med | high | CONFIRMED |

¹ Findings 1 and 2 overlap heavily (same `fast444` fork viewed at different granularities). Treat **~900 LOC as the combined ceiling**, not 1,600.

## 3. Dead code

All confirmed unreferenced anywhere in the workspace (src + tests). `cargo-machete` also reports **zero unused manifest dependencies**.

**A. Whole modules shipped in `src/` but reachable only from tests/benches (move to `tests/` or delete):**

| Unit | Location | Est LOC | Why safe | Verdict |
|---|---|---|---|---|
| j2k-transcode | `src/dct53_1d.rs` (+ `lib.rs:10` decl) | 332 | Entire `#[doc(hidden)] pub mod`; only its own tests/benches call it | CONFIRMED |
| j2k-transcode | `src/dct53_multilevel.rs` + `dct53_2d.rs:76,123,220` | 230 | `dct53_multilevel` never called by pipeline/GPU; it is the only non-test caller of the three `dct53_2d` oracle fns | CONFIRMED |
| j2k-transcode | `src/corpus_validation.rs` (+ `lib.rs:8`) | 380 | External-WSI test scaffolding in shipped lib; consumed by one integration test only | N/A (relocate) |

**B. Unused public types re-exported from crate roots:**

| Unit | Location | Est LOC | Verdict |
|---|---|---|---|
| j2k-core | `DecodeRequest` (`types.rs:116-175`) + 4 default methods (`traits.rs:178-195,243-261,407-427,623-647`) | 145 | CONFIRMED |
| j2k-core | `BackendFailureKind` (`accelerator.rs:118-135`), `WarningKind` (`types.rs:198-208`) | 30 | CONFIRMED |
| j2k-profile | `MetricUnit` enum + `ProfileFieldKind::Metric.unit` (write-only) (`field.rs:6-70`) | 28 | CONFIRMED |

**C. Unused functions/wrappers:**

| Unit | Location | Est LOC | Verdict |
|---|---|---|---|
| j2k-metal-support | `MetalDeviceSession` struct+impl+Debug (`lib.rs:232-266`) | 35 | CONFIRMED |
| j2k-metal-support | `buffer_contents_slice`/`_mut` (`lib.rs:398-430`) | 32 | CONFIRMED |
| j2k-test-support | 5 unused pub helpers (`pixels.rs:22`, `cuda.rs:21`, `fixtures.rs:31,41`, `jpeg_fixtures.rs:939`) | 28 | CONFIRMED |
| j2k-native | `encode_code_block_with_style` (`bitplane_encode.rs:327`) — the **only unconditional `#[allow(dead_code)]`** in the tree | 19 | CONFIRMED |
| j2k-profile | `emit_gpu_route_profile` (`gpu_route.rs:97-112`) | 15 | CONFIRMED |
| j2k-profile | `flush_profile_summary_to` (`emit.rs:106`) + `ProfileSummary::flush_to` (`summary.rs:100`) | 14 | CONFIRMED |
| j2k-profile | `ProfileField::raw` (`field.rs:72-75`) | 4 | CONFIRMED |

## 4. Duplication (reuse instead of rewrite)

The dominant reduction theme. Split into cross-backend kernel copies, cross-crate GPU/adapter scaffolding, within-crate copy-paste, and test fixtures.

### 4a. Cross-backend kernel reimplementations (CPU / CUDA / Metal)

The same numeric kernels/constants are written 3-4× (CPU scalar Rust, CUDA `cuda_oxide_*/simt` Rust, Metal `.metal`). CPU↔CUDA copies are both Rust and **can share a real crate**; Metal must be driven from generated `#define`s off the same source.

| Kernel / table | Locations | Est LOC | Verdict |
|---|---|---|---|
| `jidctint` integer IDCT (+4×4/2×2/1×1, dc-only) | `jpeg/idct/{scalar,downscale,avx2,neon}.rs`, cuda `jpeg_decode/simt/main.rs:488-659`, `shaders.metal:1408-1867` | 190 | CONFIRMED |
| `weights.rs` re-derives 9/7 lifting verbatim | `j2k-transcode-metal/src/weights.rs:5-294` vs `j2k-transcode/src/dct97_2d.rs` | 110 | CONFIRMED |
| Canonical JPEG Huffman build + decode | `jpeg/entropy/huffman.rs:64-333`, cuda `jpeg.rs:50-114` + `simt/main.rs:352-411,915-1016`, `shaders.metal:1150-1300+` | 50 | CONFIRMED |
| Forward 9/7 1D lifting (native vs transcode) | `native/fdwt.rs:450-492` vs `transcode/dct97_2d.rs:334-372` | 38 | CONFIRMED |
| 9/7 lifting coefficients (ALPHA/BETA/…/KAPPA) | 8 sites / 6 crates (native `fdwt.rs`,`idwt.rs`×2, transcode, 2× cuda-oxide, 2× metal) | 35 | CONFIRMED |
| Fancy chroma upsample (h2v2/h2v1 triangle) | `jpeg/color/upsample.rs`, cuda `jpeg_decode/simt`, `shaders.metal:2097-2130` | 30 | UNCERTAIN |
| MCT ICT/RCT coefficients + sample logic | `native/mct.rs`,`forward_mct.rs`, `mct.metal`, 2× cuda-oxide | 25 | CONFIRMED |
| MEL exponent table `[0,0,0,1,…,5]` + state | native HT decode/encode, 2× cuda-oxide HT | 22 | CONFIRMED |
| 5/3 reversible lifting predict/update | `native/idwt.rs`, cuda `j2k_idwt/simt:160-333`, `idwt.metal:93-340` | 22 | CONFIRMED |
| YCbCr→RGB fixed-point (91881,22554,…) | `jpeg/color/ycbcr.rs`, cuda `jpeg_decode/simt:738`, `shaders.metal:2460` | 20 | CONFIRMED |
| RGB→YCbCr fixed-point (19595,38470,…) | `jpeg/encoder.rs:415`, cuda `jpeg_encode/simt:239`, `shaders.metal:368` | 20 | CONFIRMED |
| Zigzag scan `[u8;64]` | `jpeg/entropy/mod.rs:14`, 2× cuda-oxide, `shaders.metal:299` | 16 | CONFIRMED |
| Forward DCT+quantize cosine table | `jpeg/encoder.rs:709`, cuda `jpeg_encode/simt:329`, `shaders.metal:456` | 15 | UNCERTAIN |
| SigProp spread-mask table `[16]` | native HT decode/encode, 2× cuda-oxide, `ht_cleanup.metal:785` | 10 | CONFIRMED |
| Dequant step/mantissa reconstruction | `native/quantize.rs:58-80` vs cuda `j2k_dequantize/simt:72-113` | 12 | CONFIRMED |
| **HTJ2K block coder (cleanup/SigProp/MagRef+MEL/VLC)** | `native/ht_block_{decode,encode}.rs` vs cuda-oxide HT SIMT | 80 | **UNCERTAIN — do NOT merge dispatch** (see §8) |

**Proposed shared abstractions:** (1) a `#![no_std]` **`j2k-dwt-const`/`j2k-types::dwt`** constants module (9/7+5/3 coeffs, MCT/color coeffs, zigzag, MEL/SigProp tables); (2) a `#![no_std]` **`j2k-jpeg-kernel-core`** for scalar IDCT/FDCT/Huffman/color-transform shared by the CPU path and the CUDA-host Rust kernels; (3) a `build.rs` that emits `.metal`/`.cu` `#define`s from those Rust consts so shader literals stop being hand-maintained. Extract **execution-model-independent** HT pieces (tables + MEL/VLC math + pure per-quad helpers) only — never the launch loops.

### 4b. Cross-crate GPU-runtime & adapter scaffolding

The Metal side already has `j2k-metal-support`; the CUDA side has no equivalent, and both families re-declare identical error/session/routing plumbing.

| Opportunity | Locations | Est LOC | Verdict |
|---|---|---|---|
| Parallel Cuda/Metal accelerator **harnesses** (mode enum, counters, gate/dispatch/recover) | `transcode-cuda/lib.rs:96-829` vs `transcode-metal/lib.rs:317-903` | 160 | CONFIRMED |
| Accelerator **counter-bag + accessors + DispatchMode** scaffold | `transcode-cuda/lib.rs:99-303` vs `transcode-metal/lib.rs:319-651` | 90 | CONFIRMED |
| JPEG marker-segment walkers (2× within j2k-jpeg) | `parse/markers.rs:55-171` vs `segment.rs:572-696` | 65 | CONFIRMED |
| SIZ/COD parse+validation + marker consts (within j2k-native) | `inspect.rs:12-424` vs `j2c/codestream.rs:689-1295` | 55 | CONFIRMED |
| **No shared CUDA-support crate**: surface wrap/upload + error mapping | `j2k-cuda/src/runtime.rs` vs `j2k-jpeg-cuda/src/runtime.rs` | 45 | CONFIRMED |
| `MetalBackendSession` lazy runtime init + `runtime_initialization_error` | `j2k-metal/{lib,compute}.rs` vs `j2k-jpeg-metal/{lib,compute}.rs` | 45 | CONFIRMED |
| `CudaSession` context/pool lazy-init + `AcceleratorSession` | `j2k-cuda/session.rs:34-153` vs `j2k-jpeg-cuda/session.rs:52-199` | 40 | CONFIRMED |
| `RouteDecision` enum + `decision_error`/profile mapping (Metal) | `j2k-metal/routing.rs:16-139` vs `j2k-jpeg-metal/routing.rs:15-159` | 40+35 | CONFIRMED |
| `j2k-jpeg-metal` re-implements `dispatch_1d/2d/3d_pipeline` | `jpeg-metal/compute/kernel_helpers.rs:72-146` vs `metal-support/lib.rs:463-512` | 40 | CONFIRMED |
| Metal `DeviceSurface` impl + Surface/Storage wrappers | `j2k-metal/lib.rs:506-548` vs `j2k-jpeg-metal/lib.rs:260-301` | 30 | CONFIRMED |
| **4 GPU-adapter `Error` enums + `CodecError` impls** | `j2k-metal`,`j2k-jpeg-metal`,`j2k-cuda`,`j2k-jpeg-cuda` (see errors-types findings) | 50+28+22 | CONFIRMED |
| `CudaTranscodeError`/`MetalTranscodeError` same-shape twins + `From`/recover | `transcode-cuda/lib.rs:46-93` vs `transcode-metal/lib.rs:49-85` | 25+28 | CONFIRMED |
| Batch sample-sum + Auto job-count/samples gate | 5× in `transcode-cuda`, inline folds in `transcode-metal` | 38 | CONFIRMED |

**Proposed shared abstractions:** create **`j2k-cuda-support`** (mirror of `j2k-metal-support`) for CUDA error mapping + surface validation + generic wrap; add an `adapter_error!{}` macro (or generic `AdapterError<D>` + blanket `CodecError`) in `j2k-core`; add a `TranscodeAcceleratorError` + `StageAcceleratorHarness<B: TranscodeBackendHooks>` in `j2k-transcode`; hoist `RouteDecision`/session-slot into the two support crates.

### 4c. Within-crate copy-paste

| Unit | Item | Est LOC | Verdict |
|---|---|---|---|
| j2k-jpeg-metal | `fast444` fork (see §2 #1/#2) | ~900¹ | CONFIRMED |
| j2k-cuda-runtime | PTX accessor/guard boilerplate ×10 | 300 | CONFIRMED |
| j2k-cuda-runtime | 10-arm cfg list ×28 | 250 | CONFIRMED |
| j2k-jpeg-metal | RGB-surface vs RGBA-texture batch decoders | 250 | CONFIRMED |
| j2k-compare | case-vs-mixed pipeline duplicated inside each bin | 250 | N/A |
| arch:giant-files | owned vs compact_owned + 12-bit region bodies | 160 | CONFIRMED |
| j2k-transcode-cuda | compact vs non-compact resident twins | 150 | CONFIRMED |
| j2k-transcode | integer-5/3 vs float-9/7 batch pipelines | 140 | CONFIRMED |
| j2k-cuda | 4-way region/scaled plan-builder + surface fan-out | 140 | CONFIRMED |
| j2k-native | `validate_*_htj2k97` + `validate_*_resolution` triplets | 110 | CONFIRMED |
| j2k-metal | `next_label` if/else waterfall ×7 (`compute.rs:16405-16700`) | 75 | CONFIRMED |
| j2k-types | Prequantized/Preencoded/PreencodedCompact 4-level skeletons ×3 | 72 | CONFIRMED |
| j2k-native | `prepared_resolution_packets`/`prepared_subband` builder family | 70 | CONFIRMED |
| j2k-jpeg-metal | bare/`_into_output`/`_with_output` forwarder triplets | 70 | CONFIRMED |
| j2k | 4 `decode_tiles_*` batch fns duplicate scoped-thread loop (`batch.rs`) | 80 | CONFIRMED |
| j2k-native | 5 near-identical `*_level_count` fns | 50 | CONFIRMED |
| j2k-types | `J2kForwardDwt53*` byte-identical to `Dwt97*`; `PrecomputedHtj2k53/97` | 35+24 | CONFIRMED |
| j2k | roundtrip-validation family repeats decode+map_err | 20 | CONFIRMED |

### 4d. Test-fixture duplication (`cross:test-fixtures`)

All confirmed; consolidate into `j2k-test-support` (or a per-crate `tests/common/mod.rs` where a test-support dep is intentionally avoided, e.g. j2k-native):

- PNM read/write/parse helpers vs `test-support` (`encode_compare` bin) — 75
- `minimal_baseline_jpeg` family re-typed in `jpeg/tests/inspect.rs` — 50
- APP14-RGB + progressive JPEG byte fixtures (`transcode/tests/fixtures`) — 60
- `max_abs_diff` DWT-parity helper ×4 transcode GPU tests — 40
- J2K marker-scan helpers across native + GPU transcode tests — 45
- `fnv1a64_hex` re-implemented 7× / 5 files — 38 + 35

## 5. Technical-debt hotspots

**`.expect`/`unwrap`/`panic` clusters** (per-crate, `src/` incl. in-crate tests; whole-workspace: **2,372 `.expect`, 185 `.unwrap`, 112 panic-family**):

| Crate | expect | unwrap | panic! | unsafe |
|---|---|---|---|---|
| j2k-jpeg-metal | 788 | 0 | 12 | 15 |
| j2k-metal | 590 | 1 | 10 | 81 |
| j2k-cuda-runtime | 389 | 10 | 3 | 257 |
| j2k-native | 276 | 29 | 14 | 0 |
| j2k-cuda | 180 | 0 | 5 | 1 |
| j2k-jpeg | 57 | 133 | 2 | 192 |
| j2k-compare | 21 | 0 | 0 | 29 |
| j2k-transcode | 19 | 10 | 1 | 0 |
| j2k-test-support | 13 | 2 | 0 | 0 |

The `.expect` mass is concentrated in the Metal/CUDA adapters, which is expected for GPU-buffer plumbing but is the single largest latent-panic surface; `j2k-jpeg`'s 133 `.unwrap` is the outlier for a `forbid(unsafe)` engine crate and worth a focused pass. Seed reports **ZERO TODO/FIXME** (policy-stripped), so grep-based debt discovery is blind here.

**`unsafe`/lint-policy fragmentation** (594-614 usages):
- **~160 of the unsafe usages live in the 10 embedded `cuda_oxide_*/simt/src/main.rs` GPU kernel trees**, which compile to PTX and never link into the host binary — they inflate the metric ~35%. Move them out of `src/` (e.g. `kernels/`) so unsafe accounting reflects host code (N/A verdict; hygiene).
- **`unsafe_code=forbid` is not actually workspace-wide**: FFI/SIMD/adapter crates opt out via per-crate `[lints]` (documented, intentional). Safe crates that inherit forbid: `j2k-profile`, `j2k-tilecodec`, `j2k-types`, plus overrides in `j2k`, `j2k-native`, `j2k-transcode`, `j2k-test-support`.
- **`j2k-native` uniquely disables `clippy::pedantic`** (`Cargo.toml:53-55`) while every other crate warns — the largest engine crate has the weakest lint posture.

**`#[allow(...)]` rot** (267 total; 55-59 `dead_code`):
- Only **5 `allow(dead_code)` are unconditional**, and only **one is genuine** (`native/bitplane_encode.rs:327`, already in §3); the other 54 are `cfg_attr(not(feature=…))`/target/test gating (22 in `cuda-runtime/kernels.rs`, 12 in `bytes.rs`, etc.) — legitimate.
- **164 of 267 allows are `clippy::too_many_arguments`** — a systemic wide-parameter API smell in `j2k-cuda/{encode,decoder,codec}.rs`, `j2k-transcode/jpeg_to_htj2k.rs`, `native/ht_block_encode.rs`, `j2k-core/traits.rs`. Fix with `EncodeParams`/`BlockCtx` structs, not more suppressions.
- `too_many_lines` is suppressed crate-wide in `j2k-compare` and `j2k-transcode`, hiding the god-functions in §6.

**Feature-gate sprawl** (583 `#[cfg(feature)]` gates):
- `cuda-runtime` referenced 520×; the 10 `cuda-oxide-*` sub-features drive the 28× repeated `any(...)` lists (§4c). Fix with an internal aggregate `_cuda-oxide` feature.
- **Redundant no-op passthrough features**: `j2k-cuda/Cargo.toml:34-45` (6) and `j2k-transcode-cuda/Cargo.toml:30-32` (3) each just re-enable `cuda-runtime` (15 LOC; CONFIRMED).
- `bench-libjpeg-turbo` has 0 `cfg!` refs but is a legit `required-features` gate — **keep**. `simd` has only 4 cfg refs — confirm it still gates meaningful code.

## 6. Poorly written code

**God-functions:**
- `j2k-metal/compute.rs:14200-17144` — single ~2,900-line fn (300; CONFIRMED). Extract validate/pack → alloc → dispatch → readback phases into the existing `compute/` siblings.
- `j2k-native/encode.rs:3426` `encode_impl` (543 lines), `:4540` `encode_multitile_impl` (240), `:7124` layered packets (243) (120).
- `xtask/adoption_report.rs:728-1116` `render_report` (~389 lines, 5); `adoption_benchmark.rs` `write_readme` (~208) + `write_summary` (~180) (30).
- `j2k-compare` `publication_blockers` (~217/157), `emit_metadata` (~150/125), `fixture_cases` (~140) (60).

**Needless indirection / mechanical boilerplate:**
- `j2k-cuda-runtime/bytes.rs:75-312` — ~45 one-line wrappers forwarding to `GpuAbi::as_bytes` (120).
- `j2k-transcode/jpeg_to_htj2k.rs:534-816` — 7 `push_*` helpers, ~63 `("name", val.to_string())` lines (80).
- `j2k-jpeg-metal/compute.rs` — single-caller wrapper chains around batch decoders (80).
- `j2k-cuda/encode.rs:859-946` — 18 counter getters used only by unit tests (70).
- `j2k-jpeg/backend/mod.rs:102-230` — dispatch match arms identical across all `BackendKind` (30).
- `j2k-transcode-metal` — `ProjectedBands`→`Dwt*TwoDimensional` 8-field copy ×4 (25); `new_explicit`/`for_auto` each spell ~22 fields (25).
- `j2k-transcode-cuda/cuda.rs:200-312` — ~110 lines field-by-field timing copies (40).
- `j2k/view.rs:535,587` — `decode_rows_u8/u16_bounded` duplicate stripe loop (25).

**Copy-paste hot spots:** the case-vs-mixed pipeline (§4c, 250) and the 8/16-bit × 4:2:2/4:2:0 lossless entropy encoder family (`test-support/jpeg_fixtures.rs:2030-3105`, 180; **high risk** — bit-exact fixtures).

## 7. Architecture & API problems

**Giant files (split; net LOC ~0 but unblocks the dedup above):**

| File | LOC | Note |
|---|---|---|
| `j2k-metal/src/compute.rs` | 17,757 | 273 fns / 42 structs inline despite 22 `compute/*` submodules |
| `j2k-native/src/j2c/encode.rs` | 10,815 | classic/HT × rev/irrev × single/multi × 4 flavors + 2,300 test lines |
| `j2k-jpeg-metal/src/lib.rs` | 7,352 | **61% (~4,510 lines) inline `mod tests`** — relocate to `src/tests.rs` |
| `j2k-jpeg/src/decoder.rs` | 7,422 | one ~3,900-line `impl Decoder<'_>` block, 260 fns |
| `j2k-transcode/src/jpeg_to_htj2k.rs` | 5,051 | 5 fns each `#[allow(too_many_lines)]` |
| `j2k-cuda-runtime/src/tests.rs` | 4,142 | single test god-module |
| `j2k-test-support/src/jpeg_fixtures.rs` | 3,930 | mixed fixture families + private entropy internals |
| `j2k-compare` bins | 3,878 + 2,840 | two flat single-file binaries |
| `j2k/src/encode.rs` | 2,930 | options + samples + entry points + validation + PSNR |
| `xtask` adoption_{benchmark,report}.rs | 2,622 + 2,387 | ~24% of crate each |
| `j2k-transcode-metal/src/metal.rs` | 2,607 | 97 fns; drives the metal copy-paste |

**Crate-boundary issues:**
- **`j2k-transcode-metal` pulls the entire 43.5K-LOC `j2k-metal` for one type** (`MetalEncodedJ2k`, `lib.rs:157`). Move the type to `j2k-metal-support`/`j2k-types` and drop the dep (CONFIRMED).
- **`j2k-transcode-cuda` declares `j2k-native` as non-optional** but uses it only under `cuda-runtime` (`src/cuda.rs`); make it optional + feature-tied so default builds stay lean (from map notes).
- **No `j2k-cuda-support` crate** — the CUDA family duplicates what `j2k-metal-support` centralizes for Metal (§4b).
- **Engine/facade split applied only to J2K** (`j2k-native`+`j2k` vs combined `j2k-jpeg`/`j2k-transcode`) — pick one convention and document it.
- **Two parallel contract crates** (`j2k-core` traits + narrow `j2k-types` jobs). This is a *deliberate* seam preserving `j2k-native`'s no-core-dep boundary — keep and document, do not merge blindly. `j2k-core` has minor module sprawl (5 files < 40 LOC).

**Public-API design (combinatorial thin-wrapper matrices — collapse behind options/request structs, stage public ones behind deprecation windows):**
- `j2k-jpeg/decoder.rs:1831-2154` — 15 `decode_tile*` free-fn wrappers (130); combinatorial `{into,scaled,region,region_scaled,rgba8} × {_with_scratch} × {_with_options}` decode surface (150).
- `j2k-native/encode.rs:1300-2250` — precomputed/prequantized/preencoded `× {mct, accelerator}` = 4 fns/flavor (120+90).
- `j2k-metal/lib.rs:927-1200` — Full/Region/Scaled/RegionScaled × output-kind cross-product, each with a duplicated non-macOS stub (120; **high risk / low conf**).
- `j2k-cuda/encode.rs:286-427` — 10 `encode_lossless_from_cuda_buffer*` wrappers (35); `direct_plan` module fully public with zero non-test consumers (40).
- `j2k-jpeg-metal` — 3 parallel `Option<&JpegFastNPacketV1>` params threaded through ~80 sites (150) — bundle into one `FastPacketSet`.
- Minor pub-surface leaks: `j2k-core/buffer.rs` internal helpers, `j2k-metal-support` `shader_library`/`named_pipeline`/`mtl_size`, `j2k-test-support` internal consts, `j2k` root re-exports with no consumers.
- **Versioning drift** (harmless, publish=false): `j2k-compare` pins `0.2.0`, `xtask` `0.0.0` vs workspace `0.6.2`.

## 8. Risk register

| Reduction | Risk | Verify first |
|---|---|---|
| **HTJ2K block coder scalar↔SIMT merge** (§4a, 80) | **High** — different execution models | **Do NOT merge launch loops.** Extract only tables + MEL/VLC math + pure helpers into `no_std`; keep kernel dispatch per-backend. Verdict was UNCERTAIN. |
| 8/16-bit × 4:2:2/4:2:0 lossless entropy encoder generic (test-support, 180) | High | Byte-exact fixture regeneration; diff generated JPEG bytes against current fixtures before/after. |
| Metal decode-method cross-product + stub macro (`j2k-metal/lib.rs`, 120) | High / low conf | Full macOS GPU parity run; the non-macOS stubs must still compile on Linux CI. |
| Metal/Cuda accelerator wrapper near-dup (55, low conf) | High | Behavioral parity of Auto/Explicit dispatch-decline contract; counter values in unit tests. |
| Cross-backend kernel dedup (IDCT/DWT/MCT/Huffman/color, ~600 total) | Medium | **GPU numeric parity harness** (existing `dwt97_parity`, `dct53`/`dct97` tests) must pass bit-for-bit; add codegen tests asserting Metal/CUDA constants equal Rust source. |
| JP2 box-parse consolidation (210), SIZ/COD (55), marker walkers (65) | Medium | Round-trip parse of the JP2/JPEG conformance corpus; the facade and engine currently map to different error/metadata types — verify `From` impls preserve every variant. |
| Crate-boundary moves (`transcode-metal`→support, `transcode-cuda` optional dep) | Low-Med | Feature-matrix build: default (no GPU), `cuda-runtime`, macOS target; confirm no symbol regressions. |
| Public API wrapper collapses (native/jpeg/cuda encode surfaces) | Medium | These are `pub` — SemVer. Stage removals behind a deprecation window; grep external `required-features`/bench targets first. |
| Accelerator harness generalization (§4b, 160+90) | Medium | Both GPU transcode integration tests (`tests/jpeg_to_htj2k.rs`) + counter-accessor unit tests. |
| Dead-code deletion (§3) | Low | Confirm no `#[cfg(test)]`/bench-only reachability before delete (already verified per-finding). |

## 9. Suggested execution order

| Phase | Theme | Representative items | Rough LOC |
|---|---|---|---|
| **1 — Quick safe wins** | Dead code + verbatim copy-paste + test fixtures | §3 dead code (all), §4d fixtures, `xtask` `sanitize_id`/`cargo()` triplicates, `j2k-jpeg` duplicate zigzag | ~1,300 |
| **2 — Boilerplate & feature-gate collapse** | Macro/aggregate-feature the within-crate repetition | CUDA PTX boilerplate (300), 10-arm cfg → `_cuda-oxide` (250), `next_label` waterfall (75), `bytes.rs` wrappers (120), `*_level_count` (50), validate triplets (110), adapter-error macros (~100), redundant passthrough features (15), counter getters (70) | ~1,100 |
| **3 — Cross-crate / cross-backend consolidation** | Shared const module + kernel-core + `j2k-cuda-support` + session/error/route sharing | jidctint (190), JP2 parse crate (210), accelerator harness (160+90), `weights.rs` (110), gpu-runtime scaffolding (~200), kernel constants/color/DWT (~230) | ~1,400 |
| **4 — Within-crate duplication via generics** | Trait/generic-ize the twin families | `fast444` fork (~900¹), RGB/RGBA decoders (250), case/mixed pipeline (250), compact/non-compact resident (150), 12-bit region (160), integer/float batch (140), 4-way region (140), j2k-types hierarchies (131), decode_tiles (80), prepared builders (70) | ~2,500 |
| **5 — Structural refactors** | God-file splits, oversized-fn extraction, test relocation, API-surface trim | 2,900-line fn (300), relocate inline test modules (400), `corpus_validation`→tests (380), encode/decoder/jpeg_to_htj2k/xtask splits, encode orchestrators (120), wrapper-matrix API collapses (~500) | ~2,000 |

Total across phases aligns with the **~11,351 LOC** estimate. Phases 1-2 are almost entirely CONFIRMED/low-risk and should land first to shrink review surface before the structural work.

## 10. Repo-level note (outside audited scope, largest single opportunity)

The parent folder holds **~10 near-duplicate copy-directories of this entire workspace**, each a full clone diverging only in one experiment:

`j2k-cuda-oxide-copy-u8`, `j2k-cuda-oxide-decode-store`, `j2k-cuda-oxide-dequantize`, `j2k-cuda-oxide-idwt`, `j2k-cuda-oxide-idwt-coop`, `j2k-metal-encode-dwt97`, `j2k-metal-encode-auto-routing-benchmarks`, `j2k-jpeg-metal-routing-benchmarks`, `j2k-transcode-pipeline-map`, `j2k-cuda-oxide-transcode`.

Plus **~59GB of build `target/` and corpus artifacts inside `j2k/`**.

These dwarf every in-tree finding: the copy-directories are effectively 10× the ~231K-LOC workspace duplicated on disk, and the artifacts are pure reclaimable space. **Recommended top-line action:** fold the divergent experiments back into feature branches/gates of the canonical `j2k` workspace (the feature system already isolates `cuda-oxide-*` kernels for exactly this), delete the copies, and add `target/`+corpus to ignore/cleanup. Detailed findings above remain scoped to the canonical workspace.