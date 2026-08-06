# ISO/IEC 15444-4 / ITU-T T.803 Conformance

Status: **candidate/pending**

Formal claim: **not made**

The implemented harness targets ISO/IEC 15444-4:2024 / ITU-T T.803 v3. Part 4
defines JPEG 2000 conformance-testing procedures and reference comparisons; it
is not another codestream syntax or a performance benchmark. The current
published release predates this evidence, and `main` remains candidate work
until every required exact-SHA lane passes.

## Intended decoder scope

| IUT | Intended wording after release signoff | Route boundary |
| --- | --- | --- |
| `j2k` CPU | Profile-1 Cclass-1; Profile-1 Cclass-1HF; Annex G JP2 reader | CPU implementation under test. |
| `j2k-cuda` | Profile-1 Cclass-1 adapter IUT; Profile-1 Cclass-1HF adapter IUT | Parsing, Tier-1, transforms, output, and transfers are reported per case as CPU, CUDA, or not used. |
| `j2k-metal` | Profile-1 Cclass-1 adapter IUT; Profile-1 Cclass-1HF adapter IUT | Parsing, Tier-1, transforms, output, and transfers are reported per case as CPU, Metal, or not used. |

CPU assistance is permitted for the adapter IUTs. Any such route is labelled
`hybrid`; it is not described as device-native. Annex G JP2 color and component
normalization currently runs through disclosed CPU stages for the GPU adapters.
JPX / Part 2 is outside this scope, except for JP2-compatible JPX input required
by Annex G.

The project does not use a generic “full Part 1 compliant” label. A future
claim must use the exact Profile/Cclass wording above and be tied to the
published reports for one immutable candidate SHA.

## Current development result

The current macOS arm64 and Linux x86-64 CPU development diagnostics each pass
all 90 selected decoder/JP2 cases with zero skips. Both real-hardware adapter
diagnostics record **0/90 device-native, 48/90 hybrid, and 42/90 CPU-routed
cases**: CUDA on an NVIDIA GeForce RTX 4070 SUPER and Metal on an Apple M4 Pro.
All selected outputs are within their applicable bounds, but neither adapter
result is device-native conformance evidence. The CPU Annex D/F encoder matrix
passes 28 of 28 cases. The CUDA and Metal matrices each pass 25 of 25; CUDA
records 24 hybrid encoder routes and one CPU-routed case, while Metal records
23 hybrid routes and two CPU-routed cases. These dirty-worktree development
reports are not exact-SHA release artifacts and do not establish a formal
claim. A Windows x86-64 CPU report is still required for CPU claim eligibility.

The former `c1-c0p0-13` failure was an IUT harness defect. The codestream has
257 components and enables the reversible component transform. T.803 B.2.5
requires Cclass-0 comparison before inverse MCT, so its first-component
reference is 1; the Cclass-1 component-0 reference after inverse RCT is 0. The
harness had incorrectly inferred MCT use from a display colorspace, which is
unknown for this component count. It now reads the COD transform flag and
transform kind through the existing codestream inspector, reconstructs the
pre-MCT component for Cclass-0, and reports the MCT stage from the same
metadata. A 257-component regression test prevents the colorspace inference
from returning.

The report now independently decodes every selected codestream whose COD
enables MCT and whose SIZ declares more than four components. For `p0_13.j2k`,
the production decoder and vendored OpenJPEG 2.5.3 matched component metadata
and samples exactly for all 257 components before any T.803 normalization. Both
canonical native-output hashes are
`a01808e0cbf14288274188c8bebb5ef8c2aa46304eca964a2ac71bed1713c1fd`.
As a second manual check, OpenJPEG CLI 2.5.4 emitted 257 PGX components with
zero sample mismatches; the concatenated one-sample component payload SHA-256
was `54acfbfedc4d8da40f76f275e1a98f10af8ef1fb9fb39e5a67a00aabcbe6597c`.

The investigation independently confirmed byte-identical `p0_13.j2k`,
`c0p0_13.pgx`, and `c1p0_13-0.pgx` payloads in ITU's current attachment,
ISO's 2024 electronic insert, and ITU's 2002 suite. No corpus mapping, hash,
dimension, precision, signedness, reduction, crop, tolerance, or comparison
arithmetic was changed to obtain the passing result.

## Evidence commands

The official corpus is fetched only from the URL and archive digest pinned in
`corpus/j2k-conformance/t803-v3.toml`:

```bash
cargo xtask t803 fetch
cargo xtask t803 run --iut cpu
cargo xtask t803 run --iut cuda
cargo xtask t803 run --iut metal
```

`fetch` rejects unapproved redirects, archive or file hash drift, unsafe archive
entries, duplicate paths, unexpected required-case names, and resource-limit
violations. The copyrighted corpus stays under `target/t803/`. Only versioned
JSON/Markdown reports and hashes may be retained.

Release eligibility is scoped independently. CPU wording requires the three
CPU operating-system reports; each adapter wording requires only that
adapter's real-hardware report. An unavailable adapter blocks its own claim,
not the CPU claim:

```bash
cargo xtask t803 verify --scope cpu --candidate-sha "$RC_SHA" \
  --report path/to/cpu-linux.json \
  --report path/to/cpu-macos.json \
  --report path/to/cpu-windows.json
cargo xtask t803 verify --scope cuda --candidate-sha "$RC_SHA" \
  --report path/to/cuda.json
cargo xtask t803 verify --scope metal --candidate-sha "$RC_SHA" \
  --report path/to/metal.json
```

`--scope all` verifies all five reports together for a coordinated release but
is not a prerequisite for an independently earned CPU or adapter result.

All 90 selected decoder/JP2 cases must be present with no skips, every report
must pass, source and corpus hashes must match, and the IUT/platform/route
identity must match the required lane. Reports are rejected when a route labels
CPU-assisted work as device-native.

## Encoder evidence

The CPU, CUDA, and Metal Annex F implementation compliance statements and the
stable pairwise/boundary matrix live in `corpus/j2k-conformance/`. Selected
codestreams are decoded by the pinned T.804 OpenJPEG reference implementation.
Reference-decode success is the Annex D legality result; lossless output must
also match the source exactly. Lossy rate and PSNR checks are separate project
quality gates.

Encoder testing is informative under T.803 and is not the same formal claim as
decoder compliance. Accelerator dispatch and fallback stages are reported for
every encoder case.

T.803 does not establish robustness, security, adoption, or performance. Those
properties require their own fuzzing, security review, external workload, and
benchmark evidence.
