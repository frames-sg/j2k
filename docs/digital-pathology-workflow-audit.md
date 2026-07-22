# Digital Pathology and Large-TIFF Workflow Audit

Audit date: 2026-07-14

Audited workspace: `0.7.1` release candidate

## Decision

J2K is suitable as the compressed tile and frame codec layer in a digital
pathology viewer, converter, or ingestion pipeline. It is not an end-to-end
OME-TIFF, vendor WSI, BigTIFF, or DICOM Whole Slide Imaging implementation.

Adopt it behind a container and slide-index layer that owns file parsing,
range reads, pyramid and tile coordinates, DICOM encapsulation, photometric
metadata, ICC application, caching, and output assembly. Do not hand an entire
slide file to the codec or decode an entire level when only a viewport is
needed.

## Evidence in this workspace

| Concern | Current evidence | Audit result |
| --- | --- | --- |
| JPEG 2000 and HTJ2K payloads | `j2k` accepts raw codestreams and JP2/JPH still-image wrappers. `J2kView` borrows input bytes, and `J2kDecoder` supports full, ROI, scaled, region-scaled, row, and native component-plane decode. | Ready at the codec boundary. |
| Repeated tile decode | `TileBatchDecode`, reusable decoder contexts and scratch pools, caller-owned outputs, indexed tile errors, and CPU worker limits are public. | Ready when the caller bounds queue depth and total live memory. |
| GPU decode | CUDA and Metal adapters expose strict device requests, reusable sessions, resident surfaces, and bounded scheduling. CPU remains the correctness fallback. | Metal is locally parity-tested. CUDA still requires Linux plus NVIDIA hardware validation. |
| Allocation safety | Codec parsing, output, rows, batches, and device paths use checked accounting. The default per-operation host allocation ceiling is 512 MiB. | Useful safety boundary, but not a slide-wide cache or service memory budget. |
| TIFF-style tile codecs | `j2k-tilecodec` exposes uncompressed, LZW, deflate, and Zstd tile decompression. `j2k-jpeg` prepares TIFF/WSI JPEG segments, including shared JPEG tables. | Partial payload support only. TIFF predictors, planar assembly, tags, IFDs, and pyramids remain caller responsibilities. |
| TIFF and OME containers | No production BigTIFF/IFD/SubIFD parser, 64-bit tile-offset index, OME-XML model, or vendor WSI metadata layer is exposed. | Missing outer integration. |
| DICOM WSI | Raw JPEG 2000-family codestreams and codec-profile passthrough checks exist. No production DICOM dataset parser, transfer-syntax registry, frame-fragment index, offset-table handler, functional-group parser, or slide-coordinate model is exposed. | Missing outer integration. |
| Color | JPEG 2000 component metadata and JP2/JPH color boxes, including ICC payloads, are parsed. Native component-plane output is available on CPU. | Container photometric interpretation and ICC transforms must be validated and applied outside the codec. |
| Domain validation | WSI and DICOM tile corpora are documented, with external-corpus environment controls. | A release-blocking container-level golden corpus is still needed. |

Relevant code and policy:

- [`crates/j2k/src/view.rs`](../crates/j2k/src/view.rs)
- [`crates/j2k-core/src/batch.rs`](../crates/j2k-core/src/batch.rs)
- [`crates/j2k-core/src/buffer.rs`](../crates/j2k-core/src/buffer.rs)
- [`crates/j2k-core/src/passthrough.rs`](../crates/j2k-core/src/passthrough.rs)
- [`crates/j2k-jpeg/src/segment.rs`](../crates/j2k-jpeg/src/segment.rs)
- [`crates/j2k-jpeg/src/batch_session.rs`](../crates/j2k-jpeg/src/batch_session.rs)
- [`crates/j2k-tilecodec/src/lib.rs`](../crates/j2k-tilecodec/src/lib.rs)
- [`docs/benchmark-corpora.md`](benchmark-corpora.md)

## Required production workflow

1. Parse the outer container with checked 64-bit arithmetic. Build an index of
   levels, tiles or frames, byte ranges, dimensions, physical coordinates,
   photometric interpretation, and color profiles. Validate every offset and
   byte count against the source length before reading.
2. Select a stored pyramid level for the requested magnification, then compute
   the intersecting tiles. Use codec downscaling only for residual scaling
   within that selected level; it is not a substitute for pyramid selection.
3. Range-read only the selected compressed payloads. Keep reads, compressed
   inputs, decoded outputs, GPU residency, and cache entries under one
   request-level memory and cancellation budget.
4. Normalize each payload at the container boundary:
   - For TIFF, honor compression, predictor, planar configuration,
     photometric interpretation, edge-tile geometry, and shared JPEG tables.
   - For DICOM, map the exact transfer syntax UID, resolve the frame index,
     assemble all fragments belonging to a classic JPEG 2000 frame, remove
     encapsulation padding correctly, and pass a raw codestream without a JP2
     header. HTJ2K frames must follow their single-fragment rule.
5. Inspect the compressed payload before decode. Cross-check dimensions,
   components, bit depth, signedness, lossless/lossy profile, and color meaning
   against the container metadata. Reject contradictions rather than guessing.
6. Submit a bounded tile batch through a reusable CPU, CUDA, or Metal session.
   Keep `BackendRequest::Auto` conservative, preserve CPU parity as the
   correctness oracle, and retain device surfaces through composition when
   that avoids unnecessary readback.
7. Assemble edge and sparse tiles in slide coordinates. Apply the outer
   container's ICC profile and photometric rules, including optical-path color
   semantics for DICOM. Missing tiles, overlaps, and padded pixels are
   container concerns and must not be inferred by the codec.
8. Return or cache only the requested viewport or analysis region. Apply
   backpressure across requests and avoid a decoded full-slide cache.

## Format-specific correctness conditions

### OME-TIFF and large TIFF

OME-TIFF can use standard TIFF or BigTIFF, places OME-XML in the first IFD's
`ImageDescription`, and represents pyramids with SubIFDs. Reduced-resolution
levels are not ordinary entries in the primary IFD chain. Large deployments
therefore need a BigTIFF-capable 64-bit index and explicit SubIFD traversal;
sequentially walking the primary IFD chain is not a correct pyramid reader.

Each level may use a different compression. Route each tile by its level's
compression and tags. For a JPEG 2000 tile, pass only the compatible compressed
payload to `j2k`; do not assume that every TIFF tile is JPEG 2000 because one
level is. Existing TIFF/WSI JPEG preparation in `j2k-jpeg` is complementary,
not a TIFF parser.

### DICOM WSI

DICOM WSI stores a pyramid level as a multi-frame image whose frames are
tiles. `TILED_FULL` positions are implicit; `TILED_SPARSE` positions are
explicit. Edge padding, focal planes, optical paths, overlap, total pixel
matrix dimensions, origin, and orientation belong to DICOM metadata and must
survive decode and assembly.

Each frame is independently JPEG 2000-family encoded and carries no JP2
header. A classic JPEG 2000 frame may span multiple fragments, while an HTJ2K
frame is constrained to one fragment. The Basic Offset Table can be empty, and
large instances may require extended offsets. The outer DICOM layer must
resolve frame boundaries before invoking this workspace.

The codec-level `CompressedTransferSyntax` intentionally describes profiles,
not DICOM UIDs. The DICOM integration must preserve the exact UID and its
lossless/progressive semantics rather than reconstructing it from the codec
enum after the fact.

## Release blockers for an end-to-end pathology claim

These are not blockers for publishing a codec-focused `0.7.1`, but they are
blockers for claiming that this repository alone implements a complete digital
pathology workflow:

1. Add or integrate a production TIFF/BigTIFF parser with checked IFD,
   SubIFD, tile offset, tile byte-count, predictor, planar, and OME-XML
   handling.
2. Add or integrate a production DICOM WSI index with Basic and Extended
   Offset Table support, classic multi-fragment frame assembly, `TILED_FULL`
   and `TILED_SPARSE` coordinates, optical paths, and ICC metadata.
3. Define one bounded viewport scheduler that accounts for file reads,
   compressed bytes, host outputs, GPU surfaces, cache occupancy, cancellation,
   and concurrent requests.
4. Add PHI-safe golden integration fixtures covering classic TIFF, BigTIFF,
   OME-TIFF SubIFD pyramids, representative vendor WSI files, and DICOM WSI.
   Include malformed offsets, truncated fragments, sparse frames, edge tiles,
   mixed level compression, signed and high-bit-depth samples, and ICC cases.
5. Compare decoded pixels and component metadata across CPU, Metal, and CUDA.
   Run CUDA validation on supported NVIDIA hardware; macOS cannot establish
   CUDA runtime parity.
6. Measure peak resident host and device memory while panning and zooming a
   large slide under concurrent load. A throughput benchmark without bounded
   memory and cancellation evidence is insufficient.

## Primary specifications used for this audit

- [OME-TIFF specification 6.2.2](https://docs.openmicroscopy.org/ome-model/6.2.2/ome-tiff/specification.html)
- [DICOM PS3.5: JPEG 2000 and HTJ2K transfer syntaxes](https://dicom.nema.org/medical/dicom/current/output/chtml/part05/sect_A.4.4.html)
- [DICOM PS3.5: encapsulated pixel data](https://dicom.nema.org/medical/Dicom/current/output/chtml/part05/sect_A.4.html)
- [DICOM Whole Slide Imaging overview](https://dicom.nema.org/dicom/dicomwsi/index.html)
- [DICOM PS3.3: microscope tile organization](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_c.8.12.14.html)
- [DICOM PS3.3: multi-frame dimensions](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.17.html)
