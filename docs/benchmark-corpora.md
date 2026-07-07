# Benchmark Corpora

Adoption-facing benchmark reports should not rely only on generated fixtures.
Use a mix of standards conformance vectors, independent implementation test
data, natural-image datasets encoded into J2K/HTJ2K with documented settings,
and domain tiles.

The tables below are an evidence plan, not a standing claim that every named
dataset was present in a local run. A run may claim only the corpora present in
its pinned manifest and summarized by `cargo xtask adoption-report`.

## Required Mix

| Corpus | Use | Source | Repo handling |
| --- | --- | --- | --- |
| ISO JPEG 2000 conformance files | Compliance-style claims for JPEG 2000 Part 1 and HTJ2K Part 15. | ISO/IEC 15444-4 / ITU-T T.803 electronic attachments. | User-supplied. Do not commit unless licensing permits. Track expected vectors in `corpus/j2k-conformance/manifest.tsv`. |
| OpenJPEG test data | Regression and interoperability corpus with real JP2/J2K edge cases. | `https://github.com/uclouvain/openjpeg-data` | User-supplied clone path. |
| OpenJPH / HTJ2K fixtures | HTJ2K-specific interoperability and JPH/J2K variants. | `https://github.com/aous72/OpenJPH` and released OpenJPH test assets. | User-supplied clone/path; small license-compatible fixtures may be committed with notices. |
| jpylyzer test files | JP2 parser/metadata robustness, including valid and invalid files. | `https://github.com/openpreserve/jpylyzer-test-files` | User-supplied clone/path. Invalid files should be used for robustness tests, not throughput comparisons. |
| Kodak | Classic natural-image compression benchmark. | Kodak PhotoCD image set mirrors. | User-supplied source images. Encode to J2K/HTJ2K with recorded command/options before decode benchmarking. |
| Tecnick / TESTIMAGES | Larger natural-image benchmark set. | TESTIMAGES / Tecnick archives. | User-supplied because of dataset license terms. Encode to J2K/HTJ2K with recorded command/options. |
| CLIC | Modern high-resolution photographic compression benchmark. | `https://www.compression.cc/` | User-supplied dataset. Encode to J2K/HTJ2K with recorded command/options. |
| Domain WSI/DICOM/medical tiles | Most relevant adoption workload for this project. | Internal, partner, public TCIA/IDC-style exports, or scanner-derived tiles. | User-supplied. Do not commit protected health information or restricted datasets. |

## External Benchmark Survey

The harness is intentionally not a single Kodak-only benchmark. Public JPEG 2000
and image-compression practice points to several distinct evidence classes:

| Source | What they use or measure | Rule for this repo |
| --- | --- | --- |
| OpenJPEG test suite and `openjpeg-data` | Standards conformance files, non-regression files, header dumps, and baseline comparisons. `openjpeg-data` states that it contains conformance, non-regression, and unit-test files used by the OpenJPEG test suite. | Use OpenJPEG conformance/nonregression fixtures as native decode interoperability and regression coverage. Do not use them alone as speed marketing; keep conformance claims separate from throughput claims. |
| OpenJPH | HTJ2K / JPEG 2000 Part 15 implementation and interoperability surface. | Include OpenJPH/JPH-compatible fixtures for HTJ2K-specific decode coverage and optional OpenJPH context rows, clearly labeled by CLI/file-output method. |
| JPEG committee HTJ2K white paper | HTJ2K is framed as a throughput improvement for JPEG 2000, especially at lossless and moderate-to-high bitrates. | Hardware marketing claims must separate classic J2K CPU comparisons from J2K CPU-vs-GPU HTJ2K acceleration rows and must state codec/profile. |
| OpenBenchmarking / Phoronix OpenJPEG profile | Large single-image encode stress using a NASA/JPL-Caltech/MSSS Curiosity panorama TIFF, reported as an OpenJPEG processor benchmark. | Add large source-image encode stress in addition to Kodak-sized images. Record source-image staging and exclude file I/O from codec timing. |
| IIIF HTJ2K evaluation | Real access workload over large cultural-heritage images, including tile/region-size effects and OpenJPEG/Kakadu/HTJ2K comparisons. | Keep ROI/region-scaled decode rows and domain tile batches. Report method labels for native versus emulated operations. |
| Lossless image compression benchmarks | Corpora such as CLIC2021, LPCB, and GDCC2020; metrics include compressed size plus compress/decompress time. | For lossless encode/decode adoption claims, report throughput and output size, and use larger modern RGB/gray corpora where license terms allow. |
| GPU JPEG 2000 encoder benchmarks | CPU and GPU encoders compared with MB/s, FPS, PSNR/MSE or compression ratio, and explicit note that host I/O is outside timing for batch mode. | GPU rows must report MiB/s/FPS-style throughput, profile, compression mode, batch size, and timing policy; staged input loading and output file I/O must stay outside the timed loop. |

Survey references consulted on 2026-06-24:

- `https://github.com/uclouvain/openjpeg/wiki/TestSuiteDocumentation`
- `https://github.com/uclouvain/openjpeg-data`
- `https://github.com/aous72/OpenJPH`
- `https://ds.jpeg.org/whitepapers/jpeg-htj2k-whitepaper.pdf`
- `https://openbenchmarking.org/test/pts/openjpeg`
- `https://journal.code4lib.org/articles/17596`
- `https://github.com/WangXuan95/Image-Compression-Benchmark`
- `https://www.fastcompression.com/benchmarks/benchmarks-j2k.htm`

## Additional Optional Corpora

| Corpus | Use | Source | Caveat |
| --- | --- | --- | --- |
| USC-SIPI Image Database | Classic image-processing benchmark images with textures, aerials, and natural scenes. | `https://sipi.usc.edu/database/` | Older and small by modern standards; encode to J2K/HTJ2K first. |
| Waterloo Exploration / Waterloo image sets | Legacy compression and image-analysis benchmark material. | Waterloo image-database mirrors and derivative benchmark sets. | Licensing/source stability varies by mirror. |
| DIV2K / Flickr2K | High-resolution RGB natural images used heavily in restoration and compression papers. | `https://data.vision.ee.ethz.ch/cvl/DIV2K/` and Flickr2K mirrors. | Source images are not J2K; record encode settings before decode benchmarking. |
| ImageCompression.info New Test Images | Modern high-resolution RGB and grayscale compression stress set with explicit redistribution terms. | `https://imagecompression.info/test_images/` | Large files; keep corpus source and notice with local artifacts. Good for gray/RGB and dimension diversity. |
| Imazen codec-corpus | Curated codec validation and compression benchmark corpus with sparse-checkout-friendly subdatasets. | `https://github.com/imazen/codec-corpus` | Dataset licenses vary by subfolder; pin the selected subset and license status in manifests. |
| WangXuan95 lossless image compression benchmark corpora | Public benchmark framing for lossless compression speed/size using CLIC2021, LPCB, and GDCC2020. | `https://github.com/WangXuan95/Image-Compression-Benchmark` | Use as methodology/corpus inspiration; verify upstream dataset terms before redistributing. |
| USTC-TD | 4K image and 1080p video coding challenge dataset for modern codec stress. | USTC-TD project/paper mirrors. | Source images require documented conversion before J2K/HTJ2K throughput runs. |
| LIVE, CSIQ, TID2013, KADID-10k | Objective quality / perceptual distortion evaluation. | Dataset project pages. | Better for quality metrics than raw codec throughput. |
| IIIF / cultural heritage JP2 collections | Real JP2 preservation/access workloads such as newspapers, books, maps, and manuscripts. | Institution-specific IIIF repositories and preservation masters. | Access and redistribution terms vary; many collections provide derivatives, not masters. |
| DICOM WSI / radiology HTJ2K samples | Standards-relevant medical imaging containers and transfer syntaxes. | DICOM sample data, TCIA/IDC exports, Orthanc WSI examples, partner datasets. | Must preserve DICOM transfer syntax, tile shape, and metadata in reports. |
| CAMELYON, PANDA, TCGA/TCIA pathology | Whole-slide domain representativeness and huge tiled batches. | Public challenge/data portals. | Usually SVS/TIFF/DICOM rather than JP2/JPH; extract or transcode tiles with recorded commands. |
| Remote sensing / satellite datasets | Large, high-detail imagery with different texture statistics than photos. | WHU-RS19, SpaceNet, Sentinel/Landsat-derived products. | Often multi-band or high bit depth; ensure supported component/bit-depth coverage is explicit. |
| NASA/JPL Curiosity panorama-style large TIFF | Large single-image CPU encode stress, matching the style used by the Phoronix/OpenBenchmarking OpenJPEG profile. | NASA/JPL-Caltech/MSSS public imagery or a local equivalent TIFF. | Encode-source corpus, not native J2K; record conversion and staging commands. |
| Spot / Pleiades style geospatial imagery | Remote-sensing decode/region workflows similar to Grok's published JP2Grok/JP2KAK/OpenJPEG benchmark framing. | Public or licensed satellite imagery exports. | Licensing and redistribution vary; report storage locality, region selection, and extraction route. |
| OpenSlide test data | Vendor WSI file diversity and tile extraction workflows. | `https://openslide.cs.cmu.edu/download/openslide-testdata/` | Usually source WSI formats rather than standalone J2K; extract compressed tiles or transcode with recorded commands. |
| Bio-Formats sample images | Scientific microscopy and life-sciences format diversity. | OME/Bio-Formats sample repositories and public OME data. | Use when testing file ingestion/transcode paths; not all inputs are J2K. |
| OsiriX JPEG 2000 DICOM samples | Quick DICOM JPEG2000 transfer syntax smoke coverage. | OsiriX DICOM image library. | Research/teaching terms; do not redistribute. |
| DICOM/NEMA compressed transfer syntax examples | Transfer-syntax interoperability for JPEG2000, JPEG-LS, JPEG, and related DICOM wrappers. | NEMA/DICOM sample sets and David Clunie references. | DICOM encapsulated codestreams may omit JP2 headers; benchmark decoder path must match DICOM extraction semantics. |
| Library of Congress / NDNP-style JP2 collections | Real cultural-heritage JP2 preservation/access workload. | Library of Congress and National Digital Newspaper Program style collections. | Public access varies by collection; prefer local authorized exports and record source terms. |
| JP2 structure/checker sample sets | JP2 box/metadata parser edge cases. | jp2StructCheck and related preservation-tool samples. | Parser/robustness corpus, not throughput evidence unless files are valid and representative. |
| NITF / geospatial JP2 | JP2 wrapped in geospatial/government imagery workflows. | GDAL JP2OpenJPEG/JP2Grok test data, public NITF samples, agency datasets. | Container handling may be outside raw J2K decode; track extraction route. |

## Starter Acquisition Recipes

Keep downloaded corpora under `target/` or another ignored/vendor directory
unless the dataset license explicitly permits committing the files.

Kodak is a small RGB smoke-to-adoption starter set:

```bash
mkdir -p target/j2k-public-corpora/kodak
for n in $(seq -w 1 24); do
  curl -L --fail \
    "https://r0k.us/graphics/kodak/kodak/kodim${n}.png" \
    -o "target/j2k-public-corpora/kodak/kodim${n}.png"
done
(
  cd target/j2k-public-corpora/kodak
  sha256sum -c - <<'SHA256'
a56e27cbf5f843c048b6af1d6e090760e9c92fadba88b7dee0205918a37523bd  kodim01.png
4f4b74a79237e311d72cad958237b5f7088d8bce1c82305ebefe1a70e3022dfd  kodim02.png
e25ca1ff2f0c0cb5fdfd5f9b0a0bb21ac4c3de3c84a67f35b09a85d3306249db  kodim03.png
e3b946107c5d3441c022f678d0c3caf1e224d81b1604ba840a4f88e562de61aa  kodim04.png
10349e963c5c813d327852f82c1795fa4148d69fedffc4c589bee458e3ac3d53  kodim05.png
363510303b715d4cbc384e1ce227e466b613a09e1b71ae985882bf8e7fbd9b18  kodim06.png
b77d3f006f42414bb242222e0482e750c0fb9e5ee8d4bed2f6f11c5605fe54a4  kodim07.png
ba23983c76b4832ee0e8af0592664756841a16779acd69f792e268fb6d13d6e7  kodim08.png
6a4361c2fc194feb4edaa9f9a4a0620fb9943e460ac7fdf037fb0f6dd6607a7d  kodim09.png
9dfb70f5867c29ff9ed6313683f19b3d867849e40fbc0c4c54a4a89df341cf23  kodim10.png
7936814b58b5387fce2e4e2488b4ec830dadd95fa9520f358ddb30990b50f2b6  kodim11.png
d78c37c2f04f23761ed2367dd77e2db584ddd4c3950833fecf89f199a8126980  kodim12.png
bc34a3ce58dea09dce1704c997171602de90cb34d0c8503a988b77f473d39b08  kodim13.png
55a94550ff18f3246c4074fd32b77b0c74447c26b6ad274d564d999c0450ba6e  kodim14.png
7538cbb80cb9103606c48b806eae57d56c885c7f90b9b3be70a41160f9cbb683  kodim15.png
a89c7268ccd4718ba424a99fc4643c572cf692ca6eae887185ceb4e9f11d2e54  kodim16.png
37afcc89fbdcb76d9518e04b2fc011027e2f4cd14b3b2f83cefd721641a47c5b  kodim17.png
1a9258c365988961d87a0598725b609139c303ad48a5aad6c503c3b1a87849aa  kodim18.png
b7450b264b1b0a411390d8931b112c27905a992520fc90569dc4b920aa32bbdc  kodim19.png
3b46c71e3b92a563820ba32936be8330c586c41f938efd94be938386aae4328a  kodim20.png
ac958597c82073f6bb65129c68f72b651db5b9efd82e11547d07350214bc268b  kodim21.png
1cee58eb1f2d9c7ebb254d208a03c783ce6cf2c4d8c2cf45e235dd23b4ce1b29  kodim22.png
e3111a2fd4da24af15d6459ef9eacfe54106b38e27b4a21821b75c3f5d2d5baf  kodim23.png
1071c68372cc5a01435c2c225a5cf7d4bb803846ec08bb6b3d6721b156d7cb96  kodim24.png
SHA256
)
cargo xtask adoption-materialize \
  --encode-fixtures target/j2k-public-corpora/kodak \
  --source-command "downloaded-from-r0k-us-kodak-lossless-true-color-sha256-pinned" \
  --license-status redistributable \
  --corpus-name kodak \
  --corpus-category natural-image \
  --out-dir target/j2k-public-corpora/materialized-kodak
```

ImageCompression.info adds high-resolution RGB/gray and dimension diversity:

```bash
mkdir -p target/j2k-public-corpora/testimages
curl -L --fail https://imagecompression.info/test_images/rgb8bit.zip \
  -o target/j2k-public-corpora/testimages/rgb8bit.zip
curl -L --fail https://imagecompression.info/test_images/gray8bit.zip \
  -o target/j2k-public-corpora/testimages/gray8bit.zip
unzip -q target/j2k-public-corpora/testimages/rgb8bit.zip \
  -d target/j2k-public-corpora/testimages/rgb8
unzip -q target/j2k-public-corpora/testimages/gray8bit.zip \
  -d target/j2k-public-corpora/testimages/gray8
cargo xtask adoption-materialize \
  --encode-fixtures "target/j2k-public-corpora/testimages/rgb8:target/j2k-public-corpora/testimages/gray8" \
  --source-command "downloaded-from-imagecompression-info-test-images" \
  --license-status redistributable-with-attribution \
  --corpus-name imagecompression-info-test-images \
  --corpus-category natural-image \
  --out-dir target/j2k-public-corpora/materialized-testimages
```

OpenJPEG and jpylyzer are better treated as native fixture/robustness corpora:

```bash
OPENJPEG_DATA_COMMIT=39524bd3a601d90ed8e0177559400d23945f96a9
mkdir -p target/j2k-public-corpora/openjpeg-data
git -C target/j2k-public-corpora/openjpeg-data init
git -C target/j2k-public-corpora/openjpeg-data remote add origin \
  https://github.com/uclouvain/openjpeg-data
git -C target/j2k-public-corpora/openjpeg-data fetch --depth 1 origin \
  "${OPENJPEG_DATA_COMMIT}"
git -C target/j2k-public-corpora/openjpeg-data checkout --detach \
  "${OPENJPEG_DATA_COMMIT}"
cargo xtask adoption-curate \
  --fixtures target/j2k-public-corpora/openjpeg-data/input/conformance \
  --encode-command "source-native-openjpeg-data-conformance-dir@39524bd3a601d90ed8e0177559400d23945f96a9" \
  --license-status permissive-test-fixture \
  --corpus-name openjpeg-data-conformance \
  --corpus-category conformance \
  --out-dir target/j2k-public-corpora/openjpeg-conformance-curated
cargo xtask adoption-curate \
  --fixtures target/j2k-public-corpora/openjpeg-data/input/nonregression \
  --encode-command "source-native-openjpeg-data-nonregression-dir@39524bd3a601d90ed8e0177559400d23945f96a9" \
  --license-status permissive-test-fixture \
  --corpus-name openjpeg-data \
  --corpus-category interop \
  --max-files 24 \
  --out-dir target/j2k-public-corpora/openjpeg-interop-curated

git clone --depth 1 https://github.com/openpreserve/jpylyzer-test-files \
  target/j2k-public-corpora/jpylyzer-test-files
```

`adoption-curate` copies only files that this repo can inspect and fully decode
as supported 8-bit grayscale/RGB JPEG 2000 throughput fixtures, then preflights
the same full-image decode through the OpenJPEG wrapper and the Grok wrapper
when Grok is available. Files that fail parser, decode, comparator, shape, or
metadata checks are recorded in `skipped.tsv` instead of entering the benchmark
manifest. Use jpylyzer invalid files for parser/robustness tests, not
throughput rows.
For larger modern natural-image coverage, use sparse checkout for selected
Imazen codec-corpus subsets:

```bash
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/imazen/codec-corpus \
  target/j2k-public-corpora/codec-corpus
git -C target/j2k-public-corpora/codec-corpus sparse-checkout set clic2025 gb82 gb82-sc
```

## Manifest Generation

Use `cargo xtask adoption-manifest` to create the decode and encode TSVs before
running the adoption bundle. The generator walks each configured directory
recursively, writes canonical absolute paths, infers common corpus categories
from directory names, emits fixture hashes, and requires source/license fields
so publication blockers are explicit instead of hidden in local notes:

```bash
cargo xtask adoption-manifest \
  --decode-fixtures "corpus/vendor/openjpeg-data:corpus/vendor/openjph:corpus/vendor/domain-jp2" \
  --decode-encode-command "source-native-or-recorded-transcode-command" \
  --encode-fixtures "corpus/vendor/kodak:corpus/vendor/tecnick:corpus/vendor/clic:corpus/vendor/domain-source" \
  --encode-source-command "source-original-8bit-gray-rgb" \
  --license-status "redistribution-not-committed-review-notes-present" \
  --out-dir corpus/vendor/adoption-manifests
```

Run it per corpus when license status or source commands differ materially, or
edit the generated TSVs before publication. Unknown directory names are labeled
`external-unspecified`, which remains useful for local runs but should be
replaced by a real corpus category before adoption-facing reporting.

Before publishing a bundle or rendered report, scrub or relativize `path`,
`input_source`, and manifest path fields to corpus labels or repo-relative
artifact paths. Do not publish operator home directories, runner mount points,
private share names, or partner dataset paths; keep raw absolute-path manifests
only in private run artifacts.

## Fixture Materialization

For source-image corpora such as Kodak, Tecnick/TESTIMAGES, CLIC, DIV2K,
Curiosity-style large TIFFs, or extracted WSI/domain tiles, use
`cargo xtask adoption-materialize` to create fixed benchmark inputs before
running the adoption bundle. The command stages each supported 8-bit grayscale
or RGB source image to canonical PGM/PPM, encodes classic J2K and HTJ2K
lossless codestreams through the public J2K CPU facade, emits both raw
codestream variants plus JP2 wrappers for classic J2K and JPH wrappers for
HTJ2K decode coverage, validates CPU round trips, and writes both manifests:

```bash
cargo xtask adoption-materialize \
  --encode-fixtures "corpus/vendor/kodak:corpus/vendor/tecnick:corpus/vendor/clic:corpus/vendor/domain-source" \
  --source-command "source-original-or-recorded-extraction-command" \
  --license-status "redistributable-with-attribution" \
  --out-dir corpus/vendor/materialized-j2k
```

The output layout is:

- `decode-fixtures/classic/*.{j2k,jp2}` and
  `decode-fixtures/htj2k/*.{j2k,jph}` for decode comparisons and CUDA HTJ2K
  subset runs.
- `staged-pnm/*.pgm` / `staged-pnm/*.ppm` for CPU, CUDA, and Metal encode
  source rows.
- `fixtures.tsv` and `encode-fixtures.tsv` with absolute paths, fixture hashes,
  source hashes for materialized decode variants, corpus labels, license status,
  and source/encode command labels.

Then run:

```bash
cargo xtask adoption-benchmark \
  --fixtures corpus/vendor/materialized-j2k/decode-fixtures \
  --manifest corpus/vendor/materialized-j2k/fixtures.tsv \
  --encode-fixtures corpus/vendor/materialized-j2k/staged-pnm \
  --encode-manifest corpus/vendor/materialized-j2k/encode-fixtures.tsv \
  --cuda-decode-batch-sizes 1,16,256,1024 \
  --out-dir target/j2k-adoption-benchmark/full
cargo xtask adoption-report --run-dir target/j2k-adoption-benchmark/full
```

Add `--require-cuda` and `--require-metal` on hardware runners. Those flags are
also report gates: `cargo xtask adoption-report` requires the CUDA/Metal steps
to have run, requires manifest-backed external rows, requires generated
hardware host-input rows to be disabled, and rejects missing Criterion/Metal
evidence before a hardware claim can be publishable. The
materializer requires images at least 128x128 so the downstream CPU encoder
comparator can enforce the shared three-resolution profile. Use
`adoption-manifest` directly for externally supplied native J2K/JP2/JPH files,
ISO conformance attachments, OpenJPEG data, OpenJPH data, and jpylyzer parser
fixtures that should not be re-encoded by this repo.

## Running All Available Corpora

Place or symlink each decoded corpus of J2K/JP2/JPH files into separate
directories, then pass a platform path-list. The harness walks configured
directories recursively and fails if a configured directory contains no
`.j2k`, `.j2c`, `.jp2`, `.jph`, or `.jhc` fixtures:

```bash
J2K_REQUIRE_OPENJPEG=1 J2K_REQUIRE_GROK=1 \
J2K_INCLUDE_OPENJPH=1 J2K_REQUIRE_OPENJPH=1 \
J2K_OPENJPH_EXPAND_BIN=/path/to/ojph_expand \
J2K_FIXTURE_COMPARE_INCLUDE_GENERATED=0 \
J2K_FIXTURE_COMPARE_MANIFEST="corpus/vendor/fixtures.tsv" \
J2K_FIXTURE_COMPARE_INPUT_DIRS="corpus/vendor/iso-j2k:corpus/vendor/openjpeg-data:corpus/vendor/openjph:corpus/vendor/jpylyzer-valid:corpus/vendor/kodak-htj2k:corpus/vendor/tecnick-htj2k:corpus/vendor/clic-htj2k:corpus/vendor/domain" \
cargo run -p j2k-compare --release --bin jp2k_fixture_compare
```

For a full adoption bundle that reuses the existing CPU, CUDA, and Metal
benchmark assets without duplicating benchmark logic, use:

```bash
cargo xtask adoption-benchmark \
  --fixtures "corpus/vendor/iso-j2k:corpus/vendor/openjpeg-data:corpus/vendor/openjph:corpus/vendor/kodak-htj2k:corpus/vendor/domain" \
  --manifest corpus/vendor/fixtures.tsv \
  --encode-fixtures "corpus/vendor/kodak-pnm:corpus/vendor/tecnick-pnm:corpus/vendor/clic-pnm:corpus/vendor/domain-pnm" \
  --encode-manifest corpus/vendor/encode-fixtures.tsv \
  --require-cuda \
  --cuda-decode-batch-sizes 1,16,256,1024 \
  --require-metal \
  --out-dir target/j2k-adoption-benchmark/full
cargo xtask adoption-report --run-dir target/j2k-adoption-benchmark/full
```

With `--require-cuda`, the report checks CUDA decode and encode evidence: same
pinned fixture/source manifests, external case counts, generated CUDA inputs
disabled, CUDA decode device-resident output policy, CUDA encode host-input
timing policy, and Criterion estimates. With `--require-metal`, it checks the
Metal auto-routing run: same pinned staged PNM manifest, external case counts,
generated Metal host inputs disabled, no skipped auto rows, no probe errors, and
the Metal timing policy. Omit the `--require-*` flag when the hardware rows are
diagnostic context rather than part of the adoption claim.

On the self-hosted CUDA GitHub runner, dispatch `GPU validation` with
`run-adoption-benchmark=true`. For the full pinned corpus, set repository variables
`J2K_ADOPTION_FIXTURES`, `J2K_ADOPTION_MANIFEST`,
`J2K_ADOPTION_ENCODE_FIXTURES`, and `J2K_ADOPTION_ENCODE_MANIFEST` to the
pinned corpus paths available on that runner. If those variables are absent,
the workflow builds a default public starter corpus from Kodak plus curated
OpenJPEG data under `target/j2k-public-corpora`. The fallback verifies the
Kodak SHA-256 list above and checks out OpenJPEG-data commit
`39524bd3a601d90ed8e0177559400d23945f96a9`. It then runs the same
`adoption-benchmark --cuda --require-cuda` command and uploads the bundle as
the `cuda-adoption-benchmark` artifact.

The fallback is pinned, not floating: it uses SHA-256-checked Kodak PNGs and a
fixed OpenJPEG-data commit before any adoption benchmark rows are generated.

Use `--quick --include-generated` only for local smoke checks. A smoke bundle is
not publication evidence. Full external `adoption-benchmark` runs fail after
writing artifacts if either CPU comparator reports non-publishable metadata.
Pass `--openjph` to add optional OpenJPH context rows for HTJ2K/JPH-compatible
full/scaled fixtures, or `--require-openjph` when absence of `ojph_expand`
should fail the run. These rows use `decode_method=openjph-cli-process-output-pnm`
and should be reported separately from the default in-process J2K/OpenJPEG/Grok
decoder matrix. Set `J2K_OPENJPH_EXPAND_BIN=/path/to/ojph_expand` for
non-standard OpenJPH installs.
Pass `--kakadu` to add optional Kakadu CLI context rows, or `--require-kakadu`
when absence of `kdu_expand`/`kdu_compress` should fail the run. Set
`J2K_KDU_EXPAND_BIN=/path/to/kdu_expand` and
`J2K_KDU_COMPRESS_BIN=/path/to/kdu_compress` for non-standard Kakadu installs.
These rows are proprietary CLI/file-output context rows and should be reported
separately from the default in-process/publication matrix.

The adoption bundle currently contains these classes of evidence:

- `cpu-fixture-compare`: same external J2K/JP2/JPH/JHC fixture bytes decoded by
  J2K, OpenJPEG, and Grok; this is the publishable head-to-head matrix when
  `publication_eligible=true`.
  Optional OpenJPH rows are disabled by default and limited to HTJ2K/JPH-
  compatible full/scaled operations that `ojph_expand` can decode to PGM/PPM
  files. They are useful interoperability context, not default publishable
  evidence against the in-process decoder rows unless the report clearly labels
  the CLI/file-output method.
  Optional Kakadu rows are disabled by default, labeled as
  `kakadu-cli-process-output-pnm`, and currently limited to full/scaled decode
  operations unless native ROI invocation is verified and added.
- `cpu-encode-compare`: same staged 8-bit PNM pixel bytes encoded to lossless
  classic JP2 by J2K, OpenJPEG, and Grok CLI processes. External PNG, JPEG,
  TIFF, BMP, PGM, PPM, and PNM inputs are decoded/staged outside the timed loop;
  this is the publishable CPU encoder head-to-head matrix when
  `publication_eligible=true`.
  Optional Kakadu encode rows use `kdu_compress` with the same staged PNM inputs
  and are validated against the same classic lossless JP2 profile, but remain
  separate proprietary CLI context rows.
- `cpu-public-api-encode` and `cpu-public-api-decode`: Criterion component
  microbenchmarks for J2K's public CPU encode/decode surfaces. These are not
  external encoder comparisons.
- `cuda-htj2k-decode`: Criterion CPU-vs-CUDA HTJ2K decode rows. When
  `--fixtures` and `--manifest` are supplied, the adoption runner passes the
  same pinned external fixture manifest through `J2K_CUDA_DECODE_INPUT_DIRS`
  and `J2K_CUDA_DECODE_MANIFEST`; the CUDA bench measures the supported HTJ2K
  subset and reports scanned/skipped fixture counts for classic J2K,
  unsupported-shape, and disabled-format fixtures.
  Use `--cuda-decode-batch-sizes 1,16,256,1024` for large-batch adoption
  evidence in mixed external batch rows; the selected list is emitted as
  `j2k_cuda_decode_batch_sizes` and `j2k_cuda_decode_mixed_batch_sizes`.
  Per-fixture batch rows use `j2k_cuda_decode_case_batch_sizes` so the harness
  still touches every fixture without multiplying every large image by every
  huge batch size.
  The CUDA decode bench emits `j2k_cuda_decode_sample_size` because external
  adoption runs intentionally use a bounded Criterion sample count across all
  CPU and CUDA rows.
  Mixed batch rows use the full external corpus up to batch 16 and then switch
  to tile-sized external cases for larger batches; the emitted
  `j2k_cuda_decode_mixed_large_batch_policy` and
  `j2k_cuda_decode_mixed_large_batch_tile_pixels` rows make that split explicit.
  CUDA decode emits `j2k_cuda_decode_io_policy` to distinguish preloaded
  host-memory fixture bytes and device-resident output surfaces from disk I/O
  throughput.
  Generated CUDA decode fixtures are disabled in external-only runs. When at
  least two supported external HTJ2K fixtures share an output format, the CUDA
  decode bench also emits mixed external batch rows that cycle distinct inputs.
- `cuda-htj2k-encode`: hardware Criterion rows for generated encode-stage
  component workloads and, when `--encode-fixtures` / `--encode-manifest` are
  supplied, external staged PGM/PPM host-input encode rows using the same
  manifest-pinned decoded pixels as the CPU encode source matrix. These rows
  compare J2K CPU HTJ2K encode against J2K CUDA HTJ2K encode, not against
  OpenJPEG/Grok CLI encoders.
  CUDA encode emits `j2k_cuda_encode_io_policy`; staged PNM pixels are preloaded
  and filesystem I/O is outside the timed loop.
- `metal-encode-auto-routing`: hardware component microbenchmarks for generated
  encode stages plus external staged PGM/PPM host-input auto-routing rows when
  `--encode-fixtures` / `--encode-manifest` are supplied. These compare J2K CPU
  encode against J2K Metal auto-routing on the same manifest-pinned pixels, not
  against OpenJPEG/Grok CLI encoders.
  Metal encode emits `j2k_metal_encode_io_policy` with the same no-filesystem-
  I/O timing distinction.

Do not claim OpenJPEG/Grok encoder speed from Criterion rows. Use
`cpu-encode-compare` rows, and only when its publication gate is clean.

The default `J2K_FIXTURE_COMPARE_MODE=portable-native` is the publishable
head-to-head mode. It includes only native operations that are comparable
across J2K, OpenJPEG, and Grok. Use `portable-emulated` only for
task-equivalent analysis with `decode_method` labels, and use `capability` only
for feature coverage with explicit skip rows.

The harness also runs generated fixtures by default, labeled
`corpus_category=generated-dev`. Generated-only output is for smoke/development
checks, not adoption-facing evidence. External fixture rows use the source path
in `input_source`, include a `corpus_category`, and report input byte size and
FNV-1a digest for reproducibility. Generated-only output reports
`publication_eligible=false`; adoption-facing reports need external cases and
strict comparator gates. External-corpus publication runs should set
`J2K_FIXTURE_COMPARE_INCLUDE_GENERATED=0` so generated smoke fixtures do not
hide mode exclusions or dilute corpus counts.

External publication runs should also set `J2K_FIXTURE_COMPARE_MANIFEST` to a
tab-separated manifest with one row per external fixture file:

```text
path	corpus_category	corpus_name	license_status	encode_command	input_fnv1a64	source_fnv1a64	codec	container
corpus/vendor/openjpeg-data/input/nonregression/file1.jp2	interop	openjpeg-data	BSD-compatible	source-native	0123456789abcdef	0123456789abcdef	j2k	jp2
```

Required columns are `path` and `corpus_category`. Optional columns are
`corpus_name`, `license_status`, `encode_command`, `input_fnv1a64`,
`source_fnv1a64`, `codec`, and `container`. Container values are
`raw-codestream`, `j2k`, `j2c`, `jp2`, `jph`, or `jhc`; `.jph` and `.jhc`
extensions are preserved in reported container metadata. Relative paths resolve
from the manifest directory. If `input_fnv1a64`, `codec`, or `container` is
present, the harness validates it against the file and fails on mismatch.
`source_fnv1a64` is the digest of the originating source pixels before
materialized raw/boxed fixture variants are encoded; when present, the
publication unique-input and mixed-batch distinct-input gates count this source
hash rather than letting raw plus JP2/JPH wrappers inflate source diversity.
For native compressed fixture corpora, `source_fnv1a64` should normally match
`input_fnv1a64` because the compressed file is the source artifact.
Publication runs require `input_fnv1a64`, `codec`, and `container`; rows that
are present in the manifest but not pinned emit
`manifest_status=covered-unpinned` and block publication. Rows without manifest
coverage emit `manifest_status=not-covered` and also block publication
eligibility.

Encoder publication runs should separately set `J2K_ENCODE_COMPARE_INPUT_DIRS`
and `J2K_ENCODE_COMPARE_MANIFEST` for source images. The encode comparator
accepts 8-bit grayscale/RGB `.pgm`, `.ppm`, `.pnm`, `.png`, `.jpg`, `.jpeg`,
`.tif`, `.tiff`, and `.bmp` files, stages every external source as canonical
PNM outside the timed loop, and then gives the same staged PNM bytes to every
encoder through its CLI surface:

```text
path	corpus_category	corpus_name	license_status	source_command	input_fnv1a64
corpus/vendor/kodak/kodim01.png	natural-image	kodak	cc0	source-png	0123456789abcdef
```

Required columns are `path` and `corpus_category`. Optional columns are
`corpus_name`, `license_status`, `source_command`, and `input_fnv1a64`.
Relative paths resolve from the manifest directory. `input_fnv1a64` is the
digest of the decoded 8-bit gray/RGB pixels that will be written to the staged
PNM, not the compressed PNG/JPEG/TIFF source bytes. Rows without manifest
coverage, or manifest rows without `input_fnv1a64`, are allowed for development
but block publication eligibility.

The harness emits `publication_blockers`; `publication_eligible=true` only when
that list is `none`. Blockers include case filters, missing strict comparator
gates, missing default per-case or mixed-throughput batch sizes, low repeat counts, skipped comparators,
debug builds, missing git revision, dirty worktrees including untracked files,
generated fixtures in the result table, too few external cases, too few
distinct external input digests, too few independently sourced native compressed
inputs, missing independent native classic J2K or HTJ2K coverage, missing
encoder grayscale/RGB source coverage, low encoder dimension/source-format
diversity, missing mixed external batch groups, insufficient mixed-batch
coverage for required gray/RGB decode or encode groups, missing
codec/container/operation coverage, missing manifest coverage/source terms, and
missing conformance/interoperability/workload corpus categories. Repo-materialized
natural-image codestreams are useful workload diagnostics but do not satisfy the
native compressed codec-coverage gate by themselves. External license
statuses must be explicit publishable terms such as `public-domain`, `cc0`,
`cc-by-4.0`, `mit`, `bsd-3-clause`, `apache-2.0`, `permissive`,
`redistributable`, or `redistributable-with-attribution`; values such as
`unknown`, `restricted`, or `no-redistribution` block adoption-facing
publication.
Treat `publication_blockers` as the reason a run is not marketing-ready.
`cargo xtask adoption-report` refuses to render a publishable report from a
blocked bundle unless `--allow-nonpublishable` is explicitly passed. Reports
created with that override are labeled diagnostic-only.

Decoder batch rows use
`batch_input_policy=rotating-owned-copies-built-outside-timed-loop` and
`sample_order_policy=interleaved-rotating-decoder-order`. This prevents the same
input pointer from being used for every J2K batch slot and avoids measuring all
repeats for one decoder before the next decoder. The fixture comparator also
forces J2K inner decode parallelism to `serial` so OpenJPEG/Grok
single-internal-thread rows are not compared against a J2K auto-parallel
single-image path. Per-fixture rows are still single-fixture microbenchmarks.
External runs also emit `external_mixed_*` rows with method-specific
`decode_method` labels such as `native-mixed-external-batch`,
`emulated-full-scaled-crop-mixed-external-batch`, or
`openjph-cli-process-output-pnm-mixed-external-batch`. Those rows group external
fixtures by pixel format and operation kind, then cycle the same ordered fixture
sequence through each eligible decoder. The default publication shape measures
per-fixture detail rows at batch `1` and mixed decode throughput rows at
`1,16,256,1024`; use the mixed rows for huge batch adoption claims, and use the
per-fixture rows to diagnose fixture-level behavior. The publication gate also
requires at least two independent source inputs in each gray/RGB full-image
mixed decode group and in ROI-scaled mixed groups that remain in the selected
comparable mode. The report includes a dedicated CPU decode mixed-batch section
plus `mixed_external_group_distinct_inputs`.

Encoder batch rows use `sample_order_policy=interleaved-rotating-encoder-order`
and spawn all encoders through CLI processes, including J2K's own hidden
`--encode-one` path. OpenJPEG and Grok are invoked with an explicit classic
lossless JP2 profile: `-n 3 -b 64,64 -p LRCP`, plus `-threads 1` and
`OPJ_NUM_THREADS=1` for OpenJPEG and `-H 1` for Grok. Validation decodes the
output and inspects the produced JP2 codestream for single-tile LRCP,
reversible 5/3, three resolution levels, 64x64 code blocks, no precinct
overrides, no SOP/EPH markers, and classic block coding. The default
publication shape measures per-source detail rows at batch `1` and mixed-source
encode throughput rows at `1,16,256`; use the mixed rows and MiB/s columns, not
only images/sec or tiles/sec, for mixed-dimension huge batch claims. The encode
publication gate requires separate gray/RGB mixed groups with at least two
independent inputs, and the report includes a dedicated CPU encode mixed-batch
section.

CUDA encode host-input rows accept staged binary PGM/PPM sources via
`J2K_CUDA_ENCODE_INPUT_DIRS` and validate `path` plus `input_fnv1a64` when
`J2K_CUDA_ENCODE_MANIFEST` is supplied. The adoption runner forwards
`--encode-fixtures` and `--encode-manifest` to these variables for `--cuda`
runs, so CUDA encode can use the same canonical PNM pixels as the CPU encoder
matrix. Non-PNM source formats should be staged by `jp2k_encode_compare` or
recorded in the encode manifest before CUDA encode benchmarking.

Metal auto-routing encode rows use the same staged source convention through
`J2K_METAL_ENCODE_INPUT_DIRS` and `J2K_METAL_ENCODE_MANIFEST` when `--metal` is
requested. External Metal rows are emitted as `mode=lossless_external` in
`j2k_metal_encode_auto_bench`; generated stage rows remain component
microbenchmarks and should not be described as external corpus evidence.

OpenJPEG native HTJ2K ROI+scaled rows are not part of `portable-native`
because the in-process OpenJPEG comparator currently returns non-matching
samples for that combined operation. In `capability` mode those rows remain in
the fixture matrix with
`skip_reason=openjpeg-htj2k-roi-scaled-noncomparable`. In
`portable-emulated` mode OpenJPEG is measured as
`decode_method=emulated-full-scaled-crop`, not as a native ROI+scaled decode.
Do not publish those rows as native OpenJPEG numbers. `portable-native` reports
excluded cases in metadata but does not treat those intentionally excluded rows
as a publication blocker.

## Comparator Signoff Versions

Comparator parity is required before publishing J2K or JPEG comparator benchmark
claims. CI installs OpenJPEG, Grok, and libjpeg-turbo from the active Ubuntu
runner packages and runs:

```bash
J2K_REQUIRE_OPENJPEG=1 J2K_REQUIRE_GROK=1 J2K_REQUIRE_LIBJPEG_TURBO=1 \
  cargo xtask j2k-bench-signoff
```

Accepted evidence must record:

- OpenJPEG CLI versions from `opj_compress` and `opj_decompress`, plus the
  in-process OpenJPEG library version reported by `j2k-compare`.
- Grok CLI versions from `grk_compress` and `grk_decompress`, plus the
  in-process Grok version and library path from `pkg-config libgrokj2k` or
  `J2K_GROK_ROOT`.
- libjpeg-turbo from `pkg-config --modversion libturbojpeg`.

The signoff command is fail-closed: required comparator tests must execute at
least the expected parity-test count for each comparator, so missing tools or
all-skipped test binaries cannot produce green benchmark evidence.

External fixtures that inspect successfully but are outside the benchmark
surface, for example unsupported component counts or bit depths, fail the run
instead of being silently omitted. Curate separate directories for supported
throughput fixtures and robustness/parser-only fixtures.

## Benchmark Report Rules

A report must include:

- exact command and environment variables
- host hardware, OS, architecture, and thread count
- comparator availability and versions, including a comparator version value
  for every required comparator
- input source labels, including `j2k-generated` for generated fixtures
- `benchmark_mode` and per-row `decode_method`
  or encode method, depending on comparator
- `required_comparators`, `matched_comparators`, `skipped_comparators`, and
  `publication_eligible`
- `publication_blockers`; publishable head-to-head reports require `none`
- corpus source, license status, and whether inputs were generated or external
- fixture manifest path plus manifest coverage counts
  and encode manifest path plus manifest coverage counts for encoder claims
- git revision, dirty state including untracked files, build profile, and debug assertion status
- `generated_case_count`, `external_case_count`, and
  `external_unique_input_count`
- encoder `external_component_group_count`, `external_dimension_count`, and
  `external_source_format_count`
- `mixed_external_batch_group_count` and
  `mixed_external_max_distinct_inputs`
- `corpus_category` and whether categories were aggregated or reported separately
- encode command/options when natural images are converted to J2K/HTJ2K first
- batch sizes, repeats, samples, median, and mean
- `batch_input_policy`, `batch_input_copy_counts_by_batch`, and
  `mixed_external_batch_policy`
- `sample_order_policy`
- `resolved_workers_by_batch`, `j2k_inner_parallelism_by_batch`, and
  `external_decoder_internal_threads`
- skipped cases and reasons
- `correctness_preflight` and `benchmark_complete` rows from the harness output;
  `correctness_preflight` applies to non-skipped comparator rows

Do not claim standards compliance from speed benchmarks. Compliance claims must
come from ISO/IEC 15444-4 / ITU-T T.803 conformance vectors and the repo's
conformance test results.
