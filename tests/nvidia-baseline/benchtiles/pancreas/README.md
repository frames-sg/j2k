# Pancreas WSI benchmark tiles

109 H&E tissue tiles (256×256, baseline JPEG, quality 85) for the
`transcode_compare` JPEG → HTJ2K benchmark.

## Provenance

Derived from an open-access GDC/TCGA pancreas whole-slide image (`Pancreas.svs`,
Aperio SVS). The slide stores its tiles as JPEG 2000 with YCbCr components
(Aperio compression `33003`), so they are not usable as JPEG input directly.

Each tile here was produced by `svs_extract`:
1. decode the J2K tile to component samples (`j2k-native`),
2. convert YCbCr → RGB,
3. keep only tiles with ≥ 60% tissue coverage and visible structure (skipping
   flat stroma and bright glass/background),
4. re-encode as baseline JPEG.

Re-encoding adds one lossy step, so these are realistic WSI *content* at a
realistic tile size, not byte-identical originals — appropriate for a throughput
benchmark, and the PSNR reference is self-consistent across codecs.

## Reproduce

```bash
cargo run --release --manifest-path tests/nvidia-baseline/Cargo.toml --bin svs_extract -- \
  /path/to/Pancreas.svs out_dir --limit 128 --stride 51 --quality 85 --min-tissue 0.6
```
