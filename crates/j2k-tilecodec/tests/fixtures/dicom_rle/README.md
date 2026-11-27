# DICOM RLE fixtures

DICOM RLE Lossless frames (PS3.5 Annex G) with reference samples, for the
`dicom_rle` tests. `corpus/conformance/rle_matrix.py` writes them with
pydicom 2.4.4 and numpy 2.0.2; run it again and commit the output to change
them.

- `cases.tsv`: one line per case: name, source, rows, columns, samples per
  pixel, bits allocated, then `offset:length` in `data.bin` of the RLE frame,
  the reference samples, and pydicom's encoding of the reference.
- `data.bin`: those byte ranges, concatenated (255,322 bytes). Reference
  samples are native little-endian, color-by-pixel (Planar Configuration 0).

## Frames from pydicom's test files

19 frames come from the RLE test files bundled with pydicom 2.4.4 (in
`pydicom/data/test_files/`). Each is the frame's encapsulated data, byte for
byte. pydicom's test-file README says GDCM's `gdcmconv` RLE-encoded the
`rtdose_rle` and `SC_rgb_rle` files.

| File | SHA-256 | Frames |
| --- | --- | --- |
| `MR_small_RLE.dcm` | `2e5cb60878dc0acc494298ccdad28fce2cf14c51096e5d8cedab40248ea02e6c` | 64 × 64, 16-bit signed, 1 frame |
| `rtdose_rle.dcm` | `2f83e3a2ef0de355570c38860b233fc2fa6c37626c81ad080d8661c03a413522` | 10 × 10, 32-bit, 15 frames |
| `SC_rgb_rle.dcm` | `3f98ee352e75b10ccd6d279ca30b0cb1e363a0c9dde318803f0ec660111327d6` | 100 × 100 RGB, 8-bit, 1 frame |
| `SC_rgb_rle_16bit.dcm` | `86504bea4a8cea36ef4f65e687495cea800e7012546a3202a7944b4f06b376b5` | 100 × 100 RGB, 16-bit, 1 frame |
| `SC_rgb_rle_32bit.dcm` | `222c7060be1697a24e6659bd97ef17fe9c8596ee41bba2a89768854eec00e383` | 100 × 100 RGB, 32-bit, 1 frame |

`rtdose_rle_1frame.dcm` (SHA-256
`f4e7a3b7aeb386ca1a2d7460b0c49771d923f6fd8abba08c1d4c7bc5c25cfc27`) holds
the same bytes as frame 0 of `rtdose_rle.dcm`, so it is not repeated. The
two-frame `SC_rgb_rle` variants are left out to keep the fixtures small.

The reference samples are pydicom's decode. The generator also checks them
against the uncompressed originals pydicom ships: `MR_small.dcm`
(`3f27d1c22f1a66e80d7bb7c911e8610fd0bb70325a76746a7adb1c0ddefcf2bb`),
`rtdose.dcm`
(`1d6cc092146d093e086a6bcccef4ebb7d097941343f5cd3b6395d157b64e37e4`) and
`rtdose_1frame.dcm`
(`6685273e1661562f38dbe2b1c6284b9c950c7dbcf080a71c14305d623d0b6090`).

The `SC_rgb_rle` frames are byte-identical to pydicom's encoding of the same
samples. The `MR_small_RLE` and `rtdose_rle` frames are not: their encoder
merges a two-byte repeat between literal runs into the literal run, as PS3.5
Annex G.3.1 suggests, and the `rtdose_rle` frames leave odd-length segments
unpadded.

## Synthetic frames

14 small frames, made by the generator and encoded with pydicom's RLE
encoder: 8, 16 and 32 bits allocated; 1, 3, 5 and 15 samples per pixel (up
to the 15-segment limit); odd widths; runs of 128, 129, 130, 256 and 257
bytes; rows with no repeated bytes; and single pixels. Their reference is the
generator's own samples.

## License

The test files are distributed in the pydicom 2.4.4 package, which is MIT
licensed. pydicom's license is retained in `LICENSE.pydicom`. pydicom's
test-file README says: "I believe there is no restriction on using any of
these files in this manner."
