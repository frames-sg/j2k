# JPEG-LS matrix

JPEG-LS codestreams with reference samples, for the `j2k-jpegls` tests.
`corpus/conformance/jpegls_matrix.py` writes them with CharLS 2.4.2 (the
version is in the first line of `cases.tsv`); run it again and commit the
output to change them.

- `cases.tsv`: one line per case: name, source, CharLS encoding options,
  width, height, components, bits per sample, interleave mode, NEAR, restart
  interval, preset parameters, then `offset:length` in `data.bin` of the input
  samples, the codestream, and the reference samples.
- `data.bin`: those byte ranges, concatenated, with identical ranges shared
  (123,514 bytes). Samples are interleaved by pixel, and little-endian 16-bit
  above 8 bits per sample.

## Sources

- `charls`: synthetic images (no patient data) encoded by CharLS through its
  C API; the reference is CharLS's decode.
- `j2k-jpegls`: synthetic images encoded by this crate. The restart-interval
  cases are here because CharLS 2.4 cannot write restart markers; the
  generator checks that CharLS decodes them to the input. The interleaved
  two-component cases are here because CharLS 2.4 neither decodes those scans
  nor encodes them correctly; their reference is the input, so they only
  guard against regressions.
- `dicom`: the JPEG-LS codestream of `MR_small_jpeg_ls_lossless.dcm` from the
  test files bundled with pydicom 2.4.4 (SHA-256
  `b2b69dd2ae854bf7dfada6745709cd5d8a4573ea12387adbbdc56e8be6056206`), a
  64 × 64 16-bit MR image that GDCM 2.2.4's `gdcmconv` compressed. pydicom's
  test-file README says `MR_small` was downsized from `MR1_UNC` in the NEMA
  WG04 data sets. The reference is CharLS's decode. The retained upstream MIT
  license is `LICENSE.pydicom`.

The generator also encodes every case CharLS supports with this crate and
stops unless CharLS decodes the result to the input (within NEAR).

CharLS 2.4 writes a MAXVAL below `2^P - 1` into the LSE segment but does not
code with it as T.87 A.2.1 and A.4.2 specify: when lossless, it reduces
errors modulo `2^P` and clamps predictions to `2^P - 1` instead of using
`RANGE = MAXVAL + 1` and clamping to MAXVAL, and its near-lossless
codestreams diverge from T.87 too. The two custom-MAXVAL cases use content
and parameters for which both give the same codestream; the crate's
round-trip tests cover the rest.
