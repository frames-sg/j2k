# j2k-types

Encode-stage types and helpers shared by the J2K crates.

The `j2k` crate, the `j2k-native` engine, and the GPU adapters all use these
types. The crate defines encode-stage jobs, outputs, and dispatch reports,
progression-order encoding, and packet descriptor sorting, plus the
encode-stage accelerator trait and its default CPU-only implementation.

It also defines the direct and referenced decode plans that the GPU adapters
run. Each `J2kDirectGrayscalePlan`, including the component plans of color and
RGBA plans, records the source component's signedness in its `signed` field.
The wavelet selection and result types returned by
`J2kDecoder::decode_wavelets` (`WaveletSelection`, `WaveletImage`, and related
types) are defined here and re-exported by `j2k` and `j2k-native`.

## Links

- API docs: <https://docs.rs/j2k-types>
- Repository: <https://github.com/frames-sg/j2k>
- Support policy: <https://github.com/frames-sg/j2k/blob/main/docs/public-support.md>
