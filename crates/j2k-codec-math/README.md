# j2k-codec-math

`j2k-codec-math` is the workspace-owned source of truth for codec constants,
generated backend fragments, and allocation-free helper algorithms that must
remain byte- or numerically-equivalent across CPU, CUDA-Oxide, and Metal
backends.

The crate is `no_std` and intentionally contains no allocation, I/O, backend
dispatch, or kernel-launch policy. Helpers such as canonical Huffman derivation
perform validation and ordinary control flow.

## Third-party code

Parts of this crate are derived from libjpeg-turbo and are covered by its
licenses: see `LICENSE.libjpeg-turbo` and `README.ijg` in this crate, and
[`NOTICES.md`](https://github.com/frames-sg/j2k/blob/main/NOTICES.md). This
software is based in part on the work of the Independent JPEG Group.

## Links

- API docs: <https://docs.rs/j2k-codec-math>
- Repository: <https://github.com/frames-sg/j2k>
- Support policy: <https://github.com/frames-sg/j2k/blob/main/docs/public-support.md>
