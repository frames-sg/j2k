# j2k-codec-math

`j2k-codec-math` is the workspace-owned source of truth for small codec
constants and pure math tables that must remain byte- or numerically-equivalent
across CPU, CUDA-Oxide, and Metal backends.

The crate is `no_std` and intentionally contains no backend dispatch,
allocation, I/O, or runtime control flow.

## Links

- API docs: <https://docs.rs/j2k-codec-math> (available after the first crates.io release)
- Repository: <https://github.com/frames-sg/j2k>
- Support policy: <https://github.com/frames-sg/j2k/blob/main/docs/public-support.md>
