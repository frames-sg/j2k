# signinum

Facade crate for the Signinum codec workspace.

Use this crate for application code that needs JPEG, JPEG 2000 / HTJ2K, or tile
decompression without wiring individual codec crates directly.

Backend selection defaults to `Auto`: CPU remains the portable baseline, while
validated Metal and CUDA adapter paths are available only for supported shapes.
Explicit device requests fail clearly when unsupported.

See the workspace README for current support policy.
