# j2k-transcode

JPEG to HTJ2K transcode crate for J2K.

The crate owns CPU transcode algorithms and shared accelerator hooks. CUDA and
Metal acceleration live in adapter crates. Unsupported source classes and
unsupported transcode modes return explicit errors.
