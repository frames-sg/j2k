# signinum-transcode-metal

Metal acceleration experiments for `signinum-transcode`.

The crate is intentionally optional. CPU JPEG parsing, entropy decode,
dequantization, and HTJ2K assembly stay outside this crate; this crate only
implements transform-stage acceleration hooks.
