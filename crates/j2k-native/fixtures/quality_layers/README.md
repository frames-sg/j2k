# Quality-layer decode fixture

This synthetic 96 x 80 RGB image was encoded with OpenJPEG 2.5.4 in three
quality layers without per-pass termination, so most code blocks carry their
arithmetic-coded payload in several fragments, one per layer. The tile-wide
decode test gathers those payloads.

The unsigned 8-bit red, green and blue samples at `(x, y)` are:

```text
((13*x + 29*y) mod 251, (7*x + 31*y + 50) mod 251, (17*x + 5*y + 100) mod 251)
```

Create a binary PPM (`P6`, maximum 255) of 96 x 80 with those row-major
samples, then run:

```sh
opj_compress -i 96x80.ppm -o openjpeg_rgb_96x80_3layers_97.j2k -n 4 -I -mct 1 -b 16,16 -r 40,20,10
```

It is single-tile, irreversible 9/7 with three decompositions, the colour
transform and 16 x 16 code blocks.

The samples are generated for j2k and contain no externally sourced image
content. OpenJPEG is an independent fixture encoder, not a production codec
dependency.
