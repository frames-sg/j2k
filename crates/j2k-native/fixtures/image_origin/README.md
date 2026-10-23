# Image-origin decode fixtures

These synthetic 71 x 53 grayscale images were encoded with OpenJPEG 2.5.4.
They sit at odd reference-grid origins, so every resolution's region windows
start on odd rows and columns. The region tests compare the row-streamed and
whole-window region IDWT on them.

The unsigned 8-bit sample at `(x, y)` is:

```text
(13*x + 29*y + 7*floor(x/127)) mod 251
```

Create a binary PGM (`P5`, maximum 255) with those row-major samples, then run:

```sh
opj_compress -i 71x53.pgm -o openjpeg_71x53_origin5x3_53.j2k -n 4 -d 5,3
opj_compress -i 71x53.pgm -o openjpeg_71x53_origin7x1_tile32_97.j2k -n 3 -I -d 7,1 -t 32,32 -T 3,1
```

OpenJPEG's `-n` counts resolutions, so the first is reversible 5/3 with three
decomposition levels and one tile. The second is irreversible 9/7 with two
decomposition levels and 32 x 32 tiles whose grid starts at (3, 1).

Two RGB images at odd origins cover the RGB8 decode that leaves its last
IDWT level to the pack. The unsigned 8-bit red, green and blue samples at
`(x, y)` are:

```text
((13*x + 29*y) mod 251, (7*x + 31*y + 50) mod 251, (17*x + 5*y + 100) mod 251)
```

Create a binary PPM (`P6`, maximum 255) of 75 x 56 and one of 72 x 59 with
those row-major samples, then run:

```sh
opj_compress -i 75x56.ppm -o openjpeg_rgb_75x56_origin5x3_53.j2k -n 5 -d 5,3 -mct 1 -r 8
opj_compress -i 72x59.ppm -o openjpeg_rgb_72x59_origin7x1_97.j2k -n 4 -I -d 7,1 -mct 1 -r 8
```

Both are single-tile with the colour transform: the first reversible 5/3
with four decompositions, the second irreversible 9/7 with three.

The samples are generated for j2k and contain no externally sourced image
content. OpenJPEG is an independent fixture encoder, not a production codec
dependency.
