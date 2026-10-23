# Image-origin region decode fixtures

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

The first is reversible 5/3 with four decompositions and one tile. The second
is irreversible 9/7 with three decompositions and 32 x 32 tiles whose grid
starts at (3, 1).

The samples are generated for j2k and contain no externally sourced image
content. OpenJPEG is an independent fixture encoder, not a production codec
dependency.
