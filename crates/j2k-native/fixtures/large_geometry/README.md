# Large-coordinate decode fixtures

These two synthetic grayscale images were encoded with OpenJPEG 2.5.4. Their
dimensions are 65,537 x 4 and 4 x 65,537, so each crosses the former dimension
cap while remaining small enough for whole-image correctness checks.

The unsigned 8-bit sample at `(x, y)` is:

```text
(13*x + 29*y + 7*floor(x/127)) mod 251
```

Create a binary PGM (`P5`, maximum 255) with those row-major samples, then run:

```sh
opj_compress -i 65537x4.pgm -o openjpeg_65537x4.j2k -n 2 -p RPCL -PLT
opj_compress -i 4x65537.pgm -o openjpeg_4x65537.j2k -n 2 -p RPCL -PLT
```

Both use reversible 5/3, one decomposition, one tile, 64 x 64 code-blocks,
default precincts, and packet-length markers. Tests derive expected full and
far-edge region samples directly from the formula. The reduced-region check
also exercises the final output sample at half resolution.

`openjpeg_5x5_origin3_tile4.j2k` is a 5 x 5 image with the same formula,
placed at reference-grid origin (3, 3) with 4 x 4 tiles:

```sh
opj_compress -i 5x5.pgm -o openjpeg_5x5_origin3_tile4.j2k -n 2 -d 3,3 -t 4,4
```

Three of its four tiles are one sample wide or tall, so they have no samples at
half resolution. The expected half-resolution samples come from
`opj_decompress -r 1`.

The samples are generated for j2k and contain no externally sourced image
content. OpenJPEG is an independent fixture encoder, not a production codec
dependency.
