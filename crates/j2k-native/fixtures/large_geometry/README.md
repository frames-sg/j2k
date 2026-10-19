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

The samples are generated for j2k and contain no externally sourced image
content. OpenJPEG is an independent fixture encoder, not a production codec
dependency.
