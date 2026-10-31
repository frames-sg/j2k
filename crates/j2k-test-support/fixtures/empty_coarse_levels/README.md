# Empty coarse levels fixtures

Two small images with two decomposition levels whose last tile row (and, in
the RGB image, column) is one sample at an odd position. That tile's two
coarse resolutions are empty (rows `ceil(5/2)` to `ceil(6/2)` and `ceil(5/4)`
to `ceil(6/4)`) while its full resolution is not. A region decode of the tile
must reconstruct it from the first level that has samples, as the full-image
decode does.

The unsigned 8-bit sample of component `c` at `(x, y)` is:

```text
(13*x + 29*y + 37*c + 7) mod 251
```

Create a binary PGM (`P5`, 8 x 6, component 0) and PPM (`P6`, 6 x 6,
components 0 to 2) with those row-major samples, then run OpenJPEG 2.5.4:

```sh
opj_compress -i gray8x6.pgm -o openjpeg_gray_8x6_tile8x5_53.j2k -n 3 -t 8,5
opj_compress -i rgb6x6.ppm -o openjpeg_rgb_6x6_tile5x5_97.j2k -n 3 -t 5,5 -I
```

| File | Contents | SHA-256 |
| --- | --- | --- |
| `openjpeg_gray_8x6_tile8x5_53.j2k` | Gray, 8 x 5 tiles, reversible 5/3; decodes to the samples exactly. | `09b737d7300ea4ddfc81a29740083105594e77c9029a2635c77bd05c98fb4d85` |
| `openjpeg_rgb_6x6_tile5x5_97.j2k` | RGB, 5 x 5 tiles, irreversible 9/7 with the colour transform. | `ea6d998d5d77cc63f86dbda5f9924177848eac1ea25c17cb5ea162ba687561e7` |

The samples are generated for j2k and contain no externally sourced image
content. OpenJPEG is an independent fixture encoder, not a production codec
dependency.
