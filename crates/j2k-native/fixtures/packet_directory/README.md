# Packet-directory fixture

`openjpeg_rgb_origin5x3_rpcl.j2k` was generated with OpenJPEG 2.5.4 from a
129×97 RGB8 PPM. Each source sample is
`(x*17 + y*31 + c*79 + (x^y)) % 256`, with `c` from 0 through 2.

```
opj_compress -i input.ppm -o openjpeg_rgb_origin5x3_rpcl.j2k \
  -n 4 -p RPCL -r 8,4,1 -c '[64,64]' -b 32,8 -SOP -EPH -d 5,3
```

This exercises independent packet encoding, three layers, reversible 5/3,
non-square blocks, SOP/EPH, and a nonzero image origin. It has one tile-part,
inline headers, no POC, and no PLT. It is a generated approximation, not an
archival workload source.
