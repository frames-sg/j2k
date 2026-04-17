# slidecodec-jpeg

Core library crate for `slidecodec`. See the top-level [README](../../README.md)
for project positioning and MSRV.

```rust
use slidecodec_jpeg::Decoder;

let info = Decoder::inspect(bytes)?;
println!("{}×{} {:?}", info.dimensions.0, info.dimensions.1, info.sof_kind);
```
