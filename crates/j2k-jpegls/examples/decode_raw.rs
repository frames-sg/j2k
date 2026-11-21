// SPDX-License-Identifier: MIT OR Apache-2.0

//! Decode a JPEG-LS codestream to raw interleaved samples.
//!
//! ```text
//! cargo run -p j2k-jpegls --example decode_raw -- INPUT OUTPUT
//! ```
//!
//! `OUTPUT` receives the samples interleaved by pixel: one byte each up to
//! 8 bits, two little-endian bytes above.

use std::error::Error;

use j2k_jpegls::Decoder;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [input, output] = &args[..] else {
        return Err("usage: decode_raw INPUT OUTPUT".into());
    };
    let data = std::fs::read(input)?;
    let decoder = Decoder::new(&data)?;
    let len = decoder.info().output_len().ok_or("image too large")?;
    let mut samples: Vec<u8> = std::iter::repeat_n(0, len).collect();
    let stride = decoder.info().row_bytes().ok_or("image too large")?;
    decoder.decode_into(&mut samples, stride)?;
    std::fs::write(output, samples)?;
    Ok(())
}
