// SPDX-License-Identifier: MIT OR Apache-2.0

//! Encode raw interleaved samples as a JPEG-LS codestream.
//!
//! ```text
//! cargo run -p j2k-jpegls --example encode_raw -- \
//!     INPUT OUTPUT WIDTH HEIGHT COMPONENTS BITS [NEAR [ILV [RESTART [PRESET]]]]
//! ```
//!
//! `INPUT` holds `WIDTH * HEIGHT * COMPONENTS` samples, interleaved by pixel:
//! one byte each up to 8 bits, two little-endian bytes above. `ILV` is 0, 1,
//! or 2; `RESTART` is lines per restart interval (0 for none); `PRESET` is
//! `MAXVAL,T1,T2,T3,RESET` (0 for a default) or `-`.

use std::error::Error;

use j2k_jpegls::{
    encode_jpegls, InterleaveMode, JpegLsEncodeOptions, JpegLsSamples, PresetCodingParameters,
};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if !(6..=10).contains(&args.len()) {
        return Err(
            "usage: encode_raw INPUT OUTPUT WIDTH HEIGHT COMPONENTS BITS \
                    [NEAR [ILV [RESTART [PRESET]]]]"
                .into(),
        );
    }
    let arg = |index: usize| args.get(index).map_or("0", String::as_str);
    let raw = std::fs::read(&args[0])?;
    let width: u32 = arg(2).parse()?;
    let height: u32 = arg(3).parse()?;
    let components: u8 = arg(4).parse()?;
    let bits: u8 = arg(5).parse()?;
    let mut options = JpegLsEncodeOptions::new(bits).with_near(arg(6).parse()?);
    options.interleave = match arg(7) {
        "0" => InterleaveMode::None,
        "1" => InterleaveMode::Line,
        "2" => InterleaveMode::Sample,
        other => return Err(format!("ILV must be 0, 1, or 2, got {other}").into()),
    };
    let restart: u16 = arg(8).parse()?;
    if restart != 0 {
        options = options.with_restart_interval(restart);
    }
    if let Some(preset) = args.get(9).filter(|preset| preset.as_str() != "-") {
        let values = preset
            .split(',')
            .map(str::parse)
            .collect::<Result<Vec<u16>, _>>()?;
        let [max_value, t1, t2, t3, reset] = values[..] else {
            return Err("PRESET must be MAXVAL,T1,T2,T3,RESET".into());
        };
        options = options.with_preset(PresetCodingParameters {
            max_value,
            t1,
            t2,
            t3,
            reset,
        });
    }
    let words: Vec<u16>;
    let samples = if bits > 8 {
        words = raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| u16::from_le_bytes(*pair))
            .collect();
        JpegLsSamples::U16 {
            data: &words,
            width,
            height,
            components,
        }
    } else {
        JpegLsSamples::U8 {
            data: &raw,
            width,
            height,
            components,
        }
    };
    let encoded = encode_jpegls(samples, &options)?;
    std::fs::write(&args[1], encoded.data)?;
    Ok(())
}
