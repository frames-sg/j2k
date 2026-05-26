// SPDX-License-Identifier: Apache-2.0

//! Transcode a committed grayscale JPEG fixture into an HTJ2K codestream.
//!
//! Run with:
//! `cargo run -p signinum-transcode --example jpeg_to_htj2k`

use signinum_transcode::{jpeg_to_htj2k, JpegToHtj2kOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let jpeg = include_bytes!("../fixtures/conformance/grayscale_8x8.jpg");
    let encoded = jpeg_to_htj2k(jpeg, &JpegToHtj2kOptions::lossless_53())?;
    let report = &encoded.report;

    assert!(!encoded.codestream.is_empty());
    println!(
        "transcoded {}x{} JPEG with {} component(s) into {} HTJ2K bytes",
        report.width,
        report.height,
        report.component_count,
        encoded.codestream.len()
    );
    Ok(())
}
