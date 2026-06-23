// SPDX-License-Identifier: MIT OR Apache-2.0

//! Transcode a committed grayscale JPEG fixture into an HTJ2K codestream.
//!
//! Run with:
//! `cargo run -p j2k-transcode --example jpeg_to_htj2k`

use j2k_test_support::JPEG_GRAYSCALE_8X8;
use j2k_transcode::{jpeg_to_htj2k, JpegToHtj2kOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let encoded = jpeg_to_htj2k(JPEG_GRAYSCALE_8X8, &JpegToHtj2kOptions::lossless_53())?;
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
