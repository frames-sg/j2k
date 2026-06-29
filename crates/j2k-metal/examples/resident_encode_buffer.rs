// SPDX-License-Identifier: MIT OR Apache-2.0

//! Encode a generated Gray8 tile into a Metal-backed HTJ2K codestream buffer.
//!
//! Run on macOS with:
//! `cargo run -p j2k-metal --example resident_encode_buffer`

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use j2k::{J2kBlockCodingMode, J2kEncodeValidation, J2kLosslessEncodeOptions};
    use j2k_core::{DeviceSubmission, PixelFormat};
    use j2k_metal::{
        submit_lossless_batch_to_metal, MetalBackendSession, MetalEncodeInputStaging,
        MetalLosslessEncodeBatchRequest, MetalLosslessEncodeConfig, MetalLosslessEncodeTile,
    };
    use j2k_native::{DecodeSettings, Image};
    use metal::MTLResourceOptions;

    let width = 8_u32;
    let height = 8_u32;
    let pixels: Vec<u8> = (0..width * height)
        .map(|index| ((index * 31 + index / 5) & 0xff) as u8)
        .collect();
    let session = MetalBackendSession::system_default()?;
    let buffer = session.device().new_buffer_with_bytes_no_copy(
        pixels.as_ptr().cast(),
        pixels.len() as u64,
        MTLResourceOptions::StorageModeShared,
        None,
    );
    let tiles = [MetalLosslessEncodeTile {
        buffer: &buffer,
        byte_offset: 0,
        width,
        height,
        pitch_bytes: width as usize,
        output_width: width,
        output_height: height,
        format: PixelFormat::Gray8,
    }];
    let options = J2kLosslessEncodeOptions::default()
        .with_strict_device_backend()
        .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
        .with_validation(J2kEncodeValidation::CpuRoundTrip);
    let submitted = submit_lossless_batch_to_metal(
        MetalLosslessEncodeBatchRequest {
            tiles: &tiles,
            staging: MetalEncodeInputStaging::AlreadyPaddedContiguous,
            config: MetalLosslessEncodeConfig::default(),
        },
        &options,
        &session,
    )?;
    let batch = submitted.wait()?;
    let encoded = batch
        .outcomes
        .first()
        .ok_or("resident encode batch returned no outcomes")?;
    let codestream = encoded.encoded.codestream_bytes()?;
    let decoded = Image::new(codestream, &DecodeSettings::default())?.decode_native()?;

    assert_eq!(decoded.data, pixels);
    println!("resident_encoded_bytes={}", codestream.len());
    println!(
        "resident_stages coefficient_prep={} packetization={} codestream_assembly={}",
        encoded.resident.coefficient_prep_used,
        encoded.resident.packetization_used,
        encoded.resident.codestream_assembly_used
    );
    println!("resident_input_copy_used={}", encoded.input_copy_used);
    println!(
        "resident_validation_ms={:.3}",
        encoded.validation_duration.as_secs_f64() * 1000.0
    );
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn main() {
    println!("resident Metal encode buffer example requires macOS");
}
