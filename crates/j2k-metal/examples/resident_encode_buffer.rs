// SPDX-License-Identifier: MIT OR Apache-2.0

//! Encode generated Gray8 tiles into Metal-backed HTJ2K codestream buffers.
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

    let width = 512_u32;
    let height = 512_u32;
    let session = MetalBackendSession::system_default()?;
    let frames = [
        generated_gray8(width, height, 31, 5, 0),
        generated_gray8(width, height, 23, 7, 11),
    ];
    let buffers = frames
        .iter()
        .map(|pixels| {
            session.device().new_buffer_with_data(
                pixels.as_ptr().cast(),
                pixels.len() as u64,
                MTLResourceOptions::StorageModeShared,
            )
        })
        .collect::<Vec<_>>();
    let tiles = buffers
        .iter()
        .map(|buffer| MetalLosslessEncodeTile {
            buffer,
            byte_offset: 0,
            width,
            height,
            pitch_bytes: width as usize,
            output_width: width,
            output_height: height,
            format: PixelFormat::Gray8,
        })
        .collect::<Vec<_>>();
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
    if batch.outcomes.len() != frames.len() {
        return Err("resident encode batch returned an unexpected frame count".into());
    }

    let mut total_codestream_bytes = 0usize;
    for (index, (encoded, pixels)) in batch.outcomes.iter().zip(frames.iter()).enumerate() {
        let codestream = encoded.encoded.codestream_bytes()?;
        let decoded = Image::new(&codestream, &DecodeSettings::default())?.decode_native()?;
        assert_eq!(decoded.data, *pixels);
        total_codestream_bytes = total_codestream_bytes
            .checked_add(codestream.len())
            .ok_or("resident encoded byte count overflow")?;
        println!(
            "frame={index} resident_encoded_bytes={} coefficient_prep={} packetization={} codestream_assembly={} input_copy_used={} validation_ms={:.3}",
            codestream.len(),
            encoded.resident.coefficient_prep_used,
            encoded.resident.packetization_used,
            encoded.resident.codestream_assembly_used,
            encoded.input_copy_used,
            encoded.validation_duration.as_secs_f64() * 1000.0
        );
    }
    println!("resident_batch_frames={}", batch.outcomes.len());
    println!("resident_total_encoded_bytes={total_codestream_bytes}");
    println!(
        "resident_batch_inflight effective={} memory_budget_bytes={}",
        batch.stats.effective_inflight_tiles, batch.stats.effective_memory_budget_bytes
    );
    Ok(())
}

#[cfg(target_os = "macos")]
fn generated_gray8(width: u32, height: u32, mul: u32, div: u32, bias: u32) -> Vec<u8> {
    (0..width * height)
        .map(|index| ((index * mul + index / div + bias) & 0xff) as u8)
        .collect()
}

#[cfg(not(target_os = "macos"))]
fn main() {
    println!("resident Metal encode buffer example requires macOS");
}
