// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use core::{ffi::c_void, ptr::NonNull};
    use std::sync::Arc;

    use j2k::{BatchDecodeOptions, BatchLayout, EncodedImage};
    use j2k_mpsgraph::{rgb8_nhwc_reference_cpu, MpsGraphBatchDecoder, MpsGraphProgram};
    use j2k_test_support::htj2k_rgb8_fixture_with_pixels;

    fn score(output: &j2k_mpsgraph::MpsGraphRunOutput) -> f32 {
        let mut value = 0.0_f32;
        // SAFETY: `value` is a valid writable F32 output allocation. The run
        // has completed, synchronized its result, and the reference graph
        // produces exactly one F32 value for this one-image batch.
        unsafe {
            output.results()[0].mpsndarray().readBytes_strideBytes(
                NonNull::from(&mut value).cast::<c_void>(),
                core::ptr::null_mut(),
            );
        }
        value
    }

    let (codestream, pixels) = htj2k_rgb8_fixture_with_pixels(64, 64);
    let encoded = Arc::<[u8]>::from(codestream);
    let options = BatchDecodeOptions {
        layout: BatchLayout::Nhwc,
        ..BatchDecodeOptions::default()
    };
    let mut decoder = MpsGraphBatchDecoder::system_default(options)?;
    let graph = MpsGraphProgram::rgb8_nhwc_reference(1, 64, 64)?;
    let oracle = rgb8_nhwc_reference_cpu(&pixels, 1, 64, 64)?[0];

    let completed = decoder.decode(vec![EncodedImage::full(encoded.clone())])?;
    let (mut groups, errors, group_errors) = completed.into_parts();
    if !errors.is_empty() || !group_errors.is_empty() || groups.len() != 1 {
        return Err(std::io::Error::other("completed decode was incomplete").into());
    }
    let completed_score = score(
        &graph
            .submit_completed(decoder.command_queue(), groups.remove(0))?
            .wait()?,
    );

    let prepared = decoder.prepare(vec![EncodedImage::full(encoded.clone())])?;
    let pipelined_score = score(&decoder.run_prepared_group(&graph, &prepared.groups()[0])?);

    let submitted = decoder.submit_prepared_group(&graph, &prepared.groups()[0])?;
    while !submitted.is_complete() {
        std::thread::yield_now();
    }
    let nonblocking_score = score(&submitted.wait()?);

    for (label, observed) in [
        ("completed-buffer", completed_score),
        ("pipelined-blocking", pipelined_score),
        ("nonblocking", nonblocking_score),
    ] {
        if (observed - oracle).abs() > 1.0e-5 {
            return Err(std::io::Error::other(format!(
                "{label} score {observed} differs from CPU oracle {oracle}"
            ))
            .into());
        }
        println!("{label}: score={observed:.6}, CPU oracle={oracle:.6}");
    }
    Ok(())
}

#[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
fn main() {
    eprintln!("the direct MPSGraph reference workflow requires Apple Silicon macOS 11+");
}
