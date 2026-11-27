// SPDX-License-Identifier: MIT OR Apache-2.0

//! Opening the start of a codestream with `J2K_OPEN_PARTIAL`.

use j2k::{
    encode_j2k_lossless, J2kBlockCodingMode, J2kLosslessEncodeOptions, J2kLosslessSamples,
    J2kProgressionOrder,
};
use j2k_c::*;
use std::ptr;

const SIDE: u32 = 64;

fn codestream() -> Vec<u8> {
    let pixels: Vec<u8> = (0..SIDE * SIDE)
        .map(|i| u8::try_from((((i % SIDE) * 7) ^ ((i / SIDE) * 3)) % 256).unwrap())
        .collect();
    let samples = J2kLosslessSamples::new(&pixels, SIDE, SIDE, 1, 8, false).expect("samples");
    let options = J2kLosslessEncodeOptions::default()
        .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
        .with_progression(J2kProgressionOrder::Rpcl)
        .with_max_decomposition_levels(Some(3));
    encode_j2k_lossless(samples, &options)
        .expect("encode")
        .codestream
}

fn open(bytes: &[u8], flags: u32) -> Result<*mut J2kCDecoder, u32> {
    let options = J2kCOpenOptions {
        flags,
        ..J2kCOpenOptions::default()
    };
    let mut handle = ptr::null_mut();
    // SAFETY: All buffers and output slots are valid and disjoint for this call.
    let status = unsafe {
        j2k_decoder_open(
            bytes.as_ptr(),
            bytes.len(),
            &raw const options,
            &raw mut handle,
            ptr::null_mut(),
        )
    };
    if status == J2K_OK {
        Ok(handle)
    } else {
        Err(status)
    }
}

fn close(mut handle: *mut J2kCDecoder) {
    // SAFETY: This is the only owner and its slot is valid for writing.
    let status = unsafe { j2k_decoder_close(&raw mut handle, ptr::null_mut()) };
    assert_eq!(status, J2K_OK);
}

fn completeness(handle: *mut J2kCDecoder) -> J2kCCompleteness {
    let mut out = J2kCCompleteness::default();
    // SAFETY: The handle is live and used exclusively; `out` is initialized.
    let status = unsafe { j2k_decoder_completeness(handle, &raw mut out, ptr::null_mut()) };
    assert_eq!(status, J2K_OK);
    out
}

fn decode(handle: *mut J2kCDecoder, reduction: u32) -> Result<Vec<u8>, u32> {
    let options = J2kCDecodeOptions {
        reduction,
        ..J2kCDecodeOptions::default()
    };
    let mut layout = J2kCOutputInfo::default();
    // SAFETY: The handle is live and used exclusively; arguments are disjoint.
    let status = unsafe {
        j2k_decoder_output_info(handle, &raw const options, &raw mut layout, ptr::null_mut())
    };
    if status != J2K_OK {
        return Err(status);
    }
    let mut out = vec![0_u8; usize::try_from(layout.buffer_bytes).expect("size")];
    // SAFETY: `out` holds `buffer_bytes` writable bytes with the planned stride.
    let status = unsafe {
        j2k_decoder_decode_into(
            handle,
            &raw const options,
            out.as_mut_ptr(),
            out.len(),
            usize::try_from(layout.row_bytes).expect("stride"),
            ptr::null_mut(),
        )
    };
    if status == J2K_OK {
        Ok(out)
    } else {
        Err(status)
    }
}

#[test]
fn partial_input_decodes_the_reductions_it_holds() {
    let full = codestream();
    let handle = open(&full, 0).expect("complete codestream");
    let references: Vec<Vec<u8>> = (0..=3)
        .map(|r| decode(handle, r).expect("reference"))
        .collect();
    close(handle);

    let mut reductions_seen = Vec::new();
    for len in (1..=full.len())
        .step_by((full.len() / 150).max(1))
        .chain([full.len()])
    {
        let Ok(handle) = open(&full[..len], J2K_OPEN_PARTIAL) else {
            continue;
        };
        let report = completeness(handle);
        if report.has_exact_reduction == 1 {
            let reduction = report.min_exact_reduction;
            if !reductions_seen.contains(&reduction) {
                reductions_seen.push(reduction);
            }
            assert_eq!(
                decode(handle, reduction).expect("reported reduction decodes"),
                references[reduction as usize],
                "prefix {len} at {reduction}"
            );
            if reduction > 0 {
                assert_eq!(
                    decode(handle, reduction - 1),
                    Err(J2K_INCOMPLETE),
                    "prefix {len}"
                );
            }
        } else {
            assert_eq!(decode(handle, 3), Err(J2K_INCOMPLETE), "prefix {len}");
        }
        if len == full.len() {
            assert_eq!(report.complete, 1);
            assert_eq!(report.min_exact_reduction, 0);
        }
        close(handle);
    }
    for reduction in 0..=3 {
        assert!(reductions_seen.contains(&reduction), "{reductions_seen:?}");
    }
}

#[test]
fn without_the_flag_a_cut_off_codestream_is_not_decoded() {
    let full = codestream();
    let prefix = &full[..full.len() / 2];
    if let Ok(handle) = open(prefix, 0) {
        let status = decode(handle, 3).expect_err("strict decode of a cut-off codestream");
        assert_ne!(status, J2K_INCOMPLETE);
        close(handle);
    }
}
