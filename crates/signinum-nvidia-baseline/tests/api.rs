// SPDX-License-Identifier: Apache-2.0

#[cfg(not(nvbaseline_built))]
use signinum_nvidia_baseline::{
    nvidia_decode_j2k_interleaved, nvidia_j2k_decode_available, NvBaselineError,
};
use signinum_nvidia_baseline::{NvBaselineSession, NvJ2kDecodeFormat};

#[cfg(not(nvbaseline_built))]
#[test]
fn nvbaseline_session_reports_not_built_when_cpp_baseline_is_unavailable() {
    match NvBaselineSession::new() {
        Err(NvBaselineError::NotBuilt) => {}
        Ok(_) => panic!("nvJPEG2000 session should not build without the nvjpeg2000 feature"),
        Err(error) => panic!("unexpected nvJPEG2000 session error: {error:?}"),
    }
}

#[cfg(not(nvbaseline_built))]
#[test]
fn nvjpeg2000_decode_reports_not_built_when_cpp_baseline_is_unavailable() {
    assert!(!nvidia_j2k_decode_available());
    match nvidia_decode_j2k_interleaved(&[], NvJ2kDecodeFormat::Rgb8) {
        Err(NvBaselineError::NotBuilt) => {}
        Ok(_) => panic!("nvJPEG2000 decode should not run without the C++ baseline"),
        Err(error) => panic!("unexpected nvJPEG2000 decode error: {error:?}"),
    }
}

#[cfg(nvbaseline_built)]
#[test]
fn nvbaseline_session_constructs_when_built() {
    let _session = NvBaselineSession::new().expect("nvJPEG2000 session initializes");
}

#[cfg(nvbaseline_built)]
#[test]
fn nvjpeg2000_decode_smoke_decodes_tiny_htj2k() {
    let pixels = vec![7u8; 64 * 64 * 3];
    let options = signinum_j2k_native::EncodeOptions {
        reversible: true,
        use_ht_block_coding: true,
        num_decomposition_levels: 1,
        ..signinum_j2k_native::EncodeOptions::default()
    };
    let codestream = signinum_j2k_native::encode_htj2k(&pixels, 64, 64, 3, 8, false, &options)
        .expect("encode tiny HTJ2K smoke input");

    let decoded = nvidia_decode_j2k_interleaved(&codestream, NvJ2kDecodeFormat::Rgb8)
        .expect("nvJPEG2000 decodes tiny HTJ2K smoke input");

    assert_eq!(decoded.width, 64);
    assert_eq!(decoded.height, 64);
    assert_eq!(decoded.num_components, 3);
    assert_eq!(decoded.bytes_per_sample, 1);
    assert_eq!(decoded.pixels.len(), pixels.len());
}
