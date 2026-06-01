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
fn nvjpeg2000_decode_smoke_decodes_nvidia_htj2k_pathology_tile() {
    let jpeg = include_bytes!("../benchtiles/pancreas/tile_00000.jpg");
    let mut session = NvBaselineSession::new().expect("nvJPEG2000 session initializes");
    let codestream = session
        .transcode_jpeg_to_htj2k(jpeg)
        .expect("nvJPEG2000 encodes pathology tile to HTJ2K");

    let decoded = session
        .decode_j2k_interleaved(&codestream.codestream, NvJ2kDecodeFormat::Rgb8)
        .expect("nvJPEG2000 decodes NVIDIA-generated HTJ2K smoke input");

    assert_eq!(decoded.width, codestream.width);
    assert_eq!(decoded.height, codestream.height);
    assert_eq!(decoded.num_components, 3);
    assert_eq!(decoded.bytes_per_sample, 1);
    assert_eq!(
        decoded.pixels.len(),
        codestream.width as usize * codestream.height as usize * 3
    );
}
