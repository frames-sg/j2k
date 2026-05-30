// SPDX-License-Identifier: Apache-2.0

#[cfg(not(nvbaseline_built))]
use signinum_nvidia_baseline::NvBaselineError;
use signinum_nvidia_baseline::NvBaselineSession;

#[cfg(not(nvbaseline_built))]
#[test]
fn nvbaseline_session_reports_not_built_when_cpp_baseline_is_unavailable() {
    match NvBaselineSession::new() {
        Err(NvBaselineError::NotBuilt) => {}
        Ok(_) => panic!("nvJPEG2000 session should not build without the nvjpeg2000 feature"),
        Err(error) => panic!("unexpected nvJPEG2000 session error: {error:?}"),
    }
}

#[cfg(nvbaseline_built)]
#[test]
fn nvbaseline_session_constructs_when_built() {
    let _session = NvBaselineSession::new().expect("nvJPEG2000 session initializes");
}
