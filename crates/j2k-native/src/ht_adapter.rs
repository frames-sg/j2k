// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{j2c, HtCodeBlockDecodeJob, HtUvlcTableEntry, Result};

/// Adapter HTJ2K SigProp benchmark state for backend experimentation.
#[doc(hidden)]
pub struct HtSigPropBenchmarkState(j2c::ht_block_decode::HtSigPropBenchmarkState);

impl HtSigPropBenchmarkState {
    /// Coefficient buffer length required by `decode_ht_sigprop_benchmark_state`.
    pub fn output_len(&self) -> usize {
        self.0.output_len()
    }
}

/// Adapter helper that precomputes cleanup-derived SigProp inputs for benchmarks.
#[doc(hidden)]
pub fn prepare_ht_sigprop_benchmark_state(
    job: HtCodeBlockDecodeJob<'_>,
) -> Result<HtSigPropBenchmarkState> {
    let segments = j2c::ht_block_decode::HtCodeBlockSegments::from_combined_payload(
        job.data,
        job.cleanup_length,
        job.refinement_length,
    )?;
    let state = j2c::ht_block_decode::prepare_sigprop_benchmark_state(
        &segments,
        job.missing_bit_planes,
        job.num_bitplanes,
        job.number_of_coding_passes,
        job.stripe_causal,
        job.strict,
        job.width,
        job.height,
        job.width,
    )?;
    Ok(HtSigPropBenchmarkState(state))
}

/// Adapter helper that runs only the HTJ2K significance-propagation phase.
#[doc(hidden)]
pub fn decode_ht_sigprop_benchmark_state(
    state: &mut HtSigPropBenchmarkState,
    output: &mut [u32],
) -> Result<()> {
    j2c::ht_block_decode::decode_sigprop_benchmark_state(&mut state.0, output)
}

/// Adapter HTJ2K VLC table 0 for backend experimentation.
#[doc(hidden)]
pub fn ht_vlc_table0() -> &'static [u16; 1024] {
    &j2c::ht_tables::VLC_TABLE0
}

/// Adapter HTJ2K VLC table 1 for backend experimentation.
#[doc(hidden)]
pub fn ht_vlc_table1() -> &'static [u16; 1024] {
    &j2c::ht_tables::VLC_TABLE1
}

/// Adapter HTJ2K UVLC table 0 for backend experimentation.
#[doc(hidden)]
pub fn ht_uvlc_table0() -> &'static [u16; 320] {
    &j2c::ht_tables::UVLC_TABLE0
}

/// Adapter HTJ2K UVLC table 1 for backend experimentation.
#[doc(hidden)]
pub fn ht_uvlc_table1() -> &'static [u16; 256] {
    &j2c::ht_tables::UVLC_TABLE1
}

/// Adapter HTJ2K cleanup encoder VLC table 0 for backend experimentation.
#[doc(hidden)]
pub fn ht_vlc_encode_table0() -> &'static [u16; 2048] {
    &j2c::ht_encode_tables::HT_VLC_ENCODE_TABLE0
}

/// Adapter HTJ2K cleanup encoder VLC table 1 for backend experimentation.
#[doc(hidden)]
pub fn ht_vlc_encode_table1() -> &'static [u16; 2048] {
    &j2c::ht_encode_tables::HT_VLC_ENCODE_TABLE1
}

/// Adapter HTJ2K cleanup encoder UVLC table for backend experimentation.
#[doc(hidden)]
pub fn ht_uvlc_encode_table() -> &'static [HtUvlcTableEntry; 75] {
    &j2c::ht_encode_tables::HT_UVLC_ENCODE_TABLE
}

/// Adapter HTJ2K cleanup encoder UVLC table packed for byte-addressed backends.
#[doc(hidden)]
pub fn ht_uvlc_encode_table_bytes() -> &'static [u8] {
    &j2c::ht_encode_tables::HT_UVLC_ENCODE_TABLE_BYTES
}
