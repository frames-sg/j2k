// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reversible (5/3) codestreams whose code blocks' coding passes stop early
//! decode with the T.800 Equation E-8 midpoint (r = 1/2) on every native
//! classic output path. See `fixtures/reversible_early_stop/README.md`.

use j2k_codec_math::classic::reversible_midpoint_bit;
use j2k_native::{
    decode_j2k_code_block_scalar, decode_j2k_sub_band_scalar, CpuDecodeParallelism, DecodeSettings,
    DecoderContext, HtCodeBlockDecoder, Image, J2kCodeBlockDecodeJob, J2kSubBandDecodeJob, Result,
};

/// Lossless; only its LL and coarsest LH code blocks stop early.
const LOSSLESS: &[u8] =
    include_bytes!("../fixtures/reversible_early_stop/emri_small_jpeg_2k_lossless_frame1.j2c");
const LOSSLESS_ORIGINAL: &[u8] =
    include_bytes!("../fixtures/reversible_early_stop/emri_small_frame1.u16le");
/// Every code block stops early, four to a sub-band in the finest three.
const RATE_LIMITED: &[u8] =
    include_bytes!("../fixtures/reversible_early_stop/openjpeg_53_rate3_gray12_64x64.j2k");
const RATE_LIMITED_OPENJPEG: &[u8] =
    include_bytes!("../fixtures/reversible_early_stop/openjpeg_53_rate3_gray12_64x64.u16le");

const PARALLELISMS: [CpuDecodeParallelism; 2] =
    [CpuDecodeParallelism::Serial, CpuDecodeParallelism::Auto];

fn decode_native(codestream: &[u8], parallelism: CpuDecodeParallelism) -> Vec<u8> {
    let image = Image::new(codestream, &DecodeSettings::default()).expect("parse codestream");
    let mut context = DecoderContext::default();
    context.set_cpu_decode_parallelism(parallelism);
    image
        .decode_native_with_context(&mut context)
        .expect("decode codestream")
        .data
}

/// Unsigned samples of the first component, clamped to its range as native
/// output is.
fn decode_with_adapter(codestream: &[u8], adapter: &mut dyn HtCodeBlockDecoder) -> Vec<f32> {
    let image = Image::new(codestream, &DecodeSettings::default()).expect("parse codestream");
    let mut context = DecoderContext::default();
    let decoded = image
        .decode_components_with_ht_decoder(&mut context, adapter)
        .expect("decode codestream through the adapter");
    let plane = &decoded.planes()[0];
    let max = f32::from(u16::MAX >> (16 - plane.bit_depth()));
    plane
        .samples()
        .iter()
        .map(|sample| sample.clamp(0.0, max))
        .collect()
}

fn u16_samples(bytes: &[u8]) -> impl Iterator<Item = u16> + '_ {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .copied()
        .map(u16::from_le_bytes)
}

#[test]
fn early_stopped_lossless_code_blocks_decode_to_the_original() {
    // One code block per sub-band, so Auto also uses the serial writer.
    for parallelism in PARALLELISMS {
        assert_eq!(
            decode_native(LOSSLESS, parallelism),
            LOSSLESS_ORIGINAL,
            "{parallelism:?}"
        );
    }
}

#[test]
fn early_stopped_code_blocks_match_openjpeg_on_the_serial_and_parallel_paths() {
    // Four code blocks in a sub-band are enough for Auto to decode in parallel.
    for parallelism in PARALLELISMS {
        assert_eq!(
            decode_native(RATE_LIMITED, parallelism),
            RATE_LIMITED_OPENJPEG,
            "{parallelism:?}"
        );
    }
}

#[test]
fn early_stopped_code_blocks_decode_exactly_on_the_exact_integer_path() {
    // Reversible components above 24 bits decode through exact `i64`
    // coefficients. Raising the lossless fixture's SIZ precision from 16 to 25
    // bits takes that path without touching its Tier-1 data; each sample
    // moves by the change in the DC level shift, 2^24 - 2^15.
    let mut codestream = LOSSLESS.to_vec();
    // SOC, then SIZ: marker, Lsiz, Rsiz, eight 32-bit sizes, Csiz, Ssiz.
    let ssiz = 2 + 2 + 2 + 2 + 8 * 4 + 2;
    assert_eq!(codestream[2..4], [0xff, 0x51], "SIZ follows SOC");
    assert_eq!(codestream[ssiz - 2..ssiz], [0, 1], "one component");
    assert_eq!(codestream[ssiz], 15, "unsigned 16-bit component");
    codestream[ssiz] = 24;
    let expected = u16_samples(LOSSLESS_ORIGINAL)
        .flat_map(|sample| (u32::from(sample) + (1 << 24) - (1 << 15)).to_le_bytes())
        .collect::<Vec<_>>();

    assert_eq!(
        decode_native(&codestream, CpuDecodeParallelism::Serial),
        expected
    );
}

/// An adapter with only the original classic hooks, which return decoded
/// magnitudes without a reconstruction rule. It records how many jobs reach
/// it and how many of them need the midpoint.
#[derive(Default)]
struct LegacyClassicAdapter {
    jobs: usize,
    midpoint_jobs: usize,
}

impl LegacyClassicAdapter {
    fn record(&mut self, job: &J2kCodeBlockDecodeJob<'_>) {
        // A coefficient significant from the block's first pass, with no
        // lower bits set, takes the midpoint whenever any coefficient does.
        let decoded_bitplanes = u32::from(job.total_bitplanes - job.missing_bit_planes);
        let first_significant = 1_u64 << (decoded_bitplanes - 1);
        self.jobs += 1;
        self.midpoint_jobs += usize::from(
            reversible_midpoint_bit(
                first_significant,
                decoded_bitplanes,
                u32::from(job.number_of_coding_passes),
            )
            .is_some(),
        );
    }
}

impl HtCodeBlockDecoder for LegacyClassicAdapter {
    fn decode_j2k_sub_band(
        &mut self,
        job: J2kSubBandDecodeJob<'_>,
        output: &mut [f32],
    ) -> Result<bool> {
        for batch_job in job.jobs {
            self.record(&batch_job.code_block);
        }
        // For complete code blocks the decoded magnitudes are exact.
        decode_j2k_sub_band_scalar(job, output)?;
        Ok(true)
    }

    fn decode_j2k_code_block(
        &mut self,
        job: J2kCodeBlockDecodeJob<'_>,
        output: &mut [f32],
    ) -> Result<bool> {
        self.record(&job);
        decode_j2k_code_block_scalar(job, output)?;
        Ok(true)
    }
}

/// Declines every hook, so the native decoder falls back per code block.
struct DecliningAdapter;

impl HtCodeBlockDecoder for DecliningAdapter {}

#[test]
fn legacy_adapters_never_receive_early_stopped_reversible_code_blocks() {
    for (codestream, expected, adapter_jobs) in [
        // 16 sub-bands of one code block each. Two stop early; the other 14
        // omit only their final cleanup pass, which needs no midpoint.
        (LOSSLESS, LOSSLESS_ORIGINAL, 14),
        (RATE_LIMITED, RATE_LIMITED_OPENJPEG, 0),
    ] {
        let expected = u16_samples(expected).map(f32::from).collect::<Vec<_>>();

        let mut legacy = LegacyClassicAdapter::default();
        assert_eq!(decode_with_adapter(codestream, &mut legacy), expected);
        assert_eq!(legacy.jobs, adapter_jobs);
        assert_eq!(legacy.midpoint_jobs, 0);

        assert_eq!(
            decode_with_adapter(codestream, &mut DecliningAdapter),
            expected
        );
    }
}
