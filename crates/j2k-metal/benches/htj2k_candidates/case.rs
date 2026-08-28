// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::{
    encode_j2k_lossless, encode_j2k_lossless_with_accelerator, encode_j2k_lossy,
    encode_j2k_lossy_with_accelerator, EncodeBackendPreference, J2kBlockCodingMode,
    J2kEncodeValidation, J2kLosslessEncodeOptions, J2kLosslessSamples, J2kLossyEncodeOptions,
    J2kLossySamples, J2kQualityLayer, J2kRateTarget,
};
use j2k_core::{BackendKind, PixelFormat};
use j2k_metal::MetalEncodeStageAccelerator;

pub(crate) const SMALL_TILE_SIDE: u32 = 256;
pub(crate) const MEDIUM_TILE_SIDE: u32 = 512;
pub(crate) const LARGE_TILE_SIDE: u32 = 1024;
const COMPONENTS: u16 = 3;
const BIT_DEPTH: u8 = 8;

#[derive(Debug, Clone, Copy)]
pub(crate) enum Profile {
    LosslessTwoLayers,
    LosslessThreeLayers,
    LossyTwoBudgets,
    LossyThreeBudgets,
}

impl Profile {
    pub(crate) const fn id(self) -> &'static str {
        match self {
            Self::LosslessTwoLayers => "lossless-2layers",
            Self::LosslessThreeLayers => "lossless-3layers",
            Self::LossyTwoBudgets => "lossy-1_4bpp",
            Self::LossyThreeBudgets => "lossy-0p5_2_6bpp",
        }
    }

    const fn is_lossless(self) -> bool {
        matches!(self, Self::LosslessTwoLayers | Self::LosslessThreeLayers)
    }

    const fn expects_candidate_sets(self) -> bool {
        matches!(self, Self::LossyTwoBudgets | Self::LossyThreeBudgets)
    }

    const fn quality_layer_count(self) -> u8 {
        match self {
            Self::LosslessTwoLayers | Self::LossyTwoBudgets => 2,
            Self::LosslessThreeLayers | Self::LossyThreeBudgets => 3,
        }
    }
}

pub(crate) struct Workload {
    pub(crate) id: String,
    pub(crate) side: u32,
    pub(crate) profile: Profile,
    pub(crate) pixels: Vec<u8>,
}

pub(crate) struct EncodeOutput {
    pub(crate) codestream: Vec<u8>,
    pub(crate) candidate_set_dispatches: usize,
    pub(crate) ht_dispatches: usize,
}

pub(crate) struct Preflight {
    pub(crate) codestream_bytes: usize,
    pub(crate) candidate_set_dispatches: usize,
    pub(crate) ht_dispatches: usize,
    pub(crate) psnr_db: Option<f64>,
}

pub(crate) fn workloads() -> Vec<Workload> {
    let mut workloads = Vec::new();
    workloads
        .try_reserve_exact(12)
        .expect("benchmark workload table allocation");
    for side in [SMALL_TILE_SIDE, MEDIUM_TILE_SIDE, LARGE_TILE_SIDE] {
        let size = match side {
            SMALL_TILE_SIDE => "small",
            MEDIUM_TILE_SIDE => "medium",
            LARGE_TILE_SIDE => "large",
            _ => unreachable!("benchmark tile table contains an unknown side"),
        };
        let pixels = textured_rgb8(side);
        for profile in [
            Profile::LosslessTwoLayers,
            Profile::LosslessThreeLayers,
            Profile::LossyTwoBudgets,
            Profile::LossyThreeBudgets,
        ] {
            workloads.push(Workload {
                id: format!("{size}-{side}x{side}-{}", profile.id()),
                side,
                profile,
                pixels: pixels.clone(),
            });
        }
    }
    workloads
}

pub(crate) fn encode_cpu(workload: &Workload) -> Result<EncodeOutput, String> {
    if workload.profile.is_lossless() {
        let encoded = encode_j2k_lossless(
            lossless_samples(workload)?,
            &lossless_options(workload, EncodeBackendPreference::CpuOnly),
        )
        .map_err(|error| error.to_string())?;
        Ok(EncodeOutput {
            codestream: encoded.codestream,
            candidate_set_dispatches: 0,
            ht_dispatches: encoded.dispatch_report.ht_code_block,
        })
    } else {
        let encoded = encode_j2k_lossy(
            lossy_samples(workload)?,
            &lossy_options(workload, EncodeBackendPreference::CpuOnly),
        )
        .map_err(|error| error.to_string())?;
        Ok(EncodeOutput {
            codestream: encoded.codestream,
            candidate_set_dispatches: 0,
            ht_dispatches: encoded.dispatch_report.ht_code_block,
        })
    }
}

pub(crate) fn encode_metal(
    workload: &Workload,
    accelerator: &mut MetalEncodeStageAccelerator,
) -> Result<EncodeOutput, String> {
    let candidates_before = accelerator.ht_candidate_set_dispatches();
    let (codestream, ht_dispatches) = if workload.profile.is_lossless() {
        let encoded = encode_j2k_lossless_with_accelerator(
            lossless_samples(workload)?,
            &lossless_options(workload, EncodeBackendPreference::Auto),
            BackendKind::Metal,
            accelerator,
        )
        .map_err(|error| error.to_string())?;
        (encoded.codestream, encoded.dispatch_report.ht_code_block)
    } else {
        let encoded = encode_j2k_lossy_with_accelerator(
            lossy_samples(workload)?,
            &lossy_options(workload, EncodeBackendPreference::Auto),
            BackendKind::Metal,
            accelerator,
        )
        .map_err(|error| error.to_string())?;
        (encoded.codestream, encoded.dispatch_report.ht_code_block)
    };
    Ok(EncodeOutput {
        codestream,
        candidate_set_dispatches: accelerator
            .ht_candidate_set_dispatches()
            .saturating_sub(candidates_before),
        ht_dispatches,
    })
}

pub(crate) fn preflight(
    workload: &Workload,
    accelerator: &mut MetalEncodeStageAccelerator,
) -> Preflight {
    let cpu = encode_cpu(workload)
        .unwrap_or_else(|error| panic!("CPU preflight {} failed: {error}", workload.id));
    let metal = encode_metal(workload, accelerator)
        .unwrap_or_else(|error| panic!("Metal preflight {} failed: {error}", workload.id));

    assert_output_parity(workload, &cpu.codestream, &metal.codestream);
    assert!(
        metal.ht_dispatches > 0,
        "Metal did not dispatch HT Tier-1 for {}",
        workload.id
    );
    if workload.profile.expects_candidate_sets() {
        assert!(
            metal.candidate_set_dispatches > 0,
            "Metal did not dispatch candidate sets for {}",
            workload.id
        );
    } else {
        assert_eq!(
            metal.candidate_set_dispatches, 0,
            "lossless control unexpectedly dispatched candidate sets for {}",
            workload.id
        );
    }

    let psnr_db = if workload.profile.is_lossless() {
        verify_lossless_roundtrip(workload, &cpu.codestream);
        None
    } else {
        Some(verify_lossy_parity(
            workload,
            &cpu.codestream,
            &metal.codestream,
        ))
    };
    Preflight {
        codestream_bytes: cpu.codestream.len(),
        candidate_set_dispatches: metal.candidate_set_dispatches,
        ht_dispatches: metal.ht_dispatches,
        psnr_db,
    }
}

fn assert_output_parity(workload: &Workload, cpu: &[u8], metal: &[u8]) {
    if cpu == metal {
        return;
    }
    let first_difference = cpu
        .iter()
        .zip(metal)
        .position(|(cpu_byte, metal_byte)| cpu_byte != metal_byte)
        .map(|index| (index, cpu[index], metal[index]));
    panic!(
        "CPU and Metal codestreams differ for {}: cpu_len={}, metal_len={}, first_difference={first_difference:?}",
        workload.id,
        cpu.len(),
        metal.len(),
    );
}

fn lossless_samples(workload: &Workload) -> Result<J2kLosslessSamples<'_>, String> {
    J2kLosslessSamples::new(
        &workload.pixels,
        workload.side,
        workload.side,
        COMPONENTS,
        BIT_DEPTH,
        false,
    )
    .map_err(|error| error.to_string())
}

fn lossy_samples(workload: &Workload) -> Result<J2kLossySamples<'_>, String> {
    J2kLossySamples::new(
        &workload.pixels,
        workload.side,
        workload.side,
        COMPONENTS,
        BIT_DEPTH,
        false,
    )
    .map_err(|error| error.to_string())
}

fn lossless_options(
    workload: &Workload,
    backend: EncodeBackendPreference,
) -> J2kLosslessEncodeOptions {
    J2kLosslessEncodeOptions::default()
        .with_backend(backend)
        .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
        .with_max_decomposition_levels(Some(5))
        .with_tile_size(Some((workload.side, workload.side)))
        .with_quality_layers(workload.profile.quality_layer_count())
        .with_validation(J2kEncodeValidation::External)
}

fn lossy_options(workload: &Workload, backend: EncodeBackendPreference) -> J2kLossyEncodeOptions {
    let layer_budgets: &[f64] = match workload.profile {
        Profile::LossyTwoBudgets => &[1.0, 4.0],
        Profile::LossyThreeBudgets => &[0.5, 2.0, 6.0],
        Profile::LosslessTwoLayers | Profile::LosslessThreeLayers => {
            panic!("lossy options requested for a lossless profile")
        }
    };
    let quality_layers = layer_budgets
        .iter()
        .copied()
        .map(|budget| J2kQualityLayer::new(J2kRateTarget::BitsPerPixel(budget)))
        .collect();
    let mut options = J2kLossyEncodeOptions::default()
        .with_backend(backend)
        .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
        .with_max_decomposition_levels(Some(5))
        .with_tile_size(Some((workload.side, workload.side)))
        .with_quality_layers(quality_layers)
        .with_validation(J2kEncodeValidation::External);
    options.psnr_iteration_budget = 1;
    options
}

fn verify_lossless_roundtrip(workload: &Workload, codestream: &[u8]) {
    let decoded = decode_rgb8(workload, codestream);
    assert_eq!(
        decoded, workload.pixels,
        "lossless round trip differs for {}",
        workload.id
    );
}

fn verify_lossy_parity(workload: &Workload, cpu: &[u8], metal: &[u8]) -> f64 {
    let cpu_decoded = decode_rgb8(workload, cpu);
    let metal_decoded = decode_rgb8(workload, metal);
    assert_eq!(
        cpu_decoded, metal_decoded,
        "lossy decoded outputs differ for {}",
        workload.id
    );
    let squared_error = workload
        .pixels
        .iter()
        .zip(&cpu_decoded)
        .map(|(&source, &decoded)| {
            let error = f64::from(source) - f64::from(decoded);
            error * error
        })
        .sum::<f64>();
    let sample_count = f64::from(workload.side) * f64::from(workload.side) * f64::from(COMPONENTS);
    let mse = squared_error / sample_count;
    let psnr = if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0_f64 * 255.0 / mse).log10()
    };
    assert!(
        psnr > 10.0,
        "lossy preflight PSNR is implausibly low for {}: {psnr:.3} dB",
        workload.id
    );
    psnr
}

fn decode_rgb8(workload: &Workload, codestream: &[u8]) -> Vec<u8> {
    let stride = usize::try_from(workload.side)
        .expect("tile side fits usize")
        .checked_mul(usize::from(COMPONENTS))
        .expect("RGB stride fits usize");
    let output_len = stride
        .checked_mul(usize::try_from(workload.side).expect("tile side fits usize"))
        .expect("decoded RGB length fits usize");
    let mut pixels = Vec::new();
    pixels
        .try_reserve_exact(output_len)
        .expect("decoded RGB benchmark allocation");
    pixels.resize(output_len, 0_u8);
    let mut decoder = j2k::J2kDecoder::new(codestream)
        .unwrap_or_else(|error| panic!("decode setup {} failed: {error}", workload.id));
    decoder
        .decode_into(&mut pixels, stride, PixelFormat::Rgb8)
        .unwrap_or_else(|error| panic!("decode {} failed: {error}", workload.id));
    pixels
}

fn textured_rgb8(side: u32) -> Vec<u8> {
    let pixel_count = usize::try_from(side)
        .expect("tile side fits usize")
        .checked_mul(usize::try_from(side).unwrap())
        .expect("tile pixel count fits usize");
    let sample_count = pixel_count
        .checked_mul(usize::from(COMPONENTS))
        .expect("tile sample count fits usize");
    let mut pixels = Vec::new();
    pixels
        .try_reserve_exact(sample_count)
        .expect("benchmark source allocation");
    for y in 0..side {
        for x in 0..side {
            let checker = ((x / 17) ^ (y / 13)) & 1;
            pixels.push(((x * 13 + y * 7 + checker * 61) & 0xff) as u8);
            pixels.push(((x * 3 + y * 19 + (x ^ y) * 5) & 0xff) as u8);
            pixels.push(((x * 23 + y * 11 + ((x * y) >> 4)) & 0xff) as u8);
        }
    }
    pixels
}
