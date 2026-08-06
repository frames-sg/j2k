// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::encoder::{EncoderCase, EncoderInputKind, EncoderPattern};

pub(super) struct GeneratedInput {
    pub(super) interleaved: Vec<u8>,
    pub(super) components: Vec<GeneratedComponent>,
}

pub(super) struct GeneratedComponent {
    pub(super) data: Vec<u8>,
    pub(super) dimensions: [u32; 2],
    pub(super) sampling: [u8; 2],
    pub(super) bit_depth: u8,
    pub(super) signed: bool,
    pub(super) samples: Vec<i32>,
}

pub(super) fn generate_input(case: &EncoderCase) -> Result<GeneratedInput, String> {
    let component_count = usize::from(case.components);
    let mut components = Vec::new();
    components
        .try_reserve_exact(component_count)
        .map_err(|error| format!("allocate generated component descriptors: {error}"))?;
    for component in 0..component_count {
        let sampling = case.sampling.get(component).copied().unwrap_or([1, 1]);
        let bit_depth = case
            .component_bit_depths
            .get(component)
            .copied()
            .unwrap_or(case.bit_depth);
        let signed = case
            .component_signedness
            .get(component)
            .copied()
            .unwrap_or(case.signed);
        components.push(generate_component(
            case, component, sampling, bit_depth, signed,
        )?);
    }

    let interleaved = if case.input == EncoderInputKind::Interleaved {
        interleave(case, &components)?
    } else {
        Vec::new()
    };
    Ok(GeneratedInput {
        interleaved,
        components,
    })
}

fn generate_component(
    case: &EncoderCase,
    component: usize,
    sampling: [u8; 2],
    bit_depth: u8,
    signed: bool,
) -> Result<GeneratedComponent, String> {
    let dimensions = [
        case.width.div_ceil(u32::from(sampling[0])),
        case.height.div_ceil(u32::from(sampling[1])),
    ];
    let sample_count = checked_sample_count(dimensions)?;
    let bytes_per_sample = usize::from(bit_depth).div_ceil(8);
    let byte_count = sample_count
        .checked_mul(bytes_per_sample)
        .ok_or_else(|| format!("{} generated component byte count overflows", case.id))?;
    let mut data = Vec::new();
    data.try_reserve_exact(byte_count)
        .map_err(|error| format!("allocate {} component bytes: {error}", case.id))?;
    let mut samples = Vec::new();
    samples
        .try_reserve_exact(sample_count)
        .map_err(|error| format!("allocate {} component samples: {error}", case.id))?;
    for y in 0..dimensions[1] {
        for x in 0..dimensions[0] {
            let sample = generated_sample(case.pattern, x, y, component, bit_depth, signed);
            samples.push(sample);
            append_sample(&mut data, sample, bit_depth);
        }
    }
    debug_assert_eq!(data.len(), byte_count);
    Ok(GeneratedComponent {
        data,
        dimensions,
        sampling,
        bit_depth,
        signed,
        samples,
    })
}

fn interleave(case: &EncoderCase, components: &[GeneratedComponent]) -> Result<Vec<u8>, String> {
    let pixel_count = (case.width as usize)
        .checked_mul(case.height as usize)
        .ok_or_else(|| format!("{} interleaved pixel count overflows", case.id))?;
    let bytes_per_sample = usize::from(case.bit_depth).div_ceil(8);
    let byte_count = pixel_count
        .checked_mul(components.len())
        .and_then(|count| count.checked_mul(bytes_per_sample))
        .ok_or_else(|| format!("{} interleaved byte count overflows", case.id))?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(byte_count)
        .map_err(|error| format!("allocate {} interleaved samples: {error}", case.id))?;
    for sample_index in 0..pixel_count {
        for component in components {
            append_sample(&mut output, component.samples[sample_index], case.bit_depth);
        }
    }
    debug_assert_eq!(output.len(), byte_count);
    Ok(output)
}

fn checked_sample_count(dimensions: [u32; 2]) -> Result<usize, String> {
    (dimensions[0] as usize)
        .checked_mul(dimensions[1] as usize)
        .ok_or_else(|| "generated component sample count overflows".to_string())
}

fn generated_sample(
    pattern: EncoderPattern,
    x: u32,
    y: u32,
    component: usize,
    bit_depth: u8,
    signed: bool,
) -> i32 {
    let modulus = 1_u64 << bit_depth;
    let raw = match pattern {
        EncoderPattern::Gradient => {
            (u64::from(x) * 17 + u64::from(y) * 31 + component as u64 * 47) % modulus
        }
        EncoderPattern::Checkerboard => {
            if (x + y + u32::try_from(component).unwrap_or_default()) & 1 == 0 {
                0
            } else {
                modulus - 1
            }
        }
        EncoderPattern::DeterministicNoise => {
            splitmix64(u64::from(x) | (u64::from(y) << 21) | ((component as u64) << 42)) % modulus
        }
        EncoderPattern::Impulse => {
            if x == 0 && y == 0 {
                modulus - 1
            } else if signed {
                modulus / 2
            } else {
                0
            }
        }
    };
    if signed {
        let raw = i64::try_from(raw).expect("31-bit generated sample fits i64");
        let midpoint = i64::try_from(modulus / 2).expect("31-bit midpoint fits i64");
        i32::try_from(raw - midpoint).expect("31-bit signed generated sample fits i32")
    } else {
        i32::try_from(raw).expect("31-bit unsigned generated sample fits i32")
    }
}

fn append_sample(output: &mut Vec<u8>, sample: i32, bit_depth: u8) {
    let modulus = 1_i64 << bit_depth;
    let raw = if sample < 0 {
        i64::from(sample) + modulus
    } else {
        i64::from(sample)
    };
    let raw = u64::try_from(raw).expect("generated sample is normalized to a non-negative value");
    let bytes = raw.to_le_bytes();
    output.extend_from_slice(&bytes[..usize::from(bit_depth).div_ceil(8)]);
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}
