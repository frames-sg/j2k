//! The irreversible multi-component transformation, as specified in
//! Annex G.2 and G.3.

use super::codestream::{ComponentInfo, Header, WaveletTransform};
use super::decode::TileDecodeContext;
use crate::error::{bail, err, ColorError, Result};
use crate::math::{dispatch, f32x8, floor_f32, Level, Simd};
use crate::{HtCodeBlockDecoder, J2kInverseMctJob, J2kWaveletTransform};
use j2k_codec_math::mct;

/// Apply the inverse multi-component transform, as specified in G.2 and G.3.
pub(crate) fn apply_inverse(
    tile_ctx: &mut TileDecodeContext,
    component_infos: &[super::codestream::ComponentInfo],
    header: &Header<'_>,
    backend: &mut Option<&mut dyn HtCodeBlockDecoder>,
) -> Result<()> {
    if tile_ctx.channel_data.len() < 3 {
        return if header.strict {
            err!(ColorError::Mct)
        } else {
            Ok(())
        };
    }

    let (s, _) = tile_ctx.channel_data.split_at_mut(3);
    let [s0, s1, s2] = s else { unreachable!() };

    let transform = component_infos[0].wavelet_transform();

    if transform != component_infos[1].wavelet_transform()
        || component_infos[1].wavelet_transform() != component_infos[2].wavelet_transform()
    {
        bail!(ColorError::Mct);
    }

    if s0.container.len() != s1.container.len() || s1.container.len() != s2.container.len() {
        bail!(ColorError::Mct);
    }

    let addends = [
        unsigned_level_shift(&component_infos[0]),
        unsigned_level_shift(&component_infos[1]),
        unsigned_level_shift(&component_infos[2]),
    ];

    if s0.integer_container.is_some()
        || s1.integer_container.is_some()
        || s2.integer_container.is_some()
    {
        return apply_inverse_i64(
            transform,
            s0,
            s1,
            s2,
            [
                unsigned_level_shift_i64(&component_infos[0]),
                unsigned_level_shift_i64(&component_infos[1]),
                unsigned_level_shift_i64(&component_infos[2]),
            ],
        );
    }

    let handled = if let Some(backend) = backend.as_deref_mut() {
        backend.decode_inverse_mct(J2kInverseMctJob {
            transform: J2kWaveletTransform::from(transform),
            plane0: &mut s0.container,
            plane1: &mut s1.container,
            plane2: &mut s2.container,
            addend0: addends[0],
            addend1: addends[1],
            addend2: addends[2],
        })?
    } else {
        false
    };

    if !handled {
        apply_inner(
            transform,
            &mut s0.container,
            &mut s1.container,
            &mut s2.container,
            addends,
        );
    }

    Ok(())
}

fn apply_inverse_i64(
    transform: WaveletTransform,
    s0: &mut super::ComponentData,
    s1: &mut super::ComponentData,
    s2: &mut super::ComponentData,
    addends: [i64; 3],
) -> Result<()> {
    if transform != WaveletTransform::Reversible53 {
        bail!(ColorError::Mct);
    }

    let (Some(y0), Some(y1), Some(y2)) = (
        s0.integer_container.as_mut(),
        s1.integer_container.as_mut(),
        s2.integer_container.as_mut(),
    ) else {
        bail!(ColorError::Mct);
    };
    if y0.len() != y1.len() || y1.len() != y2.len() {
        bail!(ColorError::Mct);
    }

    for ((y0, y1), y2) in y0.iter_mut().zip(y1.iter_mut()).zip(y2.iter_mut()) {
        let src0 = *y0;
        let src1 = *y1;
        let src2 = *y2;
        let green = src0 - floor_div_i64(src2 + src1, 4);
        *y0 = src2 + green + addends[0];
        *y1 = green + addends[1];
        *y2 = src1 + green + addends[2];
    }

    Ok(())
}

#[expect(
    clippy::cast_precision_loss,
    reason = "the codec float domain intentionally receives bounded integer samples or metadata at this rounding boundary"
)]
fn unsigned_level_shift(component_info: &ComponentInfo) -> f32 {
    if component_info.size_info.signed {
        0.0
    } else {
        (1_u32 << (component_info.size_info.precision - 1)) as f32
    }
}

fn unsigned_level_shift_i64(component_info: &ComponentInfo) -> i64 {
    if component_info.size_info.signed {
        0
    } else {
        1_i64 << (component_info.size_info.precision - 1)
    }
}

#[expect(
    clippy::inline_always,
    reason = "this scalar primitive is intentionally inlined into the reversible color-transform hot loop"
)]
#[inline(always)]
fn floor_div_i64(numerator: i64, denominator: i64) -> i64 {
    debug_assert!(denominator > 0);
    let quotient = numerator / denominator;
    let remainder = numerator % denominator;
    if remainder != 0 && remainder < 0 {
        quotient - 1
    } else {
        quotient
    }
}

fn apply_inner(
    transform: WaveletTransform,
    s0: &mut [f32],
    s1: &mut [f32],
    s2: &mut [f32],
    addends: [f32; 3],
) {
    dispatch!(Level::new(), simd => apply_inner_impl(simd, transform, s0, s1, s2, addends));
}

#[expect(
    clippy::inline_always,
    reason = "the SIMD implementation is intentionally specialized and inlined at the architecture dispatch boundary"
)]
#[inline(always)]
fn apply_inner_impl<S: Simd>(
    simd: S,
    transform: WaveletTransform,
    s0: &mut [f32],
    s1: &mut [f32],
    s2: &mut [f32],
    addends: [f32; 3],
) {
    match transform {
        // Irreversible MCT, specified in G.3.
        WaveletTransform::Irreversible97 => {
            let red_from_chroma = f32x8::splat(simd, mct::ICT_INV_R_CR);
            let green_from_red_chroma = f32x8::splat(simd, mct::ICT_INV_G_CR);
            let green_from_blue_chroma = f32x8::splat(simd, mct::ICT_INV_G_CB);
            let blue_from_chroma = f32x8::splat(simd, mct::ICT_INV_B_CB);
            let red_level = f32x8::splat(simd, addends[0]);
            let green_level = f32x8::splat(simd, addends[1]);
            let blue_level = f32x8::splat(simd, addends[2]);
            let mut s0_chunks = s0.chunks_exact_mut(8);
            let mut s1_chunks = s1.chunks_exact_mut(8);
            let mut s2_chunks = s2.chunks_exact_mut(8);
            for ((y0, y1), y2) in s0_chunks
                .by_ref()
                .zip(s1_chunks.by_ref())
                .zip(s2_chunks.by_ref())
            {
                let y_0 = f32x8::from_slice(simd, y0);
                let y_1 = f32x8::from_slice(simd, y1);
                let y_2 = f32x8::from_slice(simd, y2);

                let i0 = y_2.mul_add(red_from_chroma, y_0) + red_level;
                let i1 = y_2.mul_add(
                    green_from_red_chroma,
                    y_1.mul_add(green_from_blue_chroma, y_0),
                ) + green_level;
                let i2 = y_1.mul_add(blue_from_chroma, y_0) + blue_level;

                i0.store(y0);
                i1.store(y1);
                i2.store(y2);
            }
            for ((y0, y1), y2) in s0_chunks
                .into_remainder()
                .iter_mut()
                .zip(s1_chunks.into_remainder().iter_mut())
                .zip(s2_chunks.into_remainder().iter_mut())
            {
                let src0 = *y0;
                let src1 = *y1;
                let src2 = *y2;
                *y0 = src0 + mct::ICT_INV_R_CR * src2 + addends[0];
                *y1 = src0 + mct::ICT_INV_G_CB * src1 + mct::ICT_INV_G_CR * src2 + addends[1];
                *y2 = src0 + mct::ICT_INV_B_CB * src1 + addends[2];
            }
        }
        // Reversible MCT, specified in G.2.
        WaveletTransform::Reversible53 => {
            let quarter = f32x8::splat(simd, mct::RCT_QUARTER);
            let red_level = f32x8::splat(simd, addends[0]);
            let green_level = f32x8::splat(simd, addends[1]);
            let blue_level = f32x8::splat(simd, addends[2]);
            let mut s0_chunks = s0.chunks_exact_mut(8);
            let mut s1_chunks = s1.chunks_exact_mut(8);
            let mut s2_chunks = s2.chunks_exact_mut(8);
            for ((y0, y1), y2) in s0_chunks
                .by_ref()
                .zip(s1_chunks.by_ref())
                .zip(s2_chunks.by_ref())
            {
                let y_0 = f32x8::from_slice(simd, y0);
                let y_1 = f32x8::from_slice(simd, y1);
                let y_2 = f32x8::from_slice(simd, y2);

                let i1 = y_0 - ((y_2 + y_1) * quarter).floor();
                let i0 = y_2 + i1 + red_level;
                let i2 = y_1 + i1 + blue_level;

                i0.store(y0);
                (i1 + green_level).store(y1);
                i2.store(y2);
            }
            for ((y0, y1), y2) in s0_chunks
                .into_remainder()
                .iter_mut()
                .zip(s1_chunks.into_remainder().iter_mut())
                .zip(s2_chunks.into_remainder().iter_mut())
            {
                let src0 = *y0;
                let src1 = *y1;
                let src2 = *y2;
                let i1 = src0 - floor_f32((src2 + src1) * mct::RCT_QUARTER);
                *y0 = src2 + i1 + addends[0];
                *y1 = i1 + addends[1];
                *y2 = src1 + i1 + addends[2];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    #[test]
    fn inverse_mct_applies_mixed_addends_to_simd_chunks_and_scalar_tail() {
        let source0 = [-15.0, -8.0, -1.0, 0.0, 1.0, 7.0, 13.0, 21.0, 34.0];
        let source1 = [9.0, -7.0, 5.0, -3.0, 1.0, 2.0, -4.0, 6.0, -8.0];
        let source2 = [-6.0, 4.0, -2.0, 0.0, 2.0, -4.0, 6.0, -8.0, 10.0];
        let addends = [128.0, 0.0, 2048.0];

        for transform in [
            WaveletTransform::Reversible53,
            WaveletTransform::Irreversible97,
        ] {
            let mut plane0 = source0;
            let mut plane1 = source1;
            let mut plane2 = source2;
            apply_inner(transform, &mut plane0, &mut plane1, &mut plane2, addends);

            for index in 0..source0.len() {
                let (expected0, expected1, expected2) = match transform {
                    WaveletTransform::Reversible53 => {
                        let green = source0[index]
                            - floor_f32((source2[index] + source1[index]) * mct::RCT_QUARTER);
                        (source2[index] + green, green, source1[index] + green)
                    }
                    WaveletTransform::Irreversible97 => (
                        source0[index] + mct::ICT_INV_R_CR * source2[index],
                        source0[index]
                            + mct::ICT_INV_G_CB * source1[index]
                            + mct::ICT_INV_G_CR * source2[index],
                        source0[index] + mct::ICT_INV_B_CB * source1[index],
                    ),
                };
                let expected = [
                    expected0 + addends[0],
                    expected1 + addends[1],
                    expected2 + addends[2],
                ];
                let actual = [plane0[index], plane1[index], plane2[index]];
                if transform == WaveletTransform::Reversible53 {
                    assert_eq!(actual.map(f32::to_bits), expected.map(f32::to_bits));
                } else {
                    for (actual, expected) in actual.into_iter().zip(expected) {
                        assert!((actual - expected).abs() <= 0.000_25);
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "performance guard harness; run explicitly with --ignored --nocapture"]
    fn inverse_mct_shift_fusion_perf_guard() {
        const LEN: usize = 512 * 512;
        const SAMPLES: usize = 21;
        let source0 = (0..LEN)
            .map(|index| {
                f32::from(u16::try_from(index % 251).expect("bounded test sample")) - 125.0
            })
            .collect::<Vec<_>>();
        let source1 = (0..LEN)
            .map(|index| f32::from(u16::try_from(index % 127).expect("bounded test sample")) - 63.0)
            .collect::<Vec<_>>();
        let source2 = (0..LEN)
            .map(|index| f32::from(u16::try_from(index % 61).expect("bounded test sample")) - 30.0)
            .collect::<Vec<_>>();
        let mut plane0 = source0.clone();
        let mut plane1 = source1.clone();
        let mut plane2 = source2.clone();
        let addends = [128.0, 128.0, 128.0];
        let mut fused = Vec::with_capacity(SAMPLES);
        let mut separate = Vec::with_capacity(SAMPLES);

        for _ in 0..SAMPLES {
            plane0.copy_from_slice(&source0);
            plane1.copy_from_slice(&source1);
            plane2.copy_from_slice(&source2);
            let started = Instant::now();
            apply_inner(
                WaveletTransform::Irreversible97,
                &mut plane0,
                &mut plane1,
                &mut plane2,
                addends,
            );
            std::hint::black_box((&plane0, &plane1, &plane2));
            fused.push(started.elapsed());

            plane0.copy_from_slice(&source0);
            plane1.copy_from_slice(&source1);
            plane2.copy_from_slice(&source2);
            let started = Instant::now();
            apply_inner(
                WaveletTransform::Irreversible97,
                &mut plane0,
                &mut plane1,
                &mut plane2,
                [0.0; 3],
            );
            for plane in [&mut plane0, &mut plane1, &mut plane2] {
                for sample in plane {
                    *sample += 128.0;
                }
            }
            std::hint::black_box((&plane0, &plane1, &plane2));
            separate.push(started.elapsed());
        }

        let fused = median(fused);
        let separate = median(separate);
        eprintln!(
            "j2k_native_inverse_mct_shift_perf len={LEN} fused_us={} separate_us={}",
            fused.as_micros(),
            separate.as_micros()
        );
        assert!(
            fused.as_nanos().saturating_mul(100) <= separate.as_nanos().saturating_mul(95),
            "fused inverse MCT/sign shift must improve its targeted median by at least 5%"
        );
    }

    fn median(mut samples: Vec<Duration>) -> Duration {
        samples.sort_unstable();
        samples[samples.len() / 2]
    }
}
