//! The irreversible multi-component transformation, as specified in
//! Annex G.2 and G.3.

use super::codestream::{ComponentInfo, Header, WaveletTransform};
use super::decode::TileDecodeContext;
use super::idwt::{pack_last_level, LastLevel};
use crate::error::{bail, err, ColorError, DecodingError, Result};
use crate::math::{dispatch, f32x8, floor_f32, mul_add, round_ties_even_then_add, Level, Simd};
use crate::{HtCodeBlockDecoder, J2kInverseMctJob, J2kWaveletTransform};
use j2k_codec_math::mct;

/// Apply the inverse multi-component transform, as specified in G.2 and G.3.
/// `parallel` lets the CPU transform run on the Rayon pool.
pub(crate) fn apply_inverse(
    tile_ctx: &mut TileDecodeContext,
    component_infos: &[super::codestream::ComponentInfo],
    header: &Header<'_>,
    backend: &mut Option<&mut dyn HtCodeBlockDecoder>,
    parallel: bool,
) -> Result<()> {
    if tile_ctx.channel_data.len() < 3 {
        return if header.strict {
            err!(ColorError::Mct)
        } else {
            Ok(())
        };
    }

    let round_irreversible_output = tile_ctx.round_irreversible_output;
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
    let centered_output =
        round_irreversible_output && transform == WaveletTransform::Irreversible97;
    let transform_addends = if centered_output { [0.0; 3] } else { addends };

    if tile_ctx.defer_mct
        && backend.is_none()
        && s0.integer_container.is_none()
        && s1.integer_container.is_none()
        && s2.integer_container.is_none()
    {
        tile_ctx.deferred_mct = Some(DeferredMct {
            transform,
            addends: transform_addends,
            rounding: centered_output.then_some(addends),
            last_level: tile_ctx.fused_last_level.take(),
        });
        return Ok(());
    }

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

    let handled = if centered_output {
        false
    } else if let Some(backend) = backend.as_deref_mut() {
        backend.decode_inverse_mct(J2kInverseMctJob {
            transform: J2kWaveletTransform::from(transform),
            plane0: &mut s0.container,
            plane1: &mut s1.container,
            plane2: &mut s2.container,
            addend0: transform_addends[0],
            addend1: transform_addends[1],
            addend2: transform_addends[2],
        })?
    } else {
        false
    };

    let planes = (
        &mut s0.container[..],
        &mut s1.container[..],
        &mut s2.container[..],
    );
    // Centered output never goes to the backend, so its rounding always runs
    // with the CPU transform.
    if !handled {
        let rounding = centered_output.then_some(addends);
        transform_planes(transform, planes, transform_addends, rounding, parallel);
    }

    Ok(())
}

/// Samples per plane in one chunk: three planes' worth stays in cache
/// between the transform and the rounding.
const MCT_CHUNK: usize = 1 << 14;

/// Samples per plane from which the chunks run on the pool. Below it, as for
/// most region decodes, fanning the chunks out costs more than it saves once
/// other decodes share the machine.
#[cfg(any(feature = "parallel", test))]
const MCT_POOL_SAMPLES: usize = 1 << 22;

/// The CPU inverse transform of the first three planes that
/// [`apply_inverse`] left to the RGB8 pack of a plain colour image.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DeferredMct {
    transform: WaveletTransform,
    addends: [f32; 3],
    rounding: Option<[f32; 3]>,
    /// The last IDWT level the decode also left to the pack: each plane
    /// holds its sub-bands rather than samples.
    last_level: Option<LastLevel>,
}

/// Samples per plane in one fused pack chunk: the three planes' scratch
/// stays small on the stack and in L1, and chunks are multiples of the SIMD
/// width (see [`transform_planes`]).
const PACK_CHUNK: usize = 1 << 10;

/// Fewest chunks per fused pack task.
#[cfg(feature = "parallel")]
const PACK_TASK_CHUNKS: usize = 64;

impl DeferredMct {
    /// Transform the planes in place, as [`apply_inverse`] would have. Planes
    /// that hold a last level's sub-bands are left to [`Self::pack_rgb8`].
    pub(crate) fn apply(
        self,
        planes: (&mut [f32], &mut [f32], &mut [f32]),
        parallel: bool,
    ) -> Result<()> {
        if self.last_level.is_some() {
            bail!(DecodingError::IncompletePlaneStore);
        }
        transform_planes(
            self.transform,
            planes,
            self.addends,
            self.rounding,
            parallel,
        );
        Ok(())
    }

    /// Transform the planes chunk by chunk and pack each chunk as RGB8 with
    /// `pack` while it is in cache, where [`Self::apply`] transforms the
    /// whole planes and the pack reads them again. On the pool each chunk is
    /// transformed in scratch, so the planes are not written back: the
    /// workers share memory bandwidth. One thread transforms in place, which
    /// costs less than the copy. With a last level, the planes hold its
    /// high bands and the next component's LL band, with `first_ll` kept
    /// separately. Each reconstructed row is transformed and packed. On
    /// hosted ARM64 with SIMD, 9/7 rows take the fused ICT pack instead of
    /// `pack`, so `pack` must be the standard RGB8 pack there.
    pub(crate) fn pack_rgb8(
        self,
        [r, g, b]: [&mut [f32]; 3],
        first_ll: &[f32],
        scratch: [&mut [f32]; 2],
        output: &mut [u8],
        parallel: bool,
        pack: impl Fn([&[f32]; 3], &mut [u8]) + Sync,
    ) -> Result<()> {
        if let Some(level) = self.last_level {
            let planes = [&*r, &*g, &*b];
            return pack_last_level(
                level,
                planes,
                first_ll,
                scratch,
                output,
                parallel,
                |[r, g, b], output| {
                    if cfg!(all(
                        feature = "std",
                        feature = "simd",
                        target_arch = "aarch64"
                    )) && self.transform == WaveletTransform::Irreversible97
                    {
                        if self.rounding.is_some() {
                            self.pack_ict_rgb8::<true>([r, g, b], output);
                        } else {
                            self.pack_ict_rgb8::<false>([r, g, b], output);
                        }
                    } else {
                        self.transform_chunk((&mut *r, &mut *g, &mut *b));
                        pack([r, g, b], output);
                    }
                },
            );
        }
        #[cfg(feature = "parallel")]
        if parallel && r.len() > PACK_CHUNK && rayon::current_num_threads() > 1 {
            use rayon::prelude::*;
            type Scratch = [[f32; PACK_CHUNK]; 3];
            let run = |scratch: &mut Scratch,
                       output: &mut [u8],
                       ((r, g), b): ((&[f32], &[f32]), &[f32])| {
                let [s0, s1, s2] = scratch;
                let (s0, s1, s2) = (&mut s0[..r.len()], &mut s1[..g.len()], &mut s2[..b.len()]);
                s0.copy_from_slice(r);
                s1.copy_from_slice(g);
                s2.copy_from_slice(b);
                self.transform_chunk((s0, s1, s2));
                pack([s0, s1, s2], output);
            };
            output
                .par_chunks_mut(3 * PACK_CHUNK)
                .zip(r.par_chunks(PACK_CHUNK).zip(g.par_chunks(PACK_CHUNK)))
                .zip(b.par_chunks(PACK_CHUNK))
                // Each task zeroes one scratch; long tasks keep that rare.
                .with_min_len(PACK_TASK_CHUNKS)
                .for_each_init(
                    || [[0.0; PACK_CHUNK]; 3],
                    |scratch, ((output, rg), b)| run(scratch, output, (rg, b)),
                );
            return Ok(());
        }
        let _ = parallel;
        let chunks = r.chunks_mut(PACK_CHUNK).zip(g.chunks_mut(PACK_CHUNK));
        let outputs = output.chunks_mut(3 * PACK_CHUNK);
        for (output, ((s0, s1), s2)) in outputs.zip(chunks.zip(b.chunks_mut(PACK_CHUNK))) {
            self.transform_chunk((s0, s1, s2));
            pack([s0, s1, s2], output);
        }
        Ok(())
    }

    /// Retain the transform's FMA and rounding boundaries while writing RGB8
    /// directly. A separate indexed loop lets LLVM combine the conversion and
    /// interleaved stores without the streaming DWT's surrounding control flow.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "RGB8 samples use the same rounding and saturating cast as the standard pack"
    )]
    #[inline(never)]
    fn pack_ict_rgb8<const CENTERED: bool>(self, [luma, cb, cr]: [&[f32]; 3], output: &mut [u8]) {
        let pixels = output.as_chunks_mut::<3>().0;
        let len = pixels.len().min(luma.len()).min(cb.len()).min(cr.len());
        let shift = self.rounding.unwrap_or([0.0; 3]);
        for index in 0..len {
            let mut red = mul_add(cr[index], mct::ICT_INV_R_CR, luma[index]) + self.addends[0];
            let mut green = mul_add(
                cr[index],
                mct::ICT_INV_G_CR,
                mul_add(cb[index], mct::ICT_INV_G_CB, luma[index]),
            ) + self.addends[1];
            let mut blue = mul_add(cb[index], mct::ICT_INV_B_CB, luma[index]) + self.addends[2];
            if CENTERED {
                red = round_ties_even_then_add(red, shift[0]);
                green = round_ties_even_then_add(green, shift[1]);
                blue = round_ties_even_then_add(blue, shift[2]);
            }
            pixels[index] = [
                crate::math::round_f32(red) as u8,
                crate::math::round_f32(green) as u8,
                crate::math::round_f32(blue) as u8,
            ];
        }
    }

    fn transform_chunk(self, planes: (&mut [f32], &mut [f32], &mut [f32])) {
        transform_planes(self.transform, planes, self.addends, self.rounding, false);
    }
}

/// The CPU inverse transform, then (for centered 9/7 output) the rounding
/// and level shift, chunk by chunk so each chunk makes one trip to memory.
/// Chunks are multiples of the SIMD width, so every sample sees the same
/// operations as a whole-plane pass. `parallel` runs the chunks on the pool.
fn transform_planes(
    transform: WaveletTransform,
    (s0, s1, s2): (&mut [f32], &mut [f32], &mut [f32]),
    addends: [f32; 3],
    rounding: Option<[f32; 3]>,
    parallel: bool,
) {
    let run = |s0: &mut [f32], s1: &mut [f32], s2: &mut [f32]| {
        apply_inner(transform, s0, s1, s2, addends);
        if let Some(shift) = rounding {
            round_centered_and_shift(s0, shift[0]);
            round_centered_and_shift(s1, shift[1]);
            round_centered_and_shift(s2, shift[2]);
        }
    };
    #[cfg(feature = "parallel")]
    if parallel && s0.len() >= MCT_POOL_SAMPLES && rayon::current_num_threads() > 1 {
        use rayon::prelude::*;
        s0.par_chunks_mut(MCT_CHUNK)
            .zip(s1.par_chunks_mut(MCT_CHUNK))
            .zip(s2.par_chunks_mut(MCT_CHUNK))
            .for_each(|((s0, s1), s2)| run(s0, s1, s2));
        return;
    }
    let _ = parallel;
    let chunks = s0.chunks_mut(MCT_CHUNK).zip(s1.chunks_mut(MCT_CHUNK));
    for ((s0, s1), s2) in chunks.zip(s2.chunks_mut(MCT_CHUNK)) {
        run(s0, s1, s2);
    }
}

fn round_centered_and_shift(samples: &mut [f32], addend: f32) {
    for sample in samples {
        *sample = round_ties_even_then_add(*sample, addend);
    }
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

pub(crate) fn apply_inner(
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
            let (s0_chunks, s0_rest) = s0.as_chunks_mut::<8>();
            let (s1_chunks, s1_rest) = s1.as_chunks_mut::<8>();
            let (s2_chunks, s2_rest) = s2.as_chunks_mut::<8>();
            for ((y0, y1), y2) in s0_chunks.iter_mut().zip(s1_chunks).zip(s2_chunks) {
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
            for ((y0, y1), y2) in s0_rest.iter_mut().zip(s1_rest).zip(s2_rest) {
                let src0 = *y0;
                let src1 = *y1;
                let src2 = *y2;
                // Same fused expressions as the SIMD body above and Metal.
                *y0 = mul_add(src2, mct::ICT_INV_R_CR, src0) + addends[0];
                *y1 = mul_add(
                    src2,
                    mct::ICT_INV_G_CR,
                    mul_add(src1, mct::ICT_INV_G_CB, src0),
                ) + addends[1];
                *y2 = mul_add(src1, mct::ICT_INV_B_CB, src0) + addends[2];
            }
        }
        // Reversible MCT, specified in G.2.
        WaveletTransform::Reversible53 => {
            let quarter = f32x8::splat(simd, mct::RCT_QUARTER);
            let red_level = f32x8::splat(simd, addends[0]);
            let green_level = f32x8::splat(simd, addends[1]);
            let blue_level = f32x8::splat(simd, addends[2]);
            let (s0_chunks, s0_rest) = s0.as_chunks_mut::<8>();
            let (s1_chunks, s1_rest) = s1.as_chunks_mut::<8>();
            let (s2_chunks, s2_rest) = s2.as_chunks_mut::<8>();
            for ((y0, y1), y2) in s0_chunks.iter_mut().zip(s1_chunks).zip(s2_chunks) {
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
            for ((y0, y1), y2) in s0_rest.iter_mut().zip(s1_rest).zip(s2_rest) {
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
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the test pack rounds and saturates to 8 bits like the RGB8 output"
    )]
    fn chunked_inverse_mct_matches_whole_plane_passes() {
        // Serial cases: planes a few chunks long with a ragged tail. Pool
        // cases: planes past the pool floor, in a two-thread pool so the pool
        // path runs on any machine.
        let small = 2 * MCT_CHUNK + 13;
        let large = MCT_POOL_SAMPLES + 13;
        let plane = |seed: u32, len: usize| {
            (0..len)
                .map(|index| {
                    let index = u32::try_from(index).expect("index fits u32");
                    f32::from(u16::try_from((index * seed) % 4093).expect("u16")) / 8.0 - 255.0
                })
                .collect::<Vec<_>>()
        };
        let planes = |len| [plane(7, len), plane(13, len), plane(29, len)];
        let addends = [128.0, 0.0, 2048.0];
        for transform in [
            WaveletTransform::Reversible53,
            WaveletTransform::Irreversible97,
        ] {
            for (rounding, parallel, len) in [
                (None, false, small),
                (None, true, large),
                (Some(addends), false, small),
                (Some(addends), true, large),
            ] {
                let mut expected = planes(len);
                let [e0, e1, e2] = &mut expected;
                apply_inner(transform, e0, e1, e2, addends);
                if let Some(shift) = rounding {
                    for (plane, shift) in [e0, e1, e2].into_iter().zip(shift) {
                        round_centered_and_shift(plane, shift);
                    }
                }
                let mut actual = planes(len);
                let [a0, a1, a2] = &mut actual;
                let mut run =
                    || transform_planes(transform, (a0, a1, a2), addends, rounding, parallel);
                #[cfg(feature = "parallel")]
                if parallel {
                    rayon::ThreadPoolBuilder::new()
                        .num_threads(2)
                        .build()
                        .expect("two-thread pool")
                        .install(run);
                } else {
                    run();
                }
                #[cfg(not(feature = "parallel"))]
                run();
                assert!(
                    actual
                        .iter()
                        .flatten()
                        .map(|v| v.to_bits())
                        .eq(expected.iter().flatten().map(|v| v.to_bits())),
                    "{transform:?} {rounding:?} {parallel}"
                );

                // The pack that fuses the deferred transform writes the same bytes.
                let pack = |[r, g, b]: [&[f32]; 3], output: &mut [u8]| {
                    let samples = r.iter().zip(g).zip(b);
                    for (pixel, ((r, g), b)) in
                        output.as_chunks_mut::<3>().0.iter_mut().zip(samples)
                    {
                        pixel.copy_from_slice(&[*r, *g, *b].map(|v| v.round() as u8));
                    }
                };
                let mut packed = vec![0_u8; 3 * len];
                pack([&expected[0], &expected[1], &expected[2]], &mut packed);
                let [mut s0, mut s1, mut s2] = planes(len);
                let mut fused = vec![0_u8; 3 * len];
                DeferredMct {
                    transform,
                    addends,
                    rounding,
                    last_level: None,
                }
                .pack_rgb8(
                    [&mut s0, &mut s1, &mut s2],
                    &[],
                    [&mut [], &mut []],
                    &mut fused,
                    parallel,
                    pack,
                )
                .expect("fused pack");
                assert_eq!(fused, packed, "fused {transform:?} {rounding:?} {parallel}");
            }
        }
    }

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
    fn irreversible_scalar_tail_matches_simd_body_bits() {
        // These inputs round differently fused and unfused, so a tail computed
        // without FMA disagrees with the SIMD body and with Metal.
        let y0 = f32::from_bits(0xc23e_1ece);
        let y1 = f32::from_bits(0x4126_4c1f);
        let y2 = f32::from_bits(0xc129_2742);
        let expected = [
            crate::math::mul_add(y2, mct::ICT_INV_R_CR, y0),
            crate::math::mul_add(
                y2,
                mct::ICT_INV_G_CR,
                crate::math::mul_add(y1, mct::ICT_INV_G_CB, y0),
            ),
            crate::math::mul_add(y1, mct::ICT_INV_B_CB, y0),
        ];
        assert_ne!(
            expected[0].to_bits(),
            (y0 + mct::ICT_INV_R_CR * y2).to_bits(),
            "fixture must separate fused from unfused rounding"
        );

        let mut plane0 = [y0; 9];
        let mut plane1 = [y1; 9];
        let mut plane2 = [y2; 9];
        apply_inner(
            WaveletTransform::Irreversible97,
            &mut plane0,
            &mut plane1,
            &mut plane2,
            [0.0; 3],
        );

        for index in 0..plane0.len() {
            assert_eq!(
                [plane0[index], plane1[index], plane2[index]].map(f32::to_bits),
                expected.map(f32::to_bits),
                "sample {index}"
            );
        }

        // Move the FMA-sensitive values to a rounding tie. Unfused arithmetic
        // crosses this tie in either the direct or centered conversion.
        let deferred = DeferredMct {
            transform: WaveletTransform::Irreversible97,
            addends: expected.map(|sample| 0.5 - sample),
            rounding: None,
            last_level: None,
        };
        let mut packed = [0_u8; 27];
        deferred.pack_ict_rgb8::<false>([&[y0; 9], &[y1; 9], &[y2; 9]], &mut packed);
        assert_eq!(packed, [1; 27]);
        let centered = DeferredMct {
            rounding: Some([128.0; 3]),
            ..deferred
        };
        centered.pack_ict_rgb8::<true>([&[y0; 9], &[y1; 9], &[y2; 9]], &mut packed);
        assert_eq!(packed, [128; 27]);
    }

    #[test]
    fn irreversible_output_rounds_before_unsigned_level_shift() {
        let mut samples = [f32::from_bits(0xc117_fffc), -9.5, -8.5];
        round_centered_and_shift(&mut samples, 128.0);
        assert_eq!(
            samples.map(f32::to_bits),
            [119.0, 118.0, 120.0].map(f32::to_bits)
        );

        let deferred = DeferredMct {
            transform: WaveletTransform::Irreversible97,
            addends: [0.0; 3],
            rounding: Some([128.0; 3]),
            last_level: None,
        };
        let mut packed = [0_u8; 27];
        deferred.pack_ict_rgb8::<true>(
            [&[f32::from_bits(0xc117_fffc); 9], &[0.0; 9], &[0.0; 9]],
            &mut packed,
        );
        assert_eq!(packed, [119; 27]);
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
