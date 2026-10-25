// SPDX-License-Identifier: MIT OR Apache-2.0

use super::allocation::{checked_add_bytes, checked_element_bytes, host_allocation_failed};
use super::code_block_metadata::validate_accelerated_ht_code_block;
use super::tier1_allocation::{
    prepared_subbands_ownership, public_ht_blocks_ownership, Tier1PhaseTracker,
};
#[cfg(test)]
use super::NativeEncodeRetainedInput;
use super::{
    ht_block_encode, quantize, BlockCodingMode, ComponentRoiEncodeRegion,
    J2kEncodeStageAccelerator, J2kHtSubbandEncodeJob, J2kQuantizeSubbandJob,
    NativeEncodePipelineError, NativeEncodePipelineResult, NativeEncodeSession,
    PreparedEncodeCodeBlock, PreparedEncodeSubband, QuantStepSize, SubBandType, Vec,
};
use crate::j2c::fdwt::PackedSubbandView;

fn apply_roi_maxshift_encode_for_session(
    coefficients: &mut [i32],
    request: &F32SubbandEncodeRequest<'_, '_>,
    quantized_bytes: usize,
    tracker: &mut Tier1PhaseTracker<'_, '_>,
) -> NativeEncodePipelineResult<()> {
    if request.roi_shift == 0 {
        return Ok(());
    }
    let expected_len = (request.width as usize)
        .checked_mul(request.height as usize)
        .ok_or_else(|| NativeEncodePipelineError::arithmetic_overflow("ROI subband dimensions"))?;
    if coefficients.len() != expected_len {
        return Err(NativeEncodePipelineError::internal_invariant(
            "ROI subband coefficient length mismatch",
        ));
    }
    if request.roi_regions.is_empty() {
        for coefficient in coefficients {
            shift_roi_coefficient(coefficient, request.roi_shift)
                .map_err(NativeEncodePipelineError::arithmetic_overflow)?;
        }
        return Ok(());
    }

    let (mut selected, _) = tracker.try_vec::<u8>(
        coefficients.len(),
        [quantized_bytes],
        "ROI subband selection flags",
    )?;
    selected.resize(coefficients.len(), 0);
    for region in request.roi_regions {
        let Some((x0, y0, x1, y1)) =
            roi_region_subband_window(*region, request.width, request.height, request.roi_scale)
        else {
            continue;
        };
        for y in y0..y1 {
            for x in x0..x1 {
                let idx = (y as usize)
                    .checked_mul(request.width as usize)
                    .and_then(|row| row.checked_add(x as usize))
                    .ok_or_else(|| {
                        NativeEncodePipelineError::arithmetic_overflow("ROI subband index")
                    })?;
                if selected[idx] != 0 {
                    continue;
                }
                selected[idx] = 1;
                shift_roi_coefficient(&mut coefficients[idx], request.roi_shift)
                    .map_err(NativeEncodePipelineError::arithmetic_overflow)?;
            }
        }
    }
    Ok(())
}

fn shift_roi_coefficient(coefficient: &mut i32, roi_shift: u8) -> Result<(), &'static str> {
    *coefficient = coefficient
        .checked_shl(u32::from(roi_shift))
        .ok_or("ROI maxshift coefficient overflow")?;
    Ok(())
}

pub(super) fn shift_roi_coefficient_i64(
    coefficient: &mut i64,
    roi_shift: u8,
) -> Result<(), &'static str> {
    let factor = 1_i64
        .checked_shl(u32::from(roi_shift))
        .ok_or("ROI maxshift coefficient overflow")?;
    *coefficient = coefficient
        .checked_mul(factor)
        .ok_or("ROI maxshift coefficient overflow")?;
    Ok(())
}

pub(super) fn roi_region_subband_window(
    region: ComponentRoiEncodeRegion,
    width: u32,
    height: u32,
    roi_scale: u32,
) -> Option<(u32, u32, u32, u32)> {
    if width == 0 || height == 0 || roi_scale == 0 {
        return None;
    }
    let x1 = region.x.saturating_add(region.width);
    let y1 = region.y.saturating_add(region.height);
    let x0 = (region.x / roi_scale).min(width);
    let y0 = (region.y / roi_scale).min(height);
    let x1 = x1.div_ceil(roi_scale).min(width);
    let y1 = y1.div_ceil(roi_scale).min(height);
    if x0 >= x1 || y0 >= y1 {
        None
    } else {
        Some((x0, y0, x1, y1))
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "this codec boundary keeps geometry, state buffers, and validated options explicit without allocation or indirection"
)]
#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "the stable codec boundary borrows shared Copy metadata used across nested calls"
)]
#[cfg(test)]
pub(super) fn prepare_subband(
    coefficients: &[f32],
    width: u32,
    height: u32,
    step_size: &QuantStepSize,
    bit_depth: u8,
    guard_bits: u8,
    reversible: bool,
    block_coding_mode: BlockCodingMode,
    cb_width: u32,
    cb_height: u32,
    sub_band_type: SubBandType,
    roi_shift: u8,
    roi_regions: &[ComponentRoiEncodeRegion],
    roi_scale: u32,
    ht_target_coding_passes: u8,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> crate::EncodeResult<PreparedEncodeSubband> {
    let session = NativeEncodeSession::try_new(NativeEncodeRetainedInput::none())?;
    let request = F32SubbandEncodeRequest {
        origin: (0, 0),
        coefficients: SubbandSamples::Contiguous(coefficients),
        width,
        height,
        step_size,
        bit_depth,
        guard_bits,
        reversible,
        block_coding_mode,
        cb_width,
        cb_height,
        sub_band_type,
        roi_shift,
        roi_regions,
        roi_scale,
        ht_target_coding_passes,
        single_layer: false,
        session: &session,
        retained_base_bytes: 0,
    };
    prepare_subband_for_session(&request, accelerator)
        .map_err(NativeEncodePipelineError::into_encode_error)
}

/// A subband's transform coefficients.
#[derive(Clone, Copy)]
pub(super) enum SubbandSamples<'a> {
    /// Row-major samples, as the stage hooks take them.
    Contiguous(&'a [f32]),
    /// Rows of a packed transform plane. Only an accelerator that declines
    /// every stage gets these: each code block quantizes from the rows.
    Packed(PackedSubbandView<'a, f32>),
}

pub(super) struct F32SubbandEncodeRequest<'a, 'input> {
    pub(super) origin: (u32, u32),
    pub(super) coefficients: SubbandSamples<'a>,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) step_size: &'a QuantStepSize,
    pub(super) bit_depth: u8,
    pub(super) guard_bits: u8,
    pub(super) reversible: bool,
    pub(super) block_coding_mode: BlockCodingMode,
    pub(super) cb_width: u32,
    pub(super) cb_height: u32,
    pub(super) sub_band_type: SubBandType,
    pub(super) roi_shift: u8,
    pub(super) roi_regions: &'a [ComponentRoiEncodeRegion],
    pub(super) roi_scale: u32,
    pub(super) ht_target_coding_passes: u8,
    /// Whether the codestream has one quality layer, so no rate control
    /// reselects the code-block passes.
    pub(super) single_layer: bool,
    pub(super) session: &'a NativeEncodeSession<'input>,
    pub(super) retained_base_bytes: usize,
}

pub(super) fn prepare_subband_for_session(
    request: &F32SubbandEncodeRequest<'_, '_>,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> NativeEncodePipelineResult<PreparedEncodeSubband> {
    let mut tracker = Tier1PhaseTracker::new(request.session, request.retained_base_bytes);
    if request.width == 0 || request.height == 0 {
        return Ok(empty_prepared_subband(request));
    }

    let plan = F32SubbandPlan::try_new(request)?;
    let parallel = accelerator.prefer_parallel_cpu_code_block_fallback();
    let coefficients = match request.coefficients {
        SubbandSamples::Contiguous(coefficients) => coefficients,
        SubbandSamples::Packed(view)
            if accelerator.declines_every_stage() && request.roi_shift == 0 =>
        {
            let quantizer =
                quantize::Quantizer::new(*request.step_size, plan.range_bits, request.reversible);
            if request.single_layer
                && request.block_coding_mode == BlockCodingMode::HighThroughput
                && request.ht_target_coding_passes == 1
            {
                if let Some(prepared) = try_prepare_encoded_ht_blocks(
                    request,
                    plan,
                    view,
                    quantizer,
                    &mut tracker,
                    parallel,
                )? {
                    return Ok(prepared);
                }
            }
            return prepare_code_blocks(
                request,
                plan,
                0,
                &mut tracker,
                parallel,
                |x0, y0, width, height, out| {
                    for y in y0..y0 + height {
                        let row = view.row(y).ok_or_else(|| {
                            NativeEncodePipelineError::internal_invariant(
                                "packed subband row is missing",
                            )
                        })?;
                        quantizer.extend(out, &row[x0 as usize..][..width as usize]);
                    }
                    Ok(())
                },
            );
        }
        SubbandSamples::Packed(_) => {
            return Err(NativeEncodePipelineError::internal_invariant(
                "packed subband samples need an accelerator that declines every stage",
            ));
        }
    };
    if let Some(prepared) =
        try_prepare_fused_ht_subband(request, coefficients, plan, &mut tracker, accelerator)?
    {
        return Ok(prepared);
    }

    let (mut quantized, quantized_bytes) =
        try_quantize_subband(request, coefficients, plan, &mut tracker, accelerator)?;
    apply_roi_maxshift_encode_for_session(&mut quantized, request, quantized_bytes, &mut tracker)?;
    prepare_code_blocks(
        request,
        plan,
        quantized_bytes,
        &mut tracker,
        parallel,
        |x0, y0, width, height, out| {
            copy_code_block_coefficients_into(
                &quantized,
                request.width as usize,
                x0 as usize,
                y0 as usize,
                width as usize,
                height as usize,
                out,
            )
        },
    )
}

#[derive(Clone, Copy)]
struct F32SubbandPlan {
    range_bits: u8,
    total_bitplanes: u8,
    num_cbs_x: u32,
    num_cbs_y: u32,
}

impl F32SubbandPlan {
    fn try_new(request: &F32SubbandEncodeRequest<'_, '_>) -> NativeEncodePipelineResult<Self> {
        let exponent = u8::try_from(request.step_size.exponent).map_err(|_| {
            NativeEncodePipelineError::internal_invariant(
                "quantization exponent exceeds supported range",
            )
        })?;
        let base_total_bitplanes = request
            .guard_bits
            .saturating_add(exponent)
            .saturating_sub(1);
        let total_bitplanes = base_total_bitplanes
            .checked_add(request.roi_shift)
            .ok_or_else(|| {
                NativeEncodePipelineError::unsupported(
                    "ROI maxshift exceeds supported coded bitplane count",
                )
            })?;
        Ok(Self {
            range_bits: subband_range_bits(request.bit_depth, request.sub_band_type),
            total_bitplanes,
            num_cbs_x: super::tile_grid::block_count(
                request.origin.0,
                request.width,
                request.cb_width,
            )?,
            num_cbs_y: super::tile_grid::block_count(
                request.origin.1,
                request.height,
                request.cb_height,
            )?,
        })
    }
}

fn empty_prepared_subband(request: &F32SubbandEncodeRequest<'_, '_>) -> PreparedEncodeSubband {
    PreparedEncodeSubband {
        origin: request.origin,
        code_blocks: Vec::new(),
        preencoded_ht_code_blocks: None,
        preencoded_ht_maximum_cleanup_magnitude: None,
        num_cbs_x: 0,
        num_cbs_y: 0,
        code_block_width: request.cb_width,
        code_block_height: request.cb_height,
        width: request.width,
        height: request.height,
        sub_band_type: request.sub_band_type,
        total_bitplanes: 0,
        block_coding_mode: request.block_coding_mode,
        ht_target_coding_passes: request.ht_target_coding_passes,
    }
}

fn try_prepare_fused_ht_subband(
    request: &F32SubbandEncodeRequest<'_, '_>,
    coefficients: &[f32],
    plan: F32SubbandPlan,
    tracker: &mut Tier1PhaseTracker<'_, '_>,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> NativeEncodePipelineResult<Option<PreparedEncodeSubband>> {
    if request.block_coding_mode != BlockCodingMode::HighThroughput
        || request.roi_shift != 0
        || request.ht_target_coding_passes != 1
        || (!request.origin.0.is_multiple_of(request.cb_width)
            && request.width > request.cb_width - request.origin.0 % request.cb_width)
        || (!request.origin.1.is_multiple_of(request.cb_height)
            && request.height > request.cb_height - request.origin.1 % request.cb_height)
    {
        return Ok(None);
    }
    let encoded = accelerator
        .encode_ht_subband(J2kHtSubbandEncodeJob {
            coefficients,
            width: request.width,
            height: request.height,
            step_exponent: request.step_size.exponent,
            step_mantissa: request.step_size.mantissa,
            range_bits: plan.range_bits,
            reversible: request.reversible,
            code_block_width: request.cb_width,
            code_block_height: request.cb_height,
            total_bitplanes: plan.total_bitplanes,
        })
        .map_err(|source| crate::EncodeError::Accelerator {
            operation: "fused HT subband encode",
            source,
        })?;
    let Some(encoded) = encoded else {
        return Ok(None);
    };
    let maximum_cleanup_magnitude = accelerator.ht_subband_maximum_cleanup_magnitude();
    let expected_code_blocks = (plan.num_cbs_x as usize)
        .checked_mul(plan.num_cbs_y as usize)
        .ok_or_else(|| NativeEncodePipelineError::arithmetic_overflow("code-block count"))?;
    if encoded.len() != expected_code_blocks {
        return Err(crate::EncodeError::Accelerator {
            operation: "fused HT subband encode",
            source: crate::J2kEncodeStageError::internal_invariant(
                "accelerated HT subband code-block count mismatch",
            ),
        }
        .into());
    }
    for block in &encoded {
        validate_accelerated_ht_code_block(block, plan.total_bitplanes, 1).map_err(|detail| {
            crate::EncodeError::Accelerator {
                operation: "fused HT subband encode",
                source: crate::J2kEncodeStageError::internal_invariant(detail),
            }
        })?;
    }
    let encoded_bytes = public_ht_blocks_ownership(&encoded, encoded.capacity())?;
    tracker.check([encoded_bytes], "fused HT subband output")?;
    let code_blocks = code_block_shapes_for_session(
        request.width,
        request.height,
        request.cb_width,
        request.cb_height,
        encoded_bytes,
        tracker,
    )?;
    let prepared = prepared_subband(
        request,
        plan,
        code_blocks,
        Some(encoded),
        maximum_cleanup_magnitude,
    );
    let prepared_bytes =
        prepared_subbands_ownership(core::slice::from_ref(&prepared), 0)?.total()?;
    tracker.check([prepared_bytes], "prepared fused HT subband")?;
    Ok(Some(prepared))
}

fn try_quantize_subband(
    request: &F32SubbandEncodeRequest<'_, '_>,
    coefficients: &[f32],
    plan: F32SubbandPlan,
    tracker: &mut Tier1PhaseTracker<'_, '_>,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> NativeEncodePipelineResult<(Vec<i32>, usize)> {
    let requested_bytes =
        checked_element_bytes::<i32>(coefficients.len(), "quantized subband coefficients")?;
    tracker.check([requested_bytes], "quantized subband coefficients")?;
    let quantized = accelerator
        .encode_quantize_subband(J2kQuantizeSubbandJob {
            coefficients,
            step_exponent: request.step_size.exponent,
            step_mantissa: request.step_size.mantissa,
            range_bits: plan.range_bits,
            reversible: request.reversible,
        })
        .map_err(|source| crate::EncodeError::Accelerator {
            operation: "subband quantization",
            source,
        })?;
    let quantized = match quantized {
        Some(quantized) => {
            if quantized.len() != coefficients.len() {
                return Err(crate::EncodeError::Accelerator {
                    operation: "subband quantization",
                    source: crate::J2kEncodeStageError::internal_invariant(
                        "accelerated quantized subband length mismatch",
                    ),
                }
                .into());
            }
            quantized
        }
        None => quantize::try_quantize_subband(
            coefficients,
            request.step_size,
            plan.range_bits,
            request.reversible,
            accelerator.prefer_parallel_cpu_code_block_fallback(),
        )?,
    };
    let actual_bytes =
        checked_element_bytes::<i32>(quantized.capacity(), "quantized subband coefficients")?;
    tracker.check([actual_bytes], "quantized subband output")?;
    Ok((quantized, actual_bytes))
}

/// Waking the pool for a band smaller than this costs more than its few code
/// blocks take on one thread (two 64x64 blocks: about 30 us serial against
/// 35-80 us on ten M4 threads); four blocks already gain.
#[cfg(feature = "parallel")]
const MIN_PARALLEL_BAND_SAMPLES: usize = 4 * 64 * 64;

/// Cleanup-only HT blocks of a single-layer codestream quantize from the
/// packed band into a worker's buffer and are encoded at once, so their
/// coefficients never make a round trip through memory; Tier-1 then takes the
/// payloads as preencoded, with the exact magnitude bound. `None` when the
/// workers' scratch does not fit the cap, so success never depends on it.
/// Bands under `MIN_PARALLEL_BAND_SAMPLES` are coded on the calling thread.
fn try_prepare_encoded_ht_blocks(
    request: &F32SubbandEncodeRequest<'_, '_>,
    plan: F32SubbandPlan,
    view: PackedSubbandView<'_, f32>,
    quantizer: quantize::Quantizer,
    tracker: &mut Tier1PhaseTracker<'_, '_>,
    parallel: bool,
) -> NativeEncodePipelineResult<Option<PreparedEncodeSubband>> {
    let count = plan
        .num_cbs_x
        .checked_mul(plan.num_cbs_y)
        .and_then(|count| usize::try_from(count).ok())
        .ok_or_else(|| {
            NativeEncodePipelineError::arithmetic_overflow("prepared code-block count")
        })?;
    // As in the Tier-1 driver, a worker past the block count holds no scratch.
    #[cfg(feature = "parallel")]
    let parallel = parallel
        && (request.width as usize).saturating_mul(request.height as usize)
            >= MIN_PARALLEL_BAND_SAMPLES;
    #[cfg(feature = "parallel")]
    let workers = if parallel {
        rayon::current_num_threads().min(count)
    } else {
        1
    };
    #[cfg(not(feature = "parallel"))]
    let workers = 1_usize;
    let block_bytes = checked_element_bytes::<i32>(
        (request.cb_width as usize) * (request.cb_height as usize),
        "preencoded HT block buffer",
    )?;
    let scratch_bytes = workers
        .checked_mul(block_bytes + ht_block_encode::HtEncodeWorkspace::ALLOCATION_BYTES)
        .ok_or_else(|| NativeEncodePipelineError::arithmetic_overflow("preencoded HT scratch"))?;
    // As the Tier-1 driver does, charge every block's bounded output with the
    // workers' scratch and the result owners before a worker starts. A band
    // that does not fit takes the Tier-1 path, which codes it in waves.
    let output_bytes = preencoded_output_bound(request, plan, count)?;
    let owner_bytes = [
        checked_element_bytes::<Option<PreencodedHtBlock>>(count, "preencoded HT block results")?,
        checked_element_bytes::<crate::EncodedHtJ2kCodeBlock>(
            count,
            "preencoded HT block payloads",
        )?,
        checked_element_bytes::<PreparedEncodeCodeBlock>(count, "prepared code-block owners")?,
    ];
    match tracker.check(
        [scratch_bytes, output_bytes].into_iter().chain(owner_bytes),
        "preencoded HT worker wave",
    ) {
        Ok(_) => {}
        Err(crate::EncodeError::AllocationTooLarge { .. }) => return Ok(None),
        Err(error) => return Err(error.into()),
    }
    let (mut slots, slot_bytes) = tracker.try_vec::<Option<PreencodedHtBlock>>(
        count,
        [scratch_bytes, output_bytes],
        "preencoded HT block results",
    )?;
    let (mut encoded, encoded_bytes) = tracker.try_vec::<crate::EncodedHtJ2kCodeBlock>(
        count,
        [scratch_bytes, output_bytes, slot_bytes],
        "preencoded HT block payloads",
    )?;
    let (mut code_blocks, _) = tracker.try_vec::<PreparedEncodeCodeBlock>(
        count,
        [scratch_bytes, output_bytes, slot_bytes, encoded_bytes],
        "prepared code-block owners",
    )?;
    let encode = |state: &mut PreencodeState, index: usize| {
        encode_preencoded_ht_block(request, plan, view, quantizer, block_bytes, state, index)
    };
    slots.resize_with(count, || None);
    fill_preencoded_slots(&mut slots, workers, parallel, encode);

    let mut maximum = 0;
    for (index, slot) in slots.into_iter().enumerate() {
        let (block, block_maximum) = slot.ok_or_else(|| {
            NativeEncodePipelineError::internal_invariant("preencoded HT block result is missing")
        })??;
        let (_, _, width, height) = preencoded_block_span(request, plan, index)?;
        encoded.push(block);
        code_blocks.push(PreparedEncodeCodeBlock {
            coefficients: super::PreparedCodeBlockCoefficients::Empty,
            width,
            height,
        });
        maximum = maximum.max(block_maximum);
    }
    tracker.check(
        [public_ht_blocks_ownership(&encoded, encoded.capacity())?],
        "preencoded HT subband output",
    )?;
    Ok(Some(prepared_subband(
        request,
        plan,
        code_blocks,
        Some(encoded),
        Some(maximum),
    )))
}

/// The largest output the band's blocks can retain, summed.
fn preencoded_output_bound(
    request: &F32SubbandEncodeRequest<'_, '_>,
    plan: F32SubbandPlan,
    count: usize,
) -> NativeEncodePipelineResult<usize> {
    (0..count).try_fold(0, |bytes, index| {
        let (_, _, width, height) = preencoded_block_span(request, plan, index)?;
        let block = ht_block_encode::ht_worker_allocation(width as usize, height as usize, 1)?;
        Ok(checked_add_bytes(
            bytes,
            block.output_bytes,
            "preencoded HT output bound",
        )?)
    })
}

/// A preencoded HT block and its largest magnitude.
type PreencodedHtBlock = NativeEncodePipelineResult<(crate::EncodedHtJ2kCodeBlock, u64)>;
/// One worker's block buffer and, once it has coded a block, HT workspace.
type PreencodeState = (Vec<i32>, Option<ht_block_encode::HtEncodeWorkspace>);

/// Fill every slot with its encoded block, on `workers` workers when
/// `parallel`.
fn fill_preencoded_slots(
    slots: &mut [Option<PreencodedHtBlock>],
    workers: usize,
    parallel: bool,
    encode: impl Fn(&mut PreencodeState, usize) -> PreencodedHtBlock + Sync,
) {
    #[cfg(feature = "parallel")]
    if parallel && slots.len() > 1 && workers > 1 {
        use rayon::prelude::*;
        // One buffer and workspace per worker, as in the Tier-1 driver: a
        // state per Rayon split would rebuild the 20 KiB reservoirs many
        // times. Small batches keep slower cores from holding the band up.
        let batch = (slots.len() / (4 * workers)).clamp(1, 8);
        let batches = std::sync::Mutex::new(slots.chunks_mut(batch).enumerate());
        (0..workers).into_par_iter().with_max_len(1).for_each(|_| {
            let mut state = (Vec::new(), None);
            while let Some((first, slots)) =
                batches.lock().ok().and_then(|mut batches| batches.next())
            {
                for (offset, slot) in slots.iter_mut().enumerate() {
                    *slot = Some(encode(&mut state, first * batch + offset));
                }
            }
        });
        return;
    }
    let _ = (workers, parallel);
    let mut state = (Vec::new(), None);
    for (index, slot) in slots.iter_mut().enumerate() {
        *slot = Some(encode(&mut state, index));
    }
}

/// Origin and size of code block `index` of the band.
fn preencoded_block_span(
    request: &F32SubbandEncodeRequest<'_, '_>,
    plan: F32SubbandPlan,
    index: usize,
) -> NativeEncodePipelineResult<(u32, u32, u32, u32)> {
    let index = u32::try_from(index)
        .map_err(|_| NativeEncodePipelineError::arithmetic_overflow("code-block index"))?;
    let (x0, width) = super::tile_grid::block_span(
        request.origin.0,
        request.width,
        request.cb_width,
        index % plan.num_cbs_x,
    )?;
    let (y0, height) = super::tile_grid::block_span(
        request.origin.1,
        request.height,
        request.cb_height,
        index / plan.num_cbs_x,
    )?;
    Ok((x0, y0, width, height))
}

/// Quantize code block `index` from the packed band into the worker's buffer
/// and encode it at once.
fn encode_preencoded_ht_block(
    request: &F32SubbandEncodeRequest<'_, '_>,
    plan: F32SubbandPlan,
    view: PackedSubbandView<'_, f32>,
    quantizer: quantize::Quantizer,
    block_bytes: usize,
    (buffer, workspace): &mut PreencodeState,
    index: usize,
) -> PreencodedHtBlock {
    let (x0, y0, width, height) = preencoded_block_span(request, plan, index)?;
    buffer.clear();
    buffer
        .try_reserve_exact((width * height) as usize)
        .map_err(|_| host_allocation_failed("preencoded HT block buffer", block_bytes))?;
    for y in y0..y0 + height {
        let row = view.row(y).ok_or_else(|| {
            NativeEncodePipelineError::internal_invariant("packed subband row is missing")
        })?;
        quantizer.extend(buffer, &row[x0 as usize..][..width as usize]);
    }
    let maximum = buffer
        .iter()
        .map(|value| value.unsigned_abs())
        .max()
        .unwrap_or(0);
    let workspace = match workspace {
        Some(workspace) => workspace,
        None => workspace.insert(ht_block_encode::HtEncodeWorkspace::try_new()?),
    };
    let encoded = ht_block_encode::try_encode_code_block_with_passes_in_workspace(
        buffer,
        width,
        height,
        plan.total_bitplanes,
        1,
        workspace,
    )?;
    let block = crate::EncodedHtJ2kCodeBlock {
        data: encoded.data,
        cleanup_length: encoded.ht_cleanup_length,
        refinement_length: encoded.ht_refinement_length,
        num_coding_passes: encoded.num_coding_passes,
        num_zero_bitplanes: encoded.num_zero_bitplanes,
    };
    Ok((block, u64::from(maximum)))
}

/// Split a subband into code blocks, each filled by `fill(x0, y0, width,
/// height, block)` with its row-major quantized samples. `source_bytes` are
/// the bytes the fill reads from while the blocks are built.
fn prepare_code_blocks(
    request: &F32SubbandEncodeRequest<'_, '_>,
    plan: F32SubbandPlan,
    source_bytes: usize,
    tracker: &mut Tier1PhaseTracker<'_, '_>,
    parallel: bool,
    fill: impl Fn(u32, u32, u32, u32, &mut Vec<i32>) -> NativeEncodePipelineResult<()> + Sync,
) -> NativeEncodePipelineResult<PreparedEncodeSubband> {
    let code_block_count = usize::try_from(plan.num_cbs_x.checked_mul(plan.num_cbs_y).ok_or(
        crate::EncodeError::ArithmeticOverflow {
            what: "prepared code-block count",
        },
    )?)
    .map_err(|_| crate::EncodeError::ArithmeticOverflow {
        what: "prepared code-block count",
    })?;
    let (mut code_blocks, code_block_owner_bytes) = tracker.try_vec::<PreparedEncodeCodeBlock>(
        code_block_count,
        [source_bytes],
        "prepared code-block owners",
    )?;
    // The blocks tile the subband, so charging all of its samples up front
    // lets the blocks fill in any order.
    let what = "prepared i32 code-block coefficients";
    let samples = (request.width as usize)
        .checked_mul(request.height as usize)
        .ok_or_else(|| NativeEncodePipelineError::arithmetic_overflow("subband sample count"))?;
    let coefficient_bytes = checked_element_bytes::<i32>(samples, what)?;
    tracker.check(
        [source_bytes, code_block_owner_bytes, coefficient_bytes],
        what,
    )?;
    code_blocks.resize_with(code_block_count, || PreparedEncodeCodeBlock {
        coefficients: super::PreparedCodeBlockCoefficients::Empty,
        width: 0,
        height: 0,
    });
    let (origin, width, height) = (request.origin, request.width, request.height);
    let (cb_width, cb_height) = (request.cb_width, request.cb_height);
    let fill = |(index, block): (usize, &mut PreparedEncodeCodeBlock)| {
        let index = u32::try_from(index).map_err(|_| crate::EncodeError::ArithmeticOverflow {
            what: "prepared code-block count",
        })?;
        let (x0, cbw) =
            super::tile_grid::block_span(origin.0, width, cb_width, index % plan.num_cbs_x)?;
        let (y0, cbh) =
            super::tile_grid::block_span(origin.1, height, cb_height, index / plan.num_cbs_x)?;
        let count = (cbw as usize).checked_mul(cbh as usize).ok_or(
            crate::EncodeError::ArithmeticOverflow {
                what: "prepared code-block coefficient count",
            },
        )?;
        let mut cb_coeffs = Vec::new();
        cb_coeffs
            .try_reserve_exact(count)
            .map_err(|_| host_allocation_failed(what, count.saturating_mul(4)))?;
        fill(x0, y0, cbw, cbh, &mut cb_coeffs)?;
        if cb_coeffs.len() != count {
            return Err(NativeEncodePipelineError::internal_invariant(
                "prepared code-block coefficient count changed",
            ));
        }
        *block = PreparedEncodeCodeBlock {
            coefficients: super::PreparedCodeBlockCoefficients::I32(cb_coeffs),
            width: cbw,
            height: cbh,
        };
        Ok::<(), NativeEncodePipelineError>(())
    };
    #[cfg(feature = "parallel")]
    if parallel && code_block_count > 1 && rayon::current_num_threads() > 1 {
        use rayon::prelude::*;
        code_blocks
            .par_iter_mut()
            .enumerate()
            .with_min_len(4)
            .try_for_each(fill)?;
    } else {
        code_blocks.iter_mut().enumerate().try_for_each(fill)?;
    }
    #[cfg(not(feature = "parallel"))]
    {
        let _ = parallel;
        code_blocks.iter_mut().enumerate().try_for_each(fill)?;
    }

    let prepared = prepared_subband(request, plan, code_blocks, None, None);
    let prepared_bytes =
        prepared_subbands_ownership(core::slice::from_ref(&prepared), 0)?.total()?;
    tracker.check([prepared_bytes], "prepared quantized subband")?;
    Ok(prepared)
}

fn prepared_subband(
    request: &F32SubbandEncodeRequest<'_, '_>,
    plan: F32SubbandPlan,
    code_blocks: Vec<PreparedEncodeCodeBlock>,
    preencoded_ht_code_blocks: Option<Vec<super::EncodedHtJ2kCodeBlock>>,
    preencoded_ht_maximum_cleanup_magnitude: Option<u64>,
) -> PreparedEncodeSubband {
    PreparedEncodeSubband {
        origin: request.origin,
        code_blocks,
        preencoded_ht_code_blocks,
        preencoded_ht_maximum_cleanup_magnitude,
        num_cbs_x: plan.num_cbs_x,
        num_cbs_y: plan.num_cbs_y,
        code_block_width: request.cb_width,
        code_block_height: request.cb_height,
        width: request.width,
        height: request.height,
        sub_band_type: request.sub_band_type,
        total_bitplanes: plan.total_bitplanes,
        block_coding_mode: request.block_coding_mode,
        ht_target_coding_passes: request.ht_target_coding_passes,
    }
}

#[derive(Clone, Copy)]
pub(super) struct I64SubbandEncodeSettings<'a> {
    pub(super) guard_bits: u8,
    pub(super) cb_width: u32,
    pub(super) cb_height: u32,
    pub(super) roi_shift: u8,
    pub(super) roi_regions: &'a [ComponentRoiEncodeRegion],
    pub(super) roi_scale: u32,
    pub(super) block_coding_mode: BlockCodingMode,
    pub(super) ht_target_coding_passes: u8,
}

fn code_block_shapes_for_session(
    width: u32,
    height: u32,
    cb_width: u32,
    cb_height: u32,
    encoded_bytes: usize,
    tracker: &mut Tier1PhaseTracker<'_, '_>,
) -> NativeEncodePipelineResult<Vec<PreparedEncodeCodeBlock>> {
    let num_cbs_x = width.div_ceil(cb_width);
    let num_cbs_y = height.div_ceil(cb_height);
    let count = usize::try_from(num_cbs_x.checked_mul(num_cbs_y).ok_or(
        crate::EncodeError::ArithmeticOverflow {
            what: "fused HT code-block shape count",
        },
    )?)
    .map_err(|_| crate::EncodeError::ArithmeticOverflow {
        what: "fused HT code-block shape count",
    })?;
    let (mut code_blocks, _) = tracker.try_vec::<PreparedEncodeCodeBlock>(
        count,
        [encoded_bytes],
        "fused HT code-block shape owners",
    )?;
    for cby in 0..num_cbs_y {
        for cbx in 0..num_cbs_x {
            let x0 = cbx * cb_width;
            let y0 = cby * cb_height;
            let x1 = (x0 + cb_width).min(width);
            let y1 = (y0 + cb_height).min(height);
            code_blocks.push(PreparedEncodeCodeBlock {
                coefficients: super::PreparedCodeBlockCoefficients::Empty,
                width: x1 - x0,
                height: y1 - y0,
            });
        }
    }
    Ok(code_blocks)
}

fn copy_code_block_coefficients_into(
    quantized: &[i32],
    width: usize,
    x0: usize,
    y0: usize,
    cbw: usize,
    cbh: usize,
    output: &mut Vec<i32>,
) -> NativeEncodePipelineResult<()> {
    let expected = cbw
        .checked_mul(cbh)
        .ok_or(crate::EncodeError::ArithmeticOverflow {
            what: "prepared code-block coefficient count",
        })?;
    if output.capacity() < expected || !output.is_empty() {
        return Err(crate::EncodeError::InternalInvariant {
            what: "prepared code-block coefficient owner does not match its plan",
        }
        .into());
    }
    for y in 0..cbh {
        let row_start = (y0 + y)
            .checked_mul(width)
            .and_then(|row| row.checked_add(x0))
            .ok_or(crate::EncodeError::ArithmeticOverflow {
                what: "prepared code-block coefficient range",
            })?;
        let row_end = row_start
            .checked_add(cbw)
            .ok_or(crate::EncodeError::ArithmeticOverflow {
                what: "prepared code-block coefficient range",
            })?;
        output.extend_from_slice(quantized.get(row_start..row_end).ok_or_else(|| {
            NativeEncodePipelineError::internal_invariant(
                "prepared code-block coefficient range is invalid",
            )
        })?);
    }
    if output.len() != expected {
        return Err(crate::EncodeError::InternalInvariant {
            what: "prepared code-block coefficient copy length changed",
        }
        .into());
    }
    Ok(())
}

fn subband_range_bits(bit_depth: u8, sub_band_type: SubBandType) -> u8 {
    let log_gain = match sub_band_type {
        SubBandType::LowLow => 0,
        SubBandType::LowHigh | SubBandType::HighLow => 1,
        SubBandType::HighHigh => 2,
    };

    bit_depth.saturating_add(log_gain)
}

#[cfg(test)]
mod tests;
