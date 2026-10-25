// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
use crate::metal_types::prelude::*;

use std::time::Instant;

use crate::metal_types::{Buffer, ComputeCommandEncoderRef};
use j2k_native::{J2kDirectBandId, J2kDirectStoreStep, J2kWaveletTransform};

use super::{upload_cpu_decoded_coefficients, DirectComponentPlaneRequest};
use crate::engine::{
    abi::{J2kIdwtSingleDecompositionParams, J2kStoreParams, J2kZeroRect},
    band_cells,
    decode_dispatch::{
        default_metal_ht_chunk_limits, encode_metal_ht_batches_with_zero_layouts_in_encoder,
        encode_prepared_classic_sub_band_group_to_buffer_in_encoder,
        encode_prepared_classic_sub_band_to_buffer_in_encoder,
        encode_prepared_ht_sub_band_group_to_buffer_in_encoder,
        encode_prepared_ht_sub_band_to_buffer_in_encoder,
        idwt::{
            dispatch_irreversible97_single_decomposition_buffers_in_encoder_with_offsets,
            dispatch_reversible53_single_decomposition_buffers_in_encoder_with_offsets,
            IdwtSubBandBuffers, SingleIdwtDispatch,
        },
        store::dispatch_store_component_buffer_in_encoder_with_offsets,
        HtBatchInput, HtZeroLayout,
    },
    direct_cpu::{
        decode_prepared_classic_sub_band_group_on_cpu_profile,
        decode_prepared_classic_sub_band_on_cpu_profile,
        decode_prepared_ht_sub_band_group_on_cpu_profile,
        decode_prepared_ht_sub_band_on_cpu_profile,
    },
    direct_grayscale_execute::extend_preallocated_retained_buffers,
    direct_plan_types::{
        PreparedClassicSubBand, PreparedClassicSubBandGroup, PreparedDirectGrayscalePlan,
        PreparedDirectGrayscaleStep, PreparedDirectIdwt, PreparedHtSubBand, PreparedHtSubBandGroup,
    },
    direct_profile::{elapsed_us, CpuTier1DecodeSubstageCounters, DirectHybridStageTimings},
    direct_roi::{
        checked_f32_span, idwt_input_windows_from_slices, prepared_idwt_output_len,
        prepared_idwt_params, BandRequiredRegion,
    },
    direct_scratch::{take_f32_scratch_buffer, DirectScratchBuffer},
    direct_stacked_batch::{
        lookup_direct_band_slice, lookup_direct_band_slice_entry, push_member_band_slices,
        DirectBandSlice,
    },
    direct_status::DirectStatusCheck,
    direct_tier1::DirectTier1Mode,
    BandCells, MetalRuntime,
};
use crate::{profile_env::metal_profile_stages_enabled, Error};

mod final_plane;

use self::final_plane::FinalComponentPlane;

struct ComponentPlaneExecution<'a> {
    runtime: &'a MetalRuntime,
    encoder: &'a ComputeCommandEncoderRef,
    plan: &'a PreparedDirectGrayscalePlan,
    tier1_mode: DirectTier1Mode,
    profile_stages: bool,
    stage_timings: &'a mut DirectHybridStageTimings,
    retained_buffers: &'a mut Vec<Buffer>,
    status_checks: &'a mut Vec<DirectStatusCheck>,
    scratch_buffers: &'a mut Vec<DirectScratchBuffer>,
    bands: Vec<DirectBandSlice>,
    final_plane: FinalComponentPlane,
    round_centered_store: bool,
    /// Sub-bands whose only reader treats uncovered code-block cells as zero,
    /// so decoding records their cell maps instead of zeroing them.
    sparse_bands: [Option<J2kDirectBandId>; 3],
    band_cells: Vec<(J2kDirectBandId, BandCells)>,
}

impl ComponentPlaneExecution<'_> {
    fn decode_and_upload_cpu(
        &mut self,
        decode: impl FnOnce(Option<&CpuTier1DecodeSubstageCounters>) -> Result<Vec<f32>, Error>,
    ) -> Result<Buffer, Error> {
        let decode_started = self.profile_stages.then(Instant::now);
        let cpu_tier1_counters = self
            .profile_stages
            .then(CpuTier1DecodeSubstageCounters::default);
        let coefficients = decode(cpu_tier1_counters.as_ref())?;
        if let Some(started) = decode_started {
            self.stage_timings.cpu_tier1 += elapsed_us(started);
        }
        if let Some(counters) = &cpu_tier1_counters {
            counters.add_to_stage_timings(self.stage_timings);
        }
        let upload_started = self.profile_stages.then(Instant::now);
        let buffer =
            upload_cpu_decoded_coefficients(self.runtime, &coefficients, self.retained_buffers)?;
        if let Some(started) = upload_started {
            self.stage_timings.coefficient_upload += elapsed_us(started);
        }
        Ok(buffer)
    }

    fn encode_classic_group(&mut self, group: &PreparedClassicSubBandGroup) -> Result<(), Error> {
        let output_span = checked_f32_span(
            group.total_coefficients,
            1,
            "classic J2K MetalDirect grouped coefficients",
        )?;
        let buffer = match self.tier1_mode {
            DirectTier1Mode::Metal => {
                let output = take_f32_scratch_buffer(self.runtime, output_span.elements)?;
                let (buffers, status_check) =
                    encode_prepared_classic_sub_band_group_to_buffer_in_encoder(
                        self.runtime,
                        self.encoder,
                        group,
                        &output.buffer,
                        self.scratch_buffers,
                    )?;
                extend_preallocated_retained_buffers(self.retained_buffers, buffers)?;
                self.status_checks.push(status_check);
                self.encoder
                    .memory_barrier_with_resources(&[&output.buffer]);
                let buffer = output.buffer.clone();
                self.scratch_buffers.push(output);
                buffer
            }
            DirectTier1Mode::CpuUpload => self.decode_and_upload_cpu(|counters| {
                decode_prepared_classic_sub_band_group_on_cpu_profile(group, counters)
            })?,
        };
        let members = group.members.iter();
        push_member_band_slices(
            &mut self.bands,
            &buffer,
            members.map(|member| (member.band_id, member.offset_elements, member.window)),
        )
    }

    fn encode_ht_group(&mut self, group: &PreparedHtSubBandGroup) -> Result<(), Error> {
        let output_span = checked_f32_span(
            group.total_coefficients,
            1,
            "HTJ2K MetalDirect grouped coefficients",
        )?;
        let buffer = match self.tier1_mode {
            DirectTier1Mode::Metal => {
                let output = take_f32_scratch_buffer(self.runtime, output_span.elements)?;
                let sparse_zero_rects =
                    sparse_zero_rects(group, &self.sparse_bands, &mut self.band_cells)?;
                let (buffers, status_check) =
                    encode_prepared_ht_sub_band_group_to_buffer_in_encoder(
                        self.runtime,
                        self.encoder,
                        group,
                        sparse_zero_rects.as_deref().unwrap_or(&group.zero_rects),
                        &output.buffer,
                    )?;
                extend_preallocated_retained_buffers(self.retained_buffers, buffers)?;
                self.status_checks.push(status_check);
                self.encoder
                    .memory_barrier_with_resources(&[&output.buffer]);
                let buffer = output.buffer.clone();
                self.scratch_buffers.push(output);
                buffer
            }
            DirectTier1Mode::CpuUpload => self.decode_and_upload_cpu(|counters| {
                decode_prepared_ht_sub_band_group_on_cpu_profile(group, counters)
            })?,
        };
        let members = group.members.iter();
        push_member_band_slices(
            &mut self.bands,
            &buffer,
            members.map(|member| (member.band_id, member.offset_elements, member.window)),
        )
    }

    fn encode_classic_sub_band(&mut self, sub_band: &PreparedClassicSubBand) -> Result<(), Error> {
        let output_span = checked_f32_span(
            sub_band.width as usize,
            sub_band.height as usize,
            "classic J2K MetalDirect sub-band",
        )?;
        let buffer = match self.tier1_mode {
            DirectTier1Mode::Metal => {
                let output = take_f32_scratch_buffer(self.runtime, output_span.elements)?;
                let (buffers, status_check) =
                    encode_prepared_classic_sub_band_to_buffer_in_encoder(
                        self.runtime,
                        self.encoder,
                        sub_band,
                        &output.buffer,
                        self.scratch_buffers,
                    )?;
                extend_preallocated_retained_buffers(self.retained_buffers, buffers)?;
                self.status_checks.push(status_check);
                self.encoder
                    .memory_barrier_with_resources(&[&output.buffer]);
                let buffer = output.buffer.clone();
                self.scratch_buffers.push(output);
                buffer
            }
            DirectTier1Mode::CpuUpload => self.decode_and_upload_cpu(|counters| {
                decode_prepared_classic_sub_band_on_cpu_profile(sub_band, counters)
            })?,
        };
        self.bands.push(DirectBandSlice {
            band_id: sub_band.band_id,
            buffer,
            offset_bytes: 0,
            window: BandRequiredRegion::full(sub_band.width, sub_band.height),
        });
        Ok(())
    }

    fn encode_ht_sub_band(&mut self, sub_band: &PreparedHtSubBand) -> Result<(), Error> {
        let output_span = checked_f32_span(
            sub_band.width as usize,
            sub_band.height as usize,
            "HTJ2K MetalDirect sub-band",
        )?;
        let buffer = match self.tier1_mode {
            DirectTier1Mode::Metal => {
                let output = take_f32_scratch_buffer(self.runtime, output_span.elements)?;
                let (buffers, status_check) = encode_prepared_ht_sub_band_to_buffer_in_encoder(
                    self.runtime,
                    self.encoder,
                    sub_band,
                    &output.buffer,
                )?;
                extend_preallocated_retained_buffers(self.retained_buffers, buffers)?;
                self.status_checks.push(status_check);
                self.encoder
                    .memory_barrier_with_resources(&[&output.buffer]);
                let buffer = output.buffer.clone();
                self.scratch_buffers.push(output);
                buffer
            }
            DirectTier1Mode::CpuUpload => self.decode_and_upload_cpu(|counters| {
                decode_prepared_ht_sub_band_on_cpu_profile(sub_band, counters)
            })?,
        };
        self.bands.push(DirectBandSlice {
            band_id: sub_band.band_id,
            buffer,
            offset_bytes: 0,
            window: BandRequiredRegion::full(sub_band.width, sub_band.height),
        });
        Ok(())
    }

    fn encode_idwt(&mut self, idwt: &PreparedDirectIdwt) -> Result<(), Error> {
        let ll = lookup_direct_band_slice_entry(&self.bands, idwt.step.ll_band_id, idwt.step.ll)?;
        let hl = lookup_direct_band_slice_entry(&self.bands, idwt.step.hl_band_id, idwt.step.hl)?;
        let lh = lookup_direct_band_slice_entry(&self.bands, idwt.step.lh_band_id, idwt.step.lh)?;
        let hh = lookup_direct_band_slice_entry(&self.bands, idwt.step.hh_band_id, idwt.step.hh)?;
        let params = prepared_idwt_params(idwt, idwt_input_windows_from_slices(&ll, &hl, &lh, &hh));
        let output = take_f32_scratch_buffer(self.runtime, prepared_idwt_output_len(idwt)?)?;
        let encode_started = self.profile_stages.then(Instant::now);
        let dispatch = SingleIdwtDispatch {
            kernels: self.runtime.decode()?,
            sub_bands: IdwtSubBandBuffers {
                ll: &ll.buffer,
                ll_offset: ll.offset_bytes,
                hl: &hl.buffer,
                hl_offset: hl.offset_bytes,
                lh: &lh.buffer,
                lh_offset: lh.offset_bytes,
                hh: &hh.buffer,
                hh_offset: hh.offset_bytes,
            },
            params,
            decoded: &output.buffer,
            decoded_offset: 0,
        };
        match idwt.step.transform {
            J2kWaveletTransform::Reversible53 => {
                dispatch_reversible53_single_decomposition_buffers_in_encoder_with_offsets(
                    self.encoder,
                    dispatch,
                );
            }
            J2kWaveletTransform::Irreversible97 => {
                dispatch_irreversible97_single_decomposition_buffers_in_encoder_with_offsets(
                    self.encoder,
                    dispatch,
                );
            }
        }
        self.encoder
            .memory_barrier_with_resources(&[&output.buffer]);
        if let Some(started) = encode_started {
            self.stage_timings.metal_idwt_encode += elapsed_us(started);
        }
        self.bands.push(DirectBandSlice {
            band_id: idwt.step.output_band_id,
            buffer: output.buffer.clone(),
            offset_bytes: 0,
            window: idwt.output_window,
        });
        self.scratch_buffers.push(output);
        Ok(())
    }

    fn encode_store(&mut self, store: &J2kDirectStoreStep, last: bool) -> Result<(), Error> {
        let (input, input_offset) =
            lookup_direct_band_slice(&self.bands, store.input_band_id, store.input_rect)?;
        let output_span = checked_f32_span(
            store.output_width as usize,
            store.output_height as usize,
            "J2K MetalDirect stored component plane",
        )?;
        let dimensions = (store.output_width, store.output_height);
        // A complete identity store (MCT components carry no shift or rounding)
        // keeps the coefficient plane it would copy, as the stacked path does.
        // Tile assembly, cropping, level shifts and rounding still use Store.
        if last
            && self.final_plane.is_empty()
            && store.addend == 0.0
            && !self.round_centered_store
            && (
                store.source_x,
                store.source_y,
                store.output_x,
                store.output_y,
            ) == (0, 0, 0, 0)
            && (store.copy_width, store.copy_height) == dimensions
            && (store.input_rect.width(), store.input_rect.height()) == dimensions
            && input_offset == 0
            && input.length() >= output_span.bytes
        {
            let window =
                lookup_direct_band_slice_entry(&self.bands, store.input_band_id, store.input_rect)?
                    .window;
            if (window.width(), window.height()) == dimensions {
                self.final_plane
                    .adopt(input, dimensions, output_span.elements);
                self.bands.clear();
                return Ok(());
            }
        }
        let output = self.final_plane.buffer_for_store(
            self.runtime,
            dimensions,
            output_span.elements,
            output_span.bytes,
            self.scratch_buffers,
        )?;
        let encode_started = self.profile_stages.then(Instant::now);
        dispatch_store_component_buffer_in_encoder_with_offsets(
            self.runtime.decode()?,
            self.encoder,
            &input,
            input_offset,
            &output,
            0,
            J2kStoreParams {
                input_width: store.input_rect.width(),
                source_x: store.source_x,
                source_y: store.source_y,
                copy_width: store.copy_width,
                copy_height: store.copy_height,
                output_width: store.output_width,
                output_x: store.output_x,
                output_y: store.output_y,
                addend: store.addend,
                round_centered: u32::from(self.round_centered_store),
            },
        );
        if let Some(started) = encode_started {
            self.stage_timings.metal_store_encode += elapsed_us(started);
        }
        // Every referenced tile owns an independent coefficient graph. The
        // Store is the tile boundary: subsequent tiles write disjoint ranges
        // of the same final plane and no longer retain this tile's band views.
        self.bands.clear();
        Ok(())
    }

    fn encode_step(&mut self, step: &PreparedDirectGrayscaleStep, last: bool) -> Result<(), Error> {
        match step {
            PreparedDirectGrayscaleStep::ClassicSubBand(sub_band) => {
                self.encode_classic_sub_band(sub_band)
            }
            PreparedDirectGrayscaleStep::HtSubBand(sub_band) => self.encode_ht_sub_band(sub_band),
            PreparedDirectGrayscaleStep::Idwt(idwt) => self.encode_idwt(idwt),
            PreparedDirectGrayscaleStep::Store(store) => self.encode_store(store, last),
        }
    }

    /// Encodes the plan's steps before `end`.
    fn encode_steps(&mut self, start: usize, end: usize) -> Result<(), Error> {
        let plan = self.plan;
        let mut step_idx = start;
        while step_idx < end {
            if let Some(group) = plan.classic_group_starting_at(step_idx) {
                self.encode_classic_group(group)?;
                step_idx = group.end_step;
                continue;
            }
            if let Some(group) = plan.ht_group_starting_at(step_idx) {
                self.encode_ht_group(group)?;
                step_idx = group.end_step;
                continue;
            }
            self.encode_step(&plan.steps[step_idx], step_idx + 1 == plan.steps.len())?;
            step_idx += 1;
        }
        Ok(())
    }

    fn finish(self) -> Result<Buffer, Error> {
        self.final_plane.finish()
    }
}

pub(in crate::engine) fn encode_prepared_direct_component_plane_in_encoder(
    request: DirectComponentPlaneRequest<'_>,
    encoder: &ComputeCommandEncoderRef,
) -> Result<Buffer, Error> {
    let steps = request.plan.steps.len();
    let mut execution = component_plane_execution(request, encoder)?;
    execution.encode_steps(0, steps)?;
    execution.finish()
}

/// The group's zero rectangles without those of its `sparse` sub-bands whose
/// cell maps it appends to `cells`, or `None` when it appends none.
fn sparse_zero_rects(
    group: &PreparedHtSubBandGroup,
    sparse: &[Option<J2kDirectBandId>; 3],
    cells: &mut Vec<(J2kDirectBandId, BandCells)>,
) -> Result<Option<Vec<J2kZeroRect>>, Error> {
    let mut skipped = Vec::new();
    for member in &group.members {
        if !sparse.contains(&Some(member.band_id)) {
            continue;
        }
        let (width, height) = (member.window.width(), member.window.height());
        let span =
            member.offset_elements..member.offset_elements + width as usize * height as usize;
        // A sub-band its jobs cover entirely reads densely.
        if !group
            .zero_rects
            .iter()
            .any(|rect| span.contains(&(rect.offset as usize)))
        {
            continue;
        }
        if let Some(map) = band_cells(member.offset_elements, width, height, &group.jobs)? {
            cells.push((member.band_id, map));
            skipped.push(span);
        }
    }
    if skipped.is_empty() {
        return Ok(None);
    }
    let mut budget =
        crate::batch_allocation::BatchMetadataBudget::new("HTJ2K MetalDirect zero rectangles");
    let mut rects = budget.try_vec(group.zero_rects.len(), "HT sparse zero rectangles")?;
    rects.extend(group.zero_rects.iter().copied().filter(|rect| {
        !skipped
            .iter()
            .any(|range| range.contains(&(rect.offset as usize)))
    }));
    Ok(Some(rects))
}

/// The last wavelet level of `plan` when a complete identity store of the
/// whole `dimensions` plane is its only reader, so a kernel that writes the
/// final output can replace both.
pub(in crate::engine) fn final_identity_stored_level(
    plan: &PreparedDirectGrayscalePlan,
    dimensions: (u32, u32),
) -> Option<&PreparedDirectIdwt> {
    let [.., PreparedDirectGrayscaleStep::Idwt(idwt), PreparedDirectGrayscaleStep::Store(store)] =
        plan.steps.as_slice()
    else {
        return None;
    };
    let window = idwt.output_window;
    (!plan.round_centered_store
        && store.addend == 0.0
        && store.input_band_id == idwt.step.output_band_id
        && (
            store.source_x,
            store.source_y,
            store.output_x,
            store.output_y,
        ) == (0, 0, 0, 0)
        && (store.copy_width, store.copy_height) == dimensions
        && (store.output_width, store.output_height) == dimensions
        && (store.input_rect.width(), store.input_rect.height()) == dimensions
        && (window.x0, window.y0, window.x1, window.y1) == (0, 0, dimensions.0, dimensions.1))
        .then_some(idwt)
}

/// The inputs of a component's last wavelet level, with the cell maps of its
/// high-pass sub-bands whose uncovered cells were left unzeroed (`None` for a
/// sub-band zeroed as usual).
pub(in crate::engine) type FinalLevelInputs = (
    [DirectBandSlice; 4],
    J2kIdwtSingleDecompositionParams,
    [Option<BandCells>; 3],
);

/// Encodes every step of `request.plan` before `idwt`, its
/// [`final_identity_stored_level`], and returns that level's inputs. That
/// level's kernel must read uncovered high-pass cells as zero.
pub(in crate::engine) fn encode_prepared_direct_component_plane_before_final_level(
    request: DirectComponentPlaneRequest<'_>,
    encoder: &ComputeCommandEncoderRef,
    idwt: &PreparedDirectIdwt,
    decoded: Option<DecodedHtGroup>,
) -> Result<FinalLevelInputs, Error> {
    let steps = request.plan.steps.len();
    let mut execution = component_plane_execution(request, encoder)?;
    execution.sparse_bands = final_level_high_bands(idwt);
    let start = match decoded {
        Some(decoded) => {
            execution.bands = decoded.bands;
            execution.band_cells = decoded.band_cells;
            decoded.end_step
        }
        None => 0,
    };
    execution.encode_steps(start, steps.saturating_sub(2))?;
    let band = |id, rect| lookup_direct_band_slice_entry(&execution.bands, id, rect);
    let ll = band(idwt.step.ll_band_id, idwt.step.ll)?;
    let hl = band(idwt.step.hl_band_id, idwt.step.hl)?;
    let lh = band(idwt.step.lh_band_id, idwt.step.lh)?;
    let hh = band(idwt.step.hh_band_id, idwt.step.hh)?;
    let params = prepared_idwt_params(idwt, idwt_input_windows_from_slices(&ll, &hl, &lh, &hh));
    let mut cells = |id| {
        let index = execution
            .band_cells
            .iter()
            .position(|(band, _)| *band == id)?;
        Some(execution.band_cells.swap_remove(index).1)
    };
    let cells = [
        cells(idwt.step.hl_band_id),
        cells(idwt.step.lh_band_id),
        cells(idwt.step.hh_band_id),
    ];
    Ok(([ll, hl, lh, hh], params, cells))
}

fn component_plane_execution<'a>(
    request: DirectComponentPlaneRequest<'a>,
    encoder: &'a ComputeCommandEncoderRef,
) -> Result<ComponentPlaneExecution<'a>, Error> {
    let DirectComponentPlaneRequest {
        runtime,
        command_buffer: _,
        plan,
        tier1_mode,
        stage_timings,
        retained_buffers,
        status_checks,
        scratch_buffers,
    } = request;
    let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
        "J2K MetalDirect component band metadata",
    );
    let bands = budget.try_vec(plan.steps.len(), "J2K MetalDirect component band metadata")?;
    Ok(ComponentPlaneExecution {
        runtime,
        encoder,
        plan,
        tier1_mode,
        profile_stages: metal_profile_stages_enabled(),
        stage_timings,
        retained_buffers,
        status_checks,
        scratch_buffers,
        bands,
        final_plane: FinalComponentPlane::empty(),
        round_centered_store: plan.round_centered_store,
        sparse_bands: [None; 3],
        band_cells: Vec::new(),
    })
}

fn final_level_high_bands(idwt: &PreparedDirectIdwt) -> [Option<J2kDirectBandId>; 3] {
    [
        Some(idwt.step.hl_band_id),
        Some(idwt.step.lh_band_id),
        Some(idwt.step.hh_band_id),
    ]
}

/// A component's first HT group decoded by
/// [`encode_merged_final_level_ht_groups`]: its sub-band slices, the cell maps
/// of its last level's sparse high-pass sub-bands, and the step after it.
pub(in crate::engine) struct DecodedHtGroup {
    bands: Vec<DirectBandSlice>,
    band_cells: Vec<(J2kDirectBandId, BandCells)>,
    end_step: usize,
}

/// Decodes the first-step HT groups of every `(plan, last level)` into one
/// coefficient buffer with one set of HT dispatches, so the serial MEL/VLC
/// walks of all components share each dispatch; `None` when a plan does not
/// start with an HT group. The last levels' high-pass sub-bands are left
/// sparse, as [`encode_prepared_direct_component_plane_before_final_level`]
/// leaves them.
pub(in crate::engine) fn encode_merged_final_level_ht_groups(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    components: &[(&PreparedDirectGrayscalePlan, &PreparedDirectIdwt)],
    retained_buffers: &mut Vec<Buffer>,
    status_checks: &mut Vec<DirectStatusCheck>,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<Option<Vec<DecodedHtGroup>>, Error> {
    let Some(groups) = components
        .iter()
        .map(|(plan, _)| plan.ht_group_starting_at(0))
        .collect::<Option<Vec<_>>>()
    else {
        return Ok(None);
    };
    let mut budget =
        crate::batch_allocation::BatchMetadataBudget::new("HTJ2K MetalDirect merged groups");
    let mut decoded = budget.try_vec(groups.len(), "HTJ2K MetalDirect merged group results")?;
    let mut sparse_rects = budget.try_vec(groups.len(), "HTJ2K MetalDirect merged zero rects")?;
    let mut total = 0usize;
    for (group, (_, idwt)) in groups.iter().zip(components) {
        let mut band_cells = Vec::new();
        sparse_rects.push(sparse_zero_rects(
            group,
            &final_level_high_bands(idwt),
            &mut band_cells,
        )?);
        decoded.push(DecodedHtGroup {
            bands: Vec::new(),
            band_cells,
            end_step: group.end_step,
        });
        total = total
            .checked_add(group.total_coefficients)
            .ok_or_else(|| Error::MetalKernel {
                message: "HTJ2K MetalDirect merged coefficient count overflow".to_string(),
            })?;
    }
    let mut inputs = budget.try_vec(groups.len(), "HTJ2K MetalDirect merged batch inputs")?;
    let mut layouts = budget.try_vec(groups.len(), "HTJ2K MetalDirect merged zero layouts")?;
    let mut base = 0usize;
    for (source_index, (group, rects)) in groups.iter().zip(&sparse_rects).enumerate() {
        inputs.push(HtBatchInput {
            source_index,
            payload: group.payload_source.as_ht_payload_source(),
            jobs: &group.jobs,
            output_base: base,
            execution_owner: &group.execution_owner,
        });
        layouts.push(HtZeroLayout {
            extent: group.total_coefficients,
            uncovered: rects.as_deref().unwrap_or(&group.zero_rects),
        });
        base += group.total_coefficients;
    }
    let output = take_f32_scratch_buffer(
        runtime,
        checked_f32_span(total, 1, "HTJ2K MetalDirect merged coefficients")?.elements,
    )?;
    let (buffers, status_check) = encode_metal_ht_batches_with_zero_layouts_in_encoder(
        runtime,
        encoder,
        &inputs,
        Some(&layouts),
        &output.buffer,
        total,
        default_metal_ht_chunk_limits(),
    )?;
    extend_preallocated_retained_buffers(retained_buffers, buffers)?;
    status_checks.push(status_check);
    encoder.memory_barrier_with_resources(&[&output.buffer]);
    for ((group, input), result) in groups.iter().zip(&inputs).zip(&mut decoded) {
        push_member_band_slices(
            &mut result.bands,
            &output.buffer,
            group.members.iter().map(|member| {
                let offset = input.output_base + member.offset_elements;
                (member.band_id, offset, member.window)
            }),
        )?;
    }
    scratch_buffers.push(output);
    Ok(Some(decoded))
}
