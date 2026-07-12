// SPDX-License-Identifier: MIT OR Apache-2.0

//! Exact retained-owner accounting for non-compact precomputed 9/7 inputs.

use alloc::vec::Vec;

use super::{
    EncodeComponentSampleInfo, EncodeParams, NativeEncodePipelineResult, NativeEncodeSession,
    PrecomputedHtj2k53Component, PrecomputedHtj2k53Image, PrecomputedHtj2k97Component,
    PrecomputedHtj2k97Image, PreencodedHtj2k97Component, PreencodedHtj2k97Image,
    PreencodedHtj2k97Resolution, PreencodedHtj2k97Subband, PrequantizedHtj2k97Component,
    PrequantizedHtj2k97Image, PrequantizedHtj2k97Resolution, PrequantizedHtj2k97Subband,
};
use crate::j2c::encode::allocation::{
    checked_add_bytes, checked_element_bytes, host_allocation_failed,
};
use crate::EncodeResult;

pub(in crate::j2c::encode) struct ConstructionTracker<'session, 'input> {
    session: &'session NativeEncodeSession<'input>,
    retained_base_bytes: usize,
    live_bytes: usize,
}

pub(super) fn precomputed_53_image_retained_bytes(
    image: &PrecomputedHtj2k53Image,
) -> EncodeResult<usize> {
    let mut bytes = add_capacity::<PrecomputedHtj2k53Component>(
        0,
        image.components.capacity(),
        "precomputed 5/3 component owners",
    )?;
    for component in &image.components {
        bytes = add_capacity::<f32>(bytes, component.dwt.ll.capacity(), "precomputed 5/3 LL")?;
        bytes = add_capacity::<crate::J2kForwardDwt53Level>(
            bytes,
            component.dwt.levels.capacity(),
            "precomputed 5/3 level owners",
        )?;
        for level in &component.dwt.levels {
            bytes = add_capacity::<f32>(bytes, level.hl.capacity(), "precomputed 5/3 HL")?;
            bytes = add_capacity::<f32>(bytes, level.lh.capacity(), "precomputed 5/3 LH")?;
            bytes = add_capacity::<f32>(bytes, level.hh.capacity(), "precomputed 5/3 HH")?;
        }
    }
    Ok(bytes)
}

impl<'session, 'input> ConstructionTracker<'session, 'input> {
    pub(in crate::j2c::encode) const fn new(
        session: &'session NativeEncodeSession<'input>,
        retained_base_bytes: usize,
    ) -> Self {
        Self {
            session,
            retained_base_bytes,
            live_bytes: 0,
        }
    }

    pub(in crate::j2c::encode) fn retained_bytes(&self, what: &'static str) -> EncodeResult<usize> {
        checked_add_bytes(self.retained_base_bytes, self.live_bytes, what)
    }

    pub(in crate::j2c::encode) fn check_temporary(
        &self,
        temporary_bytes: usize,
        what: &'static str,
    ) -> NativeEncodePipelineResult<()> {
        let phase_bytes = checked_add_bytes(self.live_bytes, temporary_bytes, what)?;
        self.session.checked_phase(
            checked_add_bytes(self.retained_base_bytes, phase_bytes, what)?,
            what,
        )?;
        Ok(())
    }

    pub(in crate::j2c::encode) fn retain_existing(
        &mut self,
        bytes: usize,
        what: &'static str,
    ) -> NativeEncodePipelineResult<()> {
        self.live_bytes = checked_add_bytes(self.live_bytes, bytes, what)?;
        self.session
            .checked_phase(self.retained_bytes(what)?, what)?;
        Ok(())
    }

    pub(in crate::j2c::encode) fn try_vec<T>(
        &mut self,
        count: usize,
        what: &'static str,
    ) -> NativeEncodePipelineResult<Vec<T>> {
        let requested_bytes = checked_element_bytes::<T>(count, what)?;
        self.check_temporary(requested_bytes, what)?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(|_| host_allocation_failed(what, requested_bytes))?;
        self.retain_existing(checked_element_bytes::<T>(values.capacity(), what)?, what)?;
        Ok(values)
    }

    pub(in crate::j2c::encode) fn try_copy_slice<T: Copy>(
        &mut self,
        values: &[T],
        what: &'static str,
    ) -> NativeEncodePipelineResult<Vec<T>> {
        let mut copy = self.try_vec::<T>(values.len(), what)?;
        copy.extend_from_slice(values);
        Ok(copy)
    }
}

pub(in crate::j2c::encode) fn precomputed_97_image_retained_bytes(
    image: &PrecomputedHtj2k97Image,
) -> EncodeResult<usize> {
    let mut bytes = add_capacity::<PrecomputedHtj2k97Component>(
        0,
        image.components.capacity(),
        "precomputed 9/7 component owners",
    )?;
    for component in &image.components {
        bytes = add_precomputed_component(bytes, component)?;
    }
    Ok(bytes)
}

pub(in crate::j2c::encode) fn precomputed_97_images_retained_bytes(
    images: &[PrecomputedHtj2k97Image],
    outer_capacity: usize,
) -> EncodeResult<usize> {
    let mut bytes = add_capacity::<PrecomputedHtj2k97Image>(
        0,
        outer_capacity,
        "precomputed 9/7 batch image owners",
    )?;
    for image in images {
        bytes = checked_add_bytes(
            bytes,
            precomputed_97_image_retained_bytes(image)?,
            "precomputed 9/7 batch retained inputs",
        )?;
    }
    Ok(bytes)
}

fn add_precomputed_component(
    mut bytes: usize,
    component: &PrecomputedHtj2k97Component,
) -> EncodeResult<usize> {
    bytes = add_capacity::<f32>(bytes, component.dwt.ll.capacity(), "precomputed 9/7 LL")?;
    bytes = add_capacity::<crate::J2kForwardDwt97Level>(
        bytes,
        component.dwt.levels.capacity(),
        "precomputed 9/7 level owners",
    )?;
    for level in &component.dwt.levels {
        bytes = add_capacity::<f32>(bytes, level.hl.capacity(), "precomputed 9/7 HL")?;
        bytes = add_capacity::<f32>(bytes, level.lh.capacity(), "precomputed 9/7 LH")?;
        bytes = add_capacity::<f32>(bytes, level.hh.capacity(), "precomputed 9/7 HH")?;
    }
    Ok(bytes)
}

pub(in crate::j2c::encode) fn prequantized_97_image_retained_bytes(
    image: &PrequantizedHtj2k97Image,
) -> EncodeResult<usize> {
    let mut bytes = add_capacity::<PrequantizedHtj2k97Component>(
        0,
        image.components.capacity(),
        "prequantized 9/7 component owners",
    )?;
    for component in &image.components {
        bytes = add_capacity::<PrequantizedHtj2k97Resolution>(
            bytes,
            component.resolutions.capacity(),
            "prequantized 9/7 resolution owners",
        )?;
        for resolution in &component.resolutions {
            bytes = add_capacity::<PrequantizedHtj2k97Subband>(
                bytes,
                resolution.subbands.capacity(),
                "prequantized 9/7 subband owners",
            )?;
            for subband in &resolution.subbands {
                bytes = add_capacity::<crate::PrequantizedHtj2k97CodeBlock>(
                    bytes,
                    subband.code_blocks.capacity(),
                    "prequantized 9/7 code-block owners",
                )?;
                for block in &subband.code_blocks {
                    bytes = add_capacity::<i32>(
                        bytes,
                        block.coefficients.capacity(),
                        "prequantized 9/7 coefficient payloads",
                    )?;
                }
            }
        }
    }
    Ok(bytes)
}

pub(in crate::j2c::encode) fn preencoded_97_image_retained_bytes(
    image: &PreencodedHtj2k97Image,
) -> EncodeResult<usize> {
    let mut bytes = add_capacity::<PreencodedHtj2k97Component>(
        0,
        image.components.capacity(),
        "preencoded 9/7 component owners",
    )?;
    for component in &image.components {
        bytes = add_capacity::<PreencodedHtj2k97Resolution>(
            bytes,
            component.resolutions.capacity(),
            "preencoded 9/7 resolution owners",
        )?;
        for resolution in &component.resolutions {
            bytes = add_capacity::<PreencodedHtj2k97Subband>(
                bytes,
                resolution.subbands.capacity(),
                "preencoded 9/7 subband owners",
            )?;
            for subband in &resolution.subbands {
                bytes = add_capacity::<crate::PreencodedHtj2k97CodeBlock>(
                    bytes,
                    subband.code_blocks.capacity(),
                    "preencoded 9/7 code-block owners",
                )?;
                for block in &subband.code_blocks {
                    bytes = add_capacity::<u8>(
                        bytes,
                        block.encoded.data.capacity(),
                        "preencoded 9/7 payloads",
                    )?;
                }
            }
        }
    }
    Ok(bytes)
}

pub(in crate::j2c::encode) fn encode_params_retained_bytes(
    params: &EncodeParams,
) -> EncodeResult<usize> {
    let mut bytes = add_capacity::<EncodeComponentSampleInfo>(
        0,
        params.component_sample_info.capacity(),
        "precomputed 9/7 component sample metadata",
    )?;
    bytes = add_capacity::<Vec<(u16, u16)>>(
        bytes,
        params.component_quantization_step_sizes.capacity(),
        "precomputed 9/7 component quantization owners",
    )?;
    for steps in &params.component_quantization_step_sizes {
        bytes = add_capacity::<(u16, u16)>(
            bytes,
            steps.capacity(),
            "precomputed 9/7 component quantization values",
        )?;
    }
    bytes = add_capacity::<(u8, u8)>(
        bytes,
        params.component_sampling.capacity(),
        "precomputed 9/7 component sampling",
    )?;
    bytes = add_capacity::<u8>(
        bytes,
        params.roi_component_shifts.capacity(),
        "precomputed 9/7 ROI shifts",
    )?;
    add_capacity::<(u8, u8)>(
        bytes,
        params.precinct_exponents.capacity(),
        "precomputed 9/7 precinct exponents",
    )
}

pub(in crate::j2c::encode) fn add_capacity<T>(
    bytes: usize,
    capacity: usize,
    what: &'static str,
) -> EncodeResult<usize> {
    checked_add_bytes(bytes, checked_element_bytes::<T>(capacity, what)?, what)
}
