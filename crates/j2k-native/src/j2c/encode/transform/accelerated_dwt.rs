// SPDX-License-Identifier: MIT OR Apache-2.0

//! Accelerator DWT dispatch, validation, and ownership-safe representation conversion.

use super::super::single_tile::{OwnedDwtComponent, PackedF32DwtComponent};
use super::super::{
    allocation::{checked_add_bytes, checked_element_bytes, host_allocation_failed},
    fdwt, DwtDecomposition, J2kEncodeStageAccelerator, J2kForwardDwt53Job, J2kForwardDwt53Level,
    J2kForwardDwt53Output, J2kForwardDwt97Job, J2kForwardDwt97Level, J2kForwardDwt97Output,
    NativeEncodePipelineResult, NativeEncodeSession, Vec,
};

pub(in crate::j2c::encode) struct ForwardDwtRequest<'a, 'input> {
    pub(in crate::j2c::encode) component: Vec<f32>,
    pub(in crate::j2c::encode) width: u32,
    pub(in crate::j2c::encode) height: u32,
    pub(in crate::j2c::encode) num_levels: u8,
    pub(in crate::j2c::encode) reversible: bool,
    pub(in crate::j2c::encode) session: &'a NativeEncodeSession<'input>,
    pub(in crate::j2c::encode) retained_base_bytes: usize,
    pub(in crate::j2c::encode) line_scratch: &'a mut [f32],
}

pub(in crate::j2c::encode) fn encode_forward_dwt(
    request: ForwardDwtRequest<'_, '_>,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> NativeEncodePipelineResult<OwnedDwtComponent> {
    let ForwardDwtRequest {
        mut component,
        width,
        height,
        num_levels,
        reversible,
        session,
        retained_base_bytes,
        line_scratch,
    } = request;
    if reversible {
        let output = accelerator
            .encode_forward_dwt53(J2kForwardDwt53Job {
                samples: &component,
                width,
                height,
                num_levels,
            })
            .map_err(|source| crate::EncodeError::Accelerator {
                operation: "forward 5/3 DWT",
                source,
            })?;
        if let Some(output) = output {
            let decomposition = convert_forward_dwt53_output(
                output,
                session,
                retained_base_bytes,
                "forward 5/3 DWT",
            )?;
            return Ok(OwnedDwtComponent::Decomposed(decomposition));
        }
    } else {
        let output = accelerator
            .encode_forward_dwt97(J2kForwardDwt97Job {
                samples: &component,
                width,
                height,
                num_levels,
            })
            .map_err(|source| crate::EncodeError::Accelerator {
                operation: "forward 9/7 DWT",
                source,
            })?;
        if let Some(output) = output {
            let decomposition = convert_forward_dwt97_output(
                output,
                session,
                retained_base_bytes,
                "forward 9/7 DWT",
            )?;
            return Ok(OwnedDwtComponent::Decomposed(decomposition));
        }
    }

    let shape = fdwt::try_forward_dwt_packed_f32(
        &mut component,
        width,
        height,
        num_levels,
        reversible,
        line_scratch,
    )?;
    let geometry = fdwt::PackedDwtGeometry::try_new(width, height, component.len(), shape)?;
    Ok(OwnedDwtComponent::Packed(PackedF32DwtComponent {
        coefficients: component,
        geometry,
    }))
}

pub(in crate::j2c::encode) fn convert_forward_dwt53_output(
    output: J2kForwardDwt53Output,
    session: &NativeEncodeSession<'_>,
    retained_base_bytes: usize,
    operation: &'static str,
) -> NativeEncodePipelineResult<DwtDecomposition> {
    convert_accelerated_dwt_output(
        output.ll,
        output.ll_width,
        output.ll_height,
        output.levels,
        session,
        retained_base_bytes,
        operation,
    )
}

pub(in crate::j2c::encode) fn convert_forward_dwt97_output(
    output: J2kForwardDwt97Output,
    session: &NativeEncodeSession<'_>,
    retained_base_bytes: usize,
    operation: &'static str,
) -> NativeEncodePipelineResult<DwtDecomposition> {
    convert_accelerated_dwt_output(
        output.ll,
        output.ll_width,
        output.ll_height,
        output.levels,
        session,
        retained_base_bytes,
        operation,
    )
}

trait AcceleratedDwtLevel: Sized {
    fn validate(&self) -> Result<(), &'static str>;
    fn band_capacities(&self) -> [usize; 3];
    fn into_internal(self) -> fdwt::DwtLevel;
}

impl AcceleratedDwtLevel for J2kForwardDwt53Level {
    fn validate(&self) -> Result<(), &'static str> {
        validate_dwt53_level(self)
    }

    fn band_capacities(&self) -> [usize; 3] {
        [self.hl.capacity(), self.lh.capacity(), self.hh.capacity()]
    }

    fn into_internal(self) -> fdwt::DwtLevel {
        fdwt::DwtLevel {
            hl: self.hl,
            lh: self.lh,
            hh: self.hh,
            low_width: self.low_width,
            low_height: self.low_height,
            high_width: self.high_width,
            high_height: self.high_height,
        }
    }
}

impl AcceleratedDwtLevel for J2kForwardDwt97Level {
    fn validate(&self) -> Result<(), &'static str> {
        validate_dwt97_level(self)
    }

    fn band_capacities(&self) -> [usize; 3] {
        [self.hl.capacity(), self.lh.capacity(), self.hh.capacity()]
    }

    fn into_internal(self) -> fdwt::DwtLevel {
        fdwt::DwtLevel {
            hl: self.hl,
            lh: self.lh,
            hh: self.hh,
            low_width: self.low_width,
            low_height: self.low_height,
            high_width: self.high_width,
            high_height: self.high_height,
        }
    }
}

fn convert_accelerated_dwt_output<L: AcceleratedDwtLevel>(
    ll: Vec<f32>,
    ll_width: u32,
    ll_height: u32,
    source_levels: Vec<L>,
    session: &NativeEncodeSession<'_>,
    retained_base_bytes: usize,
    operation: &'static str,
) -> NativeEncodePipelineResult<DwtDecomposition> {
    validate_band_len(ll.len(), ll_width, ll_height).map_err(|detail| {
        crate::EncodeError::Accelerator {
            operation,
            source: crate::J2kEncodeStageError::internal_invariant(detail),
        }
    })?;
    for level in &source_levels {
        level
            .validate()
            .map_err(|detail| crate::EncodeError::Accelerator {
                operation,
                source: crate::J2kEncodeStageError::internal_invariant(detail),
            })?;
    }

    let source_bytes = accelerated_dwt_output_retained_bytes(&ll, &source_levels)?;
    session.checked_phase(
        checked_add_bytes(
            retained_base_bytes,
            source_bytes,
            "accelerator DWT source output",
        )?,
        "accelerator DWT source output",
    )?;
    let requested_owner_bytes = checked_element_bytes::<fdwt::DwtLevel>(
        source_levels.len(),
        "accelerator DWT destination level owners",
    )?;
    session.checked_phase(
        checked_add_bytes(
            retained_base_bytes,
            checked_add_bytes(
                source_bytes,
                requested_owner_bytes,
                "accelerator DWT conversion overlap",
            )?,
            "accelerator DWT conversion overlap",
        )?,
        "accelerator DWT conversion overlap",
    )?;
    let mut levels = Vec::new();
    levels.try_reserve_exact(source_levels.len()).map_err(|_| {
        host_allocation_failed(
            "accelerator DWT destination level owners",
            requested_owner_bytes,
        )
    })?;
    let actual_owner_bytes = checked_element_bytes::<fdwt::DwtLevel>(
        levels.capacity(),
        "accelerator DWT destination level owners",
    )?;
    session.checked_phase(
        checked_add_bytes(
            retained_base_bytes,
            checked_add_bytes(
                source_bytes,
                actual_owner_bytes,
                "accelerator DWT conversion overlap",
            )?,
            "accelerator DWT conversion overlap",
        )?,
        "accelerator DWT conversion overlap",
    )?;
    levels.extend(
        source_levels
            .into_iter()
            .map(AcceleratedDwtLevel::into_internal),
    );
    Ok(DwtDecomposition {
        ll,
        ll_width,
        ll_height,
        levels,
    })
}

fn accelerated_dwt_output_retained_bytes<L: AcceleratedDwtLevel>(
    ll: &Vec<f32>,
    levels: &Vec<L>,
) -> crate::EncodeResult<usize> {
    let mut bytes = checked_element_bytes::<f32>(ll.capacity(), "accelerator DWT LL samples")?;
    bytes = checked_add_bytes(
        bytes,
        checked_element_bytes::<L>(levels.capacity(), "accelerator DWT source level owners")?,
        "accelerator DWT source output",
    )?;
    for level in levels {
        for capacity in level.band_capacities() {
            bytes = checked_add_bytes(
                bytes,
                checked_element_bytes::<f32>(capacity, "accelerator DWT detail samples")?,
                "accelerator DWT source output",
            )?;
        }
    }
    Ok(bytes)
}

pub(in crate::j2c::encode) fn forward_dwt53_output_retained_bytes(
    output: &J2kForwardDwt53Output,
) -> crate::EncodeResult<usize> {
    accelerated_dwt_output_retained_bytes(&output.ll, &output.levels)
}

pub(in crate::j2c::encode) fn validate_dwt53_level(
    level: &J2kForwardDwt53Level,
) -> Result<(), &'static str> {
    validate_band_len(level.hl.len(), level.high_width, level.low_height)?;
    validate_band_len(level.lh.len(), level.low_width, level.high_height)?;
    validate_band_len(level.hh.len(), level.high_width, level.high_height)?;
    Ok(())
}

pub(in crate::j2c::encode) fn validate_dwt97_level(
    level: &J2kForwardDwt97Level,
) -> Result<(), &'static str> {
    validate_band_len(level.hl.len(), level.high_width, level.low_height)?;
    validate_band_len(level.lh.len(), level.low_width, level.high_height)?;
    validate_band_len(level.hh.len(), level.high_width, level.high_height)?;
    Ok(())
}

pub(in crate::j2c::encode) fn validate_band_len(
    actual: usize,
    width: u32,
    height: u32,
) -> Result<(), &'static str> {
    let expected = (width as usize)
        .checked_mul(height as usize)
        .ok_or("accelerated DWT output dimensions overflow")?;
    if actual != expected {
        return Err("accelerated DWT output length mismatch");
    }
    Ok(())
}
