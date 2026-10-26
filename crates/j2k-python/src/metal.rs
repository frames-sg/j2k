// SPDX-License-Identifier: MIT OR Apache-2.0

//! Decode into fresh private Metal storage, then transfer its sole owner to DLPack.

use std::{ffi::c_void, sync::Mutex};

use j2k::{BatchDecodeOptions, NativeSampleType, PreparedBatch, PreparedBatchGroup};
use j2k_metal::MetalBatchDecoder;
use j2k_metal_support::{checked_private_buffer, MetalImageDestination, MetalImageLayout};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::MTLBuffer;
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};

use crate::{
    decode::{shape, Decoded, GroupMeta, MaterializedGroup},
    owned_items,
    tensor::{DType, Samples, Tensor},
    value_error,
};

pub(crate) struct MetalSamples {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    len: usize,
}

// SAFETY: Metal permits retain/release on any thread. This owner never accesses
// sample contents, and it is constructed only after codec work has completed.
// Once exported, only the DLPack consumer may access the allocation.
unsafe impl Send for MetalSamples {}

impl MetalSamples {
    pub fn pointer(&self) -> *mut c_void {
        // kDLMetal stores the MTLBuffer object, not a CPU contents pointer.
        Retained::as_ptr(&self.buffer).cast::<c_void>().cast_mut()
    }

    pub fn len(&self) -> usize {
        self.len
    }
}

pub(crate) struct MetalDecoder {
    decoder: Mutex<MetalBatchDecoder>,
}

impl MetalDecoder {
    pub fn new(options: BatchDecodeOptions) -> PyResult<Self> {
        // Apple Silicon exposes one integrated GPU; its system-default Metal
        // device is the sole PyTorch MPS device (DLPack device index zero).
        Ok(Self {
            decoder: Mutex::new(
                MetalBatchDecoder::system_default_with_options(options).map_err(value_error)?,
            ),
        })
    }

    pub fn decode(&self, prepared: &PreparedBatch) -> PyResult<Decoded> {
        let mut decoder = self
            .decoder
            .lock()
            .map_err(|_| PyRuntimeError::new_err("Metal decoder is poisoned"))?;
        let mut decoded = Decoded {
            errors: prepared
                .errors()
                .iter()
                .map(|e| (e.index, e.source.to_string()))
                .collect(),
            ..Decoded::default()
        };
        for group in prepared.groups() {
            match decode_group(&mut decoder, group) {
                Ok(output) => decoded.groups.push(output),
                Err(error) => decoded.group_errors.push((
                    owned_items(group.source_indices().iter().copied())?,
                    error.to_string(),
                )),
            }
        }
        Ok(decoded)
    }
}

fn decode_group(
    decoder: &mut MetalBatchDecoder,
    group: &PreparedBatchGroup,
) -> PyResult<MaterializedGroup> {
    let info = group.info();
    let dtype = match info.sample_type {
        NativeSampleType::U8 => DType::U8,
        NativeSampleType::U16 => DType::U16,
        NativeSampleType::I16 => DType::I16,
        _ => return Err(PyValueError::new_err("unsupported Metal tensor dtype")),
    };
    let format = info
        .native_pixel_format()
        .ok_or_else(|| PyValueError::new_err("unsupported Metal pixel format"))?;
    let row_bytes = (info.dimensions.0 as usize)
        .checked_mul(format.bytes_per_pixel())
        .ok_or_else(|| PyValueError::new_err("Metal row size overflow"))?;
    let image_bytes = row_bytes
        .checked_mul(info.dimensions.1 as usize)
        .ok_or_else(|| PyValueError::new_err("Metal image size overflow"))?;
    // The group starts at offset zero and is densely packed in either layout.
    // No external buffers, suballocations, or nonzero DLPack offsets are accepted.
    let layout = MetalImageLayout::new_batch(
        0,
        info.dimensions,
        row_bytes,
        format,
        group.images().len(),
        image_bytes,
    )
    .map_err(value_error)?;
    let buffer = checked_private_buffer(decoder.backend_session().device(), layout.byte_len())
        .map_err(value_error)?;
    // SAFETY: this fresh allocation has never been published. The submission
    // owns exclusive access until wait retires GPU work and validates status,
    // including on failure. No codec-owned resident view is created or exported.
    let destination =
        unsafe { MetalImageDestination::from_exclusive_buffer(buffer.clone(), layout) }
            .map_err(value_error)?;
    let completion = decoder
        .submit_prepared_group_into(group, destination)
        .and_then(|submission| submission.wait())
        .map_err(value_error)?;
    let (rects, warnings) = completion.into_parts();
    Ok(MaterializedGroup {
        meta: GroupMeta::of(info),
        recovered: owned_items((0..group.source_indices().len()).map(|_| false))?,
        indices: owned_items(group.source_indices().iter().copied())?,
        rects,
        warnings: warnings
            .into_iter()
            .map(|v| v.iter().map(ToString::to_string).collect())
            .collect(),
        tensor: Tensor::new(
            Samples::Metal(MetalSamples {
                buffer,
                len: layout.byte_len() / usize::from(dtype.bits / 8),
            }),
            shape(info, group.images().len())?,
            dtype,
            (8, 0),
        )?,
    })
}
