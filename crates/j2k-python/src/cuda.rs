// SPDX-License-Identifier: MIT OR Apache-2.0

//! Decode directly into newly allocated PyTorch CUDA storage. PyTorch owns
//! subsequent consumer lifetimes; no codec allocation is borrowed by a model.

use std::sync::{Arc, Mutex};

use j2k::{BatchDecodeOptions, DecodeRequest, EncodedImage, NativeSampleType, PreparedBatch};
use j2k_cuda::{CudaBatchDecoder, CudaSession};
use j2k_cuda_runtime::{CudaContext, CudaExternalDeviceBufferViewMut};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::PyDict,
};

use crate::{
    decode::{shape, Batch, MaterializedGroup},
    owned_items,
    tensor::{DType, Samples, Tensor},
    value_error,
};

pub(crate) struct CudaDecoder {
    decoder: Mutex<CudaBatchDecoder>,
    context: CudaContext,
    ordinal: usize,
}

impl CudaDecoder {
    pub fn new(device: &str, options: BatchDecodeOptions) -> PyResult<Self> {
        let ordinal = if device == "cuda" {
            0
        } else {
            device
                .strip_prefix("cuda:")
                .and_then(|n| n.parse::<usize>().ok())
                .ok_or_else(|| PyValueError::new_err("device must be cpu, cuda, or cuda:N"))?
        };
        let context = CudaContext::retain_primary(ordinal).map_err(value_error)?;
        let decoder = CudaBatchDecoder::with_session_and_options(
            CudaSession::with_context(context.clone()),
            options,
        );
        Ok(Self {
            decoder: Mutex::new(decoder),
            context,
            ordinal,
        })
    }

    pub fn decode(&self, py: Python<'_>, prepared: &PreparedBatch) -> PyResult<Batch> {
        let mut groups = Vec::new();
        let mut group_errors = Vec::new();
        for group in prepared.groups() {
            match self.decode_group(py, group) {
                Ok(output) => groups.push(output.into_python(py)?),
                Err(error) => group_errors.push((
                    owned_items(group.source_indices().iter().copied())?,
                    error.to_string(),
                )),
            }
        }
        Ok(Batch {
            groups,
            errors: prepared
                .errors()
                .iter()
                .map(|e| (e.index, e.source.to_string()))
                .collect(),
            group_errors,
        })
    }

    pub fn decode_one(
        &self,
        py: Python<'_>,
        bytes: &[u8],
        request: DecodeRequest,
    ) -> PyResult<Tensor> {
        let input = EncodedImage::new(Arc::from(bytes), request);
        let prepared = py.detach(|| {
            self.decoder
                .lock()
                .map_err(|_| PyRuntimeError::new_err("CUDA decoder is poisoned"))?
                .prepare(owned_items([input].into_iter())?)
                .map_err(value_error)
        })?;
        if let Some(error) = prepared.errors().first() {
            return Err(value_error(&error.source));
        }
        if prepared.groups().len() != 1 {
            return Err(PyRuntimeError::new_err(
                "single CUDA input did not produce one group",
            ));
        }
        self.decode_group_inner(py, &prepared.groups()[0], true)
            .map(|group| group.tensor)
    }

    fn decode_group(
        &self,
        py: Python<'_>,
        group: &j2k::PreparedBatchGroup,
    ) -> PyResult<MaterializedGroup> {
        self.decode_group_inner(py, group, false)
    }

    fn decode_group_inner(
        &self,
        py: Python<'_>,
        group: &j2k::PreparedBatchGroup,
        squeeze: bool,
    ) -> PyResult<MaterializedGroup> {
        let mut dimensions = shape(group.info(), group.images().len())?;
        let dtype = match group.info().sample_type {
            NativeSampleType::U8 => DType::U8,
            NativeSampleType::U16 => DType::U16,
            NativeSampleType::I16 => DType::I16,
            _ => return Err(PyValueError::new_err("unsupported CUDA tensor dtype")),
        };
        let byte_len = dimensions
            .iter()
            .try_fold(usize::from(dtype.bits / 8), |n, &d| n.checked_mul(d))
            .ok_or_else(|| PyValueError::new_err("CUDA tensor size overflow"))?;
        let torch = py.import("torch")?;
        let device = format!("cuda:{}", self.ordinal);
        let kwargs = PyDict::new(py);
        kwargs.set_item("dtype", torch.getattr(dtype.name)?)?;
        kwargs.set_item("device", &device)?;
        let mut tensor = torch
            .call_method("empty", (&dimensions,), Some(&kwargs))?
            .unbind();
        let ptr: u64 = tensor.bind(py).call_method0("data_ptr")?.extract()?;
        let mut stream = torch
            .getattr("cuda")?
            .call_method1("current_stream", (&device,))?
            .unbind();
        let stream_ptr: u64 = stream.bind(py).getattr("cuda_stream")?.extract()?;
        let result = py.detach(|| {
            let mut decoder = self
                .decoder
                .lock()
                .map_err(|_| PyRuntimeError::new_err("CUDA decoder is poisoned"))?;
            let mut work = || {
                // SAFETY: tensor is a fresh contiguous allocation on this
                // retained primary context. It has not been published to the
                // caller. The owning Py reference stays live across submission
                // and completion, and the caller's allocation stream is ordered
                // before constructing this view below.
                let mut destination = unsafe {
                    CudaExternalDeviceBufferViewMut::from_raw_parts(
                        &self.context,
                        ptr,
                        byte_len,
                        usize::from(dtype.bits / 8),
                        &mut tensor,
                    )
                }
                .map_err(value_error)?;
                // SAFETY: the unique unpublished allocation is retained until
                // successful completion. Every error is synchronized below;
                // uncertain completion quarantines the owner instead of freeing it.
                unsafe { decoder.decode_batch_into(group, &mut destination) }.map_err(value_error)
            };
            if stream_ptr == 0 {
                // Both torch's legacy default stream and the codec use stream
                // zero in this primary context, so allocation already precedes decode.
                work()
            } else {
                // SAFETY: this PyTorch stream belongs to the selected device's
                // primary context and is retained for the entire ordered call.
                // work waits for decode and status validation before returning.
                unsafe {
                    self.context
                        .with_primary_stream_ordering(stream_ptr, &mut stream, work)
                }
                .map_err(value_error)?
            }
        });
        let metadata = match result {
            Ok(metadata) => metadata,
            Err(error) => {
                let completed = py.detach(|| {
                    self.context
                        .synchronize_for_resource_release()
                        .completion_established()
                });
                if !completed {
                    // CUDA did not establish that kernels stopped using this
                    // external allocation. Preserve its framework owner and
                    // stream indefinitely rather than returning it to the pool.
                    std::mem::forget(tensor);
                    std::mem::forget(stream);
                }
                return Err(error);
            }
        };
        if squeeze {
            tensor = tensor.bind(py).call_method1("squeeze", (0,))?.unbind();
            dimensions.remove(0);
        }
        Ok(MaterializedGroup {
            info: metadata.info().clone(),
            indices: owned_items(metadata.source_indices().iter().copied())?,
            rects: owned_items(metadata.decoded_rects().iter().copied())?,
            warnings: metadata
                .warnings()
                .iter()
                .map(|v| v.iter().map(ToString::to_string).collect())
                .collect(),
            tensor: Tensor::new(
                Samples::Torch(tensor),
                dimensions,
                dtype,
                (2, i32::try_from(self.ordinal).map_err(value_error)?),
            )?,
        })
    }
}
