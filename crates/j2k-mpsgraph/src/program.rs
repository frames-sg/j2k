// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::{BatchGroupInfo, J2kDecodeWarning, PreparedBatchGroup};
use j2k_core::Rect;
use j2k_metal::SubmittedMetalGroupDecodeInto;
use j2k_metal_support::{MetalImageDestination, MetalImageLayout};
use j2k_mpsgraph_support::{GraphExecutionError, MpsGraphSubmission};
use objc2::{rc::Retained, runtime::ProtocolObject, AnyThread};
use objc2_foundation::{NSArray, NSNumber};
use objc2_metal::{MTLBuffer, MTLCommandQueue, MTLDevice};
use objc2_metal_performance_shaders_graph::{MPSGraph, MPSGraphTensor, MPSGraphTensorData};

use crate::{
    allocation::{try_clone_slice, try_vec},
    platform::MpsGraphBatchDecoder,
    Error, MpsGraphInputGroup, MpsGraphTensorSpec,
};

/// A static rank-four `MPSGraph` program with one image placeholder.
pub struct MpsGraphProgram {
    graph: Retained<MPSGraph>,
    image_placeholder: Retained<MPSGraphTensor>,
    targets: Vec<Retained<MPSGraphTensor>>,
    input_spec: MpsGraphTensorSpec,
}

impl core::fmt::Debug for MpsGraphProgram {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("MpsGraphProgram")
            .field("input_spec", &self.input_spec)
            .field("target_count", &self.targets.len())
            .finish_non_exhaustive()
    }
}

impl MpsGraphProgram {
    /// Adopt a graph, its sole runtime image placeholder, and output targets.
    ///
    /// Other model inputs must already be constants embedded in `graph`.
    pub fn new(
        graph: Retained<MPSGraph>,
        image_placeholder: Retained<MPSGraphTensor>,
        targets: Vec<Retained<MPSGraphTensor>>,
        input_spec: MpsGraphTensorSpec,
    ) -> Result<Self, Error> {
        if targets.is_empty() {
            return Err(Error::InvalidTensorContract {
                reason: "MPSGraph program requires at least one target tensor",
            });
        }
        validate_placeholder(&image_placeholder, input_spec)?;
        Ok(Self {
            graph,
            image_placeholder,
            targets,
            input_spec,
        })
    }

    /// Static image input contract captured by this graph.
    #[must_use]
    pub const fn input_spec(&self) -> MpsGraphTensorSpec {
        self.input_spec
    }

    /// Borrow the owned graph for expert composition and benchmark tooling.
    #[must_use]
    pub fn graph(&self) -> &MPSGraph {
        &self.graph
    }

    /// Borrow the sole runtime image placeholder.
    #[must_use]
    pub fn image_placeholder(&self) -> &MPSGraphTensor {
        &self.image_placeholder
    }

    /// Borrow target tensors in result order.
    #[must_use]
    pub fn targets(&self) -> &[Retained<MPSGraphTensor>] {
        &self.targets
    }

    /// Reject a group whose rank-four shape or native dtype differs.
    pub fn validate_input_spec(&self, actual: MpsGraphTensorSpec) -> Result<(), Error> {
        if actual != self.input_spec {
            return Err(Error::InvalidTensorContract {
                reason: "MPSGraph image placeholder shape or dtype does not match the batch",
            });
        }
        Ok(())
    }

    /// Submit graph execution for a completed resident input group.
    pub fn submit_completed(
        &self,
        command_queue: &ProtocolObject<dyn MTLCommandQueue>,
        input: MpsGraphInputGroup,
    ) -> Result<SubmittedMpsGraphRun, Error> {
        self.validate_input_spec(input.spec())?;
        validate_device_registry_ids(
            input.resident_batch().device_registry_id(),
            command_queue.device().registryID(),
        )?;
        let MpsGraphInputGroup {
            tensor_data,
            resident_batch,
            spec: _,
            info,
            source_indices,
            decoded_rects,
            warnings,
        } = input;
        let metadata = RunMetadata {
            info,
            source_indices,
            completed: Some((decoded_rects, warnings)),
        };
        Ok(self.submit_graph(
            command_queue,
            &tensor_data,
            RunInputOwner::Completed {
                tensor_data: tensor_data.clone(),
                resident_batch,
            },
            None,
            metadata,
        ))
    }

    /// Submit a direct codec decode and graph run on the decoder's shared queue.
    pub(crate) fn submit_prepared_group(
        &self,
        decoder: &mut MpsGraphBatchDecoder,
        group: &PreparedBatchGroup,
    ) -> Result<SubmittedMpsGraphRun, Error> {
        let spec = MpsGraphTensorSpec::from_group_info(group.info(), group.images().len())?;
        self.validate_input_spec(spec)?;
        let pixel_format =
            group
                .info()
                .native_pixel_format()
                .ok_or(Error::InvalidTensorContract {
                    reason: "prepared group has no supported native Metal pixel format",
                })?;
        let width =
            usize::try_from(group.info().dimensions.0).map_err(|_| Error::TensorShapeOverflow)?;
        let height =
            usize::try_from(group.info().dimensions.1).map_err(|_| Error::TensorShapeOverflow)?;
        let row_bytes = width
            .checked_mul(pixel_format.bytes_per_pixel())
            .ok_or(Error::TensorShapeOverflow)?;
        let image_bytes = row_bytes
            .checked_mul(height)
            .ok_or(Error::TensorShapeOverflow)?;
        let layout = MetalImageLayout::new_batch(
            0,
            group.info().dimensions,
            row_bytes,
            pixel_format,
            group.images().len(),
            image_bytes,
        )?;
        let buffer = j2k_metal_support::checked_private_buffer(&decoder.device, layout.byte_len())?;
        // SAFETY: this crate owns every handle to the fresh allocation. The
        // codec submission owns the exclusive destination, and the graph read
        // is ordered after it on `decoder.queue` before this function returns.
        let destination =
            unsafe { MetalImageDestination::from_exclusive_buffer(buffer.clone(), layout)? };
        let codec = decoder
            .decoder
            .submit_prepared_group_into_for_consumer_queue(group, destination, &decoder.queue)?;
        let tensor_data = tensor_data_from_buffer(&buffer, spec);
        let metadata = RunMetadata {
            info: group.info().clone(),
            source_indices: try_clone_slice(
                group.source_indices(),
                "direct MPSGraph source indices",
            )?,
            completed: None,
        };
        Ok(self.submit_graph(
            &decoder.queue,
            &tensor_data,
            RunInputOwner::Direct {
                tensor_data: tensor_data.clone(),
                buffer,
            },
            Some(codec),
            metadata,
        ))
    }

    fn submit_graph(
        &self,
        command_queue: &ProtocolObject<dyn MTLCommandQueue>,
        tensor_data: &MPSGraphTensorData,
        input_owner: RunInputOwner,
        codec: Option<SubmittedMetalGroupDecodeInto>,
        metadata: RunMetadata,
    ) -> SubmittedMpsGraphRun {
        // SAFETY: the program's image shape/dtype and queue device were checked
        // before submission. The codec writes on this same queue. RunInputOwner
        // retains tensor data and its buffer or resident batch (including pool
        // leases), and the shared guard holds them through graph completion.
        let graph = unsafe {
            MpsGraphSubmission::submit(
                &self.graph,
                &self.image_placeholder,
                &self.targets,
                command_queue,
                tensor_data,
                input_owner,
            )
        };
        SubmittedMpsGraphRun {
            graph,
            codec,
            metadata: Some(metadata),
        }
    }
}

fn validate_device_registry_ids(
    image_registry_id: u64,
    requested_registry_id: u64,
) -> Result<(), Error> {
    if image_registry_id == requested_registry_id {
        return Ok(());
    }
    Err(
        j2k_metal_support::MetalSupportError::MetalImageDeviceMismatch {
            image_registry_id,
            requested_registry_id,
        }
        .into(),
    )
}

fn validate_placeholder(
    placeholder: &MPSGraphTensor,
    expected: MpsGraphTensorSpec,
) -> Result<(), Error> {
    // SAFETY: immutable graph tensor metadata remains valid while the retained
    // placeholder and its graph are alive.
    let shape = unsafe { placeholder.shape() }.ok_or(Error::InvalidTensorContract {
        reason: "MPSGraph image placeholder must have a static rank-four shape",
    })?;
    if shape.len() != 4 {
        return Err(Error::InvalidTensorContract {
            reason: "MPSGraph image placeholder must have rank four",
        });
    }
    let actual = core::array::from_fn(|index| shape.objectAtIndex(index).as_usize());
    if actual != expected.shape() {
        return Err(Error::InvalidTensorContract {
            reason: "MPSGraph image placeholder static shape does not match its contract",
        });
    }
    // SAFETY: immutable graph tensor metadata is valid for the retained tensor.
    if unsafe { placeholder.dataType() } != expected.mps_data_type() {
        return Err(Error::InvalidTensorContract {
            reason: "MPSGraph image placeholder dtype does not match its contract",
        });
    }
    Ok(())
}

fn mps_shape(shape: [usize; 4]) -> Retained<NSArray<NSNumber>> {
    let dimensions = shape.map(NSNumber::new_usize);
    NSArray::from_retained_slice(&dimensions)
}

fn tensor_data_from_buffer(
    buffer: &ProtocolObject<dyn MTLBuffer>,
    spec: MpsGraphTensorSpec,
) -> Retained<MPSGraphTensorData> {
    let shape = mps_shape(spec.shape());
    // SAFETY: the freshly allocated buffer has exactly `spec.byte_len()`
    // bytes and is retained by the run guard. Queue ordering prevents MPSGraph
    // reads until the codec's exclusive write has completed.
    unsafe {
        MPSGraphTensorData::initWithMTLBuffer_shape_dataType(
            MPSGraphTensorData::alloc(),
            buffer,
            &shape,
            spec.mps_data_type(),
        )
    }
}

struct RunMetadata {
    info: BatchGroupInfo,
    source_indices: Vec<usize>,
    completed: Option<(Vec<Rect>, Vec<Vec<J2kDecodeWarning>>)>,
}

#[expect(
    dead_code,
    reason = "variant fields are ownership guards released together after graph completion"
)]
enum RunInputOwner {
    // Tensor data is declared first so it drops before its unretained buffer owner.
    Completed {
        tensor_data: Retained<MPSGraphTensorData>,
        resident_batch: j2k_metal::MetalResidentBatch,
    },
    Direct {
        tensor_data: Retained<MPSGraphTensorData>,
        // Tensor data may not retain this allocation.
        buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    },
}

/// Completed graph outputs and the codec metadata for their input batch.
pub struct MpsGraphRunOutput {
    results: Vec<Retained<MPSGraphTensorData>>,
    info: BatchGroupInfo,
    source_indices: Vec<usize>,
    decoded_rects: Vec<Rect>,
    warnings: Vec<Vec<J2kDecodeWarning>>,
}

impl core::fmt::Debug for MpsGraphRunOutput {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("MpsGraphRunOutput")
            .field("result_count", &self.results.len())
            .field("info", &self.info)
            .field("source_indices", &self.source_indices)
            .field("decoded_rects", &self.decoded_rects)
            .field("warnings", &self.warnings)
            .finish_non_exhaustive()
    }
}

impl MpsGraphRunOutput {
    /// Target tensor data in the program's target order.
    #[must_use]
    pub fn results(&self) -> &[Retained<MPSGraphTensorData>] {
        &self.results
    }

    /// Native codec group metadata.
    #[must_use]
    pub const fn info(&self) -> &BatchGroupInfo {
        &self.info
    }

    /// Original source positions in graph batch order.
    #[must_use]
    pub fn source_indices(&self) -> &[usize] {
        &self.source_indices
    }

    /// Actual decoded rectangles in graph batch order.
    #[must_use]
    pub fn decoded_rects(&self) -> &[Rect] {
        &self.decoded_rects
    }

    /// Non-fatal codec warnings in graph batch order.
    #[must_use]
    pub fn warnings(&self) -> &[Vec<J2kDecodeWarning>] {
        &self.warnings
    }
}

/// In-flight direct decode and `MPSGraph` execution.
///
/// This guard is deliberately neither `Send` nor `Sync`. Dropping it waits for
/// graph completion before releasing the unretained input allocation.
pub struct SubmittedMpsGraphRun {
    graph: MpsGraphSubmission<RunInputOwner>,
    codec: Option<SubmittedMetalGroupDecodeInto>,
    metadata: Option<RunMetadata>,
}

impl core::fmt::Debug for SubmittedMpsGraphRun {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("SubmittedMpsGraphRun")
            .field("complete", &self.is_complete())
            .finish_non_exhaustive()
    }
}

impl SubmittedMpsGraphRun {
    /// Whether `MPSGraph` has invoked its completion callback.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.graph.is_complete()
    }

    /// Wait for codec and graph completion and return graph target data.
    pub fn wait(mut self) -> Result<MpsGraphRunOutput, Error> {
        self.finish()
    }

    fn finish(&mut self) -> Result<MpsGraphRunOutput, Error> {
        let graph_error = self.graph.wait().err();
        let mut metadata = self
            .metadata
            .take()
            .expect("MPSGraph run metadata is consumed exactly once");
        let completion = self
            .codec
            .take()
            .map(SubmittedMetalGroupDecodeInto::wait)
            .transpose()?;
        if let Some(error) = graph_error {
            return Err(graph_execution_error(error));
        }
        if let Some(completion) = completion {
            metadata.completed = Some(completion.into_parts());
        }
        let mut results = try_vec(self.graph.target_count(), "MPSGraph run outputs")?;
        for index in 0..self.graph.target_count() {
            let result = self
                .graph
                .output(index)
                .map_err(graph_execution_error)?
                .ok_or(Error::MissingGraphOutput { index })?;
            results.push(result);
        }
        let (decoded_rects, warnings) = metadata
            .completed
            .take()
            .expect("completed codec metadata exists after graph completion");
        Ok(MpsGraphRunOutput {
            results,
            info: metadata.info,
            source_indices: metadata.source_indices,
            decoded_rects,
            warnings,
        })
    }
}

impl Drop for SubmittedMpsGraphRun {
    fn drop(&mut self) {
        // Cleanup only waits: metadata/result extraction belongs to explicit wait.
        // This is also safe after codec failure consumed part of the run state.
        let _ = self.graph.wait();
        if let Some(codec) = self.codec.take() {
            let _ = codec.wait();
        }
    }
}

fn graph_execution_error(error: GraphExecutionError) -> Error {
    Error::GraphExecution {
        domain: error.domain,
        code: error.code,
        description: error.description,
    }
}

#[cfg(test)]
mod tests {
    use super::validate_device_registry_ids;
    use crate::Error;

    #[test]
    fn completed_handoff_rejects_a_command_queue_from_another_device() {
        let error = validate_device_registry_ids(17, 29)
            .expect_err("a completed buffer cannot run on another Metal device");

        assert!(matches!(
            error,
            Error::MetalRuntime(
                j2k_metal_support::MetalSupportError::MetalImageDeviceMismatch {
                    image_registry_id: 17,
                    requested_registry_id: 29,
                }
            )
        ));
    }
}
