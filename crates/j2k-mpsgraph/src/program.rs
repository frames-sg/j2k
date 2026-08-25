// SPDX-License-Identifier: MIT OR Apache-2.0

use core::{marker::PhantomData, ptr::NonNull};
use std::{
    rc::Rc,
    sync::{Arc, OnceLock},
};

use block2::RcBlock;
use j2k::{BatchGroupInfo, J2kDecodeWarning, PreparedBatchGroup};
use j2k_core::Rect;
use j2k_metal::SubmittedMetalGroupDecodeInto;
use j2k_metal_support::{MetalImageDestination, MetalImageLayout};
use objc2::{rc::Retained, runtime::ProtocolObject, AnyThread};
use objc2_foundation::{NSArray, NSDictionary, NSError, NSNumber};
use objc2_metal::{MTLBuffer, MTLCommandQueue};
use objc2_metal_performance_shaders::MPSDataType;
use objc2_metal_performance_shaders_graph::{
    MPSGraph, MPSGraphExecutionDescriptor, MPSGraphTensor, MPSGraphTensorData,
    MPSGraphTensorDataDictionary,
};

use crate::{
    allocation::{try_clone_slice, try_single, try_vec},
    platform::MpsGraphBatchDecoder,
    Error, MpsGraphInputGroup, MpsGraphTensorSpec,
};

type CompletionBlock = RcBlock<dyn Fn(NonNull<MPSGraphTensorDataDictionary>, *mut NSError)>;

#[derive(Clone, Debug)]
struct OwnedGraphError {
    domain: String,
    code: isize,
    description: String,
}

type CompletionState = OnceLock<Result<(), OwnedGraphError>>;

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

    /// Build an identity graph for interoperability tests and direct handoff.
    pub fn identity(input_spec: MpsGraphTensorSpec) -> Result<Self, Error> {
        // SAFETY: `new` is a standard owning Objective-C constructor.
        let graph = unsafe { MPSGraph::new() };
        let shape = mps_shape(input_spec.shape());
        // SAFETY: shape and dtype are static validated values retained by the graph.
        let placeholder = unsafe {
            graph.placeholderWithShape_dataType_name(Some(&shape), input_spec.mps_data_type(), None)
        };
        let targets = try_single(placeholder.clone(), "MPSGraph identity target")?;
        Self::new(graph, placeholder, targets, input_spec)
    }

    /// Build the RGB8/NHWC reference graph used by examples and benchmarks.
    ///
    /// The graph casts to F32, normalizes to `[0, 1]`, applies the fixed
    /// reference channel weights, and returns one spatially reduced score per
    /// image.
    pub fn rgb8_nhwc_reference(batch: usize, height: usize, width: usize) -> Result<Self, Error> {
        let input_spec =
            MpsGraphTensorSpec::new([batch, height, width, 3], crate::MpsGraphElementType::U8)?;
        let spatial_pixels = height
            .checked_mul(width)
            .and_then(|pixels| u32::try_from(pixels).ok())
            .ok_or(Error::TensorShapeOverflow)?;
        // SAFETY: `new` is a standard owning Objective-C constructor.
        let graph = unsafe { MPSGraph::new() };
        let shape = mps_shape(input_spec.shape());
        // SAFETY: all operations use static shapes, valid axes, and constants;
        // every returned tensor remains owned by `graph` and the program.
        let (placeholder, score) = unsafe {
            let placeholder =
                graph.placeholderWithShape_dataType_name(Some(&shape), MPSDataType::UInt8, None);
            let float = graph.castTensor_toType_name(&placeholder, MPSDataType::Float32, None);
            let scale = graph.constantWithScalar_dataType(255.0, MPSDataType::Float32);
            let normalized =
                graph.divisionWithPrimaryTensor_secondaryTensor_name(&float, &scale, None);

            let weighted_channel = |channel: isize, weight: f64| {
                let values =
                    graph.sliceTensor_dimension_start_length_name(&normalized, 3, channel, 1, None);
                let coefficient = graph.constantWithScalar_dataType(weight, MPSDataType::Float32);
                graph.multiplicationWithPrimaryTensor_secondaryTensor_name(
                    &values,
                    &coefficient,
                    None,
                )
            };
            let red = weighted_channel(0, f64::from(crate::RGB8_REFERENCE_CHANNEL_WEIGHTS[0]));
            let green = weighted_channel(1, f64::from(crate::RGB8_REFERENCE_CHANNEL_WEIGHTS[1]));
            let blue = weighted_channel(2, f64::from(crate::RGB8_REFERENCE_CHANNEL_WEIGHTS[2]));
            let red_green =
                graph.additionWithPrimaryTensor_secondaryTensor_name(&red, &green, None);
            let weighted =
                graph.additionWithPrimaryTensor_secondaryTensor_name(&red_green, &blue, None);
            let axes = NSArray::from_retained_slice(&[
                NSNumber::new_isize(1),
                NSNumber::new_isize(2),
                NSNumber::new_isize(3),
            ]);
            let summed = graph.reductionSumWithTensor_axes_name(&weighted, Some(&axes), None);
            let pixel_count =
                graph.constantWithScalar_dataType(f64::from(spatial_pixels), MPSDataType::Float32);
            let score =
                graph.divisionWithPrimaryTensor_secondaryTensor_name(&summed, &pixel_count, None);
            (placeholder, score)
        };
        Self::new(
            graph,
            placeholder,
            try_single(score, "MPSGraph reference target")?,
            input_spec,
        )
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
        let feeds = NSDictionary::from_slices(&[&*self.image_placeholder], &[tensor_data]);
        let targets = NSArray::from_retained_slice(&self.targets);
        // SAFETY: `new` is a standard owning Objective-C constructor.
        let execution_descriptor = unsafe { MPSGraphExecutionDescriptor::new() };
        let completion_state = Arc::new(CompletionState::default());
        let callback_state = Arc::clone(&completion_state);
        let completion_block: CompletionBlock = RcBlock::new(
            move |_results: NonNull<MPSGraphTensorDataDictionary>, error: *mut NSError| {
                let error = NonNull::new(error).map(|error| {
                    // SAFETY: MPSGraph guarantees that the callback NSError is
                    // valid for the duration of this invocation.
                    let error = unsafe { error.as_ref() };
                    OwnedGraphError {
                        domain: error.domain().to_string(),
                        code: error.code(),
                        description: error.localizedDescription().to_string(),
                    }
                });
                let _ = callback_state.set(error.map_or(Ok(()), Err));
            },
        );
        // SAFETY: the block pointer has the exact generated completion
        // signature. Both the descriptor and this guard retain the block until
        // completion, and Drop waits before releasing either owner.
        unsafe {
            execution_descriptor.setCompletionHandler(RcBlock::as_ptr(&completion_block));
        }
        // SAFETY: the placeholder, feed, targets, queue, descriptor, graph,
        // tensor data, and unretained underlying input buffer are all retained
        // by the returned guard until its completion callback has fired.
        let results = unsafe {
            self.graph
                .runAsyncWithMTLCommandQueue_feeds_targetTensors_targetOperations_executionDescriptor(
                    command_queue,
                    &feeds,
                    &targets,
                    None,
                    Some(&execution_descriptor),
                )
        };
        SubmittedMpsGraphRun {
            graph: self.graph.clone(),
            image_placeholder: self.image_placeholder.clone(),
            targets,
            feeds,
            results: Some(results),
            execution_descriptor,
            completion_block,
            completion_state,
            input_owner: Some(input_owner),
            codec,
            metadata: Some(metadata),
            not_send_or_sync: PhantomData,
        }
    }
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
    graph: Retained<MPSGraph>,
    image_placeholder: Retained<MPSGraphTensor>,
    targets: Retained<NSArray<MPSGraphTensor>>,
    feeds: Retained<MPSGraphTensorDataDictionary>,
    results: Option<Retained<MPSGraphTensorDataDictionary>>,
    execution_descriptor: Retained<MPSGraphExecutionDescriptor>,
    completion_block: CompletionBlock,
    completion_state: Arc<CompletionState>,
    input_owner: Option<RunInputOwner>,
    codec: Option<SubmittedMetalGroupDecodeInto>,
    metadata: Option<RunMetadata>,
    not_send_or_sync: PhantomData<Rc<()>>,
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
        self.completion_state.get().is_some()
    }

    /// Wait for codec and graph completion and return graph target data.
    pub fn wait(mut self) -> Result<MpsGraphRunOutput, Error> {
        self.finish()
    }

    fn finish(&mut self) -> Result<MpsGraphRunOutput, Error> {
        let graph_error = self.completion_state.wait().clone().err();
        let mut metadata = self
            .metadata
            .take()
            .expect("MPSGraph run metadata is consumed exactly once");
        let completion = self
            .codec
            .take()
            .map(SubmittedMetalGroupDecodeInto::wait)
            .transpose()?;
        let input_owner = self
            .input_owner
            .take()
            .expect("MPSGraph run input is consumed exactly once");
        drop(input_owner);
        if let Some(error) = graph_error {
            return Err(Error::GraphExecution {
                domain: error.domain,
                code: error.code,
                description: error.description,
            });
        }
        if let Some(completion) = completion {
            metadata.completed = Some(completion.into_parts());
        }
        let results_dictionary = self
            .results
            .take()
            .expect("MPSGraph results are consumed exactly once");
        let mut results = try_vec(self.targets.len(), "MPSGraph run outputs")?;
        for (index, target) in self.targets.iter().enumerate() {
            let result = results_dictionary
                .objectForKey(&target)
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
        if self.input_owner.is_some() {
            let _ = self.finish();
        }
        // Make the lifetime contract visible and prevent accidental removal of
        // these owners as apparently unused fields.
        let _ = (
            &self.graph,
            &self.image_placeholder,
            &self.feeds,
            &self.execution_descriptor,
            &self.completion_block,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::{CompletionState, OwnedGraphError};

    #[test]
    fn completion_state_preserves_graph_errors() {
        let state = CompletionState::new();
        assert!(state.get().is_none());

        state
            .set(Err(OwnedGraphError {
                domain: "test.domain".to_string(),
                code: 17,
                description: "test failure".to_string(),
            }))
            .expect("first completion");

        assert!(state.get().is_some());
        let error = state.wait().clone().expect_err("owned graph error");
        assert_eq!(error.domain, "test.domain");
        assert_eq!(error.code, 17);
        assert_eq!(error.description, "test failure");
    }
}
