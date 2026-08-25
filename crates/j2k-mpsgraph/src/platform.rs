// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::{
    BatchDecodeOptions, BatchGroupInfo, EncodedImage, IndexedBatchError, J2kDecodeWarning,
    PreparedBatch, PreparedBatchGroup, PreparedImage,
};
use j2k_core::Rect;
use j2k_metal::{
    MetalBackendSession, MetalBatchDecodeResult, MetalBatchDecoder, MetalBatchGroup,
    MetalBatchGroupError, MetalResidentBatch,
};
use objc2::{rc::Retained, runtime::ProtocolObject, AnyThread};
use objc2_foundation::{NSArray, NSNumber};
use objc2_metal::{MTLCommandQueue, MTLDevice};
use objc2_metal_performance_shaders::MPSDataType;
use objc2_metal_performance_shaders_graph::MPSGraphTensorData;

use crate::{Error, MpsGraphElementType, MpsGraphTensorSpec};
use crate::{MpsGraphProgram, MpsGraphRunOutput, SubmittedMpsGraphRun};

type Device = Retained<ProtocolObject<dyn MTLDevice>>;
type CommandQueue = Retained<ProtocolObject<dyn MTLCommandQueue>>;

/// Persistent decoder whose codec and `MPSGraph` work share one Metal queue.
pub struct MpsGraphBatchDecoder {
    pub(super) decoder: MetalBatchDecoder,
    pub(super) device: Device,
    pub(super) queue: CommandQueue,
}

impl core::fmt::Debug for MpsGraphBatchDecoder {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("MpsGraphBatchDecoder")
            .field("decoder", &self.decoder)
            .finish_non_exhaustive()
    }
}

impl MpsGraphBatchDecoder {
    /// Create a decoder on the system default Apple Silicon Metal device.
    pub fn system_default(options: BatchDecodeOptions) -> Result<Self, Error> {
        let device = j2k_metal_support::system_default_device()?;
        let queue = j2k_metal_support::checked_command_queue(&device)?;
        Self::with_device_and_queue(device, queue, options)
    }

    /// Create a decoder that shares an existing Metal command queue.
    pub fn with_device_and_queue(
        device: Device,
        queue: CommandQueue,
        options: BatchDecodeOptions,
    ) -> Result<Self, Error> {
        let backend = MetalBackendSession::with_command_queue(device.clone(), queue.clone())?;
        let decoder = MetalBatchDecoder::with_backend_session_and_options(backend, options);
        Ok(Self {
            decoder,
            device,
            queue,
        })
    }

    /// Parse, validate, and group encoded inputs for reuse.
    pub fn prepare(&self, inputs: Vec<EncodedImage>) -> Result<PreparedBatch, Error> {
        self.decoder.prepare(inputs).map_err(Error::from)
    }

    /// Regroup already prepared images under this decoder's batch policy.
    pub fn prepare_prepared_images(
        &self,
        images: Vec<PreparedImage>,
    ) -> Result<PreparedBatch, Error> {
        self.decoder
            .prepare_prepared_images(images)
            .map_err(Error::from)
    }

    /// Decode encoded inputs and wrap completed resident groups as `MPSGraph` inputs.
    pub fn decode(&mut self, inputs: Vec<EncodedImage>) -> Result<MpsGraphBatchDecode, Error> {
        let decoded = self.decoder.decode_batch(inputs)?;
        MpsGraphBatchDecode::from_metal(decoded)
    }

    /// Decode a reusable prepared batch and wrap its completed resident groups.
    pub fn decode_prepared(
        &mut self,
        prepared: &PreparedBatch,
    ) -> Result<MpsGraphBatchDecode, Error> {
        let decoded = self.decoder.decode_prepared(prepared)?;
        MpsGraphBatchDecode::from_metal(decoded)
    }

    /// Regroup and decode caller-owned prepared images without reparsing them.
    pub fn decode_prepared_images(
        &mut self,
        images: Vec<PreparedImage>,
    ) -> Result<MpsGraphBatchDecode, Error> {
        let decoded = self.decoder.decode_prepared_images(images)?;
        MpsGraphBatchDecode::from_metal(decoded)
    }

    /// Submit direct decode and graph execution on the shared queue.
    pub fn submit_prepared_group(
        &mut self,
        program: &MpsGraphProgram,
        group: &PreparedBatchGroup,
    ) -> Result<SubmittedMpsGraphRun, Error> {
        program.submit_prepared_group(self, group)
    }

    /// Submit direct decode and graph execution, then wait for completion.
    pub fn run_prepared_group(
        &mut self,
        program: &MpsGraphProgram,
        group: &PreparedBatchGroup,
    ) -> Result<MpsGraphRunOutput, Error> {
        self.submit_prepared_group(program, group)?.wait()
    }

    /// Number of codec group submissions made by this retained session.
    pub fn submissions(&self) -> Result<u64, Error> {
        self.decoder.submissions().map_err(Error::from)
    }

    /// Metal device retained by this decoder.
    #[must_use]
    pub fn device(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }

    /// Shared codec/MPSGraph command queue retained by this decoder.
    #[must_use]
    pub fn command_queue(&self) -> &ProtocolObject<dyn MTLCommandQueue> {
        &self.queue
    }
}

/// One completed codec-owned allocation wrapped as `MPSGraph` tensor data.
pub struct MpsGraphInputGroup {
    // Drop tensor data before the resident owner because MPSGraph may not
    // retain the MTLBuffer passed to its initializer.
    pub(super) tensor_data: Retained<MPSGraphTensorData>,
    pub(super) resident_batch: MetalResidentBatch,
    pub(super) spec: MpsGraphTensorSpec,
    pub(super) info: BatchGroupInfo,
    pub(super) source_indices: Vec<usize>,
    pub(super) decoded_rects: Vec<Rect>,
    pub(super) warnings: Vec<Vec<J2kDecodeWarning>>,
}

impl core::fmt::Debug for MpsGraphInputGroup {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("MpsGraphInputGroup")
            .field("spec", &self.spec)
            .field("info", &self.info)
            .field("source_indices", &self.source_indices)
            .field("decoded_rects", &self.decoded_rects)
            .field("warnings", &self.warnings)
            .field("resident_batch", &self.resident_batch)
            .finish_non_exhaustive()
    }
}

impl MpsGraphInputGroup {
    fn from_metal(group: MetalBatchGroup) -> Result<Self, Error> {
        let resident_batch =
            group
                .resident_batch()
                .cloned()
                .ok_or(Error::InvalidTensorContract {
                    reason: "Metal batch group did not contain resident storage",
                })?;
        let (info, source_indices, decoded_rects, warnings, _surfaces) = group.into_parts();
        if resident_batch.byte_offset() != 0 {
            return Err(Error::InvalidTensorContract {
                reason: "MPSGraph v1 requires zero-offset dense Metal batches",
            });
        }
        let spec = MpsGraphTensorSpec::from_group_info(&info, resident_batch.image_count())?;
        let expected_len = spec.byte_len()?;
        if resident_batch.byte_len() != expected_len {
            return Err(Error::InvalidTensorContract {
                reason: "resident Metal batch length does not match its tensor contract",
            });
        }

        let dimensions = spec.shape().map(NSNumber::new_usize);
        let shape = NSArray::from_retained_slice(&dimensions);
        // SAFETY: the codec submission has completed; `spec` proves the dense,
        // zero-offset buffer has the supplied rank-four shape, element type,
        // and byte length. This owner retains `resident_batch` until after
        // `tensor_data` is dropped, and exposes the allocation only read-only.
        let tensor_data = unsafe {
            MPSGraphTensorData::initWithMTLBuffer_shape_dataType(
                MPSGraphTensorData::alloc(),
                resident_batch.metal_buffer(),
                &shape,
                spec.mps_data_type(),
            )
        };
        Ok(Self {
            tensor_data,
            resident_batch,
            spec,
            info,
            source_indices,
            decoded_rects,
            warnings,
        })
    }

    /// `MPSGraph` tensor data that aliases the completed codec allocation.
    #[must_use]
    pub fn tensor_data(&self) -> &MPSGraphTensorData {
        &self.tensor_data
    }

    /// Validated tensor shape and element type.
    #[must_use]
    pub const fn spec(&self) -> MpsGraphTensorSpec {
        self.spec
    }

    /// Native codec group metadata.
    #[must_use]
    pub const fn info(&self) -> &BatchGroupInfo {
        &self.info
    }

    /// Original source positions in the tensor batch dimension.
    #[must_use]
    pub fn source_indices(&self) -> &[usize] {
        &self.source_indices
    }

    /// Actual decoded rectangles in tensor batch order.
    #[must_use]
    pub fn decoded_rects(&self) -> &[Rect] {
        &self.decoded_rects
    }

    /// Non-fatal codec warnings in tensor batch order.
    #[must_use]
    pub fn warnings(&self) -> &[Vec<J2kDecodeWarning>] {
        &self.warnings
    }

    /// Completed codec-owned Metal storage retained by this input.
    #[must_use]
    pub const fn resident_batch(&self) -> &MetalResidentBatch {
        &self.resident_batch
    }
}

/// Successful `MPSGraph` input groups plus preserved codec failures.
#[derive(Debug)]
pub struct MpsGraphBatchDecode {
    groups: Vec<MpsGraphInputGroup>,
    errors: Vec<IndexedBatchError>,
    group_errors: Vec<MetalBatchGroupError>,
}

impl MpsGraphBatchDecode {
    fn from_metal(decoded: MetalBatchDecodeResult) -> Result<Self, Error> {
        let (metal_groups, errors, group_errors) = decoded.into_parts();
        let groups = metal_groups
            .into_iter()
            .map(MpsGraphInputGroup::from_metal)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            groups,
            errors,
            group_errors,
        })
    }

    /// Completed resident `MPSGraph` input groups.
    #[must_use]
    pub fn groups(&self) -> &[MpsGraphInputGroup] {
        &self.groups
    }

    /// Indexed preparation failures preserved from the codec batch.
    #[must_use]
    pub fn errors(&self) -> &[IndexedBatchError] {
        &self.errors
    }

    /// Homogeneous codec execution failures.
    #[must_use]
    pub fn group_errors(&self) -> &[MetalBatchGroupError] {
        &self.group_errors
    }

    /// Consume the result into successful groups and both failure classes.
    #[must_use]
    pub fn into_parts(
        self,
    ) -> (
        Vec<MpsGraphInputGroup>,
        Vec<IndexedBatchError>,
        Vec<MetalBatchGroupError>,
    ) {
        (self.groups, self.errors, self.group_errors)
    }
}

impl MpsGraphTensorSpec {
    pub(super) fn mps_data_type(self) -> MPSDataType {
        match self.element_type() {
            MpsGraphElementType::U8 => MPSDataType::UInt8,
            MpsGraphElementType::U16 => MPSDataType::UInt16,
            MpsGraphElementType::I16 => MPSDataType::Int16,
        }
    }
}
