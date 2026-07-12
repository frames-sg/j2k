// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(feature = "cuda-runtime")]
use j2k::J2kEncodeStageError;

#[cfg(feature = "cuda-runtime")]
use crate::encode::stage_error::{adapter_error, arithmetic_overflow};

pub(super) const CUDA_PACKETIZATION_TAG_TREE_ALLOCATION: &str =
    "CUDA HTJ2K packetization tag-tree state";

#[derive(Debug)]
pub(in crate::encode) enum CudaHtj2kPacketizationPlanError {
    Invalid(&'static str),
    ArithmeticOverflow(&'static str),
    MemoryCapExceeded {
        what: &'static str,
        requested: usize,
        cap: usize,
    },
    HostAllocation {
        what: &'static str,
        bytes: usize,
    },
    Adapter(crate::Error),
}

impl CudaHtj2kPacketizationPlanError {
    #[cfg(feature = "cuda-runtime")]
    pub(in crate::encode) fn into_stage_error(self) -> J2kEncodeStageError {
        match self {
            Self::Invalid(what) => J2kEncodeStageError::invalid_request(what),
            Self::ArithmeticOverflow(what) => arithmetic_overflow(what),
            Self::MemoryCapExceeded {
                what,
                requested,
                cap,
            } => J2kEncodeStageError::memory_cap_exceeded(what, requested, cap),
            Self::HostAllocation { what, bytes } => {
                J2kEncodeStageError::host_allocation_failed(what, bytes)
            }
            Self::Adapter(source) => adapter_error("prepare CUDA HTJ2K packetization plan", source),
        }
    }
}

impl PartialEq for CudaHtj2kPacketizationPlanError {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Invalid(left), Self::Invalid(right))
            | (Self::ArithmeticOverflow(left), Self::ArithmeticOverflow(right)) => left == right,
            (
                Self::MemoryCapExceeded {
                    what: left_what,
                    requested: left_requested,
                    cap: left_cap,
                },
                Self::MemoryCapExceeded {
                    what: right_what,
                    requested: right_requested,
                    cap: right_cap,
                },
            ) => {
                left_what == right_what
                    && left_requested == right_requested
                    && left_cap == right_cap
            }
            (
                Self::HostAllocation {
                    what: left_what,
                    bytes: left_bytes,
                },
                Self::HostAllocation {
                    what: right_what,
                    bytes: right_bytes,
                },
            ) => left_what == right_what && left_bytes == right_bytes,
            _ => false,
        }
    }
}

pub(super) fn packetization_plan_allocation_error(
    error: crate::Error,
) -> CudaHtj2kPacketizationPlanError {
    match error {
        crate::Error::HostAllocationFailed { bytes, what } => {
            CudaHtj2kPacketizationPlanError::HostAllocation { what, bytes }
        }
        crate::Error::HostAllocationTooLarge {
            requested,
            cap,
            what,
        } => CudaHtj2kPacketizationPlanError::MemoryCapExceeded {
            what,
            requested,
            cap,
        },
        source => CudaHtj2kPacketizationPlanError::Adapter(source),
    }
}

pub(super) type PacketizationPlanResult<T> =
    core::result::Result<T, CudaHtj2kPacketizationPlanError>;
