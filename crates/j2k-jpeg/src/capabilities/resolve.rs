// SPDX-License-Identifier: MIT OR Apache-2.0

//! Backend path resolution after correctness eligibility is known.

use super::{output_geometry::output_rect_for_request, JpegCapabilityReport, JpegDecodeRequest};
use crate::{JpegError, Rect};
use j2k_core::BackendRequest;

/// Normalized JPEG decode path selected for a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JpegResolvedDecodePath {
    /// Portable CPU host decode.
    CpuHost,
    /// J2K-owned CUDA RGB8 decode path.
    OwnedCudaRgb8,
    /// J2K Metal fast-packet decode path.
    MetalFast,
    /// Request cannot be satisfied by this path resolver.
    Rejected {
        /// Backend requested by the caller.
        backend: BackendRequest,
        /// Stable rejection reason.
        reason: &'static str,
    },
}

/// Parsed JPEG metadata plus the selected backend path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JpegResolvedDecode {
    /// Original decode request.
    pub request: JpegDecodeRequest,
    /// Capability report used for the decision.
    pub capabilities: JpegCapabilityReport,
    /// Output rectangle after ROI and scale are applied.
    pub output_rect: Rect,
    /// Selected backend path.
    pub path: JpegResolvedDecodePath,
}

impl JpegResolvedDecode {
    /// Inspect JPEG bytes and resolve the requested backend path.
    ///
    /// # Errors
    ///
    /// Returns a JPEG parse or capability error when `input` is malformed or
    /// the requested operation cannot be described safely.
    pub fn inspect(input: &[u8], request: JpegDecodeRequest) -> Result<Self, JpegError> {
        let capabilities = JpegCapabilityReport::inspect(input, request.capability())?;
        Ok(Self::from_capabilities(capabilities, request))
    }

    /// Resolve a path from an existing capability report.
    #[must_use]
    pub fn from_capabilities(
        capabilities: JpegCapabilityReport,
        request: JpegDecodeRequest,
    ) -> Self {
        let output_rect = output_rect_for_request(&capabilities.info, request.op);
        let path = capabilities.resolve_path(request.backend);
        Self {
            request,
            capabilities,
            output_rect,
            path,
        }
    }
}

impl JpegCapabilityReport {
    /// Resolve a backend request using this report's correctness eligibility.
    ///
    /// `Auto` remains on the portable CPU path here. Performance promotion is
    /// owned by accelerator routing layers with measured workload context.
    #[must_use]
    pub fn resolve_path(&self, backend: BackendRequest) -> JpegResolvedDecodePath {
        match backend {
            BackendRequest::Cpu => {
                if self.cpu.eligible {
                    JpegResolvedDecodePath::CpuHost
                } else {
                    JpegResolvedDecodePath::Rejected {
                        backend,
                        reason: self
                            .cpu
                            .reason
                            .unwrap_or("JPEG CPU decode rejected this request"),
                    }
                }
            }
            BackendRequest::Auto => JpegResolvedDecodePath::CpuHost,
            BackendRequest::Cuda => {
                if self.owned_cuda.eligible {
                    JpegResolvedDecodePath::OwnedCudaRgb8
                } else {
                    JpegResolvedDecodePath::Rejected {
                        backend,
                        reason: self
                            .owned_cuda
                            .reason
                            .unwrap_or("J2K-owned CUDA JPEG decode rejected this request"),
                    }
                }
            }
            BackendRequest::Metal => {
                if self.metal_fast.eligible {
                    JpegResolvedDecodePath::MetalFast
                } else {
                    JpegResolvedDecodePath::Rejected {
                        backend,
                        reason: self
                            .metal_fast
                            .reason
                            .unwrap_or("JPEG Metal fast path rejected this request"),
                    }
                }
            }
        }
    }
}
