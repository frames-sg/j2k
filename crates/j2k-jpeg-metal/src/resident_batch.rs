// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Preflight report for RGB8 JPEG Metal resident decoder batches.
#[doc(hidden)]
pub struct JpegMetalResidentBatchReport {
    /// Requested decode operation.
    pub op: j2k_jpeg::JpegDecodeOp,
    /// Number of decoder tiles in the batch.
    pub tile_count: usize,
    /// Required output dimensions when the batch is eligible and shape-compatible.
    pub output_dimensions: Option<(u32, u32)>,
    /// Whether the batch can use reusable RGB8 Metal resident output.
    pub eligibility: j2k_jpeg::JpegBackendEligibility,
}

impl JpegMetalResidentBatchReport {
    /// Required number of tile slots in caller-owned Metal output.
    #[must_use]
    pub fn required_tile_capacity(&self) -> usize {
        self.tile_count
    }
}

pub(crate) fn report_required_output_dimensions(
    report: &JpegMetalResidentBatchReport,
) -> Result<Option<(u32, u32)>, Error> {
    if !report.eligibility.eligible {
        return Err(Error::UnsupportedMetalRequest {
            reason: report
                .eligibility
                .reason
                .unwrap_or("JPEG Metal resident batch report is not eligible"),
        });
    }
    if report.tile_count == 0 {
        return Ok(None);
    }
    report
        .output_dimensions
        .ok_or(Error::UnsupportedMetalRequest {
            reason: "JPEG Metal resident batch report is missing output dimensions",
        })
        .map(Some)
}
