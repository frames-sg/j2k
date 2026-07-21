// SPDX-License-Identifier: MIT OR Apache-2.0

//! Codec-owned resident Metal group submission.

#[cfg(target_os = "macos")]
use super::{
    allocate_codec_owned_group_destination, validate_codec_owned_resident_group, BatchColor,
    MetalResidentGroupMetadata, SubmittedMetalResidentGroup,
};
use super::{BatchDecodeOptions, Error, MetalBatchDecoder, MetalBatchGroup, PreparedBatchGroup};

impl MetalBatchDecoder {
    pub(super) fn decode_prepared_group_with_options(
        &mut self,
        group: &PreparedBatchGroup,
        options: BatchDecodeOptions,
    ) -> Result<MetalBatchGroup, Error> {
        #[cfg(target_os = "macos")]
        {
            let pending = self.submit_prepared_resident_group(group, options)?;
            pending.wait().map_err(|(_, source)| *source)
        }

        #[cfg(not(target_os = "macos"))]
        {
            let _ = group;
            let _ = options;
            Err(Error::MetalUnavailable)
        }
    }

    #[cfg(target_os = "macos")]
    pub(super) fn submit_prepared_resident_group(
        &mut self,
        group: &PreparedBatchGroup,
        options: BatchDecodeOptions,
    ) -> Result<SubmittedMetalResidentGroup, Error> {
        let fmt = validate_codec_owned_resident_group(group)?;
        let allocation =
            allocate_codec_owned_group_destination(self.backend_session().device(), group, fmt)?;
        let runtime = self.backend_session().runtime()?;
        let submission =
            match group.info().color {
                BatchColor::Gray => {
                    let plans = self.prepared_gray_group_plans(group, fmt, true)?;
                    crate::compute::submit_prepared_direct_grayscale_plan_batch_into_group(
                        runtime,
                        &plans,
                        fmt,
                        &allocation.destination,
                        Some(group.source_indices()),
                        crate::compute::DirectDestinationConsumerOrdering::HostCompletionOnly,
                    )?
                }
                BatchColor::Rgb | BatchColor::Rgba => {
                    let plans = self.prepared_color_group_plans(group, fmt)?;
                    crate::compute::submit_prepared_direct_color_plan_batch_into_group(
                        runtime,
                        &plans,
                        fmt,
                        group.info().layout,
                        &allocation.destination,
                        Some(group.source_indices()),
                        crate::compute::DirectDestinationConsumerOrdering::HostCompletionOnly,
                    )?
                }
                _ => return Err(Error::UnsupportedMetalRequest {
                    reason:
                        "J2K Metal codec-owned resident output received an unknown color contract",
                }),
            };
        self.record_submission();
        Ok(SubmittedMetalResidentGroup {
            metadata: MetalResidentGroupMetadata::from_prepared(group, options),
            submission,
            destination: allocation.destination,
            output: allocation.output,
            layout: allocation.layout,
        })
    }
}
