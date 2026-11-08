// SPDX-License-Identifier: MIT OR Apache-2.0

//! Exact operations over the existing owned decoder and CPU session.

#![forbid(unsafe_code)]

use crate::{
    abi::{
        J2kCOperationOptions, J2kCOutputInfo, J2kCRegionPlan, J2K_ALPHA_NONE, J2K_ALPHA_STRAIGHT,
        J2K_CANCELLED, J2K_INTERPRETATION_NATIVE, J2K_INTERPRETATION_SRGB8, J2K_ROUTE_NATIVE_CPU,
        J2K_ROUTE_REGION_PLANES, J2K_U8, J2K_UNSUPPORTED, J2K_WARNING_LENIENT_RECOVERY,
    },
    decoder::{in_pool, native_output_info, write_native_samples, Decoder},
    error::{Error, Result},
};
use j2k::{
    BatchGroupInfo, J2kCancellationToken, J2kDecodeControl, J2kDecoder, J2kSrgb8Layout,
    PreparedImage, PreparedImageOperation, Rect,
};
use std::num::NonZeroUsize;

pub(crate) struct OperationSource {
    pub(crate) image: PreparedImage,
    pub(crate) info: BatchGroupInfo,
    pub(crate) max_reduction: u8,
}

pub(crate) struct OperationRequest {
    roi: Rect,
    levels: u8,
    interpretation: u32,
    output_limit: u64,
    control: J2kDecodeControl,
}

impl OperationRequest {
    pub(crate) fn new(
        options: J2kCOperationOptions,
        dimensions: (u32, u32),
        cancellation: Option<J2kCancellationToken>,
    ) -> Result<Self> {
        if cancellation
            .as_ref()
            .is_some_and(J2kCancellationToken::is_cancelled)
        {
            return Err(Error::new(J2K_CANCELLED, "decode operation was cancelled"));
        }
        if options.reduction > 31 {
            return Err(Error::argument(
                "exact reduction must be between zero and 31",
            ));
        }
        if !matches!(
            options.interpretation,
            J2K_INTERPRETATION_NATIVE | J2K_INTERPRETATION_SRGB8
        ) {
            return Err(Error::argument("unknown sample interpretation"));
        }
        let roi = if [options.x, options.y, options.width, options.height] == [0; 4] {
            Rect::full(dimensions)
        } else {
            Rect {
                x: options.x,
                y: options.y,
                w: options.width,
                h: options.height,
            }
        };
        if roi.w == 0 || roi.h == 0 || !roi.is_within(dimensions) {
            return Err(Error::argument(
                "source region must be nonempty and inside the image",
            ));
        }
        let mut control = J2kDecodeControl::default();
        control.cancellation = cancellation;
        if options.max_working_memory_bytes != 0 {
            control.max_working_memory_bytes = NonZeroUsize::new(
                usize::try_from(options.max_working_memory_bytes)
                    .map_err(|_| Error::limit("working bound exceeds the host range"))?,
            );
        }
        if options.max_output_bytes != 0 {
            control.max_output_bytes = Some(
                usize::try_from(options.max_output_bytes)
                    .map_err(|_| Error::limit("output bound exceeds the host range"))?,
            );
        }
        Ok(Self {
            roi,
            levels: u8::try_from(options.reduction)
                .map_err(|_| Error::argument("invalid reduction"))?,
            interpretation: options.interpretation,
            output_limit: options.max_output_bytes,
            control,
        })
    }

    fn has_execution_control(&self) -> bool {
        self.control.max_working_memory_bytes.is_some() || self.control.cancellation.is_some()
    }

    fn check_cancelled(&self) -> Result<()> {
        if self
            .control
            .cancellation
            .as_ref()
            .is_some_and(J2kCancellationToken::is_cancelled)
        {
            return Err(Error::new(J2K_CANCELLED, "decode operation was cancelled"));
        }
        Ok(())
    }

    fn check_output(&self, bytes: u64) -> Result<()> {
        self.check_cancelled()?;
        if self.output_limit != 0 && bytes > self.output_limit {
            return Err(Error::limit("output exceeds the operation's output bound"));
        }
        Ok(())
    }
}

impl Decoder {
    /// Prepare the native source for `request`. An unprepared source is
    /// prepared under a controlled request's working limit and cancellation;
    /// its output limit applies to the operation's output, not to source
    /// preparation.
    fn prepare_operation_source(&mut self, request: &OperationRequest) -> Result<()> {
        if !request.has_execution_control() {
            return self.prepare_operations();
        }
        self.release_legacy_preparation();
        let mut preparation = request.control.clone();
        preparation.max_output_bytes = None;
        self.prepare_operations_with(Some(&preparation))?;
        self.release_legacy_preparation();
        Ok(())
    }

    pub(crate) fn plan_operation(&mut self, request: &OperationRequest) -> Result<J2kCRegionPlan> {
        request.check_cancelled()?;
        let covered = self
            .reference_grid()
            .reduced_region(request.roi, request.levels)
            .ok_or_else(|| Error::argument("invalid source region or reduction"))?;
        if covered.w == 0 || covered.h == 0 {
            return Err(Error::new(
                J2K_UNSUPPORTED,
                "region has no samples at the requested reduction",
            ));
        }
        // Refuse forged huge geometry before building packet/offset metadata,
        // even when the source's native sample layout is not classified yet.
        let minimum_bytes = u64::from(covered.w) * u64::from(covered.h);
        self.check_output_limit(minimum_bytes)?;
        request.check_output(minimum_bytes)?;
        let (output, max_reduction, route) = if request.interpretation == J2K_INTERPRETATION_NATIVE
        {
            self.prepare_operation_source(request)?;
            let source = self
                .operation_source
                .as_ref()
                .ok_or_else(|| Error::internal("prepared source is missing"))?;
            if request.levels > source.max_reduction {
                return Err(Error::new(
                    J2K_UNSUPPORTED,
                    "requested reduction exceeds a component or tile resolution ladder",
                ));
            }
            let info = native_output_info(
                &source.info,
                covered,
                source.image.used_lenient_metadata_recovery(),
            )?;
            (info, source.max_reduction, J2K_ROUTE_NATIVE_CPU)
        } else {
            let mut decoder =
                J2kDecoder::new_with_control(&self.bytes, self.settings, &request.control)?;
            let (plan, max_reduction) = in_pool(self.pool.as_ref(), || {
                let plan = decoder.srgb8_region_decode_plan_with_control(
                    request.roi,
                    request.levels,
                    &request.control,
                )?;
                Ok::<_, j2k::J2kError>((
                    plan,
                    decoder.max_reduction_levels_with_control(&request.control)?,
                ))
            })?;
            let channels = u32::try_from(plan.format.bytes_per_pixel())
                .map_err(|_| Error::internal("sRGB channel count overflow"))?;
            let output = J2kCOutputInfo {
                x: plan.output_region.x,
                y: plan.output_region.y,
                width: plan.output_region.w,
                height: plan.output_region.h,
                channels,
                sample_type: J2K_U8,
                precision: 8,
                alpha: if channels == 4 {
                    J2K_ALPHA_STRAIGHT
                } else {
                    J2K_ALPHA_NONE
                },
                row_bytes: u64::try_from(plan.row_bytes)
                    .map_err(|_| Error::limit("row size overflow"))?,
                buffer_bytes: u64::try_from(plan.buffer_bytes)
                    .map_err(|_| Error::limit("output size overflow"))?,
                ..Default::default()
            };
            (output, max_reduction, J2K_ROUTE_REGION_PLANES)
        };
        self.check_output_limit(output.buffer_bytes)?;
        request.check_output(output.buffer_bytes)?;
        if request
            .control
            .max_working_memory_bytes
            .is_some_and(|cap| output.buffer_bytes > cap.get() as u64)
        {
            return Err(Error::limit(
                "owned output alone exceeds the operation working bound",
            ));
        }
        Ok(J2kCRegionPlan {
            output,
            source_x: request.roi.x,
            source_y: request.roi.y,
            source_width: request.roi.w,
            source_height: request.roi.h,
            reduction: u32::from(request.levels),
            max_reduction: u32::from(max_reduction),
            interpretation: request.interpretation,
            route,
            ..Default::default()
        })
    }

    pub(crate) fn operation_output_span(
        &self,
        request: &OperationRequest,
        info: J2kCOutputInfo,
        stride: usize,
    ) -> Result<usize> {
        let span = self.output_span(info, stride)?;
        request.check_output(span as u64)?;
        Ok(span)
    }

    pub(crate) fn decode_operation(
        &mut self,
        request: &OperationRequest,
        plan: J2kCRegionPlan,
        output: &mut [u8],
        stride: usize,
    ) -> Result<J2kCOutputInfo> {
        request.check_cancelled()?;
        if request.interpretation == J2K_INTERPRETATION_SRGB8 {
            let mut codec =
                J2kDecoder::new_with_control(&self.bytes, self.settings, &request.control)?;
            let decoded = in_pool(self.pool.as_ref(), || {
                codec.decode_region_srgb8_at_reduction_with_control(
                    request.roi,
                    request.levels,
                    &request.control,
                )
            })?;
            let channels = match decoded.layout() {
                J2kSrgb8Layout::Gray => 1,
                J2kSrgb8Layout::Rgb => 3,
                J2kSrgb8Layout::Rgba => 4,
                _ => {
                    return Err(Error::new(
                        J2K_UNSUPPORTED,
                        "unsupported normalized output layout",
                    ))
                }
            };
            if decoded.dimensions() != (plan.output.width, plan.output.height)
                || channels != plan.output.channels
                || !decoded
                    .warnings()
                    .iter()
                    .all(|&warning| self.allows(warning))
            {
                return Err(Error::internal("normalized output disagrees with its plan"));
            }
            let row = usize::try_from(plan.output.row_bytes)
                .map_err(|_| Error::limit("row exceeds host range"))?;
            for (source, destination) in decoded
                .data()
                .chunks_exact(row)
                .zip(output.chunks_mut(stride))
            {
                destination[..row].copy_from_slice(source);
            }
            return Ok(J2kCOutputInfo {
                warnings: if decoded.warnings().is_empty() {
                    0
                } else {
                    J2K_WARNING_LENIENT_RECOVERY
                },
                ..plan.output
            });
        }
        let source = self
            .operation_source
            .as_ref()
            .ok_or_else(|| Error::internal("native source is not prepared"))?;
        let mut operation = PreparedImageOperation::new(&source.image, request.roi, request.levels);
        if request.has_execution_control() {
            operation = operation.with_control(&request.control);
        }
        let result = in_pool(self.pool.as_ref(), || {
            self.session.decode_operations(&[operation])
        })
        .map_err(Error::infrastructure)?;
        if let Some(error) = result.errors().first() {
            return Err(Error::item(&error.source));
        }
        let [group] = result.groups() else {
            return Err(Error::internal("expected one exact decoded group"));
        };
        let [rect] = group.decoded_rects() else {
            return Err(Error::internal("expected one decoded rectangle"));
        };
        if group.source_indices() != [0]
            || !group
                .warnings()
                .iter()
                .flatten()
                .all(|&warning| self.allows(warning))
        {
            return Err(Error::internal(
                "exact output changed source order or warnings",
            ));
        }
        let actual = native_output_info(
            group.info(),
            *rect,
            !group.warnings().iter().all(Vec::is_empty),
        )?;
        if actual != plan.output {
            return Err(Error::internal("native output disagrees with its plan"));
        }
        write_native_samples(group.samples(), actual, output, stride)?;
        Ok(actual)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abi::{J2kCDecodeOptions, J2kCOpenOptions};
    use std::sync::Arc;

    #[test]
    fn controlled_operations_release_unused_legacy_preparation_owners() {
        let bytes = include_bytes!(
            "../../j2k-test-support/fixtures/openjpeg_origin/rgb_23x17_origin_3_5_ht.j2c"
        );
        let mut decoder = Decoder::open(bytes, J2kCOpenOptions::default()).unwrap();
        decoder.prepare_operations().unwrap();
        decoder.prepare(J2kCDecodeOptions::default()).unwrap();
        let owners = Arc::strong_count(&decoder.bytes);
        let request = OperationRequest::new(
            J2kCOperationOptions {
                max_working_memory_bytes: 1 << 20,
                ..Default::default()
            },
            decoder.image_size(),
            None,
        )
        .unwrap();
        decoder.plan_operation(&request).unwrap();
        assert_eq!(
            Arc::strong_count(&decoder.bytes),
            owners - 1,
            "unused legacy preparation must release its input and codec-plan ownership"
        );
        assert!(
            decoder.operation_source.is_some(),
            "explicit source preparation remains reusable"
        );
    }
}
