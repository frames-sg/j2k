// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::PixelFormat;
use j2k_cuda_runtime::{
    CudaContext, CudaDeviceBuffer, CudaError, CudaJ2kInverseMctJob, CudaJ2kStoreRgb16Job,
    CudaJ2kStoreRgb16MctJob, CudaJ2kStoreRgb8Job, CudaJ2kStoreRgb8MctJob, CudaKernelOutput,
};

use super::super::resident::{bit_depth_addend, checked_area};
use super::super::{
    cuda_error, CudaHtj2kStoreStep, CudaHtj2kTransform, Error, CUDA_HTJ2K_KERNELS_NOT_READY,
    CUDA_HTJ2K_OUTPUT_FORMAT_UNSUPPORTED,
};

mod batch;

pub(super) use batch::{prepare_rgb8_mct_batch_store, rgb8_mct_batch_store_target};

#[derive(Clone, Copy)]
pub(super) struct ColorStoreInputs<'a> {
    pub(super) context: &'a CudaContext,
    pub(super) buffers: [&'a CudaDeviceBuffer; 3],
    pub(super) stores: [&'a CudaHtj2kStoreStep; 3],
    pub(super) bit_depths: [u8; 3],
}

#[derive(Clone, Copy)]
pub(super) struct ColorMctOutcome {
    store_addends: [f32; 3],
    store_route: ColorStoreRoute,
    pub(super) kernel_dispatches: usize,
    pub(super) decode_kernel_dispatches: usize,
    pub(super) elapsed_us: u128,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ColorStoreRoute {
    Separate,
    FusedReversible,
    FusedIrreversible,
}

#[derive(Clone, Copy)]
struct ColorStorePlan {
    input_widths: [u32; 3],
    source_x: [u32; 3],
    source_y: [u32; 3],
    copy_width: u32,
    copy_height: u32,
    output_width: u32,
    output_height: u32,
    output_x: u32,
    output_y: u32,
    addends: [f32; 3],
    bit_depths: [u8; 3],
    route: ColorStoreRoute,
}

impl ColorStorePlan {
    fn new(
        stores: [&CudaHtj2kStoreStep; 3],
        bit_depths: [u8; 3],
        addends: [f32; 3],
        route: ColorStoreRoute,
    ) -> Self {
        Self {
            input_widths: [
                color_store_input_width(stores[0]),
                color_store_input_width(stores[1]),
                color_store_input_width(stores[2]),
            ],
            source_x: [stores[0].source_x, stores[1].source_x, stores[2].source_x],
            source_y: [stores[0].source_y, stores[1].source_y, stores[2].source_y],
            copy_width: stores[0].copy_width,
            copy_height: stores[0].copy_height,
            output_width: stores[0].output_width,
            output_height: stores[0].output_height,
            output_x: stores[0].output_x,
            output_y: stores[0].output_y,
            addends,
            bit_depths,
            route,
        }
    }

    #[cfg(test)]
    fn route(self) -> ColorStoreRoute {
        self.route
    }

    fn rgb8_job(self, rgba: bool) -> CudaJ2kStoreRgb8Job {
        CudaJ2kStoreRgb8Job {
            input_width0: self.input_widths[0],
            input_width1: self.input_widths[1],
            input_width2: self.input_widths[2],
            source_x0: self.source_x[0],
            source_y0: self.source_y[0],
            source_x1: self.source_x[1],
            source_y1: self.source_y[1],
            source_x2: self.source_x[2],
            source_y2: self.source_y[2],
            copy_width: self.copy_width,
            copy_height: self.copy_height,
            output_width: self.output_width,
            output_height: self.output_height,
            output_x: self.output_x,
            output_y: self.output_y,
            addend0: self.addends[0],
            addend1: self.addends[1],
            addend2: self.addends[2],
            bit_depth0: u32::from(self.bit_depths[0]),
            bit_depth1: u32::from(self.bit_depths[1]),
            bit_depth2: u32::from(self.bit_depths[2]),
            rgba: u32::from(rgba),
        }
    }

    fn rgb16_job(self, rgba: bool) -> CudaJ2kStoreRgb16Job {
        CudaJ2kStoreRgb16Job {
            input_width0: self.input_widths[0],
            input_width1: self.input_widths[1],
            input_width2: self.input_widths[2],
            source_x0: self.source_x[0],
            source_y0: self.source_y[0],
            source_x1: self.source_x[1],
            source_y1: self.source_y[1],
            source_x2: self.source_x[2],
            source_y2: self.source_y[2],
            copy_width: self.copy_width,
            copy_height: self.copy_height,
            output_width: self.output_width,
            output_height: self.output_height,
            output_x: self.output_x,
            output_y: self.output_y,
            addend0: self.addends[0],
            addend1: self.addends[1],
            addend2: self.addends[2],
            bit_depth0: u32::from(self.bit_depths[0]),
            bit_depth1: u32::from(self.bit_depths[1]),
            bit_depth2: u32::from(self.bit_depths[2]),
            rgba: u32::from(rgba),
        }
    }

    fn uses_fused_store(self) -> bool {
        self.route != ColorStoreRoute::Separate
    }

    fn irreversible97(self) -> u32 {
        u32::from(self.route == ColorStoreRoute::FusedIrreversible)
    }
}

impl ColorStoreRoute {
    fn for_mct(can_fuse_store: bool, transform: CudaHtj2kTransform) -> Self {
        match (can_fuse_store, transform) {
            (false, _) => Self::Separate,
            (true, CudaHtj2kTransform::Reversible53) => Self::FusedReversible,
            (true, CudaHtj2kTransform::Irreversible97) => Self::FusedIrreversible,
        }
    }
}

pub(super) fn can_fuse_mct_store_for_stores(stores: [&CudaHtj2kStoreStep; 3]) -> bool {
    let input_width0 = color_store_input_width(stores[0]);
    let input_width1 = color_store_input_width(stores[1]);
    let input_width2 = color_store_input_width(stores[2]);
    input_width0 == input_width1
        && input_width0 == input_width2
        && stores[0].source_x == stores[1].source_x
        && stores[0].source_x == stores[2].source_x
        && stores[0].source_y == stores[1].source_y
        && stores[0].source_y == stores[2].source_y
}

pub(super) fn color_store_input_width(store: &CudaHtj2kStoreStep) -> u32 {
    store.input_rect.x1.saturating_sub(store.input_rect.x0)
}

pub(super) fn run_color_mct(
    inputs: ColorStoreInputs<'_>,
    mct_dimensions: (u32, u32),
    mct: bool,
    transform: CudaHtj2kTransform,
    collect_stage_timings: bool,
) -> Result<ColorMctOutcome, Error> {
    let irreversible97 = u32::from(transform == CudaHtj2kTransform::Irreversible97);
    let mct_store_addends = [
        bit_depth_addend(inputs.bit_depths[0]),
        bit_depth_addend(inputs.bit_depths[1]),
        bit_depth_addend(inputs.bit_depths[2]),
    ];
    let can_fuse_store = mct && can_fuse_mct_store_for_stores(inputs.stores);
    let store_route = ColorStoreRoute::for_mct(can_fuse_store, transform);
    if can_fuse_store {
        return Ok(ColorMctOutcome {
            store_addends: mct_store_addends,
            store_route,
            kernel_dispatches: 0,
            decode_kernel_dispatches: 0,
            elapsed_us: 0,
        });
    }
    if !mct {
        return Ok(ColorMctOutcome {
            store_addends: [
                inputs.stores[0].addend,
                inputs.stores[1].addend,
                inputs.stores[2].addend,
            ],
            store_route,
            kernel_dispatches: 0,
            decode_kernel_dispatches: 0,
            elapsed_us: 0,
        });
    }

    let mct_len =
        u32::try_from(checked_area(mct_dimensions.0, mct_dimensions.1)?).map_err(|_| {
            Error::UnsupportedCudaRequest {
                reason: CUDA_HTJ2K_KERNELS_NOT_READY,
            }
        })?;
    let (stats, elapsed_us) = inputs
        .context
        .time_default_stream_named_us_if(collect_stage_timings, "j2k.htj2k.decode.mct", || {
            inputs.context.j2k_inverse_mct_device(
                inputs.buffers[0],
                inputs.buffers[1],
                inputs.buffers[2],
                CudaJ2kInverseMctJob {
                    len: mct_len,
                    irreversible97,
                    addend0: mct_store_addends[0],
                    addend1: mct_store_addends[1],
                    addend2: mct_store_addends[2],
                },
            )
        })
        .map_err(cuda_error)?;
    Ok(ColorMctOutcome {
        store_addends: [0.0, 0.0, 0.0],
        store_route,
        kernel_dispatches: stats.kernel_dispatches(),
        decode_kernel_dispatches: stats.decode_kernel_dispatches(),
        elapsed_us,
    })
}

fn dispatch_color_store_u8(
    inputs: ColorStoreInputs<'_>,
    plan: ColorStorePlan,
    rgba: bool,
) -> Result<CudaKernelOutput, CudaError> {
    let store_job = plan.rgb8_job(rgba);
    if plan.uses_fused_store() {
        inputs.context.j2k_store_rgb8_mct_device(
            inputs.buffers[0],
            inputs.buffers[1],
            inputs.buffers[2],
            CudaJ2kStoreRgb8MctJob {
                store: store_job,
                irreversible97: plan.irreversible97(),
            },
        )
    } else {
        inputs.context.j2k_store_rgb8_device(
            inputs.buffers[0],
            inputs.buffers[1],
            inputs.buffers[2],
            store_job,
        )
    }
}

fn dispatch_color_store_u16(
    inputs: ColorStoreInputs<'_>,
    plan: ColorStorePlan,
    rgba: bool,
) -> Result<CudaKernelOutput, CudaError> {
    let store_job = plan.rgb16_job(rgba);
    if plan.uses_fused_store() {
        inputs.context.j2k_store_rgb16_mct_device(
            inputs.buffers[0],
            inputs.buffers[1],
            inputs.buffers[2],
            CudaJ2kStoreRgb16MctJob {
                store: store_job,
                irreversible97: plan.irreversible97(),
            },
        )
    } else {
        inputs.context.j2k_store_rgb16_device(
            inputs.buffers[0],
            inputs.buffers[1],
            inputs.buffers[2],
            store_job,
        )
    }
}

pub(super) fn dispatch_color_store(
    inputs: ColorStoreInputs<'_>,
    mct: ColorMctOutcome,
    fmt: PixelFormat,
) -> Result<CudaKernelOutput, CudaError> {
    let plan = ColorStorePlan::new(
        inputs.stores,
        inputs.bit_depths,
        mct.store_addends,
        mct.store_route,
    );
    match fmt {
        PixelFormat::Rgb8 => dispatch_color_store_u8(inputs, plan, false),
        PixelFormat::Rgba8 => dispatch_color_store_u8(inputs, plan, true),
        PixelFormat::Rgb16 => dispatch_color_store_u16(inputs, plan, false),
        PixelFormat::Rgba16 => dispatch_color_store_u16(inputs, plan, true),
        _ => Err(CudaError::InvalidArgument {
            message: CUDA_HTJ2K_OUTPUT_FORMAT_UNSUPPORTED.to_string(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CudaHtj2kRect;

    fn color_store_steps(input_widths: [u32; 3]) -> [CudaHtj2kStoreStep; 3] {
        const BAND_IDS: [u32; 3] = [0, 1, 2];
        let addends = [0.5_f32, 1.5, 2.5];
        core::array::from_fn(|index| CudaHtj2kStoreStep {
            input_band_id: BAND_IDS[index],
            input_rect: CudaHtj2kRect {
                x0: 4,
                y0: 7,
                x1: 4 + input_widths[index],
                y1: 39,
            },
            source_x: 3,
            source_y: 5,
            copy_width: 11,
            copy_height: 13,
            output_width: 17,
            output_height: 19,
            output_x: 2,
            output_y: 4,
            addend: addends[index],
        })
    }

    fn assert_expected_addends(actual: [f32; 3]) {
        for (actual, expected) in actual.into_iter().zip([1.5, -2.0, 3.25]) {
            assert!((actual - expected).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn color_store_plan_builds_rgb_and_rgba_jobs_for_both_sample_widths() {
        let stores = color_store_steps([64, 64, 64]);
        let store_refs = [&stores[0], &stores[1], &stores[2]];
        let plan = ColorStorePlan::new(
            store_refs,
            [8, 10, 12],
            [1.5, -2.0, 3.25],
            ColorStoreRoute::for_mct(true, CudaHtj2kTransform::Reversible53),
        );

        assert_eq!(plan.route(), ColorStoreRoute::FusedReversible);
        let rgb8 = plan.rgb8_job(false);
        let rgb16 = plan.rgb16_job(false);

        assert_eq!(
            [rgb8.input_width0, rgb8.input_width1, rgb8.input_width2],
            [64, 64, 64]
        );
        assert_eq!([rgb8.source_x0, rgb8.source_x1, rgb8.source_x2], [3; 3]);
        assert_eq!([rgb8.source_y0, rgb8.source_y1, rgb8.source_y2], [5; 3]);
        assert_eq!(
            (
                rgb8.copy_width,
                rgb8.copy_height,
                rgb8.output_width,
                rgb8.output_height,
                rgb8.output_x,
                rgb8.output_y,
            ),
            (11, 13, 17, 19, 2, 4)
        );
        assert_expected_addends([rgb8.addend0, rgb8.addend1, rgb8.addend2]);
        assert_eq!(
            [rgb8.bit_depth0, rgb8.bit_depth1, rgb8.bit_depth2],
            [8, 10, 12]
        );
        assert_eq!(rgb8.rgba, 0);
        assert_eq!(plan.rgb8_job(true).rgba, 1);

        assert_eq!(
            [rgb16.input_width0, rgb16.input_width1, rgb16.input_width2],
            [64, 64, 64]
        );
        assert_eq!([rgb16.source_x0, rgb16.source_x1, rgb16.source_x2], [3; 3]);
        assert_eq!([rgb16.source_y0, rgb16.source_y1, rgb16.source_y2], [5; 3]);
        assert_eq!(
            (
                rgb16.copy_width,
                rgb16.copy_height,
                rgb16.output_width,
                rgb16.output_height,
                rgb16.output_x,
                rgb16.output_y,
            ),
            (11, 13, 17, 19, 2, 4)
        );
        assert_expected_addends([rgb16.addend0, rgb16.addend1, rgb16.addend2]);
        assert_eq!(
            [rgb16.bit_depth0, rgb16.bit_depth1, rgb16.bit_depth2],
            [8, 10, 12]
        );
        assert_eq!(rgb16.rgba, 0);
        assert_eq!(plan.rgb16_job(true).rgba, 1);
    }

    #[test]
    fn color_store_plan_distinguishes_fused_transform_and_separate_routes() {
        let stores = color_store_steps([64, 64, 64]);
        let store_refs = [&stores[0], &stores[1], &stores[2]];
        assert!(can_fuse_mct_store_for_stores(store_refs));
        assert_eq!(
            ColorStoreRoute::for_mct(true, CudaHtj2kTransform::Reversible53),
            ColorStoreRoute::FusedReversible
        );
        assert_eq!(
            ColorStoreRoute::for_mct(true, CudaHtj2kTransform::Irreversible97),
            ColorStoreRoute::FusedIrreversible
        );
        assert_eq!(
            ColorStoreRoute::for_mct(false, CudaHtj2kTransform::Irreversible97),
            ColorStoreRoute::Separate
        );

        let mismatched = color_store_steps([64, 64, 32]);
        let mismatched_refs = [&mismatched[0], &mismatched[1], &mismatched[2]];
        assert!(!can_fuse_mct_store_for_stores(mismatched_refs));
    }
}
