// SPDX-License-Identifier: MIT OR Apache-2.0

mod input;

use super::{
    host_owners, CudaHtj2kColorDecodePlans, Error, HostPhaseBudget, NativeColorBatchInput,
    NativeDecoderContext, PixelFormat, CUDA_HTJ2K_OUTPUT_FORMAT_UNSUPPORTED,
};

use self::input::prepare_native_color_input;

pub(super) struct PreparedNativeColorBatch {
    pub(super) colors: Vec<CudaHtj2kColorDecodePlans>,
    pub(super) shared_payload: Vec<u8>,
    pub(super) source_indices: Vec<usize>,
}

pub(super) fn prepare_native_color_batch(
    inputs: &[NativeColorBatchInput<'_>],
    fmt: PixelFormat,
) -> Result<PreparedNativeColorBatch, Error> {
    validate_native_color_format(fmt)?;
    let mut initial_budget = HostPhaseBudget::new("j2k CUDA exact RGB batch plan owners");
    let mut prepared = PreparedNativeColorBatch {
        colors: initial_budget.try_vec_with_capacity(inputs.len())?,
        shared_payload: Vec::new(),
        source_indices: initial_budget.try_vec_with_capacity(inputs.len())?,
    };
    let mut native_context = NativeDecoderContext::default();
    for (output_index, input) in inputs.iter().enumerate() {
        let input_colors = prepare_native_color_input(
            input,
            fmt,
            &mut native_context,
            &prepared.colors,
            &mut prepared.shared_payload,
        )?;
        append_native_color_input(&mut prepared, output_index, input, input_colors)?;
    }
    Ok(prepared)
}

fn validate_native_color_format(fmt: PixelFormat) -> Result<(), Error> {
    if matches!(
        fmt,
        PixelFormat::Rgb8
            | PixelFormat::Rgb16
            | PixelFormat::RgbI16
            | PixelFormat::Rgba8
            | PixelFormat::Rgba16
            | PixelFormat::RgbaI16
    ) {
        return Ok(());
    }
    Err(Error::UnsupportedCudaRequest {
        reason: CUDA_HTJ2K_OUTPUT_FORMAT_UNSUPPORTED,
    })
}

fn append_native_color_input(
    prepared: &mut PreparedNativeColorBatch,
    output_index: usize,
    input: &NativeColorBatchInput<'_>,
    mut input_colors: Vec<CudaHtj2kColorDecodePlans>,
) -> Result<(), Error> {
    let mut budget = host_owners::color_batch_budget(
        &prepared.colors,
        &prepared.shared_payload,
        None,
        "j2k CUDA exact color tile owner append",
    )?;
    budget.account_vec(&prepared.source_indices)?;
    budget.try_vec_reserve(&mut prepared.colors, input_colors.len())?;
    budget.try_vec_reserve(&mut prepared.source_indices, input_colors.len())?;
    for color in &mut input_colors {
        color.output_index = output_index;
    }
    prepared
        .source_indices
        .extend(std::iter::repeat_n(input.source_index, input_colors.len()));
    prepared.colors.append(&mut input_colors);
    Ok(())
}
