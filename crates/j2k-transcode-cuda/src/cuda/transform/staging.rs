// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{
    checked_element_product, checked_host_byte_add, checked_host_byte_sum, checked_host_bytes,
    CudaTranscodeDwt97Bands, CudaTranscodeError, CudaTranscodeReversible53Bands,
    Dwt97TwoDimensional, HostPhaseBudget, ReversibleDwt53FirstLevel,
};

/// Flatten `&[[i16; 64]]` into the contiguous `&[i16]` the runtime job expects.
pub(super) fn flatten_blocks(blocks: &[[i16; 64]]) -> &[i16] {
    blocks.as_flattened()
}

pub(super) fn bands_to_first_level(
    bands: CudaTranscodeReversible53Bands,
) -> ReversibleDwt53FirstLevel {
    ReversibleDwt53FirstLevel {
        ll: bands.ll,
        hl: bands.hl,
        lh: bands.lh,
        hh: bands.hh,
        low_width: bands.low_width,
        low_height: bands.low_height,
        high_width: bands.high_width,
        high_height: bands.high_height,
    }
}

/// Append the job's `[[f64; 8]; 8]` natural-order DCT blocks to a contiguous
/// `f32` coefficient buffer (row-major within block) the runtime kernels consume.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the CUDA kernel ABI intentionally consumes f32 DCT coefficients"
)]
pub(super) fn append_f64_blocks_to_f32(blocks: &[[[f64; 8]; 8]], out: &mut Vec<f32>) {
    for block in blocks {
        for row in block {
            for &coefficient in row {
                out.push(coefficient as f32);
            }
        }
    }
}

/// Append natural-order dequantized i16 DCT blocks directly to the contiguous
/// i16 coefficient buffer the runtime kernels consume.
pub(in crate::cuda) fn append_i16_blocks(blocks: &[[i16; 64]], out: &mut Vec<i16>) {
    out.extend_from_slice(flatten_blocks(blocks));
}

/// Flatten one job's DCT blocks into a fresh contiguous `f32` buffer.
pub(super) fn flatten_f64_blocks_to_f32(
    blocks: &[[[f64; 8]; 8]],
    budget: &mut HostPhaseBudget,
) -> Result<Vec<f32>, CudaTranscodeError> {
    let mut out =
        budget.try_vec_for_product::<f32>(&[blocks.len(), 64], "single-job f32 DCT staging")?;
    append_f64_blocks_to_f32(blocks, &mut out);
    Ok(out)
}

pub(super) fn validate_block_grid(
    block_count: usize,
    block_cols: usize,
    block_rows: usize,
    mismatch: &'static str,
) -> Result<(), CudaTranscodeError> {
    let expected =
        checked_element_product(&[block_cols, block_rows], "CUDA transcode DCT block grid")?;
    if block_count != expected {
        return Err(CudaTranscodeError::UnsupportedJob(mismatch));
    }
    Ok(())
}

pub(super) fn validate_staging_and_readback_workspace(
    item_count: usize,
    block_cols: usize,
    block_rows: usize,
    width: usize,
    height: usize,
    what: &'static str,
) -> Result<(), CudaTranscodeError> {
    let input_count = checked_element_product(
        &[item_count, block_cols, block_rows, 64],
        "CUDA transcode input coefficient geometry",
    )?;
    let output_count = checked_element_product(
        &[item_count, width, height],
        "CUDA transcode output coefficient geometry",
    )?;
    checked_host_byte_sum(
        &[
            checked_host_bytes::<f32>(input_count, what)?,
            checked_host_bytes::<f32>(output_count, what)?,
        ],
        what,
    )?;
    Ok(())
}

fn preflight_dwt97_conversion_budget(
    budget: &HostPhaseBudget,
    bands: &[CudaTranscodeDwt97Bands],
    output_count: usize,
) -> Result<(), CudaTranscodeError> {
    let mut destination_bytes = 0usize;
    for item in bands {
        for band_len in [item.ll.len(), item.hl.len(), item.lh.len(), item.hh.len()] {
            destination_bytes = checked_host_byte_add(
                destination_bytes,
                checked_host_bytes::<f64>(band_len, "CUDA 9/7 f64 widened bands")?,
                "CUDA 9/7 f64 widened bands",
            )?;
        }
    }
    let additional = checked_host_byte_sum(
        &[
            destination_bytes,
            checked_host_bytes::<Dwt97TwoDimensional<f64>>(
                output_count,
                "CUDA 9/7 f64 band metadata",
            )?,
        ],
        "CUDA 9/7 widening workspace",
    )?;
    budget.preflight_bytes(additional)
}

fn dwt97_bands_to_f64_after_preflight(
    bands: CudaTranscodeDwt97Bands,
    budget: &mut HostPhaseBudget,
) -> Result<Dwt97TwoDimensional<f64>, CudaTranscodeError> {
    let mut widen = |band: Vec<f32>, what| -> Result<Vec<f64>, CudaTranscodeError> {
        let mut widened = budget.try_vec_with_capacity(band.len(), what)?;
        widened.extend(band.into_iter().map(f64::from));
        Ok(widened)
    };
    Ok(Dwt97TwoDimensional {
        ll: widen(bands.ll, "CUDA 9/7 LL f64 readback")?,
        hl: widen(bands.hl, "CUDA 9/7 HL f64 readback")?,
        lh: widen(bands.lh, "CUDA 9/7 LH f64 readback")?,
        hh: widen(bands.hh, "CUDA 9/7 HH f64 readback")?,
        low_width: bands.low_width,
        low_height: bands.low_height,
        high_width: bands.high_width,
        high_height: bands.high_height,
    })
}

pub(super) fn dwt97_bands_to_f64_with_live_host_bytes(
    bands: CudaTranscodeDwt97Bands,
    live_host_bytes: usize,
) -> Result<Dwt97TwoDimensional<f64>, CudaTranscodeError> {
    let mut budget =
        HostPhaseBudget::with_live_bytes("CUDA 9/7 widening workspace", live_host_bytes)?;
    account_dwt97_bands(&mut budget, &bands)?;
    preflight_dwt97_conversion_budget(&budget, core::slice::from_ref(&bands), 0)?;
    dwt97_bands_to_f64_after_preflight(bands, &mut budget)
}

pub(super) fn dwt97_batch_bands_to_f64(
    bands: Vec<CudaTranscodeDwt97Bands>,
) -> Result<Vec<Dwt97TwoDimensional<f64>>, CudaTranscodeError> {
    let mut budget = HostPhaseBudget::new("CUDA 9/7 batch widening workspace");
    budget.account_vec(&bands)?;
    for item in &bands {
        account_dwt97_bands(&mut budget, item)?;
    }
    preflight_dwt97_conversion_budget(&budget, &bands, bands.len())?;
    let mut outputs =
        budget.try_vec_with_capacity(bands.len(), "CUDA 9/7 batch widened outputs")?;
    for item_bands in bands {
        outputs.push(dwt97_bands_to_f64_after_preflight(item_bands, &mut budget)?);
    }
    Ok(outputs)
}

fn account_dwt97_bands(
    budget: &mut HostPhaseBudget,
    bands: &CudaTranscodeDwt97Bands,
) -> Result<(), CudaTranscodeError> {
    budget.account_vec(&bands.ll)?;
    budget.account_vec(&bands.hl)?;
    budget.account_vec(&bands.lh)?;
    budget.account_vec(&bands.hh)?;
    Ok(())
}

pub(super) fn account_dwt97_output(
    budget: &mut HostPhaseBudget,
    bands: &Dwt97TwoDimensional<f64>,
) -> Result<(), CudaTranscodeError> {
    budget.account_vec(&bands.ll)?;
    budget.account_vec(&bands.hl)?;
    budget.account_vec(&bands.lh)?;
    budget.account_vec(&bands.hh)?;
    Ok(())
}

pub(super) fn account_reversible_output(
    budget: &mut HostPhaseBudget,
    bands: &ReversibleDwt53FirstLevel,
) -> Result<(), CudaTranscodeError> {
    budget.account_vec(&bands.ll)?;
    budget.account_vec(&bands.hl)?;
    budget.account_vec(&bands.lh)?;
    budget.account_vec(&bands.hh)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{validate_block_grid, validate_staging_and_readback_workspace};
    use crate::CudaTranscodeError;

    #[test]
    fn block_grid_validation_rejects_mismatch_and_overflow_without_allocation() {
        assert!(matches!(
            validate_block_grid(1, 2, 2, "mismatch"),
            Err(CudaTranscodeError::UnsupportedJob("mismatch"))
        ));

        assert!(matches!(
            validate_block_grid(0, usize::MAX, 2, "mismatch"),
            Err(CudaTranscodeError::HostAllocationTooLarge {
                requested: usize::MAX,
                ..
            })
        ));

        assert!(matches!(
            validate_staging_and_readback_workspace(
                2,
                4096,
                4096,
                32_768,
                32_768,
                "test dispatch workspace",
            ),
            Err(CudaTranscodeError::HostAllocationTooLarge {
                what: "test dispatch workspace",
                ..
            })
        ));
    }
}
