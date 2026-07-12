// SPDX-License-Identifier: MIT OR Apache-2.0

//! Retained metadata and phase-wide workspace planning for JPEG decode.

use alloc::vec::Vec;

use crate::allocation::{
    checked_add_allocation_bytes, checked_allocation_bytes, ensure_allocation_bytes,
};
use crate::context::MAX_DECODER_CONTEXT_ALLOCATION_BYTES;
use crate::error::Warning;

use super::warning_ownership::warning_merge_peak_bytes;
use super::{
    checked_usize_product, Decoder, HuffmanTable, JpegError, ParsedHeader, PreparedDecodePlan,
    PreparedProgressiveComponentPlan, PreparedProgressiveScan, PreparedProgressiveScanComponent,
    SofKind, COMPONENT_IMAGE_METADATA_BYTES, DEFAULT_MAX_DECODE_BYTES,
};

impl Decoder<'_> {
    /// Exact retained host bytes that remain live while the CPU checkpoint
    /// cache grows. The cache itself is deliberately excluded so replacement
    /// growth can count the old and new cache capacities exactly once.
    pub(crate) fn retained_allocation_bytes_excluding_cpu_checkpoint_cache(
        &self,
    ) -> Result<usize, JpegError> {
        let mut total = MAX_DECODER_CONTEXT_ALLOCATION_BYTES;
        total = checked_add_allocation_bytes(total, self.plan.retained_allocation_bytes()?)?;
        if let Some(progressive) = &self.progressive_plan {
            total = checked_add_allocation_bytes(total, progressive.retained_allocation_bytes()?)?;
        }
        checked_add_allocation_bytes(
            total,
            checked_allocation_bytes::<Warning>(self.warnings.capacity())?,
        )
    }

    pub(crate) fn decode_workspace_cap(&self) -> Result<usize, JpegError> {
        let mut prepared_bytes = self.plan.retained_allocation_bytes()?;
        if let Some(progressive) = &self.progressive_plan {
            prepared_bytes = checked_add_allocation_bytes(
                prepared_bytes,
                progressive.retained_allocation_bytes()?,
            )?;
        }
        let checkpoint_bytes = self
            .cpu_entropy_checkpoints
            .lock()
            .map_err(|_| JpegError::InternalInvariant {
                reason: "CPU entropy checkpoint cache mutex poisoned",
            })?
            .retained_allocation_bytes()?;
        prepared_bytes = checked_add_allocation_bytes(prepared_bytes, checkpoint_bytes)?;
        decode_workspace_cap_for_warning_capacity(self.warnings.capacity(), prepared_bytes)
    }

    pub(super) fn decode_phase_live_bytes(
        &self,
        external_live_bytes: usize,
    ) -> Result<(usize, usize), JpegError> {
        let workspace_cap = self.decode_workspace_cap()?;
        let scratch_bytes = self.decode_scratch_bytes(workspace_cap)?;
        let requested =
            external_live_bytes
                .checked_add(scratch_bytes)
                .ok_or(JpegError::MemoryCapExceeded {
                    requested: usize::MAX,
                    cap: workspace_cap,
                })?;
        if requested > workspace_cap {
            return Err(JpegError::MemoryCapExceeded {
                requested,
                cap: workspace_cap,
            });
        }
        Ok((requested, workspace_cap))
    }
}

pub(super) fn progressive_prepared_allocation_bytes(
    component_count: usize,
    scan_count: usize,
    scan_component_count: usize,
    huffman_table_count: usize,
) -> Result<usize, JpegError> {
    let mut total = checked_allocation_bytes::<PreparedProgressiveComponentPlan>(component_count)?;
    total = checked_add_allocation_bytes(
        total,
        checked_allocation_bytes::<PreparedProgressiveScanComponent>(scan_component_count)?,
    )?;
    total = checked_add_allocation_bytes(
        total,
        checked_allocation_bytes::<HuffmanTable>(huffman_table_count)?,
    )?;
    total = checked_add_allocation_bytes(
        total,
        checked_allocation_bytes::<PreparedProgressiveScan>(scan_count)?,
    )?;
    checked_add_allocation_bytes(
        total,
        prepared_decode_plan_allocation_bytes(component_count, 0)?,
    )
}

pub(super) fn prepared_decode_plan_allocation_bytes(
    component_count: usize,
    huffman_table_count: usize,
) -> Result<usize, JpegError> {
    PreparedDecodePlan::allocation_bytes_for_counts(component_count, huffman_table_count)
}

pub(super) fn ensure_prepared_construction_fits(
    header: &ParsedHeader,
    prepared_bytes: usize,
) -> Result<(), JpegError> {
    let metadata_bytes =
        checked_add_allocation_bytes(header.retained_allocation_bytes()?, prepared_bytes)?;
    let requested =
        checked_add_allocation_bytes(metadata_bytes, MAX_DECODER_CONTEXT_ALLOCATION_BYTES)?;
    ensure_allocation_bytes(requested)
}

pub(super) fn decode_workspace_cap(
    header: &ParsedHeader,
    prepared_bytes: usize,
) -> Result<usize, JpegError> {
    decode_workspace_cap_for_warning_capacity(header.warnings.capacity(), prepared_bytes)
}

fn decode_workspace_cap_for_warning_capacity(
    warning_capacity: usize,
    prepared_bytes: usize,
) -> Result<usize, JpegError> {
    // At outcome construction the decoder retains its header warnings while
    // the scan-warning vector and the merged public warning vector coexist.
    // The shared helper also reserves allocator over-capacity headroom for the
    // merged owner and must stay identical to runtime warning construction.
    let warning_peak = warning_merge_peak_bytes(warning_capacity)?;
    let retained_bytes = checked_add_allocation_bytes(prepared_bytes, warning_peak)?;
    let reserved =
        checked_add_allocation_bytes(MAX_DECODER_CONTEXT_ALLOCATION_BYTES, retained_bytes)?;
    ensure_allocation_bytes(reserved)?;
    Ok(DEFAULT_MAX_DECODE_BYTES - reserved)
}

pub(super) fn compute_progressive_scratch_bytes(
    components: &[PreparedProgressiveComponentPlan],
    output_width: usize,
    sof_kind: SofKind,
    cap: usize,
) -> Result<usize, JpegError> {
    let coefficient_outer = checked_usize_product(
        &[components.len(), core::mem::size_of::<Vec<[i32; 64]>>()],
        cap,
    )?;
    let image_outer =
        checked_usize_product(&[components.len(), COMPONENT_IMAGE_METADATA_BYTES], cap)?;
    let dc_predictors =
        checked_usize_product(&[components.len(), core::mem::size_of::<i32>()], cap)?;
    let mut coefficient_payload = 0usize;
    let mut plane_samples = 0usize;
    for component in components {
        let blocks = checked_usize_product(
            &[component.block_cols as usize, component.block_rows as usize],
            cap,
        )?;
        let coeffs = checked_usize_product(&[blocks, core::mem::size_of::<[i32; 64]>()], cap)?;
        coefficient_payload = checked_workspace_add(coefficient_payload, coeffs, cap)?;
        let samples = checked_usize_product(
            &[
                component.block_cols as usize,
                component.block_rows as usize,
                64,
            ],
            cap,
        )?;
        plane_samples = checked_workspace_add(plane_samples, samples, cap)?;
    }

    // Component rendering owns one grayscale or three color rows. Row-sink
    // decoding simultaneously detaches two RGB rows from `ScratchPool`.
    let row_count = if components.len() == 1 { 7 } else { 9 };
    let row_payload = checked_usize_product(&[output_width, row_count], cap)?;
    let eight_bit_render = checked_workspace_add(
        checked_workspace_add(image_outer, plane_samples, cap)?,
        row_payload,
        cap,
    )?;
    let render_phase = if sof_kind == SofKind::Progressive12 {
        let extended12_planes = checked_usize_product(&[plane_samples, 2], cap)?;
        eight_bit_render.max(extended12_planes)
    } else {
        eight_bit_render
    };
    let phase_peak = dc_predictors.max(render_phase);
    checked_workspace_add(
        checked_workspace_add(coefficient_outer, coefficient_payload, cap)?,
        phase_peak,
        cap,
    )
}

fn checked_workspace_add(left: usize, right: usize, cap: usize) -> Result<usize, JpegError> {
    let requested = left
        .checked_add(right)
        .ok_or(JpegError::MemoryCapExceeded {
            requested: usize::MAX,
            cap,
        })?;
    if requested > cap {
        return Err(JpegError::MemoryCapExceeded { requested, cap });
    }
    Ok(requested)
}

#[cfg(test)]
mod tests {
    use super::super::PreparedComponentPlan;
    use super::*;
    use core::mem::size_of;
    use j2k_test_support::JPEG_BASELINE_420_16X16;

    fn progressive_component() -> PreparedProgressiveComponentPlan {
        PreparedProgressiveComponentPlan {
            h: 1,
            v: 1,
            output_index: 0,
            quant: [1; 64],
            block_cols: 1,
            block_rows: 1,
            sample_width: 8,
            sample_height: 8,
        }
    }

    #[test]
    fn progressive_prepared_formula_counts_flattened_and_host_plan_buffers() {
        let components = 4usize;
        let scans = 9usize;
        let scan_components = 12usize;
        let huffman_tables = 7usize;
        let expected = components * size_of::<PreparedProgressiveComponentPlan>()
            + scan_components * size_of::<PreparedProgressiveScanComponent>()
            + huffman_tables * size_of::<HuffmanTable>()
            + scans * size_of::<PreparedProgressiveScan>()
            + components * size_of::<PreparedComponentPlan>();
        assert_eq!(
            progressive_prepared_allocation_bytes(
                components,
                scans,
                scan_components,
                huffman_tables,
            )
            .unwrap(),
            expected
        );
    }

    #[test]
    fn prepared_decode_formula_counts_one_inline_arena() {
        let expected = size_of::<PreparedComponentPlan>() + 2 * size_of::<HuffmanTable>();
        assert_eq!(
            prepared_decode_plan_allocation_bytes(1, 2).unwrap(),
            expected
        );
    }

    #[test]
    fn retained_baseline_excludes_existing_cpu_checkpoint_cache_capacity() {
        let decoder = Decoder::new(JPEG_BASELINE_420_16X16).expect("valid JPEG fixture");
        let before = decoder
            .retained_allocation_bytes_excluding_cpu_checkpoint_cache()
            .expect("bounded decoder baseline");
        let workspace_before = decoder
            .decode_workspace_cap()
            .expect("empty-cache workspace cap");
        let retained_checkpoint_bytes = {
            let mut cache = decoder
                .cpu_entropy_checkpoints
                .lock()
                .expect("checkpoint cache lock");
            cache
                .checkpoints
                .try_reserve_exact(8)
                .expect("bounded test cache");
            let retained = cache.retained_allocation_bytes().expect("cache bytes");
            assert!(retained > 0);
            retained
        };
        let after = decoder
            .retained_allocation_bytes_excluding_cpu_checkpoint_cache()
            .expect("bounded decoder baseline");

        assert_eq!(before, after);
        assert_eq!(
            decoder
                .decode_workspace_cap()
                .expect("populated-cache workspace cap"),
            workspace_before - retained_checkpoint_bytes
        );
    }

    #[test]
    fn runtime_metadata_and_context_reserve_reduce_progressive_scratch_cap() {
        let warning_peak = 2 * size_of::<Warning>();
        let expected =
            DEFAULT_MAX_DECODE_BYTES - MAX_DECODER_CONTEXT_ALLOCATION_BYTES - warning_peak;
        assert_eq!(
            decode_workspace_cap_for_warning_capacity(0, 0).unwrap(),
            expected
        );

        let maximum_prepared = expected;
        assert_eq!(
            decode_workspace_cap_for_warning_capacity(0, maximum_prepared).unwrap(),
            0
        );
        assert!(matches!(
            decode_workspace_cap_for_warning_capacity(0, maximum_prepared + 1),
            Err(JpegError::MemoryCapExceeded { .. })
        ));
    }

    #[test]
    fn progressive_scratch_counts_every_simultaneous_outer_and_payload() {
        let expected = size_of::<Vec<[i32; 64]>>()
            + size_of::<[i32; 64]>()
            + (COMPONENT_IMAGE_METADATA_BYTES + 64 + 8 * 7).max(size_of::<i32>());
        assert_eq!(
            compute_progressive_scratch_bytes(
                &[progressive_component()],
                8,
                SofKind::Progressive8,
                DEFAULT_MAX_DECODE_BYTES,
            )
            .unwrap(),
            expected
        );
    }

    #[test]
    fn progressive_scratch_rejects_aggregate_render_peak() {
        let component = progressive_component();
        assert!(compute_progressive_scratch_bytes(
            core::slice::from_ref(&component),
            8,
            SofKind::Progressive8,
            600,
        )
        .is_ok());
        assert!(matches!(
            compute_progressive_scratch_bytes(
                &[component.clone(), component],
                8,
                SofKind::Progressive8,
                600,
            ),
            Err(JpegError::MemoryCapExceeded {
                requested,
                cap: 600
            }) if requested > 600
        ));
    }

    #[test]
    fn progressive12_scratch_counts_live_u16_render_planes() {
        let component = progressive_component();
        let coefficient_bytes = size_of::<Vec<[i32; 64]>>() + size_of::<[i32; 64]>();
        let eight_bit_render = COMPONENT_IMAGE_METADATA_BYTES + 64 + 7;
        let extended12_render = 64 * size_of::<u16>();
        let expected = coefficient_bytes + eight_bit_render.max(extended12_render);

        assert_eq!(
            compute_progressive_scratch_bytes(
                &[component],
                1,
                SofKind::Progressive12,
                DEFAULT_MAX_DECODE_BYTES,
            )
            .unwrap(),
            expected
        );
    }
}
