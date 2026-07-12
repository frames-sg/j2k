// SPDX-License-Identifier: MIT OR Apache-2.0

use std::mem::size_of;

use super::super::{
    dispatch_1d_pipeline, dispatch_3d_pipeline, new_compute_command_encoder, Buffer,
    CommandBufferRef, ComputePipelineState, Error, JpegFast420BatchParams, PreparedHuffmanHost,
};

/// Encode the split coeff-decode + IDCT-deposit passes shared by the surfaces
/// and texture drivers' `SplitCoeffIdct` debug mode.
#[cfg(all(target_os = "macos", test))]
#[derive(Clone, Copy)]
pub(in crate::compute) struct SplitCoeffIdctPasses<'a> {
    pub(in crate::compute) command_buffer: &'a CommandBufferRef,
    pub(in crate::compute) pipelines: (&'a ComputePipelineState, &'a ComputePipelineState),
    pub(in crate::compute) params: &'a JpegFast420BatchParams,
    pub(in crate::compute) quants: [&'a [u16; 64]; 3],
    pub(in crate::compute) dc_tables: &'a [PreparedHuffmanHost; 3],
    pub(in crate::compute) ac_tables: &'a [PreparedHuffmanHost; 3],
    pub(in crate::compute) entropy: (&'a Buffer, &'a Buffer, &'a Buffer, &'a Buffer),
    pub(in crate::compute) status_buffer: &'a Buffer,
    pub(in crate::compute) planes: [&'a Buffer; 3],
    pub(in crate::compute) scratch: (&'a Buffer, &'a Buffer),
    pub(in crate::compute) total_decode_threads: u32,
    pub(in crate::compute) idct_grid: (u32, u32, u32),
}

#[cfg(all(target_os = "macos", test))]
pub(in crate::compute) fn encode_split_coeff_idct_passes(
    request: SplitCoeffIdctPasses<'_>,
) -> Result<(), Error> {
    let SplitCoeffIdctPasses {
        command_buffer,
        pipelines,
        params,
        quants,
        dc_tables,
        ac_tables,
        entropy,
        status_buffer,
        planes,
        scratch,
        total_decode_threads,
        idct_grid,
    } = request;
    let (coeffs_pipeline, idct_pipeline) = pipelines;
    let (entropy_payload, entropy_offsets, entropy_lens, entropy_checkpoints) = entropy;
    let (coeff_blocks, dc_only_flags) = scratch;

    let coeff_encoder = new_compute_command_encoder(command_buffer)?;
    coeff_encoder.set_compute_pipeline_state(coeffs_pipeline);
    coeff_encoder.set_buffer(0, Some(entropy_payload), 0);
    coeff_encoder.set_buffer(1, Some(coeff_blocks), 0);
    coeff_encoder.set_buffer(2, Some(dc_only_flags), 0);
    coeff_encoder.set_bytes(
        4,
        size_of::<JpegFast420BatchParams>() as u64,
        (&raw const *params).cast(),
    );
    coeff_encoder.set_bytes(5, size_of::<[u16; 64]>() as u64, quants[0].as_ptr().cast());
    coeff_encoder.set_bytes(6, size_of::<[u16; 64]>() as u64, quants[1].as_ptr().cast());
    coeff_encoder.set_bytes(7, size_of::<[u16; 64]>() as u64, quants[2].as_ptr().cast());
    coeff_encoder.set_bytes(
        8,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[0]).cast(),
    );
    coeff_encoder.set_bytes(
        9,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[0]).cast(),
    );
    coeff_encoder.set_bytes(
        10,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[1]).cast(),
    );
    coeff_encoder.set_bytes(
        11,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[1]).cast(),
    );
    coeff_encoder.set_bytes(
        12,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[2]).cast(),
    );
    coeff_encoder.set_bytes(
        13,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[2]).cast(),
    );
    coeff_encoder.set_buffer(14, Some(entropy_offsets), 0);
    coeff_encoder.set_buffer(15, Some(entropy_lens), 0);
    coeff_encoder.set_buffer(16, Some(status_buffer), 0);
    coeff_encoder.set_buffer(17, Some(entropy_checkpoints), 0);
    dispatch_1d_pipeline(&coeff_encoder, coeffs_pipeline, total_decode_threads);
    coeff_encoder.end_encoding();

    let idct_encoder = new_compute_command_encoder(command_buffer)?;
    idct_encoder.set_compute_pipeline_state(idct_pipeline);
    idct_encoder.set_buffer(0, Some(coeff_blocks), 0);
    idct_encoder.set_buffer(1, Some(dc_only_flags), 0);
    idct_encoder.set_buffer(2, Some(planes[0]), 0);
    idct_encoder.set_buffer(3, Some(planes[1]), 0);
    idct_encoder.set_buffer(4, Some(planes[2]), 0);
    idct_encoder.set_bytes(
        5,
        size_of::<JpegFast420BatchParams>() as u64,
        (&raw const *params).cast(),
    );
    dispatch_3d_pipeline(&idct_encoder, idct_pipeline, idct_grid);
    idct_encoder.end_encoding();
    Ok(())
}
