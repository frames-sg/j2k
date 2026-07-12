// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    BlockCodingMode, EncodeComponentSampleInfo, EncodeOptions, EncodeRoiRegion, NativeEncodeSession,
};

mod execute;
mod finalize;
mod input;
mod ownership;
mod plan;
#[cfg(test)]
mod tests;
mod tile;

pub(super) use execute::encode_multitile_impl;
pub(super) use finalize::finalize_multitile_codestream;
pub(super) use input::extract_component_plane_tile_for_session;
pub(super) use ownership::{
    append_encoded_tile_parts, encode_options_retained_bytes, quantization_retained_bytes,
    reserve_tile_parts,
};
pub(super) use plan::{try_clone_options_with_component_sampling, try_copy_slice};

pub(super) struct MultiTileEncodeRequest<'request, 'input> {
    pub(super) pixels: &'request [u8],
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) num_components: u16,
    pub(super) bit_depth: u8,
    pub(super) signed: bool,
    pub(super) options: &'request EncodeOptions,
    pub(super) block_coding_mode: BlockCodingMode,
    pub(super) roi_regions: &'request [EncodeRoiRegion],
    pub(super) component_sample_info: &'request [EncodeComponentSampleInfo],
    pub(super) session: &'request NativeEncodeSession<'input>,
    pub(super) tile_width: u32,
    pub(super) tile_height: u32,
}
