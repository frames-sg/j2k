// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use super::{
    EncodeOptions, EncodeTypedComponentPlane, NativeEncodePipelineError,
    NativeEncodePipelineResult, NativeEncodeSession,
};

mod geometry;
mod multitile;
mod plan;
mod prepare;
mod single;
mod validation;

pub(in crate::j2c::encode) use prepare::{
    prepare_i64_component_packets, I64ComponentPrepareRequest,
};

pub(super) fn encode_typed_component_planes_53_i64(
    planes: &[EncodeTypedComponentPlane<'_>],
    width: u32,
    height: u32,
    options: &EncodeOptions,
    session: &NativeEncodeSession<'_>,
) -> NativeEncodePipelineResult<Vec<u8>> {
    validation::validate_high_bit_options(options)?;

    let num_components = u16::try_from(planes.len()).map_err(|_| {
        NativeEncodePipelineError::internal_invariant("validated typed component count exceeds u16")
    })?;
    if let Some((tile_width, tile_height)) = options.tile_size {
        if tile_width < width || tile_height < height {
            return multitile::encode_typed_component_planes_53_i64_multitile(
                &multitile::TypedI64MultiTileRequest {
                    planes,
                    width,
                    height,
                    options,
                    tile_width,
                    tile_height,
                    num_components,
                    session,
                },
            );
        }
    }
    single::encode_typed_component_planes_53_i64_single(
        planes,
        width,
        height,
        num_components,
        options,
        session,
    )
}

#[cfg(test)]
#[path = "typed_i64/tests.rs"]
mod tests;
