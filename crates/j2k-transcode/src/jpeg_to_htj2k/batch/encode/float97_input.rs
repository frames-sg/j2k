// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    require_all_components, Float97BatchEncodeInputs, JpegToHtj2kError, PrecomputedHtj2k97Image,
    PreencodedHtj2k97CompactImage, PreencodedHtj2k97Image, PrequantizedHtj2k97Image,
};

pub(super) enum Float97BatchEncodingInput {
    Precomputed(PrecomputedHtj2k97Image),
    Compact(PreencodedHtj2k97CompactImage),
    Preencoded(PreencodedHtj2k97Image),
    Prequantized(PrequantizedHtj2k97Image),
}

pub(super) fn select_float97_batch_encoding(
    inputs: Float97BatchEncodeInputs,
) -> Result<Float97BatchEncodingInput, JpegToHtj2kError> {
    let Float97BatchEncodeInputs {
        width,
        height,
        precomputed_components,
        preencoded_compact_payload,
        preencoded_compact_components,
        preencoded_components,
        prequantized_components,
    } = inputs;

    if preencoded_compact_components.iter().any(Option::is_some) {
        drop(precomputed_components);
        drop(preencoded_components);
        drop(prequantized_components);
        let components = require_all_components(
            preencoded_compact_components,
            "9/7 compact preencoded batch transcode did not produce all components",
        )?;
        return Ok(Float97BatchEncodingInput::Compact(
            PreencodedHtj2k97CompactImage {
                width,
                height,
                bit_depth: 8,
                signed: false,
                payload: preencoded_compact_payload,
                components,
            },
        ));
    }
    if preencoded_components.iter().any(Option::is_some) {
        drop(precomputed_components);
        drop(preencoded_compact_payload);
        drop(preencoded_compact_components);
        drop(prequantized_components);
        let components = require_all_components(
            preencoded_components,
            "9/7 preencoded batch transcode did not produce all components",
        )?;
        return Ok(Float97BatchEncodingInput::Preencoded(
            PreencodedHtj2k97Image {
                width,
                height,
                bit_depth: 8,
                signed: false,
                components,
            },
        ));
    }
    if prequantized_components.iter().any(Option::is_some) {
        drop(precomputed_components);
        drop(preencoded_compact_payload);
        drop(preencoded_compact_components);
        drop(preencoded_components);
        let components = require_all_components(
            prequantized_components,
            "9/7 code-block batch transcode did not produce all components",
        )?;
        return Ok(Float97BatchEncodingInput::Prequantized(
            PrequantizedHtj2k97Image {
                width,
                height,
                bit_depth: 8,
                signed: false,
                components,
            },
        ));
    }

    drop(preencoded_compact_payload);
    drop(preencoded_compact_components);
    drop(preencoded_components);
    drop(prequantized_components);
    let components = require_all_components(
        precomputed_components,
        "9/7 batch transcode did not produce all components",
    )?;
    Ok(Float97BatchEncodingInput::Precomputed(
        PrecomputedHtj2k97Image {
            width,
            height,
            bit_depth: 8,
            signed: false,
            components,
        },
    ))
}
