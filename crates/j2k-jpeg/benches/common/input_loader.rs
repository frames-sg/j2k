// SPDX-License-Identifier: MIT OR Apache-2.0

use super::classification::{
    classify_corpus_input, color_space_mode, CorpusInputClass, DecodeMode,
};
use j2k_jpeg::Decoder;
use j2k_test_support::{JPEG_BASELINE_420_16X16, JPEG_GRAYSCALE_8X8};
use std::path::{Path, PathBuf};

struct LoadedInput {
    name: String,
    bytes: Vec<u8>,
    dimensions: (u32, u32),
    mode: DecodeMode,
    input_class: CorpusInputClass,
}

pub(crate) fn load_inputs<T>(
    mut project: impl FnMut(String, Vec<u8>, (u32, u32), DecodeMode, CorpusInputClass) -> T,
) -> Vec<T> {
    let mut inputs = vec![
        LoadedInput {
            name: "repo/baseline_420_16x16".to_string(),
            bytes: JPEG_BASELINE_420_16X16.to_vec(),
            dimensions: (16, 16),
            mode: DecodeMode::Rgb,
            input_class: CorpusInputClass::BoundedFullFrame,
        },
        LoadedInput {
            name: "repo/grayscale_8x8".to_string(),
            bytes: JPEG_GRAYSCALE_8X8.to_vec(),
            dimensions: (8, 8),
            mode: DecodeMode::Gray,
            input_class: CorpusInputClass::BoundedFullFrame,
        },
    ];

    let mut seen = inputs
        .iter()
        .map(|input| input.name.clone())
        .collect::<Vec<_>>();
    for path in j2k_test_support::paths_from_env("J2K_BENCH_INPUTS") {
        collect_jpegs(&path, &mut inputs, &mut seen);
    }

    inputs.sort_by(|a, b| a.name.cmp(&b.name));
    inputs
        .into_iter()
        .map(|input| {
            project(
                input.name,
                input.bytes,
                input.dimensions,
                input.mode,
                input.input_class,
            )
        })
        .collect()
}

fn collect_jpegs(path: &Path, inputs: &mut Vec<LoadedInput>, seen: &mut Vec<String>) {
    for path in j2k_test_support::collect_jpeg_paths(path) {
        push_jpeg(&path, inputs, seen);
    }
}

fn push_jpeg(path: &Path, inputs: &mut Vec<LoadedInput>, seen: &mut Vec<String>) {
    let Ok(bytes) = std::fs::read(path) else {
        return;
    };
    let Ok(dec) = Decoder::new(&bytes) else {
        return;
    };
    let Some(mode) = color_space_mode(dec.info().color_space) else {
        return;
    };
    let dimensions = dec.info().dimensions;
    let input_class = classify_corpus_input(dimensions, mode);

    let name = relative_name(path);
    if seen.contains(&name) {
        return;
    }
    seen.push(name.clone());
    inputs.push(LoadedInput {
        name,
        bytes,
        dimensions,
        mode,
        input_class,
    });
}

fn relative_name(path: &Path) -> String {
    let absolute = path.canonicalize().unwrap_or_else(|_| PathBuf::from(path));
    if let Some(prefix) = std::env::var_os("HOME") {
        let prefix = PathBuf::from(prefix);
        if let Ok(stripped) = absolute.strip_prefix(prefix) {
            return stripped.display().to_string();
        }
    }
    absolute.display().to_string()
}
