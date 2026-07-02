// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;
use std::path::{Path, PathBuf};

use j2k_jpeg::{ColorSpace, Decoder as CpuDecoder};

const FULL_FRAME_MAX_OUTPUT_BYTES: usize = 512 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DecodeMode {
    Gray,
    Rgb,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CorpusInputClass {
    BoundedFullFrame,
    VeryLarge,
}

#[derive(Clone)]
pub(crate) struct BenchInput {
    pub(crate) name: String,
    pub(crate) bytes: Vec<u8>,
    pub(crate) dimensions: (u32, u32),
    pub(crate) mode: DecodeMode,
    pub(crate) input_class: CorpusInputClass,
}

pub(crate) fn load_bench_inputs(mut inputs: Vec<BenchInput>) -> Vec<BenchInput> {
    let mut seen = inputs
        .iter()
        .map(|input| input.name.clone())
        .collect::<Vec<_>>();
    for path in std::env::split_paths(&std::env::var_os("J2K_BENCH_INPUTS").unwrap_or_default()) {
        collect_jpegs(&path, &mut inputs, &mut seen);
    }
    inputs.sort_by(|a, b| a.name.cmp(&b.name));
    inputs
}

fn collect_jpegs(path: &Path, inputs: &mut Vec<BenchInput>, seen: &mut Vec<String>) {
    if path.is_file() {
        push_jpeg(path, inputs, seen);
        return;
    }
    if !path.is_dir() {
        return;
    }

    let mut stack = vec![path.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let child = entry.path();
            if child.is_dir() {
                stack.push(child);
            } else {
                push_jpeg(&child, inputs, seen);
            }
        }
    }
}

fn push_jpeg(path: &Path, inputs: &mut Vec<BenchInput>, seen: &mut Vec<String>) {
    if !is_jpeg(path) {
        return;
    }
    let Ok(bytes) = fs::read(path) else {
        return;
    };
    let Ok(decoder) = CpuDecoder::new(&bytes) else {
        return;
    };
    let Some(mode) = color_space_mode(decoder.info().color_space) else {
        return;
    };
    let name = relative_name(path);
    if seen.contains(&name) {
        return;
    }

    seen.push(name.clone());
    let dimensions = decoder.info().dimensions;
    inputs.push(BenchInput {
        name,
        bytes,
        dimensions,
        mode,
        input_class: classify_corpus_input(dimensions, mode),
    });
}

fn is_jpeg(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| matches!(ext.to_ascii_lowercase().as_str(), "jpg" | "jpeg"))
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

fn color_space_mode(color_space: ColorSpace) -> Option<DecodeMode> {
    match color_space {
        ColorSpace::Grayscale => Some(DecodeMode::Gray),
        ColorSpace::YCbCr | ColorSpace::Rgb => Some(DecodeMode::Rgb),
        ColorSpace::Cmyk | ColorSpace::Ycck => None,
    }
}

fn classify_corpus_input(dimensions: (u32, u32), mode: DecodeMode) -> CorpusInputClass {
    let bpp = match mode {
        DecodeMode::Gray => 1usize,
        DecodeMode::Rgb => 3usize,
    };
    let bytes = usize::try_from(dimensions.0)
        .ok()
        .zip(usize::try_from(dimensions.1).ok())
        .and_then(|(width, height)| width.checked_mul(height))
        .and_then(|pixels| pixels.checked_mul(bpp));
    match bytes {
        Some(bytes) if bytes <= FULL_FRAME_MAX_OUTPUT_BYTES => CorpusInputClass::BoundedFullFrame,
        _ => CorpusInputClass::VeryLarge,
    }
}
