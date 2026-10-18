// SPDX-License-Identifier: MIT OR Apache-2.0

//! Optional PGM/PPM corpus for the existing CPU encoder benchmark.

use std::path::PathBuf;
use std::sync::Arc;

use super::{CaseKind, EncodeCase, JpegBackend, JpegEncodeOptions, JpegSubsampling};

pub(super) fn load_cases() -> Option<Vec<EncodeCase>> {
    let roots = j2k_test_support::paths_from_env("J2K_ENCODE_BENCH_INPUTS");
    if roots.is_empty() {
        return None;
    }
    let qualities = std::env::var("J2K_ENCODE_BENCH_QUALITIES")
        .unwrap_or_else(|_| "50,90,100".into())
        .split(',')
        .map(|value| {
            let quality: u8 = value.trim().parse().expect("JPEG benchmark quality");
            assert!((1..=100).contains(&quality));
            quality
        })
        .collect::<Vec<_>>();
    let mut paths = Vec::new();
    for path in roots {
        collect_pnm(path, &mut paths);
    }
    paths.sort();
    paths.dedup();
    assert!(
        !paths.is_empty(),
        "JPEG encoder corpus contains no PGM/PPM inputs"
    );
    let mut cases = Vec::new();
    for path in paths {
        let input = j2k_test_support::read_pnm_image(&path).expect("JPEG encoder PNM input");
        let name = path.file_stem().expect("PNM filename").to_string_lossy();
        let (kind, sampling): (_, &[(JpegSubsampling, &str)]) = if input.channels == 1 {
            (CaseKind::Gray, &[(JpegSubsampling::Gray, "gray")])
        } else {
            (
                CaseKind::Rgb,
                &[
                    (JpegSubsampling::Ybr444, "444"),
                    (JpegSubsampling::Ybr422, "422"),
                    (JpegSubsampling::Ybr420, "420"),
                ],
            )
        };
        let data: Arc<[u8]> = input.pixels.into();
        for &quality in &qualities {
            for &(subsampling, suffix) in sampling {
                cases.push(EncodeCase {
                    name: format!("{name}_q{quality}_{suffix}"),
                    width: input.width,
                    height: input.height,
                    kind,
                    data: Arc::clone(&data),
                    options: JpegEncodeOptions {
                        quality,
                        subsampling,
                        restart_interval: None,
                        backend: JpegBackend::Cpu,
                    },
                });
            }
        }
    }
    Some(cases)
}

fn collect_pnm(path: PathBuf, output: &mut Vec<PathBuf>) {
    if path.is_dir() {
        for entry in std::fs::read_dir(path).expect("JPEG encoder corpus directory") {
            collect_pnm(entry.expect("JPEG encoder corpus entry").path(), output);
        }
    } else if path
        .extension()
        .and_then(|value| value.to_str())
        .is_some_and(|extension| {
            extension.eq_ignore_ascii_case("pgm") || extension.eq_ignore_ascii_case("ppm")
        })
    {
        output.push(path);
    }
}
