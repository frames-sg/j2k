// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    time::Instant,
};

use j2k_core::{
    try_host_vec_filled, try_host_vec_with_capacity, DeviceSurface, ImageDecode, Info, PixelFormat,
};
use j2k_cuda::{CudaSession, J2kDecoder, Surface, SurfaceResidency};
use j2k_native::{encode, EncodeOptions};

const TILE_DIM: u32 = 512;
const DEFAULT_BATCH_SIZE: usize = 128;
const DEFAULT_ITERATIONS: usize = 100;

struct ProfileFixture {
    bytes: Vec<u8>,
    info: Info,
    reference: Vec<u8>,
}

fn main() {
    let batch_size = env_usize("J2K_CUDA_PROFILE_BATCH_SIZE", DEFAULT_BATCH_SIZE);
    let iterations = env_usize("J2K_CUDA_PROFILE_ITERATIONS", DEFAULT_ITERATIONS);
    let fixture_bytes = load_fixture_bytes();
    let fixtures = prepare_fixtures(fixture_bytes);
    let format = output_format(&fixtures[0].info);
    let selected = (0..batch_size)
        .map(|index| &fixtures[index % fixtures.len()])
        .collect::<Vec<_>>();
    let inputs = selected
        .iter()
        .map(|fixture| fixture.bytes.as_slice())
        .collect::<Vec<_>>();
    let mut session = CudaSession::default();

    write_single_fixture_if_requested(&fixtures);

    // Validate pixels and populate the session's caches before measuring. Only
    // compressed input to completed, device-resident output is timed below.
    for _ in 0..3 {
        let surfaces =
            J2kDecoder::decode_batch_to_device_with_session(&inputs, format, &mut session)
                .expect("warm CUDA batch decode");
        validate_batch(&surfaces, &selected, format);
    }

    let mut samples = try_host_vec_with_capacity(iterations).expect("profile timing samples");
    let start = Instant::now();
    let mut dispatches = 0usize;
    let mut ptr_xor = 0u64;
    for _ in 0..iterations {
        let iteration_start = Instant::now();
        let surfaces =
            J2kDecoder::decode_batch_to_device_with_session(&inputs, format, &mut session)
                .expect("strict CUDA batch decode");
        assert_eq!(surfaces.len(), batch_size);
        for surface in surfaces {
            assert_eq!(surface.residency(), SurfaceResidency::CudaResidentDecode);
            let cuda = surface.cuda_surface().expect("cuda surface");
            dispatches = dispatches.saturating_add(cuda.stats().decode_kernel_dispatches());
            ptr_xor ^= cuda.device_ptr();
        }
        samples.push(iteration_start.elapsed().as_secs_f64() * 1_000_000.0);
    }
    let elapsed = start.elapsed();
    let images = batch_size.saturating_mul(iterations);
    let images_f64 = f64::from(u32::try_from(images).expect("profile image count fits in u32"));
    let batch_pixels = selected
        .iter()
        .map(|fixture| f64::from(fixture.info.dimensions.0) * f64::from(fixture.info.dimensions.1))
        .sum::<f64>();
    let total_pixels = batch_pixels
        * f64::from(u32::try_from(iterations).expect("profile iteration count fits in u32"));
    let seconds = elapsed.as_secs_f64();
    samples.sort_by(f64::total_cmp);
    let median_us = samples[samples.len() / 2];
    let p95_us = samples[(samples.len() * 95 / 100).min(samples.len() - 1)];
    let first_info = &selected[0].info;
    let mixed_dimensions = selected
        .iter()
        .any(|fixture| fixture.info.dimensions != first_info.dimensions);
    let mixed_resolution_levels = selected
        .iter()
        .any(|fixture| fixture.info.resolution_levels != first_info.resolution_levels);
    let min_bit_depth = selected
        .iter()
        .map(|fixture| fixture.info.bit_depth)
        .min()
        .expect("non-empty profile batch");
    let max_bit_depth = selected
        .iter()
        .map(|fixture| fixture.info.bit_depth)
        .max()
        .expect("non-empty profile batch");
    println!(
        "mode=batch_no_download format={format:?} width={} height={} components={} bit_depth_min={min_bit_depth} bit_depth_max={max_bit_depth} resolution_levels={} mixed_dimensions={mixed_dimensions} mixed_resolution_levels={mixed_resolution_levels} source_inputs={} batch_unique_inputs={} exact_parity=true images={images} batch_size={batch_size} iterations={iterations} elapsed_s={seconds:.6} median_us={median_us:.3} p95_us={p95_us:.3} images_per_s={:.3} pixels_per_s={:.3} decode_dispatches={dispatches} ptr_xor={ptr_xor}",
        first_info.dimensions.0,
        first_info.dimensions.1,
        first_info.components,
        first_info.resolution_levels,
        fixtures.len(),
        fixtures.len().min(batch_size),
        images_f64 / seconds,
        total_pixels / seconds,
    );
    let (_surfaces, profile) =
        J2kDecoder::decode_batch_to_device_with_session_and_profile(&inputs, format, &mut session)
            .expect("profiled CUDA decode");
    println!("stages={profile:?}");
    #[cfg(feature = "cuda-runtime")]
    println!(
        "runtime={:?}",
        session.diagnostics().expect("runtime diagnostics")
    );
}

fn load_fixture_bytes() -> Vec<Vec<u8>> {
    let single = std::env::var_os("J2K_CUDA_PROFILE_INPUT");
    let roots = std::env::var_os("J2K_CUDA_PROFILE_INPUTS");
    assert!(
        single.is_none() || roots.is_none(),
        "set only one of J2K_CUDA_PROFILE_INPUT or J2K_CUDA_PROFILE_INPUTS"
    );
    if let Some(path) = single {
        return singleton_fixture(fs::read(path).expect("read profile input"));
    }
    let Some(roots) = roots else {
        return singleton_fixture(generated_fixture());
    };
    let mut paths = Vec::new();
    for root in std::env::split_paths(&roots) {
        collect_j2k_paths(&root, &mut paths)
            .unwrap_or_else(|error| panic!("scan profile input {}: {error}", root.display()));
    }
    paths.sort();
    paths.dedup();
    assert!(
        !paths.is_empty(),
        "J2K_CUDA_PROFILE_INPUTS contains no .j2k/.j2c/.jp2/.jph/.jhc inputs"
    );
    paths
        .iter()
        .map(|path| {
            fs::read(path)
                .unwrap_or_else(|error| panic!("read profile input {}: {error}", path.display()))
        })
        .collect()
}

fn collect_j2k_paths(path: &Path, paths: &mut Vec<PathBuf>) -> Result<(), std::io::Error> {
    if path.is_file() {
        paths.push(path.to_path_buf());
        return Ok(());
    }
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if file_type.is_symlink() {
            continue;
        }
        let child = entry.path();
        if file_type.is_dir() {
            collect_j2k_paths(&child, paths)?;
        } else if file_type.is_file() && is_j2k_path(&child) {
            paths.push(child);
        }
    }
    Ok(())
}

fn is_j2k_path(path: &Path) -> bool {
    path.extension()
        .and_then(|value| value.to_str())
        .is_some_and(|extension| {
            matches!(
                extension.to_ascii_lowercase().as_str(),
                "j2k" | "j2c" | "jp2" | "jph" | "jhc"
            )
        })
}

fn prepare_fixtures(fixture_bytes: Vec<Vec<u8>>) -> Vec<ProfileFixture> {
    let mut fixtures =
        try_host_vec_with_capacity(fixture_bytes.len()).expect("profile fixture owners");
    let mut expected_format = None;
    for bytes in fixture_bytes {
        let info = J2kDecoder::inspect(&bytes).expect("inspect profile input");
        let format = output_format(&info);
        if let Some(expected) = expected_format {
            assert_eq!(
                format, expected,
                "all profile inputs must use the same output pixel format"
            );
        } else {
            expected_format = Some(format);
        }
        let stride = info.dimensions.0 as usize * format.bytes_per_pixel();
        let mut reference = try_host_vec_filled(stride * info.dimensions.1 as usize, 0)
            .expect("profile reference pixels");
        J2kDecoder::new(&bytes)
            .expect("CPU decoder")
            .decode_into(&mut reference, stride, format)
            .expect("CPU reference decode");
        fixtures.push(ProfileFixture {
            bytes,
            info,
            reference,
        });
    }
    fixtures
}

fn singleton_fixture(bytes: Vec<u8>) -> Vec<Vec<u8>> {
    let mut fixtures = try_host_vec_with_capacity(1).expect("profile fixture owner");
    fixtures.push(bytes);
    fixtures
}

fn output_format(info: &Info) -> PixelFormat {
    match (info.components, info.bit_depth) {
        (1, 1..=8) => PixelFormat::Gray8,
        (1, 9..=16) => PixelFormat::Gray16,
        (3, 1..=8) => PixelFormat::Rgb8,
        (3, 9..=16) => PixelFormat::Rgb16,
        (components, bit_depth) => panic!(
            "unsupported profile input: components={components} bit_depth={bit_depth}; expected grayscale or RGB at up to 16 bits"
        ),
    }
}

fn validate_batch(surfaces: &[Surface], fixtures: &[&ProfileFixture], format: PixelFormat) {
    assert_eq!(surfaces.len(), fixtures.len());
    let actual = Surface::download_batch_tight(surfaces).expect("CUDA readback");
    let mut offset = 0usize;
    for (surface, fixture) in surfaces.iter().zip(fixtures) {
        assert_eq!(surface.dimensions(), fixture.info.dimensions);
        assert_eq!(surface.pixel_format(), format);
        let end = offset + fixture.reference.len();
        assert_eq!(
            &actual[offset..end],
            fixture.reference,
            "CPU/CUDA exact output parity"
        );
        offset = end;
    }
    assert_eq!(actual.len(), offset);
}

fn write_single_fixture_if_requested(fixtures: &[ProfileFixture]) {
    let write_fixture = std::env::var_os("J2K_CUDA_PROFILE_WRITE_FIXTURE");
    let write_reference = std::env::var_os("J2K_CUDA_PROFILE_WRITE_REFERENCE");
    if write_fixture.is_some() || write_reference.is_some() {
        assert_eq!(
            fixtures.len(),
            1,
            "profile fixture/reference output requires exactly one source input"
        );
    }
    if let Some(path) = write_fixture {
        fs::File::create_new(path)
            .expect("create new profile fixture")
            .write_all(&fixtures[0].bytes)
            .expect("write profile fixture");
    }
    if let Some(path) = write_reference {
        fs::File::create_new(path)
            .expect("create new profile reference")
            .write_all(&fixtures[0].reference)
            .expect("write profile reference");
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    let Some(value) = std::env::var_os(name) else {
        return default;
    };
    let value = value.to_string_lossy();
    let parsed = value
        .parse::<usize>()
        .unwrap_or_else(|error| panic!("invalid {name}={value}: {error}"));
    assert!(parsed > 0, "{name} must be greater than zero");
    parsed
}

fn generated_fixture() -> Vec<u8> {
    let width = u32::try_from(env_usize("J2K_CUDA_PROFILE_WIDTH", TILE_DIM as usize))
        .expect("profile width fits u32");
    let height = u32::try_from(env_usize("J2K_CUDA_PROFILE_HEIGHT", TILE_DIM as usize))
        .expect("profile height fits u32");
    let levels = u8::try_from(env_usize("J2K_CUDA_PROFILE_DECOMPOSITION_LEVELS", 1))
        .expect("profile decomposition level count fits u8");
    let reversible = match std::env::var("J2K_CUDA_PROFILE_TRANSFORM").as_deref() {
        Ok("53") | Err(std::env::VarError::NotPresent) => true,
        Ok("97") => false,
        other => panic!("expected J2K_CUDA_PROFILE_TRANSFORM=53 or 97, got {other:?}"),
    };
    let use_ht_block_coding = match std::env::var("J2K_CUDA_PROFILE_CODING").as_deref() {
        Ok("ht") | Err(std::env::VarError::NotPresent) => true,
        Ok("classic") => false,
        other => panic!("expected J2K_CUDA_PROFILE_CODING=ht or classic, got {other:?}"),
    };
    let options = EncodeOptions {
        reversible,
        use_ht_block_coding,
        num_decomposition_levels: levels,
        ..EncodeOptions::default()
    };
    match std::env::var("J2K_CUDA_PROFILE_FORMAT").as_deref() {
        Ok("rgb8") | Err(std::env::VarError::NotPresent) => {
            let mut pixels = try_host_vec_with_capacity(width as usize * height as usize * 3)
                .expect("profile fixture pixels");
            for idx in 0..width * height {
                pixels.push(u8::try_from((idx * 17 + idx / 3) & 0xff).expect("masked red fits"));
                pixels.push(u8::try_from((idx * 29 + 7) & 0xff).expect("masked green fits"));
                pixels.push(u8::try_from((idx * 43 + 19) & 0xff).expect("masked blue fits"));
            }
            encode(&pixels, width, height, 3, 8, false, &options)
                .expect("encode RGB8 profile fixture")
        }
        Ok("gray16") => {
            let mut pixels = try_host_vec_with_capacity(width as usize * height as usize * 2)
                .expect("profile fixture pixels");
            for idx in 0..u64::from(width) * u64::from(height) {
                let value = u16::try_from(
                    (idx.wrapping_mul(257) + idx / 7 + (idx >> 5).wrapping_mul(31)) & 0xffff,
                )
                .expect("masked grayscale sample fits");
                pixels.extend_from_slice(&value.to_le_bytes());
            }
            encode(&pixels, width, height, 1, 16, false, &options)
                .expect("encode Gray16 profile fixture")
        }
        other => panic!("expected J2K_CUDA_PROFILE_FORMAT=rgb8 or gray16, got {other:?}"),
    }
}
