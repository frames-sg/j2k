// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};

use j2k::{BatchDecodeOptions, BatchLayout, EncodedImage};
use j2k_core::{
    try_host_vec_filled, try_host_vec_with_capacity, DeviceSurface, ImageDecode, Info, PixelFormat,
};
#[cfg(feature = "cuda-runtime")]
use j2k_cuda::CudaBatchDecodeResult;
use j2k_cuda::{CudaBatchDecoder, CudaSession, J2kDecoder, Surface, SurfaceResidency};
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
    let download = match std::env::var("J2K_CUDA_PROFILE_DOWNLOAD").as_deref() {
        Ok("1") => true,
        Ok("0") | Err(std::env::VarError::NotPresent) => false,
        value => panic!("expected J2K_CUDA_PROFILE_DOWNLOAD=0 or 1, got {value:?}"),
    };
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
    let api = std::env::var("J2K_CUDA_PROFILE_API").unwrap_or_else(|_| "surface".to_owned());
    let layout = profile_layout(&api);
    let mut owned = owned_profile_decoder(&api, layout, &inputs);

    write_single_fixture_if_requested(&fixtures);

    let mut decode_batch = || match &mut owned {
        Some((decoder, encoded)) => decode_owned_batch(decoder, encoded),
        None => ProfileOutput::Surfaces(
            J2kDecoder::decode_batch_to_device_with_session(&inputs, format, &mut session)
                .expect("strict CUDA batch decode"),
        ),
    };
    // Optional readback includes completed host pixels in each timed sample.
    let max_abs_diff = validate_warmups(&mut decode_batch, &selected, format);

    let (mut samples, elapsed) = measure_batches(decode_batch, iterations, batch_size, download);
    let images = batch_size.saturating_mul(iterations);
    let images_f64 = f64::from(u32::try_from(images).expect("profile image count fits in u32"));
    let batch_pixels = selected
        .iter()
        .map(|fixture| f64::from(fixture.info.dimensions.0) * f64::from(fixture.info.dimensions.1))
        .sum::<f64>();
    let total_pixels = batch_pixels
        * f64::from(u32::try_from(iterations).expect("profile iteration count fits in u32"));
    let seconds = elapsed.as_secs_f64();
    println!("samples_us={samples:?}");
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
    let mode = if download {
        "batch_download"
    } else {
        "batch_no_download"
    };
    println!(
        "mode={mode} api={api} layout={layout:?} format={format:?} width={} height={} components={} bit_depth_min={min_bit_depth} bit_depth_max={max_bit_depth} resolution_levels={} mixed_dimensions={mixed_dimensions} mixed_resolution_levels={mixed_resolution_levels} source_inputs={} batch_unique_inputs={} exact_parity={} max_abs_diff={max_abs_diff} images={images} batch_size={batch_size} iterations={iterations} elapsed_s={seconds:.6} median_us={median_us:.3} p95_us={p95_us:.3} images_per_s={:.3} pixels_per_s={:.3}",
        first_info.dimensions.0,
        first_info.dimensions.1,
        first_info.components,
        first_info.resolution_levels,
        fixtures.len(),
        fixtures.len().min(batch_size),
        max_abs_diff == 0,
        images_f64 / seconds,
        total_pixels / seconds,
    );
    if owned.is_none() {
        let (_surfaces, profile) = J2kDecoder::decode_batch_to_device_with_session_and_profile(
            &inputs,
            format,
            &mut session,
        )
        .expect("profiled CUDA decode");
        println!("stages={profile:?}");
    }
    #[cfg(feature = "cuda-runtime")]
    println!(
        "runtime={:?}",
        owned
            .as_ref()
            .map_or(&session, |(decoder, _)| decoder.session())
            .diagnostics()
            .expect("runtime diagnostics")
    );
}

fn profile_layout(api: &str) -> BatchLayout {
    match (api, std::env::var("J2K_CUDA_PROFILE_LAYOUT").as_deref()) {
        (_, Ok("nhwc") | Err(std::env::VarError::NotPresent)) => BatchLayout::Nhwc,
        ("owned", Ok("nchw")) => BatchLayout::Nchw,
        (_, value) => panic!(
            "expected J2K_CUDA_PROFILE_LAYOUT=nhwc, or nchw with J2K_CUDA_PROFILE_API=owned, got {value:?}"
        ),
    }
}

fn owned_profile_decoder(
    api: &str,
    layout: BatchLayout,
    inputs: &[&[u8]],
) -> Option<(CudaBatchDecoder, Vec<EncodedImage>)> {
    match api {
        "surface" => None,
        "owned" => Some((
            CudaBatchDecoder::with_options(BatchDecodeOptions {
                layout,
                ..BatchDecodeOptions::default()
            }),
            inputs
                .iter()
                .map(|bytes| EncodedImage::full(Arc::from(*bytes)))
                .collect::<Vec<_>>(),
        )),
        _ => panic!("expected J2K_CUDA_PROFILE_API=surface or owned"),
    }
}

/// One decoded batch: per-image surfaces, or one dense planar group from the owned API.
enum ProfileOutput {
    Surfaces(Vec<Surface>),
    #[cfg(feature = "cuda-runtime")]
    Planar(CudaBatchDecodeResult),
}

impl ProfileOutput {
    /// Match the nvJPEG2000 harness: one zero-initialized host buffer per batch.
    fn download(&self) -> Vec<u8> {
        match self {
            Self::Surfaces(surfaces) => {
                let mut host = try_host_vec_filled(tight_batch_len(surfaces), 0u8)
                    .expect("host readback buffer");
                Surface::download_batch_tight_into(surfaces, &mut host).expect("CUDA readback");
                host
            }
            #[cfg(feature = "cuda-runtime")]
            Self::Planar(result) => {
                let dense = result.groups()[0].dense_output();
                let len = dense.ranges().iter().map(|range| range.len).sum();
                let mut host = try_host_vec_filled(len, 0u8).expect("host readback buffer");
                dense
                    .buffer()
                    .copy_range_to_host(dense.ranges()[0].offset, &mut host)
                    .expect("CUDA dense readback");
                host
            }
        }
    }

    fn assert_resident(&self, batch_size: usize) {
        match self {
            Self::Surfaces(surfaces) => {
                assert_eq!(surfaces.len(), batch_size);
                for surface in surfaces {
                    assert_eq!(surface.residency(), SurfaceResidency::CudaResidentDecode);
                    let cuda = surface.cuda_surface().expect("cuda surface");
                    std::hint::black_box(cuda.device_ptr());
                }
            }
            #[cfg(feature = "cuda-runtime")]
            Self::Planar(result) => {
                let [group] = result.groups() else {
                    panic!("planar profile batch must decode to one group")
                };
                assert_eq!(group.dense_output().ranges().len(), batch_size);
                std::hint::black_box(group.dense_output().buffer().device_ptr());
            }
        }
    }
}

fn measure_batches(
    mut decode_batch: impl FnMut() -> ProfileOutput,
    iterations: usize,
    batch_size: usize,
    download: bool,
) -> (Vec<f64>, Duration) {
    let mut samples = try_host_vec_with_capacity(iterations).expect("profile timing samples");
    let start = Instant::now();
    for _ in 0..iterations {
        let iteration_start = Instant::now();
        let output = decode_batch();
        if download {
            std::hint::black_box(output.download());
        }
        output.assert_resident(batch_size);
        samples.push(iteration_start.elapsed().as_secs_f64() * 1_000_000.0);
    }
    let elapsed = start.elapsed();
    (samples, elapsed)
}

fn decode_owned_batch(decoder: &mut CudaBatchDecoder, inputs: &[EncodedImage]) -> ProfileOutput {
    // Clone only the Arc owners here; codestream preparation remains timed.
    let result = decoder
        .decode_batch(inputs.to_vec())
        .expect("strict owned CUDA batch decode");
    assert!(result.errors().is_empty(), "{:?}", result.errors());
    assert!(
        result.group_errors().is_empty(),
        "{:?}",
        result.group_errors()
    );
    #[cfg(feature = "cuda-runtime")]
    if decoder.options().layout == BatchLayout::Nchw {
        return ProfileOutput::Planar(result);
    }
    let mut ordered = try_host_vec_with_capacity(inputs.len()).expect("ordered CUDA surfaces");
    ordered.resize_with(inputs.len(), || None);
    for group in result.into_parts().0 {
        #[cfg(feature = "cuda-runtime")]
        let (_, indices, _, _, surfaces, _dense) = group.into_parts();
        #[cfg(not(feature = "cuda-runtime"))]
        let (_, indices, _, _, surfaces) = group.into_parts();
        assert_eq!(indices.len(), surfaces.len());
        for (index, surface) in indices.into_iter().zip(surfaces) {
            assert!(ordered[index].replace(surface).is_none());
        }
    }
    ProfileOutput::Surfaces(
        ordered
            .into_iter()
            .map(|surface| surface.expect("CUDA output for every input"))
            .collect(),
    )
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

fn tight_batch_len(surfaces: &[Surface]) -> usize {
    surfaces.iter().map(DeviceSurface::byte_len).sum()
}

/// Validate pixels and populate the session's caches before measuring.
fn validate_warmups(
    decode_batch: &mut impl FnMut() -> ProfileOutput,
    fixtures: &[&ProfileFixture],
    format: PixelFormat,
) -> usize {
    let tolerance = env_tolerance("J2K_CUDA_PROFILE_MAX_ABS_DIFF");
    (0..3)
        .map(|_| validate_batch(&decode_batch(), fixtures, format, tolerance))
        .max()
        .unwrap_or(0)
}

/// Validate against the CPU references and return the largest sample difference.
fn validate_batch(
    output: &ProfileOutput,
    fixtures: &[&ProfileFixture],
    format: PixelFormat,
    tolerance: usize,
) -> usize {
    // Batches above the library's single-allocation cap still validate in caller order.
    let actual = output.download();
    let surfaces = match output {
        ProfileOutput::Surfaces(surfaces) => surfaces,
        #[cfg(feature = "cuda-runtime")]
        ProfileOutput::Planar(result) => {
            return validate_planar(result, &actual, fixtures, tolerance);
        }
    };
    assert_eq!(surfaces.len(), fixtures.len());
    let mut offset = 0usize;
    let mut max_abs_diff = 0usize;
    for (surface, fixture) in surfaces.iter().zip(fixtures) {
        assert_eq!(surface.dimensions(), fixture.info.dimensions);
        assert_eq!(surface.pixel_format(), format);
        let end = offset + fixture.reference.len();
        let diff = max_sample_diff(
            &actual[offset..end],
            &fixture.reference,
            sample_bytes(fixture),
        );
        assert!(
            diff <= tolerance,
            "CPU/CUDA output differs by {diff}, above tolerance {tolerance}"
        );
        max_abs_diff = max_abs_diff.max(diff);
        offset = end;
    }
    assert_eq!(actual.len(), offset);
    max_abs_diff
}

fn sample_bytes(fixture: &ProfileFixture) -> usize {
    if fixture.info.bit_depth > 8 {
        2
    } else {
        1
    }
}

fn max_sample_diff(actual: &[u8], expected: &[u8], sample_bytes: usize) -> usize {
    assert_eq!(actual.len(), expected.len());
    actual
        .chunks_exact(sample_bytes)
        .zip(expected.chunks_exact(sample_bytes))
        .map(|(actual, expected)| sample_value(actual).abs_diff(sample_value(expected)))
        .max()
        .unwrap_or(0)
}

fn sample_value(bytes: &[u8]) -> usize {
    match bytes {
        [value] => usize::from(*value),
        [low, high] => usize::from(u16::from_ne_bytes([*low, *high])),
        _ => panic!("profile samples are one or two bytes"),
    }
}

/// Compare each NCHW image with its interleaved CPU reference, in source order,
/// and return the largest sample difference.
#[cfg(feature = "cuda-runtime")]
fn validate_planar(
    result: &CudaBatchDecodeResult,
    actual: &[u8],
    fixtures: &[&ProfileFixture],
    tolerance: usize,
) -> usize {
    let [group] = result.groups() else {
        panic!("planar profile batch must decode to one group")
    };
    let dense = group.dense_output();
    assert_eq!(group.source_indices().len(), fixtures.len());
    let base = dense.ranges()[0].offset;
    let mut max_abs_diff = 0usize;
    for (range, &source) in dense.ranges().iter().zip(group.source_indices()) {
        let fixture = fixtures[source];
        let channels = usize::from(fixture.info.components);
        let pixels = fixture.info.dimensions.0 as usize * fixture.info.dimensions.1 as usize;
        let sample_bytes = sample_bytes(fixture);
        let image = &actual[range.offset - base..range.offset - base + range.len];
        assert_eq!(image.len(), fixture.reference.len());
        for channel in 0..channels {
            for pixel in 0..pixels {
                let planar = (channel * pixels + pixel) * sample_bytes;
                let interleaved = (pixel * channels + channel) * sample_bytes;
                let diff = sample_value(&image[planar..planar + sample_bytes]).abs_diff(
                    sample_value(&fixture.reference[interleaved..interleaved + sample_bytes]),
                );
                assert!(
                    diff <= tolerance,
                    "CPU/CUDA planar output for source {source} differs by {diff}, above tolerance {tolerance}"
                );
                max_abs_diff = max_abs_diff.max(diff);
            }
        }
    }
    max_abs_diff
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

/// Parse a per-sample tolerance, where zero (the default) requires exact output.
fn env_tolerance(name: &str) -> usize {
    std::env::var_os(name).map_or(0, |value| {
        let value = value.to_string_lossy();
        value
            .parse::<usize>()
            .unwrap_or_else(|error| panic!("invalid {name}={value}: {error}"))
    })
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
