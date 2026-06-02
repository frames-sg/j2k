// SPDX-License-Identifier: Apache-2.0

use std::time::Instant;

use signinum_core::PixelFormat;
use signinum_j2k_cuda::{CudaSession, J2kDecoder, SurfaceResidency};
use signinum_j2k_native::{encode_htj2k, EncodeOptions};

const TILE_DIM: u32 = 512;
const DEFAULT_BATCH_SIZE: usize = 128;
const DEFAULT_ITERATIONS: usize = 4;

fn main() {
    let batch_size = env_usize("SIGNINUM_J2K_CUDA_PROFILE_BATCH_SIZE", DEFAULT_BATCH_SIZE);
    let iterations = env_usize("SIGNINUM_J2K_CUDA_PROFILE_ITERATIONS", DEFAULT_ITERATIONS);
    let fixture = htj2k_rgb8_fixture(TILE_DIM, TILE_DIM);
    let fixtures = vec![fixture; batch_size];
    let mut session = CudaSession::default();

    let start = Instant::now();
    let mut dispatches = 0usize;
    let mut ptr_xor = 0u64;
    for _ in 0..iterations {
        for fixture in &fixtures {
            let mut decoder = J2kDecoder::new(fixture).expect("decoder");
            let surface = decoder
                .decode_to_device_with_session(PixelFormat::Rgb8, &mut session)
                .expect("strict CUDA HTJ2K RGB8 decode");
            assert_eq!(surface.residency(), SurfaceResidency::CudaResidentDecode);
            let cuda = surface.cuda_surface().expect("cuda surface");
            dispatches = dispatches.saturating_add(cuda.stats().decode_kernel_dispatches());
            ptr_xor ^= cuda.device_ptr();
        }
    }
    let elapsed = start.elapsed();
    let tiles = batch_size.saturating_mul(iterations);
    let tiles_f64 = f64::from(u32::try_from(tiles).expect("profile tile count fits in u32"));
    let seconds = elapsed.as_secs_f64();
    println!(
        "tiles={tiles} batch_size={batch_size} iterations={iterations} elapsed_s={seconds:.6} tiles_per_s={:.3} decode_dispatches={dispatches} ptr_xor={ptr_xor}",
        tiles_f64 / seconds
    );
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

fn htj2k_rgb8_fixture(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(width as usize * height as usize * 3);
    for idx in 0..width * height {
        pixels.push(u8::try_from((idx * 17 + idx / 3) & 0xff).expect("masked red fits"));
        pixels.push(u8::try_from((idx * 29 + 7) & 0xff).expect("masked green fits"));
        pixels.push(u8::try_from((idx * 43 + 19) & 0xff).expect("masked blue fits"));
    }
    let options = EncodeOptions {
        reversible: true,
        use_ht_block_coding: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    encode_htj2k(&pixels, width, height, 3, 8, false, &options).expect("encode RGB HTJ2K fixture")
}
