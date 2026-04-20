//! Codec benchmarks using Criterion.
//!
//! Run with: `cargo bench -p openjph-core`

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use openjph_core::codestream::Codestream;
use openjph_core::file::{MemInfile, MemOutfile};
use openjph_core::types::{Point, Size};

// ---------------------------------------------------------------------------
// Helper: encode an 8-bit single-component image to HTJ2K
// ---------------------------------------------------------------------------

fn encode_image(
    width: u32,
    height: u32,
    pixels: &[i32],
    num_decomps: u32,
    reversible: bool,
) -> Vec<u8> {
    let mut cs = Codestream::new();
    cs.access_siz_mut()
        .set_image_extent(Point::new(width, height));
    cs.access_siz_mut().set_num_components(1);
    cs.access_siz_mut()
        .set_comp_info(0, Point::new(1, 1), 8, false);
    cs.access_siz_mut().set_tile_size(Size::new(width, height));
    cs.access_cod_mut().set_num_decomposition(num_decomps);
    cs.access_cod_mut().set_reversible(reversible);
    cs.access_cod_mut().set_color_transform(false);
    cs.set_planar(0);

    let mut outfile = MemOutfile::new();
    cs.write_headers(&mut outfile, &[]).unwrap();
    for y in 0..height as usize {
        let start = y * width as usize;
        cs.exchange(&pixels[start..start + width as usize], 0)
            .unwrap();
    }
    cs.flush(&mut outfile).unwrap();
    outfile.get_data().to_vec()
}

fn decode_image(data: &[u8], height: u32) -> Vec<Vec<i32>> {
    let mut infile = MemInfile::new(data);
    let mut cs = Codestream::new();
    cs.read_headers(&mut infile).unwrap();
    cs.create(&mut infile).unwrap();

    let mut lines = Vec::with_capacity(height as usize);
    for _ in 0..height {
        if let Some(line) = cs.pull(0) {
            lines.push(line);
        }
    }
    lines
}

fn generate_gradient(width: u32, height: u32) -> Vec<i32> {
    let mut pixels = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((x + y) % 256) as i32);
        }
    }
    pixels
}

// ---------------------------------------------------------------------------
// Encode benchmarks
// ---------------------------------------------------------------------------

fn bench_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode");

    for &(w, h) in &[(8, 8), (64, 64), (256, 256)] {
        let pixels = generate_gradient(w, h);
        let pixel_count = (w * h) as u64;
        let decomps = if w >= 32 { 1 } else { 0 };

        group.throughput(Throughput::Elements(pixel_count));
        group.bench_with_input(
            BenchmarkId::new("rev53", format!("{}x{}", w, h)),
            &(&pixels, w, h, decomps),
            |b, &(px, w, h, d)| {
                b.iter(|| encode_image(w, h, px, d, true));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("irv97", format!("{}x{}", w, h)),
            &(&pixels, w, h, decomps),
            |b, &(px, w, h, d)| {
                b.iter(|| encode_image(w, h, px, d, false));
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Decode benchmarks
// ---------------------------------------------------------------------------

fn bench_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode");

    for &(w, h) in &[(8, 8), (64, 64), (256, 256)] {
        let pixels = generate_gradient(w, h);
        let decomps = if w >= 32 { 1 } else { 0 };
        let pixel_count = (w * h) as u64;

        let encoded_rev = encode_image(w, h, &pixels, decomps, true);
        group.throughput(Throughput::Elements(pixel_count));
        group.bench_with_input(
            BenchmarkId::new("rev53", format!("{}x{}", w, h)),
            &(&encoded_rev, h),
            |b, &(data, h)| {
                b.iter(|| decode_image(data, h));
            },
        );

        let encoded_irv = encode_image(w, h, &pixels, decomps, false);
        group.bench_with_input(
            BenchmarkId::new("irv97", format!("{}x{}", w, h)),
            &(&encoded_irv, h),
            |b, &(data, h)| {
                b.iter(|| decode_image(data, h));
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// DWT roundtrip benchmarks (varying decomposition levels)
// ---------------------------------------------------------------------------

fn bench_dwt_levels(c: &mut Criterion) {
    let mut group = c.benchmark_group("dwt_levels");

    let (w, h) = (64u32, 64u32);
    let pixels = generate_gradient(w, h);
    let pixel_count = (w * h) as u64;

    for &decomps in &[0, 1] {
        let label = format!("{}decomp_{}x{}", decomps, w, h);

        group.throughput(Throughput::Elements(pixel_count));
        group.bench_function(BenchmarkId::new("rev53", &label), |b| {
            b.iter(|| {
                let encoded = encode_image(w, h, &pixels, decomps, true);
                decode_image(&encoded, h)
            });
        });

        group.bench_function(BenchmarkId::new("irv97", &label), |b| {
            b.iter(|| {
                let encoded = encode_image(w, h, &pixels, decomps, false);
                decode_image(&encoded, h)
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Full pipeline benchmark (encode + decode)
// ---------------------------------------------------------------------------

fn bench_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("roundtrip");

    for &(w, h) in &[(8, 8), (64, 64)] {
        let pixels = generate_gradient(w, h);
        let decomps = if w >= 32 { 1 } else { 0 };
        let pixel_count = (w * h) as u64;

        group.throughput(Throughput::Elements(pixel_count));
        group.bench_function(BenchmarkId::new("lossless", format!("{}x{}", w, h)), |b| {
            b.iter(|| {
                let encoded = encode_image(w, h, &pixels, decomps, true);
                decode_image(&encoded, h)
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion groups
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_encode,
    bench_decode,
    bench_dwt_levels,
    bench_roundtrip
);
criterion_main!(benches);
