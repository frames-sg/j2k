// SPDX-License-Identifier: Apache-2.0

use core::alloc::{GlobalAlloc, Layout};
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use signinum_jpeg::{
    encode_jpeg_baseline, JpegBackend, JpegEncodeOptions, JpegSamples, JpegSubsampling,
};
use signinum_test_support::{patterned_gray8, patterned_rgb8};

#[global_allocator]
static ALLOCATOR: CountingAllocator<std::alloc::System> =
    CountingAllocator::new(std::alloc::System);

#[derive(Debug, Clone, Copy)]
struct AllocationStats {
    allocations: usize,
    allocated_bytes: usize,
}

struct CountingAllocator<A> {
    inner: A,
    enabled: AtomicBool,
    allocations: AtomicUsize,
    allocated_bytes: AtomicUsize,
}

impl<A> CountingAllocator<A> {
    const fn new(inner: A) -> Self {
        Self {
            inner,
            enabled: AtomicBool::new(false),
            allocations: AtomicUsize::new(0),
            allocated_bytes: AtomicUsize::new(0),
        }
    }

    fn measure<R>(&self, f: impl FnOnce() -> R) -> (R, AllocationStats) {
        self.reset();
        self.enabled.store(true, Ordering::SeqCst);
        let result = f();
        self.enabled.store(false, Ordering::SeqCst);
        (result, self.stats())
    }

    fn reset(&self) {
        self.allocations.store(0, Ordering::SeqCst);
        self.allocated_bytes.store(0, Ordering::SeqCst);
    }

    fn stats(&self) -> AllocationStats {
        AllocationStats {
            allocations: self.allocations.load(Ordering::SeqCst),
            allocated_bytes: self.allocated_bytes.load(Ordering::SeqCst),
        }
    }

    fn record_alloc(&self, size: usize) {
        self.allocations.fetch_add(1, Ordering::Relaxed);
        self.allocated_bytes.fetch_add(size, Ordering::Relaxed);
    }

    fn metering_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }
}

unsafe impl<A: GlobalAlloc> GlobalAlloc for CountingAllocator<A> {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { self.inner.alloc(layout) };
        if self.metering_enabled() && !ptr.is_null() {
            self.record_alloc(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { self.inner.dealloc(ptr, layout) };
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = unsafe { self.inner.realloc(ptr, layout, new_size) };
        if self.metering_enabled() && !new_ptr.is_null() {
            self.record_alloc(new_size);
        }
        new_ptr
    }
}

#[derive(Clone, Copy)]
enum CaseKind {
    Gray,
    Rgb,
}

struct EncodeCase {
    name: &'static str,
    width: u32,
    height: u32,
    kind: CaseKind,
    data: Vec<u8>,
    options: JpegEncodeOptions,
}

impl EncodeCase {
    fn samples(&self) -> JpegSamples<'_> {
        match self.kind {
            CaseKind::Gray => JpegSamples::Gray8 {
                data: &self.data,
                width: self.width,
                height: self.height,
            },
            CaseKind::Rgb => JpegSamples::Rgb8 {
                data: &self.data,
                width: self.width,
                height: self.height,
            },
        }
    }

    fn encode(&self) -> usize {
        let encoded =
            encode_jpeg_baseline(self.samples(), self.options).expect("JPEG CPU encode bench");
        let output_bytes = encoded.data.len();
        black_box(encoded.backend);
        black_box(output_bytes)
    }
}

fn gray_case(name: &'static str, width: u32, height: u32, quality: u8) -> EncodeCase {
    EncodeCase {
        name,
        width,
        height,
        kind: CaseKind::Gray,
        data: patterned_gray8(width, height),
        options: JpegEncodeOptions {
            quality,
            subsampling: JpegSubsampling::Gray,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    }
}

fn rgb_case(
    name: &'static str,
    width: u32,
    height: u32,
    quality: u8,
    subsampling: JpegSubsampling,
    restart_interval: Option<u16>,
) -> EncodeCase {
    EncodeCase {
        name,
        width,
        height,
        kind: CaseKind::Rgb,
        data: patterned_rgb8(width, height),
        options: JpegEncodeOptions {
            quality,
            subsampling,
            restart_interval,
            backend: JpegBackend::Cpu,
        },
    }
}

fn encode_cases() -> Vec<EncodeCase> {
    vec![
        gray_case("gray8_256_default", 256, 256, 90),
        gray_case("gray8_512_default", 512, 512, 90),
        gray_case("gray8_257x263_default", 257, 263, 90),
        rgb_case(
            "rgb8_256_444_default",
            256,
            256,
            90,
            JpegSubsampling::Ybr444,
            None,
        ),
        rgb_case(
            "rgb8_256_422_default",
            256,
            256,
            90,
            JpegSubsampling::Ybr422,
            None,
        ),
        rgb_case(
            "rgb8_256_420_default",
            256,
            256,
            90,
            JpegSubsampling::Ybr420,
            None,
        ),
        rgb_case(
            "rgb8_512_444_default",
            512,
            512,
            90,
            JpegSubsampling::Ybr444,
            None,
        ),
        rgb_case(
            "rgb8_512_422_default",
            512,
            512,
            90,
            JpegSubsampling::Ybr422,
            None,
        ),
        rgb_case(
            "rgb8_512_420_default",
            512,
            512,
            90,
            JpegSubsampling::Ybr420,
            None,
        ),
        rgb_case(
            "rgb8_257x263_420_default",
            257,
            263,
            90,
            JpegSubsampling::Ybr420,
            None,
        ),
        rgb_case(
            "rgb8_512_420_high_quality",
            512,
            512,
            98,
            JpegSubsampling::Ybr420,
            None,
        ),
        rgb_case(
            "rgb8_512_420_restart_64",
            512,
            512,
            90,
            JpegSubsampling::Ybr420,
            Some(64),
        ),
    ]
}

fn report_allocations(cases: &[EncodeCase]) {
    if std::env::var_os("SIGNINUM_ALLOC_REPORT").is_none() {
        return;
    }

    for case in cases {
        black_box(case.encode());
    }

    for case in cases {
        let (output_bytes, stats) = ALLOCATOR.measure(|| case.encode());
        println!(
            "signinum_alloc case={} allocations={} allocated_bytes={} output_bytes={}",
            case.name, stats.allocations, stats.allocated_bytes, output_bytes
        );
    }
}

fn bench_encode_cpu(c: &mut Criterion) {
    let cases = encode_cases();
    report_allocations(&cases);

    let mut group = c.benchmark_group("jpeg_cpu_encode_runtime");
    for case in &cases {
        group.bench_function(case.name, |b| {
            b.iter(|| {
                black_box(case.encode());
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_encode_cpu);
criterion_main!(benches);
