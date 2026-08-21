use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

include!("../../../cuda_oxide_simt_prelude.rs");

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[expect(
        clippy::too_many_arguments,
        reason = "the CUDA kernel ABI carries validated conversion and normalization scalars"
    )]
    pub unsafe fn j2k_ml_convert(
        destination: *mut u8,
        source: *const u8,
        sample_count: u64,
        channels: u32,
        source_sample: u32,
        output_kind: u32,
        layout: u32,
        destination_offset: u64,
        normalization: u32,
        mean0: f32,
        mean1: f32,
        mean2: f32,
        mean3: f32,
        std0: f32,
        std1: f32,
        std2: f32,
        std3: f32,
    ) {
        let index = thread::index_1d().get();
        if index >= sample_count as usize {
            return;
        }
        let channel = index % channels as usize;
        let pixel = index / channels as usize;
        let pixels = sample_count as usize / channels as usize;
        let output_index = destination_offset as usize
            + if layout == 0 {
                channel * pixels + pixel
            } else {
                index
            };
        let value_u16 = if source_sample == 1 {
            simt_load(source, index) as u16
        } else {
            simt_load(source.cast::<u16>(), index)
        };
        if output_kind == 1 {
            simt_store(destination, output_index, value_u16 as u8);
            return;
        }
        if output_kind == 2 {
            simt_store(destination.cast::<u16>(), output_index, value_u16);
            return;
        }
        let denominator = if source_sample == 1 { 255.0 } else { 65_535.0 };
        let mut value = value_u16 as f32;
        if normalization >= 2 {
            value /= denominator;
        }
        if normalization == 3 {
            let mean = match channel {
                0 => mean0,
                1 => mean1,
                2 => mean2,
                _ => mean3,
            };
            let std = match channel {
                0 => std0,
                1 => std1,
                2 => std2,
                _ => std3,
            };
            value = (value - mean) / std;
        }
        simt_store(destination.cast::<f32>(), output_index, value);
    }
}

fn main() {}
