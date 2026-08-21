mod context_diagnostics;
mod context_external;

use super::{
    f32_slice_as_bytes_mut, pool_fit_buffer_index_by_len, CudaContext, CudaError, CudaKernelName,
};

fn cuda_runtime_gate() -> bool {
    j2k_test_support::cuda_runtime_gate(module_path!())
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "kernel metadata inventory is one exact host/device parity contract"
)]
fn runtime_raii_primitives_smoke_when_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let mut pinned = context.pinned_host_buffer(16).expect("pinned host buffer");
    pinned.as_mut_slice().copy_from_slice(&[7u8; 16]);
    assert_eq!(pinned.as_slice(), &[7u8; 16]);
    let pinned_upload = context
        .upload_pinned(&[1u8, 2, 3, 4])
        .expect("pinned upload");
    let mut uploaded = [0u8; 4];
    pinned_upload
        .copy_to_host(&mut uploaded)
        .expect("download pinned upload");
    assert_eq!(uploaded, [1, 2, 3, 4]);
    let pinned_float_upload = context
        .upload_f32_pinned(&[1.25, -2.5])
        .expect("pinned f32 upload");
    let mut downloaded_float_values = [0.0f32; 2];
    pinned_float_upload
        .copy_to_host(super::f32_slice_as_bytes_mut(&mut downloaded_float_values))
        .expect("download pinned f32 upload");
    assert!((downloaded_float_values[0] - 1.25).abs() < f32::EPSILON);
    assert!((downloaded_float_values[1] + 2.5).abs() < f32::EPSILON);
    let pinned_integer_upload = context
        .upload_i32_pinned(&[7, -11])
        .expect("pinned i32 upload");
    let mut downloaded_integer_values = [0i32; 2];
    pinned_integer_upload
        .copy_to_host(super::i32_slice_as_bytes_mut(
            &mut downloaded_integer_values,
        ))
        .expect("download pinned i32 upload");
    assert_eq!(downloaded_integer_values, [7, -11]);
    let ranged_upload = context
        .upload(&[9u8, 8, 7, 6, 5, 4])
        .expect("range-copy upload");
    let mut range = [0u8; 3];
    ranged_upload
        .copy_range_to_host(2, &mut range)
        .expect("copy device range");
    assert_eq!(range, [7, 6, 5]);
    let mut uninit_range = Vec::with_capacity(3);
    ranged_upload
        .copy_range_to_host_uninit(1, uninit_range.spare_capacity_mut())
        .expect("copy device range into spare capacity");
    // SAFETY: copy_range_to_host_uninit returned success after writing
    // exactly three bytes into the Vec spare capacity.
    unsafe {
        uninit_range.set_len(3);
    }
    assert_eq!(uninit_range, [8, 7, 6]);
    let pool = context.buffer_pool();
    let pooled_upload = pool.upload(&[3u8, 1, 4, 1]).expect("pooled upload");
    let pooled_output = super::copy_pooled_bytes_to_vec_uninit(&pooled_upload, 4)
        .expect("copy pooled bytes into spare capacity");
    assert_eq!(pooled_output, [3, 1, 4, 1]);

    let module = context
        .preload_kernel_module(CudaKernelName::CopyU8)
        .expect("preload copy kernel");
    assert_eq!(module.entrypoint(), "j2k_copy_u8");

    let stream = context.create_stream().expect("CUDA stream");
    let start = context.create_event().expect("start event");
    let end = context.create_event().expect("end event");
    start.record(&stream).expect("record start");
    end.record(&stream).expect("record end");
    end.synchronize().expect("synchronize event");
    let elapsed = super::CudaEvent::elapsed_time_us(&start, &end).expect("elapsed time");
    assert!(elapsed >= 0.0);

    let pool = context.buffer_pool();
    {
        let buffer = pool.take(32).expect("pooled buffer");
        assert!(buffer.device_ptr() != 0);
        assert_eq!(buffer.byte_len(), 32);
        assert!(buffer.allocation_byte_len() >= 32);
    }
    let cached_count = pool.cached_count().expect("cached count");
    assert_eq!(cached_count, 1);
    {
        let buffer = pool.take(16).expect("reused pooled buffer");
        assert_eq!(buffer.byte_len(), 16);
        assert!(buffer.allocation_byte_len() >= 32);
    }

    let samples = [1.25f32, -2.5, 3.75, 4.5];
    {
        let buffer = pool.upload_f32(&samples).expect("pooled f32 upload");
        assert_eq!(
            buffer.byte_len(),
            samples.len() * std::mem::size_of::<f32>()
        );
        let mut downloaded = vec![0.0f32; samples.len()];
        buffer
            .copy_to_host(f32_slice_as_bytes_mut(&mut downloaded))
            .expect("download pooled f32 upload");
        assert_eq!(downloaded, samples);
    }
    let i16_samples = [-12i16, 7, 19, -4];
    {
        let buffer = pool
            .upload_i16_pinned(&i16_samples)
            .expect("pooled pinned i16 upload");
        assert_eq!(
            buffer.byte_len(),
            i16_samples.len() * std::mem::size_of::<i16>()
        );
        let mut downloaded_bytes = vec![0u8; std::mem::size_of_val(&i16_samples)];
        buffer
            .copy_to_host(&mut downloaded_bytes)
            .expect("download pooled pinned i16 upload");
        let downloaded = downloaded_bytes
            .chunks_exact(std::mem::size_of::<i16>())
            .map(|chunk| i16::from_ne_bytes([chunk[0], chunk[1]]))
            .collect::<Vec<_>>();
        assert_eq!(downloaded, i16_samples);
    }
    let cached_after_upload = pool.cached_count().expect("cached after upload");
    assert!(cached_after_upload >= cached_count);
}

#[test]
fn pooled_buffer_selection_uses_smallest_sufficient_fit() {
    let buffers = [(1usize, 32usize), (0, 64)];

    assert_eq!(
        pool_fit_buffer_index_by_len(buffers.iter().copied(), 16),
        Some(1)
    );
    let mut large_pool = (0..1024).map(|index| (index, 8usize)).collect::<Vec<_>>();
    large_pool[1022] = (1022, 32);
    large_pool[1023] = (1023, 64);

    assert_eq!(
        pool_fit_buffer_index_by_len(large_pool.iter().copied(), 16),
        Some(1022)
    );
    let mut recent_fit_pool = (0..4096).map(|index| (index, 8usize)).collect::<Vec<_>>();
    recent_fit_pool[4094] = (4094, 32);
    recent_fit_pool[4095] = (4095, 64);

    assert_eq!(
        pool_fit_buffer_index_by_len(recent_fit_pool.iter().copied(), 16),
        Some(4094)
    );
    let fallback_pool = (0..4096)
        .map(|index| match index.cmp(&3000) {
            std::cmp::Ordering::Less => (index, 8usize),
            std::cmp::Ordering::Equal => (index, 32),
            std::cmp::Ordering::Greater => (index, 64),
        })
        .collect::<Vec<_>>();

    assert_eq!(
        pool_fit_buffer_index_by_len(fallback_pool.iter().copied(), 16),
        Some(3000)
    );
}

#[test]
fn pooled_take_with_trace_reports_allocation_and_reuse_when_runtime_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let pool = context.buffer_pool();
    let (fresh, fresh_trace) = pool.take_with_trace(32).expect("fresh traced take");

    assert_eq!(fresh.byte_len(), 32);
    assert_eq!(fresh_trace.requested_len, 32);
    assert_eq!(fresh_trace.free_count_before, 0);
    assert_eq!(fresh_trace.scanned_count, 0);
    assert!(!fresh_trace.reused);
    assert!(fresh_trace.allocation_byte_len >= 32);
    drop(fresh);

    let (reused, reuse_trace) = pool.take_with_trace(16).expect("reused traced take");

    assert_eq!(reused.byte_len(), 16);
    assert_eq!(reuse_trace.requested_len, 16);
    assert_eq!(reuse_trace.free_count_before, 1);
    assert_eq!(reuse_trace.scanned_count, 1);
    assert!(reuse_trace.reused);
    assert!(reuse_trace.allocation_byte_len >= 32);
}

#[test]
fn pooled_buffer_can_detach_and_recycle_when_runtime_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let pool = context.buffer_pool();
    let raw = pool
        .take(32)
        .expect("pooled buffer")
        .into_device_buffer()
        .expect("detach pooled buffer");
    assert_eq!(pool.cached_count().expect("cached after detach"), 0);

    pool.recycle(raw).expect("explicit recycle");
    assert_eq!(pool.cached_count().expect("cached after recycle"), 1);

    let (_reused, trace) = pool.take_with_trace(16).expect("reused traced take");
    assert!(trace.reused);
    assert!(trace.allocation_byte_len >= 32);
}

#[test]
fn default_stream_timer_reports_elapsed_time_when_runtime_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let input = vec![17u8; 4096];
    let (output, elapsed_us) = context
        .time_default_stream_us(|| context.copy_with_kernel(&input))
        .expect("timed CUDA copy kernel");

    assert_eq!(output.execution().kernel_dispatches(), 1);
    assert!(elapsed_us > 0);
}

#[cfg(all(feature = "cuda-oxide-copy-u8", j2k_cuda_oxide_copy_u8_built))]
#[test]
fn cuda_oxide_copy_u8_matches_builtin_copy_and_cpu_when_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let input = (0..4099)
        .map(|index| u8::try_from((index * 31 + 17) % 251).expect("modulo 251 fits u8"))
        .collect::<Vec<_>>();

    let builtin = context
        .copy_with_kernel(&input)
        .expect("builtin CUDA copy kernel");
    let cuda_oxide = context
        .copy_with_cuda_oxide_kernel(&input)
        .expect("cuda-oxide CUDA copy kernel");

    let mut builtin_bytes = vec![0u8; input.len()];
    builtin
        .buffer()
        .copy_to_host(&mut builtin_bytes)
        .expect("download builtin CUDA copy");
    let mut cuda_oxide_bytes = vec![0u8; input.len()];
    cuda_oxide
        .buffer()
        .copy_to_host(&mut cuda_oxide_bytes)
        .expect("download cuda-oxide CUDA copy");

    assert_eq!(builtin.execution().kernel_dispatches(), 1);
    assert_eq!(cuda_oxide.execution().kernel_dispatches(), 1);
    assert_eq!(builtin_bytes, input);
    assert_eq!(cuda_oxide_bytes, input);
    assert_eq!(cuda_oxide_bytes, builtin_bytes);
}

#[test]
fn named_default_stream_timer_is_available_for_profiling_ranges_when_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let input = vec![23u8; 4096];
    let (output, elapsed_us) = context
        .time_default_stream_named_us("j2k.test.copy", || context.copy_with_kernel(&input))
        .expect("named timed CUDA copy kernel");

    assert_eq!(output.execution().kernel_dispatches(), 1);
    assert!(elapsed_us > 0);
}

#[test]
fn typed_device_view_reports_element_count_when_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let mut aligned = context.allocate(16).expect("aligned buffer");
    let view = aligned.typed_view::<u32>().expect("typed immutable view");
    assert_eq!(view.len(), 4);
    let mut_view = aligned.typed_view_mut::<u64>().expect("typed mutable view");
    assert_eq!(mut_view.len(), 2);

    let unaligned = context.allocate(3).expect("unaligned buffer");
    let error = unaligned
        .typed_view::<u16>()
        .expect_err("unaligned typed view");
    assert!(matches!(
        error,
        CudaError::LengthNotElementAligned {
            bytes: 3,
            element_size: 2
        }
    ));
}
