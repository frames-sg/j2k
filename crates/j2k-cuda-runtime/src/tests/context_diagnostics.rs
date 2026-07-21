// SPDX-License-Identifier: MIT OR Apache-2.0

use super::cuda_runtime_gate;
use crate::CudaContext;

#[test]
fn runtime_diagnostics_count_device_to_host_transfers_when_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let before = context.diagnostics().expect("initial runtime diagnostics");

    let owned = context.upload(&[1_u8, 2, 3, 4]).expect("owned upload");
    let mut owned_out = [0_u8; 4];
    owned
        .copy_to_host(&mut owned_out)
        .expect("owned device-to-host copy");

    let pool = context.buffer_pool();
    let pooled = pool.upload(&[5_u8, 6, 7]).expect("pooled upload");
    let mut pooled_out = [0_u8; 3];
    pooled
        .copy_to_host(&mut pooled_out)
        .expect("pooled device-to-host copy");

    let after = context.diagnostics().expect("final runtime diagnostics");
    assert_eq!(
        after.device_to_host_operations - before.device_to_host_operations,
        2
    );
    assert_eq!(after.device_to_host_bytes - before.device_to_host_bytes, 7);
}

#[test]
fn completed_cuda_events_are_reused_by_the_context_when_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let before = context.diagnostics().expect("initial runtime diagnostics");
    {
        let event = context.create_event().expect("first event");
        event.record_default_stream().expect("record first event");
        event.synchronize().expect("complete first event");
    }
    let after_first = context.diagnostics().expect("first event diagnostics");
    {
        let event = context.create_event().expect("reused event");
        event.record_default_stream().expect("record reused event");
        event.synchronize().expect("complete reused event");
    }
    let after_second = context.diagnostics().expect("second event diagnostics");

    assert_eq!(
        after_first.event_driver_allocations - before.event_driver_allocations,
        1
    );
    assert_eq!(
        after_second.event_driver_allocations,
        after_first.event_driver_allocations
    );
    assert_eq!(after_second.event_reuses - after_first.event_reuses, 1);
    assert_eq!(after_second.cached_events, 1);
}
