// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use super::*;

fn failure(message: &str) -> CudaError {
    CudaError::InvalidArgument {
        message: message.to_string(),
    }
}

#[test]
fn recovered_operation_failure_keeps_context_available() {
    let lifecycle = ContextResourceLifecycle::new();
    let recovery_ran = AtomicBool::new(false);
    let first = lifecycle.run_recoverable(
        || Ok(()),
        || Err::<(), _>(failure("driver failure")),
        || {
            recovery_ran.store(true, Ordering::Relaxed);
            Ok(())
        },
    );
    assert!(matches!(first, Err(CudaError::InvalidArgument { .. })));
    assert!(recovery_ran.load(Ordering::Relaxed));
    assert!(!lifecycle.is_poisoned());

    let later_ran = AtomicBool::new(false);
    let later = lifecycle.run_recoverable(
        || Ok(()),
        || {
            later_ran.store(true, Ordering::Relaxed);
            Ok(())
        },
        || Ok(()),
    );
    assert!(later.is_ok());
    assert!(later_ran.load(Ordering::Relaxed));
}

#[test]
fn failed_operation_recovery_poisons_and_blocks_later_work() {
    let lifecycle = ContextResourceLifecycle::new();
    let first = lifecycle.run_recoverable(
        || Ok(()),
        || Err::<(), _>(failure("driver failure")),
        || Err(failure("completion failure")),
    );
    assert!(matches!(first, Err(CudaError::CompletionFailed { .. })));
    assert!(lifecycle.is_poisoned());

    let later_ran = AtomicBool::new(false);
    let later = lifecycle.run_recoverable(
        || Ok(()),
        || {
            later_ran.store(true, Ordering::Relaxed);
            Ok(())
        },
        || Ok(()),
    );
    assert!(matches!(later, Err(CudaError::StatePoisoned { .. })));
    assert!(!later_ran.load(Ordering::Relaxed));
}

#[test]
fn direct_completion_failure_poisons_context() {
    let lifecycle = ContextResourceLifecycle::new();
    let result =
        lifecycle.run_completion(|| Ok(()), || Err::<(), _>(failure("completion failure")));
    assert!(matches!(result, Err(CudaError::InvalidArgument { .. })));
    assert!(lifecycle.is_poisoned());
}

#[test]
fn stateful_operation_failure_quarantines_context_after_successful_completion() {
    let lifecycle = ContextResourceLifecycle::new();
    let result = lifecycle.run_stateful(
        || Ok(()),
        || Err::<(), _>(failure("resource transition failed")),
        || Ok(()),
    );

    assert!(matches!(result, Err(CudaError::InvalidArgument { .. })));
    assert!(lifecycle.is_poisoned());
}

#[test]
fn stateful_operation_and_completion_failures_are_both_retained() {
    let lifecycle = ContextResourceLifecycle::new();
    let result = lifecycle.run_stateful(
        || Ok(()),
        || Err::<(), _>(failure("resource transition failed")),
        || Err(failure("completion failed")),
    );

    assert!(matches!(result, Err(CudaError::CompletionFailed { .. })));
    assert!(lifecycle.is_poisoned());
}

#[test]
fn successfully_rolled_back_lookup_failure_keeps_context_available() {
    let lifecycle = ContextResourceLifecycle::new();
    lifecycle
        .run_stateful(|| Ok(()), || Ok(()), || Ok(()))
        .expect("module load");
    let lookup = lifecycle.run_recoverable(
        || Ok(()),
        || Err::<(), _>(failure("missing entrypoint")),
        || Ok(()),
    );
    assert!(matches!(lookup, Err(CudaError::InvalidArgument { .. })));
    lifecycle
        .run_stateful(|| Ok(()), || Ok(()), || Ok(()))
        .expect("module rollback");
    assert!(!lifecycle.is_poisoned());
    assert!(lifecycle.ensure_available().is_ok());
}

#[test]
fn stateful_failure_publishes_quarantine_before_blocking_completion() {
    let lifecycle = Arc::new(ContextResourceLifecycle::new());
    let worker_lifecycle = lifecycle.clone();
    let (recovery_entered_tx, recovery_entered_rx) = std::sync::mpsc::channel();
    let (release_recovery_tx, release_recovery_rx) = std::sync::mpsc::channel();
    let worker = std::thread::spawn(move || {
        worker_lifecycle.run_stateful(
            || Ok(()),
            || Err::<(), _>(failure("resource transition failed")),
            || {
                recovery_entered_tx
                    .send(())
                    .expect("announce blocked completion");
                release_recovery_rx
                    .recv()
                    .expect("release blocked completion");
                Ok(())
            },
        )
    });

    recovery_entered_rx
        .recv()
        .expect("stateful recovery entered");
    assert!(lifecycle.is_poisoned());
    assert!(matches!(
        lifecycle.ensure_available(),
        Err(CudaError::StatePoisoned { .. })
    ));
    release_recovery_tx
        .send(())
        .expect("release stateful recovery");
    assert!(matches!(
        worker.join().expect("join stateful operation"),
        Err(CudaError::InvalidArgument { .. })
    ));
}

#[test]
fn failed_context_binding_poisons_without_running_operation() {
    let lifecycle = ContextResourceLifecycle::new();
    let operation_ran = AtomicBool::new(false);
    let recovery_ran = AtomicBool::new(false);
    let result = lifecycle.run_recoverable(
        || Err(failure("set current")),
        || {
            operation_ran.store(true, Ordering::Relaxed);
            Ok(())
        },
        || {
            recovery_ran.store(true, Ordering::Relaxed);
            Ok(())
        },
    );
    assert!(matches!(result, Err(CudaError::InvalidArgument { .. })));
    assert!(lifecycle.is_poisoned());
    assert!(!operation_ran.load(Ordering::Relaxed));
    assert!(!recovery_ran.load(Ordering::Relaxed));
}

#[test]
fn concurrent_operations_are_serialized() {
    let lifecycle = Arc::new(ContextResourceLifecycle::new());
    let (entered_tx, entered_rx) = std::sync::mpsc::channel();
    let (release_tx, release_rx) = std::sync::mpsc::channel();
    let first_lifecycle = lifecycle.clone();
    let first = std::thread::spawn(move || {
        first_lifecycle.run_recoverable(
            || Ok(()),
            || {
                entered_tx.send(()).expect("announce first operation");
                release_rx.recv().expect("release first operation");
                Ok(())
            },
            || Ok(()),
        )
    });
    entered_rx.recv().expect("first operation entered");

    let second_entered = Arc::new(AtomicBool::new(false));
    let second_lifecycle = lifecycle.clone();
    let second_flag = second_entered.clone();
    let second = std::thread::spawn(move || {
        second_lifecycle.run_recoverable(
            || Ok(()),
            || {
                second_flag.store(true, Ordering::Release);
                Ok(())
            },
            || Ok(()),
        )
    });
    std::thread::yield_now();
    assert!(!second_entered.load(Ordering::Acquire));
    release_tx.send(()).expect("release first operation");
    first
        .join()
        .expect("join first operation")
        .expect("first result");
    second
        .join()
        .expect("join second operation")
        .expect("second result");
    assert!(second_entered.load(Ordering::Acquire));
}

#[test]
fn panic_while_gated_poisons_later_operations() {
    let lifecycle = ContextResourceLifecycle::new();
    let panic_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = lifecycle.run_recoverable(
            || Ok(()),
            || -> Result<(), CudaError> { panic!("simulated driver wrapper panic") },
            || Ok(()),
        );
    }));
    assert!(panic_result.is_err());
    assert!(matches!(
        lifecycle.ensure_available(),
        Err(CudaError::StatePoisoned { .. })
    ));
    assert!(lifecycle.is_poisoned());
    assert!(matches!(
        lifecycle.run_recoverable(|| Ok(()), || Ok(()), || Ok(())),
        Err(CudaError::StatePoisoned { .. })
    ));
}

#[test]
fn panic_during_recovery_poisons_before_releasing_gate() {
    let lifecycle = ContextResourceLifecycle::new();
    let panic_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = lifecycle.run_recoverable(
            || Ok(()),
            || Err::<(), _>(failure("driver failure")),
            || -> Result<(), CudaError> { panic!("simulated synchronization panic") },
        );
    }));

    assert!(panic_result.is_err());
    assert!(lifecycle.is_poisoned());
    assert!(matches!(
        lifecycle.ensure_available(),
        Err(CudaError::StatePoisoned { .. })
    ));
}
