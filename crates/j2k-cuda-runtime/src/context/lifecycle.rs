// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Mutex,
};

use crate::error::{select_uncertain_completion_error, CudaError};

pub(crate) struct ContextResourceLifecycle {
    gate: Mutex<()>,
    poisoned: AtomicBool,
}

struct PoisonOnUnwind<'a> {
    lifecycle: &'a ContextResourceLifecycle,
    armed: bool,
}

impl PoisonOnUnwind<'_> {
    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for PoisonOnUnwind<'_> {
    fn drop(&mut self) {
        if self.armed {
            self.lifecycle.mark_poisoned();
        }
    }
}

impl ContextResourceLifecycle {
    pub(crate) fn new() -> Self {
        Self {
            gate: Mutex::new(()),
            poisoned: AtomicBool::new(false),
        }
    }

    pub(crate) fn ensure_available(&self) -> Result<(), CudaError> {
        if self.is_poisoned() {
            Err(CudaError::StatePoisoned {
                message: "CUDA context resource completion is uncertain".to_string(),
            })
        } else {
            Ok(())
        }
    }

    pub(crate) fn is_poisoned(&self) -> bool {
        self.poisoned.load(Ordering::Acquire)
    }

    pub(crate) fn can_release_individually(&mut self) -> bool {
        if self.gate.get_mut().is_err() {
            self.mark_poisoned();
        }
        !self.is_poisoned()
    }

    pub(crate) fn run_recoverable<T>(
        &self,
        set_current: impl FnOnce() -> Result<(), CudaError>,
        operation: impl FnOnce() -> Result<T, CudaError>,
        establish_completion: impl FnOnce() -> Result<(), CudaError>,
    ) -> Result<T, CudaError> {
        self.run_gated(set_current, |lifecycle| match operation() {
            Ok(value) => Ok(value),
            Err(primary_error) => match establish_completion() {
                Ok(()) => Err(primary_error),
                Err(completion_error) => {
                    lifecycle.mark_poisoned();
                    Err(select_uncertain_completion_error(
                        primary_error,
                        Some(completion_error),
                    ))
                }
            },
        })
    }

    pub(crate) fn run_completion<T>(
        &self,
        set_current: impl FnOnce() -> Result<(), CudaError>,
        operation: impl FnOnce() -> Result<T, CudaError>,
    ) -> Result<T, CudaError> {
        self.run_gated(set_current, |lifecycle| {
            let result = operation();
            if result.is_err() {
                lifecycle.mark_poisoned();
            }
            result
        })
    }

    pub(crate) fn run_stateful<T>(
        &self,
        set_current: impl FnOnce() -> Result<(), CudaError>,
        operation: impl FnOnce() -> Result<T, CudaError>,
        establish_completion: impl FnOnce() -> Result<(), CudaError>,
    ) -> Result<T, CudaError> {
        self.run_gated(set_current, |lifecycle| match operation() {
            Ok(value) => Ok(value),
            Err(primary_error) => {
                // Stateful failures quarantine immediately. Pool operations
                // consult this atomic state without taking the lifecycle gate,
                // so publishing after a blocking sync would leave a reuse race.
                lifecycle.mark_poisoned();
                let completion_result = establish_completion();
                match completion_result {
                    Ok(()) => Err(primary_error),
                    Err(completion_error) => Err(select_uncertain_completion_error(
                        primary_error,
                        Some(completion_error),
                    )),
                }
            }
        })
    }

    fn run_gated<T>(
        &self,
        set_current: impl FnOnce() -> Result<(), CudaError>,
        operation: impl FnOnce(&Self) -> Result<T, CudaError>,
    ) -> Result<T, CudaError> {
        let _gate = match self.gate.lock() {
            Ok(gate) => gate,
            Err(error) => {
                self.mark_poisoned();
                return Err(CudaError::StatePoisoned {
                    message: error.to_string(),
                });
            }
        };
        self.ensure_available()?;
        let mut poison_on_unwind = PoisonOnUnwind {
            lifecycle: self,
            armed: true,
        };
        if let Err(error) = set_current() {
            self.mark_poisoned();
            poison_on_unwind.disarm();
            return Err(error);
        }
        let result = operation(self);
        poison_on_unwind.disarm();
        result
    }

    fn mark_poisoned(&self) {
        self.poisoned.store(true, Ordering::Release);
    }
}

#[cfg(test)]
mod tests;
