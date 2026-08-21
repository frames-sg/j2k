// SPDX-License-Identifier: MIT OR Apache-2.0

use super::CudaQueuedHtj2kCleanup;

impl Drop for CudaQueuedHtj2kCleanup {
    fn drop(&mut self) {
        if self.pool_reuse_guard.is_some() {
            // Last-resort protection for callers that abandon queued work or
            // unwind. `finish` surfaces synchronization failures normally.
            let outcome = self.context.synchronize_for_resource_release();
            if outcome.completion_established() {
                let _ = self.release_after_stream_completion();
            } else {
                // Do not recycle or free allocations that the driver could
                // still reference. The active pool hold is intentionally
                // retained when completion could not be attempted.
                self.abandon_resources();
            }
        }
    }
}
