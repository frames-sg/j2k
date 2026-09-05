// SPDX-License-Identifier: MIT OR Apache-2.0

use core::{marker::PhantomData, ptr::NonNull};
use std::{
    rc::Rc,
    sync::{Arc, OnceLock},
};

use block2::RcBlock;
use objc2::{rc::Retained, runtime::ProtocolObject, Message};
use objc2_foundation::{NSArray, NSDictionary, NSError};
use objc2_metal::MTLCommandQueue;
use objc2_metal_performance_shaders_graph::{
    MPSGraph, MPSGraphExecutionDescriptor, MPSGraphTensor, MPSGraphTensorData,
    MPSGraphTensorDataDictionary,
};

type CompletionBlock = RcBlock<dyn Fn(NonNull<MPSGraphTensorDataDictionary>, *mut NSError)>;
type CompletionState = OnceLock<Result<(), GraphExecutionError>>;

/// An asynchronous Foundation error copied while the callback owns its source.
#[derive(Clone, Debug)]
pub struct GraphExecutionError {
    /// Foundation error domain.
    pub domain: String,
    /// Foundation error code.
    pub code: isize,
    /// Owned localized error description.
    pub description: String,
}

impl core::fmt::Display for GraphExecutionError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            formatter,
            "MPSGraph execution failed ({}, code {}): {}",
            self.domain, self.code, self.description
        )
    }
}

impl std::error::Error for GraphExecutionError {}

/// An asynchronous graph run and the owner of its unretained input storage.
///
/// The input owner is concrete so a resident batch can keep pool leases alive,
/// while a direct submission can retain its standalone buffer. This guard is
/// neither `Send` nor `Sync`. Drop waits for completion, including failure, before
/// releasing graph resources and finally the input owner.
pub struct MpsGraphSubmission<InputOwner> {
    _graph: Retained<MPSGraph>,
    _image_placeholder: Retained<MPSGraphTensor>,
    targets: Retained<NSArray<MPSGraphTensor>>,
    _feeds: Retained<MPSGraphTensorDataDictionary>,
    results: Retained<MPSGraphTensorDataDictionary>,
    _execution_descriptor: Retained<MPSGraphExecutionDescriptor>,
    _completion_block: CompletionBlock,
    completion_state: Arc<CompletionState>,
    _not_send_or_sync: PhantomData<Rc<()>>,
    // Tensor data may not retain its buffer. Drop this after the graph resources.
    _input_owner: InputOwner,
}

impl<InputOwner> core::fmt::Debug for MpsGraphSubmission<InputOwner> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("MpsGraphSubmission")
            .field("complete", &self.is_complete())
            .field("target_count", &self.target_count())
            .finish_non_exhaustive()
    }
}

impl<InputOwner> MpsGraphSubmission<InputOwner> {
    /// Submit a validated single-placeholder graph on the supplied command queue.
    ///
    /// # Safety
    /// The placeholder and every target must belong to `graph`. Tensor data must
    /// match the placeholder's shape, dtype and device. `input_owner` must keep
    /// all unretained tensor storage alive and prevent its reuse until this guard
    /// drops. Writes to that storage must precede graph reads on this queue, and
    /// callers must not mutate the graph, tensors or input during execution.
    pub unsafe fn submit(
        graph: &MPSGraph,
        image_placeholder: &MPSGraphTensor,
        targets: &[Retained<MPSGraphTensor>],
        command_queue: &ProtocolObject<dyn MTLCommandQueue>,
        tensor_data: &MPSGraphTensorData,
        input_owner: InputOwner,
    ) -> Self {
        let feeds = NSDictionary::from_slices(&[image_placeholder], &[tensor_data]);
        let targets = NSArray::from_retained_slice(targets);
        // SAFETY: new is the standard owning Objective-C constructor.
        let execution_descriptor = unsafe { MPSGraphExecutionDescriptor::new() };
        let completion_state = Arc::new(CompletionState::default());
        let callback_state = Arc::clone(&completion_state);
        let completion_block: CompletionBlock = RcBlock::new(
            move |_results: NonNull<MPSGraphTensorDataDictionary>, error: *mut NSError| {
                let error = NonNull::new(error).map(|error| {
                    // SAFETY: MPSGraph guarantees the NSError is valid for this callback.
                    let error = unsafe { error.as_ref() };
                    GraphExecutionError {
                        domain: error.domain().to_string(),
                        code: error.code(),
                        description: error.localizedDescription().to_string(),
                    }
                });
                let _ = callback_state.set(error.map_or(Ok(()), Err));
            },
        );
        // SAFETY: the block matches the generated signature. Both descriptor and
        // returned guard retain it until Drop has waited for callback completion.
        unsafe {
            execution_descriptor.setCompletionHandler(RcBlock::as_ptr(&completion_block));
        }
        // SAFETY: the caller validates graph/feed/queue compatibility and input
        // write ordering. The guard retains every resource and storage owner until
        // the completion callback fires, including when the guard is dropped early.
        let results = unsafe {
            graph.runAsyncWithMTLCommandQueue_feeds_targetTensors_targetOperations_executionDescriptor(
                command_queue, &feeds, &targets, None, Some(&execution_descriptor),
            )
        };
        Self {
            _graph: graph.retain(),
            _image_placeholder: image_placeholder.retain(),
            targets,
            _feeds: feeds,
            results,
            _execution_descriptor: execution_descriptor,
            _completion_block: completion_block,
            completion_state,
            _not_send_or_sync: PhantomData,
            _input_owner: input_owner,
        }
    }

    /// Whether the completion callback has fired, for either success or failure.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.completion_state.get().is_some()
    }

    /// Number of requested targets, preserving the caller's order.
    #[must_use]
    pub fn target_count(&self) -> usize {
        self.targets.len()
    }

    /// Wait for graph completion. Repeated calls return the same outcome.
    ///
    /// # Errors
    /// Returns the owned asynchronous Foundation error if execution failed.
    pub fn wait(&self) -> Result<(), GraphExecutionError> {
        self.completion_state.wait().clone()
    }

    /// Wait and retain one output in target order; absent or out-of-range targets
    /// return `None`. Adapters decide how to report missing outputs and allocate
    /// their result collection.
    ///
    /// # Errors
    /// Returns the owned asynchronous Foundation error if execution failed.
    pub fn output(
        &self,
        index: usize,
    ) -> Result<Option<Retained<MPSGraphTensorData>>, GraphExecutionError> {
        self.wait()?;
        if index >= self.targets.len() {
            return Ok(None);
        }
        Ok(self
            .results
            .objectForKey(&self.targets.objectAtIndex(index)))
    }
}

impl<InputOwner> Drop for MpsGraphSubmission<InputOwner> {
    fn drop(&mut self) {
        // No metadata extraction or output allocation on the cleanup path.
        let _ = self.completion_state.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::{CompletionState, GraphExecutionError};

    #[test]
    fn completion_state_preserves_graph_errors() {
        let state = CompletionState::new();
        assert!(state.get().is_none());

        state
            .set(Err(GraphExecutionError {
                domain: "test.domain".to_string(),
                code: 17,
                description: "test failure".to_string(),
            }))
            .expect("first completion");

        assert!(state.get().is_some());
        let error = state.wait().clone().expect_err("owned graph error");
        assert_eq!(error.domain, "test.domain");
        assert_eq!(error.code, 17);
        assert_eq!(error.description, "test failure");
    }
}
