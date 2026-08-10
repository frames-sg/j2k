// SPDX-License-Identifier: MIT OR Apache-2.0

use objc2::{rc::Retained, runtime::ProtocolObject, Message};
use objc2_foundation::{NSError, NSString};
use objc2_metal::{MTLCompileOptions, MTLComputePipelineState, MTLDevice, MTLLibrary};

use crate::MetalSupportError;

type DeviceHandle = Retained<ProtocolObject<dyn MTLDevice>>;
type LibraryHandle = Retained<ProtocolObject<dyn MTLLibrary>>;
type PipelineHandle = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

fn objective_c_error_message(error: &NSError) -> String {
    error.localizedDescription().to_string()
}

/// Compile a Metal shader source string with default compile options.
///
/// # Errors
///
/// Returns [`MetalSupportError::ShaderLibrary`] when Metal rejects the source.
pub fn shader_library(
    device: &ProtocolObject<dyn MTLDevice>,
    source: &str,
) -> Result<LibraryHandle, MetalSupportError> {
    let options = MTLCompileOptions::new();
    let source = NSString::from_str(source);
    device
        .newLibraryWithSource_options_error(&source, Some(&options))
        .map_err(|error| MetalSupportError::ShaderLibrary {
            message: objective_c_error_message(&error),
        })
}

/// Load a named compute pipeline from an already compiled shader library.
///
/// # Errors
///
/// Returns [`MetalSupportError::PipelineFunction`] when the function is absent,
/// or [`MetalSupportError::PipelineState`] when pipeline construction fails.
pub fn named_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
    function_name: &str,
) -> Result<PipelineHandle, MetalSupportError> {
    let name = NSString::from_str(function_name);
    let function =
        library
            .newFunctionWithName(&name)
            .ok_or_else(|| MetalSupportError::PipelineFunction {
                function_name: function_name.to_string(),
                message: format!("Function '{function_name}' does not exist"),
            })?;
    device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|error| MetalSupportError::PipelineState {
            function_name: function_name.to_string(),
            message: objective_c_error_message(&error),
        })
}

/// Convenience loader for many pipelines from one Metal shader library.
pub struct MetalPipelineLoader {
    device: DeviceHandle,
    library: LibraryHandle,
}

impl MetalPipelineLoader {
    /// Compile `source` and keep the resulting library for named pipeline loads.
    ///
    /// # Errors
    ///
    /// Returns [`MetalSupportError::ShaderLibrary`] when Metal rejects the source.
    pub fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        source: &str,
    ) -> Result<Self, MetalSupportError> {
        Ok(Self {
            device: device.retain(),
            library: shader_library(device, source)?,
        })
    }

    /// Load one compute pipeline from the cached shader library.
    ///
    /// # Errors
    ///
    /// Returns a typed lookup or pipeline-construction error.
    pub fn pipeline(&self, function_name: &str) -> Result<PipelineHandle, MetalSupportError> {
        named_pipeline(&self.device, &self.library, function_name)
    }

    /// Borrow the compiled shader library.
    #[must_use]
    pub fn library(&self) -> &ProtocolObject<dyn MTLLibrary> {
        &self.library
    }
}
