// SPDX-License-Identifier: MIT OR Apache-2.0

macro_rules! cuda_kernel_params {
    ($($arg:ident),+ $(,)?) => {
        [$(cuda_kernel_param(&mut $arg)),+]
    };
}

macro_rules! impl_cuda_htj2k_encoded_status_accessors {
    () => {
        /// HTJ2K cleanup segment length in bytes.
        pub fn cleanup_length(&self) -> u32 {
            htj2k_encoded_cleanup_length(self.status)
        }

        /// HTJ2K refinement segment length in bytes.
        pub fn refinement_length(&self) -> u32 {
            htj2k_encoded_refinement_length(self.status)
        }

        /// Number of coding passes in the encoded payload.
        pub fn num_coding_passes(&self) -> u8 {
            htj2k_encoded_num_coding_passes(self.status)
        }

        /// Number of missing most-significant bitplanes.
        pub fn num_zero_bitplanes(&self) -> u8 {
            htj2k_encoded_num_zero_bitplanes(self.status)
        }

        /// Kernel status row downloaded after dispatch.
        pub fn status(&self) -> CudaHtj2kEncodeStatus {
            self.status
        }

        /// CUDA execution counters for the encode dispatch.
        pub fn execution(&self) -> CudaExecutionStats {
            self.execution
        }

        /// CUDA event timings for the encode dispatch.
        pub fn stage_timings(&self) -> CudaHtj2kEncodeStageTimings {
            self.stage_timings
        }
    };
}
