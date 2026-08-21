// SPDX-License-Identifier: MIT OR Apache-2.0

//! Internal kernel-parameter macros.

macro_rules! cuda_kernel_params {
    ($($arg:ident),+ $(,)?) => {
        [$(cuda_kernel_param(&mut $arg)),+]
    };
}
