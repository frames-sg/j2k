// SPDX-License-Identifier: MIT OR Apache-2.0

macro_rules! cuda_kernel_params {
    ($($arg:ident),+ $(,)?) => {
        [$(cuda_kernel_param(&mut $arg)),+]
    };
}
