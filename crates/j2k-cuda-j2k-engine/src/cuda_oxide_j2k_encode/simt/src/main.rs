#![allow(
    clippy::manual_div_ceil,
    reason = "CUDA device toolchain compatibility requires explicit integer ceiling division"
)]
#![allow(
    clippy::manual_is_multiple_of,
    reason = "CUDA device toolchain compatibility requires explicit remainder checks"
)]
#![allow(
    clippy::too_many_arguments,
    reason = "flat device helpers mirror CUDA ABI buffers and launch metadata"
)]
#![allow(
    static_mut_refs,
    reason = "CUDA shared-memory statics are accessed through device-scoped references"
)]

mod abi;
mod constants;
mod dwt53;
mod dwt97;
mod exports;
mod helpers;
mod packet_writer;
mod packetization;
mod quantization;
mod tag_tree;

include!("../../../cuda_oxide_simt_prelude.rs");

fn main() {}
