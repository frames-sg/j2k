#![allow(
    clippy::manual_div_ceil,
    clippy::manual_is_multiple_of,
    clippy::too_many_arguments,
    static_mut_refs
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
