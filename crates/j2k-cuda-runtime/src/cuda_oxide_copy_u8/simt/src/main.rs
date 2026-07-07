use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

include!("../../../cuda_oxide_simt_prelude.rs");

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub unsafe fn j2k_copy_u8(dst: *mut u8, src: *const u8, len: u64) {
        let index = thread::index_1d().get();
        if index < len as usize {
            simt_store(dst, index, simt_load(src, index));
        }
    }
}

fn main() {}
