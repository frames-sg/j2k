// SPDX-License-Identifier: MIT OR Apache-2.0

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct CudaJ2kStoreGray8Job {
    pub(crate) input_width: u32,
    pub(crate) source_x: u32,
    pub(crate) source_y: u32,
    pub(crate) copy_width: u32,
    pub(crate) copy_height: u32,
    pub(crate) output_width: u32,
    pub(crate) output_height: u32,
    pub(crate) output_x: u32,
    pub(crate) output_y: u32,
    pub(crate) addend: f32,
    pub(crate) bit_depth: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct CudaJ2kStoreGray16Job {
    pub(crate) input_width: u32,
    pub(crate) source_x: u32,
    pub(crate) source_y: u32,
    pub(crate) copy_width: u32,
    pub(crate) copy_height: u32,
    pub(crate) output_width: u32,
    pub(crate) output_height: u32,
    pub(crate) output_x: u32,
    pub(crate) output_y: u32,
    pub(crate) addend: f32,
    pub(crate) bit_depth: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct CudaJ2kStoreGray8BatchJob {
    pub(crate) input_ptr: u64,
    pub(crate) output_ptr: u64,
    pub(crate) job: CudaJ2kStoreGray8Job,
    pub(crate) reserved_tail: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct CudaJ2kStoreGray16BatchJob {
    pub(crate) input_ptr: u64,
    pub(crate) output_ptr: u64,
    pub(crate) job: CudaJ2kStoreGray16Job,
    pub(crate) reserved_tail: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct CudaJ2kStoreGrayI16BatchJob {
    pub(crate) input_ptr: u64,
    pub(crate) output_ptr: u64,
    pub(crate) job: CudaJ2kStoreGray16Job,
    pub(crate) reserved_tail: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct CudaJ2kInverseMctJob {
    pub(crate) len: u32,
    pub(crate) irreversible97: u32,
    pub(crate) addend0: f32,
    pub(crate) addend1: f32,
    pub(crate) addend2: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct CudaJ2kStoreRgb8Job {
    pub(crate) input_width0: u32,
    pub(crate) input_width1: u32,
    pub(crate) input_width2: u32,
    pub(crate) source_x0: u32,
    pub(crate) source_y0: u32,
    pub(crate) source_x1: u32,
    pub(crate) source_y1: u32,
    pub(crate) source_x2: u32,
    pub(crate) source_y2: u32,
    pub(crate) copy_width: u32,
    pub(crate) copy_height: u32,
    pub(crate) output_width: u32,
    pub(crate) output_height: u32,
    pub(crate) output_x: u32,
    pub(crate) output_y: u32,
    pub(crate) addend0: f32,
    pub(crate) addend1: f32,
    pub(crate) addend2: f32,
    pub(crate) bit_depth0: u32,
    pub(crate) bit_depth1: u32,
    pub(crate) bit_depth2: u32,
    pub(crate) rgba: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct CudaJ2kStoreRgb16Job {
    pub(crate) input_width0: u32,
    pub(crate) input_width1: u32,
    pub(crate) input_width2: u32,
    pub(crate) source_x0: u32,
    pub(crate) source_y0: u32,
    pub(crate) source_x1: u32,
    pub(crate) source_y1: u32,
    pub(crate) source_x2: u32,
    pub(crate) source_y2: u32,
    pub(crate) copy_width: u32,
    pub(crate) copy_height: u32,
    pub(crate) output_width: u32,
    pub(crate) output_height: u32,
    pub(crate) output_x: u32,
    pub(crate) output_y: u32,
    pub(crate) addend0: f32,
    pub(crate) addend1: f32,
    pub(crate) addend2: f32,
    pub(crate) bit_depth0: u32,
    pub(crate) bit_depth1: u32,
    pub(crate) bit_depth2: u32,
    pub(crate) rgba: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct CudaJ2kStoreRgb8MctJob {
    pub(crate) store: CudaJ2kStoreRgb8Job,
    pub(crate) irreversible97: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct CudaJ2kStoreRgb8MctBatchJob {
    pub(crate) plane0_ptr: u64,
    pub(crate) plane1_ptr: u64,
    pub(crate) plane2_ptr: u64,
    pub(crate) output_ptr: u64,
    pub(crate) job: CudaJ2kStoreRgb8MctJob,
    pub(crate) reserved_tail: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct CudaJ2kStoreRgb16MctJob {
    pub(crate) store: CudaJ2kStoreRgb16Job,
    pub(crate) irreversible97: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct CudaJ2kStoreRgbNativeJob {
    pub(crate) input_width0: u32,
    pub(crate) input_width1: u32,
    pub(crate) input_width2: u32,
    pub(crate) source_x0: u32,
    pub(crate) source_y0: u32,
    pub(crate) source_x1: u32,
    pub(crate) source_y1: u32,
    pub(crate) source_x2: u32,
    pub(crate) source_y2: u32,
    pub(crate) copy_width: u32,
    pub(crate) copy_height: u32,
    pub(crate) output_width: u32,
    pub(crate) output_height: u32,
    pub(crate) output_x: u32,
    pub(crate) output_y: u32,
    pub(crate) addend0: f32,
    pub(crate) addend1: f32,
    pub(crate) addend2: f32,
    pub(crate) bit_depth0: u32,
    pub(crate) bit_depth1: u32,
    pub(crate) bit_depth2: u32,
    pub(crate) layout: u32,
    pub(crate) transform: u32,
    pub(crate) reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct CudaJ2kStoreRgbNativeBatchJob {
    pub(crate) plane0_ptr: u64,
    pub(crate) plane1_ptr: u64,
    pub(crate) plane2_ptr: u64,
    pub(crate) output_ptr: u64,
    pub(crate) job: CudaJ2kStoreRgbNativeJob,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct CudaJ2kStoreRgbaNativeJob {
    pub(crate) input_width0: u32,
    pub(crate) input_width1: u32,
    pub(crate) input_width2: u32,
    pub(crate) input_width3: u32,
    pub(crate) source_x0: u32,
    pub(crate) source_y0: u32,
    pub(crate) source_x1: u32,
    pub(crate) source_y1: u32,
    pub(crate) source_x2: u32,
    pub(crate) source_y2: u32,
    pub(crate) source_x3: u32,
    pub(crate) source_y3: u32,
    pub(crate) copy_width: u32,
    pub(crate) copy_height: u32,
    pub(crate) output_width: u32,
    pub(crate) output_height: u32,
    pub(crate) output_x: u32,
    pub(crate) output_y: u32,
    pub(crate) addend0: f32,
    pub(crate) addend1: f32,
    pub(crate) addend2: f32,
    pub(crate) addend3: f32,
    pub(crate) bit_depth0: u32,
    pub(crate) bit_depth1: u32,
    pub(crate) bit_depth2: u32,
    pub(crate) bit_depth3: u32,
    pub(crate) layout: u32,
    pub(crate) transform: u32,
    pub(crate) reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct CudaJ2kStoreRgbaNativeBatchJob {
    pub(crate) plane0_ptr: u64,
    pub(crate) plane1_ptr: u64,
    pub(crate) plane2_ptr: u64,
    pub(crate) plane3_ptr: u64,
    pub(crate) output_ptr: u64,
    pub(crate) job: CudaJ2kStoreRgbaNativeJob,
    pub(crate) reserved_tail: u32,
}
