// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{fs, path::Path};

fn compact(source: &str) -> String {
    source
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect()
}

#[test]
fn cuda_j2k_encode_kernel_parameter_order_remains_stable() {
    let launch =
        fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("src/j2k_encode/launch.rs"))
            .expect("read CUDA J2K encode launch module");
    let launch = compact(&launch);

    for parameters in [
        "cuda_kernel_params!(pixels_ptr,output_ptr,num_pixels_u64,num_components_u32,bit_depth_u32,signed_u32)",
        "cuda_kernel_params!(pixels_ptr,output_ptr,width_u64,height_u64,byte_offset_u64,pitch_bytes_u64,num_components_u32,bit_depth_u32,signed_u32)",
        "cuda_kernel_params!(input_ptr,output_ptr,full_width,current_width,current_height,low_extent)",
        "cuda_kernel_params!(samples_ptr,coefficients_ptr,len_u64,step_exponent,step_mantissa,range_bits,reversible)",
        "cuda_kernel_params!(samples_ptr,coefficients_ptr,x0,y0,width,height,stride,step_exponent,step_mantissa,range_bits,reversible)",
    ] {
        assert_eq!(
            launch.matches(parameters).count(),
            1,
            "kernel parameter ABI must contain exactly one {parameters}"
        );
    }
    assert_eq!(
        launch
            .matches("cuda_kernel_params!(plane0_ptr,plane1_ptr,plane2_ptr,len_u64)")
            .count(),
        2,
        "RCT and ICT must retain their identical four-argument ABI"
    );
}
