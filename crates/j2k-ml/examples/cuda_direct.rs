// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(not(target_os = "macos"))]
mod support;

#[cfg(not(target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use burn_cuda::CudaDevice;
    use j2k::{BatchDecodeOptions, EncodedImage};
    use j2k_ml::CudaDirectBurnDecoder;

    let inputs = (0..4)
        .map(|seed| support::generated_rgb8(seed).map(EncodedImage::full))
        .collect::<Result<Vec<_>, _>>()?;
    let mut decoder =
        CudaDirectBurnDecoder::new(CudaDevice::default(), BatchDecodeOptions::default());
    let output = decoder.decode(inputs)?;
    if !output.errors().is_empty()
        || !output.group_errors().is_empty()
        || output.groups().len() != 1
    {
        return Err(std::io::Error::other(format!("incomplete CUDA decode: {output:?}")).into());
    }
    let tensor = output
        .into_groups()
        .into_iter()
        .next()
        .ok_or_else(|| std::io::Error::other("CUDA decode returned no tensor"))?
        .into_tensor()
        .into_tensor();
    println!(
        "direct CUDA-to-Burn decode: shape={:?}, dtype={:?}",
        tensor.dims(),
        tensor.dtype()
    );
    Ok(())
}

#[cfg(target_os = "macos")]
fn main() {
    eprintln!("the direct CUDA example requires a supported CUDA host");
}
