// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
mod support;

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use j2k::{BatchDecodeOptions, EncodedImage};
    use j2k_ml::MetalUploadBurnDecoder;

    let inputs = (0..4)
        .map(|seed| support::generated_rgb8(seed).map(EncodedImage::full))
        .collect::<Result<Vec<_>, _>>()?;
    let mut decoder = MetalUploadBurnDecoder::system_default(BatchDecodeOptions::default())?;
    let output = decoder.decode(inputs)?;
    if !output.errors.is_empty() || !output.group_errors.is_empty() || output.groups.len() != 1 {
        return Err(std::io::Error::other(format!("incomplete Metal decode: {output:?}")).into());
    }
    let tensor = output
        .groups
        .into_iter()
        .next()
        .ok_or_else(|| std::io::Error::other("Metal decode returned no tensor"))?
        .tensor
        .into_tensor();
    println!(
        "staged Metal-to-Burn upload: shape={:?}, dtype={:?}",
        tensor.dims(),
        tensor.dtype()
    );
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("the Metal upload example requires macOS");
}
