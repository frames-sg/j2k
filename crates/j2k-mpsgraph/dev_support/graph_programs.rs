// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_mpsgraph::{Error, MpsGraphElementType, MpsGraphProgram, MpsGraphTensorSpec};
use objc2_foundation::{NSArray, NSNumber};
use objc2_metal_performance_shaders::MPSDataType;
use objc2_metal_performance_shaders_graph::MPSGraph;

pub(crate) fn average_program(
    batch: usize,
    height: usize,
    width: usize,
) -> Result<MpsGraphProgram, Error> {
    let spec = MpsGraphTensorSpec::new([batch, height, width, 3], MpsGraphElementType::U8)?;
    let sample_count = height
        .checked_mul(width)
        .and_then(|pixels| pixels.checked_mul(3))
        .and_then(|samples| u32::try_from(samples).ok())
        .ok_or(Error::TensorShapeOverflow)?;
    let dimensions = spec.shape().map(NSNumber::new_usize);
    let shape = NSArray::from_retained_slice(&dimensions);
    let axes = NSArray::from_retained_slice(&[
        NSNumber::new_isize(1),
        NSNumber::new_isize(2),
        NSNumber::new_isize(3),
    ]);
    // SAFETY: the graph uses a validated shape, valid reduction axes, and a
    // finite nonzero divisor. The program retains every graph object.
    let (graph, placeholder, average) = unsafe {
        let graph = MPSGraph::new();
        let placeholder =
            graph.placeholderWithShape_dataType_name(Some(&shape), MPSDataType::UInt8, None);
        let float = graph.castTensor_toType_name(&placeholder, MPSDataType::Float32, None);
        let summed = graph.reductionSumWithTensor_axes_name(&float, Some(&axes), None);
        let divisor = graph
            .constantWithScalar_dataType(255.0 * f64::from(sample_count), MPSDataType::Float32);
        let average = graph.divisionWithPrimaryTensor_secondaryTensor_name(&summed, &divisor, None);
        (graph, placeholder, average)
    };
    MpsGraphProgram::new(graph, placeholder, vec![average], spec)
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "the oracle accumulates in F64 and returns a bounded normalized F32 score"
)]
pub(crate) fn average_cpu(
    pixels: &[u8],
    batch: usize,
    height: usize,
    width: usize,
) -> Result<Vec<f32>, Error> {
    let image_samples = height
        .checked_mul(width)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or(Error::TensorShapeOverflow)?;
    let expected = batch
        .checked_mul(image_samples)
        .ok_or(Error::TensorShapeOverflow)?;
    if batch == 0 || image_samples == 0 || pixels.len() != expected {
        return Err(Error::InvalidTensorContract {
            reason: "average CPU oracle input does not match its nonzero shape",
        });
    }
    let mut scores = Vec::new();
    scores
        .try_reserve_exact(batch)
        .map_err(|source| Error::Allocation {
            what: "average CPU oracle scores",
            requested: batch,
            source,
        })?;
    for image in pixels.chunks_exact(image_samples) {
        let sum = image.iter().map(|&sample| f64::from(sample)).sum::<f64>();
        scores.push((sum / (255.0 * image_samples as f64)) as f32);
    }
    Ok(scores)
}
