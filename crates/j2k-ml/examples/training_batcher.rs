// SPDX-License-Identifier: MIT OR Apache-2.0

mod support;

use std::sync::{Arc, Mutex};

use burn_core::data::{
    dataloader::{batcher::Batcher, DataLoaderBuilder},
    dataset::InMemDataset,
};
use burn_core::tensor::{backend::Backend, Int, Tensor, TensorData};
#[cfg(not(all(target_arch = "aarch64", target_os = "linux")))]
use burn_flex::{Flex as ExampleBackend, FlexDevice as ExampleDevice};
#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
use burn_ndarray::{NdArray as ExampleBackend, NdArrayDevice};
use j2k::{BatchDecodeOptions, EncodedImage};
use j2k_ml::{BurnBatchTensor, CpuBurnDecoder};

#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
type PortableBackend = ExampleBackend<f32, i64, i8>;
#[cfg(not(all(target_arch = "aarch64", target_os = "linux")))]
type PortableBackend = ExampleBackend;

#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
fn example_device() -> NdArrayDevice {
    NdArrayDevice::Cpu
}

#[cfg(not(all(target_arch = "aarch64", target_os = "linux")))]
fn example_device() -> ExampleDevice {
    ExampleDevice
}

#[derive(Clone, Debug)]
struct TrainingItem {
    encoded: Arc<[u8]>,
    label: i32,
}

#[derive(Clone, Debug)]
struct TrainingBatch<B: Backend> {
    images: Tensor<B, 4>,
    labels: Tensor<B, 1, Int>,
}

#[derive(Clone, Debug, thiserror::Error)]
enum TrainingBatchError {
    #[error("the persistent decoder mutex was poisoned")]
    DecoderPoisoned,
    #[error("the persistent decoder could not be initialized")]
    DecoderUnavailable,
    #[error("batch decode failed: {0}")]
    Decode(String),
    #[error("batch contained item failures: {0:?}")]
    ItemFailures(Vec<String>),
    #[error("expected one homogeneous image group, received {0}")]
    GroupCount(usize),
    #[error("decoded source index {0} has no matching label")]
    MissingLabel(usize),
    #[error("the generated RGB8 example unexpectedly produced a non-U8 tensor")]
    UnexpectedSampleType,
}

struct J2kTrainingBatcher<B: Backend> {
    decoder: Mutex<Option<CpuBurnDecoder<B>>>,
}

impl<B: Backend> J2kTrainingBatcher<B> {
    fn new() -> Self {
        Self {
            decoder: Mutex::new(None),
        }
    }
}

impl<B: Backend> Batcher<B, TrainingItem, Result<TrainingBatch<B>, TrainingBatchError>>
    for J2kTrainingBatcher<B>
{
    fn batch(
        &self,
        items: Vec<TrainingItem>,
        device: &B::Device,
    ) -> Result<TrainingBatch<B>, TrainingBatchError> {
        let labels = items.iter().map(|item| item.label).collect::<Vec<_>>();
        let inputs = items
            .into_iter()
            .map(|item| EncodedImage::full(item.encoded))
            .collect();
        let mut decoder = self
            .decoder
            .lock()
            .map_err(|_| TrainingBatchError::DecoderPoisoned)?;
        if decoder
            .as_ref()
            .is_none_or(|decoder| decoder.device() != device)
        {
            *decoder = Some(CpuBurnDecoder::new(
                device.clone(),
                BatchDecodeOptions::default(),
            ));
        }
        let output = decoder
            .as_mut()
            .ok_or(TrainingBatchError::DecoderUnavailable)?
            .decode(inputs)
            .map_err(|error| TrainingBatchError::Decode(error.to_string()))?;

        let mut failures = output
            .errors
            .iter()
            .map(|error| format!("{error:?}"))
            .collect::<Vec<_>>();
        failures.extend(
            output
                .group_errors
                .iter()
                .map(std::string::ToString::to_string),
        );
        if !failures.is_empty() {
            return Err(TrainingBatchError::ItemFailures(failures));
        }
        if output.groups.len() != 1 {
            return Err(TrainingBatchError::GroupCount(output.groups.len()));
        }
        let group = output
            .groups
            .into_iter()
            .next()
            .ok_or(TrainingBatchError::GroupCount(0))?;
        let ordered_labels = group
            .source_indices
            .iter()
            .map(|index| {
                labels
                    .get(*index)
                    .copied()
                    .ok_or(TrainingBatchError::MissingLabel(*index))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let images = match group.tensor {
            BurnBatchTensor::U8(tensor) => tensor.float().div_scalar(255.0),
            BurnBatchTensor::U16(_) | BurnBatchTensor::I16(_) => {
                return Err(TrainingBatchError::UnexpectedSampleType);
            }
        };
        let label_count = ordered_labels.len();
        let labels =
            Tensor::<B, 1, Int>::from_data(TensorData::new(ordered_labels, [label_count]), device);
        Ok(TrainingBatch { images, labels })
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let items = (0_u8..8)
        .map(|index| {
            Ok(TrainingItem {
                encoded: support::generated_rgb8(index)?,
                label: i32::from(index % 2),
            })
        })
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
    let loader = DataLoaderBuilder::<PortableBackend, _, _>::new(J2kTrainingBatcher::<
        PortableBackend,
    >::new())
    .batch_size(4)
    .num_workers(0)
    .set_device(example_device())
    .build(InMemDataset::new(items));

    for batch in loader.iter() {
        let batch = batch?;
        println!(
            "training batch: images={:?}, labels={:?}",
            batch.images.dims(),
            batch.labels.dims()
        );
    }
    Ok(())
}
