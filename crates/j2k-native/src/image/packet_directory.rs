// SPDX-License-Identifier: MIT OR Apache-2.0

//! Bounded packet-boundary reuse behind the existing Image decode calls.

use alloc::sync::Arc;
use core::mem::size_of;
use std::sync::Mutex;

use crate::j2c::PacketDirectory;

const MAX_DIRECTORY_BYTES: usize = 16 << 20;

#[derive(Default)]
enum State {
    #[default]
    FirstRegion,
    RepeatedRegion,
    Ready(Arc<PacketDirectory>),
    Disabled,
}

#[derive(Default)]
pub(crate) struct PacketDirectoryCache {
    state: Mutex<State>,
    #[cfg(test)]
    limit: Option<usize>,
}

impl PacketDirectoryCache {
    pub(crate) fn for_region(
        &self,
        build: impl FnOnce(usize) -> Option<PacketDirectory>,
    ) -> Option<Arc<PacketDirectory>> {
        // Only preparation is serialized. Requests borrow immutable offsets
        // after releasing this guard and decode concurrently as before.
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match &*state {
            State::FirstRegion => {
                *state = State::RepeatedRegion;
                None
            }
            State::RepeatedRegion => {
                #[cfg(test)]
                let limit = self.limit.unwrap_or(MAX_DIRECTORY_BYTES);
                #[cfg(not(test))]
                let limit = MAX_DIRECTORY_BYTES;
                let overhead = size_of::<PacketDirectory>() + 2 * size_of::<usize>();
                if let Some(directory) = build(limit.saturating_sub(overhead)) {
                    let directory = Arc::new(directory);
                    *state = State::Ready(Arc::clone(&directory));
                    Some(directory)
                } else {
                    *state = State::Disabled;
                    None
                }
            }
            State::Ready(directory) => Some(Arc::clone(directory)),
            State::Disabled => None,
        }
    }

    pub(crate) fn snapshot(&self) -> Option<Arc<PacketDirectory>> {
        let state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match &*state {
            State::Ready(directory) => Some(Arc::clone(directory)),
            _ => None,
        }
    }

    pub(crate) fn retained_bytes(&self) -> usize {
        self.snapshot().as_deref().map_or(0, Self::directory_bytes)
    }

    pub(crate) fn directory_bytes(directory: &PacketDirectory) -> usize {
        // The boundary allocation plus Arc's value and reference counters.
        directory.retained_bytes() + size_of::<PacketDirectory>() + 2 * size_of::<usize>()
    }

    pub(crate) fn evict_if_unshared(&self, directory: &Arc<PacketDirectory>) -> bool {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // Only this request and the Image own it. Drop both before
        // retrying, so fallback really recovers the original headroom.
        let evict = matches!(&*state, State::Ready(retained)
            if Arc::ptr_eq(retained, directory) && Arc::strong_count(directory) == 2);
        if evict {
            *state = State::Disabled;
        }
        drop(state);
        evict
    }

    #[cfg(test)]
    pub(crate) fn disabled() -> Self {
        Self {
            state: Mutex::new(State::Disabled),
            limit: None,
        }
    }

    #[cfg(test)]
    pub(crate) fn with_limit(limit: usize) -> Self {
        Self {
            state: Mutex::new(State::FirstRegion),
            limit: Some(limit),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        ColorSpace, CpuDecodeParallelism, DecodeSettings, DecoderContext, Image,
        DEFAULT_MAX_DECODE_BYTES,
    };

    #[test]
    fn packet_directory_output_allocation_retries_successfully() {
        let bytes =
            include_bytes!("../../fixtures/packet_directory/openjpeg_rgb_origin5x3_rpcl.j2k");
        let mut image = Image::new(bytes, &DecodeSettings::default()).unwrap();
        // A retained profile is cloned only during output packing. Its size
        // makes that phase, rather than coefficient decoding, the peak owner.
        image.color_space = ColorSpace::Icc {
            profile: vec![0; 1 << 20],
            num_channels: 3,
        };
        let metadata = image.retained_allocation_bytes().unwrap();
        let mut ctx = DecoderContext::default();
        ctx.set_cpu_decode_parallelism(CpuDecodeParallelism::Serial);
        for _ in 0..2 {
            image
                .decode_region_components_with_context((60, 40, 16, 16), &mut ctx)
                .unwrap();
        }
        let expected = image
            .decode_native_components_with_context(&mut ctx)
            .unwrap();
        let channels = &ctx.tile_decode_context.channel_data;
        let channel_bytes = crate::image::allocation::DecodeOwnerBudget::for_components(
            0,
            channels,
            channels.capacity(),
        )
        .unwrap()
        .bytes();
        let baseline =
            DEFAULT_MAX_DECODE_BYTES - channel_bytes - expected.allocated_bytes().unwrap();
        // Isolate the late failure: the core already fits with the cache live.
        image
            .run_decode(&mut ctx, None, None, true, baseline)
            .expect("core fits with cache");
        assert!(image.retained_allocation_bytes().unwrap() > metadata);
        let actual = image
            .decode_native_components_with_context_and_retained_baseline(&mut ctx, baseline)
            .expect("owned output must evict and retry");
        assert_eq!(image.retained_allocation_bytes().unwrap(), metadata);
        for (actual, expected) in actual.planes().iter().zip(expected.planes()) {
            assert_eq!(actual.data(), expected.data());
        }
    }

    #[test]
    fn packet_directory_capacity_retry_succeeds_for_existing_region_apis() {
        let bytes =
            include_bytes!("../../fixtures/packet_directory/openjpeg_rgb_origin5x3_rpcl.j2k");
        let roi = (60, 40, 16, 16);
        for prepared in [false, true] {
            let decode = |image: &Image<'_>, baseline| -> crate::Result<Vec<u32>> {
                let mut ctx = DecoderContext::default();
                ctx.set_cpu_decode_parallelism(CpuDecodeParallelism::Serial);
                if prepared {
                    let mut session = image.prepare_region_decoder_with_context(&mut ctx)?;
                    session.retained_image_bytes = baseline;
                    session.retained_session_bytes =
                        baseline + session.tiles.metadata_owner_bytes();
                    let output = session.decode_region_components(roi)?;
                    Ok(output
                        .planes()
                        .iter()
                        .flat_map(|p| p.samples().iter().map(|v| v.to_bits()))
                        .collect())
                } else {
                    image.run_decode(&mut ctx, Some(roi), None, false, baseline)?;
                    Ok(ctx
                        .tile_decode_context
                        .channel_data
                        .iter()
                        .flat_map(|p| p.container.truncated().iter().map(|v| v.to_bits()))
                        .collect())
                }
            };
            let mut control = Image::new(bytes, &DecodeSettings::default()).unwrap();
            control.packet_cache = super::PacketDirectoryCache::disabled();
            // Locate the real capacity boundary without allocating a cap-sized
            // fixture. The baseline represents an already-live external owner.
            let (mut fits, mut fails) = (0, DEFAULT_MAX_DECODE_BYTES);
            while fails - fits > 1 {
                let candidate = fits + (fails - fits) / 2;
                match decode(&control, candidate) {
                    Ok(_) => fits = candidate,
                    Err(error) => {
                        assert!(crate::j2c::is_capacity_error(&error), "{error:?}");
                        fails = candidate;
                    }
                }
            }
            let expected = decode(&control, fits).unwrap();
            let image = Image::new(bytes, &DecodeSettings::default()).unwrap();
            let metadata = image.retained_allocation_bytes().unwrap();
            for _ in 0..2 {
                decode(&image, metadata).unwrap();
            }
            assert!(image.retained_allocation_bytes().unwrap() > metadata);
            assert_eq!(
                decode(&image, fits).expect("eviction must restore a fitting request"),
                expected
            );
            assert_eq!(image.retained_allocation_bytes().unwrap(), metadata);
        }
    }

    #[test]
    fn packet_directory_capacity_retry_releases_only_unshared_owners() {
        let bytes =
            include_bytes!("../../fixtures/packet_directory/openjpeg_rgb_origin5x3_rpcl.j2k");
        let image = Image::new(bytes, &DecodeSettings::default()).unwrap();
        let metadata = image.retained_allocation_bytes().unwrap();
        let roi = (60, 40, 16, 16);
        let mut ctx = DecoderContext::default();
        for _ in 0..2 {
            image
                .decode_region_components_with_context(roi, &mut ctx)
                .unwrap();
        }
        let reader = image.packet_cache.snapshot().unwrap();
        let retained = image.retained_allocation_bytes().unwrap();
        let directory_bytes = retained - metadata;
        let tight_baseline = DEFAULT_MAX_DECODE_BYTES - directory_bytes + 1;
        let error = image
            .run_decode(&mut ctx, Some(roi), None, false, tight_baseline)
            .unwrap_err();
        assert!(crate::j2c::is_capacity_error(&error));
        assert_eq!(
            image.retained_allocation_bytes().unwrap(),
            retained,
            "a concurrent reader still owns the directory, so its charge remains live"
        );
        drop(reader);
        let error = image
            .run_decode(&mut ctx, Some(roi), None, false, tight_baseline)
            .unwrap_err();
        assert!(crate::j2c::is_capacity_error(&error));
        assert_eq!(
            image.retained_allocation_bytes().unwrap(),
            metadata,
            "the optional directory is released before fallback retries"
        );
        image
            .decode_region_components_with_context(roi, &mut ctx)
            .unwrap();
    }

    #[test]
    fn packet_directory_prepared_session_releases_cache_before_capacity_retry() {
        let bytes =
            include_bytes!("../../fixtures/packet_directory/openjpeg_rgb_origin5x3_rpcl.j2k");
        let image = Image::new(bytes, &DecodeSettings::default()).unwrap();
        let metadata = image.retained_allocation_bytes().unwrap();
        let roi = (60, 40, 16, 16);
        let mut ctx = DecoderContext::default();
        for _ in 0..2 {
            image
                .decode_region_components_with_context(roi, &mut ctx)
                .unwrap();
        }
        let directory_bytes = image.retained_allocation_bytes().unwrap() - metadata;
        let mut prepared = image.prepare_region_decoder_with_context(&mut ctx).unwrap();
        prepared.retained_image_bytes = DEFAULT_MAX_DECODE_BYTES - directory_bytes + 1;
        prepared.retained_session_bytes =
            prepared.retained_image_bytes + prepared.tiles.metadata_owner_bytes();
        let error = prepared.decode_region_components(roi).err().unwrap();
        assert!(crate::j2c::is_capacity_error(&error));
        assert_eq!(image.retained_allocation_bytes().unwrap(), metadata);
    }
}
