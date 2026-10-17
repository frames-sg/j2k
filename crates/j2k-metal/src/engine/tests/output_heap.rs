// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::{BackendRequest, PixelFormat};
use j2k_native::{encode_htj2k, EncodeOptions};
use objc2::Message as _;
use objc2_metal::MTLResource as _;

use super::runtime::should_run_metal_runtime;
use crate::{J2kDecoder, MetalDecodeRequest};

fn gray16(seed: u32) -> Vec<u8> {
    let (width, height) = (300_u32, 200_u32);
    let mut pixels = Vec::with_capacity(width as usize * height as usize * 2);
    for y in 0..height {
        for x in 0..width {
            let value = u16::try_from((x * 31 + y * 17 + seed * 977) % 4096).expect("fits u16");
            pixels.extend_from_slice(&value.to_le_bytes());
        }
    }
    encode_htj2k(
        &pixels,
        width,
        height,
        1,
        16,
        false,
        &EncodeOptions::default(),
    )
    .expect("encode")
}

fn decode(bytes: &[u8]) -> crate::Surface {
    J2kDecoder::new(bytes)
        .expect("decoder")
        .decode_request_to_device(MetalDecodeRequest::full(
            PixelFormat::Gray16,
            BackendRequest::Metal,
        ))
        .expect("Metal decode")
}

#[test]
fn outputs_keep_their_bytes_while_any_reference_is_held() {
    if !should_run_metal_runtime() {
        return;
    }
    let first = gray16(1);
    let held = decode(&first);
    let expected = held.as_bytes().expect("held bytes").into_owned();

    // A caller may keep the Metal buffer after dropping its surface.
    let dropped = decode(&first);
    // SAFETY: the decode completed, and this test only reads the buffer.
    let (buffer, offset) = unsafe { dropped.metal_buffer() }.expect("Metal-backed surface");
    assert!(
        buffer.heap().is_some(),
        "outputs come from the runtime's heap"
    );
    let retained = buffer.retain();
    drop(dropped);

    // Later outputs of the same size come from the same heap.
    for seed in 2..10 {
        drop(decode(&gray16(seed)));
    }

    assert_eq!(
        held.as_bytes().expect("held bytes").as_ref(),
        expected.as_slice()
    );
    // SAFETY: no GPU work references the retained buffer.
    let retained_bytes = unsafe {
        j2k_metal_support::checked_buffer_read_vec::<u8>(&retained, offset, expected.len())
    }
    .expect("retained bytes");
    assert_eq!(retained_bytes, expected);
}
