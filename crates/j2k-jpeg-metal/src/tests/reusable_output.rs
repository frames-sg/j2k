// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[test]
fn metal_batch_output_buffer_ensure_reuses_matching_allocation_and_grows_capacity() {
    use metal::foreign_types::ForeignTypeRef;

    if !should_run_metal_runtime() {
        return;
    }

    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let mut output =
        MetalBatchOutputBuffer::new_rgb8_tiles(&session, (16, 16), 2).expect("output buffer");
    let original_buffer = output.buffer_trusted().as_ptr();

    output
        .ensure_rgb8_tiles(&session, (16, 16), 1)
        .expect("ensure smaller matching output");
    assert_eq!(output.buffer_trusted().as_ptr(), original_buffer);
    assert_eq!(output.dimensions(), (16, 16));
    assert_eq!(output.tile_capacity(), 2);

    output
        .ensure_rgb8_tiles(&session, (16, 16), 3)
        .expect("ensure larger output");
    assert_ne!(output.buffer_trusted().as_ptr(), original_buffer);
    assert_eq!(output.dimensions(), (16, 16));
    assert_eq!(output.tile_capacity(), 3);
    assert_eq!(
        output.byte_len(),
        16 * 16 * PixelFormat::Rgb8.bytes_per_pixel() * 3
    );
}

#[test]
fn cloned_reusable_output_surface_readback_waits_for_safe_output_access() {
    if !should_run_metal_runtime() {
        return;
    }

    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let output =
        MetalBatchOutputBuffer::new_rgb8_tiles(&session, (1, 1), 1).expect("output buffer");
    let output_alias = output.clone();
    let surface =
        Surface::from_batch_output_buffer_offset(&output_alias, (1, 1), PixelFormat::Rgb8, 0);
    assert!(
        output.shares_access_gate_with(&surface),
        "cloned output and derived surface must share one allocation gate"
    );
    let output_access = output
        .lock_for_safe_access()
        .expect("lock reusable output access");
    let (started_tx, started_rx) = std::sync::mpsc::channel();
    let (finished_tx, finished_rx) = std::sync::mpsc::channel();
    let reader = std::thread::spawn(move || {
        started_tx.send(()).expect("signal readback start");
        let byte_len = surface.as_bytes().len();
        finished_tx
            .send(byte_len)
            .expect("signal readback completion");
    });

    started_rx.recv().expect("readback thread started");
    assert!(
        finished_rx
            .recv_timeout(std::time::Duration::from_millis(100))
            .is_err(),
        "readback must wait while the output allocation is locked for a safe writer"
    );

    drop(output_access);
    assert_eq!(
        finished_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("readback completed after releasing output access"),
        PixelFormat::Rgb8.bytes_per_pixel()
    );
    reader.join().expect("readback thread joined");
}

#[test]
fn reusable_output_surface_download_reports_poisoned_access_gate() {
    if !should_run_metal_runtime() {
        return;
    }

    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let output =
        MetalBatchOutputBuffer::new_rgb8_tiles(&session, (1, 1), 1).expect("output buffer");
    let surface = Surface::from_batch_output_buffer_offset(&output, (1, 1), PixelFormat::Rgb8, 0);
    let poison_output = output.clone();
    let poisoner = std::thread::spawn(move || {
        let _access = poison_output
            .lock_for_safe_access()
            .expect("lock reusable output access before poisoning");
        panic!("poison reusable output access gate for regression coverage");
    });
    assert!(poisoner.join().is_err(), "poisoning thread must panic");

    let mut bytes = [0_u8; 3];
    let error = surface
        .download_into(&mut bytes, 3)
        .expect_err("fallible download must report a poisoned access gate");
    assert!(
        matches!(&error, Error::MetalKernel { message } if message.contains("access gate was poisoned")),
        "unexpected poisoned-gate error: {error:?}"
    );
}

#[test]
fn metal_batch_texture_output_ensure_reuses_matching_textures_and_grows_capacity() {
    use metal::foreign_types::ForeignTypeRef;

    if !should_run_metal_runtime() {
        return;
    }

    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let mut output =
        MetalBatchTextureOutput::new_rgba8_tiles(&session, (16, 16), 2).expect("texture output");
    let original_texture = output.texture_trusted(0).expect("texture").as_ptr();

    output
        .ensure_rgba8_tiles(&session, (16, 16), 1)
        .expect("ensure smaller matching texture output");
    assert_eq!(
        output.texture_trusted(0).expect("texture").as_ptr(),
        original_texture
    );
    assert_eq!(output.dimensions(), (16, 16));
    assert_eq!(output.tile_capacity(), 2);

    output
        .ensure_rgba8_tiles(&session, (16, 16), 3)
        .expect("ensure larger texture output");
    assert_ne!(
        output.texture_trusted(0).expect("texture").as_ptr(),
        original_texture
    );
    assert_eq!(output.dimensions(), (16, 16));
    assert_eq!(output.tile_capacity(), 3);
    assert_eq!(output.pixel_format(), PixelFormat::Rgba8);
}

#[test]
fn reusable_texture_output_clones_and_subsets_share_one_access_gate() {
    if !should_run_metal_runtime() {
        return;
    }

    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let mut output =
        MetalBatchTextureOutput::new_rgba8_tiles(&session, (4, 4), 2).expect("texture output");
    let output_clone = output.clone();
    let output_subset = output.clone_slots(&[1]).expect("texture output subset");
    assert!(output.shares_access_gate_with(&output_clone));
    assert!(output.shares_access_gate_with(&output_subset));

    let output_access = output
        .lock_for_safe_access()
        .expect("lock texture output access");
    let (started_tx, started_rx) = std::sync::mpsc::channel();
    let (finished_tx, finished_rx) = std::sync::mpsc::channel();
    let writer = std::thread::spawn(move || {
        started_tx.send(()).expect("report gate wait start");
        let _clone_access = output_clone
            .lock_for_safe_access()
            .expect("lock cloned texture output access");
        finished_tx.send(()).expect("report cloned gate acquired");
    });
    started_rx.recv().expect("cloned writer started");
    assert!(
        finished_rx
            .recv_timeout(std::time::Duration::from_millis(100))
            .is_err(),
        "a cloned output must wait while the shared allocation gate is held"
    );

    drop(output_access);
    finished_rx
        .recv_timeout(std::time::Duration::from_secs(5))
        .expect("cloned writer acquired the released allocation gate");
    writer.join().expect("cloned writer joined");

    let old_allocation = output_subset;
    output
        .ensure_rgba8_tiles(&session, (8, 8), 1)
        .expect("replace texture allocation");
    assert!(
        !output.shares_access_gate_with(&old_allocation),
        "a resized output must receive a gate for its new allocation"
    );
}
