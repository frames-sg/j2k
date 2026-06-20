use j2k_jpeg::adapter::{assemble_jpeg_baseline_frame, baseline_encode_tables};
use j2k_jpeg::{
    encode_jpeg_baseline, JpegBackend, JpegEncodeOptions, JpegSamples, JpegSubsampling,
};

#[test]
fn baseline_encode_helper_assembles_same_frame_as_cpu_encoder() {
    let pixels = j2k_test_support::patterned_rgb8(2, 2);
    let options = JpegEncodeOptions {
        quality: 87,
        subsampling: JpegSubsampling::Ybr444,
        restart_interval: Some(4),
        backend: JpegBackend::Cpu,
    };
    let encoded = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &pixels,
            width: 2,
            height: 2,
        },
        options,
    )
    .expect("cpu encode");

    let entropy = entropy_payload(&encoded.data);
    let tables = baseline_encode_tables(options).expect("baseline tables");
    let assembled = assemble_jpeg_baseline_frame(entropy, 2, 2, &tables, options, JpegBackend::Cpu)
        .expect("assemble frame");

    assert_eq!(assembled, encoded);
}

#[test]
fn baseline_encode_tables_expose_sampling_for_adapters() {
    let tables = baseline_encode_tables(JpegEncodeOptions {
        subsampling: JpegSubsampling::Ybr420,
        ..JpegEncodeOptions::default()
    })
    .expect("baseline tables");

    assert_eq!(tables.sampling.components, 3);
    assert_eq!(tables.sampling.h, [2, 1, 1]);
    assert_eq!(tables.sampling.v, [2, 1, 1]);
    assert_eq!(tables.sampling.max_h, 2);
    assert_eq!(tables.sampling.max_v, 2);
}

fn entropy_payload(data: &[u8]) -> &[u8] {
    let mut offset = 0usize;
    while offset + 4 <= data.len() {
        assert_eq!(data[offset], 0xff, "expected marker at offset {offset}");
        let marker = data[offset + 1];
        offset += 2;
        if marker == 0xd8 {
            continue;
        }
        if marker == 0xda {
            let len = u16::from_be_bytes([data[offset], data[offset + 1]]) as usize;
            let entropy_start = offset + len;
            let entropy_end = data
                .windows(2)
                .enumerate()
                .skip(entropy_start)
                .find_map(|(idx, marker)| (marker == [0xff, 0xd9]).then_some(idx))
                .expect("EOI marker");
            return &data[entropy_start..entropy_end];
        }
        let len = u16::from_be_bytes([data[offset], data[offset + 1]]) as usize;
        offset += len;
    }
    panic!("SOS marker not found");
}
