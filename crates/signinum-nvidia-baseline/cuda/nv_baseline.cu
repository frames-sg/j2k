// SPDX-License-Identifier: Apache-2.0
//
// NVIDIA GPU baseline for the JPEG -> HTJ2K transcode comparison: nvJPEG decodes
// the JPEG to planar RGB on the GPU, then nvJPEG2000 encodes it to a
// High-Throughput JPEG 2000 (HTJ2K) codestream on the GPU. This is the
// apples-to-apples NVIDIA path against signinum's coefficient-domain transcode
// (which skips the pixel round-trip). Exposes a single C ABI entry point so the
// Rust side keeps a tiny FFI surface.
//
// Compiled by build.rs with nvcc only when the `nvjpeg2000` feature is on and
// the libraries are present. nvJPEG2000 ships separately from the CUDA toolkit.

#include <cstring>
#include <cuda_runtime.h>
#include <nvjpeg.h>
#include <nvjpeg2k.h>

// HT enablement macros (guarded in case an older header predates them).
#ifndef NVJPEG2K_RSIZ_HT
#define NVJPEG2K_RSIZ_HT 0x4000
#endif
#ifndef NVJPEG2K_MODE_HT
#define NVJPEG2K_MODE_HT 0x40
#endif

extern "C" {

// Probe: returns 1 if the nvJPEG and nvJPEG2000 handles can be created.
int nvb_available(void) {
    nvjpegHandle_t jpeg = nullptr;
    nvjpeg2kEncoder_t enc = nullptr;
    int ok = 1;
    if (nvjpegCreateSimple(&jpeg) != NVJPEG_STATUS_SUCCESS) {
        ok = 0;
    }
    if (nvjpeg2kEncoderCreateSimple(&enc) != NVJPEG2K_STATUS_SUCCESS) {
        ok = 0;
    }
    if (enc) {
        nvjpeg2kEncoderDestroy(enc);
    }
    if (jpeg) {
        nvjpegDestroy(jpeg);
    }
    return ok;
}

// Reference decode (untimed): JPEG -> interleaved RGB on the host, for PSNR.
// `out_rgb` must hold width*height*3 bytes. Returns 0 on success.
int nvb_decode_jpeg_rgb(
    const unsigned char* jpeg, size_t jpeg_len,
    unsigned char* out_rgb, size_t out_cap, int* width, int* height) {
    int rc = 0;
    cudaStream_t stream = nullptr;
    nvjpegHandle_t handle = nullptr;
    nvjpegJpegState_t state = nullptr;
    unsigned char* dev = nullptr;

    if (cudaStreamCreate(&stream) != cudaSuccess) { return 901; }
    if (nvjpegCreateSimple(&handle) != NVJPEG_STATUS_SUCCESS) { rc = 101; goto cleanup; }
    if (nvjpegJpegStateCreate(handle, &state) != NVJPEG_STATUS_SUCCESS) { rc = 102; goto cleanup; }

    int comps;
    nvjpegChromaSubsampling_t subsampling;
    int widths[NVJPEG_MAX_COMPONENT];
    int heights[NVJPEG_MAX_COMPONENT];
    if (nvjpegGetImageInfo(handle, jpeg, jpeg_len, &comps, &subsampling, widths, heights)
        != NVJPEG_STATUS_SUCCESS) { rc = 103; goto cleanup; }

    int w = widths[0];
    int h = heights[0];
    *width = w;
    *height = h;
    size_t rgb_bytes = (size_t)w * (size_t)h * 3;
    if (rgb_bytes > out_cap) { rc = 120; goto cleanup; }
    if (cudaMalloc((void**)&dev, rgb_bytes) != cudaSuccess) { rc = 902; goto cleanup; }

    nvjpegImage_t dest;
    memset(&dest, 0, sizeof(dest));
    dest.channel[0] = dev;          // interleaved RGB lands in channel[0]
    dest.pitch[0] = (size_t)w * 3;
    if (nvjpegDecode(handle, state, jpeg, jpeg_len, NVJPEG_OUTPUT_RGBI, &dest, stream)
        != NVJPEG_STATUS_SUCCESS) { rc = 110; goto cleanup; }
    cudaStreamSynchronize(stream);
    if (cudaMemcpy(out_rgb, dev, rgb_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        rc = 903; goto cleanup;
    }

cleanup:
    if (dev) { cudaFree(dev); }
    if (state) { nvjpegJpegStateDestroy(state); }
    if (handle) { nvjpegDestroy(handle); }
    if (stream) { cudaStreamDestroy(stream); }
    return rc;
}

// Full GPU transcode: JPEG bytes -> HTJ2K bytes. Returns 0 on success, or a
// non-zero stage code (1xx nvJPEG decode, 2xx nvJPEG2000 encode, 9xx CUDA).
// `decode_ms` / `encode_ms` are GPU stage times (cudaEvent). `out` must have
// `out_cap` bytes; on success `*out_len` holds the codestream length.
int nvb_transcode_jpeg_to_htj2k(
    const unsigned char* jpeg, size_t jpeg_len,
    unsigned char* out, size_t out_cap, size_t* out_len,
    double* decode_ms, double* encode_ms,
    int* width, int* height, int* num_components) {
    int rc = 0;
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr, mid = nullptr, stop = nullptr;
    nvjpegHandle_t jpeg_handle = nullptr;
    nvjpegJpegState_t jpeg_state = nullptr;
    nvjpeg2kEncoder_t enc = nullptr;
    nvjpeg2kEncodeState_t enc_state = nullptr;
    nvjpeg2kEncodeParams_t enc_params = nullptr;
    unsigned char* planes[3] = {nullptr, nullptr, nullptr};

    if (cudaStreamCreate(&stream) != cudaSuccess) { return 901; }
    cudaEventCreate(&start);
    cudaEventCreate(&mid);
    cudaEventCreate(&stop);

    if (nvjpegCreateSimple(&jpeg_handle) != NVJPEG_STATUS_SUCCESS) { rc = 101; goto cleanup; }
    if (nvjpegJpegStateCreate(jpeg_handle, &jpeg_state) != NVJPEG_STATUS_SUCCESS) { rc = 102; goto cleanup; }

    int comps;
    nvjpegChromaSubsampling_t subsampling;
    int widths[NVJPEG_MAX_COMPONENT];
    int heights[NVJPEG_MAX_COMPONENT];
    if (nvjpegGetImageInfo(jpeg_handle, jpeg, jpeg_len, &comps, &subsampling, widths, heights)
        != NVJPEG_STATUS_SUCCESS) { rc = 103; goto cleanup; }

    int w = widths[0];
    int h = heights[0];
    *width = w;
    *height = h;
    *num_components = 3;

    // Planar RGB destination (one plane per channel, tightly packed).
    size_t plane_bytes = (size_t)w * (size_t)h;
    for (int c = 0; c < 3; ++c) {
        if (cudaMalloc((void**)&planes[c], plane_bytes) != cudaSuccess) { rc = 902; goto cleanup; }
    }
    nvjpegImage_t dest;
    memset(&dest, 0, sizeof(dest));
    for (int c = 0; c < 3; ++c) {
        dest.channel[c] = planes[c];
        dest.pitch[c] = (size_t)w;
    }

    cudaEventRecord(start, stream);
    if (nvjpegDecode(jpeg_handle, jpeg_state, jpeg, jpeg_len, NVJPEG_OUTPUT_RGB, &dest, stream)
        != NVJPEG_STATUS_SUCCESS) { rc = 110; goto cleanup; }
    cudaEventRecord(mid, stream);

    // --- nvJPEG2000 HTJ2K encode of the planar RGB ---
    if (nvjpeg2kEncoderCreateSimple(&enc) != NVJPEG2K_STATUS_SUCCESS) { rc = 201; goto cleanup; }
    if (nvjpeg2kEncodeStateCreate(enc, &enc_state) != NVJPEG2K_STATUS_SUCCESS) { rc = 202; goto cleanup; }
    if (nvjpeg2kEncodeParamsCreate(&enc_params) != NVJPEG2K_STATUS_SUCCESS) { rc = 203; goto cleanup; }

    nvjpeg2kImageComponentInfo_t comp_info[3];
    for (int c = 0; c < 3; ++c) {
        comp_info[c].component_width = (uint32_t)w;
        comp_info[c].component_height = (uint32_t)h;
        comp_info[c].precision = 8;
        comp_info[c].sgn = 0;
    }

    // Resolutions: cap decomposition levels so the LL band stays >= 1 sample.
    int levels = 0;
    int axis = (w < h) ? w : h;
    while (axis > 1 && levels < 5) { axis >>= 1; ++levels; }

    nvjpeg2kEncodeConfig_t config;
    memset(&config, 0, sizeof(config));
    config.stream_type = NVJPEG2K_STREAM_J2K;
    config.color_space = NVJPEG2K_COLORSPACE_SRGB;
    config.rsiz = NVJPEG2K_RSIZ_HT;
    config.image_width = (uint32_t)w;
    config.image_height = (uint32_t)h;
    config.enable_tiling = 0;
    config.num_components = 3;
    config.image_comp_info = comp_info;
    config.prog_order = NVJPEG2K_LRCP;
    config.num_layers = 1;
    config.mct_mode = 1; // RGB input: apply the multi-component (color) transform.
    config.num_resolutions = (uint32_t)(levels + 1);
    config.code_block_w = 64;
    config.code_block_h = 64;
    config.encode_modes = NVJPEG2K_MODE_HT;
    config.irreversible = 1; // 9/7 irreversible path.

    if (nvjpeg2kEncodeParamsSetEncodeConfig(enc_params, &config) != NVJPEG2K_STATUS_SUCCESS) {
        rc = 204; goto cleanup;
    }

    void* plane_ptrs[3] = {planes[0], planes[1], planes[2]};
    size_t pitches[3] = {(size_t)w, (size_t)w, (size_t)w};
    nvjpeg2kImage_t input;
    memset(&input, 0, sizeof(input));
    input.pixel_data = plane_ptrs;
    input.pitch_in_bytes = pitches;
    input.pixel_type = NVJPEG2K_UINT8;
    input.num_components = 3;

    if (nvjpeg2kEncode(enc, enc_state, enc_params, &input, stream) != NVJPEG2K_STATUS_SUCCESS) {
        rc = 210; goto cleanup;
    }
    cudaEventRecord(stop, stream);
    cudaStreamSynchronize(stream);

    size_t length = 0;
    if (nvjpeg2kEncodeRetrieveBitstream(enc, enc_state, nullptr, &length, stream)
        != NVJPEG2K_STATUS_SUCCESS) { rc = 211; goto cleanup; }
    if (length > out_cap) { rc = 212; goto cleanup; }
    if (nvjpeg2kEncodeRetrieveBitstream(enc, enc_state, out, &length, stream)
        != NVJPEG2K_STATUS_SUCCESS) { rc = 213; goto cleanup; }
    cudaStreamSynchronize(stream);
    *out_len = length;

    float decode_elapsed = 0.0f, encode_elapsed = 0.0f;
    cudaEventElapsedTime(&decode_elapsed, start, mid);
    cudaEventElapsedTime(&encode_elapsed, mid, stop);
    *decode_ms = (double)decode_elapsed;
    *encode_ms = (double)encode_elapsed;

cleanup:
    for (int c = 0; c < 3; ++c) {
        if (planes[c]) { cudaFree(planes[c]); }
    }
    if (enc_params) { nvjpeg2kEncodeParamsDestroy(enc_params); }
    if (enc_state) { nvjpeg2kEncodeStateDestroy(enc_state); }
    if (enc) { nvjpeg2kEncoderDestroy(enc); }
    if (jpeg_state) { nvjpegJpegStateDestroy(jpeg_state); }
    if (jpeg_handle) { nvjpegDestroy(jpeg_handle); }
    if (start) { cudaEventDestroy(start); }
    if (mid) { cudaEventDestroy(mid); }
    if (stop) { cudaEventDestroy(stop); }
    if (stream) { cudaStreamDestroy(stream); }
    return rc;
}

} // extern "C"
