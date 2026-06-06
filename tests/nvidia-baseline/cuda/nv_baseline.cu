// SPDX-License-Identifier: Apache-2.0
//
// NVIDIA GPU baseline for the JPEG -> HTJ2K transcode comparison: nvJPEG decodes
// the JPEG to RGB on the GPU, then nvJPEG2000 encodes it to a High-Throughput
// JPEG 2000 (HTJ2K) codestream on the GPU. This is the apples-to-apples NVIDIA
// path against signinum's coefficient-domain transcode (which skips the pixel
// round-trip). Exposes a tiny C ABI surface for the Rust wrapper.
//
// Compiled by build.rs with nvcc only when the `nvjpeg2000` feature is on and
// the libraries are present. nvJPEG2000 ships separately from the CUDA toolkit.
//
// All locals are declared up front: the cleanup paths use `goto`, and C++
// forbids a goto from jumping over a variable's initialization.

#include <cstring>
#include <cuda_runtime.h>
#include <new>
#include <nvjpeg.h>
#include <nvjpeg2k.h>

// HT enablement macros (guarded in case an older header predates them).
#ifndef NVJPEG2K_RSIZ_HT
#define NVJPEG2K_RSIZ_HT 0x4000
#endif
#ifndef NVJPEG2K_MODE_HT
#define NVJPEG2K_MODE_HT 0x40
#endif

struct NvbSession {
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t mid = nullptr;
    cudaEvent_t stop = nullptr;
    nvjpegHandle_t jpeg_handle = nullptr;
    nvjpegJpegState_t jpeg_state = nullptr;
    nvjpeg2kHandle_t dec_handle = nullptr;
    nvjpeg2kDecodeState_t dec_state = nullptr;
    nvjpeg2kDecodeParams_t dec_params = nullptr;
    nvjpeg2kStream_t dec_stream = nullptr;
    nvjpeg2kEncoder_t enc = nullptr;
    nvjpeg2kEncodeState_t enc_state = nullptr;
    nvjpeg2kEncodeParams_t enc_params = nullptr;
    unsigned char* planes[3] = {nullptr, nullptr, nullptr};
    size_t plane_capacity = 0;
    unsigned char* decode_interleaved = nullptr;
    size_t decode_interleaved_capacity = 0;
};

static void nvb_session_release_planes(NvbSession* session) {
    for (int c = 0; c < 3; ++c) {
        if (session->planes[c]) {
            cudaFree(session->planes[c]);
            session->planes[c] = nullptr;
        }
    }
    session->plane_capacity = 0;
}

static int nvb_session_ensure_planes(NvbSession* session, size_t plane_bytes) {
    if (session->plane_capacity >= plane_bytes) {
        return 0;
    }
    nvb_session_release_planes(session);
    for (int c = 0; c < 3; ++c) {
        if (cudaMalloc((void**)&session->planes[c], plane_bytes) != cudaSuccess) {
            nvb_session_release_planes(session);
            return 902;
        }
    }
    session->plane_capacity = plane_bytes;
    return 0;
}

static void nvb_session_release_decode_interleaved(NvbSession* session) {
    if (session->decode_interleaved) {
        cudaFree(session->decode_interleaved);
        session->decode_interleaved = nullptr;
    }
    session->decode_interleaved_capacity = 0;
}

static int nvb_session_ensure_decode_interleaved(NvbSession* session, size_t bytes) {
    if (session->decode_interleaved_capacity >= bytes) {
        return 0;
    }
    nvb_session_release_decode_interleaved(session);
    if (cudaMalloc((void**)&session->decode_interleaved, bytes) != cudaSuccess) {
        return 902;
    }
    session->decode_interleaved_capacity = bytes;
    return 0;
}

extern "C" {

void nvb_session_destroy(NvbSession* session) {
    if (!session) {
        return;
    }
    if (session->stream) { cudaStreamSynchronize(session->stream); }
    nvb_session_release_decode_interleaved(session);
    nvb_session_release_planes(session);
    if (session->enc_params) { nvjpeg2kEncodeParamsDestroy(session->enc_params); }
    if (session->enc_state) { nvjpeg2kEncodeStateDestroy(session->enc_state); }
    if (session->enc) { nvjpeg2kEncoderDestroy(session->enc); }
    if (session->dec_stream) { nvjpeg2kStreamDestroy(session->dec_stream); }
    if (session->dec_params) { nvjpeg2kDecodeParamsDestroy(session->dec_params); }
    if (session->dec_state) { nvjpeg2kDecodeStateDestroy(session->dec_state); }
    if (session->dec_handle) { nvjpeg2kDestroy(session->dec_handle); }
    if (session->jpeg_state) { nvjpegJpegStateDestroy(session->jpeg_state); }
    if (session->jpeg_handle) { nvjpegDestroy(session->jpeg_handle); }
    if (session->start) { cudaEventDestroy(session->start); }
    if (session->mid) { cudaEventDestroy(session->mid); }
    if (session->stop) { cudaEventDestroy(session->stop); }
    if (session->stream) { cudaStreamDestroy(session->stream); }
    delete session;
}

int nvb_session_create(NvbSession** out) {
    int rc = 0;
    NvbSession* session = nullptr;
    if (!out) { return 900; }
    *out = nullptr;
    session = new (std::nothrow) NvbSession();
    if (!session) { return 904; }

    if (cudaStreamCreate(&session->stream) != cudaSuccess) { rc = 901; goto cleanup; }
    if (cudaEventCreate(&session->start) != cudaSuccess) { rc = 905; goto cleanup; }
    if (cudaEventCreate(&session->mid) != cudaSuccess) { rc = 905; goto cleanup; }
    if (cudaEventCreate(&session->stop) != cudaSuccess) { rc = 905; goto cleanup; }
    if (nvjpegCreateSimple(&session->jpeg_handle) != NVJPEG_STATUS_SUCCESS) { rc = 101; goto cleanup; }
    if (nvjpegJpegStateCreate(session->jpeg_handle, &session->jpeg_state) != NVJPEG_STATUS_SUCCESS) { rc = 102; goto cleanup; }
    if (nvjpeg2kCreateSimple(&session->dec_handle) != NVJPEG2K_STATUS_SUCCESS) { rc = 221; goto cleanup; }
    if (nvjpeg2kDecodeStateCreate(session->dec_handle, &session->dec_state) != NVJPEG2K_STATUS_SUCCESS) { rc = 222; goto cleanup; }
    if (nvjpeg2kDecodeParamsCreate(&session->dec_params) != NVJPEG2K_STATUS_SUCCESS) { rc = 223; goto cleanup; }
    if (nvjpeg2kStreamCreate(&session->dec_stream) != NVJPEG2K_STATUS_SUCCESS) { rc = 224; goto cleanup; }
    if (nvjpeg2kEncoderCreateSimple(&session->enc) != NVJPEG2K_STATUS_SUCCESS) { rc = 201; goto cleanup; }
    if (nvjpeg2kEncodeStateCreate(session->enc, &session->enc_state) != NVJPEG2K_STATUS_SUCCESS) { rc = 202; goto cleanup; }
    if (nvjpeg2kEncodeParamsCreate(&session->enc_params) != NVJPEG2K_STATUS_SUCCESS) { rc = 203; goto cleanup; }

    *out = session;
    return 0;

cleanup:
    nvb_session_destroy(session);
    return rc;
}

int nvb_session_decode_j2k_interleaved(
    NvbSession* session,
    const unsigned char* j2k, size_t j2k_len,
    int requested_format,
    unsigned char* out, size_t out_cap, size_t* out_len,
    double* decode_ms,
    int* width, int* height, int* num_components,
    int* bit_depth, int* bytes_per_sample) {
    int rc = 0;
    nvjpeg2kImageInfo_t image_info;
    nvjpeg2kImageComponentInfo_t comp_info;
    nvjpeg2kImage_t output;
    size_t needed = 0;
    size_t pitch = 0;
    void* pixel_data[1] = {nullptr};
    size_t pitch_in_bytes[1] = {0};
    float decode_elapsed = 0.0f;

    if (!session) { return 900; }
    if (!j2k || j2k_len == 0 || !out || !out_len) { return 920; }

#ifdef NVB_STREAM_PARSE_USES_OUT_POINTER
    if (nvjpeg2kStreamParse(session->dec_handle, j2k, j2k_len, 0, 0, &session->dec_stream)
        != NVJPEG2K_STATUS_SUCCESS) { return 225; }
#else
    if (nvjpeg2kStreamParse(session->dec_handle, j2k, j2k_len, 0, 0, session->dec_stream)
        != NVJPEG2K_STATUS_SUCCESS) { return 225; }
#endif

    memset(&image_info, 0, sizeof(image_info));
    if (nvjpeg2kStreamGetImageInfo(session->dec_stream, &image_info) != NVJPEG2K_STATUS_SUCCESS) {
        return 226;
    }
    if (image_info.num_components == 0 || image_info.num_components > 4) { return 227; }
    memset(&comp_info, 0, sizeof(comp_info));
    if (nvjpeg2kStreamGetImageComponentInfo(session->dec_stream, &comp_info, 0)
        != NVJPEG2K_STATUS_SUCCESS) { return 228; }

    *width = (int)image_info.image_width;
    *height = (int)image_info.image_height;
    *num_components = (int)image_info.num_components;
    *bit_depth = (int)comp_info.precision;
    *bytes_per_sample = (comp_info.precision <= 8) ? 1 : 2;

    if (requested_format == 1) {
        if (image_info.num_components != 3) { return 229; }
        *num_components = 3;
        *bytes_per_sample = 1;
        if (comp_info.precision > 8 || comp_info.sgn != 0) { return 230; }
    } else if (requested_format == 2) {
        if (image_info.num_components != 1) { return 229; }
        *num_components = 1;
        *bytes_per_sample = 1;
        if (comp_info.precision > 8 || comp_info.sgn != 0) { return 230; }
    } else {
        return 231;
    }

    if (nvjpeg2kDecodeParamsSetOutputFormat(session->dec_params, NVJPEG2K_FORMAT_INTERLEAVED)
        != NVJPEG2K_STATUS_SUCCESS) { return 232; }
    if (nvjpeg2kDecodeParamsSetRGBOutput(session->dec_params, requested_format == 1 ? 1 : 0)
        != NVJPEG2K_STATUS_SUCCESS) { return 233; }

    needed = (size_t)(*width) * (size_t)(*height) * (size_t)(*num_components) * (size_t)(*bytes_per_sample);
    pitch = (size_t)(*width) * (size_t)(*num_components) * (size_t)(*bytes_per_sample);
    if (needed > out_cap) { return 234; }
    rc = nvb_session_ensure_decode_interleaved(session, needed);
    if (rc != 0) { return rc; }

    pixel_data[0] = session->decode_interleaved;
    pitch_in_bytes[0] = pitch;
    memset(&output, 0, sizeof(output));
    output.pixel_data = pixel_data;
    output.pitch_in_bytes = pitch_in_bytes;
    output.pixel_type = (*bytes_per_sample == 1) ? NVJPEG2K_UINT8 : NVJPEG2K_UINT16;
    output.num_components = (uint32_t)(*num_components);

    cudaEventRecord(session->start, session->stream);
    if (nvjpeg2kDecodeImage(session->dec_handle, session->dec_state, session->dec_stream, session->dec_params, &output, session->stream)
        != NVJPEG2K_STATUS_SUCCESS) { rc = 235; goto drain; }
    cudaEventRecord(session->stop, session->stream);
    if (cudaStreamSynchronize(session->stream) != cudaSuccess) { return 906; }
    if (cudaMemcpy(out, session->decode_interleaved, needed, cudaMemcpyDeviceToHost) != cudaSuccess) {
        return 903;
    }
    cudaEventElapsedTime(&decode_elapsed, session->start, session->stop);
    *decode_ms = (double)decode_elapsed;
    *out_len = needed;
    return 0;

drain:
    cudaStreamSynchronize(session->stream);
    return rc;
}

// Probe: returns 1 if the nvJPEG and nvJPEG2000 handles can be created.
int nvb_available(void) {
    NvbSession* session = nullptr;
    const int rc = nvb_session_create(&session);
    nvb_session_destroy(session);
    return rc == 0 ? 1 : 0;
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
    int comps = 0;
    nvjpegChromaSubsampling_t subsampling;
    int widths[NVJPEG_MAX_COMPONENT] = {0};
    int heights[NVJPEG_MAX_COMPONENT] = {0};
    int w = 0;
    int h = 0;
    size_t rgb_bytes = 0;
    nvjpegImage_t dest;

    if (cudaStreamCreate(&stream) != cudaSuccess) { return 901; }
    if (nvjpegCreateSimple(&handle) != NVJPEG_STATUS_SUCCESS) { rc = 101; goto cleanup; }
    if (nvjpegJpegStateCreate(handle, &state) != NVJPEG_STATUS_SUCCESS) { rc = 102; goto cleanup; }
    if (nvjpegGetImageInfo(handle, jpeg, jpeg_len, &comps, &subsampling, widths, heights)
        != NVJPEG_STATUS_SUCCESS) { rc = 103; goto cleanup; }

    w = widths[0];
    h = heights[0];
    *width = w;
    *height = h;
    rgb_bytes = (size_t)w * (size_t)h * 3;
    if (rgb_bytes > out_cap) { rc = 120; goto cleanup; }
    if (cudaMalloc((void**)&dev, rgb_bytes) != cudaSuccess) { rc = 902; goto cleanup; }

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
    if (stream) { cudaStreamSynchronize(stream); }
    if (dev) { cudaFree(dev); }
    if (state) { nvjpegJpegStateDestroy(state); }
    if (handle) { nvjpegDestroy(handle); }
    if (stream) { cudaStreamDestroy(stream); }
    return rc;
}

// Reused-session decode timing: JPEG -> interleaved RGB in device memory.
// Returns only a CUDA event duration for the decode submission; no host download.
int nvb_session_decode_jpeg_rgb_interleaved_timed(
    NvbSession* session,
    const unsigned char* jpeg, size_t jpeg_len,
    double* decode_ms,
    int* width, int* height) {
    int comps = 0;
    nvjpegChromaSubsampling_t subsampling;
    int widths[NVJPEG_MAX_COMPONENT] = {0};
    int heights[NVJPEG_MAX_COMPONENT] = {0};
    int w = 0;
    int h = 0;
    size_t rgb_bytes = 0;
    nvjpegImage_t dest;
    float decode_elapsed = 0.0f;

    if (!session || !jpeg || jpeg_len == 0 || !decode_ms || !width || !height) { return 920; }
    if (nvjpegGetImageInfo(session->jpeg_handle, jpeg, jpeg_len, &comps, &subsampling, widths, heights)
        != NVJPEG_STATUS_SUCCESS) { return 103; }

    w = widths[0];
    h = heights[0];
    *width = w;
    *height = h;
    rgb_bytes = (size_t)w * (size_t)h * 3;
    if (nvb_session_ensure_decode_interleaved(session, rgb_bytes) != 0) { return 902; }

    memset(&dest, 0, sizeof(dest));
    dest.channel[0] = session->decode_interleaved;
    dest.pitch[0] = (size_t)w * 3;

    cudaEventRecord(session->start, session->stream);
    if (nvjpegDecode(session->jpeg_handle, session->jpeg_state, jpeg, jpeg_len, NVJPEG_OUTPUT_RGBI, &dest, session->stream)
        != NVJPEG_STATUS_SUCCESS) { return 110; }
    cudaEventRecord(session->stop, session->stream);
    if (cudaEventSynchronize(session->stop) != cudaSuccess) { return 906; }
    if (cudaEventElapsedTime(&decode_elapsed, session->start, session->stop) != cudaSuccess) { return 907; }
    *decode_ms = (double)decode_elapsed;
    return 0;
}

// Reused-session GPU transcode: JPEG bytes -> HTJ2K bytes. Returns 0 on success,
// or a non-zero stage code (1xx nvJPEG decode, 2xx nvJPEG2000 encode, 9xx CUDA).
// `decode_ms` / `encode_ms` are GPU stage times (cudaEvent). `out` must have
// `out_cap` bytes; on success `*out_len` holds the codestream length.
int nvb_session_transcode_jpeg_to_htj2k(
    NvbSession* session,
    const unsigned char* jpeg, size_t jpeg_len,
    unsigned char* out, size_t out_cap, size_t* out_len,
    double* decode_ms, double* encode_ms,
    int* width, int* height, int* num_components) {
    int rc = 0;
    int comps = 0;
    nvjpegChromaSubsampling_t subsampling;
    int widths[NVJPEG_MAX_COMPONENT] = {0};
    int heights[NVJPEG_MAX_COMPONENT] = {0};
    int w = 0;
    int h = 0;
    size_t plane_bytes = 0;
    nvjpegImage_t dest;
    nvjpeg2kImageComponentInfo_t comp_info[3];
    int levels = 0;
    int axis = 0;
    nvjpeg2kEncodeConfig_t config;
    void* plane_ptrs[3] = {nullptr, nullptr, nullptr};
    size_t pitches[3] = {0, 0, 0};
    nvjpeg2kImage_t input;
    size_t length = 0;
    float decode_elapsed = 0.0f;
    float encode_elapsed = 0.0f;

    if (!session) { return 900; }

    if (nvjpegGetImageInfo(session->jpeg_handle, jpeg, jpeg_len, &comps, &subsampling, widths, heights)
        != NVJPEG_STATUS_SUCCESS) { return 103; }

    w = widths[0];
    h = heights[0];
    *width = w;
    *height = h;
    *num_components = 3;

    // Planar RGB destination (one plane per channel, tightly packed).
    plane_bytes = (size_t)w * (size_t)h;
    rc = nvb_session_ensure_planes(session, plane_bytes);
    if (rc != 0) { return rc; }
    memset(&dest, 0, sizeof(dest));
    for (int c = 0; c < 3; ++c) {
        dest.channel[c] = session->planes[c];
        dest.pitch[c] = (size_t)w;
    }

    cudaEventRecord(session->start, session->stream);
    if (nvjpegDecode(session->jpeg_handle, session->jpeg_state, jpeg, jpeg_len, NVJPEG_OUTPUT_RGB, &dest, session->stream)
        != NVJPEG_STATUS_SUCCESS) { rc = 110; goto drain; }
    cudaEventRecord(session->mid, session->stream);

    // --- nvJPEG2000 HTJ2K encode of the planar RGB ---
    for (int c = 0; c < 3; ++c) {
        comp_info[c].component_width = (uint32_t)w;
        comp_info[c].component_height = (uint32_t)h;
        comp_info[c].precision = 8;
        comp_info[c].sgn = 0;
    }

    // Resolutions: cap decomposition levels so the LL band stays >= 1 sample.
    axis = (w < h) ? w : h;
    while (axis > 1 && levels < 5) { axis >>= 1; ++levels; }

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

    if (nvjpeg2kEncodeParamsSetEncodeConfig(session->enc_params, &config) != NVJPEG2K_STATUS_SUCCESS) {
        rc = 204; goto drain;
    }

    plane_ptrs[0] = session->planes[0];
    plane_ptrs[1] = session->planes[1];
    plane_ptrs[2] = session->planes[2];
    pitches[0] = (size_t)w;
    pitches[1] = (size_t)w;
    pitches[2] = (size_t)w;
    memset(&input, 0, sizeof(input));
    input.pixel_data = plane_ptrs;
    input.pitch_in_bytes = pitches;
    input.pixel_type = NVJPEG2K_UINT8;
    input.num_components = 3;

    if (nvjpeg2kEncode(session->enc, session->enc_state, session->enc_params, &input, session->stream) != NVJPEG2K_STATUS_SUCCESS) {
        rc = 210; goto drain;
    }

    if (nvjpeg2kEncodeRetrieveBitstream(session->enc, session->enc_state, nullptr, &length, session->stream)
        != NVJPEG2K_STATUS_SUCCESS) { rc = 211; goto drain; }
    if (length > out_cap) { rc = 212; goto drain; }
    if (nvjpeg2kEncodeRetrieveBitstream(session->enc, session->enc_state, out, &length, session->stream)
        != NVJPEG2K_STATUS_SUCCESS) { rc = 213; goto drain; }
    cudaEventRecord(session->stop, session->stream);
    if (cudaStreamSynchronize(session->stream) != cudaSuccess) { return 906; }
    *out_len = length;

    cudaEventElapsedTime(&decode_elapsed, session->start, session->mid);
    cudaEventElapsedTime(&encode_elapsed, session->mid, session->stop);
    *decode_ms = (double)decode_elapsed;
    *encode_ms = (double)encode_elapsed;
    return 0;

drain:
    cudaStreamSynchronize(session->stream);
    return rc;
}

// Compatibility one-shot wrapper.
int nvb_transcode_jpeg_to_htj2k(
    const unsigned char* jpeg, size_t jpeg_len,
    unsigned char* out, size_t out_cap, size_t* out_len,
    double* decode_ms, double* encode_ms,
    int* width, int* height, int* num_components) {
    int rc = 0;
    NvbSession* session = nullptr;
    rc = nvb_session_create(&session);
    if (rc != 0) { return rc; }
    rc = nvb_session_transcode_jpeg_to_htj2k(
        session,
        jpeg,
        jpeg_len,
        out,
        out_cap,
        out_len,
        decode_ms,
        encode_ms,
        width,
        height,
        num_components
    );
    nvb_session_destroy(session);
    return rc;
}

} // extern "C"
