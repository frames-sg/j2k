#include "grok.h"
#include "j2k_shim_common.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if GRK_VERSION_MAJOR == 10
void j2k_grok_initialize(uint32_t threads) { (void)grk_initialize(NULL, threads); }
#elif GRK_VERSION_MAJOR == 20
void j2k_grok_initialize(uint32_t threads) { grk_initialize(NULL, threads, NULL); }
#else
#error "j2k Grok shim supports Grok 10.x and 20.x"
#endif

static uint8_t j2k_clamp_u8(int32_t value) {
  if (value < 0) {
    return 0;
  }
  if (value > 255) {
    return 255;
  }
  return (uint8_t)value;
}

static int j2k_grok_checked_mul_size(size_t lhs, size_t rhs, size_t *out) {
  if (!out) {
    return 0;
  }
  if (lhs != 0 && rhs > SIZE_MAX / lhs) {
    return 0;
  }
  *out = lhs * rhs;
  return 1;
}

static int j2k_grok_checked_add_size(size_t lhs, size_t rhs, size_t *out) {
  if (!out) {
    return 0;
  }
  if (rhs > SIZE_MAX - lhs) {
    return 0;
  }
  *out = lhs + rhs;
  return 1;
}

static int j2k_grok_validate_component_count(uint32_t numcomps,
                                             uint32_t channels) {
  if (numcomps == 0) {
    return 0;
  }
  if (channels == 1) {
    return 1;
  }
  if (channels == 3 && (numcomps == 1 || numcomps >= 3)) {
    return 1;
  }
  return 0;
}

static int j2k_grok_component_index(const grk_image_comp *component,
                                    uint32_t row, uint32_t col,
                                    size_t *index) {
  size_t row_offset = 0;
  if (!component || !component->data || !index) {
    return 0;
  }
  if (component->w == 0 || component->h == 0 || component->stride == 0) {
    return 0;
  }
  if (row >= component->h || col >= component->w ||
      component->stride < component->w) {
    return 0;
  }
  if (!j2k_grok_checked_mul_size((size_t)row, (size_t)component->stride,
                                 &row_offset)) {
    return 0;
  }
  return j2k_grok_checked_add_size(row_offset, (size_t)col, index);
}

static int j2k_grok_component_sample_u8(const grk_image_comp *component,
                                        size_t index, uint8_t *sample) {
  if (!component || !component->data || !sample) {
    return 0;
  }
#if GRK_VERSION_MAJOR == 10
  *sample = j2k_clamp_u8(component->data[index]);
  return 1;
#else
  switch (component->data_type) {
  case GRK_INT_8:
    if (!component->sgnd) {
      *sample = ((const uint8_t *)component->data)[index];
    } else {
      *sample = j2k_clamp_u8(((const int8_t *)component->data)[index]);
    }
    return 1;
  case GRK_INT_16:
    if (!component->sgnd) {
      uint16_t value = ((const uint16_t *)component->data)[index];
      *sample = value > UINT8_MAX ? UINT8_MAX : (uint8_t)value;
    } else {
      *sample = j2k_clamp_u8(((const int16_t *)component->data)[index]);
    }
    return 1;
  case GRK_INT_32:
    if (!component->sgnd) {
      uint32_t value = ((const uint32_t *)component->data)[index];
      *sample = value > UINT8_MAX ? UINT8_MAX : (uint8_t)value;
    } else {
      *sample = j2k_clamp_u8(((const int32_t *)component->data)[index]);
    }
    return 1;
  default:
    return 0;
  }
#endif
}

#if GRK_VERSION_MAJOR != 10
/* Whether every row the packer reads lies inside the component plane. */
static int j2k_grok_component_covers(const grk_image_comp *c, uint32_t width,
                                     uint32_t height) {
  return c && c->data && c->stride != 0 && c->stride >= c->w &&
         c->w >= width && c->h >= height;
}

static inline uint8_t j2k_u8_from_u8(uint8_t v) { return v; }
static inline uint8_t j2k_u8_from_s8(int8_t v) { return j2k_clamp_u8(v); }
static inline uint8_t j2k_u8_from_u16(uint16_t v) { return v > UINT8_MAX ? UINT8_MAX : (uint8_t)v; }
static inline uint8_t j2k_u8_from_s16(int16_t v) { return j2k_clamp_u8(v); }
static inline uint8_t j2k_u8_from_u32(uint32_t v) { return v > UINT8_MAX ? UINT8_MAX : (uint8_t)v; }
static inline uint8_t j2k_u8_from_s32(int32_t v) { return j2k_clamp_u8(v); }

/* Packs rows with the components' sample type known outside the loops, the
 * same conversions as j2k_grok_component_sample_u8 applies per sample. */
#define J2K_GROK_PACK(T, CONV)                                                  \
  do {                                                                          \
    for (uint32_t row = 0; row < height; ++row) {                               \
      uint8_t *out = packed + (size_t)row * width * channels;                   \
      const T *r0 = (const T *)c0->data + (size_t)row * c0->stride;             \
      if (channels == 1) {                                                      \
        for (uint32_t col = 0; col < width; ++col) out[col] = CONV(r0[col]);    \
      } else if (image->numcomps == 1) {                                        \
        for (uint32_t col = 0; col < width; ++col) {                            \
          uint8_t v = CONV(r0[col]);                                            \
          out[3 * (size_t)col] = v;                                             \
          out[3 * (size_t)col + 1] = v;                                         \
          out[3 * (size_t)col + 2] = v;                                         \
        }                                                                       \
      } else {                                                                  \
        const T *r1 = (const T *)c1->data + (size_t)row * c1->stride;           \
        const T *r2 = (const T *)c2->data + (size_t)row * c2->stride;           \
        for (uint32_t col = 0; col < width; ++col) {                            \
          out[3 * (size_t)col] = CONV(r0[col]);                                 \
          out[3 * (size_t)col + 1] = CONV(r1[col]);                             \
          out[3 * (size_t)col + 2] = CONV(r2[col]);                             \
        }                                                                       \
      }                                                                         \
    }                                                                           \
  } while (0)

/* Fast path: every used component covers the output and shares one sample
 * type. Returns 0 when that does not hold, so the caller falls back to the
 * checked per-sample path. */
static int j2k_grok_pack_fast(const grk_image *image, uint32_t width,
                              uint32_t height, uint32_t channels,
                              uint8_t *packed) {
  const grk_image_comp *c0 = &image->comps[0];
  const grk_image_comp *c1 = c0;
  const grk_image_comp *c2 = c0;
  if (!j2k_grok_component_covers(c0, width, height)) {
    return 0;
  }
  if (channels == 3 && image->numcomps >= 3) {
    c1 = &image->comps[1];
    c2 = &image->comps[2];
    if (!j2k_grok_component_covers(c1, width, height) ||
        !j2k_grok_component_covers(c2, width, height) ||
        c1->data_type != c0->data_type || c2->data_type != c0->data_type ||
        c1->sgnd != c0->sgnd || c2->sgnd != c0->sgnd) {
      return 0;
    }
  }
  switch (c0->data_type) {
  case GRK_INT_8:
    if (c0->sgnd) J2K_GROK_PACK(int8_t, j2k_u8_from_s8); else J2K_GROK_PACK(uint8_t, j2k_u8_from_u8);
    return 1;
  case GRK_INT_16:
    if (c0->sgnd) J2K_GROK_PACK(int16_t, j2k_u8_from_s16); else J2K_GROK_PACK(uint16_t, j2k_u8_from_u16);
    return 1;
  case GRK_INT_32:
    if (c0->sgnd) J2K_GROK_PACK(int32_t, j2k_u8_from_s32); else J2K_GROK_PACK(uint32_t, j2k_u8_from_u32);
    return 1;
  default:
    return 0;
  }
}
#undef J2K_GROK_PACK

static inline uint16_t j2k_u16_clamp(int64_t v) {
  return v < 0 ? 0 : v > UINT16_MAX ? UINT16_MAX : (uint16_t)v;
}

/* 16-bit gray: component 0's values, unscaled, clamped to 0..65535, stored as
 * native-endian uint16 through memcpy (the destination has no alignment
 * guarantee). */
#define J2K_GROK_PACK16(T)                                                      \
  do {                                                                          \
    for (uint32_t row = 0; row < height; ++row) {                               \
      uint8_t *out = packed + (size_t)row * width * 2;                          \
      const T *r0 = (const T *)c0->data + (size_t)row * c0->stride;             \
      for (uint32_t col = 0; col < width; ++col) {                              \
        uint16_t v = j2k_u16_clamp((int64_t)r0[col]);                           \
        memcpy(out + 2 * (size_t)col, &v, sizeof v);                            \
      }                                                                         \
    }                                                                           \
  } while (0)

static int j2k_grok_pack_gray16(const grk_image *image, uint32_t width,
                                uint32_t height, uint8_t *packed) {
  const grk_image_comp *c0 = &image->comps[0];
  if (!j2k_grok_component_covers(c0, width, height)) {
    return 0;
  }
  switch (c0->data_type) {
  case GRK_INT_8:
    if (c0->sgnd) J2K_GROK_PACK16(int8_t); else J2K_GROK_PACK16(uint8_t);
    return 1;
  case GRK_INT_16:
    if (c0->sgnd) J2K_GROK_PACK16(int16_t); else J2K_GROK_PACK16(uint16_t);
    return 1;
  case GRK_INT_32:
    if (c0->sgnd) J2K_GROK_PACK16(int32_t); else J2K_GROK_PACK16(uint32_t);
    return 1;
  default:
    return 0;
  }
}
#undef J2K_GROK_PACK16
#endif

int j2k_grok_decode_u8(const uint8_t *bytes, size_t len, uint8_t reduce,
                       int has_region, uint32_t x0, uint32_t y0,
                       uint32_t x1, uint32_t y1, uint32_t channels,
                       uint32_t threads, uint32_t bytes_per_sample,
                       uint8_t *dest, size_t dest_len,
                       uint8_t **out_data, size_t *out_len,
                       uint32_t *out_width, uint32_t *out_height) {
  grk_object *codec = NULL;
  grk_image *image = NULL;
  grk_stream_params stream_params;
#if GRK_VERSION_MAJOR == 10
  grk_decompress_core_params params;
#else
  grk_decompress_parameters params;
#endif
  grk_header_info header_info;
  uint8_t *packed = NULL;

  if (!bytes || !out_data || !out_len || !out_width || !out_height ||
      (bytes_per_sample != 1 && bytes_per_sample != 2) ||
      (bytes_per_sample == 2 && channels != 1)) {
    return 0;
  }

  *out_data = NULL;
  *out_len = 0;
  *out_width = 0;
  *out_height = 0;

  memset(&stream_params, 0, sizeof(stream_params));
  memset(&params, 0, sizeof(params));
  memset(&header_info, 0, sizeof(header_info));

  stream_params.buf = (uint8_t *)bytes;
#if GRK_VERSION_MAJOR == 10
  stream_params.len = len;
#else
  stream_params.buf_len = len;
  stream_params.stream_len = len;
  stream_params.is_read_stream = true;
#endif

#if GRK_VERSION_MAJOR == 10
  if (bytes_per_sample != 1) {
    return 0; /* the Grok 10 path packs 8-bit samples only */
  }
  grk_decompress_set_default_params(&params);
  params.reduce = reduce;
  (void)threads; /* Grok 10 sizes decoding only through grk_initialize. */
#else
  params.core.reduce = reduce;
  params.force_rgb = channels == 3;
  params.upsample = channels == 3;
  /* 1 decodes on Grok's inline executor; more uses the global pool that
   * j2k_grok_initialize sized. */
  params.num_threads = threads ? threads : 1;
  if (has_region) {
    params.dw_x0 = x0;
    params.dw_y0 = y0;
    params.dw_x1 = x1;
    params.dw_y1 = y1;
  }
#endif

#if GRK_VERSION_MAJOR == 10
  header_info.decompressFormat = GRK_FMT_PXM;
  header_info.forceRGB = channels == 3;
  header_info.upsample = channels == 3;
#else
  header_info.color_space = channels == 3 ? GRK_CLRSPC_SRGB : GRK_CLRSPC_GRAY;
  header_info.decompress_fmt = GRK_FMT_PXM;
  header_info.force_rgb = channels == 3;
  header_info.upsample = channels == 3;
#endif

  codec = grk_decompress_init(&stream_params, &params);
  if (!codec) {
    return 0;
  }
  if (!grk_decompress_read_header(codec, &header_info)) {
    grk_object_unref(codec);
    return 0;
  }
#if GRK_VERSION_MAJOR == 10
  if (has_region &&
      !grk_decompress_set_window(codec, (float)x0, (float)y0, (float)x1,
                                 (float)y1)) {
    grk_object_unref(codec);
    return 0;
  }
#endif
  if (!grk_decompress(codec, NULL)) {
    grk_object_unref(codec);
    return 0;
  }

#if GRK_VERSION_MAJOR == 10
  image = grk_decompress_get_composited_image(codec);
#else
  image = grk_decompress_get_image(codec);
#endif
  if (!image || image->numcomps == 0 || !image->comps) {
    grk_object_unref(codec);
    return 0;
  }

#if GRK_VERSION_MAJOR == 10
  uint32_t width =
      image->decompressWidth ? image->decompressWidth : image->comps[0].w;
  uint32_t height =
      image->decompressHeight ? image->decompressHeight : image->comps[0].h;
#else
  uint32_t width = image->decompress_width ? image->decompress_width : image->comps[0].w;
  uint32_t height = image->decompress_height ? image->decompress_height : image->comps[0].h;
#endif
  size_t total = 0;
  size_t last_index = 0;
  int ok = 0;
  if (!j2k_grok_validate_component_count(image->numcomps, channels) ||
      !j2k_shim_output_len(width, height, channels, bytes_per_sample, &total) ||
      !j2k_grok_component_index(&image->comps[0], height - 1, width - 1,
                                &last_index)) {
    goto done;
  }
  if (channels == 3 && image->numcomps >= 3 &&
      (!j2k_grok_component_index(&image->comps[1], height - 1, width - 1,
                                 &last_index) ||
       !j2k_grok_component_index(&image->comps[2], height - 1, width - 1,
                                 &last_index))) {
    goto done;
  }
  if (dest && dest_len != total) {
    goto done;
  }
  packed = dest ? dest : (uint8_t *)malloc(total);
  if (!packed) {
    goto done;
  }

#if GRK_VERSION_MAJOR != 10
  if (bytes_per_sample == 2) {
    ok = j2k_grok_pack_gray16(image, width, height, packed);
    goto done;
  }
  if (j2k_grok_pack_fast(image, width, height, channels, packed)) {
    ok = 1;
    goto done;
  }
#endif

  for (uint32_t row = 0; row < height; ++row) {
    for (uint32_t col = 0; col < width; ++col) {
      size_t dst = ((size_t)row * width + col) * channels;
      grk_image_comp *c0 = &image->comps[0];
      size_t c0_index = 0;
      uint8_t v0 = 0;
      if (!j2k_grok_component_index(c0, row, col, &c0_index) ||
          !j2k_grok_component_sample_u8(c0, c0_index, &v0)) {
        goto done;
      }
      if (channels == 1) {
        packed[dst] = v0;
        continue;
      }
      if (image->numcomps == 1) {
        packed[dst] = v0;
        packed[dst + 1] = v0;
        packed[dst + 2] = v0;
        continue;
      }
      grk_image_comp *c1 = &image->comps[1];
      grk_image_comp *c2 = &image->comps[2];
      size_t c1_index = 0;
      size_t c2_index = 0;
      uint8_t v1 = 0;
      uint8_t v2 = 0;
      if (!j2k_grok_component_index(c1, row, col, &c1_index) ||
          !j2k_grok_component_index(c2, row, col, &c2_index) ||
          !j2k_grok_component_sample_u8(c1, c1_index, &v1) ||
          !j2k_grok_component_sample_u8(c2, c2_index, &v2)) {
        goto done;
      }
      packed[dst] = v0;
      packed[dst + 1] = v1;
      packed[dst + 2] = v2;
    }
  }
  ok = 1;

done:
  if (ok) {
    *out_data = packed;
    *out_len = total;
    *out_width = width;
    *out_height = height;
  } else if (!dest) {
    free(packed);
  }
  grk_object_unref(codec);
  return ok;
}

void j2k_grok_free(void *ptr) { free(ptr); }
