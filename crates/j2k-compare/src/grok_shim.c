#include "grok.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define GROK_OUTPUT_CAP_BYTES ((size_t)512 * 1024 * 1024)

static void j2k_grok_init_once(void) {
  static int initialized = 0;
  if (!initialized) {
    grk_initialize(NULL, 1, NULL);
    initialized = 1;
  }
}

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

static int j2k_grok_checked_output_len(uint32_t width, uint32_t height,
                                       uint32_t channels, size_t *out) {
  size_t pixels = 0;
  size_t total = 0;
  if (channels != 1 && channels != 3) {
    return 0;
  }
  if (width == 0 || height == 0) {
    return 0;
  }
  if (!j2k_grok_checked_mul_size((size_t)width, (size_t)height, &pixels)) {
    return 0;
  }
  if (!j2k_grok_checked_mul_size(pixels, (size_t)channels, &total)) {
    return 0;
  }
  if (total > GROK_OUTPUT_CAP_BYTES) {
    return 0;
  }
  *out = total;
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

static int32_t j2k_grok_component_sample(const grk_image_comp *component,
                                              size_t index) {
  if (!component || !component->data) {
    return 0;
  }
  switch (component->data_type) {
  case GRK_INT_8:
    if (!component->sgnd) {
      return ((const uint8_t *)component->data)[index];
    }
    return ((const int8_t *)component->data)[index];
  case GRK_INT_16:
    if (!component->sgnd) {
      return ((const uint16_t *)component->data)[index];
    }
    return ((const int16_t *)component->data)[index];
  case GRK_INT_32:
    if (!component->sgnd) {
      return (int32_t)((const uint32_t *)component->data)[index];
    }
    return ((const int32_t *)component->data)[index];
  default:
    return ((const int32_t *)component->data)[index];
  }
}

int j2k_grok_decode_u8(const uint8_t *bytes, size_t len, uint32_t reduce,
                              int has_region, uint32_t x0, uint32_t y0,
                              uint32_t x1, uint32_t y1, uint32_t channels,
                              uint8_t **out_data, size_t *out_len,
                              uint32_t *out_width, uint32_t *out_height) {
  grk_object *codec = NULL;
  grk_image *image = NULL;
  grk_stream_params stream_params;
  grk_decompress_parameters params;
  grk_header_info header_info;
  uint8_t *packed = NULL;

  if (!bytes || !out_data || !out_len || !out_width || !out_height) {
    return 0;
  }

  *out_data = NULL;
  *out_len = 0;
  *out_width = 0;
  *out_height = 0;

  j2k_grok_init_once();
  memset(&stream_params, 0, sizeof(stream_params));
  memset(&params, 0, sizeof(params));
  memset(&header_info, 0, sizeof(header_info));

  stream_params.buf = (uint8_t *)bytes;
  stream_params.buf_len = len;
  stream_params.stream_len = len;
  stream_params.is_read_stream = true;

  params.core.reduce = (uint8_t)reduce;
  params.force_rgb = channels == 3;
  params.upsample = channels == 3;
  params.num_threads = 1;
  if (has_region) {
    params.dw_x0 = x0;
    params.dw_y0 = y0;
    params.dw_x1 = x1;
    params.dw_y1 = y1;
  }

  header_info.color_space = channels == 3 ? GRK_CLRSPC_SRGB : GRK_CLRSPC_GRAY;
  header_info.decompress_fmt = GRK_FMT_PXM;
  header_info.force_rgb = channels == 3;
  header_info.upsample = channels == 3;

  codec = grk_decompress_init(&stream_params, &params);
  if (!codec) {
    return 0;
  }
  if (!grk_decompress_read_header(codec, &header_info)) {
    grk_object_unref(codec);
    return 0;
  }
  if (!grk_decompress(codec, NULL)) {
    grk_object_unref(codec);
    return 0;
  }

  image = grk_decompress_get_image(codec);
  if (!image || image->numcomps == 0 || !image->comps) {
    grk_object_unref(codec);
    return 0;
  }

  uint32_t width = image->decompress_width ? image->decompress_width : image->comps[0].w;
  uint32_t height = image->decompress_height ? image->decompress_height : image->comps[0].h;
  size_t total = 0;
  size_t last_index = 0;
  if (!j2k_grok_validate_component_count(image->numcomps, channels) ||
      !j2k_grok_checked_output_len(width, height, channels, &total)) {
    grk_object_unref(codec);
    return 0;
  }
  if (!j2k_grok_component_index(&image->comps[0], height - 1, width - 1,
                                &last_index)) {
    grk_object_unref(codec);
    return 0;
  }
  if (channels == 3 && image->numcomps >= 3 &&
      (!j2k_grok_component_index(&image->comps[1], height - 1, width - 1,
                                 &last_index) ||
       !j2k_grok_component_index(&image->comps[2], height - 1, width - 1,
                                 &last_index))) {
    grk_object_unref(codec);
    return 0;
  }
  packed = (uint8_t *)malloc(total);
  if (!packed) {
    grk_object_unref(codec);
    return 0;
  }

  for (uint32_t row = 0; row < height; ++row) {
    for (uint32_t col = 0; col < width; ++col) {
      size_t dst = ((size_t)row * width + col) * channels;
      grk_image_comp *c0 = &image->comps[0];
      size_t c0_index = 0;
      if (!j2k_grok_component_index(c0, row, col, &c0_index)) {
        free(packed);
        grk_object_unref(codec);
        return 0;
      }
      int32_t v0 = j2k_grok_component_sample(c0, c0_index);
      if (channels == 1) {
        packed[dst] = j2k_clamp_u8(v0);
        continue;
      }
      if (image->numcomps == 1) {
        uint8_t gray = j2k_clamp_u8(v0);
        packed[dst] = gray;
        packed[dst + 1] = gray;
        packed[dst + 2] = gray;
        continue;
      }
      grk_image_comp *c1 = &image->comps[1];
      grk_image_comp *c2 = &image->comps[2];
      size_t c1_index = 0;
      size_t c2_index = 0;
      if (!j2k_grok_component_index(c1, row, col, &c1_index) ||
          !j2k_grok_component_index(c2, row, col, &c2_index)) {
        free(packed);
        grk_object_unref(codec);
        return 0;
      }
      int32_t v1 = j2k_grok_component_sample(c1, c1_index);
      int32_t v2 = j2k_grok_component_sample(c2, c2_index);
      packed[dst] = j2k_clamp_u8(v0);
      packed[dst + 1] = j2k_clamp_u8(v1);
      packed[dst + 2] = j2k_clamp_u8(v2);
    }
  }

  *out_data = packed;
  *out_len = total;
  *out_width = width;
  *out_height = height;
  grk_object_unref(codec);
  return 1;
}

void j2k_grok_free(void *ptr) { free(ptr); }
