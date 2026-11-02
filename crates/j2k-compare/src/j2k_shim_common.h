/* SPDX-License-Identifier: MIT OR Apache-2.0 */

/* Output checks and sample packing shared by the C and C++ comparator shims,
 * so every external decoder's samples reach the caller's buffer through the
 * same loops. Each loop keeps its bounds and pointers in locals and restricts
 * its pointers: stores through `uint8_t *` may alias anything, so a loop that
 * reads its bounds or source through memory reloads them for every sample
 * and does not vectorize. */

#ifndef J2K_SHIM_COMMON_H
#define J2K_SHIM_COMMON_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
#define J2K_SHIM_RESTRICT __restrict
#else
#define J2K_SHIM_RESTRICT restrict
#endif

/* The largest output a shim writes, as j2k-compare's
 * `MAX_EXTERNAL_OUTPUT_BYTES`. */
#define J2K_SHIM_OUTPUT_CAP_BYTES ((size_t)4 * 1024 * 1024 * 1024)

/* Sets `*out` to the bytes of a `width` x `height` image of `channels` (1 or
 * 3) samples of `bytes_per_sample` (1 or 2) bytes. Returns 0, leaving `*out`
 * alone, for a zero dimension, another shape, or more than the cap. */
static inline int j2k_shim_output_len(uint32_t width, uint32_t height,
                                      uint32_t channels,
                                      uint32_t bytes_per_sample, size_t *out) {
  if ((channels != 1 && channels != 3) ||
      (bytes_per_sample != 1 && bytes_per_sample != 2) || width == 0 ||
      height == 0) {
    return 0;
  }
  const size_t row = (size_t)width * channels * bytes_per_sample;
  if (row > J2K_SHIM_OUTPUT_CAP_BYTES / height) {
    return 0;
  }
  *out = row * height;
  return 1;
}

/* How one component's samples become 8-bit: `(sample + offset) >> shift`,
 * clamped to 0..255. A signed component's offset centres it, and above 8
 * bits the shift keeps its top 8. */
typedef struct {
  int32_t offset;
  uint32_t shift;
} j2k_shim_u8_scale;

static inline j2k_shim_u8_scale j2k_shim_u8_scale_for(uint32_t depth,
                                                      int is_signed) {
  j2k_shim_u8_scale scale;
  scale.offset =
      is_signed && depth != 0 && depth <= 31 ? (int32_t)1 << (depth - 1) : 0;
  scale.shift = depth > 8 ? (depth - 8 < 31 ? depth - 8 : 31) : 0;
  return scale;
}

static inline int j2k_shim_u8_is_plain(j2k_shim_u8_scale scale) {
  return scale.offset == 0 && scale.shift == 0;
}

/* The offset wraps instead of overflowing; in range it is the plain sum. */
static inline uint8_t j2k_shim_u8(int32_t sample, int32_t offset,
                                  uint32_t shift) {
  const int32_t value = (int32_t)((uint32_t)sample + (uint32_t)offset) >> shift;
  return (uint8_t)(value < 0 ? 0 : value > 255 ? 255 : value);
}

static inline uint8_t j2k_shim_clamp_u8(int32_t sample) {
  return (uint8_t)(sample < 0 ? 0 : sample > 255 ? 255 : sample);
}

/* One row of one component as 8-bit gray. */
static inline void j2k_shim_pack_gray8(uint8_t *J2K_SHIM_RESTRICT dst,
                                       const int32_t *J2K_SHIM_RESTRICT src,
                                       uint32_t width,
                                       j2k_shim_u8_scale scale) {
  if (j2k_shim_u8_is_plain(scale)) {
    for (uint32_t x = 0; x < width; ++x) {
      dst[x] = j2k_shim_clamp_u8(src[x]);
    }
    return;
  }
  const int32_t offset = scale.offset;
  const uint32_t shift = scale.shift;
  for (uint32_t x = 0; x < width; ++x) {
    dst[x] = j2k_shim_u8(src[x], offset, shift);
  }
}

/* One row of three components as interleaved 8-bit RGB. */
static inline void j2k_shim_pack_rgb8(uint8_t *J2K_SHIM_RESTRICT dst,
                                      const int32_t *J2K_SHIM_RESTRICT red,
                                      const int32_t *J2K_SHIM_RESTRICT green,
                                      const int32_t *J2K_SHIM_RESTRICT blue,
                                      uint32_t width,
                                      const j2k_shim_u8_scale scales[3]) {
  const j2k_shim_u8_scale r = scales[0];
  const j2k_shim_u8_scale g = scales[1];
  const j2k_shim_u8_scale b = scales[2];
  if (j2k_shim_u8_is_plain(r) && j2k_shim_u8_is_plain(g) &&
      j2k_shim_u8_is_plain(b)) {
    for (uint32_t x = 0; x < width; ++x) {
      dst[3 * (size_t)x] = j2k_shim_clamp_u8(red[x]);
      dst[3 * (size_t)x + 1] = j2k_shim_clamp_u8(green[x]);
      dst[3 * (size_t)x + 2] = j2k_shim_clamp_u8(blue[x]);
    }
    return;
  }
  for (uint32_t x = 0; x < width; ++x) {
    dst[3 * (size_t)x] = j2k_shim_u8(red[x], r.offset, r.shift);
    dst[3 * (size_t)x + 1] = j2k_shim_u8(green[x], g.offset, g.shift);
    dst[3 * (size_t)x + 2] = j2k_shim_u8(blue[x], b.offset, b.shift);
  }
}

/* One row of one component as native-endian 16-bit samples holding each
 * value unscaled, clamped to 0..65535. The destination has no alignment
 * guarantee, so each sample is stored through memcpy. */
static inline void j2k_shim_pack_gray16(uint8_t *J2K_SHIM_RESTRICT dst,
                                        const int32_t *J2K_SHIM_RESTRICT src,
                                        uint32_t width) {
  for (uint32_t x = 0; x < width; ++x) {
    const int32_t sample = src[x];
    const uint16_t value =
        (uint16_t)(sample < 0 ? 0 : sample > 65535 ? 65535 : sample);
    memcpy(dst + 2 * (size_t)x, &value, sizeof value);
  }
}

#endif
