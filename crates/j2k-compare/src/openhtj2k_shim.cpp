// SPDX-License-Identifier: MIT OR Apache-2.0

#include "decoder.hpp"
#include "j2k_shim_common.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

// Decodes at resolution reduction `reduce`, optionally only the region
// [x0, x1) x [y0, y1) given in full-resolution coordinates, using OpenHTJ2K's
// own viewport API (set_col_range / set_row_range, as its region_decode_bench
// does). The output covers the region scaled by 2^reduce (floor of the start,
// ceiling of the end), clipped to the image. The pixels are written to
// `dest`, whose length must equal the output's.
extern "C" int j2k_openhtj2k_decode_u8(
    const std::uint8_t *bytes, std::size_t len, std::uint8_t reduce,
    std::uint32_t threads, std::uint32_t channels, std::uint32_t bytes_per_sample,
    int has_region,
    std::uint32_t x0, std::uint32_t y0, std::uint32_t x1, std::uint32_t y1,
    std::uint8_t *dest, std::size_t dest_len, std::size_t *out_len,
    std::uint32_t *out_width, std::uint32_t *out_height) {
  if (bytes == nullptr || len == 0 || dest == nullptr || out_len == nullptr ||
      out_width == nullptr || out_height == nullptr ||
      (channels != 1 && channels != 3) || reduce > 31 ||
      (bytes_per_sample != 1 && bytes_per_sample != 2) ||
      (bytes_per_sample == 2 && channels != 1) ||
      (has_region && (x1 <= x0 || y1 <= y0))) {
    return 0;
  }
  *out_len = 0;
  *out_width = 0;
  *out_height = 0;

  try {
    open_htj2k::openhtj2k_decoder decoder(bytes, len, reduce,
                                           std::max(threads, 1u));
    decoder.parse();
    const std::uint16_t components = decoder.get_num_component();
    if ((channels == 1 && components == 0) ||
        (channels == 3 && components < 3)) {
      return 0;
    }
    const std::uint32_t denom = std::uint32_t{1} << reduce;
    const std::uint32_t rx0 = has_region ? x0 / denom : 0;
    const std::uint32_t ry0 = has_region ? y0 / denom : 0;
    const std::uint32_t rx1 =
        has_region ? x1 / denom + (x1 % denom != 0) : UINT32_MAX;
    const std::uint32_t ry1 =
        has_region ? y1 / denom + (y1 % denom != 0) : UINT32_MAX;
    if (has_region) {
      decoder.set_col_range(rx0, rx1);
      decoder.set_row_range(ry0, ry1);
    }

    std::vector<std::uint32_t> widths;
    std::vector<std::uint32_t> heights;
    std::vector<std::uint8_t> depths;
    std::vector<bool> signedness;
    std::uint8_t *output = nullptr;
    j2k_shim_u8_scale scales[3] = {};
    std::uint32_t ox0 = 0, oy0 = 0, out_w = 0, out_h = 0, rows_written = 0;
    std::size_t required = 0;

    decoder.invoke_line_based_stream(
        [&](std::uint32_t y, std::int32_t *const *rows, std::uint16_t nc) {
          if (rows == nullptr) {
            throw std::runtime_error("OpenHTJ2K returned null component rows");
          }
          if (output == nullptr) {
            if (nc < channels || widths.size() < channels ||
                heights.size() < channels || depths.size() < channels ||
                signedness.size() < channels) {
              throw std::runtime_error("OpenHTJ2K component metadata mismatch");
            }
            const std::uint32_t width = widths[0];
            const std::uint32_t height = heights[0];
            for (std::uint32_t c = 1; c < channels; ++c) {
              if (widths[c] != width || heights[c] != height) {
                throw std::runtime_error("subsampled OpenHTJ2K output unsupported");
              }
            }
            ox0 = std::min(rx0, width);
            oy0 = std::min(ry0, height);
            out_w = std::min(rx1, width) - ox0;
            out_h = std::min(ry1, height) - oy0;
            if (!j2k_shim_output_len(out_w, out_h, channels, bytes_per_sample,
                                     &required)) {
              throw std::runtime_error("OpenHTJ2K output size invalid");
            }
            if (dest_len != required) {
              throw std::runtime_error("destination length mismatch");
            }
            for (std::uint32_t c = 0; c < channels; ++c) {
              scales[c] = j2k_shim_u8_scale_for(depths[c], signedness[c]);
            }
            output = dest;
          }
          if (y < oy0 || y >= oy0 + out_h) {
            return;
          }
          for (std::uint32_t c = 0; c < channels; ++c) {
            if (rows[c] == nullptr) {
              throw std::runtime_error("OpenHTJ2K returned a null component row");
            }
          }
          std::uint8_t *const row = output + static_cast<std::size_t>(y - oy0) *
                                                 out_w * channels * bytes_per_sample;
          if (bytes_per_sample == 2) {
            j2k_shim_pack_gray16(row, rows[0] + ox0, out_w);
          } else if (channels == 1) {
            j2k_shim_pack_gray8(row, rows[0] + ox0, out_w, scales[0]);
          } else {
            j2k_shim_pack_rgb8(row, rows[0] + ox0, rows[1] + ox0, rows[2] + ox0, out_w,
                               scales);
          }
          ++rows_written;
        },
        widths, heights, depths, signedness);

    if (output == nullptr || rows_written != out_h) {
      return 0;
    }
    *out_len = required;
    *out_width = out_w;
    *out_height = out_h;
    return 1;
  } catch (...) {
    return 0;
  }
}

