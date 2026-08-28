// SPDX-License-Identifier: MIT OR Apache-2.0

#include "decoder.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
constexpr std::size_t OUTPUT_CAP_BYTES = std::size_t{512} * 1024 * 1024;

bool checked_output_len(std::uint32_t width, std::uint32_t height,
                        std::uint32_t channels, std::size_t &out) {
  if ((channels != 1 && channels != 3) || width == 0 || height == 0) {
    return false;
  }
  if (static_cast<std::size_t>(width) >
      std::numeric_limits<std::size_t>::max() / channels) {
    return false;
  }
  const std::size_t row = static_cast<std::size_t>(width) * channels;
  if (row > std::numeric_limits<std::size_t>::max() / height) {
    return false;
  }
  out = row * height;
  return out <= OUTPUT_CAP_BYTES;
}

std::uint8_t sample_u8(std::int32_t sample, std::uint8_t depth,
                       bool is_signed) {
  if (is_signed && depth != 0 && depth <= 31) {
    sample += std::int32_t{1} << (depth - 1);
  }
  if (depth > 8) {
    sample >>= depth - 8;
  }
  return static_cast<std::uint8_t>(std::clamp(sample, 0, 255));
}
} // namespace

extern "C" int j2k_openhtj2k_decode_u8(
    const std::uint8_t *bytes, std::size_t len, std::uint8_t reduce,
    std::uint32_t threads, std::uint32_t channels, std::uint8_t **out_data,
    std::size_t *out_len, std::uint32_t *out_width,
    std::uint32_t *out_height) {
  if (bytes == nullptr || len == 0 || out_data == nullptr || out_len == nullptr ||
      out_width == nullptr || out_height == nullptr ||
      (channels != 1 && channels != 3)) {
    return 0;
  }
  *out_data = nullptr;
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

    std::vector<std::uint32_t> widths;
    std::vector<std::uint32_t> heights;
    std::vector<std::uint8_t> depths;
    std::vector<bool> signedness;
    std::vector<std::uint8_t> output;
    bool initialized = false;
    std::uint32_t width = 0;
    std::uint32_t height = 0;

    decoder.invoke_line_based_stream(
        [&](std::uint32_t y, std::int32_t *const *rows, std::uint16_t nc) {
          if (rows == nullptr) {
            throw std::runtime_error("OpenHTJ2K returned null component rows");
          }
          if (!initialized) {
            if (nc < channels || widths.size() < channels ||
                heights.size() < channels || depths.size() < channels ||
                signedness.size() < channels) {
              throw std::runtime_error("OpenHTJ2K component metadata mismatch");
            }
            width = widths[0];
            height = heights[0];
            for (std::uint32_t c = 1; c < channels; ++c) {
              if (widths[c] != width || heights[c] != height) {
                throw std::runtime_error("subsampled OpenHTJ2K output unsupported");
              }
            }
            std::size_t required = 0;
            if (!checked_output_len(width, height, channels, required)) {
              throw std::runtime_error("OpenHTJ2K output size invalid");
            }
            output.resize(required);
            initialized = true;
          }
          if (y >= height) {
            throw std::runtime_error("OpenHTJ2K row exceeds output height");
          }
          const std::size_t row_offset =
              static_cast<std::size_t>(y) * width * channels;
          for (std::uint32_t c = 0; c < channels; ++c) {
            if (rows[c] == nullptr) {
              throw std::runtime_error("OpenHTJ2K returned a null component row");
            }
          }
          for (std::uint32_t x = 0; x < width; ++x) {
            for (std::uint32_t c = 0; c < channels; ++c) {
              output[row_offset + static_cast<std::size_t>(x) * channels + c] =
                  sample_u8(rows[c][x], depths[c], signedness[c]);
            }
          }
        },
        widths, heights, depths, signedness);

    if (!initialized || output.empty()) {
      return 0;
    }
    auto *allocation = static_cast<std::uint8_t *>(std::malloc(output.size()));
    if (allocation == nullptr) {
      return 0;
    }
    std::memcpy(allocation, output.data(), output.size());
    *out_data = allocation;
    *out_len = output.size();
    *out_width = width;
    *out_height = height;
    return 1;
  } catch (...) {
    return 0;
  }
}

extern "C" void j2k_openhtj2k_free(void *ptr) { std::free(ptr); }
