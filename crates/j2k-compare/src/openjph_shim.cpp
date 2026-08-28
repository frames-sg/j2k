// SPDX-License-Identifier: MIT OR Apache-2.0

#include <openjph/ojph_codestream.h>
#include <openjph/ojph_file.h>
#include <openjph/ojph_mem.h>
#include <openjph/ojph_params.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
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

std::uint8_t sample_u8(std::int32_t sample, std::uint32_t depth,
                       bool is_signed) {
  if (is_signed && depth != 0 && depth <= 31) {
    sample += std::int32_t{1} << (depth - 1);
  }
  if (depth > 8) {
    sample >>= depth - 8;
  }
  return static_cast<std::uint8_t>(std::min(std::max(sample, 0), 255));
}
} // namespace

extern "C" int j2k_openjph_decode_u8(
    const std::uint8_t *bytes, std::size_t len, std::uint8_t reduce,
    std::uint32_t channels, std::uint8_t **out_data, std::size_t *out_len,
    std::uint32_t *out_width, std::uint32_t *out_height) {
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
    ojph::mem_infile input;
    input.open(bytes, len);
    ojph::codestream codestream;
    codestream.set_planar(false);
    codestream.read_headers(&input);
    codestream.restrict_input_resolution(reduce, reduce);
    const ojph::param_siz siz = codestream.access_siz();
    if (siz.get_num_components() < channels) {
      return 0;
    }
    const std::uint32_t width = siz.get_recon_width(0);
    const std::uint32_t height = siz.get_recon_height(0);
    for (std::uint32_t component = 1; component < channels; ++component) {
      if (siz.get_recon_width(component) != width ||
          siz.get_recon_height(component) != height) {
        return 0;
      }
    }
    std::size_t required = 0;
    if (!checked_output_len(width, height, channels, required)) {
      return 0;
    }
    std::vector<std::uint8_t> output(required);
    codestream.create();
    for (std::uint32_t y = 0; y < height; ++y) {
      const std::size_t row_offset =
          static_cast<std::size_t>(y) * width * channels;
      for (std::uint32_t component = 0; component < channels; ++component) {
        ojph::ui32 pulled_component = 0;
        const ojph::line_buf *line = codestream.pull(pulled_component);
        if (line == nullptr || pulled_component != component || line->i32 == nullptr ||
            line->size < width) {
          return 0;
        }
        const std::uint32_t depth = siz.get_bit_depth(component);
        const bool signedness = siz.is_signed(component);
        for (std::uint32_t x = 0; x < width; ++x) {
          output[row_offset + static_cast<std::size_t>(x) * channels + component] =
              sample_u8(line->i32[x], depth, signedness);
        }
      }
    }
    codestream.close();
    input.close();

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

extern "C" void j2k_openjph_free(void *ptr) { std::free(ptr); }
