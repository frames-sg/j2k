// SPDX-License-Identifier: MIT OR Apache-2.0

#include <openjph/ojph_codestream.h>
#include <openjph/ojph_file.h>
#include <openjph/ojph_mem.h>
#include <openjph/ojph_params.h>

#include "j2k_shim_common.h"

#include <cstddef>
#include <cstdint>

// Decodes into `dest`, whose length must equal the decoded image's, so a
// benchmark caller's buffer receives the pixels directly, as the native
// decoder's does.
extern "C" int j2k_openjph_decode_u8(
    const std::uint8_t *bytes, std::size_t len, std::uint8_t reduce,
    std::uint32_t channels, std::uint32_t bytes_per_sample, std::uint8_t *dest,
    std::size_t dest_len, std::size_t *out_len, std::uint32_t *out_width,
    std::uint32_t *out_height) {
  if (bytes == nullptr || len == 0 || dest == nullptr || out_len == nullptr ||
      out_width == nullptr || out_height == nullptr ||
      (channels != 1 && channels != 3) ||
      (bytes_per_sample != 1 && bytes_per_sample != 2) ||
      (bytes_per_sample == 2 && channels != 1)) {
    return 0;
  }
  *out_len = 0;
  *out_width = 0;
  *out_height = 0;

  try {
    ojph::mem_infile input;
    input.open(bytes, len);
    ojph::codestream codestream;
    codestream.read_headers(&input);
    // read_headers chooses a default from the color-transform flag.
    // Override it afterwards, including RGB streams without an MCT.
    codestream.set_planar(false);
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
    if (!j2k_shim_output_len(width, height, channels, bytes_per_sample,
                             &required) ||
        dest_len != required) {
      return 0;
    }
    j2k_shim_u8_scale scales[3] = {};
    for (std::uint32_t component = 0; component < channels; ++component) {
      scales[component] = j2k_shim_u8_scale_for(siz.get_bit_depth(component),
                                                 siz.is_signed(component));
    }
    codestream.create();
    // OpenJPH hands over one component's line at a time; RGB rows pack
    // once all three have arrived.
    const std::int32_t *lines[3] = {};
    const std::size_t row_bytes =
        static_cast<std::size_t>(width) * channels * bytes_per_sample;
    for (std::uint32_t y = 0; y < height; ++y) {
      for (std::uint32_t component = 0; component < channels; ++component) {
        ojph::ui32 pulled_component = 0;
        const ojph::line_buf *line = codestream.pull(pulled_component);
        if (line == nullptr || pulled_component != component || line->i32 == nullptr ||
            line->size < width) {
          return 0;
        }
        lines[component] = line->i32;
      }
      std::uint8_t *const row = dest + static_cast<std::size_t>(y) * row_bytes;
      if (bytes_per_sample == 2) {
        j2k_shim_pack_gray16(row, lines[0], width);
      } else if (channels == 1) {
        j2k_shim_pack_gray8(row, lines[0], width, scales[0]);
      } else {
        j2k_shim_pack_rgb8(row, lines[0], lines[1], lines[2], width, scales);
      }
    }
    codestream.close();
    input.close();

    *out_len = required;
    *out_width = width;
    *out_height = height;
    return 1;
  } catch (...) {
    return 0;
  }
}
