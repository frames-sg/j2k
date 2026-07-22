// SPDX-License-Identifier: MIT OR Apache-2.0

#include <cassert>
#include <cstdint>
#include <cstring>
#include <cstdlib>

#include "ojph_codestream.h"
#include "ojph_defs.h"
#include "ojph_file.h"
#include "ojph_mem.h"
#include "ojph_params.h"

namespace {

constexpr std::uint32_t kWidth = 19;
constexpr std::uint32_t kHeight = 13;

std::int32_t sample(std::uint32_t x, std::uint32_t y,
                    std::uint32_t component, std::uint32_t precision) {
  const std::uint32_t modulus = 1U << precision;
  const std::uint32_t code =
      (x * 37U + y * 73U + component * 109U + x * y * 3U +
       component * y * 11U) %
      modulus;
  return static_cast<std::int32_t>(code) -
         static_cast<std::int32_t>(modulus / 2U);
}

}  // namespace

int main(int argc, char** argv) {
  assert(argc >= 3 && argc <= 5);
  const char* output_path = argv[1];
  const std::uint32_t precision =
      static_cast<std::uint32_t>(std::strtoul(argv[2], nullptr, 10));
  assert(precision == 8 || precision == 12 || precision == 16);
  bool single_tile = false;
  bool reversible = true;
  for (int index = 3; index < argc; ++index) {
    if (std::strcmp(argv[index], "single-tile") == 0) {
      single_tile = true;
    } else if (std::strcmp(argv[index], "irreversible") == 0) {
      reversible = false;
    } else {
      assert(false && "unsupported fixture mode");
    }
  }

  ojph::codestream codestream;
  ojph::param_siz siz = codestream.access_siz();
  siz.set_image_extent(ojph::point(kWidth, kHeight));
  siz.set_num_components(3);
  for (std::uint32_t component = 0; component < 3; ++component) {
    siz.set_component(component, ojph::point(1, 1), precision, true);
  }
  siz.set_tile_size(single_tile ? ojph::size(kWidth, kHeight)
                                : ojph::size(11, 7));

  ojph::param_cod cod = codestream.access_cod();
  cod.set_num_decomposition(2);
  cod.set_block_dims(8, 8);
  cod.set_color_transform(false);
  cod.set_reversible(reversible);
  if (!reversible) {
    const float qstep = 0.003F / static_cast<float>(1U << (precision - 8U));
    codestream.access_qcd().set_irrev_quant(qstep);
  }
  codestream.set_planar(true);

  ojph::j2c_outfile output;
  output.open(output_path);
  codestream.write_headers(&output);

  ojph::ui32 next_component = 0;
  ojph::line_buf* line = codestream.exchange(nullptr, next_component);
  for (std::uint32_t component = 0; component < 3; ++component) {
    for (std::uint32_t y = 0; y < kHeight; ++y) {
      assert(next_component == component);
      assert(line != nullptr && line->i32 != nullptr && line->size >= kWidth);
      for (std::uint32_t x = 0; x < kWidth; ++x) {
        line->i32[x] = sample(x, y, component, reversible ? precision : 8U);
      }
      line = codestream.exchange(line, next_component);
    }
  }

  codestream.flush();
  codestream.close();
  output.close();
  return 0;
}
