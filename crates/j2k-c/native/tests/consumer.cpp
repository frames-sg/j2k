// SPDX-License-Identifier: MIT OR Apache-2.0
// Exercise C++ ownership and exception behavior against the installed binary.
#include <j2k.hpp>
#include <fstream>
#include <iostream>
#include <iterator>
#include <type_traits>
#include <vector>

static_assert(!std::is_copy_constructible_v<j2k::decoder>);
static_assert(std::is_nothrow_move_constructible_v<j2k::decoder>);
static_assert(std::is_nothrow_move_assignable_v<j2k::decoder>);

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    try {
        std::ifstream file(argv[1], std::ios::binary);
        if (!file) throw std::runtime_error("cannot open fixture");
        std::vector<uint8_t> encoded{std::istreambuf_iterator<char>(file), {}};
        j2k::decoder first(encoded.data(), encoded.size());
        j2k::decoder moved(std::move(first));
        j2k::decoder target(encoded.data(), encoded.size());
        target = std::move(moved); // Releases target's previous handle.
        const auto source = target.info();
        const auto component = target.component(0);
        if (source.width != 32 || source.height != 24 || component.precision != 8)
            throw std::runtime_error("source metadata mismatch");
        const auto info = target.output_info();
        std::vector<uint8_t> pixels(static_cast<size_t>(info.buffer_bytes));
        target.decode_into(pixels.data(), pixels.size(), static_cast<size_t>(info.row_bytes));
        if (pixels.size() != 768) throw std::runtime_error("output size mismatch");
        for (size_t i = 0; i < pixels.size(); ++i)
            if (pixels[i] != (i * 13 + 7) % 256) throw std::runtime_error("pixel mismatch");
        bool rejected = false;
        try { j2k::decoder empty(encoded.data(), 0); }
        catch (const j2k::error &error) {
            rejected = error.status() == J2K_INVALID_INPUT && error.what()[0] != '\0';
        }
        if (!rejected) throw std::runtime_error("empty input was not rejected with a diagnostic");
        rejected = false;
        try { target.decode_into(pixels.data(), 1, static_cast<size_t>(info.row_bytes)); }
        catch (const j2k::error &error) { rejected = error.status() == J2K_BUFFER_TOO_SMALL; }
        if (!rejected) throw std::runtime_error("short output was not rejected");
        rejected = false;
        try { first.info(); }
        catch (const j2k::error &error) { rejected = error.status() == J2K_INVALID_ARGUMENT; }
        if (!rejected) throw std::runtime_error("moved-from handle was not rejected");

        // Rows: the full image in order, a stop, and an exception that crosses no C frame.
        std::vector<uint8_t> streamed;
        const auto layout = target.decode_rows([&](uint32_t, const uint8_t *row, size_t bytes) {
            streamed.insert(streamed.end(), row, row + bytes);
            return true;
        }, 4);
        if (!layout || layout->row_bytes != 32 || streamed != pixels)
            throw std::runtime_error("streamed rows differ from decode_into");
        uint32_t seen = 0;
        if (target.decode_rows([&](uint32_t, const uint8_t *, size_t) { return ++seen < 2; }))
            throw std::runtime_error("a stopped decode reported completion");
        if (seen != 2) throw std::runtime_error("rows continued after the stop");
        bool rethrown = false;
        try {
            target.decode_rows([](uint32_t y, const uint8_t *, size_t) -> bool {
                if (y == 1) throw std::logic_error("from the row callback");
                return true;
            });
        } catch (const std::logic_error &) { rethrown = true; }
        if (!rethrown) throw std::runtime_error("a row callback exception was not rethrown");

        // Encode and decode again.
        j2k_encode_options options = J2K_ENCODE_OPTIONS_INIT;
        options.width = 32; options.height = 24; options.components = 1; options.bit_depth = 8;
        const auto codestream = j2k::encode_lossless(pixels.data(), pixels.size(), options);
        j2k::decoder again(codestream.data(), codestream.size());
        std::vector<uint8_t> decoded(pixels.size());
        again.decode_into(decoded.data(), decoded.size(), 32);
        if (decoded != pixels) throw std::runtime_error("encode round trip mismatch");
        rejected = false;
        options.flags = 64;
        try { j2k::encode_lossless(pixels.data(), pixels.size(), options); }
        catch (const j2k::error &error) { rejected = error.status() == J2K_INVALID_ARGUMENT; }
        if (!rejected) throw std::runtime_error("an unknown encode flag was not rejected");
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
