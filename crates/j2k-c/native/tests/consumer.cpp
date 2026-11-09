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
static_assert(!std::is_copy_constructible_v<j2k::cancellation_token>);
static_assert(std::is_nothrow_move_constructible_v<j2k::cancellation_token>);

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

        target.prepare_operations();
        j2k_operation_options operation = J2K_OPERATION_OPTIONS_INIT;
        operation.max_working_memory_bytes = UINT64_C(8) << 20;
        const auto plan = target.plan_region(operation);
        std::vector<uint8_t> exact(static_cast<size_t>(plan.output.buffer_bytes));
        const auto actual = target.decode_region_into(exact.data(), exact.size(), 32, operation);
        if (actual.precision != 8 || exact != pixels)
            throw std::runtime_error("exact native operation mismatch");
        j2k::cancellation_token token;
        j2k::cancellation_token moved_token(std::move(token));
        operation.cancellation = moved_token.native_handle();
        moved_token.cancel();
        rejected = false;
        try { target.decode_region_into(exact.data(), exact.size(), 32, operation); }
        catch (const j2k::error &error) { rejected = error.status() == J2K_CANCELLED; }
        if (!rejected) throw std::runtime_error("cancelled operation was not rejected");
        operation.cancellation = nullptr;
        target.set_operation_cache_limit(0);

        // Encode and decode again.
        j2k_encode_options options = J2K_ENCODE_OPTIONS_INIT;
        options.width = 32; options.height = 24; options.components = 1; options.bit_depth = 8;
        const auto codestream = j2k::encode_lossless(pixels.data(), pixels.size(), options);
        j2k::decoder again(codestream.data(), codestream.size());
        std::vector<uint8_t> decoded(pixels.size());
        again.decode_into(decoded.data(), decoded.size(), 32);
        if (decoded != pixels) throw std::runtime_error("encode round trip mismatch");
        if (!again.icc_profile().empty() || again.color_info().specification != J2K_COLOR_NONE)
            throw std::runtime_error("a raw codestream reported a colour specification");

        // Lossy to a rate target, as a JP2 file with an ICC profile.
        const std::vector<uint8_t> profile{'t', 'e', 's', 't', '-', 'i', 'c', 'c'};
        const double rate[] = {1.5};
        j2k_encode_params params = J2K_ENCODE_PARAMS_INIT;
        params.mode = J2K_MODE_LOSSY;
        params.rate = J2K_RATE_BITS_PER_PIXEL;
        params.target_count = 1;
        params.targets = rate;
        params.icc_profile = profile.data();
        params.icc_profile_len = profile.size();
        options.flags = J2K_ENCODE_FILE;
        j2k_encode_report report = J2K_ENCODE_REPORT_INIT;
        const auto tagged_file = j2k::encode(pixels.data(), pixels.size(), options, params, &report);
        j2k::decoder tagged(tagged_file.data(), tagged_file.size());
        if (tagged.icc_profile() != profile || report.codestream_bytes == 0)
            throw std::runtime_error("ICC-tagged lossy encode mismatch");
        options.flags = 0;
        if (j2k::encode(pixels.data(), pixels.size(), options) != codestream)
            throw std::runtime_error("default parameters differ from lossless encode");
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
