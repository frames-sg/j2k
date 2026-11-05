// SPDX-License-Identifier: MIT OR Apache-2.0
// Decode using the installed C++17 wrapper. Pixel storage stays caller-owned.
#include <j2k.hpp>
#include <fstream>
#include <iostream>
#include <vector>

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " IMAGE.jp2\n";
        return 2;
    }
    try {
        std::ifstream file(argv[1], std::ios::binary | std::ios::ate);
        if (!file) throw std::runtime_error("cannot open input");
        const auto length = file.tellg();
        if (length <= 0 || length > 512LL * 1024 * 1024)
            throw std::runtime_error("input must be nonempty and no larger than 512 MiB");
        std::vector<uint8_t> encoded(static_cast<size_t>(length));
        file.seekg(0);
        if (!file.read(reinterpret_cast<char *>(encoded.data()), length))
            throw std::runtime_error("cannot read complete input");

        j2k::decoder decoder(encoded.data(), encoded.size());
        const auto info = decoder.output_info();
        std::vector<uint8_t> pixels(static_cast<size_t>(info.buffer_bytes));
        decoder.decode_into(pixels.data(), pixels.size(), static_cast<size_t>(info.row_bytes));
        std::cout << info.width << 'x' << info.height << ", " << info.channels
                  << " channels, " << info.precision << " significant bits, "
                  << pixels.size() << " bytes\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
