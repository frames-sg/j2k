/* SPDX-License-Identifier: MIT OR Apache-2.0 */
/* Decode a JP2/JPH file or raw codestream using only the installed package. */
#include <j2k.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s IMAGE.jp2\n", argv[0]);
        return 2;
    }
    FILE *file = fopen(argv[1], "rb");
    if (!file) { perror("open input"); return 1; }
    if (fseek(file, 0, SEEK_END) != 0) { fclose(file); return 1; }
    const long length = ftell(file);
    if (length <= 0 || length > 512L * 1024 * 1024 || fseek(file, 0, SEEK_SET) != 0) {
        fprintf(stderr, "input must be a nonempty file no larger than 512 MiB\n");
        fclose(file);
        return 1;
    }
    uint8_t *encoded = (uint8_t *)malloc((size_t)length);
    if (!encoded) { fclose(file); return 1; }
    const size_t read = fread(encoded, 1, (size_t)length, file);
    fclose(file);
    if (read != (size_t)length) { free(encoded); return 1; }

    j2k_decoder *decoder = NULL;
    j2k_error error = {0};
    j2k_status status = j2k_decoder_open(encoded, read, NULL, &decoder, &error);
    free(encoded); /* The decoder owns its copy after a successful open. */
    if (status != J2K_OK) {
        fprintf(stderr, "open (%" PRIu32 "): %s\n", status, error.message);
        return 1;
    }

    j2k_output_info info = J2K_OUTPUT_INFO_INIT;
    status = j2k_decoder_output_info(decoder, NULL, &info, &error);
    uint8_t *pixels = NULL;
    if (status == J2K_OK) {
        pixels = (uint8_t *)malloc((size_t)info.buffer_bytes);
        if (!pixels) {
            fprintf(stderr, "output allocation failed\n");
            j2k_decoder_close(&decoder, NULL);
            return 1;
        }
        status = j2k_decoder_decode_into(decoder, NULL, pixels,
            (size_t)info.buffer_bytes, (size_t)info.row_bytes, &error);
    }
    if (status != J2K_OK) {
        fprintf(stderr, "decode (%" PRIu32 "): %s\n", status, error.message);
    } else {
        printf("%" PRIu32 "x%" PRIu32 ", %" PRIu32 " channels, %" PRIu32
               " significant bits, sample type %" PRIu32 ", %" PRIu64 " bytes\n",
               info.width, info.height, info.channels, info.precision,
               info.sample_type, info.buffer_bytes);
    }
    free(pixels);
    j2k_decoder_close(&decoder, NULL);
    return status == J2K_OK ? 0 : 1;
}
