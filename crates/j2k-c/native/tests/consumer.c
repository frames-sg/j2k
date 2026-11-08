/* SPDX-License-Identifier: MIT OR Apache-2.0 */
/* A real C translation unit checks the installed header against the shared ABI. */
#include <j2k.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define REQUIRE(value) do { if (!(value)) { \
    fprintf(stderr, "failed at line %d: %s\n", __LINE__, #value); return 1; \
} } while (0)

_Static_assert(sizeof(j2k_open_options) == 32, "ABI 1 open options");
_Static_assert(sizeof(j2k_decode_options) == 32, "ABI 1 decode options");
_Static_assert(sizeof(j2k_image_info) == 56, "ABI 1 image info");
_Static_assert(sizeof(j2k_component_info) == 24, "ABI 1 component info");
_Static_assert(sizeof(j2k_output_info) == 64, "ABI 1 output info");
_Static_assert(sizeof(j2k_encode_options) == 32, "ABI 1 encode options");
_Static_assert(sizeof(j2k_error) == 260, "ABI 1 error");
_Static_assert(sizeof(j2k_region_plan) == 104, "exact region plan layout");
_Static_assert(offsetof(j2k_operation_options, cancellation) == 48, "operation token offset");

/* Copies each row after the previous one; stops after stop_after rows. */
typedef struct rows {
    uint8_t bytes[768];
    size_t filled;
    uint32_t next_y;
    uint32_t stop_after;
    int out_of_order;
} rows;

static int32_t take_row(void *user_data, uint32_t y, const uint8_t *row, size_t row_bytes) {
    rows *state = (rows *)user_data;
    if (y != state->next_y || state->filled + row_bytes > sizeof(state->bytes)) {
        state->out_of_order = 1;
        return 1;
    }
    memcpy(state->bytes + state->filled, row, row_bytes);
    state->filled += row_bytes;
    state->next_y = y + 1;
    return state->next_y == state->stop_after;
}

int main(int argc, char **argv) {
    REQUIRE(argc == 2);
    REQUIRE(j2k_abi_version() == J2K_ABI_VERSION);
    REQUIRE(strlen(j2k_version()) > 0);
    FILE *file = fopen(argv[1], "rb");
    REQUIRE(file != NULL);
    uint8_t encoded[8192];
    const size_t length = fread(encoded, 1, sizeof(encoded), file);
    const int complete = feof(file) && !ferror(file);
    fclose(file);
    REQUIRE(length > 0 && complete);

    j2k_error error = {0};
    j2k_decoder *decoder = NULL;
    j2k_open_options open_options = J2K_OPEN_OPTIONS_INIT;
    open_options.max_input_bytes = length;
    open_options.max_output_bytes = 768;
    REQUIRE(j2k_decoder_open(encoded, length, &open_options, &decoder, &error) == J2K_OK);
    memset(encoded, 0, length); /* No retained borrow of the caller's input. */
    j2k_image_info image = J2K_IMAGE_INFO_INIT;
    REQUIRE(j2k_decoder_info(decoder, &image, &error) == J2K_OK);
    REQUIRE(image.width == 32 && image.height == 24 && image.components == 1);
    j2k_component_info component = J2K_COMPONENT_INFO_INIT;
    REQUIRE(j2k_decoder_component_info(decoder, 0, &component, &error) == J2K_OK);
    REQUIRE(component.precision == 8 && component.is_signed == 0);
    REQUIRE(component.sampling_x == 1 && component.sampling_y == 1);

    j2k_output_info output = J2K_OUTPUT_INFO_INIT;
    REQUIRE(j2k_decoder_output_info(decoder, NULL, &output, &error) == J2K_OK);
    REQUIRE(output.width == 32 && output.height == 24 && output.channels == 1);
    REQUIRE(output.sample_type == J2K_U8 && output.precision == 8);
    REQUIRE(output.row_bytes == 32 && output.buffer_bytes == 768);
    uint8_t pixels[768];
    REQUIRE(j2k_decoder_decode_into(decoder, NULL, pixels, sizeof(pixels), 32, &error) == J2K_OK);
    for (size_t i = 0; i < sizeof(pixels); ++i) REQUIRE(pixels[i] == (i * 13 + 7) % 256);

    j2k_decode_options region = J2K_DECODE_OPTIONS_INIT;
    region.x = 3; region.y = 5; region.width = 7; region.height = 6;
    REQUIRE(j2k_decoder_output_info(decoder, &region, &output, &error) == J2K_OK);
    REQUIRE(output.x == 3 && output.y == 5 && output.width == 7 && output.height == 6);
    uint8_t padded[66];
    memset(padded, 0xA5, sizeof(padded));
    REQUIRE(j2k_decoder_decode_into(decoder, &region, padded, 61, 11, &error) == J2K_BUFFER_TOO_SMALL);
    REQUIRE(error.code == J2K_BUFFER_TOO_SMALL && error.message[0] != '\0');
    for (size_t i = 0; i < sizeof(padded); ++i) REQUIRE(padded[i] == 0xA5);
    REQUIRE(j2k_decoder_decode_into(decoder, &region, padded, 62, 11, &error) == J2K_OK);
    REQUIRE(error.code == J2K_OK && error.message[0] == '\0');
    for (size_t y = 0; y < 6; ++y) {
        for (size_t x = 0; x < 7; ++x) REQUIRE(padded[y * 11 + x] == pixels[(y + 5) * 32 + x + 3]);
        for (size_t x = 7; x < 11; ++x) REQUIRE(padded[y * 11 + x] == 0xA5);
    }

    /* Rows stream the full image in decode_into's layout, in order. */
    j2k_output_info row_layout = J2K_OUTPUT_INFO_INIT;
    REQUIRE(j2k_decoder_row_info(decoder, NULL, &row_layout, &error) == J2K_OK);
    REQUIRE(row_layout.width == 32 && row_layout.height == 24 && row_layout.row_bytes == 32);
    rows streamed = {{0}, 0, 0, 0, 0};
    j2k_output_info streamed_layout = J2K_OUTPUT_INFO_INIT;
    REQUIRE(j2k_decoder_decode_rows(decoder, NULL, 5, take_row, &streamed, &streamed_layout, &error) == J2K_OK);
    REQUIRE(!streamed.out_of_order && streamed.next_y == 24 && streamed.filled == 768);
    REQUIRE(memcmp(streamed.bytes, pixels, 768) == 0 && streamed_layout.warnings == 0);
    rows stopped = {{0}, 0, 0, 3, 0};
    REQUIRE(j2k_decoder_decode_rows(decoder, NULL, 0, take_row, &stopped, NULL, &error) == J2K_STOPPED);
    REQUIRE(error.code == J2K_STOPPED && stopped.next_y == 3);
    REQUIRE(j2k_decoder_decode_rows(decoder, NULL, 0, NULL, NULL, NULL, &error) == J2K_INVALID_ARGUMENT);

    /* Additive exact operations preserve ABI-1 native output and token ownership. */
    /* Preparation is optional: an unprepared decoder prepares its source under
     * the operation's working limit and keeps only a successful preparation. */
    j2k_operation_options operation = J2K_OPERATION_OPTIONS_INIT;
    j2k_region_plan plan = J2K_REGION_PLAN_INIT;
    operation.max_working_memory_bytes = 1;
    REQUIRE(j2k_decoder_plan_region(decoder, &operation, &plan, &error) == J2K_RESOURCE_LIMIT);
    operation.max_working_memory_bytes = UINT64_C(8) << 20;
    REQUIRE(j2k_decoder_plan_region(decoder, &operation, &plan, &error) == J2K_OK);
    REQUIRE(j2k_decoder_prepare_operations(decoder, &error) == J2K_OK); /* already prepared */
    REQUIRE(j2k_decoder_plan_region(decoder, &operation, &plan, &error) == J2K_OK);
    REQUIRE(plan.output.buffer_bytes == 768 && plan.route == J2K_ROUTE_NATIVE_CPU);
    uint8_t exact[768];
    j2k_output_info actual = J2K_OUTPUT_INFO_INIT;
    REQUIRE(j2k_decoder_decode_region_into(decoder, &operation, exact, sizeof(exact), 32, &actual, &error) == J2K_OK);
    REQUIRE(memcmp(exact, pixels, sizeof(exact)) == 0 && actual.precision == 8);
    j2k_cancellation *token = NULL;
    REQUIRE(j2k_cancellation_create(&token, &error) == J2K_OK);
    REQUIRE(j2k_cancellation_cancel(token, &error) == J2K_OK);
    operation.cancellation = token;
    memset(exact, 0xA5, sizeof(exact));
    REQUIRE(j2k_decoder_decode_region_into(decoder, &operation, exact, sizeof(exact), 32, NULL, &error) == J2K_CANCELLED);
    for (size_t i = 0; i < sizeof(exact); ++i) REQUIRE(exact[i] == 0xA5);
    operation.cancellation = NULL;
    REQUIRE(j2k_cancellation_free(&token, &error) == J2K_OK && token == NULL);
    REQUIRE(j2k_decoder_set_operation_cache_limit(decoder, 0, &error) == J2K_OK);
    operation.max_working_memory_bytes = 0;
    operation.interpretation = J2K_INTERPRETATION_SRGB8;
    REQUIRE(j2k_decoder_plan_region(decoder, &operation, &plan, &error) == J2K_OK);
    REQUIRE(plan.route == J2K_ROUTE_REGION_PLANES && plan.output.sample_type == J2K_U8);
    REQUIRE(j2k_decoder_decode_region_into(decoder, &operation, exact, sizeof(exact), 32, &actual, &error) == J2K_OK);
    REQUIRE(memcmp(exact, pixels, sizeof(exact)) == 0);

    REQUIRE(j2k_decoder_close(&decoder, &error) == J2K_OK);
    REQUIRE(decoder == NULL);
    REQUIRE(j2k_decoder_close(&decoder, NULL) == J2K_OK);

    /* Encode the decoded pixels as JPH and decode them again, on own threads. */
    j2k_encode_options encode = J2K_ENCODE_OPTIONS_INIT;
    encode.width = 32; encode.height = 24; encode.components = 1; encode.bit_depth = 8;
    encode.flags = J2K_ENCODE_HTJ2K | J2K_ENCODE_FILE;
    uint8_t *file_bytes = (uint8_t *)1;
    size_t file_length = 1;
    REQUIRE(j2k_encode_lossless(pixels, sizeof(pixels), &encode, &file_bytes, &file_length, &error) == J2K_OK);
    REQUIRE(file_bytes != NULL && file_length > 12 && file_bytes[4] == 'j' && file_bytes[5] == 'P');
    j2k_open_options threaded = J2K_OPEN_OPTIONS_INIT;
    threaded.workers = 2;
    REQUIRE(j2k_decoder_open(file_bytes, file_length, &threaded, &decoder, &error) == J2K_OK);
    j2k_buffer_free(file_bytes, file_length);
    uint8_t round_trip[768];
    REQUIRE(j2k_decoder_decode_into(decoder, NULL, round_trip, sizeof(round_trip), 32, &error) == J2K_OK);
    REQUIRE(memcmp(round_trip, pixels, sizeof(pixels)) == 0);
    REQUIRE(j2k_decoder_close(&decoder, &error) == J2K_OK);
    encode.bit_depth = 17;
    REQUIRE(j2k_encode_lossless(pixels, sizeof(pixels), &encode, &file_bytes, &file_length, &error) == J2K_INVALID_ARGUMENT);
    REQUIRE(file_bytes == NULL && file_length == 0);
    j2k_buffer_free(NULL, 0);
    return 0;
}
