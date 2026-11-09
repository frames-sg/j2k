// SPDX-License-Identifier: MIT OR Apache-2.0
// C++17 ownership wrapper over the C ABI.
#ifndef J2K_HPP
#define J2K_HPP

#include "j2k.h"
#include <exception>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace j2k {

class error : public std::runtime_error {
public:
    error(j2k_status status, const char *message)
        : std::runtime_error(message), status_(status) {}
    j2k_status status() const noexcept { return status_; }
private:
    j2k_status status_;
};

inline void check(j2k_status status, const j2k_error &detail) {
    if (status != J2K_OK) throw error(status, detail.message);
}

namespace detail {
template <typename OnRow> struct row_context {
    OnRow *on_row;
    std::exception_ptr failure;
};

// No exception crosses the C ABI: one thrown by on_row stops the decode and
// is rethrown by decoder::decode_rows.
template <typename OnRow>
int32_t forward_row(void *user_data, uint32_t y, const uint8_t *row, size_t row_bytes) noexcept {
    auto &context = *static_cast<row_context<OnRow> *>(user_data);
    try {
        return (*context.on_row)(y, row, row_bytes) ? 0 : 1;
    } catch (...) {
        context.failure = std::current_exception();
        return 1;
    }
}
} // namespace detail

// Move-only token. cancel() may run concurrently; destruction/move may not.
class cancellation_token {
public:
    cancellation_token() {
        j2k_error detail{};
        check(j2k_cancellation_create(&handle_, &detail), detail);
    }
    ~cancellation_token() noexcept { j2k_cancellation_free(&handle_, nullptr); }
    cancellation_token(const cancellation_token &) = delete;
    cancellation_token &operator=(const cancellation_token &) = delete;
    cancellation_token(cancellation_token &&other) noexcept
        : handle_(std::exchange(other.handle_, nullptr)) {}
    cancellation_token &operator=(cancellation_token &&other) noexcept {
        if (this != &other) {
            j2k_cancellation_free(&handle_, nullptr);
            handle_ = std::exchange(other.handle_, nullptr);
        }
        return *this;
    }
    const j2k_cancellation *native_handle() const noexcept { return handle_; }
    void cancel() const {
        j2k_error detail{};
        check(j2k_cancellation_cancel(handle_, &detail), detail);
    }
private:
    j2k_cancellation *handle_ = nullptr;
};

namespace detail {
// Copies a library-allocated buffer into a vector and releases it either way.
inline std::vector<uint8_t> take_buffer(uint8_t *data, size_t size) {
    try {
        std::vector<uint8_t> result(data, data + size);
        j2k_buffer_free(data, size);
        return result;
    } catch (...) {
        j2k_buffer_free(data, size);
        throw;
    }
}
} // namespace detail

// Encode interleaved samples losslessly (see j2k_encode_lossless).
inline std::vector<uint8_t> encode_lossless(const uint8_t *samples, size_t length,
                                            const j2k_encode_options &options) {
    uint8_t *data = nullptr;
    size_t size = 0;
    j2k_error detail{};
    check(j2k_encode_lossless(samples, length, &options, &data, &size, &detail), detail);
    return detail::take_buffer(data, size);
}

// Encode interleaved samples with coding parameters (see j2k_encode). report,
// when given, receives what the codec measured.
inline std::vector<uint8_t> encode(const uint8_t *samples, size_t length,
                                   const j2k_encode_options &options,
                                   const j2k_encode_params &params = J2K_ENCODE_PARAMS_INIT,
                                   j2k_encode_report *report = nullptr) {
    uint8_t *data = nullptr;
    size_t size = 0;
    j2k_error detail{};
    check(j2k_encode(samples, length, &options, &params, &data, &size, report, &detail), detail);
    return detail::take_buffer(data, size);
}

// Move-only; serial access per instance. The constructor copies encoded bytes.
class decoder {
public:
    decoder(const uint8_t *input, size_t length,
            const j2k_open_options &options = J2K_OPEN_OPTIONS_INIT) {
        j2k_error detail{};
        const auto status = j2k_decoder_open(input, length, &options, &handle_, &detail);
        check(status, detail);
    }

    ~decoder() noexcept { j2k_decoder_close(&handle_, nullptr); }
    decoder(const decoder &) = delete;
    decoder &operator=(const decoder &) = delete;
    decoder(decoder &&other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}
    decoder &operator=(decoder &&other) noexcept {
        if (this != &other) {
            j2k_decoder_close(&handle_, nullptr);
            handle_ = std::exchange(other.handle_, nullptr);
        }
        return *this;
    }

    j2k_image_info info() {
        j2k_image_info result = J2K_IMAGE_INFO_INIT;
        j2k_error detail{};
        const auto status = j2k_decoder_info(handle_, &result, &detail);
        check(status, detail);
        return result;
    }

    j2k_component_info component(uint32_t index) {
        j2k_component_info result = J2K_COMPONENT_INFO_INIT;
        j2k_error detail{};
        const auto status = j2k_decoder_component_info(handle_, index, &result, &detail);
        check(status, detail);
        return result;
    }

    // The first COLR box, which decoding applies.
    j2k_color_info color_info() {
        j2k_color_info result = J2K_COLOR_INFO_INIT;
        j2k_error detail{};
        check(j2k_decoder_color_info(handle_, &result, &detail), detail);
        return result;
    }

    // The ICC profile decoding applies; empty without one.
    std::vector<uint8_t> icc_profile() {
        const auto color = color_info();
        if (color.specification != J2K_COLOR_ICC) return {};
        std::vector<uint8_t> result(static_cast<size_t>(color.icc_profile_bytes));
        j2k_error detail{};
        check(j2k_decoder_icc_profile(handle_, result.data(), result.size(), &detail), detail);
        return result;
    }

    j2k_output_info output_info(
            const j2k_decode_options &options = J2K_DECODE_OPTIONS_INIT) {
        j2k_output_info result = J2K_OUTPUT_INFO_INIT;
        j2k_error detail{};
        const auto status = j2k_decoder_output_info(handle_, &options, &result, &detail);
        check(status, detail);
        return result;
    }

    void decode_into(uint8_t *output, size_t length, size_t stride,
                     const j2k_decode_options &options = J2K_DECODE_OPTIONS_INIT) {
        j2k_error detail{};
        const auto status = j2k_decoder_decode_into(handle_, &options, output, length, stride, &detail);
        check(status, detail);
    }

    // Optional explicit input preparation, outside later operation controls.
    // Controlled native operations prepare an unprepared source themselves.
    void prepare_operations() {
        j2k_error detail{};
        check(j2k_decoder_prepare_operations(handle_, &detail), detail);
    }

    j2k_region_plan plan_region(
            const j2k_operation_options &options = J2K_OPERATION_OPTIONS_INIT) {
        j2k_region_plan result = J2K_REGION_PLAN_INIT;
        j2k_error detail{};
        check(j2k_decoder_plan_region(handle_, &options, &result, &detail), detail);
        return result;
    }

    j2k_output_info decode_region_into(uint8_t *output, size_t length, size_t stride,
            const j2k_operation_options &options = J2K_OPERATION_OPTIONS_INIT) {
        j2k_output_info result = J2K_OUTPUT_INFO_INIT;
        j2k_error detail{};
        check(j2k_decoder_decode_region_into(handle_, &options, output, length,
                                            stride, &result, &detail), detail);
        return result;
    }

    void set_operation_cache_limit(size_t bytes) {
        j2k_error detail{};
        check(j2k_decoder_set_operation_cache_limit(handle_, bytes, &detail), detail);
    }

    // The layout decode_rows delivers: the full image at full resolution.
    j2k_output_info row_info() {
        j2k_output_info result = J2K_OUTPUT_INFO_INIT;
        j2k_error detail{};
        check(j2k_decoder_row_info(handle_, nullptr, &result, &detail), detail);
        return result;
    }

    // Calls on_row(y, row, row_bytes) for each row of the full image, in order;
    // on_row returns false to stop. Returns the row layout with its warnings,
    // or nothing when on_row stopped the decode. An exception thrown by on_row
    // stops the decode and is rethrown here.
    template <typename OnRow>
    std::optional<j2k_output_info> decode_rows(OnRow &&on_row, uint32_t max_rows_per_stripe = 0) {
        using callable = std::remove_reference_t<OnRow>;
        detail::row_context<callable> context{&on_row, nullptr};
        j2k_output_info result = J2K_OUTPUT_INFO_INIT;
        j2k_error diagnostic{};
        const auto status = j2k_decoder_decode_rows(
            handle_, nullptr, max_rows_per_stripe, &detail::forward_row<callable>, &context,
            &result, &diagnostic);
        if (context.failure) std::rethrow_exception(context.failure);
        if (status == J2K_STOPPED) return std::nullopt;
        check(status, diagnostic);
        return result;
    }

private:
    j2k_decoder *handle_ = nullptr;
};

} // namespace j2k
#endif
