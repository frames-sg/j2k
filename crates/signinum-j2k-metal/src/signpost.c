#include <stdint.h>

#include <dispatch/dispatch.h>
#include <os/log.h>
#include <os/signpost.h>

enum {
    SIGNINUM_SIGNPOST_DECODE_HYBRID_CPU_TIER1 = 1,
    SIGNINUM_SIGNPOST_DECODE_HYBRID_COEFFICIENT_UPLOAD = 2,
    SIGNINUM_SIGNPOST_DECODE_HYBRID_COMMAND_WAIT = 3,
    SIGNINUM_SIGNPOST_DECODE_HYBRID_IDWT_COMMAND_ENCODE = 4,
    SIGNINUM_SIGNPOST_DECODE_HYBRID_STORE_COMMAND_ENCODE = 5,
    SIGNINUM_SIGNPOST_DECODE_HYBRID_MCT_PACK_COMMAND_ENCODE = 6,
    SIGNINUM_SIGNPOST_ENCODE_HYBRID_COMMAND_WAIT = 7,
    SIGNINUM_SIGNPOST_ENCODE_HYBRID_RESULT_HARVEST = 8,
    SIGNINUM_SIGNPOST_ENCODE_HYBRID_CLASSIC_TIER1_SETUP = 9,
    SIGNINUM_SIGNPOST_ENCODE_HYBRID_CLASSIC_TIER1_COMMAND_ENCODE = 10,
    SIGNINUM_SIGNPOST_ENCODE_HYBRID_CLASSIC_PACKET_PLAN = 11,
    SIGNINUM_SIGNPOST_ENCODE_HYBRID_CLASSIC_PACKET_BUFFER_SETUP = 12,
    SIGNINUM_SIGNPOST_ENCODE_HYBRID_CLASSIC_PACKETIZATION_COMMAND_ENCODE = 13,
    SIGNINUM_SIGNPOST_ENCODE_HYBRID_CLASSIC_PAYLOAD_COPY_COMMAND_ENCODE = 14,
    SIGNINUM_SIGNPOST_ENCODE_HYBRID_CLASSIC_CODESTREAM_ASSEMBLY_COMMAND_ENCODE = 15,
    SIGNINUM_SIGNPOST_ENCODE_HYBRID_HT_TIER1_SETUP = 16,
    SIGNINUM_SIGNPOST_ENCODE_HYBRID_HT_TIER1_COMMAND_ENCODE = 17,
    SIGNINUM_SIGNPOST_ENCODE_HYBRID_HT_PACKET_PLAN = 18,
    SIGNINUM_SIGNPOST_ENCODE_HYBRID_HT_PACKET_BUFFER_SETUP = 19,
    SIGNINUM_SIGNPOST_ENCODE_HYBRID_HT_PACKET_BLOCK_PREP_COMMAND_ENCODE = 20,
    SIGNINUM_SIGNPOST_ENCODE_HYBRID_HT_PACKETIZATION_COMMAND_ENCODE = 21,
    SIGNINUM_SIGNPOST_ENCODE_HYBRID_HT_PAYLOAD_COPY_COMMAND_ENCODE = 22,
    SIGNINUM_SIGNPOST_ENCODE_HYBRID_HT_CODESTREAM_ASSEMBLY_COMMAND_ENCODE = 23,
};

static os_log_t signinum_j2k_metal_log_handle;
static dispatch_once_t signinum_j2k_metal_log_once;

static void signinum_j2k_metal_log_init(void *context) {
    (void)context;
    signinum_j2k_metal_log_handle = os_log_create(
        "com.frames.signinum.j2k-metal",
        OS_LOG_CATEGORY_POINTS_OF_INTEREST);
}

static os_log_t signinum_j2k_metal_log(void) {
    dispatch_once_f(&signinum_j2k_metal_log_once, NULL, signinum_j2k_metal_log_init);
    return signinum_j2k_metal_log_handle;
}

#define SIGNINUM_SIGNPOST_BEGIN_CASE(id, name_literal) \
    case id: \
        os_signpost_interval_begin(log, signpost_id, name_literal); \
        return (uint64_t)signpost_id

#define SIGNINUM_SIGNPOST_END_CASE(id, name_literal) \
    case id: \
        os_signpost_interval_end(log, signpost_id, name_literal); \
        return

uint64_t signinum_j2k_metal_signpost_begin(uint32_t name_id) {
    os_log_t log = signinum_j2k_metal_log();
    if (log == NULL || !os_signpost_enabled(log)) {
        return 0;
    }

    os_signpost_id_t signpost_id = os_signpost_id_generate(log);
    if (signpost_id == OS_SIGNPOST_ID_NULL || signpost_id == OS_SIGNPOST_ID_INVALID) {
        return 0;
    }

    switch (name_id) {
        SIGNINUM_SIGNPOST_BEGIN_CASE(SIGNINUM_SIGNPOST_DECODE_HYBRID_CPU_TIER1,
            "signinum-j2k decode hybrid cpu tier1");
        SIGNINUM_SIGNPOST_BEGIN_CASE(SIGNINUM_SIGNPOST_DECODE_HYBRID_COEFFICIENT_UPLOAD,
            "signinum-j2k decode hybrid coefficient upload");
        SIGNINUM_SIGNPOST_BEGIN_CASE(SIGNINUM_SIGNPOST_DECODE_HYBRID_COMMAND_WAIT,
            "signinum-j2k decode hybrid command wait");
        SIGNINUM_SIGNPOST_BEGIN_CASE(SIGNINUM_SIGNPOST_DECODE_HYBRID_IDWT_COMMAND_ENCODE,
            "signinum-j2k decode hybrid idwt command encode");
        SIGNINUM_SIGNPOST_BEGIN_CASE(SIGNINUM_SIGNPOST_DECODE_HYBRID_STORE_COMMAND_ENCODE,
            "signinum-j2k decode hybrid store command encode");
        SIGNINUM_SIGNPOST_BEGIN_CASE(SIGNINUM_SIGNPOST_DECODE_HYBRID_MCT_PACK_COMMAND_ENCODE,
            "signinum-j2k decode hybrid mct pack command encode");
        SIGNINUM_SIGNPOST_BEGIN_CASE(SIGNINUM_SIGNPOST_ENCODE_HYBRID_COMMAND_WAIT,
            "signinum-j2k encode hybrid command wait");
        SIGNINUM_SIGNPOST_BEGIN_CASE(SIGNINUM_SIGNPOST_ENCODE_HYBRID_RESULT_HARVEST,
            "signinum-j2k encode hybrid result harvest");
        SIGNINUM_SIGNPOST_BEGIN_CASE(SIGNINUM_SIGNPOST_ENCODE_HYBRID_CLASSIC_TIER1_SETUP,
            "signinum-j2k encode hybrid classic tier1 setup");
        SIGNINUM_SIGNPOST_BEGIN_CASE(SIGNINUM_SIGNPOST_ENCODE_HYBRID_CLASSIC_TIER1_COMMAND_ENCODE,
            "signinum-j2k encode hybrid classic tier1 command encode");
        SIGNINUM_SIGNPOST_BEGIN_CASE(SIGNINUM_SIGNPOST_ENCODE_HYBRID_CLASSIC_PACKET_PLAN,
            "signinum-j2k encode hybrid classic packet plan");
        SIGNINUM_SIGNPOST_BEGIN_CASE(SIGNINUM_SIGNPOST_ENCODE_HYBRID_CLASSIC_PACKET_BUFFER_SETUP,
            "signinum-j2k encode hybrid classic packet buffer setup");
        SIGNINUM_SIGNPOST_BEGIN_CASE(
            SIGNINUM_SIGNPOST_ENCODE_HYBRID_CLASSIC_PACKETIZATION_COMMAND_ENCODE,
            "signinum-j2k encode hybrid classic packetization command encode");
        SIGNINUM_SIGNPOST_BEGIN_CASE(SIGNINUM_SIGNPOST_ENCODE_HYBRID_CLASSIC_PAYLOAD_COPY_COMMAND_ENCODE,
            "signinum-j2k encode hybrid classic payload copy command encode");
        SIGNINUM_SIGNPOST_BEGIN_CASE(
            SIGNINUM_SIGNPOST_ENCODE_HYBRID_CLASSIC_CODESTREAM_ASSEMBLY_COMMAND_ENCODE,
            "signinum-j2k encode hybrid classic codestream assembly command encode");
        SIGNINUM_SIGNPOST_BEGIN_CASE(SIGNINUM_SIGNPOST_ENCODE_HYBRID_HT_TIER1_SETUP,
            "signinum-j2k encode hybrid ht tier1 setup");
        SIGNINUM_SIGNPOST_BEGIN_CASE(SIGNINUM_SIGNPOST_ENCODE_HYBRID_HT_TIER1_COMMAND_ENCODE,
            "signinum-j2k encode hybrid ht tier1 command encode");
        SIGNINUM_SIGNPOST_BEGIN_CASE(SIGNINUM_SIGNPOST_ENCODE_HYBRID_HT_PACKET_PLAN,
            "signinum-j2k encode hybrid ht packet plan");
        SIGNINUM_SIGNPOST_BEGIN_CASE(SIGNINUM_SIGNPOST_ENCODE_HYBRID_HT_PACKET_BUFFER_SETUP,
            "signinum-j2k encode hybrid ht packet buffer setup");
        SIGNINUM_SIGNPOST_BEGIN_CASE(
            SIGNINUM_SIGNPOST_ENCODE_HYBRID_HT_PACKET_BLOCK_PREP_COMMAND_ENCODE,
            "signinum-j2k encode hybrid ht packet block prep command encode");
        SIGNINUM_SIGNPOST_BEGIN_CASE(SIGNINUM_SIGNPOST_ENCODE_HYBRID_HT_PACKETIZATION_COMMAND_ENCODE,
            "signinum-j2k encode hybrid ht packetization command encode");
        SIGNINUM_SIGNPOST_BEGIN_CASE(SIGNINUM_SIGNPOST_ENCODE_HYBRID_HT_PAYLOAD_COPY_COMMAND_ENCODE,
            "signinum-j2k encode hybrid ht payload copy command encode");
        SIGNINUM_SIGNPOST_BEGIN_CASE(SIGNINUM_SIGNPOST_ENCODE_HYBRID_HT_CODESTREAM_ASSEMBLY_COMMAND_ENCODE,
            "signinum-j2k encode hybrid ht codestream assembly command encode");
        default:
            return 0;
    }
}

void signinum_j2k_metal_signpost_end(uint32_t name_id, uint64_t raw_signpost_id) {
    if (raw_signpost_id == OS_SIGNPOST_ID_NULL || raw_signpost_id == OS_SIGNPOST_ID_INVALID) {
        return;
    }

    os_log_t log = signinum_j2k_metal_log();
    if (log == NULL) {
        return;
    }

    os_signpost_id_t signpost_id = (os_signpost_id_t)raw_signpost_id;
    switch (name_id) {
        SIGNINUM_SIGNPOST_END_CASE(SIGNINUM_SIGNPOST_DECODE_HYBRID_CPU_TIER1,
            "signinum-j2k decode hybrid cpu tier1");
        SIGNINUM_SIGNPOST_END_CASE(SIGNINUM_SIGNPOST_DECODE_HYBRID_COEFFICIENT_UPLOAD,
            "signinum-j2k decode hybrid coefficient upload");
        SIGNINUM_SIGNPOST_END_CASE(SIGNINUM_SIGNPOST_DECODE_HYBRID_COMMAND_WAIT,
            "signinum-j2k decode hybrid command wait");
        SIGNINUM_SIGNPOST_END_CASE(SIGNINUM_SIGNPOST_DECODE_HYBRID_IDWT_COMMAND_ENCODE,
            "signinum-j2k decode hybrid idwt command encode");
        SIGNINUM_SIGNPOST_END_CASE(SIGNINUM_SIGNPOST_DECODE_HYBRID_STORE_COMMAND_ENCODE,
            "signinum-j2k decode hybrid store command encode");
        SIGNINUM_SIGNPOST_END_CASE(SIGNINUM_SIGNPOST_DECODE_HYBRID_MCT_PACK_COMMAND_ENCODE,
            "signinum-j2k decode hybrid mct pack command encode");
        SIGNINUM_SIGNPOST_END_CASE(SIGNINUM_SIGNPOST_ENCODE_HYBRID_COMMAND_WAIT,
            "signinum-j2k encode hybrid command wait");
        SIGNINUM_SIGNPOST_END_CASE(SIGNINUM_SIGNPOST_ENCODE_HYBRID_RESULT_HARVEST,
            "signinum-j2k encode hybrid result harvest");
        SIGNINUM_SIGNPOST_END_CASE(SIGNINUM_SIGNPOST_ENCODE_HYBRID_CLASSIC_TIER1_SETUP,
            "signinum-j2k encode hybrid classic tier1 setup");
        SIGNINUM_SIGNPOST_END_CASE(SIGNINUM_SIGNPOST_ENCODE_HYBRID_CLASSIC_TIER1_COMMAND_ENCODE,
            "signinum-j2k encode hybrid classic tier1 command encode");
        SIGNINUM_SIGNPOST_END_CASE(SIGNINUM_SIGNPOST_ENCODE_HYBRID_CLASSIC_PACKET_PLAN,
            "signinum-j2k encode hybrid classic packet plan");
        SIGNINUM_SIGNPOST_END_CASE(SIGNINUM_SIGNPOST_ENCODE_HYBRID_CLASSIC_PACKET_BUFFER_SETUP,
            "signinum-j2k encode hybrid classic packet buffer setup");
        SIGNINUM_SIGNPOST_END_CASE(
            SIGNINUM_SIGNPOST_ENCODE_HYBRID_CLASSIC_PACKETIZATION_COMMAND_ENCODE,
            "signinum-j2k encode hybrid classic packetization command encode");
        SIGNINUM_SIGNPOST_END_CASE(SIGNINUM_SIGNPOST_ENCODE_HYBRID_CLASSIC_PAYLOAD_COPY_COMMAND_ENCODE,
            "signinum-j2k encode hybrid classic payload copy command encode");
        SIGNINUM_SIGNPOST_END_CASE(
            SIGNINUM_SIGNPOST_ENCODE_HYBRID_CLASSIC_CODESTREAM_ASSEMBLY_COMMAND_ENCODE,
            "signinum-j2k encode hybrid classic codestream assembly command encode");
        SIGNINUM_SIGNPOST_END_CASE(SIGNINUM_SIGNPOST_ENCODE_HYBRID_HT_TIER1_SETUP,
            "signinum-j2k encode hybrid ht tier1 setup");
        SIGNINUM_SIGNPOST_END_CASE(SIGNINUM_SIGNPOST_ENCODE_HYBRID_HT_TIER1_COMMAND_ENCODE,
            "signinum-j2k encode hybrid ht tier1 command encode");
        SIGNINUM_SIGNPOST_END_CASE(SIGNINUM_SIGNPOST_ENCODE_HYBRID_HT_PACKET_PLAN,
            "signinum-j2k encode hybrid ht packet plan");
        SIGNINUM_SIGNPOST_END_CASE(SIGNINUM_SIGNPOST_ENCODE_HYBRID_HT_PACKET_BUFFER_SETUP,
            "signinum-j2k encode hybrid ht packet buffer setup");
        SIGNINUM_SIGNPOST_END_CASE(
            SIGNINUM_SIGNPOST_ENCODE_HYBRID_HT_PACKET_BLOCK_PREP_COMMAND_ENCODE,
            "signinum-j2k encode hybrid ht packet block prep command encode");
        SIGNINUM_SIGNPOST_END_CASE(SIGNINUM_SIGNPOST_ENCODE_HYBRID_HT_PACKETIZATION_COMMAND_ENCODE,
            "signinum-j2k encode hybrid ht packetization command encode");
        SIGNINUM_SIGNPOST_END_CASE(SIGNINUM_SIGNPOST_ENCODE_HYBRID_HT_PAYLOAD_COPY_COMMAND_ENCODE,
            "signinum-j2k encode hybrid ht payload copy command encode");
        SIGNINUM_SIGNPOST_END_CASE(SIGNINUM_SIGNPOST_ENCODE_HYBRID_HT_CODESTREAM_ASSEMBLY_COMMAND_ENCODE,
            "signinum-j2k encode hybrid ht codestream assembly command encode");
        default:
            return;
    }
}
