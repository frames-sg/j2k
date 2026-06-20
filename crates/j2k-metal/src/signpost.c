#include <stdint.h>

#include <dispatch/dispatch.h>
#include <os/log.h>
#include <os/signpost.h>

enum {
    J2K_SIGNPOST_DECODE_HYBRID_CPU_TIER1 = 1,
    J2K_SIGNPOST_DECODE_HYBRID_COEFFICIENT_UPLOAD = 2,
    J2K_SIGNPOST_DECODE_HYBRID_COMMAND_WAIT = 3,
    J2K_SIGNPOST_DECODE_HYBRID_IDWT_COMMAND_ENCODE = 4,
    J2K_SIGNPOST_DECODE_HYBRID_STORE_COMMAND_ENCODE = 5,
    J2K_SIGNPOST_DECODE_HYBRID_MCT_PACK_COMMAND_ENCODE = 6,
    J2K_SIGNPOST_ENCODE_HYBRID_COMMAND_WAIT = 7,
    J2K_SIGNPOST_ENCODE_HYBRID_RESULT_HARVEST = 8,
    J2K_SIGNPOST_ENCODE_HYBRID_CLASSIC_TIER1_SETUP = 9,
    J2K_SIGNPOST_ENCODE_HYBRID_CLASSIC_TIER1_COMMAND_ENCODE = 10,
    J2K_SIGNPOST_ENCODE_HYBRID_CLASSIC_PACKET_PLAN = 11,
    J2K_SIGNPOST_ENCODE_HYBRID_CLASSIC_PACKET_BUFFER_SETUP = 12,
    J2K_SIGNPOST_ENCODE_HYBRID_CLASSIC_PACKETIZATION_COMMAND_ENCODE = 13,
    J2K_SIGNPOST_ENCODE_HYBRID_CLASSIC_PAYLOAD_COPY_COMMAND_ENCODE = 14,
    J2K_SIGNPOST_ENCODE_HYBRID_CLASSIC_CODESTREAM_ASSEMBLY_COMMAND_ENCODE = 15,
    J2K_SIGNPOST_ENCODE_HYBRID_HT_TIER1_SETUP = 16,
    J2K_SIGNPOST_ENCODE_HYBRID_HT_TIER1_COMMAND_ENCODE = 17,
    J2K_SIGNPOST_ENCODE_HYBRID_HT_PACKET_PLAN = 18,
    J2K_SIGNPOST_ENCODE_HYBRID_HT_PACKET_BUFFER_SETUP = 19,
    J2K_SIGNPOST_ENCODE_HYBRID_HT_PACKET_BLOCK_PREP_COMMAND_ENCODE = 20,
    J2K_SIGNPOST_ENCODE_HYBRID_HT_PACKETIZATION_COMMAND_ENCODE = 21,
    J2K_SIGNPOST_ENCODE_HYBRID_HT_PAYLOAD_COPY_COMMAND_ENCODE = 22,
    J2K_SIGNPOST_ENCODE_HYBRID_HT_CODESTREAM_ASSEMBLY_COMMAND_ENCODE = 23,
};

static os_log_t j2k_metal_log_handle;
static dispatch_once_t j2k_metal_log_once;

static void j2k_metal_log_init(void *context) {
    (void)context;
   j2k_metal_log_handle = os_log_create(
        "com.frames. j2k.j2k-metal",
        OS_LOG_CATEGORY_POINTS_OF_INTEREST);
}

static os_log_t j2k_metal_log(void) {
    dispatch_once_f(&j2k_metal_log_once, NULL, j2k_metal_log_init);
    return j2k_metal_log_handle;
}

#define J2K_SIGNPOST_BEGIN_CASE(id, name_literal) \
    case id: \
        os_signpost_interval_begin(log, signpost_id, name_literal); \
        return (uint64_t)signpost_id

#define J2K_SIGNPOST_END_CASE(id, name_literal) \
    case id: \
        os_signpost_interval_end(log, signpost_id, name_literal); \
        return

uint64_t j2k_metal_signpost_begin(uint32_t name_id) {
    os_log_t log = j2k_metal_log();
    if (log == NULL || !os_signpost_enabled(log)) {
        return 0;
    }

    os_signpost_id_t signpost_id = os_signpost_id_generate(log);
    if (signpost_id == OS_SIGNPOST_ID_NULL || signpost_id == OS_SIGNPOST_ID_INVALID) {
        return 0;
    }

    switch (name_id) {
        J2K_SIGNPOST_BEGIN_CASE(J2K_SIGNPOST_DECODE_HYBRID_CPU_TIER1,
            "j2k decode hybrid cpu tier1");
        J2K_SIGNPOST_BEGIN_CASE(J2K_SIGNPOST_DECODE_HYBRID_COEFFICIENT_UPLOAD,
            "j2k decode hybrid coefficient upload");
        J2K_SIGNPOST_BEGIN_CASE(J2K_SIGNPOST_DECODE_HYBRID_COMMAND_WAIT,
            "j2k decode hybrid command wait");
        J2K_SIGNPOST_BEGIN_CASE(J2K_SIGNPOST_DECODE_HYBRID_IDWT_COMMAND_ENCODE,
            "j2k decode hybrid idwt command encode");
        J2K_SIGNPOST_BEGIN_CASE(J2K_SIGNPOST_DECODE_HYBRID_STORE_COMMAND_ENCODE,
            "j2k decode hybrid store command encode");
        J2K_SIGNPOST_BEGIN_CASE(J2K_SIGNPOST_DECODE_HYBRID_MCT_PACK_COMMAND_ENCODE,
            "j2k decode hybrid mct pack command encode");
        J2K_SIGNPOST_BEGIN_CASE(J2K_SIGNPOST_ENCODE_HYBRID_COMMAND_WAIT,
            "j2k encode hybrid command wait");
        J2K_SIGNPOST_BEGIN_CASE(J2K_SIGNPOST_ENCODE_HYBRID_RESULT_HARVEST,
            "j2k encode hybrid result harvest");
        J2K_SIGNPOST_BEGIN_CASE(J2K_SIGNPOST_ENCODE_HYBRID_CLASSIC_TIER1_SETUP,
            "j2k encode hybrid classic tier1 setup");
        J2K_SIGNPOST_BEGIN_CASE(J2K_SIGNPOST_ENCODE_HYBRID_CLASSIC_TIER1_COMMAND_ENCODE,
            "j2k encode hybrid classic tier1 command encode");
        J2K_SIGNPOST_BEGIN_CASE(J2K_SIGNPOST_ENCODE_HYBRID_CLASSIC_PACKET_PLAN,
            "j2k encode hybrid classic packet plan");
        J2K_SIGNPOST_BEGIN_CASE(J2K_SIGNPOST_ENCODE_HYBRID_CLASSIC_PACKET_BUFFER_SETUP,
            "j2k encode hybrid classic packet buffer setup");
        J2K_SIGNPOST_BEGIN_CASE(
            J2K_SIGNPOST_ENCODE_HYBRID_CLASSIC_PACKETIZATION_COMMAND_ENCODE,
            "j2k encode hybrid classic packetization command encode");
        J2K_SIGNPOST_BEGIN_CASE(J2K_SIGNPOST_ENCODE_HYBRID_CLASSIC_PAYLOAD_COPY_COMMAND_ENCODE,
            "j2k encode hybrid classic payload copy command encode");
        J2K_SIGNPOST_BEGIN_CASE(
            J2K_SIGNPOST_ENCODE_HYBRID_CLASSIC_CODESTREAM_ASSEMBLY_COMMAND_ENCODE,
            "j2k encode hybrid classic codestream assembly command encode");
        J2K_SIGNPOST_BEGIN_CASE(J2K_SIGNPOST_ENCODE_HYBRID_HT_TIER1_SETUP,
            "j2k encode hybrid ht tier1 setup");
        J2K_SIGNPOST_BEGIN_CASE(J2K_SIGNPOST_ENCODE_HYBRID_HT_TIER1_COMMAND_ENCODE,
            "j2k encode hybrid ht tier1 command encode");
        J2K_SIGNPOST_BEGIN_CASE(J2K_SIGNPOST_ENCODE_HYBRID_HT_PACKET_PLAN,
            "j2k encode hybrid ht packet plan");
        J2K_SIGNPOST_BEGIN_CASE(J2K_SIGNPOST_ENCODE_HYBRID_HT_PACKET_BUFFER_SETUP,
            "j2k encode hybrid ht packet buffer setup");
        J2K_SIGNPOST_BEGIN_CASE(
            J2K_SIGNPOST_ENCODE_HYBRID_HT_PACKET_BLOCK_PREP_COMMAND_ENCODE,
            "j2k encode hybrid ht packet block prep command encode");
        J2K_SIGNPOST_BEGIN_CASE(J2K_SIGNPOST_ENCODE_HYBRID_HT_PACKETIZATION_COMMAND_ENCODE,
            "j2k encode hybrid ht packetization command encode");
        J2K_SIGNPOST_BEGIN_CASE(J2K_SIGNPOST_ENCODE_HYBRID_HT_PAYLOAD_COPY_COMMAND_ENCODE,
            "j2k encode hybrid ht payload copy command encode");
        J2K_SIGNPOST_BEGIN_CASE(J2K_SIGNPOST_ENCODE_HYBRID_HT_CODESTREAM_ASSEMBLY_COMMAND_ENCODE,
            "j2k encode hybrid ht codestream assembly command encode");
        default:
            return 0;
    }
}

void j2k_metal_signpost_end(uint32_t name_id, uint64_t raw_signpost_id) {
    if (raw_signpost_id == OS_SIGNPOST_ID_NULL || raw_signpost_id == OS_SIGNPOST_ID_INVALID) {
        return;
    }

    os_log_t log = j2k_metal_log();
    if (log == NULL) {
        return;
    }

    os_signpost_id_t signpost_id = (os_signpost_id_t)raw_signpost_id;
    switch (name_id) {
        J2K_SIGNPOST_END_CASE(J2K_SIGNPOST_DECODE_HYBRID_CPU_TIER1,
            "j2k decode hybrid cpu tier1");
        J2K_SIGNPOST_END_CASE(J2K_SIGNPOST_DECODE_HYBRID_COEFFICIENT_UPLOAD,
            "j2k decode hybrid coefficient upload");
        J2K_SIGNPOST_END_CASE(J2K_SIGNPOST_DECODE_HYBRID_COMMAND_WAIT,
            "j2k decode hybrid command wait");
        J2K_SIGNPOST_END_CASE(J2K_SIGNPOST_DECODE_HYBRID_IDWT_COMMAND_ENCODE,
            "j2k decode hybrid idwt command encode");
        J2K_SIGNPOST_END_CASE(J2K_SIGNPOST_DECODE_HYBRID_STORE_COMMAND_ENCODE,
            "j2k decode hybrid store command encode");
        J2K_SIGNPOST_END_CASE(J2K_SIGNPOST_DECODE_HYBRID_MCT_PACK_COMMAND_ENCODE,
            "j2k decode hybrid mct pack command encode");
        J2K_SIGNPOST_END_CASE(J2K_SIGNPOST_ENCODE_HYBRID_COMMAND_WAIT,
            "j2k encode hybrid command wait");
        J2K_SIGNPOST_END_CASE(J2K_SIGNPOST_ENCODE_HYBRID_RESULT_HARVEST,
            "j2k encode hybrid result harvest");
        J2K_SIGNPOST_END_CASE(J2K_SIGNPOST_ENCODE_HYBRID_CLASSIC_TIER1_SETUP,
            "j2k encode hybrid classic tier1 setup");
        J2K_SIGNPOST_END_CASE(J2K_SIGNPOST_ENCODE_HYBRID_CLASSIC_TIER1_COMMAND_ENCODE,
            "j2k encode hybrid classic tier1 command encode");
        J2K_SIGNPOST_END_CASE(J2K_SIGNPOST_ENCODE_HYBRID_CLASSIC_PACKET_PLAN,
            "j2k encode hybrid classic packet plan");
        J2K_SIGNPOST_END_CASE(J2K_SIGNPOST_ENCODE_HYBRID_CLASSIC_PACKET_BUFFER_SETUP,
            "j2k encode hybrid classic packet buffer setup");
        J2K_SIGNPOST_END_CASE(
            J2K_SIGNPOST_ENCODE_HYBRID_CLASSIC_PACKETIZATION_COMMAND_ENCODE,
            "j2k encode hybrid classic packetization command encode");
        J2K_SIGNPOST_END_CASE(J2K_SIGNPOST_ENCODE_HYBRID_CLASSIC_PAYLOAD_COPY_COMMAND_ENCODE,
            "j2k encode hybrid classic payload copy command encode");
        J2K_SIGNPOST_END_CASE(
            J2K_SIGNPOST_ENCODE_HYBRID_CLASSIC_CODESTREAM_ASSEMBLY_COMMAND_ENCODE,
            "j2k encode hybrid classic codestream assembly command encode");
        J2K_SIGNPOST_END_CASE(J2K_SIGNPOST_ENCODE_HYBRID_HT_TIER1_SETUP,
            "j2k encode hybrid ht tier1 setup");
        J2K_SIGNPOST_END_CASE(J2K_SIGNPOST_ENCODE_HYBRID_HT_TIER1_COMMAND_ENCODE,
            "j2k encode hybrid ht tier1 command encode");
        J2K_SIGNPOST_END_CASE(J2K_SIGNPOST_ENCODE_HYBRID_HT_PACKET_PLAN,
            "j2k encode hybrid ht packet plan");
        J2K_SIGNPOST_END_CASE(J2K_SIGNPOST_ENCODE_HYBRID_HT_PACKET_BUFFER_SETUP,
            "j2k encode hybrid ht packet buffer setup");
        J2K_SIGNPOST_END_CASE(
            J2K_SIGNPOST_ENCODE_HYBRID_HT_PACKET_BLOCK_PREP_COMMAND_ENCODE,
            "j2k encode hybrid ht packet block prep command encode");
        J2K_SIGNPOST_END_CASE(J2K_SIGNPOST_ENCODE_HYBRID_HT_PACKETIZATION_COMMAND_ENCODE,
            "j2k encode hybrid ht packetization command encode");
        J2K_SIGNPOST_END_CASE(J2K_SIGNPOST_ENCODE_HYBRID_HT_PAYLOAD_COPY_COMMAND_ENCODE,
            "j2k encode hybrid ht payload copy command encode");
        J2K_SIGNPOST_END_CASE(J2K_SIGNPOST_ENCODE_HYBRID_HT_CODESTREAM_ASSEMBLY_COMMAND_ENCODE,
            "j2k encode hybrid ht codestream assembly command encode");
        default:
            return;
    }
}
