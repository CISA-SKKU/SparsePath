// rdma_common.h
// Common RDMA utilities and error handling
#pragma once

#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <stdexcept>
#include <iostream>
#include <vector>
#include <memory>
#include <atomic>

namespace rdma {

// Configuration constants
constexpr uint32_t DEFAULT_PORT = 18515;
constexpr uint32_t DEFAULT_GID_INDEX = 3;  // RoCE v2 with IPv4
// SQ depth is a QUEUE length, not parallelism: in-flight READs per QP stay
// capped by ORD (max_rd_atomic=16, CX-6 HW limit). A deep SQ lets software
// post a whole interleaved super-hop (factor x (slack_R+1) WRs, e.g. 4x85=340)
// without blocking in the drain loop; the NIC drains it 16 at a time.
// 128 -> 512 (2026-06-12): interleave q4 demand 260 > 128 serialized posting
// and regressed below q1. CQ must hold one CQE per outstanding signaled WR,
// so it scales with SQ depth (send side; 2x margin).
constexpr uint32_t DEFAULT_SQ_DEPTH = 512;
constexpr uint32_t DEFAULT_RQ_DEPTH = 128;
constexpr uint32_t DEFAULT_CQ_SIZE = 1024;
constexpr uint32_t DEFAULT_MAX_SGE = 30;  // NVIDIA ConnectX/BlueField supports 30+
constexpr uint32_t DEFAULT_MAX_INLINE = 256;

// RDMA operation types
enum class OpType : uint8_t {
    SEND = 0,
    RECV = 1,
    READ = 2,
    WRITE = 3,
    CAS = 4,
    FAA = 5
};

// Message types for protocol
enum class MsgType : uint8_t {
    CONN_REQ = 0,      // Connection request
    CONN_ACK = 1,      // Connection acknowledgment
    MR_INFO = 2,       // Memory region info exchange
    MR_INFO_ACK = 3,   // MR info acknowledgment
    VECTOR_REQ = 4,    // Vector fetch request
    VECTOR_RESP = 5,   // Vector fetch response
    PING = 6,          // Ping
    PONG = 7,          // Pong
    DISCONNECT = 8     // Disconnect
};

// Remote memory region info (for one-sided operations)
struct RemoteMemoryRegion {
    uint64_t addr;      // Remote virtual address
    uint32_t rkey;      // Remote key for RDMA access
    uint64_t size;      // Size of the region

    RemoteMemoryRegion() : addr(0), rkey(0), size(0) {}
    RemoteMemoryRegion(uint64_t a, uint32_t r, uint64_t s)
        : addr(a), rkey(r), size(s) {}
};

// Message header for protocol messages
struct MessageHeader {
    MsgType type;
    uint32_t payload_size;
    uint64_t request_id;

    MessageHeader() : type(MsgType::PING), payload_size(0), request_id(0) {}
    MessageHeader(MsgType t, uint32_t size, uint64_t id)
        : type(t), payload_size(size), request_id(id) {}
};

// Memory region info message payload
struct MRInfoPayload {
    RemoteMemoryRegion mr;
    uint32_t num_vectors;
    uint32_t vector_dim;

    MRInfoPayload() : num_vectors(0), vector_dim(0) {}
};

// Vector fetch request payload
struct VectorFetchRequest {
    uint32_t vector_id;       // Which vector to fetch
    uint32_t num_vectors;     // How many consecutive vectors

    VectorFetchRequest() : vector_id(0), num_vectors(1) {}
    VectorFetchRequest(uint32_t id, uint32_t n = 1) : vector_id(id), num_vectors(n) {}
};

// SGL entry for scatter-gather operations
struct SGLEntry {
    void* local_addr;         // Local buffer address
    uint32_t length;          // Size of this segment in bytes
    uint64_t remote_offset;   // Offset from remote MR base address

    SGLEntry() : local_addr(nullptr), length(0), remote_offset(0) {}
    SGLEntry(void* addr, uint32_t len, uint64_t offset = 0)
        : local_addr(addr), length(len), remote_offset(offset) {}
};

// Device capabilities for advanced features
struct DeviceCapabilities {
    uint32_t max_sge;              // Maximum SGE per work request
    uint32_t max_qp_wr;            // Maximum outstanding WRs per QP
    uint64_t max_mr_size;          // Maximum memory region size
    bool odp_supported;            // On-Demand Paging support
    bool odp_implicit;             // Implicit ODP support
    bool umr_supported;            // User Memory Region (mlx5) support

    DeviceCapabilities()
        : max_sge(1), max_qp_wr(128), max_mr_size(0),
          odp_supported(false), odp_implicit(false), umr_supported(false) {}

    void print() const {
        fprintf(stderr, "[INFO] Device Capabilities:\n");
        fprintf(stderr, "[INFO]   max_sge: %u\n", max_sge);
        fprintf(stderr, "[INFO]   max_qp_wr: %u\n", max_qp_wr);
        fprintf(stderr, "[INFO]   max_mr_size: %lu\n", max_mr_size);
        fprintf(stderr, "[INFO]   ODP supported: %s\n", odp_supported ? "yes" : "no");
        fprintf(stderr, "[INFO]   ODP implicit: %s\n", odp_implicit ? "yes" : "no");
        fprintf(stderr, "[INFO]   UMR supported: %s\n", umr_supported ? "yes" : "no");
    }
};

// RDMA exception class
class RDMAException : public std::runtime_error {
public:
    explicit RDMAException(const std::string& msg)
        : std::runtime_error("RDMA Error: " + msg) {}

    RDMAException(const std::string& msg, int err)
        : std::runtime_error("RDMA Error: " + msg + " (errno=" +
                            std::to_string(err) + ": " + strerror(err) + ")") {}
};

// Helper macros for error checking
#define RDMA_CHECK(cond, msg) \
    do { \
        if (!(cond)) { \
            throw RDMAException(msg, errno); \
        } \
    } while(0)

#define RDMA_CHECK_NULL(ptr, msg) \
    do { \
        if ((ptr) == nullptr) { \
            throw RDMAException(msg, errno); \
        } \
    } while(0)

// Utility functions
inline std::string gid_to_string(const union ibv_gid& gid) {
    char buf[64];
    snprintf(buf, sizeof(buf),
             "%02x%02x:%02x%02x:%02x%02x:%02x%02x:"
             "%02x%02x:%02x%02x:%02x%02x:%02x%02x",
             gid.raw[0], gid.raw[1], gid.raw[2], gid.raw[3],
             gid.raw[4], gid.raw[5], gid.raw[6], gid.raw[7],
             gid.raw[8], gid.raw[9], gid.raw[10], gid.raw[11],
             gid.raw[12], gid.raw[13], gid.raw[14], gid.raw[15]);
    return std::string(buf);
}

inline void string_to_gid(const std::string& str, union ibv_gid& gid) {
    sscanf(str.c_str(),
           "%02hhx%02hhx:%02hhx%02hhx:%02hhx%02hhx:%02hhx%02hhx:"
           "%02hhx%02hhx:%02hhx%02hhx:%02hhx%02hhx:%02hhx%02hhx",
           &gid.raw[0], &gid.raw[1], &gid.raw[2], &gid.raw[3],
           &gid.raw[4], &gid.raw[5], &gid.raw[6], &gid.raw[7],
           &gid.raw[8], &gid.raw[9], &gid.raw[10], &gid.raw[11],
           &gid.raw[12], &gid.raw[13], &gid.raw[14], &gid.raw[15]);
}

// QP state to string
inline const char* qp_state_to_string(enum ibv_qp_state state) {
    switch(state) {
        case IBV_QPS_RESET: return "RESET";
        case IBV_QPS_INIT: return "INIT";
        case IBV_QPS_RTR: return "RTR";
        case IBV_QPS_RTS: return "RTS";
        case IBV_QPS_SQD: return "SQD";
        case IBV_QPS_SQE: return "SQE";
        case IBV_QPS_ERR: return "ERR";
        default: return "UNKNOWN";
    }
}

// Work completion status to string
inline const char* wc_status_to_string(enum ibv_wc_status status) {
    switch(status) {
        case IBV_WC_SUCCESS: return "SUCCESS";
        case IBV_WC_LOC_LEN_ERR: return "LOCAL_LENGTH_ERROR";
        case IBV_WC_LOC_QP_OP_ERR: return "LOCAL_QP_OP_ERROR";
        case IBV_WC_LOC_EEC_OP_ERR: return "LOCAL_EEC_OP_ERROR";
        case IBV_WC_LOC_PROT_ERR: return "LOCAL_PROTECTION_ERROR";
        case IBV_WC_WR_FLUSH_ERR: return "WR_FLUSH_ERROR";
        case IBV_WC_MW_BIND_ERR: return "MW_BIND_ERROR";
        case IBV_WC_BAD_RESP_ERR: return "BAD_RESPONSE_ERROR";
        case IBV_WC_LOC_ACCESS_ERR: return "LOCAL_ACCESS_ERROR";
        case IBV_WC_REM_INV_REQ_ERR: return "REMOTE_INVALID_REQUEST_ERROR";
        case IBV_WC_REM_ACCESS_ERR: return "REMOTE_ACCESS_ERROR";
        case IBV_WC_REM_OP_ERR: return "REMOTE_OP_ERROR";
        case IBV_WC_RETRY_EXC_ERR: return "RETRY_EXCEEDED_ERROR";
        case IBV_WC_RNR_RETRY_EXC_ERR: return "RNR_RETRY_EXCEEDED_ERROR";
        case IBV_WC_LOC_RDD_VIOL_ERR: return "LOCAL_RDD_VIOLATION_ERROR";
        case IBV_WC_REM_INV_RD_REQ_ERR: return "REMOTE_INVALID_RD_REQUEST_ERROR";
        case IBV_WC_REM_ABORT_ERR: return "REMOTE_ABORT_ERROR";
        case IBV_WC_INV_EECN_ERR: return "INVALID_EECN_ERROR";
        case IBV_WC_INV_EEC_STATE_ERR: return "INVALID_EEC_STATE_ERROR";
        case IBV_WC_FATAL_ERR: return "FATAL_ERROR";
        case IBV_WC_RESP_TIMEOUT_ERR: return "RESPONSE_TIMEOUT_ERROR";
        case IBV_WC_GENERAL_ERR: return "GENERAL_ERROR";
        default: return "UNKNOWN";
    }
}

// Simple logging
class Logger {
public:
    enum Level { DEBUG = 0, INFO = 1, WARN = 2, ERROR = 3 };

    static void set_level(Level l) { level_ = l; }

    template<typename... Args>
    static void debug(const char* fmt, Args... args) {
        if (level_ <= DEBUG) log("DEBUG", fmt, args...);
    }

    template<typename... Args>
    static void info(const char* fmt, Args... args) {
        if (level_ <= INFO) log("INFO", fmt, args...);
    }

    template<typename... Args>
    static void warn(const char* fmt, Args... args) {
        if (level_ <= WARN) log("WARN", fmt, args...);
    }

    template<typename... Args>
    static void error(const char* fmt, Args... args) {
        if (level_ <= ERROR) log("ERROR", fmt, args...);
    }

private:
    static inline Level level_ = INFO;

    // Overload for no format arguments - avoids -Wformat-security warning
    static void log(const char* level, const char* msg) {
        fprintf(stderr, "[%s] %s\n", level, msg);
    }

    template<typename... Args>
    static void log(const char* level, const char* fmt, Args... args) {
        fprintf(stderr, "[%s] ", level);
        fprintf(stderr, fmt, args...);
        fprintf(stderr, "\n");
    }
};

} // namespace rdma
