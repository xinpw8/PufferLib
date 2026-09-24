#ifndef REK_NATIVE5_BC_DATASET_H
#define REK_NATIVE5_BC_DATASET_H
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <set>
#include <stdexcept>
#include <vector>

namespace rek_bc {
constexpr int OBS = 223, ACTIONS = 33, HEADER_BYTES = 256, ROW_BYTES = 1056;
struct Row {
    uint32_t split = 0, sequence = 0, reset = 0;
    int32_t action = -1;
    float weight = 0;
    double time = 0;
    std::array<float, OBS> obs{};
    // Vocabulary support, not an inferred native runtime legality mask.
    std::array<float, ACTIONS> support{};
};
struct Sequence { size_t begin, end; uint32_t split; };
struct Dataset {
    std::array<uint8_t, OBS> feature_mask{};
    std::vector<Row> rows;
    std::vector<Sequence> sequences;
};
inline void require(bool condition, const char* reason) {
    if (!condition) throw std::runtime_error(reason);
}
template<class T> T read_scalar(const uint8_t* p) {
    T result; std::memcpy(&result, p, sizeof(T)); return result;
}
inline void validate(Dataset& d) {
    require(!d.rows.empty(), "empty dataset");
    int kept = 0;
    for (uint8_t v : d.feature_mask) { require(v <= 1, "feature mask is not binary"); kept += v; }
    require(kept > 0, "empty feature mask");
    d.sequences.clear();
    std::set<uint32_t> seen;
    for (size_t i = 0; i < d.rows.size(); ++i) {
        const Row& r = d.rows[i];
        require(r.split <= 1 && r.reset <= 1, "invalid split/reset");
        require(std::isfinite(r.time) && r.time >= 0, "invalid timestamp");
        require(std::isfinite(r.weight) && r.weight >= 0 && r.weight <= 1000, "invalid label weight");
        require(r.action >= -1 && r.action < ACTIONS, "invalid action");
        require((r.action == -1) == (r.weight == 0), "unlabeled action/weight disagreement");
        const bool start = i == 0 || r.sequence != d.rows[i - 1].sequence;
        require(r.reset == static_cast<uint32_t>(start), "reset must mark true sequence start only");
        if (start) {
            require(seen.insert(r.sequence).second, "noncontiguous sequence");
            if (!d.sequences.empty()) d.sequences.back().end = i;
            d.sequences.push_back({i, d.rows.size(), r.split});
        } else {
            require(r.split == d.rows[i - 1].split, "sequence leaks across split");
            require(r.time > d.rows[i - 1].time, "nonincreasing sequence time");
        }
        for (float v : r.obs) require(std::isfinite(v), "nonfinite observation");
        int available = 0;
        for (float v : r.support) { require(v == 0 || v == 1, "nonbinary class support"); available += int(v); }
        require(available > 0, "empty class support");
        require(r.action < 0 || r.support[r.action] == 1, "target outside class support");
    }
}
inline Dataset parse(const std::vector<uint8_t>& bytes) {
    const uint32_t endian = 1;
    require(*reinterpret_cast<const uint8_t*>(&endian) == 1, "little endian host required");
    require(bytes.size() >= HEADER_BYTES, "truncated header");
    const auto* p = bytes.data();
    require(std::memcmp(p, "REKBC001", 8) == 0, "invalid dataset magic");
    require(read_scalar<uint32_t>(p + 8) == 1 && read_scalar<uint32_t>(p + 12) == OBS
        && read_scalar<uint32_t>(p + 16) == ACTIONS && read_scalar<uint32_t>(p + 24) == ROW_BYTES,
        "unsupported dataset layout");
    const uint32_t count = read_scalar<uint32_t>(p + 20);
    require(count > 0 && count <= 1000000, "invalid row count");
    require(bytes.size() == HEADER_BYTES + size_t(count) * ROW_BYTES, "truncated or trailing dataset bytes");
    require(read_scalar<uint32_t>(p + 28) == 0 && p[255] == 0, "nonzero reserved header");
    Dataset d;
    std::memcpy(d.feature_mask.data(), p + 32, OBS);
    d.rows.resize(count);
    for (size_t i = 0; i < count; ++i) {
        p = bytes.data() + HEADER_BYTES + i * ROW_BYTES;
        Row& r = d.rows[i];
        r.split = read_scalar<uint32_t>(p); r.sequence = read_scalar<uint32_t>(p + 4);
        r.reset = read_scalar<uint32_t>(p + 8); r.action = read_scalar<int32_t>(p + 12);
        r.weight = read_scalar<float>(p + 16); r.time = read_scalar<double>(p + 24);
        require(read_scalar<uint32_t>(p + 20) == 0, "nonzero reserved row");
        std::memcpy(r.obs.data(), p + 32, OBS * sizeof(float));
        std::memcpy(r.support.data(), p + 924, ACTIONS * sizeof(float));
    }
    validate(d);
    return d;
}
inline Dataset load(const char* path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    require(bool(f), "cannot open dataset");
    const auto n = f.tellg();
    require(n >= HEADER_BYTES && n <= HEADER_BYTES + int64_t(1000000) * ROW_BYTES, "invalid dataset size");
    std::vector<uint8_t> bytes(static_cast<size_t>(n));
    f.seekg(0); f.read(reinterpret_cast<char*>(bytes.data()), n);
    require(bool(f), "dataset read failed");
    return parse(bytes);
}
} // namespace rek_bc
#endif
