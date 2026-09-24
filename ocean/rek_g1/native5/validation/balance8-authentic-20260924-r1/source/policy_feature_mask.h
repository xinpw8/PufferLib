#ifndef REK_POLICY_FEATURE_MASK_H
#define REK_POLICY_FEATURE_MASK_H
#include <array>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <openssl/sha.h>

namespace rek_policy_features {
constexpr int count = 223;
struct Mask {
    std::array<unsigned char, count> values{};
    std::string sha256;
    bool enabled = false;
    Mask() { values.fill(1); }
};
// Missing historical features are removed identically in BC, PPO and live
// inference. Zeroed inputs are excluded features, not measured zero values.
inline Mask load(const char* path) {
    Mask result;
    if (!path || !*path) return result;
    std::unique_ptr<FILE, decltype(&std::fclose)> file(std::fopen(path, "rb"), std::fclose);
    if (!file) throw std::runtime_error("feature_mask_open_failed");
    if (std::fread(result.values.data(), 1, count, file.get()) != count ||
        std::fgetc(file.get()) != EOF || std::ferror(file.get()))
        throw std::runtime_error("feature_mask_requires_exactly_223_bytes");
    int retained = 0;
    for (auto value : result.values) {
        if (value > 1) throw std::runtime_error("feature_mask_requires_binary_bytes");
        retained += value;
    }
    if (!retained) throw std::runtime_error("feature_mask_cannot_exclude_all_inputs");
    unsigned char hash[SHA256_DIGEST_LENGTH];
    if (!SHA256(result.values.data(), count, hash))
        throw std::runtime_error("feature_mask_hash_failed");
    constexpr char hex[] = "0123456789abcdef";
    for (auto value : hash) { result.sha256 += hex[value >> 4]; result.sha256 += hex[value & 15]; }
    result.enabled = true;
    return result;
}
inline void apply(float* observation, const Mask& mask) {
    for (int i = 0; i < count; ++i) if (!mask.values[i]) observation[i] = 0.0f;
}
}
#endif
