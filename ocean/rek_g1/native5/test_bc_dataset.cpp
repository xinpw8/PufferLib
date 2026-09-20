#include "bc_dataset.h"
#include <functional>
#include <iostream>
#include <limits>
using namespace rek_bc;
template<class T> void put(std::vector<uint8_t>& b, size_t offset, T v) { std::memcpy(b.data() + offset, &v, sizeof(v)); }
int main() {
    int checks = 0;
    std::vector<uint8_t> good(HEADER_BYTES + 4 * ROW_BYTES);
    std::memcpy(good.data(), "REKBC001", 8);
    put<uint32_t>(good, 8, 1); put<uint32_t>(good, 12, OBS); put<uint32_t>(good, 16, ACTIONS);
    put<uint32_t>(good, 20, 4); put<uint32_t>(good, 24, ROW_BYTES);
    std::fill(good.begin() + 32, good.begin() + 255, 1);
    for (size_t i = 0; i < 4; ++i) {
        const auto p = HEADER_BYTES + i * ROW_BYTES;
        put<uint32_t>(good, p, i / 2); put<uint32_t>(good, p + 4, i / 2);
        put<uint32_t>(good, p + 8, i % 2 == 0); put<int32_t>(good, p + 12, i % 2 ? 2 : -1);
        put<float>(good, p + 16, i % 2 ? 1 : 0); put<double>(good, p + 24, (i % 2) * .02);
        for (int j = 0; j < ACTIONS; ++j) put<float>(good, p + 924 + 4 * j, 1);
    }
    const auto d = parse(good); require(d.sequences.size() == 2 && d.rows[1].action == 2, "valid parse failed"); ++checks;
    auto rejects = [&](const std::function<void(std::vector<uint8_t>&)>& mutate) {
        auto b = good; mutate(b); bool rejected = false;
        try { parse(b); } catch (const std::exception&) { rejected = true; }
        require(rejected, "invalid input accepted"); ++checks;
    };
    for (size_t size = 0; size < good.size(); ++size) rejects([&](auto& b) { b.resize(size); });
    rejects([](auto& b) { b.push_back(0); });
    rejects([](auto& b) { b[0] = 0; });
    rejects([](auto& b) { b[32] = 2; });
    rejects([](auto& b) { b[255] = 1; });
    rejects([](auto& b) { put<uint32_t>(b, 24, ROW_BYTES - 1); });
    rejects([](auto& b) { put<uint32_t>(b, HEADER_BYTES + ROW_BYTES, 1); });
    rejects([](auto& b) { put<uint32_t>(b, HEADER_BYTES + 8, 0); });
    rejects([](auto& b) { put<uint32_t>(b, HEADER_BYTES + ROW_BYTES + 8, 1); });
    rejects([](auto& b) { put<double>(b, HEADER_BYTES + ROW_BYTES + 24, 0); });
    rejects([](auto& b) { put<int32_t>(b, HEADER_BYTES + ROW_BYTES + 12, 33); });
    rejects([](auto& b) { put<float>(b, HEADER_BYTES + ROW_BYTES + 16, 0); });
    rejects([](auto& b) { put<float>(b, HEADER_BYTES + ROW_BYTES + 924 + 8, 0); });
    rejects([](auto& b) { put<float>(b, HEADER_BYTES + 32, std::numeric_limits<float>::quiet_NaN()); });
    rejects([](auto& b) { put<float>(b, HEADER_BYTES + 924, .5f); });
    rejects([](auto& b) { put<uint32_t>(b, HEADER_BYTES + 20, 1); });
    std::cout << "{\"dataset_checks\":" << checks << ",\"passed\":true}\n";
}
