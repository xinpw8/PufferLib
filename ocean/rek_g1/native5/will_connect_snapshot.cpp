// CPU-only JSONL diagnostic for archived native snapshots. No simulator/input IO.
#include "will_connect_json.h"
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <limits>

namespace {
using Json = std::unique_ptr<cJSON, decltype(&cJSON_Delete)>;
const cJSON* field(const cJSON* j, const char* key) {
    return cJSON_GetObjectItemCaseSensitive(j, key);
}
template<size_t N> void floats(const cJSON* j, const char* key, float (&out)[N]) {
    const cJSON* a = field(j, key);
    if (!cJSON_IsArray(a) || cJSON_GetArraySize(a) != int(N))
        throw std::runtime_error(std::string("Wrong array length: ") + key);
    for (size_t i = 0; i < N; ++i) {
        const cJSON* v = cJSON_GetArrayItem(a, int(i));
        if (!cJSON_IsNumber(v) || !std::isfinite(v->valuedouble) ||
            std::fabs(v->valuedouble) > std::numeric_limits<float>::max())
            throw std::runtime_error(std::string("Nonfinite/non-numeric input: ") + key);
        out[i] = float(v->valuedouble);
    }
}
int integer(const cJSON* j, const char* key, int maximum) {
    const cJSON* v = field(j, key);
    if (!cJSON_IsNumber(v) || !std::isfinite(v->valuedouble) ||
        v->valuedouble < 0 || v->valuedouble > maximum || v->valuedouble != v->valueint)
        throw std::runtime_error(std::string("Invalid integer: ") + key);
    return v->valueint;
}
}
int main() {
    try {
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.size() > 65536) throw std::runtime_error("Snapshot line exceeds limit");
            Json input(cJSON_ParseWithLengthOpts(line.c_str(), line.size() + 1, nullptr, 1), cJSON_Delete);
            if (!input || !cJSON_IsObject(input.get())) throw std::runtime_error("Invalid snapshot JSON");
            float p[72], v[70], mask_values[66]; uint8_t masks[66];
            floats(input.get(), "qpos", p); floats(input.get(), "qvel", v);
            floats(input.get(), "mask", mask_values);
            for (int k = 0; k < 66; ++k) {
                if (mask_values[k] != 0 && mask_values[k] != 1) throw std::runtime_error("Invalid action mask");
                masks[k] = uint8_t(mask_values[k]);
            }
            const int phase = integer(input.get(), "phase", REK_G1_FIGHT_OVER);
            const int terminal = integer(input.get(), "terminal", 1);
            const int failure = integer(input.get(), "failureBits", 2147483647);
            const auto diagnostic = rek5_will_connect::evaluate(p, v, masks, phase, terminal != 0, uint32_t(failure));
            Json output(rek5_will_connect::json(diagnostic), cJSON_Delete);
            char* serialized = cJSON_PrintUnformatted(output.get());
            if (!serialized) throw std::runtime_error("JSON allocation failed");
            std::cout << serialized << '\n'; cJSON_free(serialized);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
