#include "policy_feature_mask.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace {
unsigned checks = 0;
void check(bool ok, const char* why) {
    ++checks;
    if (!ok) throw std::runtime_error(why);
}
bool joint(int i) { return (i >= 13 && i <= 70) || (i >= 99 && i <= 156); }
void write_new(const std::string& path, const std::vector<unsigned char>& bytes) {
    std::unique_ptr<FILE, decltype(&std::fclose)> file(std::fopen(path.c_str(), "wbx"), std::fclose);
    if (!file) throw std::runtime_error("fixture_exists_or_cannot_create");
    if (std::fwrite(bytes.data(), 1, bytes.size(), file.get()) != bytes.size() ||
        std::fflush(file.get()) != 0) throw std::runtime_error("fixture_write_failed");
}
}

int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::runtime_error("usage: joint-mask-cpu-test NEW_MASK NEW_ALL_ONES_MASK");
        std::vector<unsigned char> bytes(223, 1);
        for (int i = 0; i < 223; ++i) if (joint(i)) bytes[i] = 0;
        write_new(argv[1], bytes);
        write_new(argv[2], std::vector<unsigned char>(223, 1));
        const auto selected = rek_policy_features::load(argv[1]);
        const auto ones = rek_policy_features::load(argv[2]);
        const auto omitted = rek_policy_features::load(nullptr);
        check(selected.enabled && selected.sha256.size() == 64, "selected_mask_identity");
        check(!omitted.enabled && omitted.sha256.empty(), "omitted_mask_identity");
        check(std::count(selected.values.begin(), selected.values.end(), 1) == 107, "retained_count");
        for (int i = 0; i < 223; ++i) check(selected.values[i] == !joint(i), "exact_joint_ranges");
        for (int row = 0; row < 257; ++row) {
            std::array<float, 223> source{}, modified{};
            for (int i = 0; i < 223; ++i) {
                source[i] = i % 13 == 0 ? -0.0f : std::sin(float(row * 223 + i));
                modified[i] = joint(i) ? source[i] + 1000.0f : source[i];
            }
            auto masked = source, all_ones = source, no_mask = source;
            rek_policy_features::apply(masked.data(), selected);
            rek_policy_features::apply(modified.data(), selected);
            rek_policy_features::apply(all_ones.data(), ones);
            rek_policy_features::apply(no_mask.data(), omitted);
            check(!std::memcmp(source.data(), all_ones.data(), sizeof(source)), "all_ones_changes_input");
            check(!std::memcmp(source.data(), no_mask.data(), sizeof(source)), "omitted_changes_input");
            check(!std::memcmp(masked.data(), modified.data(), sizeof(source)), "excluded_inputs_leak");
            for (int i = 0; i < 223; ++i) {
                check(joint(i) ? masked[i] == 0.0f && !std::signbit(masked[i]) :
                    !std::memcmp(&masked[i], &source[i], sizeof(float)), "mask_changes_wrong_field");
            }
        }
        for (const auto& item : std::vector<std::pair<std::string, std::vector<unsigned char>>>{
                 {"short", std::vector<unsigned char>(222, 1)},
                 {"long", std::vector<unsigned char>(224, 1)},
                 {"nonbinary", std::vector<unsigned char>(223, 2)},
                 {"empty", std::vector<unsigned char>(223, 0)}}) {
            const std::string path = std::string(argv[1]) + "." + item.first;
            write_new(path, item.second);
            bool rejected = false;
            try { (void)rek_policy_features::load(path.c_str()); }
            catch (const std::runtime_error&) { rejected = true; }
            check(rejected, "malformed_mask_accepted");
        }
        std::printf("{\"passed\":true,\"checks\":%u,\"rows\":257,\"excluded\":116,\"retained\":107,\"mask_sha256\":\"%s\",\"all_ones_sha256\":\"%s\",\"gpu_invocations\":0}\n",
            checks, selected.sha256.c_str(), ones.sha256.c_str());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
