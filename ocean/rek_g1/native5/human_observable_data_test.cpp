#define REK_HUMAN_OBSERVABLE_NO_MAIN
#include "human_observable_data.cpp"

namespace {
int checks = 0;
void check(bool ok) { ++checks; rek_bc::require(ok, "human observable test failed"); }
template<class F> void rejects(F f) { bool failed = false; try { f(); } catch (const std::exception&) { failed = true; } check(failed); }
template<class T> void put(human_observable::Bytes& bytes, size_t offset, T value) { std::memcpy(bytes.data() + offset, &value, sizeof(value)); }
}
int main() {
    using namespace human_observable;
    try {
        Bytes original(256 + 3 * 1056); std::memcpy(original.data(), "REKBC001", 8);
        for (const auto& pair : {std::pair<int, uint32_t>{8,1}, {12,223}, {16,33}, {20,3}, {24,1056}}) put(original, pair.first, pair.second);
        std::fill(original.begin() + 32, original.begin() + 255, 1);
        for (int i = 0; i < 3; ++i) {
            const size_t p = 256 + i * 1056; put(original, p, uint32_t(i == 2)); put(original, p + 4, uint32_t(i == 2));
            put(original, p + 8, uint32_t(i != 1)); put(original, p + 12, int32_t(i == 1 ? -1 : 17));
            put(original, p + 16, i == 1 ? 0.f : 1.f); put(original, p + 24, i == 1 ? .02 : 0.);
            for (int c = 0; c < 223; ++c) put(original, p + 32 + 4 * c, 999.f);
            for (int c = 0; c < 33; ++c) put(original, p + 924 + 4 * c, float(c != 32));
        }
        auto data = rek_bc::parse(original);
        const std::string sample_text = R"({"event":"sample","sample_index":0,"client_fixed_tick":0,"unity_unscaled_time":100,"fight_epoch":0,"phase_value":1,"local_fighter_index":0,"round":{"number":1,"active":true,"redo":false,"result_value":0,"duration":120,"time_remaining":100,"clean_hits":[0,0]},"fighter_0":{"root_position":[0,1,0],"root_rotation":[0.70710678,0,0,-0.70710678]},"fighter_1":{"root_position":[1,1,0],"root_rotation":[0,0,0,-1]}})";
        auto sample = parse(sample_text);
        std::vector<Binding> bindings{{10,0,0,1,0,100}, {20,1,10,1,0,100.02}, {10,0,0,1,0,100}};
        std::vector<balance::Snapshot> snapshots(3, snapshot(sample.get(), bindings[0], 1));
        snapshots[1].sample_seconds = 100.02; snapshots[1].fighter[0].root_xyz[2] += .02f; snapshots[1].points[0] = 2;
        snapshots[2].round_key = 2; snapshots[2].points[0] = 7;
        auto output = reproject(original, data, bindings, snapshots); auto migrated = rek_bc::parse(output);
        check(std::count(output.begin() + 32, output.begin() + 255, uint8_t(1)) == 166);
        for (int i = 0; i < 3; ++i) {
            const size_t p = 256 + i * 1056;
            check(std::memcmp(original.data() + p, output.data() + p, 32) == 0);
            check(std::memcmp(original.data() + p + 924, output.data() + p + 924, 132) == 0);
            const auto& obs = migrated.rows[i].obs;
            for (int c = 0; c < 223; ++c) { check(std::isfinite(obs[c]) && obs[c] != 999.f); if (!balance::structurally_available(c)) check(obs[c] == 0.f); }
            for (int base : {0,86}) {
                for (int c = 13; c <= 70; ++c) check(obs[base + c] == 0);
                check(obs[base + 74] == 0 && obs[base + 75] == 0);
            }
            check(obs[202] == 0 && obs[204] == 0 && obs[205] == 0);
            check(obs[203] == float(i == 1)); check(std::abs(obs[72] - .5f) < 1e-6f);
        }
        check(std::abs(migrated.rows[1].obs[9] - 1.f) < 2e-6f);
        check(migrated.rows[1].obs[217] == 2.f);
        check(migrated.rows[0].obs[9] == 0 && migrated.rows[2].obs[217] == 0);
        auto poisoned = original;
        for (int i = 0; i < 3; ++i) for (int c = 0; c < 223; ++c) put(poisoned, 256 + i * 1056 + 32 + 4 * c, float(c - 300));
        check(reproject(poisoned, rek_bc::parse(poisoned), bindings, snapshots) == output);
        auto broken = snapshots; broken[1].points[0] = -1; rejects([&] { reproject(original, data, bindings, broken); });
        broken = snapshots; broken[0].points[0] = 3; rejects([&] { reproject(original, data, bindings, broken); });
        auto bad_bindings = bindings; bad_bindings[1].fixed_tick++; rejects([&] { reproject(original, data, bad_bindings, snapshots); });
        bad_bindings = bindings; bad_bindings[0].sample++; rejects([&] { snapshot(sample.get(), bad_bindings[0], 1); });
        broken = snapshots; broken[1].sample_seconds = 101; rejects([&] { reproject(original, data, bindings, broken); });
        std::ostringstream ledger;
        for (int i = 0; i < 3; ++i) ledger << "{\"split\":" << (i == 2) << ",\"sequence_id\":" << (i == 2)
            << ",\"reset_before\":" << (i != 1) << ",\"action\":" << (i == 1 ? -1 : 17) << ",\"weight\":" << (i != 1)
            << ",\"time\":" << (i == 1 ? .02 : 0.) << ",\"source_line\":" << bindings[i].source_line
            << ",\"sample_index\":" << bindings[i].sample << ",\"client_fixed_tick\":" << bindings[i].fixed_tick
            << ",\"round_number\":1,\"fight_epoch\":0,\"unscaled_time\":100}\n";
        const auto ledger_text = ledger.str(); Bytes ledger_bytes(ledger_text.begin(), ledger_text.end());
        check(bind_ledger(ledger_bytes, data).size() == 3);
        auto wrong_labels = data; wrong_labels.rows[0].action = 18; rejects([&] { bind_ledger(ledger_bytes, wrong_labels); });
        std::cout << "{\"native_cpu_adapter_tests\":\"passed\",\"checks\":" << checks << ",\"stale_features_independent\":true,\"physics\":false,\"gpu\":false}\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 2; }
}
