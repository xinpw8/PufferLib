// Offline schema migration of two pinned human recordings. No policy or physics.
#include "observable_balance.h"
#include "bc_dataset.h"
#include "../../../vendor/cJSON.h"
#include <openssl/evp.h>
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

namespace human_observable {
namespace balance = rek_observable_balance;
namespace fs = std::filesystem;
using Json = std::unique_ptr<cJSON, decltype(&cJSON_Delete)>;
using Bytes = std::vector<uint8_t>;
using rek_bc::require;
constexpr const char* RAW_HASHES[] = {
    "547cec42f700e97b9594f8d2c6df88f966052c0e0e5ce7395c4862b619fb6d7f",
    "ec55c32a6e2272a8e7656d73260ca35876be2cd6c8d8d6fe8c9dfe77cd25a309"};
constexpr const char* DATA_HASH = "71238da150db6e926170efdc27662f4426145aac7f69a7c93f619ece54d6cf1d";
constexpr const char* MANIFEST_HASH = "eb69ee0bfc75b5a3a0c4fe00a57c94bcf295884ea18240b98839052ad55d12a8";
constexpr const char* ROW_HASH = "3f24a13f09a145d41f6c3c33d1da1fec0ca17b72d9c3c370cad668b55938ac83";
constexpr const char* COMMAND_HASH = "8a1f6f625bbb1ed50452c705ad7dba65642357428120a0ecf3c5cbb6f1694797";

std::string sha(const void* data, size_t size) {
    unsigned char hash[32]; unsigned length = 0;
    require(EVP_Digest(data, size, hash, &length, EVP_sha256(), nullptr) == 1 && length == 32, "SHA256 failed");
    std::string out; constexpr char hex[] = "0123456789abcdef";
    for (unsigned char b : hash) { out += hex[b >> 4]; out += hex[b & 15]; }
    return out;
}
std::string sha(const Bytes& data) { return sha(data.data(), data.size()); }
Bytes read(const fs::path& file) {
    std::ifstream in(file, std::ios::binary | std::ios::ate);
    require(bool(in), "cannot read input"); const auto size = in.tellg();
    require(size > 0 && size <= 100000000, "input size outside bounded capture scope");
    Bytes out(static_cast<size_t>(size)); in.seekg(0);
    in.read(reinterpret_cast<char*>(out.data()), size); require(bool(in), "input read failed"); return out;
}
Bytes pinned(const fs::path& file, const char* expected) {
    auto bytes = read(file); require(sha(bytes) == expected, "pinned input SHA256 mismatch"); return bytes;
}
Json parse(const std::string& line) {
    Json out(cJSON_ParseWithLengthOpts(line.c_str(), line.size() + 1, nullptr, 1), cJSON_Delete);
    require(out && cJSON_IsObject(out.get()), "invalid JSON object"); return out;
}
const cJSON* get(const cJSON* j, const char* key) {
    const auto* out = cJSON_GetObjectItemCaseSensitive(j, key); require(out, "missing JSON field"); return out;
}
double number(const cJSON* j) { require(cJSON_IsNumber(j) && std::isfinite(j->valuedouble), "invalid finite number"); return j->valuedouble; }
int integer(const cJSON* j, int low = 0, int high = 100000000) {
    const double n = number(j); require(n >= low && n <= high && n == std::floor(n), "invalid integer"); return int(n);
}
bool boolean(const cJSON* j) { require(cJSON_IsBool(j), "invalid boolean"); return cJSON_IsTrue(j); }
std::string string(const cJSON* j) { require(cJSON_IsString(j) && j->valuestring, "invalid string"); return j->valuestring; }
const cJSON* array(const cJSON* j, int size) { require(cJSON_IsArray(j) && cJSON_GetArraySize(j) == size, "invalid array size"); return j; }
std::string json_string(const cJSON* j) { char* p = cJSON_PrintUnformatted(j); require(p, "JSON print failed"); std::string out(p); cJSON_free(p); return out; }
void add(cJSON* j, const char* key, const std::string& value) { cJSON_AddStringToObject(j, key, value.c_str()); }
template<class Visit> void lines(const Bytes& bytes, Visit visit) {
    std::istringstream input(std::string(bytes.begin(), bytes.end())); std::string line; size_t index = 0;
    while (std::getline(input, line)) { ++index; require(!line.empty(), "unexpected empty source line"); auto j = parse(line); visit(j.get(), index); }
}

struct Binding { int source_line, sample, fixed_tick, round, epoch; double seconds; };
std::vector<Binding> bind_ledger(const Bytes& bytes, const rek_bc::Dataset& data) {
    std::vector<Binding> out;
    lines(bytes, [&](const cJSON* j, size_t) {
        require(out.size() < data.rows.size(), "too many ledger rows"); const auto& r = data.rows[out.size()];
        require(integer(get(j, "split"), 0, 1) == int(r.split) && integer(get(j, "sequence_id")) == int(r.sequence) &&
            integer(get(j, "reset_before"), 0, 1) == int(r.reset) && integer(get(j, "action"), -1, 32) == r.action &&
            number(get(j, "weight")) == r.weight && number(get(j, "time")) == r.time, "ledger row metadata mismatch");
        Binding b{integer(get(j, "source_line"), 1), integer(get(j, "sample_index")), integer(get(j, "client_fixed_tick")),
            integer(get(j, "round_number")), integer(get(j, "fight_epoch")), number(get(j, "unscaled_time"))};
        if (!out.empty() && r.split == data.rows[out.size() - 1].split) require(b.source_line > out.back().source_line, "source order regressed");
        out.push_back(b);
    });
    require(out.size() == data.rows.size(), "missing ledger rows"); return out;
}

balance::Snapshot snapshot(const cJSON* s, const Binding& b, uint64_t key) {
    require(string(get(s, "event")) == "sample" && integer(get(s, "sample_index")) == b.sample &&
        integer(get(s, "client_fixed_tick")) == b.fixed_tick && number(get(s, "unity_unscaled_time")) == b.seconds &&
        integer(get(s, "fight_epoch")) == b.epoch, "raw sample identity mismatch");
    const auto* r = get(s, "round");
    require(integer(get(r, "number")) == b.round && boolean(get(r, "active")) && !boolean(get(r, "redo")) &&
        integer(get(r, "result_value")) == 0 && integer(get(s, "phase_value")) == 1 && integer(get(s, "local_fighter_index")) == 0,
        "sample round/phase/actor mismatch");
    balance::Snapshot out{}; out.round_key = key; out.sample_seconds = b.seconds; out.actor_slot = 0; out.round_active = 1;
    out.round_duration_seconds = float(number(get(r, "duration"))); out.round_remaining_seconds = float(number(get(r, "time_remaining")));
    const auto* points = array(get(r, "clean_hits"), 2);
    for (int slot = 0; slot < 2; ++slot) {
        out.points[slot] = integer(cJSON_GetArrayItem(points, slot), 0, 1000000);
        const auto* fighter = get(s, slot ? "fighter_1" : "fighter_0");
        const auto* p = array(get(fighter, "root_position"), 3); const auto* q = array(get(fighter, "root_rotation"), 4);
        float xyz[3], xyzw[4];
        for (int k = 0; k < 3; ++k) xyz[k] = float(number(cJSON_GetArrayItem(p, k)));
        for (int k = 0; k < 4; ++k) xyzw[k] = float(number(cJSON_GetArrayItem(q, k)));
        require(balance::from_unity_root(xyz, xyzw, out.fighter[slot]), "invalid root pose");
        out.fighter[slot].joint_pose_available = 0;
    }
    // V5 has raw referee bytes, but cannot prove the current QPC/lifecycle receipt
    // contract. Zero is unavailable padding, never a guessed inactive count.
    out.referee_available = 0; out.count_mask = 0; return out;
}
std::vector<balance::Snapshot> load_snapshots(const std::array<Bytes, 2>& captures,
        const std::vector<Binding>& bindings, const rek_bc::Dataset& data) {
    std::vector<balance::Snapshot> out(data.rows.size()); size_t cursor = 0;
    for (int split = 0; split < 2; ++split) {
        bool started = false, ended = false; int samples = 0;
        lines(captures[split], [&](const cJSON* j, size_t line) {
            const auto event = string(get(j, "event"));
            if (event == "capture_start") {
                require(!started && line == 1 && string(get(j, "schema")) == "rek.private_ai.protocol.v5", "invalid capture start");
                const auto* scope = get(j, "scope");
                require(boolean(get(scope, "allowed")) && integer(get(scope, "local_fighter_index")) == 0 &&
                    boolean(get(scope, "opponent_is_ai")) && !boolean(get(scope, "human_in_opponent_slot")), "invalid capture scope");
                started = true; return;
            }
            require(started && !ended && event != "capture_error", "record outside valid capture");
            if (event == "sample") ++samples;
            if (cursor < data.rows.size() && int(data.rows[cursor].split) == split && size_t(bindings[cursor].source_line) == line) {
                out[cursor] = snapshot(j, bindings[cursor], uint64_t(split + 1)); ++cursor;
            }
            if (event == "capture_end") {
                require(integer(get(j, "capture_error_count")) == 0 && integer(get(j, "sample_count")) == samples, "invalid capture end"); ended = true;
            }
        });
        require(started && ended && samples == 5998, "incomplete pinned capture");
        require(cursor == data.rows.size() || int(data.rows[cursor].split) > split, "referenced source line missing");
    }
    require(cursor == data.rows.size(), "missing source samples"); return out;
}

Bytes reproject(const Bytes& original, const rek_bc::Dataset& data,
        const std::vector<Binding>& bindings, const std::vector<balance::Snapshot>& samples) {
    require(data.rows.size() == samples.size() && samples.size() == bindings.size(), "projection row count mismatch");
    Bytes out = original; balance::feature_mask(out.data() + 32);
    for (size_t i = 0; i < data.rows.size(); ++i) {
        const auto& row = data.rows[i]; const auto* previous = row.reset ? nullptr : &samples[i - 1];
        if (previous) require(bindings[i].sample == bindings[i - 1].sample + 1 &&
            bindings[i].fixed_tick == bindings[i - 1].fixed_tick + 10 && bindings[i].round == bindings[i - 1].round &&
            bindings[i].epoch == bindings[i - 1].epoch && samples[i].sample_seconds > previous->sample_seconds &&
            samples[i].sample_seconds - previous->sample_seconds <= .025, "noncontiguous preserved segment");
        std::array<float, 223> obs{};
        require(balance::project(samples[i], previous, obs.data()) == balance::kOk, "observable projection rejected sample");
        require(obs[203] == float(!row.reset), "unexpected observation history availability");
        const size_t p = rek_bc::HEADER_BYTES + i * rek_bc::ROW_BYTES;
        std::memcpy(out.data() + p + 32, obs.data(), sizeof(obs));
        require(std::memcmp(out.data() + p, original.data() + p, 32) == 0 &&
            std::memcmp(out.data() + p + 924, original.data() + p + 924, 132) == 0, "row metadata or class support changed");
    }
    require(std::memcmp(out.data(), original.data(), 32) == 0 && out[255] == original[255], "nonmask header changed");
    rek_bc::parse(out); return out;
}
void write(const fs::path& file, const void* bytes, size_t size) {
    FILE* out = std::fopen(file.c_str(), "wbx"); require(out, "output already exists or cannot be created");
    const auto written = std::fwrite(bytes, 1, size, out); const int closed = std::fclose(out);
    require(written == size && closed == 0, "output write failed");
    require(sha(read(file)) == sha(bytes, size), "output readback mismatch");
}
void write(const fs::path& file, const Bytes& bytes) { write(file, bytes.data(), bytes.size()); }
void write(const fs::path& file, const std::string& text) { write(file, text.data(), text.size()); }

void run(const fs::path& source, const std::array<fs::path, 2>& raw, const fs::path& output) {
    require(!fs::exists(output), "output directory already exists");
    const auto original = pinned(source / "human-commands.bin", DATA_HASH);
    const auto old_manifest = pinned(source / "manifest.json", MANIFEST_HASH);
    const auto ledger = pinned(source / "row-ledger.jsonl", ROW_HASH);
    const auto commands = pinned(source / "command-ledger.jsonl", COMMAND_HASH);
    const auto data = rek_bc::parse(original); require(data.rows.size() == 11985 && data.sequences.size() == 4, "unexpected pinned row/sequence counts");
    const auto bindings = bind_ledger(ledger, data);
    const std::array<Bytes, 2> captures{pinned(raw[0], RAW_HASHES[0]), pinned(raw[1], RAW_HASHES[1])};
    const auto samples = load_snapshots(captures, bindings, data); const auto migrated = reproject(original, data, bindings, samples);
    const auto projected = rek_bc::parse(migrated);
    std::array<int, 2> rows{}, labels{}, kicks{}, sequences{}; int changed_rows = 0, history = 0; double max_tilt = 0;
    std::ostringstream projection_ledger;
    for (size_t i = 0; i < data.rows.size(); ++i) {
        const auto& r = data.rows[i]; const auto& obs = projected.rows[i].obs;
        rows[r.split]++; labels[r.split] += r.weight > 0; kicks[r.split] += r.action == 17; sequences[r.split] += r.reset;
        changed_rows += std::memcmp(r.obs.data(), obs.data(), sizeof(r.obs)) != 0; history += obs[203] != 0;
        max_tilt = std::max(max_tilt, double(std::max(obs[72], obs[158])));
        projection_ledger << "{\"row\":" << i << ",\"split\":" << r.split << ",\"source_line\":" << bindings[i].source_line
            << ",\"previous_source_line\":" << (r.reset ? "null" : std::to_string(bindings[i - 1].source_line))
            << ",\"history_available\":" << (r.reset ? "false" : "true") << "}\n";
    }
    require(rows == std::array<int, 2>{5993, 5992} && labels == std::array<int, 2>{4943, 4990} &&
        kicks == std::array<int, 2>{16, 2} && sequences == std::array<int, 2>{2, 2}, "pinned split counts changed");
    require(changed_rows == 11985 && history == 11981 && max_tilt > 0, "missing new observation features");
    Json manifest(cJSON_CreateObject(), cJSON_Delete); auto* m = manifest.get();
    add(m, "schema", "rek.human_observable_migration.v1"); add(m, "observation_schema", balance::kSchema);
    add(m, "source_observation_schema", "rek.native5.scaled_polar_xy.v1");
    add(m, "source_dataset_sha256", DATA_HASH); add(m, "dataset_sha256", sha(migrated));
    add(m, "row_ledger_sha256", ROW_HASH); add(m, "command_ledger_sha256", COMMAND_HASH);
    add(m, "source_manifest_sha256", MANIFEST_HASH); add(m, "original_feature_mask_sha256", sha(original.data() + 32, 223));
    add(m, "feature_mask_sha256", sha(migrated.data() + 32, 223));
    add(m, "projection_ledger_sha256", sha(projection_ledger.str().data(), projection_ledger.str().size()));
    add(m, "clock", "recorded sample.unity_unscaled_time; no fabricated QPC; preceding retained observation within each preserved segment");
    add(m, "joint_pose", "unavailable in both fighters, matching physical/live adapter contract");
    add(m, "referee", "unavailable; protocol-v5 cannot prove new QPC/lifecycle freshness contract; no inferred falls/counts");
    add(m, "targets", "unchanged observed outgoing command requests; no acceptance, execution, success or held-key inference");
    add(m, "split", "first whole round train, second whole round development holdout; same human session; four preserved historical segments");
    add(m, "ledger_semantics", "original ledgers are byte-identical label evidence, including old observation provenance; projection-ledger.jsonl is the new feature chronology");
    add(m, "class_support", "unchanged vocabulary support, not measured action legality");
    cJSON_AddBoolToObject(m, "nonobservation_row_bytes_identical", true); cJSON_AddBoolToObject(m, "original_ledgers_identical", true);
    cJSON_AddBoolToObject(m, "gpu_used", false); cJSON_AddBoolToObject(m, "physics_stepped", false); cJSON_AddBoolToObject(m, "training_performed", false);
    cJSON_AddNumberToObject(m, "rows", data.rows.size()); cJSON_AddNumberToObject(m, "changed_observation_rows", changed_rows);
    cJSON_AddNumberToObject(m, "history_available_rows", history); cJSON_AddNumberToObject(m, "maximum_tilt_fraction", max_tilt);
    cJSON_AddNumberToObject(m, "retained_features", std::count(migrated.begin() + 32, migrated.begin() + 255, uint8_t(1)));
    auto* sources = cJSON_AddArrayToObject(m, "sources");
    for (int split = 0; split < 2; ++split) {
        auto* j = cJSON_CreateObject(); add(j, "file", raw[split].string()); add(j, "sha256", RAW_HASHES[split]);
        cJSON_AddNumberToObject(j, "split", split); cJSON_AddNumberToObject(j, "rows", rows[split]); cJSON_AddNumberToObject(j, "labels", labels[split]);
        cJSON_AddNumberToObject(j, "left_front_labels", kicks[split]); cJSON_AddNumberToObject(j, "segments", sequences[split]); cJSON_AddItemToArray(sources, j);
    }
    require(fs::create_directory(output), "cannot create fresh output directory");
    write(output / "human-observable.bin", migrated); write(output / "row-ledger.jsonl", ledger); write(output / "command-ledger.jsonl", commands);
    write(output / "source-manifest.json", old_manifest); write(output / "projection-ledger.jsonl", projection_ledger.str());
    write(output / "feature-mask.bin", migrated.data() + 32, 223); write(output / "original-feature-mask.bin", original.data() + 32, 223);
    write(output / "manifest.json", json_string(m) + "\n"); std::cout << json_string(m) << '\n';
}
} // namespace human_observable

#ifndef REK_HUMAN_OBSERVABLE_NO_MAIN
int main(int argc, char** argv) {
    try {
        rek_bc::require(argc == 5, "Usage: human-observable-data ORIGINAL_DATASET_DIR TRAIN_RAW HELDOUT_RAW NEW_OUTPUT_DIR");
        human_observable::run(argv[1], {argv[2], argv[3]}, argv[4]); return 0;
    } catch (const std::exception& e) { std::cerr << "human_observable_error: " << e.what() << '\n'; return 2; }
}
#endif
