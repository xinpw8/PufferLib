#include "motion_assets.cuh"
#include "../../../vendor/cJSON.h"

#include <openssl/evp.h>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>

namespace {
void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

using Json = std::unique_ptr<cJSON, decltype(&cJSON_Delete)>;

const cJSON* member(const cJSON* object, const char* key) {
    const cJSON* value = cJSON_GetObjectItemCaseSensitive(object, key);
    require(value != nullptr, std::string("missing asset field: ") + key);
    return value;
}

std::string string_value(const cJSON* value) {
    require(cJSON_IsString(value) && value->valuestring != nullptr, "expected asset string");
    return value->valuestring;
}

std::string string_member(const cJSON* object, const char* key) {
    return string_value(member(object, key));
}

double number(const cJSON* value) {
    require(cJSON_IsNumber(value) && std::isfinite(value->valuedouble), "expected finite asset number");
    return value->valuedouble;
}

int integer(const cJSON* value) {
    const double n = number(value);
    require(n >= INT32_MIN && n <= INT32_MAX && std::floor(n) == n,
        "asset integer exceeds int32 range");
    return static_cast<int>(n);
}

size_t size_value(const cJSON* value) {
    const double n = number(value);
    require(n >= 0 && n < static_cast<double>(SIZE_MAX) && std::floor(n) == n,
        "invalid asset size");
    return static_cast<size_t>(n);
}

int flag(const cJSON* value) {
    if (cJSON_IsBool(value)) return cJSON_IsTrue(value) ? 1 : 0;
    const int n = integer(value);
    require(n == 0 || n == 1, "asset boolean must be zero or one");
    return n;
}

std::string bundle_file(const char* root, const std::string& name) {
    require(root != nullptr && root[0] != '\0', "asset bundle root is required");
    require(!name.empty() && name != "." && name != ".."
        && name.find('/') == std::string::npos && name.find('\\') == std::string::npos,
        "asset filename must be relative to its bundle");
    return std::string(root) + "/" + name;
}

std::vector<unsigned char> read_bytes(const std::string& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    require(input.good(), "could not open asset: " + path);
    const auto end = input.tellg();
    require(end >= 0 && static_cast<uint64_t>(end) <= std::numeric_limits<size_t>::max(),
        "could not size asset: " + path);
    std::vector<unsigned char> bytes(static_cast<size_t>(end));
    input.seekg(0);
    if (!bytes.empty()) input.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
    require(input.good(), "could not read complete asset: " + path);
    return bytes;
}

std::string sha256(const std::vector<unsigned char>& bytes) {
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int length = 0;
    require(EVP_Digest(bytes.data(), bytes.size(), digest, &length, EVP_sha256(), nullptr) == 1
        && length == 32, "SHA-256 failed");
    static constexpr char hex[] = "0123456789abcdef";
    std::string result(64, '0');
    for (size_t i = 0; i < 32; ++i) {
        result[2 * i] = hex[digest[i] >> 4];
        result[2 * i + 1] = hex[digest[i] & 15];
    }
    return result;
}

Json parse(const std::vector<unsigned char>& bytes) {
    std::string text(bytes.begin(), bytes.end());
    Json value(cJSON_ParseWithLengthOpts(text.c_str(), text.size() + 1, nullptr, 1), cJSON_Delete);
    require(value && cJSON_IsObject(value.get()), "invalid asset JSON object");
    return value;
}

struct HostArray {
    std::vector<size_t> shape;
    std::vector<float> values;
};

std::vector<float> floats(const std::vector<unsigned char>& bytes) {
    const uint16_t endian = 1;
    require(*reinterpret_cast<const uint8_t*>(&endian) == 1, "float32_le requires little endian host");
    require(bytes.size() % sizeof(float) == 0, "float asset byte count is not divisible by four");
    std::vector<float> values(bytes.size() / sizeof(float));
    if (!bytes.empty()) std::memcpy(values.data(), bytes.data(), bytes.size());
    for (const float value : values) require(std::isfinite(value), "nonfinite motion asset");
    return values;
}

void normalize_heading(std::vector<float>& values) {
    require(!values.empty() && values.size() % 4 == 0, "clip roots must be a nonempty WXYZ matrix");
    const auto check_units = [&]() {
        for (size_t i = 0; i < values.size(); i += 4) {
            double sum = 0;
            for (int j = 0; j < 4; ++j) sum += double(values[i + j]) * values[i + j];
            require(std::isfinite(sum) && std::abs(std::sqrt(sum) - 1.0) <= 1.0e-4,
                "clip roots must contain unit WXYZ quaternions");
        }
    };
    check_units();
    const float w = values[0], x = values[1], y = values[2], z = values[3];
    volatile float xy = x * y, zw = z * w, yy = y * y, zz = z * z;
    volatile float cross = xy + zw, squares = yy + zz;
    volatile float numerator = 2.0f * cross, twice = 2.0f * squares;
    volatile float denominator = 1.0f - twice;
    const float angle = atan2f(numerator, denominator);
    require(std::isfinite(angle), "clip frame-zero heading must be finite");
    if (fabsf(angle) >= 1.0e-6f) {
        volatile float half = -0.5f * angle;
        const float cosine = cosf(half), sine = sinf(half);
        for (size_t i = 0; i < values.size(); i += 4) {
            const float old[4] = {values[i], values[i+1], values[i+2], values[i+3]};
            volatile float wc = old[0] * cosine, zs = old[3] * sine;
            volatile float xc = old[1] * cosine, ys = old[2] * sine;
            volatile float xs = old[1] * sine, yc = old[2] * cosine;
            volatile float zc = old[3] * cosine, ws = old[0] * sine;
            values[i] = wc - zs;
            values[i+1] = xc - ys;
            values[i+2] = xs + yc;
            values[i+3] = zc + ws;
        }
    }
    check_units();
}

struct HostClip {
    int id;
    size_t frames;
    float fps;
    std::string positions, roots, role;
    std::vector<float> feet;
};

struct HostRoute {
    int clip_id;
    int kind;
    int move;
    SonicMotionComposerNativeConfig config;
};

struct HostAssets {
    std::string manifest_sha256;
    std::map<std::string, HostArray> arrays;
    std::vector<HostClip> clips;
    std::vector<HostRoute> routes;
};

HostAssets load_assets(const char* root, const char* features) {
    HostAssets out;
    const auto manifest_bytes = read_bytes(bundle_file(root, "semantic_duel_assets_manifest.json"));
    out.manifest_sha256 = sha256(manifest_bytes);
    const Json manifest = parse(manifest_bytes);
    require(string_member(manifest.get(), "schema") == "rek.g1_semantic_duel_assets.v1",
        "unsupported semantic asset schema");
    const Json feature_manifest = parse(read_bytes(bundle_file(features, "foot_features_manifest.json")));
    require(string_member(feature_manifest.get(), "asset_manifest_sha256") == out.manifest_sha256,
        "foot features belong to another asset manifest");
    const cJSON* files = member(manifest.get(), "files");
    require(cJSON_IsObject(files), "asset files must be an object");
    for (const cJSON* record = files->child; record; record = record->next) {
        require(record->string != nullptr, "asset file has no name");
        const std::string name = record->string;
        const auto bytes = read_bytes(bundle_file(root, name));
        require(bytes.size() == size_value(member(record, "bytes"))
            && sha256(bytes) == string_member(record, "sha256"), "semantic asset identity mismatch: " + name);
        const cJSON* dtype = cJSON_GetObjectItemCaseSensitive(record, "dtype");
        if (!dtype || string_value(dtype) != "float32_le") continue;
        HostArray array;
        const cJSON* shape = member(record, "shape");
        require(cJSON_IsArray(shape), "asset shape must be an array");
        size_t product = 1;
        for (const cJSON* dimension = shape->child; dimension; dimension = dimension->next) {
            const size_t n = size_value(dimension);
            require(n > 0 && product <= SIZE_MAX / n, "invalid motion asset shape");
            array.shape.push_back(n);
            product *= n;
        }
        array.values = floats(bytes);
        require(product == array.values.size(), "motion asset shape/byte count mismatch: " + name);
        require(out.arrays.emplace(name, std::move(array)).second, "duplicate asset file: " + name);
    }

    std::map<int, const cJSON*> feature_records;
    const cJSON* feature_clips = member(feature_manifest.get(), "clips");
    require(cJSON_IsArray(feature_clips), "feature clips must be an array");
    for (const cJSON* record = feature_clips->child; record; record = record->next)
        require(feature_records.emplace(integer(member(record, "npz_path_id")), record).second,
            "duplicate foot feature clip ID");
    const cJSON* clips = member(manifest.get(), "clips");
    require(cJSON_IsArray(clips) && cJSON_GetArraySize(clips) > 0, "motion clips are required");
    std::map<int, size_t> clip_indices;
    for (const cJSON* record = clips->child; record; record = record->next) {
        HostClip clip;
        clip.id = integer(member(record, "npz_path_id"));
        clip.role = string_member(record, "role");
        clip.frames = size_value(member(record, "frames"));
        clip.fps = static_cast<float>(number(member(record, "fps")));
        require(clip.frames > 0 && clip.frames <= 16777216 && clip.fps > 0 && std::isfinite(clip.fps),
            "invalid motion clip frame count or frame rate");
        const cJSON* clip_files = member(record, "files");
        clip.positions = string_member(clip_files, "mujoco_joint_order");
        clip.roots = string_member(clip_files, "wxyz");
        auto& positions = out.arrays.at(clip.positions);
        auto& roots = out.arrays.at(clip.roots);
        auto& xyzw = out.arrays.at(string_member(clip_files, "xyzw"));
        require(positions.shape == std::vector<size_t>({clip.frames, 29})
            && roots.shape == std::vector<size_t>({clip.frames, 4}) && roots.shape == xyzw.shape,
            "clip motion/root shape mismatch");
        normalize_heading(roots.values);
        for (size_t i = 0; i < roots.values.size(); i += 4) {
            xyzw.values[i] = roots.values[i+1]; xyzw.values[i+1] = roots.values[i+2];
            xyzw.values[i+2] = roots.values[i+3]; xyzw.values[i+3] = roots.values[i];
        }
        const cJSON* feature = feature_records.at(clip.id);
        const auto foot_bytes = read_bytes(bundle_file(features, string_member(feature, "file")));
        require(foot_bytes.size() == clip.frames * 6 * sizeof(float)
            && sha256(foot_bytes) == string_member(feature, "sha256"),
            "foot feature identity/shape mismatch");
        clip.feet = floats(foot_bytes);
        require(clip_indices.emplace(clip.id, out.clips.size()).second, "duplicate motion clip ID");
        out.clips.push_back(std::move(clip));
    }

    const cJSON* routes = member(manifest.get(), "routes");
    require(cJSON_IsArray(routes) && cJSON_GetArraySize(routes) == 24, "expected original 24 motion routes");
    bool moves[17] = {};
    for (const cJSON* record = routes->child; record; record = record->next) {
        require(integer(member(record, "route_id")) == static_cast<int>(out.routes.size()),
            "motion routes must retain contiguous original IDs");
        HostRoute route = {};
        route.clip_id = integer(member(record, "npz_path_id"));
        require(clip_indices.count(route.clip_id) == 1, "route references an unknown motion clip");
        const std::string kind = string_member(record, "kind");
        if (kind == "idle") route.kind = 0;
        else if (kind == "translation") route.kind = 1;
        else if (kind == "turn") route.kind = 2;
        else if (kind == "discrete_move" || kind == "kick") route.kind = 3;
        else throw std::runtime_error("unknown motion route kind: " + kind);
        const cJSON* move = member(record, "runtime_move_index");
        route.move = cJSON_IsNull(move) ? -1 : integer(move);
        if (route.move != -1) {
            require(route.move >= 0 && route.move < 17 && !moves[route.move], "invalid or duplicate runtime move ID");
            moves[route.move] = true;
        }
        const cJSON* config = member(record, "config");
        route.config = {
            flag(member(config, "mirror")), flag(member(config, "loop")),
            static_cast<float>(number(member(config, "playback_speed"))),
            integer(member(config, "start_frame")), integer(member(config, "end_frame")),
            static_cast<float>(number(member(config, "blend_in_seconds"))),
            static_cast<float>(number(member(config, "blend_out_seconds"))),
            static_cast<float>(number(member(config, "yaw_blend")))
        };
        out.routes.push_back(route);
    }
    for (bool present : moves) require(present, "original 17-move route table is incomplete");
    return out;
}
} // namespace

void* RekNative5Motion::allocate(size_t bytes, bool zero) {
    require(bytes > 0, "zero-size motion allocation");
    void* pointer = nullptr;
    cuda_check(cudaMalloc(&pointer, bytes), "allocate native motion memory");
    allocations_.push_back(pointer);
    if (zero) cuda_check(cudaMemset(pointer, 0, bytes), "zero native motion memory");
    return pointer;
}

void* RekNative5Motion::upload(const void* source, size_t bytes) {
    void* pointer = allocate(bytes);
    cuda_check(cudaMemcpy(pointer, source, bytes, cudaMemcpyHostToDevice), "upload native motion memory");
    return pointer;
}

void RekNative5Motion::release() noexcept {
    for (auto it = allocations_.rbegin(); it != allocations_.rend(); ++it) cudaFree(*it);
    allocations_.clear();
}

RekNative5Motion::~RekNative5Motion() { release(); }

std::string RekNative5Motion::validate_assets(const char* root, const char* features) {
    return load_assets(root, features).manifest_sha256;
}

RekNative5Motion::RekNative5Motion(const RekNative5Config& config, cudaStream_t stream) {
    try {
        require(config.arenas > 0 && config.arenas <= INT32_MAX / 2, "invalid native motion arena count");
        require(config.locomotion_segment_ticks > 0, "locomotion segment must be positive");
        for (uint32_t duration : config.move_duration_ticks) require(duration > 0, "move durations must be positive");
        count = static_cast<size_t>(config.arenas) * 2;
        HostAssets host = load_assets(config.assets_path, config.motion_features_path);
        manifest_sha256 = host.manifest_sha256;
        asset_count = host.clips.size();
        bool have_idle = false;
        for (const auto& clip : host.clips) {
            if (clip.role != "idle") continue;
            require(!have_idle, "multiple idle motion clips");
            const auto& p = host.arrays.at(clip.positions).values;
            const auto& q = host.arrays.at(clip.roots).values;
            for (size_t joint = 0; joint < 29; ++joint) idle_positions[joint] = p[joint];
            idle_root_xyzw = {q[1], q[2], q[3], q[0]};
            have_idle = true;
        }
        require(have_idle, "idle motion reference is missing");
        std::map<std::string, float*> arrays;
        for (const auto& entry : host.arrays)
            arrays[entry.first] = static_cast<float*>(upload(entry.second.values.data(), entry.second.values.size() * sizeof(float)));
        std::vector<RekG1CudaMotionAsset> assets;
        std::map<int, SonicMotionComposerNativeClip> clips;
        for (const auto& clip : host.clips) {
            const SonicMotionComposerNativeClip view = {arrays.at(clip.positions), arrays.at(clip.roots),
                clip.frames * 29, clip.frames * 4, clip.frames, clip.fps};
            clips[clip.id] = view;
            assets.push_back({view, static_cast<float*>(upload(clip.feet.data(), clip.feet.size() * sizeof(float))), clip.feet.size()});
        }
        auto* device_assets = static_cast<RekG1CudaMotionAsset*>(upload(assets.data(), assets.size() * sizeof(assets[0])));
        std::vector<RekG1CudaComposerCommand> routes;
        int32_t route_kinds[24] = {}, move_routes[17] = {};
        for (size_t i = 0; i < host.routes.size(); ++i) {
            const auto& route = host.routes[i];
            routes.push_back({REK_G1_CUDA_COMPOSER_PLAY, clips.at(route.clip_id), route.config, 0.0f});
            route_kinds[i] = route.kind;
            if (route.move >= 0) move_routes[route.move] = static_cast<int32_t>(i);
        }
        composers = static_cast<SonicMotionComposerNative*>(allocate(count * sizeof(*composers)));
        matchers = static_cast<SonicMotionEntryMatcherNative*>(allocate(count * sizeof(*matchers)));
        auto* slots = static_cast<SonicMotionEntryMatcherNativeFeatureSlot*>(allocate(count * asset_count * sizeof(SonicMotionEntryMatcherNativeFeatureSlot)));
        auto* init_status = static_cast<RekG1CudaMotionInitStatus*>(allocate(count * sizeof(RekG1CudaMotionInitStatus)));
        cuda_check(rek_g1_cuda_motion_init(composers, matchers, slots, device_assets, asset_count, 50, init_status, count, stream),
            "initialize native motion");
        cuda_check(cudaStreamSynchronize(stream), "wait for native motion initialization");
        std::vector<RekG1CudaMotionInitStatus> statuses(count);
        cuda_check(cudaMemcpy(statuses.data(), init_status, count * sizeof(statuses[0]), cudaMemcpyDeviceToHost), "read motion initialization status");
        for (size_t i = 0; i < count; ++i)
            require(statuses[i].composer == 0 && statuses[i].matcher == 0, "native motion initialization rejected fighter " + std::to_string(i));

        positions = static_cast<float*>(allocate(count * 290 * sizeof(float), true));
        next_positions = static_cast<float*>(allocate(count * 290 * sizeof(float), true));
        rotations = static_cast<float*>(allocate(count * 40 * sizeof(float), true));
        std::vector<SonicMotionComposerNativeReferenceOutput> outputs(count);
        for (size_t i = 0; i < count; ++i)
            outputs[i] = {positions + i * 290, next_positions + i * 290, rotations + i * 40, 290, 290, 40};
        const uint32_t mirror_indices[29] = {6,7,8,9,10,11,0,1,2,3,4,5,12,13,14,22,23,24,25,26,27,28,15,16,17,18,19,20,21};
        const uint8_t mirror_negate[29] = {0,1,1,0,0,1,0,1,1,0,0,1,1,1,0,0,1,1,0,1,0,1,0,1,1,0,1,0,1};
        const SonicMotionComposerNativeMirrorTable mirror = {
            static_cast<uint32_t*>(upload(mirror_indices, sizeof(mirror_indices))),
            static_cast<uint8_t*>(upload(mirror_negate, sizeof(mirror_negate))), 29, 29};
        SonicMotionComposerNativeReferenceTiming timing = {};
        for (int i = 0; i < 10; ++i) { timing.current_offsets[i] = i * 5; timing.next_offsets[i] = i * 5 + 1; }
        const RekG1CudaSemanticConfig semantic_config = {{0.02f, 0.5f}, {1.0f, 1.0f, 1.0f, 50}, {0.03f, 0.03f, 2.0f, 1}};
        auto* table = static_cast<RekG1SemanticActionTableStorage*>(allocate(sizeof(RekG1SemanticActionTableStorage)));
        auto* table_status = static_cast<int*>(allocate(sizeof(int)));
        auto* durations = static_cast<uint32_t*>(upload(config.move_duration_ticks, sizeof(config.move_duration_ticks)));
        cuda_check(rek_g1_cuda_semantic_table_init(table, config.locomotion_segment_ticks, durations, table_status, stream), "initialize semantic action table");
        cuda_check(cudaStreamSynchronize(stream), "wait for semantic action table");
        int table_result = -1;
        cuda_check(cudaMemcpy(&table_result, table_status, sizeof(int), cudaMemcpyDeviceToHost), "read semantic action table status");
        require(table_result == 0, "native semantic action table rejected configuration");

        rows = static_cast<RekG1CudaSemanticRow*>(allocate(count * sizeof(*rows), true));
        std::vector<float> headings(count * 4, 0.0f);
        for (size_t i = 0; i < count; ++i) headings[i * 4] = 1.0f;
        heading = static_cast<float*>(upload(headings.data(), headings.size() * sizeof(float)));
        observation12 = static_cast<float*>(allocate(count * 12 * sizeof(float)));
        masks = static_cast<uint8_t*>(allocate(count * 33));
        zero_flags = static_cast<uint8_t*>(allocate(count, true));
        all_flags = static_cast<uint8_t*>(allocate(count));
        cuda_check(cudaMemset(all_flags, 1, count), "initialize semantic reset flags");
        scheduler_host = {rows, composers, table,
            static_cast<RekG1CudaSemanticConfig*>(upload(&semantic_config, sizeof(semantic_config))),
            static_cast<RekG1CudaComposerCommand*>(upload(routes.data(), routes.size() * sizeof(routes[0]))),
            static_cast<int32_t*>(upload(route_kinds, sizeof(route_kinds))),
            static_cast<int32_t*>(upload(move_routes, sizeof(move_routes))),
            static_cast<SonicMotionComposerNativeReferenceTiming*>(upload(&timing, sizeof(timing))),
            static_cast<SonicMotionComposerNativeMirrorTable*>(upload(&mirror, sizeof(mirror))),
            static_cast<SonicMotionComposerNativeReferenceOutput*>(upload(outputs.data(), outputs.size() * sizeof(outputs[0]))),
            heading, observation12, masks};
        scheduler = static_cast<RekG1CudaSemanticBuffers*>(upload(&scheduler_host, sizeof(scheduler_host)));
        reset(all_flags, heading, stream);
        check_status(stream);
    } catch (...) {
        release();
        throw;
    }
}

void RekNative5Motion::reset(const uint8_t* flags, const float* headings, cudaStream_t stream) {
    cuda_check(rek_g1_cuda_semantic_reset(scheduler, flags, headings, count, stream), "reset semantic scheduler");
}

void RekNative5Motion::pre(const float* actions, const float* velocity, const uint8_t* suspended, cudaStream_t stream) {
    cuda_check(rek_g1_cuda_semantic_pre(scheduler, actions, velocity, suspended ? suspended : zero_flags, count, stream), "semantic pre-step");
}

void RekNative5Motion::pre_direct(const float* actions, const RekG1CudaDirectCommand* commands,
        const uint8_t* enabled, RekG1CudaDirectResult* results, const float* velocity,
        const uint8_t* suspended, cudaStream_t stream) {
    cuda_check(rek_g1_cuda_semantic_pre_direct(scheduler, actions, commands, enabled,
        results, velocity, suspended ? suspended : zero_flags, count, stream), "direct semantic pre-step");
}

void RekNative5Motion::post(const float* velocity, const int32_t* phase, const uint8_t* suspended,
        const uint8_t* input_reset, const uint8_t* reset_event, const uint8_t* terminal, cudaStream_t stream) {
    cuda_check(rek_g1_cuda_semantic_post(scheduler, velocity, phase, suspended ? suspended : zero_flags,
        input_reset ? input_reset : zero_flags, reset_event ? reset_event : zero_flags,
        terminal ? terminal : zero_flags, count, stream), "semantic post-step");
}

void RekNative5Motion::check_status(cudaStream_t stream) const {
    cuda_check(cudaStreamSynchronize(stream), "wait for semantic scheduler status");
    std::vector<RekG1CudaSemanticRow> host(count);
    cuda_check(cudaMemcpy(host.data(), rows, count * sizeof(host[0]), cudaMemcpyDeviceToHost), "read semantic scheduler status");
    for (size_t i = 0; i < count; ++i)
        require(host[i].status == 0, "semantic scheduler fighter " + std::to_string(i) + " failed with status " + std::to_string(host[i].status));
}

#ifdef REK_NATIVE5_MOTION_ASSET_MAIN
#include <iostream>
int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: validate_motion_assets ASSET_BUNDLE FOOT_FEATURE_BUNDLE\n";
        return 2;
    }
    try {
        const std::string digest = RekNative5Motion::validate_assets(argv[1], argv[2]);
        std::cout << "motion_assets_validated manifest_sha256=" << digest
            << " routes=24 moves=17 cuda_calls=0\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
#endif
