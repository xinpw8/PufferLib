#pragma once
#include "round_reward.h"
#include <openssl/sha.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace rek_authentic {
constexpr int OBS = 223, ACTIONS = 33, LOGITS = 34;
constexpr size_t HEADER_BYTES = 256, ROW_BYTES = 1128;
constexpr size_t IDENTITY_HEADER_BYTES = 384;
constexpr size_t REPLAY_HEADER_BYTES = 128, REPLAY_ROW_BYTES = 152;
inline void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
inline std::string hex(const unsigned char* data, size_t n) {
    const char* digits = "0123456789abcdef"; std::string out(n * 2, '0');
    for (size_t i = 0; i < n; ++i) { out[2*i] = digits[data[i] >> 4]; out[2*i+1] = digits[data[i] & 15]; }
    return out;
}
inline std::string sha256(const void* data, size_t n) {
    unsigned char out[SHA256_DIGEST_LENGTH]; SHA256(static_cast<const unsigned char*>(data), n, out);
    return hex(out, sizeof(out));
}
inline std::vector<unsigned char> read_file(const char* path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    require(bool(f), "cannot open authentic trajectory file"); const auto n = f.tellg();
    require(n >= 0 && uint64_t(n) <= 1024ULL*1024*1024, "authentic file exceeds 1 GiB bound");
    std::vector<unsigned char> b(static_cast<size_t>(n)); f.seekg(0);
    f.read(reinterpret_cast<char*>(b.data()), b.size()); require(bool(f), "authentic file read failed"); return b;
}
template<class T> inline T scalar(const unsigned char* b) {
    const uint16_t endian = 1; require(*reinterpret_cast<const unsigned char*>(&endian) == 1, "little endian host required");
    T value; std::memcpy(&value, b, sizeof(T)); return value;
}
struct Row {
    uint32_t split=0, sequence=0, reset=0;
    int32_t action=0;
    float policy_weight=0, value_weight=0;
    double time=0, next_time=0, dt=0;
    std::array<float, OBS> obs{};
    std::array<float, ACTIONS> support{}; // Actual behavior legality, unlike the BC vocabulary mask.
    float gamma=0, lambda=0, reward=0, outcome=0;
    uint32_t source_seq=0, next_source_seq=0;
    int32_t own_points=0, opponent_points=0, next_own_points=0, next_opponent_points=0;
    uint32_t terminal_after=0, applied=0;
    uint64_t behavior_seed=73;
};
struct Sequence { size_t begin=0, end=0; uint32_t sequence=0, split=0; };
struct Dataset {
    std::array<uint8_t, OBS> feature_mask{};
    std::vector<Row> rows;
    std::vector<Sequence> sequences;
    std::string digest;
    uint32_t format_version=1;
    std::string identity_sha256, worker_sha256, checkpoint_sha256, native_object_sha256;
};
inline void validate(Dataset& d, uint32_t round_count) {
    require(!d.rows.empty(), "empty authentic dataset");
    for (auto x : d.feature_mask) require(d.format_version==3 ? x<=1 : x==1, "invalid authentic feature mask");
    d.sequences.clear(); std::set<uint32_t> seen;
    for (size_t i=0; i<d.rows.size(); ++i) {
        const auto& r=d.rows[i];
        require(r.split == 0 && r.reset <= 1 && r.action >= 0 && r.action < ACTIONS, "invalid authentic row identity");
        for (float v:r.obs) require(std::isfinite(v), "nonfinite authentic observation");
        for (float v:r.support) require(v == 0 || v == 1, "invalid actual action mask");
        require(r.support[r.action] == 1, "chosen action outside actual mask");
        require(std::isfinite(r.time) && r.time >= 0 && std::isfinite(r.next_time) && r.next_time > r.time &&
            std::isfinite(r.dt) && r.dt > 0 && std::abs(r.next_time-r.time-r.dt)<1e-10, "invalid authentic time");
        require(std::isfinite(r.gamma) && r.gamma > 0 && r.gamma <= 1 &&
            std::isfinite(r.lambda) && r.lambda > 0 && r.lambda <= 1, "invalid time-scaled discount");
        require(r.terminal_after <= 1 && r.applied <= 1 && r.policy_weight == float(r.applied) &&
            r.value_weight == 1 && (r.applied || r.terminal_after), "invalid authentic loss/ack flags");
        require((r.outcome == -1 || r.outcome == 0 || r.outcome == 1) &&
            (r.terminal_after || r.outcome == 0), "invalid terminal outcome");
        require(r.own_points >= 0 && r.opponent_points >= 0 && r.next_own_points >= r.own_points &&
            r.next_opponent_points >= r.opponent_points && r.next_own_points <= 32767 &&
            r.next_opponent_points <= 32767 && r.next_source_seq > r.source_seq, "invalid point/sequence transition");
        const float after=r.terminal_after ? 0.f : rek5_round_reward::potential(r.next_own_points,r.next_opponent_points);
        const float expected=r.outcome+r.gamma*after-rek5_round_reward::potential(r.own_points,r.opponent_points);
        require(std::isfinite(r.reward) && std::abs(r.reward-expected)<1e-6f, "authentic reward contract mismatch");
        const bool new_sequence=i==0 || r.sequence != d.rows[i-1].sequence;
        require(bool(r.reset)==new_sequence, "authentic recurrent reset mismatch");
        if (new_sequence) {
            require(seen.insert(r.sequence).second && r.time==0, "reused sequence or nonzero start time");
            if(i) { require(d.rows[i-1].terminal_after==1, "prior sequence lacks terminal"); d.sequences.back().end=i; }
            d.sequences.push_back({i,d.rows.size(),r.sequence,r.split});
        } else {
            const auto& p=d.rows[i-1];
            require(p.behavior_seed==r.behavior_seed,"behavior seed changed within round");
            require(!p.terminal_after && p.next_time==r.time && p.next_source_seq==r.source_seq &&
                p.next_own_points==r.own_points && p.next_opponent_points==r.opponent_points,
                "broken authentic transition chain");
        }
    }
    require(d.rows.back().terminal_after==1 && d.sequences.size()==round_count, "terminal/round count mismatch");
}
inline Dataset decode(const std::vector<unsigned char>& b) {
    require(b.size()>=HEADER_BYTES, "short authentic header");
    const bool identity=std::memcmp(b.data(),"REKRL003",8)==0;
    const size_t header=identity?IDENTITY_HEADER_BYTES:HEADER_BYTES;
    require(b.size()>=header && (identity || std::memcmp(b.data(),"REKRL001",8)==0), "invalid authentic magic");
    require(scalar<uint32_t>(&b[8])==(identity?3:1) && scalar<uint32_t>(&b[12])==OBS &&
        scalar<uint32_t>(&b[16])==ACTIONS && scalar<uint32_t>(&b[24])==ROW_BYTES && b[255]==0,
        "invalid authentic header");
    const auto n=scalar<uint32_t>(&b[20]), rounds=scalar<uint32_t>(&b[28]);
    require(n>0 && n<1000000 && b.size()==header+uint64_t(n)*ROW_BYTES, "authentic shape mismatch");
    Dataset d; std::copy(b.begin()+32,b.begin()+255,d.feature_mask.begin()); d.rows.resize(n);
    d.format_version=identity?3:1;
    if(identity) {
        d.identity_sha256=hex(&b[256],32); d.worker_sha256=hex(&b[288],32);
        d.checkpoint_sha256=hex(&b[320],32); d.native_object_sha256=hex(&b[352],32);
    }
    for(size_t i=0;i<n;++i) {
        const auto* p=b.data()+header+i*ROW_BYTES; auto& r=d.rows[i];
        r.split=scalar<uint32_t>(p); r.sequence=scalar<uint32_t>(p+4); r.reset=scalar<uint32_t>(p+8);
        r.action=scalar<int32_t>(p+12); r.policy_weight=scalar<float>(p+16); r.time=scalar<double>(p+24);
        if(identity) r.behavior_seed=uint64_t(scalar<uint32_t>(p+20)) | (uint64_t(scalar<uint32_t>(p+1124))<<32);
        else require(scalar<uint32_t>(p+20)==0 && scalar<uint32_t>(p+1124)==0, "nonzero authentic reserved field");
        for(int j=0;j<OBS;++j) r.obs[j]=scalar<float>(p+32+4*j);
        for(int j=0;j<ACTIONS;++j) r.support[j]=scalar<float>(p+924+4*j);
        r.next_time=scalar<double>(p+1056); r.dt=scalar<double>(p+1064);
        r.gamma=scalar<float>(p+1072); r.lambda=scalar<float>(p+1076); r.reward=scalar<float>(p+1080);
        r.outcome=scalar<float>(p+1084); r.source_seq=scalar<uint32_t>(p+1088); r.next_source_seq=scalar<uint32_t>(p+1092);
        r.own_points=scalar<int32_t>(p+1096); r.opponent_points=scalar<int32_t>(p+1100);
        r.next_own_points=scalar<int32_t>(p+1104); r.next_opponent_points=scalar<int32_t>(p+1108);
        r.terminal_after=scalar<uint32_t>(p+1112); r.applied=scalar<uint32_t>(p+1116); r.value_weight=scalar<float>(p+1120);
    }
    validate(d,rounds); d.digest=sha256(b.data(),b.size()); return d;
}
inline Dataset load(const char* path) { return decode(read_file(path)); }
struct ReplayRow {
    uint32_t index=0, action=0;
    float old_logprob=0, old_value=0;
    std::array<float,LOGITS> logits{};
};
struct Replay {
    std::vector<ReplayRow> rows;
    std::string dataset_sha256, checkpoint_sha256;
    uint64_t seed=0;
};
inline Replay decode_replay(const std::vector<unsigned char>& b,const Dataset& d) {
    const bool identity=d.format_version==3;
    require(b.size()>=REPLAY_HEADER_BYTES && std::memcmp(b.data(),identity?"REKBR003":"REKBR001",8)==0 &&
        scalar<uint32_t>(&b[8])==(identity?3:1) && scalar<uint32_t>(&b[12])==d.rows.size() &&
        scalar<uint32_t>(&b[16])==REPLAY_ROW_BYTES && scalar<uint32_t>(&b[20])==0 &&
        b.size()==REPLAY_HEADER_BYTES+d.rows.size()*REPLAY_ROW_BYTES, "invalid behavior replay header");
    Replay out; out.dataset_sha256=hex(&b[24],32); out.checkpoint_sha256=hex(&b[56],32);
    out.seed=scalar<uint64_t>(&b[88]);
    require(out.dataset_sha256==d.digest && out.seed==(identity?0:73), "behavior replay source mismatch");
    if(identity) require(hex(&b[96],32)==d.identity_sha256 && out.checkpoint_sha256==d.checkpoint_sha256,
        "behavior replay identity mismatch");
    else for(size_t i=96;i<128;++i) require(b[i]==0,"nonzero behavior header reserved bytes");
    for(size_t i=0;i<d.rows.size();++i) {
        const auto* p=b.data()+REPLAY_HEADER_BYTES+i*REPLAY_ROW_BYTES; ReplayRow r;
        r.index=scalar<uint32_t>(p); r.action=scalar<uint32_t>(p+4);
        r.old_logprob=scalar<float>(p+8); r.old_value=scalar<float>(p+12);
        require(r.index==i && r.action==uint32_t(d.rows[i].action) && std::isfinite(r.old_logprob) &&
            r.old_logprob<=0.0001f && std::isfinite(r.old_value), "behavior replay action/value mismatch");
        for(int j=0;j<LOGITS;++j) { r.logits[j]=scalar<float>(p+16+4*j); require(std::isfinite(r.logits[j]),"nonfinite replay logit"); }
        require(r.old_value==r.logits[33], "behavior value head mismatch"); out.rows.push_back(r);
    }
    return out;
}
inline Replay load_replay(const char* path,const Dataset& d) { return decode_replay(read_file(path),d); }
} // namespace rek_authentic
