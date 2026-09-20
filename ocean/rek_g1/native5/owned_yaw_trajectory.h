#pragma once
#include "authentic_trajectory.h"
#include "owned_yaw_observation.h"

// Explicit v2 format. Existing REKRL001/REKBR001 loaders reject these files.
// Common non-observation validation is reused only after checking v2 identity.
namespace rek_owned_yaw_trajectory {
inline rek_authentic::Dataset decode(const std::vector<unsigned char>& bytes) {
    using rek_authentic::require;
    require(bytes.size()>=rek_authentic::HEADER_BYTES &&
        std::memcmp(bytes.data(),"REKRL002",8)==0 && rek_authentic::scalar<uint32_t>(&bytes[8])==2,
        "expected explicit owned-yaw-v2 trajectory");
    auto normalized=bytes;
    std::memcpy(normalized.data(),"REKRL001",8);uint32_t version=1;
    std::memcpy(normalized.data()+8,&version,sizeof(version));
    auto data=rek_authentic::decode(normalized);
    for(const auto& row:data.rows) {
        const float yaw=row.obs[rek_owned_yaw::kColumn];
        require((yaw==-1||yaw==0||yaw==1) && (row.obs[182]==0||row.obs[182]==1) &&
            row.obs[182]==row.obs[183] && (row.obs[182]==1||yaw==0), "invalid owned-yaw column/busy contract");
    }
    data.digest=rek_authentic::sha256(bytes.data(),bytes.size());
    return data;
}
inline rek_authentic::Dataset load(const char* path) { return decode(rek_authentic::read_file(path)); }
inline rek_authentic::Replay decode_replay(const std::vector<unsigned char>& bytes,const rek_authentic::Dataset& data) {
    rek_authentic::require(bytes.size()>=rek_authentic::REPLAY_HEADER_BYTES &&
        std::memcmp(bytes.data(),"REKBR002",8)==0 && rek_authentic::scalar<uint32_t>(&bytes[8])==2,
        "expected explicit owned-yaw-v2 replay");
    auto normalized=bytes;std::memcpy(normalized.data(),"REKBR001",8);uint32_t version=1;
    std::memcpy(normalized.data()+8,&version,sizeof(version));
    return rek_authentic::decode_replay(normalized,data);
}
inline void verify_column_only_upgrade(const std::vector<unsigned char>& old_bytes,
        const std::vector<unsigned char>& new_bytes) {
    rek_authentic::require(old_bytes.size()==new_bytes.size(),"v2 upgrade changed byte count");
    const auto old_data=rek_authentic::decode(old_bytes);const auto new_data=decode(new_bytes);
    rek_authentic::require(old_data.rows.size()==new_data.rows.size(),"v2 upgrade changed row count");
    auto restored=new_bytes;
    std::memcpy(restored.data(),old_bytes.data(),12);
    for(size_t i=0;i<old_data.rows.size();++i) {
        const size_t offset=rek_authentic::HEADER_BYTES+i*rek_authentic::ROW_BYTES+32+4*rek_owned_yaw::kColumn;
        rek_authentic::require(rek_authentic::scalar<uint32_t>(old_bytes.data()+offset)==0,
            "legacy column187 is not positive zero");
        std::memcpy(restored.data()+offset,old_bytes.data()+offset,sizeof(float));
    }
    rek_authentic::require(restored==old_bytes,"v2 upgrade changed data outside schema header/column187");
}
}
