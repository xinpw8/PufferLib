#include "fast_mode_config.h"
#include <cassert>
#include <cstring>
#include <memory>

int main(){
    using Json=std::unique_ptr<cJSON,decltype(&cJSON_Delete)>;
    setenv("REK_FAST_OPPONENT","recovered_bot1_v1",1);
    setenv("REK_FAST_OBSERVATION","rendered_pose_v1",1);
    auto legacy=rek5_modes::configure(nullptr);
    assert(legacy.opponent=="v4_scripted"&&legacy.observation=="v4_logical");
    assert(std::strcmp(std::getenv("REK_FAST_OPPONENT"),"v4_scripted")==0);
    assert(std::strcmp(std::getenv("REK_FAST_OBSERVATION"),"v4_logical")==0);
    Json config(cJSON_Parse("{\"opponent_controller\":\"recovered_bot1_v1\",\"observation_mode\":\"rendered_pose_v1\"}"),cJSON_Delete);
    auto current=rek5_modes::configure(config.get());
    assert(current.opponent=="recovered_bot1_v1"&&current.observation=="rendered_pose_v1");
    assert(std::strcmp(std::getenv("REK_FAST_OPPONENT"),current.opponent.c_str())==0);
    assert(std::strcmp(std::getenv("REK_FAST_OBSERVATION"),current.observation.c_str())==0);
    const auto serialized="{\"test\":true"+rek5_modes::json_fields(current,"recovered_hit_rules_v2")+"}";
    Json result(cJSON_Parse(serialized.c_str()),cJSON_Delete);
    assert(result&&std::strcmp(cJSON_GetObjectItem(result.get(),"scoring_mode")->valuestring,"recovered_hit_rules_v2")==0);
    int rejected=0;
    for(const char* bad:{"[]","{\"opponent_controller\":false}","{\"observation_mode\":\"unknown\"}"}){
        Json invalid(cJSON_Parse(bad),cJSON_Delete);
        try{rek5_modes::configure(invalid.get());}catch(const std::runtime_error&){rejected++;}
    }
    assert(rejected==3);
    std::puts("PASS explicit mode configuration, ambient override isolation, JSON identity, invalid-mode rejection");
}
