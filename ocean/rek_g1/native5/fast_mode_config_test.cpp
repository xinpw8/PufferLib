#include "fast_mode_config.h"
#include <cassert>
#include <cstring>
#include <memory>
#include <utility>

int main(){
    using Json=std::unique_ptr<cJSON,decltype(&cJSON_Delete)>;
    setenv("REK_FAST_OPPONENT","recovered_bot1_v1",1);
    setenv("REK_FAST_OBSERVATION","rendered_pose_v1",1);
    setenv("REK_FAST_GEOMETRY","primitive_samples_v1",1);
    setenv("REK_FAST_CONTACT_SUBSTEPS","16",1);
    auto legacy=rek5_modes::configure(nullptr);
    assert(legacy.opponent=="v4_scripted"&&legacy.observation=="v4_logical");
    assert(legacy.geometry=="bounding_spheres"&&legacy.contact_substeps==0);
    assert(std::strcmp(std::getenv("REK_FAST_OPPONENT"),"v4_scripted")==0);
    assert(std::strcmp(std::getenv("REK_FAST_OBSERVATION"),"v4_logical")==0);
    assert(std::strcmp(std::getenv("REK_FAST_GEOMETRY"),"bounding_spheres")==0);
    // The runtime requires a valid positive sampling count even when unused.
    assert(std::strcmp(std::getenv("REK_FAST_CONTACT_SUBSTEPS"),"4")==0);
    Json config(cJSON_Parse("{\"opponent_controller\":\"recovered_bot1_v1\",\"observation_mode\":\"rendered_pose_v1\"}"),cJSON_Delete);
    auto current=rek5_modes::configure(config.get());
    assert(current.opponent=="recovered_bot1_v1"&&current.observation=="rendered_pose_v1");
    assert(current.geometry=="bounding_spheres"&&current.contact_substeps==0);
    assert(std::strcmp(std::getenv("REK_FAST_OPPONENT"),current.opponent.c_str())==0);
    assert(std::strcmp(std::getenv("REK_FAST_OBSERVATION"),current.observation.c_str())==0);
    auto check_identity=[](const rek5_modes::Identity& identity){
        const auto serialized="{\"test\":true"+rek5_modes::json_fields(identity,"recovered_hit_rules_v2")+"}";
        Json result(cJSON_Parse(serialized.c_str()),cJSON_Delete);
        assert(result);
        for(const auto& entry:{std::make_pair("opponent_controller",identity.opponent),
                std::make_pair("observation_mode",identity.observation),
                std::make_pair("geometry_mode",identity.geometry),
                std::make_pair("scoring_mode",std::string("recovered_hit_rules_v2"))}){
            const auto* value=cJSON_GetObjectItem(result.get(),entry.first);
            assert(cJSON_IsString(value)&&entry.second==value->valuestring);
        }
        const auto* samples=cJSON_GetObjectItem(result.get(),"contact_substeps");
        assert(cJSON_IsNumber(samples)&&samples->valuedouble==identity.contact_substeps);
    };
    check_identity(legacy);check_identity(current);
    setenv("REK_FAST_GEOMETRY","invalid_ambient_geometry",1);
    setenv("REK_FAST_CONTACT_SUBSTEPS","invalid_ambient_count",1);
    Json primitive_default(cJSON_Parse("{\"geometry_mode\":\"primitive_samples_v1\"}"),cJSON_Delete);
    auto primitive=rek5_modes::configure(primitive_default.get());
    assert(primitive.geometry=="primitive_samples_v1"&&primitive.contact_substeps==4);
    assert(std::strcmp(std::getenv("REK_FAST_GEOMETRY"),"primitive_samples_v1")==0);
    assert(std::strcmp(std::getenv("REK_FAST_CONTACT_SUBSTEPS"),"4")==0);
    check_identity(primitive);
    for(int samples:{1,8,16}){
        const auto text="{\"geometry_mode\":\"primitive_samples_v1\",\"contact_substeps\":"+std::to_string(samples)+"}";
        Json configured(cJSON_Parse(text.c_str()),cJSON_Delete);
        auto explicit_samples=rek5_modes::configure(configured.get());
        assert(explicit_samples.contact_substeps==samples);
        assert(std::strcmp(std::getenv("REK_FAST_CONTACT_SUBSTEPS"),std::to_string(samples).c_str())==0);
        check_identity(explicit_samples);
    }
    Json unused_samples(cJSON_Parse("{\"geometry_mode\":\"bounding_spheres\",\"contact_substeps\":16}"),cJSON_Delete);
    auto sphere=rek5_modes::configure(unused_samples.get());
    assert(sphere.geometry=="bounding_spheres"&&sphere.contact_substeps==0);
    check_identity(sphere);
    int rejected=0;
    for(const char* bad:{"[]","{\"opponent_controller\":false}","{\"observation_mode\":\"unknown\"}",
            "{\"geometry_mode\":false}","{\"geometry_mode\":\"unknown\"}","{\"geometry_mode\":null}",
            "{\"contact_substeps\":0}","{\"contact_substeps\":-1}","{\"contact_substeps\":17}",
            "{\"contact_substeps\":1.5}","{\"contact_substeps\":\"4\"}","{\"contact_substeps\":true}",
            "{\"contact_substeps\":null}","{\"contact_substeps\":1e999}"}){
        Json invalid(cJSON_Parse(bad),cJSON_Delete);
        assert(invalid);
        try{rek5_modes::configure(invalid.get());}catch(const std::runtime_error&){rejected++;}
        assert(std::strcmp(std::getenv("REK_FAST_GEOMETRY"),"bounding_spheres")==0);
        assert(std::strcmp(std::getenv("REK_FAST_CONTACT_SUBSTEPS"),"16")==0);
    }
    assert(rejected==14);
    std::puts("PASS explicit mode configuration, geometry defaults and boundaries, ambient override isolation, complete JSON identity, 14 invalid-mode/count rejections");
}
