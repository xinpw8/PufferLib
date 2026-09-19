#include "contact_potential_loader.h"
#include <iostream>
#include <vector>

int main(int argc,char** argv){
    try{
        using namespace rek5_contact_potential;
        int checks=0;
        auto check=[&](bool value){++checks;if(!value)throw std::runtime_error("test "+std::to_string(checks));};
        Model m{};m.count=2;
        for(float& scale:m.scale)scale=1;
        m.samples[0][0]=.5f;m.samples[0][1]=1;
        m.samples[1][0]=.7f;m.samples[1][1]=0;m.samples[1][2]=1;
        check(potential(m,.5f,0)==0);
        check(fabsf(potential(m,.7f,float(M_PI/2)))<1e-6f);
        check(potential(m,2,0)<potential(m,1,0));
        check(fabsf(potential(m,.5f,float(M_PI))-potential(m,.5f,float(-M_PI)))<1e-6f);
        // Unity+Z target from Unity+X facing has evidence bearing -pi/2,
        // opposite the reflected candidate's +pi/2.
        float ux=0,uz=1,forward_x=1,forward_z=0;
        float evidence=atan2f(ux,uz)-atan2f(forward_x,forward_z);
        float candidate=atan2f(uz,ux)-atan2f(forward_z,forward_x);
        check(fabsf(evidence+candidate)<1e-6f);
        // The discounted shaping sum telescopes to -Phi(start) with terminal
        // Phi=0, regardless of intermediate actions/loops or episode length.
        const float gamma=.999f;
        for(int n:{1,5,250,6000}){
            double accumulated=0,discount=1;float initial=potential(m,1.8f,.8f),previous=initial;
            for(int t=0;t<n;t++){
                const float next=potential(m,.6f+.4f*sinf(float(t)),cosf(float(t)));
                accumulated+=discount*shaping_delta(previous,next,t==n-1,gamma);
                discount*=gamma;previous=next;
            }
            check(fabs(accumulated+initial)<2e-5);
        }
        if(argc==2){
            const auto loaded=load(argv[1]);
            check(loaded.model.count==12);check(hash_string(loaded.file_sha256));
            for(int i=0;i<loaded.model.count;i++){
                const auto* s=loaded.model.samples[i];
                check(fabsf(potential(loaded.model,s[0],atan2f(s[2],s[1])))<2e-5f);
            }
            std::ifstream stream(argv[1]);std::string text((std::istreambuf_iterator<char>(stream)),{});
            auto rejected=[&](const char* field_name,const char* replacement){
                auto* json=cJSON_Parse(text.c_str());
                cJSON_ReplaceItemInObjectCaseSensitive(json,field_name,cJSON_Parse(replacement));
                char* changed=cJSON_PrintUnformatted(json);bool caught=false;
                try{parse(changed);}catch(const std::exception&){caught=true;}
                cJSON_free(changed);cJSON_Delete(json);check(caught);
            };
            rejected("feature_scale","[0,1,1]");rejected("fit_round","2");
            rejected("sample_count","65");rejected("sample_count","12.5");
            rejected("model_id","\"invalid\"");rejected("holdout_event_ids","[\"R1-E0001\"]");
        }
        std::cout<<"contact_potential_checks="<<checks<<" passed\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
