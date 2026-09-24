#include "bc_dataset.h"
#include <iostream>
int main(int argc,char**argv){try{
    rek_bc::require(argc==2,"dataset path required");const auto d=rek_bc::load(argv[1]);
    rek_bc::require(d.rows.size()==11985&&d.sequences.size()==4,"human chronology mismatch");
    int labels[2]={},rows[2]={},kick[2]={},history=0;
    for(int i=0;i<223;i++){bool unknown=(i>=176&&i<=183)||i==202||i==204||i==205;rek_bc::require(d.feature_mask[i]==!unknown,"partial feature mask mismatch");}
    for(const auto&r:d.rows){
        rows[r.split]++;history+=int(r.obs[203]);
        rek_bc::require((r.action==-1&&r.weight==0)||(r.action>=16&&r.action<=32&&r.weight==1),"conditional target mismatch");
        if(r.weight){labels[r.split]++;kick[r.split]+=r.action==17;}
        for(int a=0;a<33;a++)rek_bc::require(r.support[a]==(a>=16?1:0),"conditional support mismatch");
        for(int i=0;i<223;i++)if(!d.feature_mask[i])rek_bc::require(r.obs[i]==0,"unknown feature not zero");
        for(int i:{72,158})rek_bc::require(r.obs[i]>=0&&r.obs[i]<=1,"tilt range");
        rek_bc::require(r.obs[203]==0||r.obs[203]==1,"history availability");
        if(!r.obs[203])rek_bc::require(r.obs[9]==0&&r.obs[95]==0,"missing history rates");
    }
    rek_bc::require(rows[0]==5993&&rows[1]==5992&&labels[0]==77&&labels[1]==109&&kick[0]==16&&kick[1]==2&&history==11981,"source counts mismatch");
    std::cout<<"{\"native_dataset_reader\":\"passed\",\"rows\":11985,\"train_attack_labels\":77,\"heldout_attack_labels\":109,\"train_category17\":16,\"heldout_category17\":2,\"sequences\":4,\"gpu_used\":false}\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}}
