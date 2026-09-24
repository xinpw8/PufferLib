#include "bc_dataset.h"
#include <iostream>
int main(int argc,char**argv){try{
 rek_bc::require(argc==2,"dataset required");const auto d=rek_bc::load(argv[1]);rek_bc::require(d.rows.size()==11985&&d.sequences.size()==4,"chronology");
 int labels[2]={},counts[2][33]={};
 for(int c=0;c<223;c++)rek_bc::require(d.feature_mask[c]==!((c>=176&&c<=183)||c==202||c==204||c==205),"mask");
 for(const auto&r:d.rows){rek_bc::require((r.action==-1&&r.weight==0)||(r.action>=2&&r.action<=15&&r.weight==1),"movement label");if(r.weight){labels[r.split]++;counts[r.split][r.action]++;}for(int a=0;a<33;a++)rek_bc::require(r.support[a]==(a>=2&&a<=15?1:0),"support");for(int c=0;c<223;c++)if(!d.feature_mask[c])rek_bc::require(r.obs[c]==0,"unknown feature");}
 const int expected[2][14]={{499,68,75,51,315,212,120,3,12,0,0,0,0,0},{932,42,0,0,392,107,40,112,0,0,0,0,0,0}};
 for(int s=0;s<2;s++)for(int a=2;a<=15;a++)rek_bc::require(counts[s][a]==expected[s][a-2],"original movement class count");
 rek_bc::require(labels[0]==1355&&labels[1]==1625,"label totals");std::cout<<"{\"passed\":true,\"rows\":11985,\"sequences\":4,\"train_labels\":1355,\"heldout_labels\":1625,\"gpu_used\":false}\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}}
