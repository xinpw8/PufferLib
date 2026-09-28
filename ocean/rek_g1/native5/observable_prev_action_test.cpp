#include "observable_prev_action.h"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace pa=rek_observable_prev_action;
namespace {
unsigned checks=0;
void check(bool condition){++checks;if(!condition)throw std::runtime_error("previous-action contract check failed");}
void preserved(const std::array<float,223>& before,const std::array<float,223>& after){
    for(int i=0;i<223;i++)if(!pa::history_column(i))
        check(std::memcmp(&before[i],&after[i],sizeof(float))==0);
}
}
int main(){try{
    std::array<unsigned char,223> mask{};pa::feature_mask(mask.data());
    int base=0,added=0,total=0;
    for(int i=0;i<223;i++){
        base+=rek_observable_balance::structurally_available(i);
        added+=pa::history_column(i);total+=mask[i];
        check(!pa::history_column(i)||!rek_observable_balance::structurally_available(i));
        check(!pa::history_column(i)||i>=176);
    }
    check(base==166&&added==34&&total==200);
    check(pa::column(-1)==-1&&pa::column(33)==-1);
    for(int a=0;a<33;a++)for(int b=0;b<33;b++)check((a==b)==(pa::column(a)==pa::column(b)));
    std::array<float,223> source{};
    for(int i=0;i<223;i++)source[i]=float(i-111)/64.f;
    source[0]=-0.f;
    pa::History history;
    auto row=source;check(pa::write(row.data(),history));check(pa::valid_features(row.data()));preserved(source,row);
    check(row[pa::column(0)]==0&&row[pa::kAvailableColumn]==0);
    for(int action=0;action<33;action++){
        const auto before=history;
        row=source;check(pa::write(row.data(),before));
        check(pa::record(history,float(action)));
        // The already-produced observation is not retroactively changed.
        check(row[pa::kAvailableColumn]==float(before.available));
        auto next=source;check(pa::write(next.data(),history));check(pa::valid_features(next.data()));
        for(int a=0;a<33;a++)check(next[pa::column(a)]==float(a==action));
        check(next[pa::kAvailableColumn]==1);preserved(source,next);
        auto copy=next;check(pa::write(copy.data(),history));check(std::memcmp(copy.data(),next.data(),sizeof(next))==0);
    }
    const auto saved=history;
    for(float invalid:{-1.f,33.f,.5f,std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}){
        check(!pa::record(history,invalid));check(history.action==saved.action&&history.available==saved.available);
    }
    check(pa::record(history,0));row=source;check(pa::write(row.data(),history));
    check(row[pa::column(0)]==1&&row[pa::kAvailableColumn]==1);
    // No record/clear occurs for rejected delivery, repeated read, value-only
    // bootstrap, rollout boundary or counted-fall body reset.
    for(int repeat=0;repeat<1025;repeat++){
        auto next=source;check(pa::write(next.data(),history));check(next==row);
    }
    pa::clear(history);row=source;check(pa::write(row.data(),history));
    for(int a=0;a<33;a++)check(row[pa::column(a)]==0);
    check(row[pa::kAvailableColumn]==0&&pa::valid_features(row.data()));
    check(pa::record(history,1));check(pa::write(row.data(),history));
    check(row[pa::column(1)]==1&&row[pa::column(0)]==0);
    for(pa::History invalid:std::array<pa::History,3>{{{0,2},{-1,1},{33,1}}}){
        auto next=source;check(!pa::write(next.data(),invalid));check(next==source);
    }
    check(!pa::write(nullptr,history)&&!pa::valid_features(nullptr));
    for(int i=0;i<34;i++){
        auto broken=row;broken[i==33?pa::kAvailableColumn:pa::column(i)]=.25f;
        check(!pa::valid_features(broken.data()));
    }
    row[pa::column(2)]=1;check(!pa::valid_features(row.data()));
    std::printf("{\"cpu_tests\":\"passed\",\"checks\":%u,\"base_features\":166,\"added_features\":34,\"joint_columns_reused\":false,\"cuda_calls\":0}\n",checks);
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
