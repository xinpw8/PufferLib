#pragma once
// Per-arena aggregation for lite_fall_dataset.cu, shared with the CPU tests.
// Every feature is derived from exported raw observations, which the physical
// runtime and the lite compact runtime publish with the same field meanings.
#include "lite_falls.h"
#include <stdint.h>
#include <cstdio>
#include <string>
#include <vector>

namespace rek_lite_dataset {
using namespace rek_lite_falls;
constexpr int kCells=kMoveBins*kMoveBins*kDistanceBins*kClosingBins*2;
// Raw observation fields (223 per fighter row).
constexpr int kX=0,kY=1,kQw=3,kQx=4,kQy=5,kQz=6,kOpponent=86,kTilt=72,kRatio=73,kPhase=79,kGrace=83,kEvents=85;
constexpr int kForward=176,kStrafe=177,kYaw=178,kRoute=179,kAttacking=182,kOpponentScoreDelta=184+34;

struct Settings {
    float distance_edges[kDistanceBins-1],closing_edges[kClosingBins-1];
    uint32_t durations[17];
    int max_delay;
};
struct RowState { int32_t route,move_ticks,phase,onset_tick,onset_class; };
struct Sink {
    unsigned long long *exposures,*onsets;           // [kCells]
    unsigned long long *recover,*fallen,*censored;   // [3][max_delay+1]
    double* observation;  // falling tilt, falling ratio, falling n, fallen tilt, fallen ratio, fallen n
};
REK_LITE_FN int cell_index(int own,int opponent,int distance,int closing,int struck){
    return (((own*kMoveBins+opponent)*kDistanceBins+distance)*kClosingBins+closing)*2+struck;
}
// raw: both fighter rows of one arena after a step. previous_distance < 0
// means no previous sample. Add is called as add(pointer, value).
template<class Add>
REK_LITE_FN void observe_arena(const float* raw,RowState* rows,float& previous_distance,const Settings& s,
        bool terminal,int tick,const Sink& sink,const Add& add){
    const float* o[2]={raw,raw+223};
    const float distance=hypotf(o[0][kOpponent+kX]-o[0][kX],o[0][kOpponent+kY]-o[0][kY]);
    const float closing=previous_distance>=0?(previous_distance-distance)/.02f:0.f;
    previous_distance=distance;
    int move[2];float progress[2];bool moving[2],turning[2];
    for(int side=0;side<2;side++){
        RowState& r=rows[side];const float* x=o[side];
        const int route=x[kAttacking]!=0?int(x[kRoute]):-1;
        // A move's first exported tick has already advanced once.
        if(route!=r.route){r.route=route;r.move_ticks=route>=0?1:0;}else if(route>=0)r.move_ticks++;
        move[side]=route_move(route);
        const uint32_t duration=move[side]>=0?s.durations[move[side]]:1u;
        progress[side]=move[side]>=0?float(r.move_ticks)/float(duration>0?duration:1u):0.f;
        moving[side]=x[kForward]!=0||x[kStrafe]!=0;turning[side]=x[kYaw]!=0;
    }
    for(int side=0;side<2;side++){
        RowState& r=rows[side];const float* x=o[side];
        const int before=r.phase,phase=int(x[kPhase]);const uint32_t events=uint32_t(x[kEvents]);
        const bool started=(events&FallingStarted)||(before==Upright&&phase!=Upright);
        // Exposure: upright before this step and outside the post-reset grace.
        if(before==Upright&&x[kGrace]<=0.f){
            const int own=move_bin(move[side],progress[side],moving[side],turning[side]);
            const int opp=move_bin(move[side^1],progress[side^1],moving[side^1],turning[side^1]);
            const int cell=cell_index(own,opp,bucket(distance,s.distance_edges,kDistanceBins-1),
                bucket(closing,s.closing_edges,kClosingBins-1),x[kOpponentScoreDelta]>0?1:0);
            add(sink.exposures+cell,1ull);
            if(started)add(sink.onsets+cell,1ull);
        }
        if(started){r.onset_tick=tick;r.onset_class=move_class(move[side]);}
        // An open fall resolves by recovery or by becoming fallen, or is
        // censored when a spawn reset or the round end clears it first.
        if(before==Falling||started){
            int elapsed=tick-r.onset_tick;elapsed=elapsed<0?0:elapsed>s.max_delay?s.max_delay:elapsed;
            const size_t slot=size_t(r.onset_class)*size_t(s.max_delay+1)+size_t(elapsed);
            if(events&BecameFallen)add(sink.fallen+slot,1ull);
            else if(events&FallingCleared)add(sink.recover+slot,1ull);
            else if(phase==Upright||terminal)add(sink.censored+slot,1ull);
        }
        if(phase==Falling){add(sink.observation+0,double(x[kTilt]));add(sink.observation+1,double(x[kRatio]));add(sink.observation+2,1.0);}
        if(phase==Fallen){add(sink.observation+3,double(x[kTilt]));add(sink.observation+4,double(x[kRatio]));add(sink.observation+5,1.0);}
        r.phase=terminal?Upright:phase;
    }
}

struct HostAdd {
    void operator()(unsigned long long* p,unsigned long long v) const {*p+=v;}
    void operator()(double* p,double v) const {*p+=v;}
};
struct HostSink {
    std::vector<unsigned long long> exposures,onsets,recover,fallen,censored;
    double observation[6]{};
    explicit HostSink(int max_delay):exposures(kCells),onsets(kCells),recover(size_t(3)*(max_delay+1)),
        fallen(size_t(3)*(max_delay+1)),censored(size_t(3)*(max_delay+1)){}
    Sink view(){return {exposures.data(),onsets.data(),recover.data(),fallen.data(),censored.data(),observation};}
};

inline void write_dataset(FILE* f,const std::string& dataset_id,const Settings& s,const HostSink& d,const std::string& provenance){
    const auto array=[&](const char* name,const std::vector<unsigned long long>& v){
        std::fprintf(f,"\"%s\":[",name);for(size_t i=0;i<v.size();i++)std::fprintf(f,i?",%llu":"%llu",v[i]);std::fprintf(f,"]");
    };
    const auto rows=[&](const char* name,const std::vector<unsigned long long>& v){
        const size_t width=size_t(s.max_delay+1);std::fprintf(f,"\"%s\":[",name);
        for(int c=0;c<kOutcomeClasses;c++){std::fprintf(f,c?",[":"[");for(size_t i=0;i<width;i++)std::fprintf(f,i?",%llu":"%llu",v[c*width+i]);std::fprintf(f,"]");}
        std::fprintf(f,"]");
    };
    unsigned long long counts[3][3]{};const size_t width=size_t(s.max_delay+1);
    for(int c=0;c<3;c++)for(size_t i=0;i<width;i++){counts[c][0]+=d.recover[c*width+i];counts[c][1]+=d.fallen[c*width+i];counts[c][2]+=d.censored[c*width+i];}
    std::fprintf(f,"{\"schema\":\"rek.lite_fall_dataset.v1\",\"dataset_id\":\"%s\",",dataset_id.c_str());
    std::fprintf(f,"\"dimensions\":{\"move_bins\":%d,\"distance_bins\":%d,\"closing_bins\":%d,\"struck\":2,\"phase_bins\":%d},",kMoveBins,kDistanceBins,kClosingBins,kPhaseBins);
    std::fprintf(f,"\"distance_edges_m\":[");for(int i=0;i<kDistanceBins-1;i++)std::fprintf(f,i?",%.9g":"%.9g",s.distance_edges[i]);
    std::fprintf(f,"],\"closing_edges_m_s\":[");for(int i=0;i<kClosingBins-1;i++)std::fprintf(f,i?",%.9g":"%.9g",s.closing_edges[i]);std::fprintf(f,"],");
    array("exposures",d.exposures);std::fprintf(f,",");array("onsets",d.onsets);
    std::fprintf(f,",\"outcomes\":{\"max_delay_ticks\":%d,",s.max_delay);
    rows("recover_delay_histogram",d.recover);std::fprintf(f,",");rows("fallen_delay_histogram",d.fallen);std::fprintf(f,",");
    rows("censored_delay_histogram",d.censored);
    std::fprintf(f,",\"counts_by_class\":[[%llu,%llu,%llu],[%llu,%llu,%llu],[%llu,%llu,%llu]]},",
        counts[0][0],counts[0][1],counts[0][2],counts[1][0],counts[1][1],counts[1][2],counts[2][0],counts[2][1],counts[2][2]);
    const double* o=d.observation;
    std::fprintf(f,"\"observation_means\":{\"falling_tilt_degrees\":%.9g,\"falling_height_ratio\":%.9g,\"fallen_tilt_degrees\":%.9g,\"fallen_height_ratio\":%.9g},",
        o[2]>0?o[0]/o[2]:60.0,o[2]>0?o[1]/o[2]:.5,o[5]>0?o[3]/o[5]:92.0,o[5]>0?o[4]/o[5]:.12);
    std::fprintf(f,"\"provenance\":%s}\n",provenance.c_str());
}
}  // namespace rek_lite_dataset
