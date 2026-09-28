#include "robot_state.cuh"

// The unchanged existing C implementation is the observation/history oracle.
// Inference stubs are unreachable because this test calls its assembly stages.
extern "C" {
#include "gear_sonic_native_batch.c"
int gear_sonic_ort_encode(GearSonicOrtBatch*,float*,float*,char*,size_t) { return 0; }
int gear_sonic_ort_decode(GearSonicOrtBatch*,float*,float*,char*,size_t) { return 0; }
}

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(bool condition,const char* message) { if (!condition) throw std::runtime_error(message); }
void ck(cudaError_t status) { if (status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status)); }
template<class T> struct Buffer {
    T* data=nullptr;std::size_t count=0;
    explicit Buffer(std::size_t n):count(n) { ck(cudaMalloc(reinterpret_cast<void**>(&data),n*sizeof(T))); }
    ~Buffer() { cudaFree(data); }
    void upload(const std::vector<T>& value) { check(value.size()==count,"upload length");ck(cudaMemcpy(data,value.data(),count*sizeof(T),cudaMemcpyHostToDevice)); }
};
std::vector<float> read(const float* values,std::size_t count) {
    std::vector<float> result(count);ck(cudaMemcpy(result.data(),values,count*4,cudaMemcpyDeviceToHost));return result;
}
std::size_t checked_values=0;
void equal(const std::vector<float>& actual,const std::vector<float>& expected,const char* label,double tolerance=0) {
    check(actual.size()==expected.size(),"comparison length");
    for (std::size_t i=0;i<actual.size();++i) {
        ++checked_values;
        if (!std::isfinite(actual[i]) || std::abs(double(actual[i])-expected[i])>tolerance) {
            std::fprintf(stderr,"%s index=%zu actual=%.9g expected=%.9g\n",label,i,actual[i],expected[i]);
            throw std::runtime_error(label);
        }
    }
}
std::vector<float> flatten(const std::array<GearSonicNativeBatch,8>& states,float* GearSonicNativeBatch::*field,std::size_t width) {
    std::vector<float> result(8*width);
    for (std::size_t row=0;row<8;++row) std::copy_n(states[row].*field,width,result.data()+row*width);
    return result;
}
const float TEST_KP[29]={99.098428f,99.098428f,40.179238f,99.098428f,28.501246f,28.501246f,
    99.098428f,99.098428f,40.179238f,99.098428f,28.501246f,28.501246f,40.179238f,28.501246f,28.501246f,
    14.250623f,14.250623f,14.250623f,14.250623f,14.250623f,16.778327f,16.778327f,
    14.250623f,14.250623f,14.250623f,14.250623f,14.250623f,16.778327f,16.778327f};
const float TEST_KD[29]={6.308802f,6.308802f,2.55789f,6.308802f,1.814446f,1.814446f,
    6.308802f,6.308802f,2.55789f,6.308802f,1.814446f,1.814446f,2.55789f,1.814446f,1.814446f,
    .907223f,.907223f,.907223f,.907223f,.907223f,1.068142f,1.068142f,
    .907223f,.907223f,.907223f,.907223f,.907223f,1.068142f,1.068142f};
const float TEST_LIMIT[29]={139,139,88,139,25,25,139,139,88,139,25,25,88,25,25,
    25,25,25,25,25,5,5,25,25,25,25,25,5,5};
} // namespace

int main() {
    try {
        constexpr std::size_t rows=8;
        cudaStream_t stream=nullptr;ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        std::uint8_t limited[58];float ranges[116];
        for (int i=0;i<58;++i) { limited[i]=i%29!=1;ranges[i*2]=-.2f;ranges[i*2+1]=.3f; }
        char error[1024]={};
        std::unique_ptr<RobotState,decltype(&robot_state_destroy)> state(
            robot_state_create(rows,stream,limited,ranges,error,sizeof(error)),robot_state_destroy);
        check(state!=nullptr,error);
        std::array<GearSonicNativeBatch,rows> cpu{};
        for (auto& row:cpu) check(gear_sonic_native_batch_open(&row,1,error,sizeof(error)),error);
        Buffer<float> base(rows*4),omega(rows*3),joints(rows*29),velocity(rows*29),heading(rows*4);
        Buffer<float> reference(rows*290),next(rows*290),root(rows*40),tokens(rows*64),actions(rows*29);
        Buffer<std::uint8_t> mask(rows),enabled(rows);
        std::mt19937 generator(7301);
        std::uniform_real_distribution<float> distribution(-1,1);
        auto random=[&](std::size_t count,float scale=1) {
            std::vector<float> v(count);for (float& x:v) x=distribution(generator)*scale;return v;
        };
        auto normalized=[&](std::size_t count) {
            auto values=random(count*4);
            for (std::size_t q=0;q<count;++q) {
                double norm=0;for (int i=0;i<4;++i) norm+=double(values[q*4+i])*values[q*4+i];
                norm=std::sqrt(norm);for (int i=0;i<4;++i) values[q*4+i]=float(values[q*4+i]/norm);
            }return values;
        };
        auto ht=random(rows*64);tokens.upload(ht);
        for (int tick=0;tick<18;++tick) {
            std::vector<std::uint8_t> active(rows),reset_mask(rows,0);
            for (std::size_t row=0;row<rows;++row) active[row]=row%4==0 || row%4==3 || (row%4==1 ? tick%2==0 : tick<7);
            if (tick==12) {
                reset_mask[0]=reset_mask[1]=reset_mask[6]=1;mask.upload(reset_mask);
                ck(robot_state_reset_history(state.get(),mask.data));
                for (std::size_t row=0;row<rows;++row) if (reset_mask[row]) {
                    gear_sonic_native_batch_reset(&cpu[row]);
                    std::fill_n(cpu[row].targets_mujoco,29,0);
                }
            }
            enabled.upload(active);ck(robot_state_set_active(state.get(),enabled.data));
            auto hb=normalized(rows),hh=normalized(rows),hr=normalized(rows*10);
            auto ho=random(rows*3),hq=random(rows*29),hv=random(rows*29),hp=random(rows*290,.2f),hn=random(rows*290,.2f),ha=random(rows*29,120);
            base.upload(hb);heading.upload(hh);root.upload(hr);omega.upload(ho);joints.upload(hq);velocity.upload(hv);
            reference.upload(hp);next.upload(hn);actions.upload(ha);
            std::vector<double> db(hb.begin(),hb.end()),dh(hh.begin(),hh.end()),domega(ho.begin(),ho.end()),dq(hq.begin(),hq.end()),dv(hv.begin(),hv.end());
            auto cpu_prepare=[&] {
                for (std::size_t row=0;row<rows;++row) {
                    GearSonicNativeStateInput input{db.data()+row*4,domega.data()+row*3,dq.data()+row*29,dv.data()+row*29,dh.data()+row*4,nullptr};
                    GearSonicNativeReferenceInput ref{hp.data()+row*290,hn.data()+row*290,hr.data()+row*40};
                    if (active[row]) append_history(&cpu[row],0,&input);
                    build_encoder_reference_row(&cpu[row],0,&ref,&input);
                    std::copy_n(ht.data()+row*64,64,cpu[row].tokens);build_decoder_row(&cpu[row],0);
                }
            };
            auto gpu_prepare=[&] {
                ck(robot_state_prepare(state.get(),base.data,omega.data,joints.data,velocity.data,heading.data,reference.data,next.data,root.data));
                ck(robot_state_decoder_input(state.get(),tokens.data));
            };
            auto cpu_actions=[&] { for (std::size_t row=0;row<rows;++row) if (active[row]) {
                std::copy_n(ha.data()+row*29,29,cpu[row].actions_policy);transform_actions(&cpu[row]);
            }};
            cpu_prepare();gpu_prepare();ck(cudaStreamSynchronize(stream));
            equal(read(state->encoder_observations,rows*1762),flatten(cpu,&GearSonicNativeBatch::encoder_observations,1762),"encoder_vs_existing_native",1e-7);
            equal(read(state->decoder_observations,rows*994),flatten(cpu,&GearSonicNativeBatch::decoder_observations,994),"decoder_history_vs_existing_native");
            cpu_actions();ck(robot_state_apply_actions(state.get(),actions.data));ck(cudaStreamSynchronize(stream));
            equal(read(state->targets,rows*29),flatten(cpu,&GearSonicNativeBatch::targets_mujoco,29),"targets_vs_existing_native");
            equal(read(state->last_actions,rows*29),flatten(cpu,&GearSonicNativeBatch::clipped_actions_policy,29),"last_actions_vs_existing_native");
            if (tick==17) {
                cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
                ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
                gpu_prepare();ck(robot_state_apply_actions(state.get(),actions.data));
                ck(cudaStreamEndCapture(stream,&graph));ck(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
                for (int i=0;i<3;++i) { cpu_prepare();cpu_actions();ck(cudaGraphLaunch(executable,stream)); }
                ck(cudaStreamSynchronize(stream));
                equal(read(state->decoder_observations,rows*994),flatten(cpu,&GearSonicNativeBatch::decoder_observations,994),"captured_history_vs_existing_native");
                ck(cudaGraphExecDestroy(executable));ck(cudaGraphDestroy(graph));
            }
        }
        ck(robot_state_complete_reset(state.get(),nullptr));
        auto set_targets=[&](float value) { std::vector<float> target(rows*29,value);ck(cudaMemcpy(state->targets,target.data(),target.size()*4,cudaMemcpyHostToDevice)); };
        auto zero=std::vector<float>(rows*29,0);joints.upload(zero);velocity.upload(zero);
        set_targets(2);ck(robot_state_drive_prepare(state.get(),joints.data,velocity.data,0));ck(cudaStreamSynchronize(stream));
        std::vector<float> expected(rows*29,.3f);for (std::size_t row=0;row<rows;++row) expected[row*29+1]=2;
        equal(read(state->controls,rows*29),expected,"active_joint_clamp");
        equal(read(state->filtered,rows*29),std::vector<float>(rows*29,2),"unclipped_filter_retained");
        set_targets(-2);ck(robot_state_drive_prepare(state.get(),joints.data,velocity.data,1));ck(cudaStreamSynchronize(stream));
        equal(read(state->controls,rows*29),expected,"odd_substep_holds_filter");
        ck(robot_state_drive_prepare(state.get(),joints.data,velocity.data,2));ck(cudaStreamSynchronize(stream));
        const float filtered=2.0f+(-2.0f-2.0f)*.5568627119064331f;
        equal(read(state->filtered,rows*29),std::vector<float>(rows*29,filtered),"even_substep_filter");
        std::vector<std::uint8_t> reset_rows(rows,0),dampened(rows,0);reset_rows[0]=1;dampened[1]=1;
        auto live=std::vector<float>(rows*29,1.5f);actions.upload(live);mask.upload(reset_rows);enabled.upload(dampened);
        ck(robot_state_begin_reset(state.get(),mask.data,actions.data));
        ck(robot_state_set_dampened(state.get(),enabled.data,actions.data));
        joints.upload(std::vector<float>(rows*29,.4f));velocity.upload(std::vector<float>(rows*29,-3));
        ck(robot_state_drive_prepare(state.get(),joints.data,velocity.data,3));ck(cudaStreamSynchronize(stream));
        auto controls=read(state->controls,rows*29);
        equal(std::vector<float>(controls.begin(),controls.begin()+29),std::vector<float>(29,1.5f),"reset_retained_target");
        std::vector<float> dampened_expected(29);
        for (int j=0;j<29;++j) {
            const double kp=TEST_KP[j],kd=TEST_KD[j];
            const double retained_kp=float(TEST_KP[j]*.1f),retained_kd=float(TEST_KD[j]*.1f),limit=float(TEST_LIMIT[j]*.1f);
            double force=retained_kp*(1.5-double(.4f))-retained_kd*(-3.0);
            force=std::max(-limit,std::min(limit,force));
            const double bias=-kp*double(.4f)-kd*(-3.0);
            dampened_expected[j]=float((force-bias)/kp);
        }
        equal(std::vector<float>(controls.begin()+29,controls.begin()+58),dampened_expected,"dampened_retained_force");
        ck(robot_state_complete_reset(state.get(),mask.data));ck(cudaStreamSynchronize(stream));
        controls=read(state->controls,rows*29);
        equal(std::vector<float>(controls.begin(),controls.begin()+29),std::vector<float>(29,0),"selected_drive_reset");
        equal(std::vector<float>(controls.begin()+29,controls.begin()+58),dampened_expected,"unselected_drive_reset_isolation");
        for (auto& row:cpu) gear_sonic_native_batch_close(&row);
        state.reset();ck(cudaStreamDestroy(stream));
        std::printf("native_robot_state_pass checked_values=%zu rows=8 ticks=18 history_capture_replays=3 python_used=0\n",checked_values);
        return 0;
    } catch (const std::exception& exception) { std::fprintf(stderr,"native_robot_state_failed: %s\n",exception.what());return 1; }
}
