#include "robot_state.cuh"

#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace {
__constant__ int TO_POLICY[29] = {
    0,6,12,1,7,13,2,8,14,3,9,15,22,4,10,16,23,5,11,17,24,18,25,19,26,20,27,21,28};
__constant__ int TO_MUJOCO[29] = {
    0,3,6,9,13,17,1,4,7,10,14,18,2,5,8,11,15,19,21,23,25,27,12,16,20,22,24,26,28};
__constant__ float DEFAULT_ANGLES[29] = {
    -.312f,0,0,.669f,-.363f,0,-.312f,0,0,.669f,-.363f,0,0,0,0,
    .2f,.2f,0,.6f,0,0,0,.2f,-.2f,0,.6f,0,0,0};
__constant__ float ACTION_SCALE[29] = {
    .350661f,.350661f,.547546f,.350661f,.438577f,.438577f,
    .350661f,.350661f,.547546f,.350661f,.438577f,.438577f,
    .547546f,.438577f,.438577f,
    .438577f,.438577f,.438577f,.438577f,.438577f,.074501f,.074501f,
    .438577f,.438577f,.438577f,.438577f,.438577f,.074501f,.074501f};
__constant__ float KP[29] = {
    99.098428f,99.098428f,40.179238f,99.098428f,28.501246f,28.501246f,
    99.098428f,99.098428f,40.179238f,99.098428f,28.501246f,28.501246f,
    40.179238f,28.501246f,28.501246f,
    14.250623f,14.250623f,14.250623f,14.250623f,14.250623f,16.778327f,16.778327f,
    14.250623f,14.250623f,14.250623f,14.250623f,14.250623f,16.778327f,16.778327f};
__constant__ float KD[29] = {
    6.308802f,6.308802f,2.55789f,6.308802f,1.814446f,1.814446f,
    6.308802f,6.308802f,2.55789f,6.308802f,1.814446f,1.814446f,
    2.55789f,1.814446f,1.814446f,
    .907223f,.907223f,.907223f,.907223f,.907223f,1.068142f,1.068142f,
    .907223f,.907223f,.907223f,.907223f,.907223f,1.068142f,1.068142f};
__constant__ float FORCE_LIMIT[29] = {
    139,139,88,139,25,25,139,139,88,139,25,25,88,25,25,
    25,25,25,25,25,5,5,25,25,25,25,25,5,5};

__device__ double add(double a, double b) { return __dadd_rn(a,b); }
__device__ double sub(double a, double b) { return __dsub_rn(a,b); }
__device__ double mul(double a, double b) { return __dmul_rn(a,b); }

__device__ void quaternion_multiply(const double a[4], const double b[4], double out[4]) {
    out[0] = sub(sub(sub(mul(a[0],b[0]),mul(a[1],b[1])),mul(a[2],b[2])),mul(a[3],b[3]));
    out[1] = sub(add(add(mul(a[0],b[1]),mul(a[1],b[0])),mul(a[2],b[3])),mul(a[3],b[2]));
    out[2] = add(add(sub(mul(a[0],b[2]),mul(a[1],b[3])),mul(a[2],b[0])),mul(a[3],b[1]));
    out[3] = add(sub(add(mul(a[0],b[3]),mul(a[1],b[2])),mul(a[2],b[1])),mul(a[3],b[0]));
}

__device__ void rotation_six(const double q[4], float out[6]) {
    const double w=q[0],x=q[1],y=q[2],z=q[3];
    out[0] = float(sub(1,mul(2,add(mul(y,y),mul(z,z)))));
    out[1] = float(mul(2,sub(mul(x,y),mul(z,w))));
    out[2] = float(mul(2,add(mul(x,y),mul(z,w))));
    out[3] = float(sub(1,mul(2,add(mul(x,x),mul(z,z)))));
    out[4] = float(mul(2,sub(mul(x,z),mul(y,w))));
    out[5] = float(mul(2,add(mul(y,z),mul(x,w))));
}

__device__ float gravity(const float* q, int channel) {
    const double w=q[0],x=-double(q[1]),y=-double(q[2]),z=-double(q[3]);
    if (channel == 0) return float(add(mul(mul(-y,w),2),mul(mul(x,-z),2)));
    if (channel == 1) return float(add(mul(mul(x,w),2),mul(mul(y,-z),2)));
    return float(add(-sub(mul(mul(2,w),w),1),mul(mul(z,-z),2)));
}

__global__ void prepare(RobotState s, const float* base, const float* omega,
    const float* joints, const float* velocities, const float* heading,
    const float* reference, const float* next_reference, const float* root) {
    const std::size_t row = blockIdx.x;
    float* observation = s.encoder_observations + row * 1762;
    for (int i=threadIdx.x;i<1762;i+=blockDim.x) observation[i]=0;
    __syncthreads();
    for (int i=threadIdx.x;i<290;i+=blockDim.x) {
        const int sample=i/29, policy=i%29;
        const auto source=(row*10+sample)*29+TO_POLICY[policy];
        observation[4+i]=reference[source];
        observation[294+i]=float(__ddiv_rn(sub(double(next_reference[source]),double(reference[source])),.02));
    }
    if (threadIdx.x<10) {
        const int sample=threadIdx.x;
        const auto offset=(row*10+sample)*4;
        const double reference_q[4]={root[offset+3],root[offset],root[offset+1],root[offset+2]};
        const double heading_q[4]={heading[row*4],heading[row*4+1],heading[row*4+2],heading[row*4+3]};
        const double conjugate[4]={base[row*4],-double(base[row*4+1]),-double(base[row*4+2]),-double(base[row*4+3])};
        double aligned[4],relative[4];
        quaternion_multiply(heading_q,reference_q,aligned);
        quaternion_multiply(conjugate,aligned,relative);
        rotation_six(relative,observation+601+sample*6);
    }
    if (threadIdx.x>=93 || !s.active[row]) return;
    const int channel=threadIdx.x;
    int group=0,width=3,column=channel;
    float entry;
    if (channel<3) entry=omega[row*3+channel];
    else if (channel<32) {
        group=30;width=29;column=channel-3;
        int joint=TO_POLICY[column];
        entry=float(sub(double(joints[row*29+joint]),double(DEFAULT_ANGLES[joint])));
    } else if (channel<61) {
        group=320;width=29;column=channel-32;
        entry=velocities[row*29+TO_POLICY[column]];
    } else if (channel<90) {
        group=610;width=29;column=channel-61;
        entry=s.last_actions[row*29+column];
    } else {
        group=900;width=3;column=channel-90;
        entry=gravity(base+row*4,column);
    }
    float* history=s.history+row*930+group+column;
    for (int slot=0;slot<9;++slot) history[slot*width]=history[(slot+1)*width];
    history[9*width]=entry;
}

__global__ void decoder_input(RobotState s, const float* tokens) {
    const std::size_t row=blockIdx.x;
    for (int i=threadIdx.x;i<994;i+=blockDim.x)
        s.decoder_observations[row*994+i]=i<64 ? tokens[row*64+i] : s.history[row*930+i-64];
}

__device__ float clamp_action(float value) {
    return value < -100 ? -100 : value > 100 ? 100 : value;
}

__global__ void apply_actions(RobotState s, const float* raw) {
    const std::size_t row=blockIdx.x;
    const int joint=threadIdx.x;
    if (joint>=29 || !s.active[row]) return;
    s.last_actions[row*29+joint]=clamp_action(raw[row*29+joint]);
    const float scaled=__fmul_rn(clamp_action(raw[row*29+TO_MUJOCO[joint]]),ACTION_SCALE[joint]);
    s.targets[row*29+joint]=__fadd_rn(DEFAULT_ANGLES[joint],scaled);
}

__global__ void set_active(RobotState s, const std::uint8_t* enabled) {
    const std::size_t row=blockIdx.x*blockDim.x+threadIdx.x;
    if (row>=s.rows) return;
    s.controller_enabled[row]=!enabled || enabled[row];
    s.active[row]=s.controller_enabled[row] && !s.dampened[row] && !s.resetting[row];
}

__global__ void reset(RobotState s, const std::uint8_t* mask, bool complete) {
    const std::size_t row=blockIdx.x;
    if (mask && !mask[row]) return;
    for (int i=threadIdx.x;i<930;i+=blockDim.x) s.history[row*930+i]=0;
    if (threadIdx.x<29) {
        const auto index=row*29+threadIdx.x;
        s.last_actions[index]=0;s.targets[index]=0;
        if (complete) {
            s.filtered[index]=0;s.retained_targets[index]=0;s.controls[index]=0;
        }
    }
    if (complete && threadIdx.x==0) {
        s.initialized[row]=0;s.dampened[row]=0;s.resetting[row]=0;
        s.controller_enabled[row]=1;s.active[row]=1;
    }
}

__global__ void change_suspension(RobotState s, const std::uint8_t* desired,
    const float* live_controls, bool begin_reset) {
    const std::size_t row=blockIdx.x;
    const bool selected=!desired || desired[row];
    const bool entering=begin_reset
        ? selected && !s.dampened[row] && !s.resetting[row]
        : selected && !s.dampened[row];
    if (entering && threadIdx.x<29)
        s.retained_targets[row*29+threadIdx.x]=live_controls[row*29+threadIdx.x];
    __syncthreads();
    if (threadIdx.x==0) {
        if (begin_reset) s.resetting[row]=s.resetting[row] || selected;
        else s.dampened[row]=selected;
        s.active[row]=s.controller_enabled[row] && !s.dampened[row] && !s.resetting[row];
    }
}

__global__ void drive_prepare(RobotState s, const float* joints, const float* velocities, int substep) {
    const std::size_t row=blockIdx.x;
    const int joint=threadIdx.x;
    const bool active=!s.dampened[row] && !s.resetting[row];
    if (joint<29) {
        const auto index=row*29+joint;
        if (substep%2==0 && active) {
            const float next=s.initialized[row]
                ? __fadd_rn(s.filtered[index],__fmul_rn(__fsub_rn(s.targets[index],s.filtered[index]),.5568627119064331f))
                : s.targets[index];
            s.filtered[index]=next;
        }
        float control;
        if (s.dampened[row]) {
            const double kp=KP[joint],kd=KD[joint];
            const double retained_kp=__fmul_rn(KP[joint],.1f);
            const double retained_kd=__fmul_rn(KD[joint],.1f);
            const double limit=__fmul_rn(FORCE_LIMIT[joint],.1f);
            double force=sub(mul(retained_kp,sub(double(s.retained_targets[index]),double(joints[index]))),
                mul(retained_kd,double(velocities[index])));
            force=force>limit ? limit : force<-limit ? -limit : force;
            const double bias=sub(mul(-kp,double(joints[index])),mul(kd,double(velocities[index])));
            control=float(__ddiv_rn(sub(force,bias),kp));
        } else if (s.resetting[row]) control=s.retained_targets[index];
        else {
            control=s.filtered[index];
            const auto side_joint=(row%2)*29+joint;
            if (s.joint_limited[side_joint]) {
                const float lower=s.joint_ranges[side_joint*2],upper=s.joint_ranges[side_joint*2+1];
                control=control>upper ? upper : control<lower ? lower : control;
            }
        }
        s.controls[index]=control;
    }
    __syncthreads();
    if (threadIdx.x==0 && substep%2==0 && active) s.initialized[row]=1;
}

void cuda_check(cudaError_t status) {
    if (status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void set_error(char* error,std::size_t capacity,const char* detail) {
    if (error && capacity) { std::snprintf(error,capacity,"%s",detail);error[capacity-1]='\0'; }
}
} // namespace

void robot_state_destroy(RobotState* state) {
    if (!state) return;
    cudaFree(state->float_storage);
    cudaFree(state->flag_storage);
    delete state;
}

RobotState* robot_state_create(std::size_t rows,cudaStream_t stream,
    const std::uint8_t* limited,const float* ranges,char* error,std::size_t capacity) {
    std::unique_ptr<RobotState,decltype(&robot_state_destroy)> state(nullptr,robot_state_destroy);
    try {
        constexpr std::size_t floats_per_row=1762+994+930+29*5;
        if (!rows || rows%2 || rows>2147483647 || rows>(std::numeric_limits<std::size_t>::max()/4-116)/floats_per_row
            || !limited || !ranges) throw std::runtime_error("invalid paired robot-state metadata");
        for (int i=0;i<58;++i) if (limited[i] &&
            (!std::isfinite(ranges[i*2]) || !std::isfinite(ranges[i*2+1]) || ranges[i*2]>ranges[i*2+1]))
            throw std::runtime_error("invalid joint range");
        state.reset(new RobotState);
        state->rows=rows;state->stream=stream;
        const auto float_count=rows*floats_per_row+116;
        cuda_check(cudaMalloc(&state->float_storage,float_count*sizeof(float)));
        cuda_check(cudaMalloc(&state->flag_storage,rows*5+58));
        cuda_check(cudaMemsetAsync(state->float_storage,0,float_count*sizeof(float),stream));
        cuda_check(cudaMemsetAsync(state->flag_storage,0,rows*5+58,stream));
        float* cursor=static_cast<float*>(state->float_storage);
        state->encoder_observations=cursor;cursor+=rows*1762;
        state->decoder_observations=cursor;cursor+=rows*994;
        state->history=cursor;cursor+=rows*930;
        state->last_actions=cursor;cursor+=rows*29;
        state->targets=cursor;cursor+=rows*29;
        state->filtered=cursor;cursor+=rows*29;
        state->retained_targets=cursor;cursor+=rows*29;
        state->controls=cursor;cursor+=rows*29;
        state->joint_ranges=cursor;
        auto* flags=static_cast<std::uint8_t*>(state->flag_storage);
        state->active=flags;flags+=rows;
        state->controller_enabled=flags;flags+=rows;
        state->initialized=flags;flags+=rows;
        state->dampened=flags;flags+=rows;
        state->resetting=flags;flags+=rows;
        state->joint_limited=flags;
        cuda_check(cudaMemcpyAsync(state->joint_ranges,ranges,116*sizeof(float),cudaMemcpyHostToDevice,stream));
        cuda_check(cudaMemcpyAsync(state->joint_limited,limited,58,cudaMemcpyHostToDevice,stream));
        cuda_check(robot_state_set_active(state.get(),nullptr));
        cuda_check(cudaStreamSynchronize(stream));
        set_error(error,capacity,"");
        return state.release();
    } catch (const std::exception& exception) { set_error(error,capacity,exception.what());return nullptr; }
}

cudaError_t robot_state_set_active(RobotState* s,const std::uint8_t* enabled) {
    if (!s) return cudaErrorInvalidValue;
    set_active<<<unsigned((s->rows+255)/256),256,0,s->stream>>>(*s,enabled);
    return cudaPeekAtLastError();
}

cudaError_t robot_state_prepare(RobotState* s,const float* base,const float* omega,
    const float* joints,const float* velocities,const float* heading,
    const float* reference,const float* next,const float* root) {
    if (!s || !base || !omega || !joints || !velocities || !heading || !reference || !next || !root)
        return cudaErrorInvalidValue;
    prepare<<<unsigned(s->rows),256,0,s->stream>>>(*s,base,omega,joints,velocities,heading,reference,next,root);
    return cudaPeekAtLastError();
}
cudaError_t robot_state_decoder_input(RobotState* s,const float* tokens) {
    if (!s || !tokens) return cudaErrorInvalidValue;
    decoder_input<<<unsigned(s->rows),256,0,s->stream>>>(*s,tokens);
    return cudaPeekAtLastError();
}
cudaError_t robot_state_apply_actions(RobotState* s,const float* actions) {
    if (!s || !actions) return cudaErrorInvalidValue;
    apply_actions<<<unsigned(s->rows),32,0,s->stream>>>(*s,actions);
    return cudaPeekAtLastError();
}
cudaError_t robot_state_reset_history(RobotState* s,const std::uint8_t* rows) {
    if (!s) return cudaErrorInvalidValue;
    reset<<<unsigned(s->rows),256,0,s->stream>>>(*s,rows,false);
    return cudaPeekAtLastError();
}
cudaError_t robot_state_complete_reset(RobotState* s,const std::uint8_t* rows) {
    if (!s) return cudaErrorInvalidValue;
    reset<<<unsigned(s->rows),256,0,s->stream>>>(*s,rows,true);
    return cudaPeekAtLastError();
}
cudaError_t robot_state_set_dampened(RobotState* s,const std::uint8_t* desired,const float* controls) {
    if (!s || !desired || !controls) return cudaErrorInvalidValue;
    change_suspension<<<unsigned(s->rows),32,0,s->stream>>>(*s,desired,controls,false);
    return cudaPeekAtLastError();
}
cudaError_t robot_state_begin_reset(RobotState* s,const std::uint8_t* rows,const float* controls) {
    if (!s || !controls) return cudaErrorInvalidValue;
    change_suspension<<<unsigned(s->rows),32,0,s->stream>>>(*s,rows,controls,true);
    return cudaPeekAtLastError();
}
cudaError_t robot_state_drive_prepare(RobotState* s,const float* joints,const float* velocities,int substep) {
    if (!s || !joints || !velocities || substep<0 || substep>9) return cudaErrorInvalidValue;
    drive_prepare<<<unsigned(s->rows),32,0,s->stream>>>(*s,joints,velocities,substep);
    return cudaPeekAtLastError();
}
