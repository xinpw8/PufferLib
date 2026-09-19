'use strict';

// Compile the actual, pinned advantage kernel in isolation. This is a
// deterministic CUDA diagnostic; it never constructs a policy or trains one.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const {spawnSync} = require('node:child_process');
const EXPECTED = '20c5f33b036ad43c66492ca6be82265675b8f620bfcb989eb433d2f04f27cd98';

function kernelSource(text) {
  const start = text.indexOf('constexpr int ADV_VEC_WIDTH');
  const signature = text.indexOf('__global__ void puff_advantage(', start);
  const body = text.indexOf('{', signature);
  if (start < 0 || signature < 0 || body < 0) throw new Error('Pinned kernel markers missing');
  let depth = 1, end = body + 1;
  for (; end < text.length && depth; end++) {
    if (text[end] === '{') depth++;
    else if (text[end] === '}') depth--;
  }
  if (depth) throw new Error('Pinned kernel is incomplete');
  return text.slice(start, end);
}

const harness = String.raw`
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
using precision_t = float;
__host__ __device__ inline float to_float(float x) { return x; }
KERNEL_SOURCE
PATCHED_KERNEL_SOURCE
static void require(cudaError_t e) { if(e != cudaSuccess) { std::fprintf(stderr,"%s\n",cudaGetErrorString(e)); std::exit(2); } }
int main() {
    constexpr int H=8, N=4;
    float *v,*r,*d,*a,*out;
    require(cudaMallocManaged(&v,N*H*sizeof(float)));
    require(cudaMallocManaged(&r,N*H*sizeof(float)));
    require(cudaMallocManaged(&d,N*H*sizeof(float)));
    require(cudaMallocManaged(&a,N*H*sizeof(float)));
    require(cudaMallocManaged(&out,N*H*sizeof(float)));
    for(int i=0;i<N*H;i++)v[i]=r[i]=d[i]=a[i]=out[i]=0;
    // Row 0: reward and true termination are stored in final observed slot.
    r[7]=5; d[7]=1;
    // Row 1: final executed action pays five and terminates, stored as slot 0
    // of the next rollout. The previous rollout has no slot for it.
    // Row 2 models that next rollout. Reward[0] is not consumed by the kernel.
    r[2*H]=5;d[2*H]=1;
    // Row 3: an internal terminal prevents credit from a later episode leaking.
    r[3*H+3]=2;d[3*H+3]=1;r[3*H+7]=5;d[3*H+7]=1;
    puff_advantage<<<1,64>>>(v,r,d,nullptr,a,out,1,1,1,1,N,H);
    require(cudaGetLastError());require(cudaDeviceSynchronize());
    bool controls=a[6]==5 && a[7]==0 && a[3*H+2]==2 && a[3*H+3]==5;
    bool omitted=a[H+7]==0 && a[2*H]==0;
    std::printf("{\"schema\":\"rek.pufferlib5.boundary_probe.v1\",\"actual_pinned_cuda_kernel\":true,\"horizon\":8,\"internal_terminal_advantage\":%.9g,\"internal_terminal_expected\":5,\"boundary_terminal_last_action_advantage\":%.9g,\"boundary_terminal_expected\":5,\"next_rollout_reward_zero_slot_advantage\":%.9g,\"internal_terminal_no_leak\":%.9g,\"positive_controls_pass\":%s,\"boundary_omission_reproduced\":%s,\"boundary_contract_pass\":false}\n",a[6],a[H+7],a[2*H],a[3*H+2],controls?"true":"false",omitted?"true":"false");
    require(cudaFree(v));require(cudaFree(r));require(cudaFree(d));require(cudaFree(a));require(cudaFree(out));
    PATCHED_TEST
    return controls&&omitted?0:1;
}
`;

const fixedTest = String.raw`
    // Test exact patched source with terminal, continuing and rollout-cut
    // transitions. The pinned API has no independent truncation flag.
    constexpr int F=5;
    float *fv,*fr,*fd,*fa,*fo,*bv,*br,*bd;
    require(cudaMallocManaged(&fv,F*H*sizeof(float)));
    require(cudaMallocManaged(&fr,F*H*sizeof(float)));
    require(cudaMallocManaged(&fd,F*H*sizeof(float)));
    require(cudaMallocManaged(&fa,F*H*sizeof(float)));
    require(cudaMallocManaged(&fo,F*H*sizeof(float)));
    require(cudaMallocManaged(&bv,F*sizeof(float)));
    require(cudaMallocManaged(&br,F*sizeof(float)));
    require(cudaMallocManaged(&bd,F*sizeof(float)));
    for(int j=0;j<F;j++) {
        bv[j]=float(j+2);br[j]=float(j+1);bd[j]=(j==0||j==4)?1:0;
        for(int t=0;t<H;t++) {
            int i=j*H+t;fv[i]=float((t+j)%3);fr[i]=float((t*j)%4);fd[i]=0;fa[i]=fo[i]=0;
        }
    }
    br[0]=5;bv[0]=100;fv[H-1]=0;
    br[1]=2;bv[1]=7;fv[2*H-1]=0;
    fd[2*H+4]=1;fr[2*H+4]=3; // Internal true terminal.
    fixed::puff_advantage<<<1,64>>>(fv,fr,fd,nullptr,fa,fo,.9f,.8f,1,1,F,H,bv,br,bd);
    require(cudaGetLastError());require(cudaDeviceSynchronize());
    double largest=0;
    for(int j=0;j<F;j++) {
        double next_adv=0;
        for(int t=H-1;t>=0;t--) {
            int i=j*H+t;
            double next_r=t==H-1?br[j]:fr[i+1];
            double next_v=t==H-1?bv[j]:fv[i+1];
            bool done=t==H-1?bd[j]!=0:fd[i+1]!=0;
            double expected=next_r+(done?0:.9*next_v)-fv[i]+(done?0:.9*.8*next_adv);
            largest=std::fmax(largest,std::fabs(fa[i]-expected));
            largest=std::fmax(largest,std::fabs(fo[i]-(fv[i]+expected)));
            next_adv=expected;
        }
    }
    bool fixed_pass=largest<2e-5 && fa[H-1]==5 && std::fabs(fa[2*H-1]-8.3f)<1e-5;
    std::printf("{\"schema\":\"rek.pufferlib5.fixed_boundary_probe.v1\",\"actual_patched_cuda_kernel\":true,\"terminal_final_action_advantage\":%.9g,\"terminal_expected\":5,\"nonterminal_final_action_advantage\":%.9g,\"nonterminal_expected\":8.3,\"all_40_advantages_and_returns_max_error\":%.9g,\"boundary_contract_pass\":%s,\"external_truncation_supported\":false}\n",fa[H-1],fa[2*H-1],largest,fixed_pass?"true":"false");
    require(cudaFree(fv));require(cudaFree(fr));require(cudaFree(fd));require(cudaFree(fa));require(cudaFree(fo));require(cudaFree(bv));require(cudaFree(br));require(cudaFree(bd));
    if(!fixed_pass)return 1;
`;

function main(args) {
  if (args.length !== 2 && args.length !== 3) throw new Error('Usage: node probe_pufferlib5_temporal_credit.cjs PINNED_ALGO_CU NEW_OUTPUT_DIRECTORY [PATCHED_ALGO_CU]');
  const source = fs.readFileSync(args[0]);
  const hash = crypto.createHash('sha256').update(source).digest('hex');
  if (hash !== EXPECTED) throw new Error(`Unexpected algo.cu SHA-256: ${hash}`);
  const output = path.resolve(args[1]);
  fs.mkdirSync(output); // Never overwrite an existing diagnostic directory.
  const cu = path.join(output, 'pinned_advantage_probe.cu'), exe = path.join(output, 'pinned_advantage_probe');
  const patched = args[2] ? fs.readFileSync(args[2]) : null;
  const generated = harness.replace('KERNEL_SOURCE', kernelSource(source.toString('utf8')))
    .replace('PATCHED_KERNEL_SOURCE', patched ? 'namespace fixed {\n'+kernelSource(patched.toString('utf8'))+'\n}' : '')
    .replace('PATCHED_TEST', patched ? fixedTest : '');
  fs.writeFileSync(cu, generated);
  const nvcc = process.env.NVCC || '/usr/local/cuda/bin/nvcc';
  const compile = spawnSync(nvcc, ['-std=c++17', '-O2', '-arch=sm_121', cu, '-o', exe], {encoding:'utf8'});
  fs.writeFileSync(path.join(output, 'compile.stdout.txt'), compile.stdout || '');
  fs.writeFileSync(path.join(output, 'compile.stderr.txt'), compile.stderr || '');
  if (compile.status !== 0) throw new Error(compile.error?.message || compile.stderr || 'CUDA compile failed');
  const run = spawnSync(exe, [], {encoding:'utf8'});
  fs.writeFileSync(path.join(output, 'result.jsonl'), run.stdout || '');
  fs.writeFileSync(path.join(output, 'stderr.txt'), run.stderr || '');
  fs.writeFileSync(path.join(output, 'source.json'), JSON.stringify({source:path.resolve(args[0]),sha256:hash,
    patched_source:args[2] ? path.resolve(args[2]) : null,
    patched_sha256:patched ? crypto.createHash('sha256').update(patched).digest('hex') : null},null,2)+'\n');
  process.stdout.write(run.stdout || '');
  if (run.status !== 0) throw new Error(run.error?.message || run.stderr || 'CUDA probe failed');
}
if (require.main === module) {
  try { main(process.argv.slice(2)); }
  catch(error) { process.stderr.write(error.message+'\n');process.exitCode=2; }
}
module.exports = {kernelSource, EXPECTED};
