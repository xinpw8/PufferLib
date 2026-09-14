import test from 'node:test';
import assert from 'node:assert/strict';
import {buildWindows,catalogLookup,classifyKernel} from './profile_training.mjs';

const event=(name,start,end,streamId=1)=>({name,start,end,streamId,deviceId:0,contextId:1});
function context(anchors=[],catalog={modules:[]}){return {...buildWindows(anchors),catalog:catalogLookup(catalog)};}

test('three exact GPU markers define PPO regions and preserve controller boundaries',()=>{
  const value=context([
    event('puf_stamp(unsigned long long *)',0,1),event('puf_stamp(unsigned long long *)',10,11),event('puf_stamp(unsigned long long *)',20,21),
    event('<unnamed>::pack_encoder(const float *,float *,size_t)',30,31,2),event('<unnamed>::apply_actions(RobotState,const float *)',50,51,2)
  ]);
  assert.equal(classifyKernel(event('custom_copy_kernel()',2,3),value).category,'ppo_training');
  assert.equal(classifyKernel(event('sgemm_library_kernel()',35,40,2),value).category,'robot_controller');
  assert.equal(classifyKernel(event('sgemm_library_kernel()',55,60,2),value).category,'unclassified_gemm');
  assert.deepEqual(value.warnings,[]);
});
test('cached physics entries are exact catalog matches; similar names are unknown',()=>{
  const value=context([],{modules:[{name:'wp_mujoco_warp._src.solver_123',kernels:[{symbol:'_solve_123_cuda_kernel_forward'}]}]});
  assert.deepEqual(classifyKernel(event('_solve_123_cuda_kernel_forward',0,1),value),{category:'physics',detail:'constraint_solver',evidence:'exact_cached_kernel_catalog'});
  assert.equal(classifyKernel(event('_solve_wrong_cuda_kernel_forward',0,1),value).category,'unclassified');
});
test('missing graph markers are surfaced rather than fabricated',()=>{
  const value=buildWindows([event('puf_stamp(unsigned long long *)',0,1),event('<unnamed>::pack_encoder(float*)',2,3,2)]);
  assert.equal(value.training.size,0);assert.equal(value.controller.size,0);assert.equal(value.warnings.length,2);
});
test('controller source signatures and unknown GEMMs remain distinguishable',()=>{
  const value=context();
  assert.equal(classifyKernel(event('<unnamed>::drive_prepare(RobotState,float*,int)',0,1),value).category,'robot_controller');
  assert.equal(classifyKernel(event('magma_sgemmEx_kernel<float>()',0,1),value).category,'unclassified_gemm');
  assert.equal(classifyKernel(event('mystery_kernel()',0,1),value).evidence,'unknown');
  assert.equal(classifyKernel(event('memset32',0,1),value).category,'gpu_memory_maintenance');
});
