import {test} from 'node:test';
import assert from 'node:assert/strict';
import {parameters,kernels,blockDimension,sharedBytes} from './catalog_kernels.mjs';
test('nested generated vector and matrix argument types remain intact',()=>{
  const p=parameters('wp::launch_bounds_t dim, wp::array_t<wp::vec_t<3, wp::float32>> var_xpos, bool var_sparse');
  assert.equal(p.length,3);assert.deepEqual(p[1],{type:'wp::array_t<wp::vec_t<3, wp::float32>>',name:'var_xpos'});
});
test('only exported kernel entry points are inventoried',()=>{
  const result=kernels('static void helper(int x) {}\nextern "C" __global__ void actual(wp::launch_bounds_t dim, wp::array_t<wp::mat_t<3, 3, wp::float32>> var_x) { }');
  assert.equal(result.length,1);assert.equal(result[0].symbol,'actual');assert.equal(result[0].parameters.length,2);
});
test('malformed nested declarations are rejected',()=>assert.throws(()=>parameters('wp::array_t<wp::vec_t<3, float> value')));
test('generated block dimensions are part of the launch ABI',()=>{
  assert.equal(blockDimension('\n#define WP_TILE_BLOCK_DIM 32\n'),32);
  assert.throws(()=>blockDimension(''));
  assert.throws(()=>blockDimension('#define WP_TILE_BLOCK_DIM 2048\n'));
});
test('tiled dynamic shared memory comes from matching kernel metadata',()=>{
  assert.equal(sharedBytes('k',{k_smem_bytes:6144}),6144);
  assert.equal(sharedBytes('k',{k_smem_bytes:0}),0);
  assert.throws(()=>sharedBytes('k',{}));
  assert.throws(()=>sharedBytes('k',{k_smem_bytes:-1}));
});
