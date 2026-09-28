#include "solver_schedule.h"
#include <mujoco/mujoco.h>
#include <stdexcept>

namespace rek_mjgpu {
namespace {
// Specializations present in the original MuJoCo-Warp cache. KernelProgram
// verifies their source/PTX hashes, argument ABI and launch shared memory.
constexpr const char* solver_module = "wp_mujoco_warp._src.solver_b116efb";
constexpr const char* jaref_module = "wp_solve_init_jaref__locals__kernel_95afaf28_95afaf2";
constexpr const char* jv_module = "wp_linesearch_jv_fused__locals__kernel_b495af1b_b495af1";
constexpr const char* constraint_module = "wp_update_constraint_efc__locals__kernel_81015163_8101516";
constexpr const char* gauss_module = "wp_update_constraint_gauss_cost__locals__kernel_b7364b7d_b7364b7";
constexpr const char* mul_module = "wp_mul_m_sparse__locals___mul_m_sparse_aa423e43_aa423e4";
constexpr const char* line_module = "wp_linesearch_iterative__locals__kernel_a4af0200_d13ae26";
constexpr const char* cholesky_module = "wp_update_gradient_cholesky_blocked__locals__kernel_d7c8d7e5_4145b3d";

void require(bool ok, const char* reason) {
    if (!ok) throw std::runtime_error(std::string("native Newton solver: ")+reason);
}
void check(CUresult result, const char* where) {
    if (result != CUDA_SUCCESS) {
        const char* reason=nullptr; cuGetErrorString(result,&reason);
        throw std::runtime_error(std::string(where)+": "+(reason?reason:"CUDA error"));
    }
}
void kernel(ScheduleSpec& out, const char* module, const char* entry,
            std::initializer_list<int> dimensions,
            std::initializer_list<ScheduleParameter> parameters,
            int block=0, int shared=0) {
    ScheduleNode node;
    node.module=module; node.entry_prefix=entry;
    node.bounds=launch_bounds(dimensions); node.parameters=parameters;
    node.block_dim=block; node.shared_bytes=shared;
    out.push_back(std::move(node));
}
void zero(ScheduleSpec& out, const char* name) {
    ScheduleNode node; node.kind=ScheduleNode::Kind::Zero; node.destination=name;
    out.push_back(std::move(node));
}
void copy(ScheduleSpec& out, const char* destination, const char* source) {
    ScheduleNode node; node.kind=ScheduleNode::Kind::Copy;
    node.destination=destination; node.source=source; out.push_back(std::move(node));
}
void mul_m(ScheduleSpec& out, const ModelData& d, const char* vector, const char* result) {
    kernel(out,mul_module,"mul_m_sparse__locals___mul_m_sparse_6b17933c_cuda_kernel_forward",
        {d.integer("d.nworld"),d.integer("m.nv")},
        {"m.qM_mulm_rowadr","m.qM_mulm_col","m.qM_mulm_madr","d.qM",vector,"solver.done",result});
}
void update_constraint(ScheduleSpec& out, const ModelData& d) {
    const int worlds=d.integer("d.nworld"), rows=d.integer("d.njmax");
    kernel(out,solver_module,"update_constraint_init_cost_",{worlds},
        {"solver.cost","solver.done","solver.gauss","solver.cost","solver.prev_cost"});
    kernel(out,constraint_module,"update_constraint_efc__locals__kernel_9a9ed9bb_cuda_kernel_forward",{worlds,rows},
        {"m.opt.impratio_invsqrt","d.ne","d.nf","d.nefc","d.contact.friction","d.contact.dim",
         "d.contact.efc_address","d.efc.type","d.efc.id","d.efc.D","d.efc.frictionloss","d.nacon",
         "solver.Jaref","solver.done","d.efc.force","d.efc.state","solver.cost",
         "solver.changed_efc_ids","solver.changed_efc_count"});
    zero(out,"d.qfrc_constraint");
    kernel(out,solver_module,"update_constraint_init_qfrc_constraint_sparse_",{worlds,rows},
        {"d.nefc","d.efc.J_rownnz","d.efc.J_rowadr","d.efc.J_colind","d.efc.J",
         "d.efc.force","solver.done","d.qfrc_constraint"});
    kernel(out,gauss_module,"update_constraint_gauss_cost__locals__kernel_706504b4_cuda_kernel_forward",{worlds,4},
        {"d.qacc","d.qfrc_smooth","d.qacc_smooth","d.efc.Ma","solver.done","solver.gauss","solver.cost"});
}
void update_gradient(ScheduleSpec& out, const ModelData& d, int sm_count) {
    const int worlds=d.integer("d.nworld"), nv=d.integer("m.nv"), rows=d.integer("d.njmax");
    kernel(out,solver_module,"update_gradient_zero_grad_dot_",{worlds},{"solver.done","solver.grad_dot"});
    kernel(out,solver_module,"update_gradient_grad_",{worlds,nv},
        {"d.qfrc_smooth","d.qfrc_constraint","d.efc.Ma","solver.done","solver.grad","solver.grad_dot"});
    zero(out,"solver.h");
    kernel(out,solver_module,"_JTDAJ_sparse_",{worlds,rows},
        {"d.nefc","d.efc.J_rownnz","d.efc.J_rowadr","d.efc.J_colind","d.efc.J","d.efc.D",
         "d.efc.state","solver.done","solver.h"});
    kernel(out,solver_module,"update_gradient_set_h_qM_lower_sparse_",
        {worlds,d.array_ref("m.qM_fullm_i").shape[0]},
        {"m.qM_fullm_i","m.qM_fullm_j","d.qM","solver.done","solver.h"});
    const int triangle=d.array_ref("m.dof_tri_row").shape[0];
    const int blocks=(sm_count*6*256+triangle-1)/triangle;
    const int contacts=d.integer("d.naconmax"), perblock=(contacts+blocks-1)/blocks;
    kernel(out,solver_module,"update_gradient_JTCJ_sparse_",{blocks,triangle},
        {"m.opt.impratio_invsqrt","m.dof_tri_row","m.dof_tri_col","d.contact.dist",
         "d.contact.includemargin","d.contact.friction","d.contact.dim","d.contact.efc_address",
         "d.contact.worldid","d.efc.J_rownnz","d.efc.J_rowadr","d.efc.J_colind","d.efc.J",
         "d.efc.D","d.efc.state",contacts,"d.nacon","solver.Jaref","solver.done",perblock,blocks,"solver.h"});
    kernel(out,solver_module,"padding_h_",{worlds,d.integer("m.nv_pad")-nv},
        {nv,"solver.done","solver.h"});
    // launch_tiled(dim=worlds,block_dim=32) adds the lane dimension. A plain
    // {worlds} launch gives incorrect wp.tid() coordinates for tiled kernels.
    kernel(out,cholesky_module,"update_gradient_cholesky_blocked__locals__kernel_3b712445_cuda_kernel_forward",{worlds,32},
        {"solver.done","solver.grad_3d","solver.h","solver.hfactor","solver.Mgrad_3d"},32,6144);
}
void linesearch(ScheduleSpec& out, const ModelData& d) {
    const int worlds=d.integer("d.nworld"), rows=d.integer("d.njmax");
    mul_m(out,d,"solver.search","solver.mv");
    kernel(out,solver_module,"linesearch_zero_jv_",{worlds,rows},
        {"d.nefc","solver.done","solver.jv"});
    kernel(out,jv_module,"linesearch_jv_fused__locals__kernel_2d0c3c0f_cuda_kernel_forward",{worlds,rows,4},
        {"d.nefc","d.efc.J_rownnz","d.efc.J_rowadr","d.efc.J_colind","d.efc.J",
         "solver.search","solver.done","solver.jv"});
    kernel(out,line_module,"linesearch_iterative__locals__kernel_2e02bc1f_cuda_kernel_forward",{worlds,32},
        {d.integer("m.nv"),"m.opt.tolerance","m.opt.ls_tolerance","m.opt.impratio_invsqrt",
         "m.stat.meaninertia","d.ne","d.nf","d.nefc","d.qfrc_smooth","d.contact.friction",
         "d.contact.dim","d.contact.efc_address","d.efc.type","d.efc.id","d.efc.J_rownnz",
         "d.efc.J_rowadr","d.efc.J_colind","d.efc.J","d.efc.D","d.efc.frictionloss",rows,
         "d.nacon","solver.Jaref","solver.search","solver.search_dot","solver.gauss",
         "solver.mv","solver.jv","solver.quad","solver.done","d.qacc","d.efc.Ma",
         "solver.Jaref","solver.jv","solver.quad"},32,96);
}
}

SolverSpec build_newton_solver(ModelData& d) {
    const int worlds=d.integer("d.nworld"), nv=d.integer("m.nv"), padded=d.integer("m.nv_pad");
    const int rows=d.integer("d.njmax");
    require(worlds>0&&nv==70&&padded==80&&rows>0,"cached kernels require nv=70, nv_pad=80 and positive capacities");
    require(d.integer("m.opt.solver")==mjSOL_NEWTON&&d.integer("m.opt.cone")==mjCONE_ELLIPTIC&&d.integer("m.is_sparse"),
            "requires sparse Newton with elliptic contacts");
    require(d.integer("m.opt.ls_iterations")==50&&!d.integer("m.opt.ls_parallel"),
            "cached line search requires 50 iterative line-search iterations");
    require(d.integer("m.opt.graph_conditional"),"native solve requires GPU conditional graph support");
    require(d.integer("m.block_dim.linesearch_iterative")==32&&d.integer("m.block_dim.update_gradient_cholesky_blocked")==32,
            "cached tiled solver kernels require 32 threads");
    require(!d.contains("solver.nsolving"),"solver context was already constructed");
    CUdevice device; int sm_count=0;
    check(cuCtxGetDevice(&device),"solver device");
    check(cuDeviceGetAttribute(&sm_count,CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,device),"solver SM count");
    require(sm_count>0,"invalid GPU multiprocessor count");
    for(const char* name:{"search_dot","gauss","cost","prev_cost","grad_dot","alpha","beta"})
        d.allocate(std::string("solver.")+name,{worlds});
    d.allocate("solver.done",{worlds},Element::U8);
    for(const char* name:{"grad","Mgrad"})d.allocate(std::string("solver.")+name,{worlds,padded});
    for(const char* name:{"search","mv","prev_grad","prev_Mgrad"})d.allocate(std::string("solver.")+name,{worlds,nv});
    for(const char* name:{"Jaref","jv"})d.allocate(std::string("solver.")+name,{worlds,rows});
    d.allocate("solver.quad",{worlds,rows},Element::F32,3);
    d.allocate("solver.quad_gauss",{worlds},Element::F32,3);
    d.allocate("solver.changed_efc_ids",{worlds,rows},Element::I32);
    d.allocate("solver.changed_efc_count",{worlds},Element::I32);
    for(const char* name:{"h","hfactor"})d.allocate(std::string("solver.")+name,{worlds,padded,padded});
    d.reshape_alias("solver.grad_3d","solver.grad",{worlds,padded,1});
    d.reshape_alias("solver.Mgrad_3d","solver.Mgrad",{worlds,padded,1});
    d.allocate("solver.nsolving",{1},Element::I32);
    d.allocate("solver.nsolving_initial",{1},Element::I32);
    d.upload("solver.nsolving_initial",&worlds,sizeof(worlds));

    SolverSpec spec; spec.iteration_limit=d.integer("m.opt.iterations");
    require(spec.iteration_limit>=0,"negative solver iteration limit");
    auto& init=spec.initialize;
    // Original create_solver_context zeros these four arrays per solve. Reset
    // their padding too; stale padding may enter the 80x80 blocked solve.
    for(const char* name:{"solver.grad","solver.Mgrad","solver.h","solver.hfactor"})zero(init,name);
    copy(init,"d.qacc",d.integer("m.opt.disableflags")&mjDSBL_WARMSTART?"d.qacc_smooth":"d.qacc_warmstart");
    kernel(init,solver_module,"solve_init_efc_",{worlds},
        {"d.solver_niter","solver.search_dot","solver.cost","solver.done"});
    zero(init,"solver.Jaref");
    kernel(init,jaref_module,"solve_init_jaref__locals__kernel_110c885f_cuda_kernel_forward",{worlds,rows,4},
        {"d.nefc","d.qacc","d.efc.J_rownnz","d.efc.J_rowadr","d.efc.J_colind","d.efc.J","d.efc.aref","solver.Jaref"});
    mul_m(init,d,"d.qacc","d.efc.Ma");
    update_constraint(init,d); update_gradient(init,d,sm_count);
    kernel(init,solver_module,"solve_init_search_",{worlds,nv},
        {"solver.Mgrad","solver.search","solver.search_dot"});
    copy(init,"solver.nsolving","solver.nsolving_initial");
    auto& iteration=spec.iteration;
    linesearch(iteration,d); update_constraint(iteration,d); update_gradient(iteration,d,sm_count);
    kernel(iteration,solver_module,"solve_zero_search_dot_",{worlds},{"solver.done","solver.search_dot"});
    kernel(iteration,solver_module,"solve_search_update_",{worlds,nv},
        {d.integer("m.opt.solver"),"solver.Mgrad","solver.search","solver.beta","solver.done","solver.search","solver.search_dot"});
    kernel(iteration,solver_module,"solve_done_",{worlds},
        {nv,"m.opt.tolerance",spec.iteration_limit,"m.stat.meaninertia","solver.grad_dot","solver.cost",
         "solver.prev_cost","solver.done","d.solver_niter","solver.nsolving","solver.done"});
    return spec;
}
}
