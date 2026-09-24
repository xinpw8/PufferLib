'use strict';
const stage='/home/spark-advantage/rek-training/native-hparam-score-sweep-20260924-r1';
const previous='/home/spark-advantage/rek-training/f7-action-id-fix-20260924-r1';
const warm=previous+'/runs/F7-no-unmeasured-kick-prior-16777216/checkpoints/rek_native5/F7-no-unmeasured-kick-prior-16777216/0000000016777216.bin';
const warmSha='7561859790d0ed55a55471d96442fe954a593c52d8617b51c143687ba9d98f96';
const trainer=previous+'/build/puffer-rek-native5';
const trainerSha='1754a66278059fdaa207e02d7231bf2a8d7dcc235880b2d055c4d628a344072a';
const assets='/home/spark-advantage/codexrook-runtime/build-validation/l100-yaw-move-buffer-20260910T0448Z/semantic-duel-assets-contact';
const features='/home/spark-advantage/rek-training/gpu-runtime-20260910/motion-foot-features';
const physics='/home/spark-advantage/rek-training/paired-physics-training-20260913T0423Z/model-export-capsule-diagnostic.json';
const environment={REK_ALLOW_CPU_EVALUATION:'0',REK_PHYSICS_BACKEND:'semantic_cuda',REK_OBSERVATION_SCHEMA:'rek.native5.scaled_polar_xy.v1',REK_POLICY_ACTION_STRIDE:'1',
 REK_FAST_ATTEMPT_COST:'0.15',REK_FAST_BOT_AWARD_P:'0.25',REK_FAST_CALIBRATION_ACTION_IDS:'1',REK_FAST_CALIBRATION_COHERENT:'1',REK_FAST_CALIBRATION_INDEPENDENT_RNG:'0',REK_FAST_KICK_FALL_P:'0',
 REK_FAST_LIVE_CALIBRATION:'1',REK_FAST_INTERRUPT_ON_HIT:'1',REK_FAST_CONTACT_ENTRY:'geom_pair_v1',REK_FAST_CONTACT_SUBSTEPS:'8',REK_FAST_CONTACT_VELOCITY:'body_cvel_v1',REK_FAST_GEOMETRY:'primitive_samples_v1',
 REK_FAST_OBSERVATION:'rendered_pose_v1',REK_FAST_OPPONENT:'recovered_bot1_v1',REK_FAST_OPPONENT_MODE:'scripted',REK_FAST_RANDOM_RESETS:'1',REK_FAST_RESET_GAP_MAX:'2.5',REK_FAST_RESET_GAP_MIN:'.55',REK_FAST_RESET_HEADING_SPREAD_RAD:'3.14159265',
 REK_FAST_REWARD:'normalized_points_falls_v1',REK_FAST_SCORING:'recovered_hit_rules_v2',REK_FAST_SHAPING_WEIGHT:'0',REK_FAST_YAW_COMMAND:'keyboard_reset_v1'};
const temporal=[{name:'short',horizon:256,discountHalfLifeSeconds:60,traceHalfLifeSeconds:3.08},{name:'long',horizon:512,discountHalfLifeSeconds:120,traceHalfLifeSeconds:6.16}].map(t=>({...t,gamma:2**(-.02/t.discountHalfLifeSeconds),lambda:2**(-.02/t.traceHalfLifeSeconds)/2**(-.02/t.discountHalfLifeSeconds)}));
const control={id:'control',lr:5.5e-5,entropy:1.7e-4,clip:.13,vf:1.02,horizon:128,gamma:.9998844821426083,lambda:.9978673240629938};
const arms=[control];
for(const lr of [5.5e-5,.0003,.001,.003,.0075,.015])for(const entropy of [.00017,.002])for(const time of temporal)arms.push({id:'arm-'+String(arms.length).padStart(2,'0'),lr,entropy,clip:.2,vf:.5,...time});
module.exports={stage,previous,warm,warmSha,trainer,trainerSha,assets,features,physics,environment,temporal,control,arms,
 screenSteps:8388608,promotionSteps:33554432,screenSeeds:[10001,10003],confirmationSeeds:[20011,20021,20031],
 assumptions:{nativeOptimizer:'Muon',controlHz:50,envs:512,minibatch:8192,replay:1,hidden:256,layers:2,reward:'(own point delta - opponent point delta - .15 per accepted attack)/100, clipped point term [-1,1]',
 physicalFallsModeled:false,fullMatchesModeled:false,selectionSide:0,checkpointSemantics:'scaled_polar_xy.v1, 223 floats; incompatible with balance8 despite equal width',
 promotions:'Restart every larger-budget run from the identical 756 checkpoint, fresh optimizer; not an exact resume',
 selection:'Screen on held-out score per round with point-margin and round-win non-regression against unchanged checkpoint; if none pass, report tradeoffs, no promotion claim',
 authenticWins:'Requires actual REK complete-match evidence; simulator rounds are not matches',pythonRuntime:false,cpuEnvironmentStepping:false}};
