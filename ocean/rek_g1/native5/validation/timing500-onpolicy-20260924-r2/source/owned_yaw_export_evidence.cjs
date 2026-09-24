'use strict';
// Pure observation evidence shared by historical upgrades and actual-v2 exports.
const LEGACY = 'rek.native5.scaled_polar_xy.v1';
const SCHEMA = 'rek.native5.scaled_polar_xy.owned_yaw_v2';
const COLUMN = 187;
const check = (ok, why) => { if (!ok) throw new Error(why); };
function desiredYaw(category) {
  check(Number.isInteger(category) && category >= 1 && category <= 15, 'active desired_action must be owned category1..15');
  return [6,8,10,12,14].includes(category) ? 1 : [7,9,11,13,15].includes(category) ? -1 : 0;
}
function sameArray(actual, expected, label, float32 = false) {
  check(Array.isArray(actual) && actual.length === expected.length && actual.every((v,i) =>
    Number.isFinite(v) && (float32 ? Math.fround(v) : v) === expected[i]), label);
}
function ownedYawEvidence(obs, mask, seq, source, encoded, worker, identity, schema) {
  check(schema === LEGACY || schema === SCHEMA, 'unsupported owned-yaw evidence schema');
  const terminal = worker?.terminal === true;
  check(source?.event === 'g1_policy_state' && source.observation_sequence === seq &&
    source.round_identity_sha256 === identity && source.round?.active === !terminal &&
    (terminal || (source.stream_active === true && source.input?.active === true)), 'missing same-source active owned state');
  check(encoded?.event === 'policy_observation' && encoded.ready === true, 'missing ready encoder provenance');
  const request = encoded.worker_request, p = encoded.provenance;
  check(request?.seq === seq && request.round_id === identity && request.observation_schema === schema &&
    request.type === 'step' && request.terminal === terminal && worker?.seq === seq && worker.round_id === identity &&
    worker.observation_schema === schema && worker.type === 'step' && worker.terminal === terminal,
    'worker/encoder sequence, schema, or terminal mismatch');
  check(p?.source_qpc_ticks === source.clock?.qpc_ticks &&
    p?.source_qpc_frequency_hz === source.clock?.qpc_frequency_hz &&
    typeof p.stream_active === 'boolean' && p.stream_active === source.stream_active &&
    typeof p.projected_busy === 'boolean' &&
    ['dispatched_request_v4_duration','native_controller_busy'].includes(p.busy_projection), 'busy/source provenance unavailable');
  sameArray(request.observation, obs, 'encoder observation differs from recorded row', true);
  sameArray(worker.observation, obs, 'worker observation differs from recorded row', true);
  sameArray(request.mask, mask, 'encoder mask differs from recorded row');
  sameArray(worker.mask, mask, 'worker mask differs from recorded row');
  check(obs[182] === Number(p.projected_busy) && obs[183] === Number(p.projected_busy), 'busy feature/provenance disagreement');
  // Active ownership must remain known even while busy is false.
  const yaw = terminal ? 0 : desiredYaw(source.input.desired_action);
  const value = p.projected_busy ? yaw : 0;
  if (schema === SCHEMA) check(obs[COLUMN] === value, 'recorded owned-yaw column differs from pre-action intent');
  return { source_sequence:seq, source_qpc_ticks:source.clock.qpc_ticks,
    desired_action:terminal ? null : source.input.desired_action, projected_busy:p.projected_busy,
    busy_projection:p.busy_projection, owned_pending_yaw:value };
}
module.exports = { LEGACY, SCHEMA, COLUMN, desiredYaw, ownedYawEvidence };
