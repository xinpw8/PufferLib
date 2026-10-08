'use strict';
// Physics variants spanning what the surrogate does not know. A Claudiolo
// parameter set is only trusted if it wins across all of them.
const WORLDS = Object.freeze({
  calibrated: {},
  slowLegs: {vForward: 0.40, vBackward: 0.30, vStrafe: 0.25},
  fastLegs: {vForward: 0.70, vBackward: 0.55, vStrafe: 0.45},
  sluggish: {tauLinear: 0.40, tauYaw: 0.2, settleSpeed: 0.03},
  shortReach: {handReach: 0.62, kickReach: 0.72, handMin: 0.40},
  longReach: {handReach: 0.74, kickReach: 0.86},
  laggy: {obsDelay: 0.08, actDelay: 0.08},
  stingy: {handAccept: 0.30, kickAccept: 0.18},
  slippery: {pushSelf: 3.5, pushOther: 0.6, kickSelfFall: 0.3},
  sticky: {pushSelf: 0.9, pushOther: 0.15, kickSelfFall: 0.1},
});
module.exports = {WORLDS};
