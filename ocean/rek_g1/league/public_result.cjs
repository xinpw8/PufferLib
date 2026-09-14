'use strict';
function publicStanding(row){
  const {checkpoint,...value}=row;
  return {...value,checkpoint:checkpoint?{sha256:checkpoint.sha256,format:checkpoint.format,
    trainingSteps:checkpoint.trainingSteps,model:checkpoint.model}:null};
}
module.exports={publicStanding};
