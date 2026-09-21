'use strict';
// Independent offline evaluation of already fitted native weights. No fitting.
const fs=require('node:fs'),crypto=require('node:crypto');
const check=(ok,why)=>{if(!ok)throw Error(why);};
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const sigmoid=z=>z>=0?1/(1+Math.exp(-z)):Math.exp(z)/(1+Math.exp(z));
function evaluate(weights,bytes,split,calibrated) {
  const p=bytes.readUInt32LE(12),count=bytes.readUInt32LE(20),scored=[];
  for(let i=0;i<count;i++) {
    const at=24+i*(p+6)*4;
    if(bytes.readFloatLE(at)!==split)continue;
    let z=weights.weights[0]+(calibrated?weights.calibration_intercept:0);
    weights.columns.forEach((j,k)=>{z+=weights.weights[k+1]*(bytes.readFloatLE(at+16+j*4)-weights.mean[k])/weights.scale[k];});
    scored.push({p:sigmoid(z),y:bytes.readFloatLE(at+16+(p+weights.target)*4)});
  }
  const n=scored.length,positive=scored.reduce((s,r)=>s+r.y,0);
  const brier=scored.reduce((s,r)=>s+(r.p-r.y)**2,0)/n;
  const logLoss=scored.reduce((s,r)=>{const p=Math.max(1e-12,Math.min(1-1e-12,r.p));return s-r.y*Math.log(p)-(1-r.y)*Math.log1p(-p);},0)/n;
  const mean=scored.reduce((s,r)=>s+r.p,0)/n;
  scored.sort((a,b)=>b.p-a.p);
  let tp=0,fp=0,ap=0,area=0,lastTP=0,lastFP=0;
  for(let i=0;i<n;) {
    let end=i,yes=0;
    while(end<n&&scored[end].p===scored[i].p){yes+=scored[end].y;end++;}
    tp+=yes;fp+=end-i-yes;ap+=positive?yes/positive*tp/(tp+fp):0;
    area+=(fp-lastFP)*(tp+lastTP)/2;lastTP=tp;lastFP=fp;i=end;
  }
  let begin=0,ece=0;const bins=[];
  for(let b=0;b<10&&begin<n;b++) {
    let end=Math.max(begin,Math.floor((b+1)*n/10));
    while(end<n&&end>0&&scored[end].p===scored[end-1].p)end++;
    if(end>begin){const group=scored.slice(begin,end),p=group.reduce((s,r)=>s+r.p,0),y=group.reduce((s,r)=>s+r.y,0);
      bins.push({n:end-begin,predicted:p/(end-begin),observed:y/(end-begin),positive:y});ece+=Math.abs(p-y)/n;}
    begin=end;
  }
  return {rows:n,positive,prevalence:positive/n,mean_predicted_probability:mean,log_loss:logLoss,brier,
    average_precision:ap,roc_auc:positive&&positive<n?area/(positive*(n-positive)):0,
    tie_aware_equal_count_ece10:ece,calibration_bins:bins};
}
function verify(dataFile,weightsFile,reportFile,newOutput) {
  const data=fs.readFileSync(dataFile),rawWeights=fs.readFileSync(weightsFile),rawReport=fs.readFileSync(reportFile);
  const weights=JSON.parse(rawWeights),report=JSON.parse(rawReport);
  check(data.toString('ascii',0,7)==='REKFAL1'&&weights.runtime_enabled===false&&report.runtime_enabled===false,'invalid_inputs');
  let maximumDifference=0;
  for(const m of report.models) {
    const w=weights.models.find(w=>w.name===m.name&&w.target===m.target);check(w,'missing_weights');
    for(const [name,split,calibrated] of [['test_uncalibrated',2,false],['test_calibrated',2,true],['calibration_calibrated',1,true]]) {
      const independent=evaluate(w,data,split,calibrated);
      for(const key of ['rows','positive','prevalence','mean_predicted_probability','log_loss','brier','average_precision','roc_auc']) {
        const difference=Math.abs(independent[key]-m[name][key]);maximumDifference=Math.max(maximumDifference,difference);
        check(difference<1e-10,'native_metric_mismatch_'+key);
      }
      m[name]=independent;
    }
  }
  report.independent_verification={verified:true,refit:false,maximum_absolute_metric_difference:maximumDifference,
    dataset_sha256:sha(data),weights_sha256:sha(rawWeights),native_report_sha256:sha(rawReport),
    verifier_sha256:sha(fs.readFileSync(__filename)),calibration_bin_revision:'keep identical predicted probabilities in one bin'};
  fs.writeFileSync(newOutput,JSON.stringify(report,null,2)+'\n',{flag:'wx'});
  return report.independent_verification;
}
module.exports={evaluate,verify};
if(require.main===module){try{console.log(JSON.stringify(verify(...process.argv.slice(2))));}catch(e){console.error(e.message);process.exitCode=1;}}
