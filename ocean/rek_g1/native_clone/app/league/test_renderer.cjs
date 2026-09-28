'use strict';
class TestRenderer {
  constructor(options){
    if(!options.renderOnly)throw Error('Renderer role required');
    this.ready=Promise.resolve({rendererOnly:true});this.closed=false;
  }
  async request(op,args){
    if(op!=='frame')throw Error('Renderer received physics operation');
    return {png:Buffer.from('test frame').toString('base64'),generation:args.generation,snapshotTick:args.snapshotTick};
  }
  async close(){this.closed=true;}
}
module.exports={TestRenderer};
