"""Replay recorded frame requests through one render-only worker; report per-frame latency and PNG hashes."""
import base64,hashlib,json,os,statistics,subprocess,sys,time
binary,config,server,frames_path,follow,out=sys.argv[1:7]
env={**os.environ,**json.load(open(server))['backends'][0]['env']}
frames=json.load(open(frames_path))
p=subprocess.Popen(['taskset','-c','5-9',binary,'--render-only','--config',config],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,env=env,text=True,bufsize=1)
ready=json.loads(p.stdout.readline());assert ready.get('rendererOnly') is True,ready
times=[];hashes=[]
try:
    for i,a in enumerate(frames):
        req={'op':'frame','id':i+1,'qpos':a['qpos'],'snapshotTick':a['snapshotTick'],'generation':a['generation']}
        if follow!='none':req['followSide']=int(follow)
        t=time.perf_counter();p.stdin.write(json.dumps(req)+'\n');r=json.loads(p.stdout.readline());dt=(time.perf_counter()-t)*1000
        assert r.get('ok') and r['id']==i+1,r
        png=base64.b64decode(r['png']) if not r['png'].startswith('\x89PNG') else r['png'].encode('latin1')
        hashes.append(hashlib.sha256(png).hexdigest())
        if i>=20:times.append(dt)
        if i in (0,400,800,1200,1499):open(f'{out}-{i:04d}.png','wb').write(png)
finally:
    p.stdin.close();p.wait(timeout=20)
q=statistics.quantiles(times,n=100)
print(json.dumps({'binary':binary,'follow':follow,'frames':len(frames),'mean_ms':round(statistics.mean(times),2),'p50_ms':round(q[49],2),'p95_ms':round(q[94],2),'p99_ms':round(q[98],2),'max_ms':round(max(times),2)}))
json.dump(hashes,open(out+'-hashes.json','w'))
