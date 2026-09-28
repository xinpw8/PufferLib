"""Pure cached GETs and exact process identities for both human viewers."""
IDENTITIES={1092025:'125031755',1092037:'125031762',2949931:'131757839',2949941:'131757847',2949942:'131757847'}
URLS=['http://127.0.0.1:18771/api/snapshot','http://127.0.0.1:18772/api/snapshot']


def make_guard(bench,max_seconds=90):
    observations=[]
    def fetch():
        values=[]
        for url in URLS:
            value=bench.http(url,timeout=.75)
            if value.get('paused') is not True or value.get('ok') is not True:
                raise RuntimeError('Human viewer resumed or became unhealthy: '+url)
            values.append(value)
        observations.append([{'url':url,'paused':v['paused'],'ok':v['ok'],'tick':v.get('tick')} for url,v in zip(URLS,values)])
        return values[0]
    guard=bench.Guard(fetch=fetch,identities=lambda:all(bench.proc_start(pid)==start for pid,start in IDENTITIES.items()),max_seconds=max_seconds)
    guard.dual_observations=observations
    return guard
