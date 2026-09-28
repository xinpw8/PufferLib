"""One authorized candidate trajectory; exact reviewed harness, own affinity only."""
from pathlib import Path
import argparse, json, os
import benchmark as b

FAST_CPUS = {5,6,7,8,9,15,16,17,18,19}
BINARY = Path('/home/spark-advantage/rek-training/rek-playback-native-20260928-r1/build-r1/rek-native-clone')
PIN = 'ef519db4c8b3b3a6696ebc8dfd7b686bffc789a7b91f1ef8d60dfab3494e0975'


class OwnFastCpu(b.Owned):
    def __init__(self, argv, env, directory, guard):
        super().__init__(['/usr/bin/taskset','--cpu-list','5-9,15-19',*argv],env,directory,guard)

    def call(self, *args, **kwargs):
        value = super().call(*args, **kwargs)
        affinity = set(os.sched_getaffinity(self.process.pid))
        assert affinity == FAST_CPUS
        s = Path(f'/proc/{self.process.pid}/stat').read_text()
        cpu = int(s[s.rfind(')')+2:].split()[36])
        self.identity['affinity_cpu_ids'] = sorted(affinity)
        self.identity['last_observed_cpu'] = cpu
        self.emit('owned_cpu', {'allowed':sorted(affinity),'last_cpu':cpu})
        return value


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    if not a.execute:
        print(json.dumps({'execute':False,'ticks':150,'extra_cache_ticks':3,'affinity':sorted(FAST_CPUS)}));return
    assert b.sha(BINARY)==PIN
    assert b.sha(Path(b.__file__))=='79db557c319a4ebc4c4a5d2d6aacfd82eab9052be0b89b74cb522a042d49d75e'
    _,_,backend=b.prepared(b.LIVE_ROOT/'run-r4')
    assert not a.out.exists();a.out.mkdir(parents=True)
    env={k:v for k,v in os.environ.items() if not k.startswith('REK_')};env.update(backend['env'])
    guard=b.Guard(max_seconds=60);guard.start();summary={'success':False}
    b.Owned=OwnFastCpu
    try:
        commands=b.schedule();(a.out/'commands.json').write_text(json.dumps(commands)+'\n')
        _,result=b.run_trajectory('candidate-fast-cpu',BINARY,backend['workerConfig'],env,a.out,guard,commands)
        summary.update(success=True,result=result)
    except Exception as error:
        summary['error']=str(error);raise
    finally:
        guard.close();summary['guard_failure']=guard.failure
        (a.out/'guard.json').write_text(json.dumps(guard.observations)+'\n')
        (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        files=[{'path':str(f.relative_to(a.out)),'bytes':f.stat().st_size,'sha256':b.sha(f)} for f in sorted(a.out.rglob('*')) if f.is_file()]
        (a.out/'MANIFEST.json').write_text(json.dumps(files,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
