"""Passive Linux resource log for one exact viewer identity and its native children.

No input, process signals, service changes, command-line reads or environment reads.
GPU queries run in one bounded background job; process sampling does not wait for them.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse,csv,datetime,json,math,os,subprocess,time

GPU_FIELDS=['index','utilization.gpu','utilization.memory','clocks.current.sm',
            'clocks.current.memory','power.draw','temperature.gpu','memory.used','memory.total']
IO_FIELDS={'rchar','wchar','syscr','syscw','read_bytes','write_bytes','cancelled_write_bytes'}

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()

class IdentityLost(RuntimeError):pass

def read_stat(path):
    text=path.read_text();prefix,separator,tail=text.rpartition(') ')
    if not separator:raise ValueError('Malformed process stat')
    pid=int(prefix.split(' ',1)[0]);fields=tail.split()
    if len(fields)<37:raise ValueError('Incomplete process stat')
    return {'pid':pid,'state':fields[0],'ppid':int(fields[1]),
        'cpu_ticks':int(fields[11])+int(fields[12]),'threads':int(fields[17]),
        'start_ticks':int(fields[19]),'virtual_bytes':int(fields[20]),
        'rss_pages':int(fields[21]),'processor':int(fields[36])}

class ProcessSampler:
    def __init__(self,pid,start_ticks,native_exe,proc=Path('/proc'),clock_ticks=None,page_size=None,readlink=os.readlink):
        self.pid=pid;self.start_ticks=start_ticks;self.native_exe=str(native_exe)
        self.proc=proc;self.clock_ticks=clock_ticks or os.sysconf('SC_CLK_TCK')
        self.page_size=page_size or os.sysconf('SC_PAGE_SIZE');self.readlink=readlink;self.previous={}
    def parent(self):
        try:s=read_stat(self.proc/str(self.pid)/'stat')
        except (OSError,ValueError) as error:raise IdentityLost('Parent unavailable: '+type(error).__name__) from error
        if s['pid']!=self.pid or s['start_ticks']!=self.start_ticks or s['state'] in ['Z','X']:
            raise IdentityLost('Parent identity changed or exited')
        return s
    def sample(self,now_ns):
        parent=self.parent();errors=[];identities=[(parent,'viewer')]
        try:
            children=(self.proc/str(self.pid)/'task'/str(self.pid)/'children').read_text().split()
            for value in children:
                try:
                    child=int(value);folder=self.proc/str(child)
                    before=read_stat(folder/'stat')
                    if before['ppid']!=self.pid or before['state'] in ['Z','X']:continue
                    if self.readlink(folder/'exe')!=self.native_exe:continue
                    after=read_stat(folder/'stat')
                    if (after['pid'],after['start_ticks'],after['ppid'])!=(child,before['start_ticks'],self.pid):
                        raise ValueError('Child identity changed during discovery')
                    identities.append((after,'native_child'))
                except (OSError,ValueError) as error:errors.append('Child unavailable: '+type(error).__name__)
        except (OSError,ValueError) as error:errors.append('Child list unavailable: '+type(error).__name__)
        result=[];next_previous={}
        for record,role in identities:
            pid=record['pid'];folder=self.proc/str(pid);io=None;entry_errors=[]
            try:
                io={}
                for line in (folder/'io').read_text().splitlines():
                    key,separator,value=line.partition(':')
                    if separator and key in IO_FIELDS:io[key]=int(value.strip())
                if not io:raise ValueError('No I/O counters')
            except (OSError,ValueError) as error:io=None;entry_errors.append('I/O unavailable: '+type(error).__name__)
            try:
                after=read_stat(folder/'stat')
                if (after['pid'],after['start_ticks'])!=(pid,record['start_ticks']):raise ValueError('PID reused')
            except (OSError,ValueError) as error:
                if role=='viewer':raise IdentityLost('Parent changed during sample') from error
                errors.append('Child changed during sample');continue
            key=(pid,record['start_ticks']);previous=self.previous.get(key)
            delta_seconds=None;cpu_percent=None;io_delta=None
            if previous:
                delta_seconds=(now_ns-previous['monotonic_ns'])/1e9
                delta_ticks=record['cpu_ticks']-previous['cpu_ticks']
                if delta_seconds>0 and delta_ticks>=0:
                    cpu_percent=100*delta_ticks/self.clock_ticks/delta_seconds
                    if io is not None and previous['io'] is not None:
                        io_delta={k:io[k]-previous['io'][k] for k in io.keys()&previous['io'].keys()}
                else:entry_errors.append('Invalid counter interval; delta unavailable')
            result.append({**record,'role':role,'cpu_seconds':record['cpu_ticks']/self.clock_ticks,
                'rss_bytes':record['rss_pages']*self.page_size,'interval_seconds':delta_seconds,
                'cpu_percent_one_core':cpu_percent,'io_cumulative':io,'io_delta':io_delta,'errors':entry_errors})
            next_previous[key]={'monotonic_ns':now_ns,'cpu_ticks':record['cpu_ticks'],'io':io}
        self.parent() # Never publish a sample across parent PID reuse.
        self.previous=next_previous
        return {'processes':result,'errors':errors}

def number(value):
    value=value.strip()
    if value in ['-','N/A','[N/A]','[Not Supported]','Not Supported']:return None
    try:
        parsed=float(value)
        return parsed if math.isfinite(parsed) else None
    except ValueError:return None

def parse_gpu(text):
    records=[]
    for values in csv.reader(text.splitlines()):
        if not values:continue
        if len(values)!=len(GPU_FIELDS):raise ValueError('Unexpected GPU columns')
        records.append(dict(zip(GPU_FIELDS,map(number,values))))
    return records

def parse_apps(text):
    columns=None;rows=[]
    for line in text.splitlines():
        words=line.split()
        if not words:continue
        if words[0]=='#':
            if 'gpu' in words and 'pid' in words and 'command' in words:columns=words[1:]
            continue
        if columns is None:continue
        # Retain numeric counters and process type only. Never retain command text.
        values=dict(zip(columns,words));pid=number(values.get('pid','-'))
        if pid is None:continue
        row={'gpu_index':number(values.get('gpu','-')),'pid':int(pid),'type':values.get('type')}
        for name in ['sm','mem','enc','dec','jpg','ofa','fb','ccpm']:
            if name in values:row[name]=number(values[name])
        rows.append(row)
    if columns is None:raise ValueError('Per-process GPU sampling unsupported or unrecognized')
    return rows

def gpu_sample(runner=subprocess.run):
    started=time.monotonic_ns();result={'started_utc':utc(),'started_monotonic_ns':started,'errors':[]}
    queries=[('gpu',['nvidia-smi','--query-gpu='+','.join(GPU_FIELDS),'--format=csv,noheader,nounits'],parse_gpu),
             ('apps',['nvidia-smi','pmon','-c','1','-s','um'],parse_apps)]
    for name,argv,parser in queries:
        try:
            response=runner(argv,capture_output=True,text=True,timeout=2,check=False)
            if response.returncode:raise RuntimeError('nvidia-smi exit '+str(response.returncode))
            result[name]=parser(response.stdout)
        except (OSError,ValueError,RuntimeError,subprocess.TimeoutExpired) as error:
            result[name]=None;result['errors'].append(name+': '+type(error).__name__+': '+str(error)[:300])
    result['finished_monotonic_ns']=time.monotonic_ns();result['finished_utc']=utc()
    return result

class GpuPoller:
    def __init__(self,period=5,executor=None,sample=gpu_sample):
        self.period=period;self.executor=executor or ThreadPoolExecutor(max_workers=1,thread_name_prefix='resource-gpu')
        self.sample=sample;self.future=None;self.next_due=0;self.latest=None
    def poll(self,now_ns):
        if self.future is not None and self.future.done():
            try:self.latest=self.future.result()
            except Exception as error:self.latest={'finished_monotonic_ns':now_ns,'errors':['GPU sampler: '+type(error).__name__]}
            self.future=None
        if self.future is None and now_ns>=self.next_due:
            self.future=self.executor.submit(self.sample);self.next_due=now_ns+int(self.period*1e9)
        return {'sample':self.latest,'in_flight':self.future is not None,
            'sample_age_seconds':None if self.latest is None else max(0,(now_ns-self.latest['finished_monotonic_ns'])/1e9)}
    def close(self):self.executor.shutdown(wait=True,cancel_futures=True)

def run(args,sampler=None,gpu=None,clock=time.monotonic_ns,sleep=time.sleep):
    sampler=sampler or ProcessSampler(args.pid,args.start_ticks,args.native_exe)
    sampler.parent() # Fail before creating output if the requested identity is stale.
    if args.stop_file.exists():raise RuntimeError('Own STOP marker already exists')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    poller=gpu or GpuPoller(args.gpu_seconds)
    try:
        with args.output.open('x',encoding='utf-8',buffering=1) as output:
            def emit(record):output.write(json.dumps(record,separators=(',',':'),allow_nan=False)+'\n');output.flush()
            emit({'event':'header','schema':'rek.viewer_resource_watch.v1','utc':utc(),'monotonic_ns':clock(),
                'viewer_pid':args.pid,'viewer_start_ticks':args.start_ticks,'native_exe':str(args.native_exe),
                'clock_ticks':sampler.clock_ticks,'page_size':sampler.page_size,'sample_interval_seconds':args.interval,
                'gpu_interval_seconds':args.gpu_seconds,'scope':'Process counters cover the exact parent and direct children matching native executable. GPU counters and numeric pmon PID rows are system-wide. No cmdline/environ reads, input or process signals.',
                'units':'CPU percent is one-core based; GPU clocks MHz, power W, temperature degC, memory MiB. pmon is sampled utilization, not exclusive GPU time.',
                'unavailable':'Missing counters remain null. First process observation has no CPU/I/O delta. GPU sample age is explicit.'})
            reason='STOP';samples=0;next_due=clock()
            while not args.stop_file.exists():
                now=clock()
                if now<next_due:sleep(min(.2,(next_due-now)/1e9));continue
                try:processes=sampler.sample(now)
                except IdentityLost as error:reason=str(error);break
                emit({'event':'sample','utc':utc(),'monotonic_ns':now,**processes,'gpu':poller.poll(now)})
                samples+=1;next_due=now+int(args.interval*1e9)
            emit({'event':'stopped','utc':utc(),'monotonic_ns':clock(),'reason':reason,'samples':samples})
    finally:poller.close()

def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pid',type=int,required=True);p.add_argument('--start-ticks',type=int,required=True)
    p.add_argument('--native-exe',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stop-file',type=Path,required=True);p.add_argument('--interval',type=float,default=1)
    p.add_argument('--gpu-seconds',type=float,default=5);args=p.parse_args(argv)
    if args.pid<1 or args.start_ticks<1:raise ValueError('Positive process identity required')
    if not args.native_exe.is_absolute():raise ValueError('Exact absolute native executable required')
    if not .5<=args.interval<=60 or not 5<=args.gpu_seconds<=60:raise ValueError('Sampling interval out of bounds')
    if args.output.resolve()==args.stop_file.resolve():raise ValueError('Output cannot be its STOP marker')
    return args

if __name__=='__main__':run(parse_args())
