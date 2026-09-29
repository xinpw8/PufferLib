from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import Future
import json,subprocess,tempfile,unittest
import resource_watch as r

def stat(pid,ppid,start,cpu=0,rss=10,state='S'):
    fields=['0']*40
    for index,value in {0:state,1:ppid,11:cpu,12:0,17:3,19:start,20:123456,21:rss,36:7}.items():fields[index]=str(value)
    return f'{pid} (comm name ) with parens) '+ ' '.join(fields)+'\n'

class ProcTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='rek-resource-proc-');self.root=Path(self.temp.name)
        self.write(100,1,1000,0);self.write(101,100,1100,10);self.write(102,100,1200,1)
        children=self.root/'100/task/100';children.mkdir(parents=True);(children/'children').write_text('101 102')
        self.links={101:'/native',102:'/other'}
        self.s=r.ProcessSampler(100,1000,'/native',proc=self.root,clock_ticks=100,page_size=4096,
            readlink=lambda p:self.links[int(p.parent.name)])
    def tearDown(self):self.temp.cleanup()
    def write(self,pid,ppid,start,cpu,io=50):
        folder=self.root/str(pid);folder.mkdir(exist_ok=True)
        (folder/'stat').write_text(stat(pid,ppid,start,cpu));(folder/'io').write_text(f'read_bytes: {io}\nwrite_bytes: 20\nrchar: 200\n')
    def test_exact_identity_native_scope_and_initial_null_deltas(self):
        sample=self.s.sample(1_000_000_000);self.assertEqual([p['pid'] for p in sample['processes']],[100,101])
        for p in sample['processes']:
            self.assertIsNone(p['cpu_percent_one_core']);self.assertIsNone(p['io_delta']);self.assertEqual(p['rss_bytes'],40960)
        self.assertEqual(sample['processes'][1]['cpu_seconds'],.1)
    def test_cpu_and_io_interval_deltas(self):
        self.s.sample(1_000_000_000);self.write(100,1,1000,75,io=80);self.write(101,100,1100,160,io=130)
        values=self.s.sample(2_000_000_000)['processes']
        self.assertEqual([v['cpu_percent_one_core'] for v in values],[75,150]);self.assertEqual(values[1]['io_delta']['read_bytes'],80)
    def test_parent_reuse_and_zombie_stop_without_children_effects(self):
        self.write(100,1,9999,0)
        with self.assertRaises(r.IdentityLost):self.s.sample(1)
        (self.root/'100/stat').write_text(stat(100,1,1000,state='Z'))
        with self.assertRaises(r.IdentityLost):self.s.parent()
        self.assertTrue((self.root/'101/stat').exists())
    def test_child_reuse_and_missing_io(self):
        self.s.sample(1_000_000_000);self.write(101,100,2222,4);(self.root/'101/io').unlink()
        child=self.s.sample(2_000_000_000)['processes'][1]
        self.assertIsNone(child['cpu_percent_one_core']);self.assertIsNone(child['io_cumulative']);self.assertTrue(child['errors'])
    def test_parent_change_during_sample(self):
        original=self.s.readlink
        def changed(path):self.write(100,1,9999,0);return original(path)
        self.s.readlink=changed
        with self.assertRaises(r.IdentityLost):self.s.sample(1)
    def test_nonmonotonic_interval(self):
        self.s.sample(1000);self.write(100,1,1000,10)
        result=self.s.sample(1000)['processes'][0]
        self.assertIsNone(result['cpu_percent_one_core']);self.assertTrue(result['errors'])
    def test_run_stop_flush_and_exclusive_create(self):
        output=self.root/'resource.jsonl';stop=self.root/'STOP'
        class FakeGPU:
            closed=False
            def poll(self,now):stop.touch();return {'sample':None,'in_flight':False,'sample_age_seconds':None}
            def close(self):self.closed=True
        gpu=FakeGPU();args=SimpleNamespace(pid=100,start_ticks=1000,native_exe=Path('/native'),output=output,stop_file=stop,interval=1,gpu_seconds=5)
        r.run(args,sampler=self.s,gpu=gpu,clock=lambda:0,sleep=lambda t:None)
        rows=[json.loads(x) for x in output.read_text().splitlines()]
        self.assertEqual([x['event'] for x in rows],['header','sample','stopped']);self.assertEqual(rows[-1]['reason'],'STOP');self.assertTrue(gpu.closed)
        before=output.read_bytes();stop.unlink()
        with self.assertRaises(FileExistsError):r.run(args,sampler=self.s,gpu=gpu)
        self.assertEqual(output.read_bytes(),before)

class GpuTests(unittest.TestCase):
    def test_nonfinite_unavailable_and_columns(self):
        for value in ['nan','inf','-inf','N/A','[Not Supported]','-','garbage']:self.assertIsNone(r.number(value))
        row=r.parse_gpu('0, 50, 12, 300, N/A, 75.5, 42, 100, 128000')[0]
        self.assertEqual(row['power.draw'],75.5);self.assertIsNone(row['clocks.current.memory'])
        with self.assertRaises(ValueError):r.parse_gpu('0, 1')
    def test_pmon_command_and_unavailable_pid_not_retained(self):
        values=r.parse_apps('# gpu pid type sm mem enc dec jpg ofa command\n# Idx # C/G % % % % % % name\n0 101 C 20 3 - - - - sensitive-command-argument\n0 - - - - - - - - -\n')
        self.assertEqual(values,[{'gpu_index':0.0,'pid':101,'type':'C','sm':20.0,'mem':3.0,'enc':None,'dec':None,'jpg':None,'ofa':None}])
        self.assertNotIn('sensitive',json.dumps(values))
        with self.assertRaises(ValueError):r.parse_apps('Not Supported')
    def test_bounded_query_errors(self):
        calls=[]
        def fail(argv,**kwargs):calls.append((argv,kwargs));raise subprocess.TimeoutExpired(argv,2)
        sample=r.gpu_sample(fail)
        self.assertIsNone(sample['gpu']);self.assertIsNone(sample['apps']);self.assertEqual(len(sample['errors']),2)
        self.assertEqual(len(calls),2);self.assertTrue(all(k['timeout']==2 and not k['check'] for _,k in calls))
    def test_one_pending_job_and_age(self):
        class FakeExecutor:
            def __init__(self):self.jobs=[];self.closed=False
            def submit(self,fn):future=Future();self.jobs.append(future);return future
            def shutdown(self,**kwargs):self.closed=True
        executor=FakeExecutor();p=r.GpuPoller(executor=executor)
        p.poll(0);p.poll(9_000_000_000);self.assertEqual(len(executor.jobs),1)
        executor.jobs[0].set_result({'finished_monotonic_ns':4_000_000_000,'errors':[]})
        sample=p.poll(9_000_000_000);self.assertEqual(sample['sample_age_seconds'],5);self.assertEqual(len(executor.jobs),2)
        p.close();self.assertTrue(executor.closed)

if __name__=='__main__':unittest.main()
