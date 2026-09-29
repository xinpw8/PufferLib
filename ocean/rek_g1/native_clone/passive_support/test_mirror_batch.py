"""Mirror batching regressions using local files and fake SFTP only."""
import io,json,os,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import passive as p

PNG=b'\x89PNG\r\n\x1a\n'+b'fixture'+b'\x00\x00\x00\x00IEND\xaeB`\x82'

class RemoteFile:
    def __init__(self,path):self.path=path;self.f=path.open('rb')
    def read(self,n):return self.f.read(n)
    def seek(self,n):return self.f.seek(n)
    def stat(self):return self.path.stat()
    def __enter__(self):return self
    def __exit__(self,*args):self.f.close()

class TreeSftp:
    def __init__(self,root,events):self.root=root;self.events=events
    def path(self,remote):return self.root/remote.removeprefix(p.REMOTE+'/')
    def lstat(self,remote):return self.path(remote).lstat()
    def open(self,remote,mode):
        self.events.append(('open',remote.removeprefix(p.REMOTE+'/')))
        return RemoteFile(self.path(remote))
    def listdir_attr(self,remote):
        self.events.append(('list_frames',None))
        return [SimpleNamespace(filename=q.name,st_mode=q.stat().st_mode,st_size=q.stat().st_size,st_mtime=q.stat().st_mtime) for q in self.path(remote).iterdir()]
    def get_channel(self):return self
    def settimeout(self,n):pass
    def __enter__(self):return self
    def __exit__(self,*args):pass

class Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='mirror-batch-test-');self.root=Path(self.temp.name)
        self.remote=self.root/'remote';self.remote.mkdir()
        self.control=self.root/'control';self.control.mkdir();self.nas=self.root/'nas'
        self.events=[];self.clock=[0.0]
    def tearDown(self):self.temp.cleanup()
    def populate(self,count):
        (self.remote/'session.jsonl').write_bytes(b'{"tick":0}\n')
        (self.remote/'resource.jsonl').write_bytes(b'{"hardware_sample":0}\n')
        (self.remote/'recorder-health.json').write_text('{"steps":0}')
        (self.remote/'identity.json').write_text('{"id":"fake-session"}')
        (self.remote/'frames').mkdir()
        for i in range(count):(self.remote/'frames'/f'{i:012d}.png').write_bytes(PNG+bytes())
    def run_mirror(self,resume=False,on_cycle=None,copy=None,atomic=None):
        owner=self
        class Client:
            def open_sftp(client):
                owner.events.append(('cycle',None))
                if on_cycle:on_cycle(sum(k=='cycle' for k,_ in owner.events))
                return TreeSftp(owner.remote,owner.events)
            def close(client):pass
        def idle(_):
            (self.control/'STOP').write_text('test stop')
        args=SimpleNamespace(local=self.control,nas=self.nas,resume=resume,until_stop=True)
        with patch.object(p,'connect',return_value=Client()),patch.object(p.time,'monotonic',side_effect=lambda:self.clock[0]),patch.object(p.time,'sleep',side_effect=idle),patch.object(p,'immutable_file',side_effect=copy or p.immutable_file),patch.object(p,'atomic',side_effect=atomic or p.atomic):
            p.mirror(args)
    def test_checkpoints_bound_record_count_time_and_force(self):
        calls=[]
        with patch.object(p.time,'monotonic',side_effect=lambda:self.clock[0]),patch.object(p,'atomic',side_effect=lambda q,v:calls.append((q,json.loads(json.dumps(v))))):
            c=p.InventoryCheckpoint(self.root/'local',self.root/'nas','id',{})
            for i in range(31):c.record(str(i),{'bytes':i});self.assertFalse(c.flush())
            self.assertEqual(calls,[])
            c.record('31',{'bytes':31});self.assertTrue(c.flush());self.assertEqual(len(calls),2)
            c.record('32',{'bytes':32});self.clock[0]=2.0;self.assertTrue(c.flush())
            c.record('33',{'bytes':33});self.assertTrue(c.flush(force=True));self.assertFalse(c.flush(force=True))
            self.assertEqual([len(v['files']) for _,v in calls],[32,32,33,33,34,34])
    def test_inventory_serialization_growth_is_batched_not_per_file(self):
        reports=[]
        for n in (128,512):
            writes=[];files={};old_bytes=0
            def record(q,v):writes.append((q,len(json.dumps(v,indent=2))+1))
            with patch.object(p.time,'monotonic',return_value=0),patch.object(p,'utc',return_value='fixed'),patch.object(p,'atomic',side_effect=record):
                c=p.InventoryCheckpoint(self.root/'local',self.root/'nas','id',files)
                for i in range(n):
                    c.record(f'frames/{i:012d}.png',{'bytes':500000,'sha256':'a'*64,'source_signature':[500000,1]})
                    old_bytes+=len(json.dumps({'identity':'id','files':files},indent=2))+1
                    c.flush()
                c.flush(force=True)
            per_destination=[size for q,size in writes if q.name=='local']
            self.assertEqual(len(per_destination),n//32)
            self.assertLess(sum(per_destination),old_bytes*.045)
            reports.append({'frames':n,'old_local_inventory_writes':n,'new_local_inventory_writes':len(per_destination),'old_local_serialized_bytes':old_bytes,'new_local_serialized_bytes':sum(per_destination),'new_both_destinations_serialized_bytes':sum(size for _,size in writes)})
        print('BATCH-GROWTH '+json.dumps(reports))
    def test_failed_nas_checkpoint_keeps_batch_retryable(self):
        local=self.root/'inventory-local.json';nas=self.root/'inventory-nas.json';real=p.atomic;fail=[True]
        def write(q,v):
            if q==nas and fail[0]:fail[0]=False;raise OSError('synthetic NAS error')
            real(q,v)
        c=p.InventoryCheckpoint(local,nas,'id',{})
        with patch.object(p,'atomic',side_effect=write):
            c.record('frame',{'sha256':'a'})
            with self.assertRaisesRegex(OSError,'NAS'):c.flush(force=True)
            self.assertEqual(c.dirty,1);self.assertTrue(local.exists());self.assertFalse(nas.exists())
            c.flush(force=True)
        self.assertEqual(c.dirty,0);self.assertEqual(local.read_bytes(),nas.read_bytes())
    def test_frame_batch_refreshes_trace_health_status_and_copies_every_frame(self):
        self.populate(70);real=p.atomic;statuses=[]
        def cycle(n):
            with (self.remote/'session.jsonl').open('ab') as f:f.write(f'{{"cycle":{n}}}\n'.encode())
            with (self.remote/'resource.jsonl').open('ab') as f:f.write(f'{{"hardware_cycle":{n}}}\n'.encode())
            (self.remote/'recorder-health.json').write_text(json.dumps({'steps':n}))
            os.utime(self.remote/'recorder-health.json',(n,n))
        def write(q,v):
            if q==self.nas/'mirror-status.json' and v.get('phase')=='watching':
                statuses.append((v.get('source_recorder_health',{}).get('steps'),len(list((self.nas/'frames').glob('*.png'))) if (self.nas/'frames').exists() else 0))
            real(q,v)
        self.run_mirror(on_cycle=cycle,atomic=write)
        self.assertEqual([n for n,_ in statuses[::2]],[1,2,3])
        self.assertEqual([count for _,count in statuses[::2]],[0,32,64])
        self.assertEqual(len(list((self.nas/'frames').glob('*.png'))),70)
        self.assertEqual((self.nas/'session.jsonl').read_bytes(),(self.remote/'session.jsonl').read_bytes())
        self.assertEqual((self.nas/'resource.jsonl').read_bytes(),(self.remote/'resource.jsonl').read_bytes())
        for block in self.split_cycles():
            self.assertLess(block.index(('open','recorder-health.json')),block.index(('list_frames',None)))
            self.assertLess(block.index(('open','session.jsonl')),block.index(('list_frames',None)))
            self.assertLess(block.index(('open','resource.jsonl')),block.index(('list_frames',None)))
        ledger=json.loads((self.nas/'inventory.json').read_text());self.assertEqual(len(ledger['files']),74)
        for i in range(70):self.assertEqual((self.nas/'frames'/f'{i:012d}.png').read_bytes(),PNG)
    def split_cycles(self):
        blocks=[]
        for event in self.events:
            if event[0]=='cycle':blocks.append([])
            else:blocks[-1].append(event)
        return blocks
    def test_slow_frames_yield_by_time_before_batch_limit(self):
        self.populate(8);real=p.immutable_file
        def copy(*args):
            value=real(*args)
            if '/frames/' in args[1]:self.clock[0]+=.4
            return value
        self.run_mirror(copy=copy)
        counts=[sum(kind=='open' and name.startswith('frames/') for kind,name in b) for b in self.split_cycles()]
        self.assertEqual(counts,[3,3,2])
    def test_stop_during_copy_checkpoints_completed_files_only_then_resume(self):
        self.populate(4);real=p.immutable_file
        def copy(*args):
            if args[1].endswith('000000000002.png'):
                (self.control/'STOP').write_text('stop during frame');raise InterruptedError('stop')
            return real(*args)
        self.run_mirror(copy=copy)
        ledger=json.loads((self.nas/'inventory.json').read_text())
        self.assertIn('frames/000000000001.png',ledger['files']);self.assertNotIn('frames/000000000002.png',ledger['files'])
        self.assertEqual(json.loads((self.nas/'mirror-status.json').read_text())['reason'],'STOP')
        (self.control/'STOP').unlink();self.run_mirror(resume=True)
        self.assertEqual(len(list((self.nas/'frames').glob('*.png'))),4)
    def test_copy_error_checkpoints_successes_without_certifying_failed_file(self):
        self.populate(4);real=p.immutable_file
        def copy(*args):
            if args[1].endswith('000000000002.png'):raise IOError('synthetic bad source')
            return real(*args)
        self.run_mirror(copy=copy)
        ledger=json.loads((self.nas/'inventory.json').read_text())
        self.assertIn('frames/000000000001.png',ledger['files']);self.assertNotIn('frames/000000000002.png',ledger['files'])
        final=json.loads((self.nas/'mirror-status.json').read_text())
        self.assertIn('synthetic bad source',str(final['historical_errors']))
    def test_resume_revalidates_checkpointed_immutable_file_and_preserves_conflict(self):
        self.populate(2);self.run_mirror()
        damaged=self.nas/'frames/000000000000.png';damaged.write_bytes(b'X'*len(PNG))
        statuses=[];real=p.atomic
        def write(q,v):
            if q==self.nas/'mirror-status.json' and v.get('phase')=='watching':statuses.append(dict(v))
            real(q,v)
        (self.control/'STOP').unlink();self.events=[];self.run_mirror(resume=True,atomic=write)
        self.assertEqual(damaged.read_bytes(),b'X'*len(PNG))
        self.assertIn('Immutable destination differs',str(json.loads((self.nas/'mirror-status.json').read_text())['historical_errors']))
        self.assertEqual(statuses[0]['files_verified'],6)
        self.assertEqual(statuses[0]['files_verified_this_process'],4)
    def test_resume_recovers_uncheckpointed_files_by_hash_without_overwrite(self):
        self.populate(2)
        local=self.control/'spool';identity=p.initialize(local,self.nas,False)
        p.atomic(local/'inventory.json',{'identity':identity['id'],'files':{}})
        p.atomic(self.nas/'inventory.json',{'identity':identity['id'],'files':{}})
        q='frames/000000000000.png';p.immutable_file(TreeSftp(self.remote,self.events),p.REMOTE+'/'+q,local/q,self.nas/q,lambda:False)
        before=(self.nas/q).stat().st_mtime_ns
        self.run_mirror(resume=True)
        self.assertEqual((self.nas/q).stat().st_mtime_ns,before)
        self.assertIn(q,json.loads((self.nas/'inventory.json').read_text())['files'])
    def test_inventory_wrong_identity_rejects_before_connection(self):
        identity=p.initialize(self.control/'spool',self.nas,False)
        p.atomic(self.control/'spool/inventory.json',{'identity':identity['id']+'wrong','files':{}})
        with patch.object(p,'connect') as connection:
            with self.assertRaisesRegex(RuntimeError,'inventory identity'):
                p.mirror(SimpleNamespace(local=self.control,nas=self.nas,resume=True,until_stop=True))
            connection.assert_not_called()
    def test_final_checkpoint_failure_is_reported_and_completed_bytes_preserved(self):
        self.populate(1);real=p.atomic
        def write(q,v):
            if q==self.nas/'inventory.json':raise OSError('NAS inventory denied')
            real(q,v)
        self.run_mirror(atomic=write)
        self.assertEqual((self.nas/'frames/000000000000.png').read_bytes(),PNG)
        final=json.loads((self.control/'mirror-status.json').read_text())
        self.assertIn('Final inventory checkpoint failed',str(final['errors']))
        self.assertTrue((self.control/'spool/inventory.json').is_file())

if __name__=='__main__':unittest.main(verbosity=2)
