import io,json,os,sys,tempfile,unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch
import passive as p

class RemoteFile:
    def __init__(self,path):self.path=path;self.f=path.open('rb')
    def read(self,n):return self.f.read(n)
    def seek(self,n):return self.f.seek(n)
    def stat(self):return self.path.stat()
    def __enter__(self):return self
    def __exit__(self,*args):self.f.close()
class Sftp:
    def __init__(self,path):self.path=path
    def open(self,*args):return RemoteFile(self.path)
class Tests(unittest.TestCase):
    def test_independent_viewer_port_preserves_default_and_bounds(self):
        self.assertEqual(p.parse_args(['tunnel']).port,18771)
        self.assertEqual(p.parse_args(['tunnel','--port','18772']).port,18772)
        for value in ['0','1023','65536','-1']:
            with self.assertRaisesRegex(ValueError,'port'):p.parse_args(['tunnel','--port',value])
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='clone-passive-test-');self.root=Path(self.temp.name).resolve()
    def tearDown(self):
        assert self.root.name.startswith('clone-passive-test-')
        self.temp.cleanup()
    def test_append_resume_and_tail_guard(self):
        dest=self.root/'stream';src=io.BytesIO(b'a'*70000)
        r=p.append_file(src,dest,70000,lambda:False);self.assertEqual(r['added_bytes'],70000)
        src=io.BytesIO(b'a'*70000+b'new\n')
        r=p.append_file(src,dest,70004,lambda:False);self.assertEqual(r['added_bytes'],4);self.assertEqual(dest.read_bytes(),src.getvalue())
        before=dest.read_bytes()
        with self.assertRaisesRegex(IOError,'prefix'):p.append_file(io.BytesIO(b'a'*69999+b'bnew\nmore'),dest,70008,lambda:False)
        self.assertEqual(dest.read_bytes(),before)
        with self.assertRaisesRegex(IOError,'truncated'):p.append_file(io.BytesIO(b'a'),dest,1,lambda:False)
        self.assertEqual(dest.read_bytes(),before)
    def test_interrupted_partial_resumes_without_duplicate(self):
        data=b'x'*(p.CHUNK+128);dest=self.root/'stream';calls=0
        def stop():
            nonlocal calls
            calls+=1;return calls>1
        with self.assertRaises(InterruptedError):p.append_file(io.BytesIO(data),dest,len(data),stop)
        self.assertEqual(dest.stat().st_size,p.CHUNK)
        p.append_file(io.BytesIO(data),dest,len(data),lambda:False)
        self.assertEqual(dest.read_bytes(),data)
    def test_immutable_full_source_local_nas_hash_and_conflict(self):
        source=self.root/'source.json';source.write_text('{"n":1}')
        local=self.root/'local.json';nas=self.root/'nas.json'
        r=p.immutable_file(Sftp(source),'remote',local,nas,lambda:False)
        self.assertEqual(r['sha256'],p.sha(source));self.assertEqual(p.sha(local),p.sha(nas))
        source.write_text('{"n":2}')
        with self.assertRaisesRegex(IOError,'differs'):p.immutable_file(Sftp(source),'remote',local,nas,lambda:False)
        self.assertEqual(nas.read_text(),'{"n":1}')
    def test_incomplete_png_is_not_published(self):
        source=self.root/'source';source.write_bytes(b'\x89PNG\r\n\x1a\n'+b'0'*30)
        with self.assertRaisesRegex(IOError,'footer'):p.immutable_file(Sftp(source),'remote',self.root/'local.png',self.root/'nas.png',lambda:False)
        self.assertFalse((self.root/'nas.png').exists())
    def test_scoped_paths_and_fresh_identity(self):
        for name in ['../x','/x','a/../../b',r'..\x','a:b']:
            with self.assertRaises(ValueError):p.scoped(self.root,name)
        self.assertEqual(p.scoped(self.root,'frames/1.png'),self.root/'frames/1.png')
        local=self.root/'spool';nas=self.root/'nas';identity=p.initialize(local,nas,False)
        self.assertEqual(p.initialize(local,nas,True),identity)
        with self.assertRaises(FileExistsError):p.initialize(local,nas,False)
        wrong=json.loads((nas/'MIRROR-IDENTITY.json').read_text());wrong['id']='different';p.atomic(nas/'MIRROR-IDENTITY.json',wrong)
        with self.assertRaisesRegex(RuntimeError,'identity'):p.initialize(local,nas,True)
    def test_stop_prevents_any_connection(self):
        (self.root/'STOP').write_text('stop')
        for mode in ['tunnel','mirror']:
            for lifetime in [[],['--until-stop']]:
                with patch.object(sys,'argv',['passive.py',mode,'--local',str(self.root)]+lifetime),patch.object(p,'connect') as connect:
                    with self.assertRaisesRegex(RuntimeError,'STOP'):p.main()
                    connect.assert_not_called()
    def test_singleton_is_exclusive_and_released_on_exit(self):
        path=self.root/'owned.lock'
        with p.OwnedLock(path):
            with self.assertRaises(OSError):
                with p.OwnedLock(path):pass
        with p.OwnedLock(path):pass
    def test_stop_during_mirror_retries_is_published_to_owned_nas(self):
        control=self.root/'control';control.mkdir();nas=self.root/'nas'
        def unavailable():
            (control/'STOP').write_text('stop');raise OSError('synthetic disconnected SSH')
        with patch.object(p,'connect',side_effect=unavailable):
            p.mirror(SimpleNamespace(local=control,nas=nas,resume=False,minutes=1))
        local_status=json.loads((control/'mirror-status.json').read_text())
        self.assertEqual(local_status['phase'],'stopped');self.assertEqual(local_status['reason'],'STOP')
        self.assertEqual(json.loads((nas/'mirror-status.json').read_text()),local_status)
    def test_until_stop_has_no_clock_expiry_and_timed_default_retains_cap(self):
        for mode in ['mirror','tunnel']:
            untimed=p.parse_args([mode,'--until-stop']);self.assertIsNone(untimed.minutes)
            with patch.object(p.time,'monotonic',return_value=1e12):
                self.assertIsNone(p.deadline_for(untimed));self.assertFalse(p.expired(None))
            default=p.parse_args([mode]);self.assertEqual(default.minutes,240)
            with patch.object(p.time,'monotonic',return_value=100):self.assertEqual(p.deadline_for(default),14500)
            with patch.object(p.time,'monotonic',return_value=14500):self.assertTrue(p.expired(14500))
            self.assertEqual(p.parse_args([mode,'--minutes','1']).minutes,1)
            for value in ['0','-1','241','nan','inf']:
                with self.assertRaises(ValueError):p.parse_args([mode,'--minutes',value])
            with patch.object(sys,'stderr',io.StringIO()),self.assertRaises(SystemExit):p.parse_args([mode,'--minutes','240','--until-stop'])
    def test_untimed_mirror_still_honors_stop_and_publishes_it(self):
        control=self.root/'control';control.mkdir();nas=self.root/'nas'
        def stopped_connection():
            (control/'STOP').write_text('stop');raise OSError('synthetic disconnect')
        args=SimpleNamespace(local=control,nas=nas,resume=False,minutes=None,until_stop=True)
        with patch.object(p,'connect',side_effect=stopped_connection):p.mirror(args)
        self.assertEqual(json.loads((nas/'mirror-status.json').read_text())['reason'],'STOP')
    def test_untimed_tunnel_loop_honors_stop_without_real_socket_or_ssh(self):
        control=self.root/'control';control.mkdir()
        class Transport:
            def is_active(self):return True
        class Client:
            closed=False
            def get_transport(self):return Transport()
            def close(self):self.closed=True
        client=Client()
        def init(server,address,handler):
            self.assertEqual(address,('127.0.0.1',18772));server.fake_address=address
        def handle(server):(control/'STOP').write_text('stop')
        args=SimpleNamespace(local=control,minutes=None,until_stop=True,port=18772)
        with patch.object(p,'connect',return_value=client),patch.object(p.socketserver.ThreadingTCPServer,'__init__',init),patch.object(p.socketserver.ThreadingTCPServer,'handle_request',handle),patch.object(p.socketserver.ThreadingTCPServer,'server_close'):
            p.tunnel(args)
        self.assertTrue(client.closed)
        self.assertEqual(json.loads((control/'tunnel-status.json').read_text())['reason'],'STOP')
if __name__=='__main__':unittest.main(verbosity=2)
