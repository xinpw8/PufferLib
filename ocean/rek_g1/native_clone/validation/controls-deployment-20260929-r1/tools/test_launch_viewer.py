import importlib.util,json,tempfile,unittest
from pathlib import Path
from unittest.mock import Mock,patch
spec=importlib.util.spec_from_file_location('launch_viewer',Path(__file__).with_name('launch_viewer.py'))
launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)

class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.app=self.root/'app';self.app.mkdir();self.run=self.root/'run';self.run.mkdir()
        (self.app/'saved-g1-bindings.json').write_text('{}');(self.app/'launch_logged.cjs').write_text('// fixture only')
        self.binary=self.root/'native-fixture';self.binary.write_bytes(b'fixture, never executable')
        self.binary_sha=launcher.sha(self.binary)
        self.worker={'round_seconds':120,'arenas':4};self.env={'REK_PHYSICS_BACKEND':'mujoco_cuda'}
        self.identity={'schema':'rek.native_clone.app_run.v1','port':18772,'worker':self.worker,'env':self.env,
            'filePins':{str(self.binary.resolve()):self.binary_sha},'controlsSha256':launcher.sha(self.app/'saved-g1-bindings.json')}
        self.config={'port':18772,'leagueFile':str(self.run/'league.json'),
            'backends':[{'id':'mujoco','executable':str(self.binary),'workerConfig':str(self.run/'worker.json'),
                         'env':self.env,'logFile':str(self.run/'worker.stderr.log')}],
            'initial':{'backend':'mujoco','opponent':'bot1','humanSide':0,'roundSeconds':120}}
        self.write('identity.json',self.identity);self.write('server.json',self.config);self.write('worker.json',self.worker)
        (self.app/'SOURCE-MANIFEST.json').write_text(json.dumps({'files':[]}))
    def write(self,name,value): (self.run/name).write_text(json.dumps(value))
    def validate(self): return launcher.validate_run(self.app,self.run,self.binary_sha)
    def test_prepared_files_match_identity_and_expected_binary(self):
        config,identity=self.validate();self.assertEqual(config['port'],18772);self.assertEqual(identity,self.identity)
    def test_explicit_new_port_preserves_prepared_identity_binding(self):
        self.config['port']=18773;self.identity['port']=18773
        self.write('server.json',self.config);self.write('identity.json',self.identity)
        config,_=launcher.validate_run(self.app,self.run,self.binary_sha,18773)
        self.assertEqual(config['port'],18773)
        with self.assertRaises(AssertionError):self.validate()
        self.identity['port']=18772;self.write('identity.json',self.identity)
        with self.assertRaises(AssertionError):launcher.validate_run(self.app,self.run,self.binary_sha,18773)
    def test_invalid_requested_port_refused(self):
        for port in [1,65536,'18773']:
            with self.subTest(port=port),self.assertRaises(AssertionError):
                launcher.validate_run(self.app,self.run,self.binary_sha,port)
    def test_guard_transport_failure_cannot_be_retried_as_new_viewer_readiness(self):
        with patch.object(launcher.urllib.request,'urlopen',side_effect=OSError('unavailable')):
            with self.assertRaisesRegex(RuntimeError,'guard unavailable'):
                launcher.check_preserved_viewers([{'port':18772,'processes':[]}])
    def test_changed_worker_environment_executable_port_and_log_rejected(self):
        original=json.loads(json.dumps(self.config))
        changes=[('env',{'REK_PHYSICS_BACKEND':'other'}),('executable',str(self.root/'other-binary')),
                 ('workerConfig',str(self.root/'old-worker.json')),('logFile',str(self.root/'old.log'))]
        for key,value in changes:
            with self.subTest(field=key):
                value_config=json.loads(json.dumps(original));value_config['backends'][0][key]=value;self.write('server.json',value_config)
                with self.assertRaises(AssertionError):self.validate()
        self.write('server.json',original)
        self.write('worker.json',{**self.worker,'round_seconds':20})
        with self.assertRaises(AssertionError):self.validate()
        self.write('worker.json',self.worker);self.write('server.json',{**original,'port':18771})
        with self.assertRaises(AssertionError):self.validate()
    def test_executable_bytes_and_external_expected_hash_are_checked(self):
        with self.assertRaises(AssertionError):launcher.validate_run(self.app,self.run,'0'*64)
        self.binary.write_bytes(b'different executable')
        with self.assertRaises(AssertionError):self.validate()
    def test_post_spawn_record_failure_reaches_owned_group_cleanup(self):
        process=Mock(pid=333);real_open=Path.open
        def file_open(path,*args,**kwargs):
            if path.name=='server-process.json':raise OSError('injected record failure')
            return real_open(path,*args,**kwargs)
        identity={'pid':333,'start_ticks':777,'process_group':333,'session':333}
        socket=Mock();socket.__enter__=Mock(return_value=socket);socket.__exit__=Mock(return_value=False);socket.connect_ex.return_value=111
        argv=['launcher','--app',str(self.app),'--run',str(self.run),'--manifest-sha256',launcher.sha(self.app/'SOURCE-MANIFEST.json'),'--binary-sha256',self.binary_sha]
        with patch.object(launcher.subprocess,'Popen',return_value=process) as popen,patch.object(launcher.socket,'socket',return_value=socket),\
             patch.object(launcher,'process_identity',return_value=identity),patch.object(launcher.os,'sched_getaffinity',return_value={0,1},create=True),\
             patch.object(launcher,'cleanup_owned_group') as cleanup,patch.object(Path,'open',file_open),patch('sys.argv',argv):
            with self.assertRaisesRegex(OSError,'record failure'):launcher.main()
            self.assertTrue(popen.call_args.kwargs['start_new_session']);cleanup.assert_called_once_with(process,777)
    def test_human_resume_after_spawn_reaches_owned_cleanup_without_retry(self):
        process=Mock(pid=333)
        identity={'pid':333,'start_ticks':777,'process_group':333,'session':333}
        socket=Mock();socket.__enter__=Mock(return_value=socket);socket.__exit__=Mock(return_value=False);socket.connect_ex.return_value=111
        argv=['launcher','--app',str(self.app),'--run',str(self.run),'--manifest-sha256',launcher.sha(self.app/'SOURCE-MANIFEST.json'),'--binary-sha256',self.binary_sha]
        with patch.object(launcher.subprocess,'Popen',return_value=process),patch.object(launcher.socket,'socket',return_value=socket),\
             patch.object(launcher,'process_identity',return_value=identity),patch.object(launcher.os,'sched_getaffinity',return_value={0,1},create=True),\
             patch.object(launcher,'check_preserved_viewers',side_effect=[None,AssertionError('human resumed')]) as guard,\
             patch.object(launcher,'cleanup_owned_group') as cleanup,patch('sys.argv',argv):
            with self.assertRaisesRegex(AssertionError,'human resumed'):launcher.main()
            self.assertEqual(guard.call_count,2);cleanup.assert_called_once_with(process,777)
    def test_cleanup_escalates_only_owned_group_after_grace_period(self):
        process=Mock(pid=333)
        with patch.object(launcher,'owned_group_alive',return_value=True),patch.object(launcher.time,'monotonic',side_effect=[0,21]),\
             patch.object(launcher.os,'killpg',create=True) as kill,patch.object(launcher.signal,'SIGKILL',9,create=True):
            launcher.cleanup_owned_group(process,777)
        self.assertEqual([call.args for call in kill.call_args_list],[(333,launcher.signal.SIGTERM),(333,9)])
        process.wait.assert_called_once_with(timeout=5)
    def test_reused_pid_or_different_group_never_signalled(self):
        process=Mock(pid=333)
        for identity in [{'pid':333,'start_ticks':999,'process_group':333,'session':333},
                         {'pid':333,'start_ticks':777,'process_group':222,'session':222}]:
            with self.subTest(identity=identity),patch.object(launcher,'process_identity',return_value=identity),\
                 patch.object(launcher.os,'killpg',create=True) as kill:
                with self.assertRaises(AssertionError):launcher.cleanup_owned_group(process,777)
                kill.assert_not_called()
    def test_orphaned_worker_in_original_group_still_found(self):
        process=Mock(pid=333)
        def identity(pid):
            if pid==333:raise FileNotFoundError()
            return {'pid':444,'start_ticks':778,'process_group':333,'session':333}
        with patch.object(launcher,'process_identity',side_effect=identity),patch.object(Path,'iterdir',return_value=iter([Path('/proc/444')])):
            self.assertTrue(launcher.owned_group_alive(process,777))

if __name__=='__main__':unittest.main()
