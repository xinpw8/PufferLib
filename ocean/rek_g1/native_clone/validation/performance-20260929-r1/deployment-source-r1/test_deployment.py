import copy,io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import Mock,patch
import deployment_config as c
import prepare_viewer,start_viewer,start_resource_watch

class Tests(unittest.TestCase):
    def test_exact_package_and_all_three_guards(self):
        c.verify_package();guards=c.guards()
        self.assertEqual([v['port'] for v in guards],[18771,18772,18773])
        self.assertEqual(sum(len(v['processes']) for v in guards),8)
        self.assertEqual(guards[2]['processes'][0],{'pid':3543413,'start_ticks':133443582})
    def test_real_baseline_graph_change_is_only_requested_field(self):
        baseline=json.loads((c.HERE/'baseline/worker.json').read_text())
        for mode in ('on','off'):
            value=c.worker_for(mode,baseline)
            self.assertIs(value['cuda_graph_step'],mode=='on')
            value['cuda_graph_step']=False;self.assertEqual(value,baseline)
        self.assertIs(baseline['cuda_graph_step'],False)
        for mode in ('true','',True):
            with self.assertRaises(ValueError):c.worker_for(mode,baseline)
    def test_reject_unrelated_worker_mutation(self):
        baseline=json.loads((c.HERE/'baseline/worker.json').read_text())
        expected=c.worker_for('on',baseline)
        expected['render_model_path']=str(c.APP/'presentation/assets/presentation.playable.xml')
        c.validate_worker(expected,'on',baseline)
        for field,value in [('arenas',4),('seed',1),('round_seconds',30),('cuda_graph_step',False),('controller_decoder_path','/other')]:
            bad=copy.deepcopy(expected);bad[field]=value
            with self.assertRaises(AssertionError):c.validate_worker(bad,'on',baseline)
    def test_fixed_start_command_uses_guarded_generic_launcher(self):
        command=start_viewer.command()
        self.assertEqual(command[1],str(c.HERE/'launch_viewer.py'))
        for name,value in [('--port','18774'),('--run',str(c.RUN)),('--manifest-sha256',c.APP_SHA),('--binary-sha256',c.BINARY_SHA),('--guard-viewers',str(c.HERE/'human-viewers.json')),('--cpus',c.CPUS)]:
            self.assertEqual(command[command.index(name)+1],value)
    def test_existing_run_rejected_before_subprocess_or_guard(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(c,'RUN',Path(directory)),patch.object(prepare_viewer.subprocess,'run') as run,patch.object(prepare_viewer.launcher,'check_preserved_viewers') as guard:
            with self.assertRaisesRegex(AssertionError,'Fresh run-r8'):prepare_viewer.main(['--graph-mode','on'])
            run.assert_not_called();guard.assert_not_called()
    def test_missing_mode_does_not_default_to_graph(self):
        with patch('sys.stderr',io.StringIO()),patch.object(prepare_viewer.subprocess,'run') as run:
            with self.assertRaises(SystemExit):prepare_viewer.main([])
            run.assert_not_called()
    def test_watcher_postspawn_failure_sets_only_own_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory);parent={'pid':987,'start_ticks':1234}
            (run/'STARTED.json').write_text(json.dumps({'process':parent,'paused':True,'tick':0,'url':'http://127.0.0.1:18774/'}))
            (run/'server-process.json').write_text(json.dumps(parent))
            process=Mock(pid=999)
            response=io.BytesIO(b'{"ok":true,"paused":true,"tick":0}')
            with patch.object(c,'RUN',run),patch.object(c,'validate_prepared'),patch.object(Path,'read_text',autospec=True,side_effect=lambda q,*a,**kw: '{} ' if q==c.HERE/'PREPARED.json' else q.read_bytes().decode()),patch.object(start_resource_watch.launcher,'process_identity',side_effect=[{'start_ticks':1234},OSError('post-spawn receipt failure')]),patch.object(start_resource_watch.launcher,'check_preserved_viewers'),patch.object(start_resource_watch.urllib.request,'urlopen',return_value=response),patch.object(start_resource_watch.subprocess,'Popen',return_value=process):
                with self.assertRaisesRegex(OSError,'post-spawn'):start_resource_watch.main()
            self.assertTrue((run/'STOP_RESOURCE_WATCH').is_file());process.wait.assert_called_once_with(timeout=8)
            process.terminate.assert_not_called();process.kill.assert_not_called()

if __name__=='__main__':unittest.main(verbosity=2)
