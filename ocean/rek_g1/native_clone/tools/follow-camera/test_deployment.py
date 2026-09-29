import copy,json,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import deployment_config as c
import prepare_viewer,start_viewer
import launch_viewer as launcher

class Tests(unittest.TestCase):
    def test_exact_package_and_all_four_guards(self):
        c.verify_package();guards=c.guards()
        self.assertEqual([v['port'] for v in guards],[18771,18772,18773,18774])
        self.assertEqual(sum(len(v['processes']) for v in guards),11)
        self.assertEqual(guards[3]['processes'][0]['pid'],3702006)
    def test_baseline_is_run_r8_graph_on_single_arena(self):
        baseline=json.loads((c.HERE/'baseline/worker.json').read_text())
        self.assertEqual(c.worker_template(baseline),baseline)
        self.assertIs(baseline['cuda_graph_step'],True);self.assertEqual(baseline['arenas'],1)
    def test_reject_unrelated_worker_mutation(self):
        baseline=json.loads((c.HERE/'baseline/worker.json').read_text())
        expected=c.worker_template(baseline)
        expected['render_model_path']=str(c.APP/'presentation/assets/presentation.playable.xml')
        c.validate_worker(expected,baseline)
        for field,value in [('arenas',4),('seed',1),('round_seconds',30),('cuda_graph_step',False),
                            ('controller_decoder_path','/other'),('render_model_path',baseline['render_model_path'])]:
            bad=copy.deepcopy(expected);bad[field]=value
            with self.assertRaises(AssertionError):c.validate_worker(bad,baseline)
    def test_fixed_start_command_uses_guarded_generic_launcher(self):
        command=start_viewer.command()
        self.assertEqual(command[1],str(c.HERE/'launch_viewer.py'))
        for name,value in [('--port','18775'),('--run',str(c.RUN)),('--manifest-sha256',c.APP_SHA),('--binary-sha256',c.BINARY_SHA),('--guard-viewers',str(c.HERE/'human-viewers.json')),('--cpus',c.CPUS)]:
            self.assertEqual(command[command.index(name)+1],value)
    def test_existing_run_rejected_before_subprocess_or_guard(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(c,'RUN',Path(directory)),patch.object(prepare_viewer.subprocess,'run') as run,patch.object(prepare_viewer.launcher,'check_preserved_viewers') as guard:
            with self.assertRaisesRegex(AssertionError,'Fresh run-r9'):prepare_viewer.main()
            run.assert_not_called();guard.assert_not_called()
    def test_disk_cap_stops_only_its_own_viewer_group(self):
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory);(run/'frames').mkdir();(run/'frames/00000001.png').write_bytes(b'x'*4096)
            viewer=subprocess.Popen([sys.executable,'-c','import subprocess,sys,time;subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"]);time.sleep(60)'],start_new_session=True)
            bystander=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True)
            try:
                time.sleep(.3);ticks=launcher.process_identity(viewer.pid)['start_ticks']
                cap=subprocess.run([sys.executable,str(c.HERE/'disk_cap.py'),'--pid',str(viewer.pid),'--start-ticks',str(ticks),
                    '--run',str(run),'--cap-bytes','1024','--disk-fraction','.99','--interval','.2'],timeout=40)
                self.assertEqual(cap.returncode,0)
                self.assertEqual(viewer.wait(timeout=5),-15)
                self.assertTrue((run/'DISK-CAP-TRIPPED.json').is_file());self.assertTrue((run/'STOP_RESOURCE_WATCH').is_file())
                self.assertGreater(json.loads((run/'DISK-CAP-TRIPPED.json').read_text())['run_bytes'],1024)
                self.assertIsNone(bystander.poll())
            finally:
                for p in (viewer,bystander):
                    if p.poll() is None:p.kill()
                    p.wait(timeout=5)
    def test_disk_cap_under_cap_exits_with_viewer(self):
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory)
            viewer=subprocess.Popen([sys.executable,'-c','import time;time.sleep(1.5)'],start_new_session=True)
            time.sleep(.3);ticks=launcher.process_identity(viewer.pid)['start_ticks']
            cap=subprocess.run([sys.executable,str(c.HERE/'disk_cap.py'),'--pid',str(viewer.pid),'--start-ticks',str(ticks),
                '--run',str(run),'--cap-bytes',str(10**9),'--disk-fraction','.99','--interval','.2'],timeout=20)
            self.assertEqual(cap.returncode,0);self.assertEqual(viewer.wait(timeout=5),0)
            self.assertFalse((run/'DISK-CAP-TRIPPED.json').exists())
            self.assertEqual(json.loads((run/'disk-cap-status.json').read_text())['viewer_pid'],viewer.pid)

if __name__=='__main__':unittest.main(verbosity=2)
