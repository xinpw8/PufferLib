from pathlib import Path
import json,tempfile,unittest
from prepare_inputs import prepare,digest
class StageTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.r=Path(self.tmp.name);self.g=self.r/'session/game';self.g.mkdir(parents=True);self.i=self.g.parent/'input';(self.g/'REK.exe').write_bytes(b'fake');(self.g.parent/'STAGED.json').write_text('{}')
        self.d={'schema':'rek.original_model_inventory.fixture.v1','robot_id':'g1','catalog_resource':'Workshop/RobotCatalog','game_files':[{'file':'data','bytes':4,'sha256':digest(b'data')}]};(self.g/'data').write_bytes(b'data');p=self.g/'BepInEx/interop';p.mkdir(parents=True)
        for n,k in [('Mujoco.Runtime.dll','mujoco_interop_sha256'),('UnityEngine.CoreModule.dll','unity_core_interop_sha256'),('UnityEngine.JSONSerializeModule.dll','unity_json_interop_sha256')]:
            (p/n).write_bytes(n.encode());self.d[k]=digest(n.encode())
        self.f=self.r/'fixture.json';self.save()
    def tearDown(self):self.tmp.cleanup()
    def save(self):self.f.write_text(json.dumps(self.d))
    def run_stage(self):return prepare(self.f,self.i,self.g,'test_inventory_r1')
    def test_success_and_no_overwrite(self):
        x=self.run_stage();self.assertFalse(x['launch_performed']);self.assertEqual(json.loads((self.i/'run-config.json').read_text())['schema'],self.d['schema'].replace('fixture','run'))
        with self.assertRaises(AssertionError):self.run_stage()
    def test_pin_change_rejected_before_writes(self):
        (self.g/'data').write_bytes(b'evil')
        with self.assertRaises(AssertionError):self.run_stage()
        self.assertFalse(self.i.exists())
    def test_escape_and_duplicate_rejected(self):
        self.d['game_files'][0]['file']='../data';self.save()
        with self.assertRaises(AssertionError):self.run_stage()
        self.d['game_files'][0]['file']='data';self.d['game_files'].append(self.d['game_files'][0]);self.save()
        with self.assertRaises(AssertionError):self.run_stage()
    def test_existing_output_and_bad_schema_rejected(self):
        self.d['schema']='wrong';self.save()
        with self.assertRaises(AssertionError):self.run_stage()
        self.d['schema']='rek.original_model_inventory.fixture.v1';self.save();(self.g.parent/'out/oracle').mkdir(parents=True)
        with self.assertRaises(AssertionError):self.run_stage()
if __name__=='__main__':unittest.main()
