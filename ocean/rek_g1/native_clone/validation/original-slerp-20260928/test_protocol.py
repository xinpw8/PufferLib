import copy,json,struct,tempfile,unittest
from pathlib import Path
import slerp_protocol as p

class Protocol(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.calls=self.root/'calls.jsonl';q=p.bits([1,0,0,0,0,0,0,1,.25]);out=p.bits([1,0,0,0])
        self.rows=[{'event':'header','schema':'rek.slerp_calls.v1','mode':'libm_capture','fixture_sha256':'f'*64}]
        self.rows += [{'event':'call','call':i,'trial':0,'tick':50,'phase':'references','input_bits':q,'output_bits':out} for i in range(2)]
        self.rows += [{'event':'end','complete':True,'calls':2}];self.write(self.calls,self.rows)
        self.fixture=self.root/'fixture.json';self.fixture.write_text(json.dumps(p.make_fixture(self.calls)))
        self.f=json.loads(self.fixture.read_text());self.plugin='a'*64
        self.trace=self.root/'oracle.jsonl'
        self.oracle=[{'event':'header','schema':'rek.original_slerp.trace.v1','fixture_sha256':p.digest(self.fixture),'game_sha256':p.GAME,'metadata_sha256':p.META,
            'plugin_sha256':self.plugin,'unity_player_sha256':'277953a7035b1633c239904853bfbea7b2948937ef5567e70c1911c260dd1414',
            'rek_interop_sha256':'faa94fb58e24fda95e2c06810e28b9eb2d6d9f9f8327541976a0dc1011f646d2','unity_interop_sha256':'3ac45305a21f5107c0a9e813c48503cf27944c15ab9f6495396f20f61840104a','game_root':'Z:/session/game'},
            {'event':'entrypoints','methods':[{'method':name,'module':'Z:/session/game/GameAssembly.dll','first32_sha256':'b'*64} for name in ['SlerpWxyz','Slerp']]}]
        for repeat in range(2):
            for c in self.f['cases']:self.oracle.append({'event':'slerp','repeat':repeat,'case_id':c['id'],'input_bits':c['input_bits'],'rek_wxyz_bits':out,'unity_wxyz_bits':out})
        self.oracle.append({'event':'oracle_end','success':True,'rows':2*len(self.f['cases'])})
        self.write(self.trace,self.oracle)
    def tearDown(self):self.tmp.cleanup()
    @staticmethod
    def write(path,rows):path.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    def test_dedup_and_controls_labeled(self):
        self.assertEqual(self.f['cases'][0]['occurrences'],2)
        self.assertEqual(sum(c['kind']=='recorded_native_callback' for c in self.f['cases']),1)
        self.assertTrue(all(c['kind']=='synthetic_control' for c in self.f['cases'][1:]))
    def test_incomplete_capture_rejected(self):
        self.write(self.calls,self.rows[:-1])
        with self.assertRaises(ValueError):p.make_fixture(self.calls)
    def test_changed_duplicate_output_rejected(self):
        self.rows[2]['output_bits']=p.bits([.5,0,0,0]);self.write(self.calls,self.rows)
        with self.assertRaises(ValueError):p.make_fixture(self.calls)
    def test_finite_canonical_bit_contract(self):
        for word in ['0x7fc00000','0x7f800000','3f800000','0x3F800000']:
            with self.subTest(word=word),self.assertRaises(ValueError):p.words([word],1)
    def test_synthetic_closed_table_layout(self):
        blob,report=p.oracle_table(self.fixture,self.trace,self.plugin)
        self.assertEqual(blob[:8],b'RSLPTB1\0');self.assertEqual(len(blob),12+52*len(self.f['cases']))
        self.assertFalse(report['component_replay_performed'])
    def test_repeat_disagreement_rejected(self):
        i=2+len(self.f['cases']);self.oracle[i]['rek_wxyz_bits']=p.bits([.5,0,0,0]);self.oracle[i]['unity_wxyz_bits']=p.bits([.5,0,0,0]);self.write(self.trace,self.oracle)
        with self.assertRaises(ValueError):p.oracle_table(self.fixture,self.trace,self.plugin)
    def test_wrong_plugin_or_module_rejected(self):
        with self.assertRaises(ValueError):p.oracle_table(self.fixture,self.trace,'c'*64)
        self.oracle[1]['methods'][0]['module']='Z:/other/GameAssembly.dll';self.write(self.trace,self.oracle)
        with self.assertRaises(ValueError):p.oracle_table(self.fixture,self.trace,self.plugin)
    def test_partial_duplicate_and_changed_input_rejected(self):
        for kind in ['partial','duplicate','input']:
            rows=copy.deepcopy(self.oracle)
            if kind=='partial':rows.pop()
            elif kind=='duplicate':rows[3]=copy.deepcopy(rows[2])
            else:rows[2]['input_bits'][8]='0x3f000000'
            self.write(self.trace,rows)
            with self.subTest(kind=kind),self.assertRaises(ValueError):p.oracle_table(self.fixture,self.trace,self.plugin)

if __name__=='__main__':unittest.main()
