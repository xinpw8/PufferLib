import unittest
from compare_workers import differences,requests,comparison_scope

class ComparisonTests(unittest.TestCase):
    def test_identical_nested_state(self):
        s={'state':{'qpos':[0,1.25,-3],'tick':1,'ok':True},'rounds':[],'commandEvents':[]}
        self.assertEqual(differences(s,s),[])
    def test_one_ulp_rejected(self):
        self.assertEqual(differences({'qpos':[1.]},{'qpos':[1.0000001192092896]})[0]['path'],'$.qpos[0]')
    def test_negative_zero_rejected(self):
        self.assertTrue(differences([0.],[-0.]))
    def test_events_and_keys_preserved(self):
        self.assertTrue(differences({'rounds':[{'winner':0}]},{'rounds':[{'winner':1}]}))
        self.assertTrue(differences({'commandEvents':[]},{}))
    def test_fixed_schedule(self):
        s=requests()
        self.assertEqual(s[0],{'op':'snapshot'})
        self.assertEqual(sum(bool(x.get('benchmark')) for x in s),1)
        self.assertEqual(s[-1]['steps'],512)
        self.assertEqual(sum(x['op']=='reset' for x in s),5)
        self.assertTrue(any(x.get('command',{}).get('cancelAction') for x in s))
        self.assertFalse(any(x['op']=='frame' for x in s))
    def test_failure_scope_does_not_claim_exact(self):
        self.assertIn('equality failed',comparison_scope(False))
        self.assertNotIn('fields compare exactly',comparison_scope(False))
        self.assertIn('fields compare exactly',comparison_scope(True))
if __name__=='__main__':unittest.main()
