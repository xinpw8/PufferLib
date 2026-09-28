import unittest
from smoke_native import wait_phase,assert_inactive_edge,assert_no_deferred

def reply(phase,attempted=0,accepted=0,reason=0):
    result={'attempted':attempted,'accepted':accepted,'rejected':int(attempted and not accepted),'reason':reason}
    return {'state':{'phase':phase,'fightResult':0,'commandResults':[result,{}]},'commandEvents':[dict(side=0,**result)] if attempted or reason else []}
class Tests(unittest.TestCase):
    def test_inactive_batch_requires_one_rejected_edge(self):
        value=reply(3,1,0,4);assert_inactive_edge(value)
        for wrong in [reply(3,1,1,0),reply(3,1,0,3),reply(3)]:
            with self.assertRaises(AssertionError):assert_inactive_edge(wrong)
        value['commandEvents']*=2
        with self.assertRaises(AssertionError):assert_inactive_edge(value)
    def test_phase_wait_uses_snapshots_and_single_neutral_steps(self):
        calls=[];states=iter([reply(3),reply(3,reason=4),reply(2)])
        def call(op,**args):
            calls.append((op,args));return next(states)
        self.assertEqual(wait_phase(call,2,no_deferred=True)['phase'],2)
        self.assertEqual(calls,[('snapshot',{}),('step',{'steps':1,'command':{}}),('step',{'steps':1,'command':{}})])
    def test_deferred_edge_and_unbounded_wait_fail(self):
        for value in [reply(2,1,1),reply(2,1,0,2)]:
            with self.assertRaises(AssertionError):assert_no_deferred(value)
        with self.assertRaisesRegex(AssertionError,'bounded'):wait_phase(lambda *a,**k:reply(3),2,max_ticks=3)
        calls=iter([reply(3),reply(2,1,1)])
        with self.assertRaises(AssertionError):wait_phase(lambda *a,**k:next(calls),2,no_deferred=True)
if __name__=='__main__':unittest.main(verbosity=2)
