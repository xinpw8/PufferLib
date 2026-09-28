import unittest
from benchmark import Guard, compare, schedule


def reply(x=0., tick=1):
    return {'state': {'tick': tick, 'phase': 2, 'score': [0, 0],
        'roundNumber': 1, 'terminal': 0, 'roundResult': 0, 'winner': -1,
        'fightResult': 0, 'fightWinner': -1, 'commandResults': [], 'mask': [],
        'actions': [-1, 1], 'qpos': [x], 'qvel': [0.], 'raw': [0.]}, 'commandEvents': []}


class Tests(unittest.TestCase):
    def test_pause_guard_passes_without_input(self):
        calls = []
        g = Guard(fetch=lambda: calls.append('read') or
            {'paused': True, 'ok': True, 'tick': 260}, identities=lambda: True)
        g.check()
        self.assertEqual(calls, ['read'])
        self.assertIsNone(g.failure)

    def test_resume_aborts_only_registered_owned_work(self):
        calls = []
        g = Guard(fetch=lambda: {'paused': False, 'ok': True}, identities=lambda: True)
        g.abort_callbacks.append(lambda: calls.append('owned-abort'))
        with self.assertRaisesRegex(RuntimeError, 'Human resumed'):
            g.check()
        self.assertEqual(calls, ['owned-abort'])
        with self.assertRaises(RuntimeError):
            g.poll()

    def test_changed_identity_fails_before_read(self):
        g = Guard(fetch=lambda: self.fail('Should not query replaced live viewer'),
            identities=lambda: False)
        with self.assertRaisesRegex(RuntimeError, 'identity'):
            g.check()

    def test_unreachable_live_guard_fails_closed(self):
        def fail():
            raise TimeoutError('offline')
        g = Guard(fetch=fail, identities=lambda: True)
        with self.assertRaisesRegex(RuntimeError, 'offline'):
            g.check()

    def test_empty_error_is_latched(self):
        def fail():
            raise TimeoutError()
        g = Guard(fetch=fail, identities=lambda: True)
        with self.assertRaisesRegex(RuntimeError, 'TimeoutError'):
            g.check()
        with self.assertRaisesRegex(RuntimeError, 'TimeoutError'):
            g.poll()

    def test_whole_run_deadline_aborts(self):
        aborted=[]
        g=Guard(fetch=lambda: self.fail('Expired guard must not query'),
            identities=lambda: True, max_seconds=-1)
        g.abort_callbacks.append(lambda: aborted.append(True))
        with self.assertRaisesRegex(RuntimeError, 'deadline'):
            g.poll()
        self.assertEqual(aborted,[True])

    def test_same_fixture_is_exact(self):
        r = compare([reply()], [reply()])
        self.assertTrue(r['exact_observed_trajectory'])

    def test_numeric_drift_is_never_parity(self):
        r = compare([reply(0.)], [reply(.000001)])
        self.assertFalse(r['exact_observed_trajectory'])
        self.assertEqual(r['first_numeric_difference']['component'], 0)
        self.assertEqual(r['numeric']['qpos']['max_abs'], .000001)
        self.assertTrue(r['similar_scatter_is_not_parity_proof'])

    def test_missing_fields_cannot_pass(self):
        with self.assertRaisesRegex(AssertionError, 'Required state'):
            compare([{'state': {'qpos': [], 'qvel': [], 'raw': []}}],
                [{'state': {'qpos': [], 'qvel': [], 'raw': []}}])

    def test_event_and_tick_mismatch_are_not_ignored(self):
        a, b = reply(), reply(tick=2)
        b['commandEvents'] = [{'side': 0, 'accepted': 1}]
        r = compare([a], [b])
        self.assertEqual([d['field'] for d in r['discrete_differences']],
            ['tick', 'commandEvents'])

    def test_schedule_has_releases_edges_and_no_shared_objects(self):
        c = schedule()
        self.assertEqual(len(c), 150)
        self.assertEqual(c[0]['forward'], 0)
        self.assertEqual(c[30]['forward'], .5)
        self.assertEqual(c[60]['yaw'], .5)
        self.assertEqual(c[90]['strafe'], -.25)
        self.assertEqual(c[120]['moveIndex'], 0)
        self.assertTrue(c[121]['cancelAction'])
        self.assertEqual(c[122]['moveIndex'], -1)
        self.assertFalse(c[122]['cancelAction'])
        c[0]['forward'] = 9
        self.assertEqual(c[1]['forward'], 0)


if __name__ == '__main__':
    unittest.main()
