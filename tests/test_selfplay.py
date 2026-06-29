import pytest

from pufferlib import selfplay


def test_resolve_opponent_pool_directory_glob_and_list(tmp_path):
    a = tmp_path / 'a.bin'
    b = tmp_path / 'b.bin'
    ignored = tmp_path / 'notes.txt'
    a.write_bytes(b'a')
    b.write_bytes(b'b')
    ignored.write_text('ignore')

    assert selfplay.resolve_opponent_pool(tmp_path) == [str(a), str(b)]
    assert selfplay.resolve_opponent_pool(str(tmp_path / '*.bin')) == [str(a), str(b)]
    assert selfplay.resolve_opponent_pool([a, str(tmp_path / '*.bin')]) == [str(a), str(b)]


def test_external_pool_entries_are_path_only(tmp_path):
    opp = tmp_path / 'opp.bin'
    opp.write_bytes(b'weights')

    entries = selfplay._external_pool_entries({
        'opponent_pool': str(tmp_path),
    })

    assert entries == [{'path': str(opp)}]


def test_resolve_opponent_pool_falls_back_to_repo_relative(monkeypatch, tmp_path):
    repo = tmp_path / 'repo'
    package_dir = repo / 'pufferlib'
    pool_dir = repo / 'robopool'
    package_dir.mkdir(parents=True)
    pool_dir.mkdir()
    opp = pool_dir / 'melee.bin'
    opp.write_bytes(b'weights')
    monkeypatch.setattr(selfplay, '__file__', str(package_dir / 'selfplay.py'))
    monkeypatch.chdir(tmp_path)

    assert selfplay.resolve_opponent_pool('robopool') == [str(opp)]


def test_external_pool_entries_reject_missing_configured_pool(tmp_path):
    with pytest.raises(RuntimeError, match='resolved no .bin files'):
        selfplay._external_pool_entries({'opponent_pool': str(tmp_path / 'missing')})


class _FakePufferl:
    global_step = 10


class _FakeBackend:
    def __init__(self):
        self.saved = []

    def save_weights(self, pufferl, path):
        self.saved.append(path)
        with open(path, 'wb') as f:
            f.write(b'self')

    def count_aligned(self, pufferl, tag_value, reset):
        return 0


def test_step_adds_self_checkpoints_to_external_pool(tmp_path):
    external = tmp_path / 'external.bin'
    external.write_bytes(b'external')
    backend = _FakeBackend()
    pool_state = {
        'pool_dir': str(tmp_path / 'pool'),
        'artifact_owner': True,
        'pool': [{'path': str(external)}],
        'rng': selfplay.np.random.default_rng(0),
        'max_size': 10,
        'snapshot_interval': 1,
        'opp_timeout_steps': 0,
        'num_banks': 1,
        'banks': [selfplay.make_bank_state(str(external), 0, 0)],
        'world_size': 1,
        'last_snapshot_step': 0,
        'shared_state_path': str(tmp_path / 'shared.json'),
        'shared_state_version': 0,
    }

    selfplay.step(_FakePufferl(), backend, pool_state, {'env/n': 0.0}, epoch=0)

    assert pool_state['pool'][0]['path'] == str(external)
    assert len(pool_state['pool']) == 2
    assert backend.saved == [pool_state['pool'][1]['path']]
