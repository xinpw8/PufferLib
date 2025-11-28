"""
Native PufferEnv wrapper around the real DevilutionX-AI Gym env.
This launches the DevilutionX binary in headless step mode, shares
memory with the game, and converts the dungeon bitfield + status
into a (21, 21, 24) float tensor that matches the original CNN
training pipeline (19 environment flags + 5 status channels).
"""

import configparser
import os
import sys
from pathlib import Path

import gymnasium
import numpy as np

import pufferlib


def _resolve_ai_path():
    root = Path(__file__).resolve().parents[4]
    ai_dir = root / "DevilutionX-AI" / "ai"
    if str(ai_dir) not in sys.path:
        sys.path.append(str(ai_dir))
    return ai_dir


AI_PATH = _resolve_ai_path()
# Import after path injection
try:
    import diablo_env  # type: ignore  # noqa: E402
    import diablo_state  # type: ignore  # noqa: E402
except ImportError as e:
    raise ImportError(
        "Diablo integration requires the DevilutionX-AI Python deps "
        "(e.g., numba, futex). Install them via "
        "`pip install -r ../DevilutionX-AI/ai/requirements.txt` from the repo root."
    ) from e


ENV_ID_DEFAULT = "Diablo-FindNextLevel-v1"
ENV_STATUS_HIGH = 0xFFFFF
ENV_STATUS_LEN = 5  # monsters_cnt, hp, mode, x, y
ENV_FLAG_COUNT = len(diablo_state.EnvironmentFlag)
OBS_CHANNELS = ENV_FLAG_COUNT + ENV_STATUS_LEN
OBS_SHAPE = (21, 21, OBS_CHANNELS)  # view radius 10 -> 21x21


def _bitfield_to_channels(env_bitfield: np.ndarray, env_status: np.ndarray) -> np.ndarray:
    """Convert (H,W) uint32 dungeon bitfield + status into (H,W,C) float32."""
    bit_indices = np.arange(ENV_FLAG_COUNT, dtype=np.uint32)
    bit_planes = ((env_bitfield[..., None] >> bit_indices) & 1).astype(np.float32)
    status_norm = np.clip(env_status.astype(np.float32) / ENV_STATUS_HIGH, 0.0, 1.0)
    status_planes = np.broadcast_to(status_norm.reshape(1, 1, -1), env_bitfield.shape + (ENV_STATUS_LEN,))
    return np.concatenate([bit_planes, status_planes], axis=-1)


def _load_defaults(ai_path: Path):
    cfg_path = ai_path / "diablo-ai.ini"
    cfg = configparser.ConfigParser()
    cfg.read(cfg_path)
    defaults = {
        "diablo_build_path": (ai_path.parent / "build"),
        "mshared_filename": cfg["default"]["diablo-mshared-filename"],
    }
    return defaults


def _env_to_ascii(env_bitfield: np.ndarray, game=None) -> str:
    """Convert environment bitfield to ASCII representation."""
    EF = diablo_state.EnvironmentFlag
    lines = []
    h, w = env_bitfield.shape

    for j in range(h):
        row_chars = []
        for i in range(w):
            tile = env_bitfield[j, i]
            if tile == 0:
                s = ' '
            elif tile & EF.Player.value:
                s = '↓'  # Default player direction
            elif tile & EF.Monster.value:
                s = '@'
            elif tile & EF.Goal.value:
                s = '⚑'
            elif tile & EF.Wall.value:
                s = '#'
            elif tile & EF.Door.value:
                s = 'd' if tile & EF.Open.value else 'D'
            elif tile & EF.Chest.value:
                s = 'C' if tile & EF.Interactable.value else 'c'
            elif tile & EF.Barrel.value:
                s = 'B'
            elif tile & EF.Item.value:
                s = 'I'
            elif tile & EF.NextTrigger.value:
                s = 'v'
            elif tile & EF.PrevTrigger.value:
                s = '^'
            elif tile & EF.WarpTrigger.value:
                s = '$'
            elif tile & EF.Visible.value:
                s = '.'
            elif tile & EF.Explored.value:
                s = ' '
            else:
                s = ' '
            row_chars.append(s)
        lines.append(' '.join(row_chars))

    return '\n'.join(lines)


class _WrappedEnv:
    """Thin wrapper holding a single DevilutionX-AI Gym env instance."""

    def __init__(self, env_id, game_cfg):
        env_cls = {e["id"]: e["entry_point"] for e in diablo_env.DIABLO_ENVS}[env_id]
        # diablo_state expects to find diablo.ini.template in the CWD
        cwd = os.getcwd()
        try:
            os.chdir(AI_PATH)
            game = diablo_state.DiabloGame.run_or_attach(game_cfg)
        finally:
            os.chdir(cwd)
        env_cfg = game_cfg.copy()
        env_cfg["seed"] = game_cfg["seed"]
        env_cls.tune_config(env_cfg)
        self.env = env_cls(env_cfg, game=game)
        self.game = game
        self.ep_return = 0.0
        self.ep_len = 0
        self._last_obs = None

    def reset(self, seed=None):
        obs, _ = self.env.reset(seed=seed)
        self.ep_return = 0.0
        self.ep_len = 0
        self._last_obs = obs
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(int(action))
        self.ep_return += reward
        self.ep_len += 1
        self._last_obs = obs
        done = terminated or truncated
        log = None
        if done:
            log = {
                "episode_return": float(self.ep_return),
                "episode_length": float(self.ep_len),
                "perf": float(self.ep_return),
                "success_rate": 1.0 if reward > 0 else 0.0,
                "n": 1.0,
            }
            obs, _ = self.env.reset()
            self.ep_return = 0.0
            self.ep_len = 0
            self._last_obs = obs
        return obs, reward, terminated, truncated, info, log

    def render_ascii(self) -> str:
        """Return ASCII representation of the current game state."""
        if self._last_obs is None:
            return ""
        env_bitfield = self._last_obs.get("env", np.zeros((21, 21), dtype=np.uint32))
        header = f"Episode Return: {self.ep_return:.2f} | Steps: {self.ep_len}"
        ascii_map = _env_to_ascii(env_bitfield, self.game)
        return f"{header}\n\n{ascii_map}"

    def close(self):
        self.env.close()


class Diablo(pufferlib.PufferEnv):
    def __init__(
        self,
        render_mode=None,
        num_envs=1,
        seed=0,
        env_id=ENV_ID_DEFAULT,
        game_ticks_per_step=10,
        view_radius=10,
        no_monsters=False,
        harmless_barrels=False,
        fixed_seed=False,
        diablo_build_path=None,
        mshared_filename=None,
        buf=None,
    ):
        defaults = _load_defaults(AI_PATH)
        build_path = Path(diablo_build_path) if diablo_build_path else Path(defaults["diablo_build_path"])
        build_path = build_path.expanduser().resolve()
        mshared = mshared_filename or defaults["mshared_filename"]

        if not (build_path / "devilutionx").exists():
            raise FileNotFoundError(f"devilutionx binary not found at {build_path}/devilutionx")
        if not (build_path / "spawn.mpq").exists():
            raise FileNotFoundError(f"spawn.mpq missing in {build_path}; download per DevilutionX-AI README")

        self.num_envs = num_envs
        self.num_agents = num_envs
        self.agents_per_batch = num_envs
        # Handle empty string from INI config as None
        self.render_mode = render_mode if render_mode else None
        self.single_observation_space = gymnasium.spaces.Box(
            low=0.0, high=1.0, shape=OBS_SHAPE, dtype=np.float32
        )
        self.single_action_space = gymnasium.spaces.Discrete(11)
        self.env_id = env_id

        super().__init__(buf=buf)

        self._envs = []
        seeds = [seed + i for i in range(num_envs)]
        for idx in range(num_envs):
            game_cfg = {
                "mshared-filename": mshared,
                "diablo-bin-path": str((build_path / "devilutionx").resolve()),
                "seed": seeds[idx],
                "no-monsters": no_monsters,
                "harmless-barrels": harmless_barrels,
                "no-auto-walk-on-seconday-action": True,
                "view-radius": view_radius,
                "game-ticks-per-step": game_ticks_per_step,
                "step-mode": True,
                "gui": False if render_mode is None else True,
                "fixed-seed": fixed_seed,
                "log-to-stdout": False,
                "no-actions": False,
                "exploration-door-attraction": False,
                "exploration-door-backtrack-penalty": False,
            }
            self._envs.append(_WrappedEnv(env_id, game_cfg))

        self._reset_all()

    def _reset_all(self):
        for i, env in enumerate(self._envs):
            obs = env.reset()
            self._write_obs(i, obs)
        self.rewards[:] = 0.0
        self.terminals[:] = 0
        self.truncations[:] = 0

    def _write_obs(self, idx, obs_dict):
        obs_img = _bitfield_to_channels(obs_dict["env"], obs_dict["env-status"])
        self.observations[idx] = obs_img

    def reset(self, seed=0):
        for i, env in enumerate(self._envs):
            obs = env.reset(seed=seed + i)
            self._write_obs(i, obs)
        return self.observations, []

    def step(self, actions):
        info_list = []
        self.terminals[:] = 0
        self.truncations[:] = 0
        for i, (env, action) in enumerate(zip(self._envs, actions)):
            obs, reward, terminated, truncated, info, log = env.step(action)
            self.rewards[i] = reward
            self.terminals[i] = 1 if terminated else 0
            self.truncations[i] = 1 if truncated else 0
            self._write_obs(i, obs)
            if log:
                info_list.append(log)
        return self.observations, self.rewards, self.terminals, self.truncations, info_list

    def render(self):
        """Render the environment.

        - render_mode='ansi': returns ASCII string representation
        - render_mode='human': GUI is handled by DevilutionX (launched with gui=True)
        """
        if self.render_mode == 'ansi' and self._envs:
            return self._envs[0].render_ascii()
        return None

    def close(self):
        for env in self._envs:
            env.close()
