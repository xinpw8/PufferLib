"""
Diablo PufferEnv - Native C binding with DevilutionX shared memory interface.

This launches the DevilutionX binary and interfaces via shared memory,
with observation computation and action submission handled in C for performance.
"""

import configparser
import gymnasium
import numpy as np
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pufferlib


def _resolve_ai_path():
    """Find DevilutionX-AI directory."""
    root = Path(__file__).resolve().parents[4]
    ai_dir = root / "DevilutionX-AI" / "ai"
    if str(ai_dir) not in sys.path:
        sys.path.insert(0, str(ai_dir))
    return ai_dir


AI_PATH = _resolve_ai_path()

# Import DevilutionX-AI modules for game launching
try:
    import procutils
except ImportError as e:
    raise ImportError(
        "Diablo integration requires the DevilutionX-AI Python deps. "
        "Install them via `pip install -r ../DevilutionX-AI/ai/requirements.txt`"
    ) from e

# Import C binding
try:
    from pufferlib.ocean.diablo import binding
except ImportError:
    binding = None


# Constants
ENV_ID_DEFAULT = "Diablo-FindNextLevel-v1"
VIEW_RADIUS = 10
VIEW_SIZE = 2 * VIEW_RADIUS + 1  # 21
ENV_FLAG_COUNT = 19
ENV_STATUS_LEN = 5
OBS_CHANNELS = ENV_FLAG_COUNT + ENV_STATUS_LEN  # 24
OBS_SHAPE = (VIEW_SIZE, VIEW_SIZE, OBS_CHANNELS)


def _load_defaults(ai_path: Path):
    """Load default config from DevilutionX-AI."""
    cfg_path = ai_path / "diablo-ai.ini"
    cfg = configparser.ConfigParser()
    cfg.read(cfg_path)
    defaults = {
        "diablo_build_path": (ai_path.parent / "build"),
        "mshared_filename": cfg.get("default", "diablo-mshared-filename", fallback="devilutionx-shared.mem"),
    }
    return defaults


def _find_shared_memory_offset(pid, mshared_path, timeout=30):
    """
    Wait for shared memory file to be created and find the memory offset
    by examining /proc/PID/maps.
    """
    # Wait for file to exist
    start = time.time()
    while not os.path.exists(mshared_path):
        if time.time() - start > timeout:
            raise TimeoutError(f"Shared memory file {mshared_path} not created within {timeout}s")
        time.sleep(0.1)

    # Give the game a moment to fully initialize
    time.sleep(0.5)

    # Find memory offset from /proc/PID/maps
    try:
        offset = procutils.find_shared_memory_offset(pid, mshared_path)
    except Exception:
        # Fallback: use a common base offset
        # This is the minimum address from VARS in devilutionx.py
        offset = 5865768
    return offset


class _GameInstance:
    """Manages a single DevilutionX game instance."""

    def __init__(self, game_cfg, ai_path):
        self.proc = None
        self.state_dir = None
        self.log_file = None
        self.mshared_path = None
        self.base_offset = 0

        # Read and format config template
        cfg_template_path = ai_path / "diablo.ini.template"
        with open(cfg_template_path, "r") as f:
            cfg = f.read()

        cfg = cfg.format(
            seed=game_cfg["seed"],
            fixed_seed=1 if game_cfg.get("fixed-seed", False) else 0,
            automap_active=1 if game_cfg.get("gui", False) else 0,
            skip_progress=1 if game_cfg.get("gui", False) else 0,
            skip_animation=0 if game_cfg.get("gui", False) else 1,  # Show animations in GUI
            headless=0 if game_cfg.get("gui", False) else 1,
            game_ticks_per_step=game_cfg.get("game-ticks-per-step", 10),
            step_mode=1,  # Always use step mode - game reads from ring buffer
            mshared_filename=game_cfg["mshared-filename"],
            no_monsters=1 if game_cfg.get("no-monsters", False) else 0,
            harmless_barrels=1 if game_cfg.get("harmless-barrels", False) else 0,
            no_auto_walk_on_seconday_action=1 if game_cfg.get("no-auto-walk-on-seconday-action", True) else 0,
        )

        # Create temp directory for game state
        prefix = f"diablo-{game_cfg['seed']}-{os.getpid()}-"
        self.state_dir = tempfile.TemporaryDirectory(prefix=prefix)
        cfg_file_path = os.path.join(self.state_dir.name, "diablo.ini")
        with open(cfg_file_path, "w") as f:
            f.write(cfg)

        self.log_file = open(os.path.join(self.state_dir.name, "diablo.log"), "w", buffering=1)

        # Launch game binary
        cmd = [
            game_cfg["diablo-bin-path"],
            "--config-dir", self.state_dir.name,
            "--save-dir", self.state_dir.name,
        ]
        if not game_cfg.get("gui", False):
            cmd.append("-n")  # Skip intro videos

        # Clean environment (pygame can mess with SDL_AUDIODRIVER)
        env = os.environ.copy()
        if "SDL_AUDIODRIVER" in env:
            del env["SDL_AUDIODRIVER"]

        self.proc = subprocess.Popen(cmd, stdout=self.log_file, stderr=self.log_file, env=env)
        self.mshared_path = os.path.abspath(
            os.path.join(self.state_dir.name, game_cfg["mshared-filename"])
        )

        # Find memory offset
        self.base_offset = _find_shared_memory_offset(self.proc.pid, self.mshared_path)

    def close(self):
        """Terminate game and clean up."""
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None
        if self.log_file:
            self.log_file.close()
            self.log_file = None
        if self.state_dir:
            self.state_dir.cleanup()
            self.state_dir = None


class Diablo(pufferlib.PufferEnv):
    """
    Diablo environment using native C bindings for performance.

    Launches DevilutionX game binaries and interfaces via shared memory.
    Observation computation and action submission are handled in C.
    """

    def __init__(
        self,
        render_mode=None,
        num_envs=1,
        seed=0,
        env_id=ENV_ID_DEFAULT,
        game_ticks_per_step=10,
        view_radius=VIEW_RADIUS,
        no_monsters=False,
        harmless_barrels=False,
        fixed_seed=False,
        diablo_build_path=None,
        mshared_filename=None,
        max_steps=10000,
        buf=None,
    ):
        if binding is None:
            raise ImportError(
                "Diablo C binding not compiled. Run: python setup.py build_ext --inplace --force"
            )

        defaults = _load_defaults(AI_PATH)
        build_path = Path(diablo_build_path) if diablo_build_path else Path(defaults["diablo_build_path"])
        build_path = build_path.expanduser().resolve()
        mshared = mshared_filename or defaults["mshared_filename"]

        if not (build_path / "devilutionx").exists():
            raise FileNotFoundError(f"devilutionx binary not found at {build_path}/devilutionx")
        if not (build_path / "spawn.mpq").exists():
            raise FileNotFoundError(f"spawn.mpq missing in {build_path}; download per DevilutionX-AI README")

        # Force single env when rendering (you only want to watch one game)
        self.render_mode = render_mode if render_mode else None
        if self.render_mode == "human":
            num_envs = 1

        self.num_envs = num_envs
        self.num_agents = num_envs
        self.agents_per_batch = num_envs
        self.view_radius = view_radius
        self.max_steps = max_steps
        self.game_ticks_per_step = game_ticks_per_step

        self.single_observation_space = gymnasium.spaces.Box(
            low=0.0, high=1.0, shape=OBS_SHAPE, dtype=np.float32
        )
        self.single_action_space = gymnasium.spaces.Discrete(11)

        super().__init__(buf=buf)

        # Launch game instances
        self._games = []
        self._c_envs = []

        for idx in range(num_envs):
            game_cfg = {
                "mshared-filename": mshared,
                "diablo-bin-path": str((build_path / "devilutionx").resolve()),
                "seed": seed + idx,
                "no-monsters": no_monsters,
                "harmless-barrels": harmless_barrels,
                "no-auto-walk-on-seconday-action": True,
                "view-radius": view_radius,
                "game-ticks-per-step": game_ticks_per_step,
                "step-mode": True,
                "gui": render_mode == "human",
                "fixed-seed": fixed_seed,
            }

            # Launch game
            game = _GameInstance(game_cfg, AI_PATH)
            self._games.append(game)

            # Initialize C environment
            env_id = binding.env_init(
                self.observations[idx : idx + 1],
                self.actions[idx : idx + 1],
                self.rewards[idx : idx + 1],
                self.terminals[idx : idx + 1],
                self.truncations[idx : idx + 1],
                seed + idx,
                view_radius=view_radius,
                max_steps=max_steps,
                game_ticks_per_step=game_ticks_per_step,
            )

            # Pass mmap info to C binding
            binding.env_put(
                env_id,
                mmap_path=game.mshared_path,
                base_offset=game.base_offset,
                step_mode=True,  # Always use step mode
                goal_x=0,  # Will be set by reset
                goal_y=0,
            )

            self._c_envs.append(env_id)

        # Create vectorized env handle
        self.c_vec = binding.vectorize(*self._c_envs)
        self._reset_all()

    def _reset_all(self):
        """Reset all environments."""
        binding.vec_reset(self.c_vec, 0)

    def reset(self, seed=0):
        """Reset all environments."""
        binding.vec_reset(self.c_vec, seed)
        return self.observations, []

    def step(self, actions):
        """Step all environments."""
        self.actions[:] = actions
        binding.vec_step(self.c_vec)

        info_list = []
        log = binding.vec_log(self.c_vec)
        if log:
            info_list.append(log)

        return self.observations, self.rewards, self.terminals, self.truncations, info_list

    def render(self):
        """Render is handled by DevilutionX when gui=True."""
        if self.render_mode == "human" and self._c_envs:
            binding.vec_render(self.c_vec, 0)
        return None

    def close(self):
        """Clean up all resources."""
        if hasattr(self, "c_vec") and self.c_vec:
            binding.vec_close(self.c_vec)
            self.c_vec = None
        for game in getattr(self, "_games", []):
            game.close()
        self._games = []
        self._c_envs = []
