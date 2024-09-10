import os

import gymnasium
import numpy as np

from raylib import rl, colors

import pufferlib
from pufferlib.environments.ocean import render
from pufferlib.environments.ocean.breakout.c_breakout import CBreakout


import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


class PufferBreakout(pufferlib.PufferEnv):
    def __init__(
        self,
        report_interval: int = 1,
        num_agents: int = 1,
        # render_mode: str = "rgb_array",
    ) -> None:

        self.report_interval = report_interval

        self.c_env: CBreakout | None = None
        self.tick = 0
        self.reward_sum = 0
        self.score_sum = 0
        self.episodic_returns = np.zeros(num_agents, dtype=np.float32)
        self.dones = np.zeros(num_agents, dtype=np.uint8)
        self.scores = np.zeros(num_agents, dtype=np.int32)

        # This block required by advanced PufferLib env spec
        self.obs_size = 5*5 + 5  # image_size + flat_size
        low = 0
        high = 1
        self.observation_space = gymnasium.spaces.Box(
            low=low, high=high, shape=(5, 5), dtype=np.float32
        )
        # self.observation_space = gymnasium.spaces.Tuple(
        #     gymnasium.spaces.Box(low=low, high=high, shape=(5, 5), dtype=np.float32),
        #     gymnasium.spaces.Box(low=low, high=high, shape=(5,), dtype=np.int8),
        # )
        
        
        self.action_space = gymnasium.spaces.Discrete(1)
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space
        self.num_agents = num_agents
        # self.render_mode = render_mode
        self.emulated = None
        self.done = False
        buf_observations = np.ascontiguousarray(np.zeros((self.num_agents, *self.observation_space.shape), dtype=np.float32))
        self.buf = pufferlib.namespace(buf_observations=buf_observations,
            rewards=np.zeros(self.num_agents, dtype=np.float32),
            terminals=np.zeros(self.num_agents, dtype=bool),
            truncations=np.zeros(self.num_agents, dtype=bool),
            masks=np.ones(self.num_agents, dtype=bool),
        )
        self.actions = np.zeros(self.num_agents, dtype=np.uint8)


        # if render_mode == "ansi":
        #     self.client = render.AnsiRender()
        # elif render_mode == "rgb_array":
        #     self.client = render.RGBArrayRender()
        # elif render_mode == "human":
        #     self.client = RaylibClient(
        #         self.width,
        #         self.height,
        #         self.num_brick_rows,
        #         self.num_brick_cols,
        #         self.brick_positions,
        #         self.ball_width,
        #         self.ball_height,
        #         self.brick_width,
        #         self.brick_height,
        #         self.fps,
        #     )
        # else:
        #     raise ValueError(f"Invalid render mode: {render_mode}")

    def step(self, actions):
        self.actions[:] = actions

        # if self.render_mode == "human" and self.human_action is not None:
        #     self.actions[0] = self.human_action
        # elif self.render_mode == "human":
        #     self.actions[0] = 0

        self.c_env.step(self.actions)

        info = {}
        self.reward_sum += self.buf.rewards.mean()
        self.score_sum += self.scores.mean()

        if self.tick % self.report_interval == 0:
            info["episodic_return"] = self.episodic_returns.mean()
            info["reward"] = self.reward_sum / self.report_interval

            self.reward_sum = 0
            self.score_sum = 0
            self.tick = 0

        self.tick += 1

        return (
            self.buf.observations,
            self.buf.rewards,
            self.buf.terminals,
            self.buf.truncations,
            info,
        )

    def reset(self, seed=None):
        if self.c_env is None:
            self.c_env = CBreakout(
                # dt=self.dt,
                observations=self.buf.observations,
                rewards=self.buf.rewards,
                scores=self.scores,
                episodic_returns=self.episodic_returns,
                dones=self.dones,
                num_agents=self.num_agents,
                obs_size=self.obs_size,
            )

        return self.buf.observations, {}
    

    def close(self):
        pass

    def _calculate_scores(self, action):
        score = 0
        for agent_idx in range(self.num_agents):
            self.scores[agent_idx] = 0
            if self.image_sign == action['image']:
                reward += 0.5
            # if self.flat_sign == action['flat']:
            #     reward += 0.5
            
            
            
    def render(self):
        pass
        # if self.render_mode == "ansi":
        #     return self.client.render(self.grid)
        # elif self.render_mode == "rgb_array":
        #     return self.client.render(self.grid)
        # elif self.render_mode == "raylib":
        #     return self.client.render(self.grid)
        # elif self.render_mode == "human":
        #     action = None

        #     if rl.IsKeyDown(rl.KEY_UP) or rl.IsKeyDown(rl.KEY_W):
        #         action = 1
        #     if rl.IsKeyDown(rl.KEY_LEFT) or rl.IsKeyDown(rl.KEY_A):
        #         action = 2
        #     if rl.IsKeyDown(rl.KEY_RIGHT) or rl.IsKeyDown(rl.KEY_D):
        #         action = 3

        #     self.human_action = action

        #     return self.client.render(
        #         self.paddle_positions[0],
        #         self.paddle_widths[0],
        #         self.paddle_heights[0],
        #         self.ball_positions[0],
        #         self.brick_states[0],
        #         self.scores[0],
        #         self.num_balls[0],
        #     )
        # else:
        #     raise ValueError(f"Invalid render mode: {self.render_mode}")

# class RaylibClient:
#     def __init__(
#         self,
#         width: int,
#         height: int,
#         num_brick_rows: int,
#         num_brick_cols: int,
#         brick_positions: np.ndarray,
#         ball_width: float,
#         ball_height: float,
#         brick_width: int,
#         brick_height: int,
#         fps: float,
#     ) -> None:
#         self.width = width
#         self.height = height
#         self.ball_width = ball_width
#         self.ball_height = ball_height
#         self.brick_width = brick_width
#         self.brick_height = brick_height
#         self.num_brick_rows = num_brick_rows
#         self.num_brick_cols = num_brick_cols
#         self.brick_positions = brick_positions

#         self.running = False

#         self.BRICK_COLORS = [
#             colors.RED,
#             colors.ORANGE,
#             colors.YELLOW,
#             colors.GREEN,
#             colors.SKYBLUE,
#             colors.BLUE,
#         ]

#         # Initialize raylib window
#         rl.InitWindow(width, height, "PufferLib Ray Breakout".encode())
#         rl.SetTargetFPS(fps)

#         sprite_sheet_path = os.path.join(
#             *self.__module__.split(".")[:-1], "puffer_chars.png"
#         )
#         self.puffer = rl.LoadTexture(sprite_sheet_path.encode())

#         sound_path = os.path.join(*self.__module__.split(".")[:-1], "hit.wav")

#         # self.sound = rl.LoadSound(sound_path.encode())

#     def render(
#         self,
#         paddle_position: np.ndarray,
#         paddle_width: int,
#         paddle_height: int,
#         ball_position: np.ndarray,
#         brick_states: np.ndarray,
#         score: float,
#         num_balls: int,
#     ) -> None:
#         rl.BeginDrawing()
#         rl.ClearBackground(render.PUFF_BACKGROUND)

#         # Draw the paddle
#         paddle_x, paddle_y = paddle_position
#         rl.DrawRectangle(
#             int(paddle_x),
#             int(paddle_y),
#             paddle_width,
#             paddle_height,
#             colors.DARKGRAY,
#         )

#         # Draw the ball
#         ball_x, ball_y = ball_position

#         # source_rect = (128, 0, 128, 128)
#         # dest_rect = (ball_x, ball_y, self.ball_width, self.ball_height)

#         # rl.DrawTexturePro(
#         #     self.puffer, source_rect, dest_rect, (0, 0), 0.0, colors.WHITE
#         # )
#         rl.DrawRectangle(
#             int(ball_x), int(ball_y), self.ball_width, self.ball_height, colors.WHITE
#         )

#         # Draw the bricks
#         for row in range(self.num_brick_rows):
#             for col in range(self.num_brick_cols):
#                 idx = row * self.num_brick_cols + col
#                 if brick_states[idx] == 1:
#                     continue

#                 x, y = self.brick_positions[idx]
#                 brick_color = self.BRICK_COLORS[row]
#                 rl.DrawRectangle(
#                     int(x),
#                     int(y),
#                     self.brick_width,
#                     self.brick_height,
#                     brick_color,
#                 )

#         # Draw Score
#         score_text = f"Score: {int(score)}".encode("ascii")
#         rl.DrawText(score_text, 10, 10, 20, colors.WHITE)

#         num_balls_text = f"Balls: {num_balls}".encode("ascii")
#         rl.DrawText(num_balls_text, self.width - 80, 10, 20, colors.WHITE)

#         rl.EndDrawing()

#         # rl.PlaySound(self.sound)

#         return render.cdata_to_numpy()

#     def close(self) -> None:
#         rl.close_window()


def test_performance(timeout=20, atn_cache=1024, num_envs=400):
    tick = 0

    import time

    start = time.time()
    while time.time() - start < timeout:
        atns = actions[tick % atn_cache]
        env.step(atns)
        tick += 1

    print(f"SPS: %f", num_envs * tick / (time.time() - start))


if __name__ == "__main__":
    # Run with c profile
    from cProfile import run

    num_envs = 100
    env = PufferBreakout(num_agents=num_envs)
    env.reset()
    actions = np.random.randint(0, 9, (1024, num_envs))
    test_performance(20, 1024, num_envs)
    # exit(0)

    run("test_performance(20)", "stats.profile")
    import pstats
    from pstats import SortKey

    p = pstats.Stats("stats.profile")
    p.sort_stats(SortKey.TIME).print_stats(25)
    exit(0)

    # test_performance(10)
