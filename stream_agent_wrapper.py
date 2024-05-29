import asyncio
import websockets
import json
import gymnasium as gym
import colorsys
import time
from pokegym import data
from pokegym.data import LEVEL, POKE
from pokegym.ram_map import read_m, symbol_lookup  # Ensure correct imports

X_POS_ADDRESS, Y_POS_ADDRESS = 0xD362, 0xD361
MAP_N_ADDRESS = 0xD35E

def color_generator(step=5):  # step=1
    """Generates a continuous spectrum of colors in hex format."""
    hue = 0
    while True:
        # Convert HSL (Hue, Saturation, Lightness) to RGB, then to HEX
        rgb = colorsys.hls_to_rgb(hue / 360, 0.5, 1)
        hex_color = '#{:02x}{:02x}{:02x}'.format(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
        yield hex_color
        hue = (hue + step) % 360

class StreamWrapper(gym.Wrapper):
    def __init__(self, env, stream_metadata={}):
        super().__init__(env)
        self.color_generator = color_generator(step=2)  # step=1
        self.ws_address = "wss://transdimensional.xyz/broadcast"
        self.stream_metadata = stream_metadata
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.websocket = None
        self.loop.run_until_complete(self.establish_wc_connection())

        if hasattr(env, "pyboy"):
            self.emulator = env.pyboy
        elif hasattr(env, "game"):
            self.emulator = env.game
        else:
            raise Exception("Could not find emulator!")

        self.upload_interval = 270  # 25 for 1 agent (i.e. solo play) # 150 ############################################################
        self.steam_step_counter = 0
        self.coord_list = []
        self.start_time = time.time()

        # Initialize Pokémon data once
        self.pokemon_data = self.retrieve_pokemon_data()

    def retrieve_pokemon_data(self):
        pokemon_data = []
        for i in range(6):
            poke = read_m(self.emulator, POKE[i])
            lvl = read_m(self.emulator, LEVEL[i])
            name_info = data.poke_dict.get(poke, {})
            name = name_info.get("name", "")
            pokemon_data.append((name, lvl))
        return pokemon_data

    def step(self, action):
        self.update_pokemon_data()
        x_pos = read_m(self.emulator, X_POS_ADDRESS)
        y_pos = read_m(self.emulator, Y_POS_ADDRESS)
        map_n = read_m(self.emulator, MAP_N_ADDRESS)
        reset_count = self.env.reset_count
        env_id = self.env.env_id
        self.coord_list.append([x_pos, y_pos, map_n])

        # Update stream metadata
        self.stream_metadata['extra'] = f"bet_fixed_window\n{self.pokemon_data[0][0]}: {self.pokemon_data[0][1]}\nenv_id={env_id}"
        self.stream_metadata["color"] = "#FFA550"

        if self.steam_step_counter >= self.upload_interval:
            self.loop.run_until_complete(
                self.broadcast_ws_message(
                    json.dumps(
                        {
                            "metadata": self.stream_metadata,
                            "coords": self.coord_list
                        }
                    )
                )
            )
            self.steam_step_counter = 0
            self.coord_list = []

        self.steam_step_counter += 1

        return self.env.step(action)

    def update_pokemon_data(self):
        # Update only if there is a change in Pokémon data
        updated_data = self.retrieve_pokemon_data()
        if updated_data != self.pokemon_data:
            self.pokemon_data = updated_data

    async def broadcast_ws_message(self, message):
        if self.websocket is None:
            await self.establish_wc_connection()
        if self.websocket is not None:
            try:
                await self.websocket.send(message)
            except websockets.exceptions.WebSocketException as e:
                self.websocket = None

    async def establish_wc_connection(self):
        try:
            self.websocket = await websockets.connect(self.ws_address)
        except:
            self.websocket = None

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def uptime(self):
        return (time.time() - self.start_time) / 60
