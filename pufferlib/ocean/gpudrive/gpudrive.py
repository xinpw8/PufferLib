import numpy as np
import gymnasium
import json
import struct

import pufferlib
# from pufferlib.ocean.gpudrive.cy_gpudrive import CyGPUDrive

# class GPUDrive(pufferlib.PufferEnv):
#     def __init__(self, num_envs=1, render_mode=None, report_interval=1,
#             width=1280, height=1024,
#             num_agents=4,
#             active_agents=4,
#             human_agent_idx=0,
#             buf = None):

#         # env
#         self.num_agents = num_envs*num_agents
#         self.render_mode = render_mode
#         self.report_interval = report_interval
        
#         self.num_obs = 3000
#         self.single_observation_space = gymnasium.spaces.Box(low=-1, high=1,
#             shape=(self.num_obs,), dtype=np.float32)
#         self.single_action_space = gymnasium.spaces.Discrete(90)

#         super().__init__(buf=buf)
#         self.c_envs = CyGPUDrive(self.observations, self.actions, self.rewards,
#             self.terminals, num_envs, width, height, num_agents, active_agents)


#     def reset(self, seed=None):
#         self.c_envs.reset()
#         self.tick = 0
#         return self.observations, []

#     def step(self, actions):
#         self.actions[:] = actions
#         self.c_envs.step()
#         self.tick += 1

#         info = []
#         if self.tick % self.report_interval == 0:
#             log = self.c_envs.log()
#             if log['episode_length'] > 0:
#                 info.append(log)
#         return (self.observations, self.rewards,
#             self.terminals, self.truncations, info)

#     def render(self):
#         self.c_envs.render()
        
#     def close(self):
#         self.c_envs.close() 

def entity_dtype():
    return np.dtype([
        ('type', np.int32),
        ('road_object_id', np.int32),
        ('road_point_id', np.int32),
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('vx', np.float32),
        ('vy', np.float32),
        ('vz', np.float32),
        ('width', np.float32),
        ('length', np.float32),
        ('height', np.float32),
        ('heading', np.float32),
        ('valid', np.int32),
        ('goal_position_x', np.float32),
        ('goal_position_y', np.float32),
        ('goal_position_z', np.float32),
        ('collision_state', np.int32),
    ])

def save_map_binary(map_data, output_file):
    """Saves map data in a binary format readable by C"""
    with open(output_file, 'wb') as f:
        # Count total entities
        num_entities = len(map_data.get('objects', [])) + len(map_data.get('roads', []))
        f.write(struct.pack('i', num_entities))
        active_agent_indices = []
        metadata = map_data.get('metadata', {})
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
            
        if isinstance(metadata, dict) and 'tracks_to_predict' in metadata:
            for track in metadata['tracks_to_predict']:
                active_agent_indices.append(track['track_index'])
        print("Active agent indices:", active_agent_indices)
        # Write active agent indices
        f.write(struct.pack('i', len(active_agent_indices)))
        for idx in active_agent_indices:
            f.write(struct.pack('i', idx))
        
        # Write objects
        for obj in map_data.get('objects', []):
            # Write base entity data
            obj_type = obj.get('type', 1)
            if(obj_type =='vehicle'):
                obj_type = 1
            elif(obj_type == 'pedestrian'):
                obj_type = 2
            elif(obj_type == 'cyclist'):
                obj_type = 3
            f.write(struct.pack('i', obj_type))  # type
            f.write(struct.pack('i', obj.get('id', 0)))    # road_object_id
            f.write(struct.pack('i', obj.get('road_point_id', 0)))  # road_point_id
            f.write(struct.pack('i', 91))                  # array_size
            
            # Write position arrays
            positions = obj.get('position', [])
            for i in range(91):
                pos = positions[i] if i < len(positions) else {'x': 0.0, 'y': 0.0, 'z': 0.0}
                f.write(struct.pack('f', float(pos.get('x', 0.0))))
            for i in range(91):
                pos = positions[i] if i < len(positions) else {'x': 0.0, 'y': 0.0, 'z': 0.0}
                f.write(struct.pack('f', float(pos.get('y', 0.0))))
            for i in range(91):
                pos = positions[i] if i < len(positions) else {'x': 0.0, 'y': 0.0, 'z': 0.0}
                f.write(struct.pack('f', float(pos.get('z', 0.0))))
            
            # Write velocity arrays
            velocities = obj.get('velocity', [])
            for arr, key in [(velocities, 'x'), (velocities, 'y')]:
                for i in range(91):
                    vel = arr[i] if i < len(arr) else {'x': 0.0, 'y': 0.0}
                    f.write(struct.pack('f', float(vel.get(key, 0.0))))
            f.write(struct.pack('91f', *[0.0] * 91))  # vz (unused)
            
            # Write heading and valid arrays
            headings = obj.get('heading', [])
            f.write(struct.pack('91f', *[float(headings[i]) if i < len(headings) else 0.0 for i in range(91)]))
            
            valids = obj.get('valid', [])
            f.write(struct.pack('91i', *[int(valids[i]) if i < len(valids) else 0 for i in range(91)]))
            
            # Write scalar fields
            f.write(struct.pack('f', float(obj.get('width', 0.0))))
            f.write(struct.pack('f', float(obj.get('length', 0.0))))
            f.write(struct.pack('f', float(obj.get('height', 0.0))))
            goal_pos = obj.get('goalPosition', {'x': 0, 'y': 0, 'z': 0})  # Get goalPosition object with default
            f.write(struct.pack('f', float(goal_pos.get('x', 0.0))))  # Get x value
            f.write(struct.pack('f', float(goal_pos.get('y', 0.0))))  # Get y value
            f.write(struct.pack('f', float(goal_pos.get('z', 0.0))))  # Get z value
        
        # Write roads
        for idx, road in enumerate(map_data.get('roads', [])):
            geometry = road.get('geometry', [])
            size = len(geometry)
            road_type = road.get('map_element_id', 0)
            if(road_type >=0 and road_type <=3):
                road_type = 4
            elif(road_type >=5 and road_type <=13):
                road_type = 5
            elif(road_type >=14 and road_type <=16):
                road_type = 6
            elif(road_type == 17):
                road_type = 7
            elif(road_type == 18):
                road_type = 8
            elif(road_type == 19):
                road_type = 9
            elif(road_type == 20):
                road_type = 10
            # Write base entity data
            f.write(struct.pack('i', road_type))  # type
            f.write(struct.pack('i', 0))          # road_object_id
            f.write(struct.pack('i', road.get('id', 0)))    # road_point_id
            f.write(struct.pack('i', size))                 # array_size
            
            # Write position arrays
            for coord in ['x', 'y', 'z']:
                for point in geometry:
                    f.write(struct.pack('f', float(point.get(coord, 0.0))))
            # Write scalar fields
            f.write(struct.pack('f', float(road.get('width', 0.0))))
            f.write(struct.pack('f', float(road.get('length', 0.0))))
            f.write(struct.pack('f', float(road.get('height', 0.0))))
            goal_pos = road.get('goalPosition', {'x': 0, 'y': 0, 'z': 0})  # Get goalPosition object with default
            f.write(struct.pack('f', float(goal_pos.get('x', 0.0))))  # Get x value
            f.write(struct.pack('f', float(goal_pos.get('y', 0.0))))  # Get y value
            f.write(struct.pack('f', float(goal_pos.get('z', 0.0))))  # Get z value

def load_map(map_name, binary_output=None):
    """Loads a JSON map and optionally saves it as binary"""
    with open(map_name, 'r') as f:
        map_data = json.load(f)
    
    if binary_output:
        save_map_binary(map_data, binary_output)
    
    # Create numpy array for Python usage
    total_entities = len(map_data.get('roads', [])) + len(map_data.get('objects', []))
    entities_flat = np.zeros(total_entities, dtype=entity_dtype())
    # entities = np.frombuffer(entities_flat, dtype=entity_dtype()).view(np.recarray)
    
    # # Process roads
    # current_idx = 0
    # for entity in map_data.get('roads', []):
    #     for json_key, struct_key in ROAD_MAPPING.items():
    #         if json_key in entity:
    #             if struct_key in ['type', 'road_object_id', 'valid', 'collision_state']:
    #                 entities[struct_key][current_idx] = entity[json_key]
    #             else:
    #                 entities[struct_key][current_idx] = float(entity[json_key])
    #     current_idx += 1
    
    # # Process objects
    # for entity in map_data.get('objects', []):
    #     for json_key, struct_key in OBJECT_MAPPING.items():
    #         if json_key in entity:
    #             if struct_key in ['type', 'road_object_id', 'valid', 'collision_state']:
    #                 entities[struct_key][current_idx] = entity[json_key]
    #             else:
    #                 entities[struct_key][current_idx] = float(entity[json_key])
    #     current_idx += 1
    
    # return entities_flat


if __name__ == '__main__':
    load_map('resources/tfrecord-00000-of-01000_4.json', 'map.bin')
