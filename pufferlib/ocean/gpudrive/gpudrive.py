import numpy as np
import gymnasium
import json
import struct

import pufferlib
from pufferlib.ocean.gpudrive.cy_gpudrive import CyGPUDrive, entity_dtype

class GPUDrive(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, report_interval=1,
            width=1280, height=1024,
            active_agent_count=5,
            human_agent_idx=0,
            num_agents = 5,
            buf = None):

        # env
        self.num_agents = num_envs*num_agents
        self.render_mode = render_mode
        self.report_interval = report_interval
        
        self.num_obs = 1030
        self.single_observation_space = gymnasium.spaces.Box(low=-1, high=1,
            shape=(self.num_obs,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.MultiDiscrete([7, 13])


        super().__init__(buf=buf)
        self.c_envs = CyGPUDrive(self.observations, self.actions, self.rewards,
            self.terminals, num_envs, num_agents, human_agent_idx)


    def reset(self, seed=None):
        self.c_envs.reset()
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        self.c_envs.step()
        self.tick+=1
        info = []
        if self.tick % self.report_interval == 0:
            log = self.c_envs.log()
            if log['episode_length'] > 0:
                info.append(log)
        return (self.observations, self.rewards,
            self.terminals, self.truncations, info)

    def render(self):
        self.c_envs.render()
        
    def close(self):
        self.c_envs.close() 
def calculate_area(p1, p2, p3):
    # Calculate the area of the triangle using the determinant method
    return 0.5 * abs((p1['x'] - p3['x']) * (p2['y'] - p1['y']) - (p1['x'] - p2['x']) * (p3['y'] - p1['y']))

def simplify_polyline(geometry, polyline_reduction_threshold):
    """Simplify the given polyline using a method inspired by Visvalingham-Whyatt, optimized for Python."""
    num_points = len(geometry)
    if num_points < 3:
        return geometry  # Not enough points to simplify

    skip = [False] * num_points
    skip_changed = True

    while skip_changed:
        skip_changed = False
        k = 0
        while k < num_points - 1:
            k_1 = k + 1
            while k_1 < num_points - 1 and skip[k_1]:
                k_1 += 1
            if k_1 >= num_points - 1:
                break

            k_2 = k_1 + 1
            while k_2 < num_points and skip[k_2]:
                k_2 += 1
            if k_2 >= num_points:
                break

            point1 = geometry[k]
            point2 = geometry[k_1]
            point3 = geometry[k_2]
            area = calculate_area(point1, point2, point3)

            if area < polyline_reduction_threshold:
                skip[k_1] = True
                skip_changed = True
                k = k_2
            else:
                k = k_1

    return [geometry[i] for i in range(num_points) if not skip[i]]

def save_map_binary(map_data, output_file):
    """Saves map data in a binary format readable by C"""
    with open(output_file, 'wb') as f:
        # Count total entities
        print(len(map_data.get('objects', [])))
        print(len(map_data.get('roads', [])))
        num_entities = len(map_data.get('objects', [])) + len(map_data.get('roads', []))
        f.write(struct.pack('i', num_entities))
        # Write objects
        for obj in map_data.get('objects', []):
            # Write base entity data
            obj_type = obj.get('type', 1)
            if(obj_type =='vehicle'):
                obj_type = 1
            elif(obj_type == 'pedestrian'):
                obj_type = 2;
            elif(obj_type == 'cyclist'):
                obj_type = 3;
            f.write(struct.pack('i', obj_type))  # type
            f.write(struct.pack('i', obj.get('id', 0)))   # id  
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
            for arr, key in [(velocities, 'x'), (velocities, 'y'), (velocities, 'z')]:
                for i in range(91):
                    vel = arr[i] if i < len(arr) else {'x': 0.0, 'y': 0.0, 'z': 0.0}
                    f.write(struct.pack('f', float(vel.get(key, 0.0))))
            
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
            road_type = road.get('map_element_id', 0)
            # breakpoint()
            if(len(geometry) > 10 and road_type >= 14 and road_type <=16):
                geometry = simplify_polyline(geometry, .1)
            size = len(geometry)
            # breakpoint()
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
            f.write(struct.pack('i', road.get('id', 0)))    # id
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
    # objects = [obj for obj in map_data.get('objects', []) 
    #            if obj.get('type') == 'vehicle' or obj.get('type') == 1]
    # roads = map_data.get('roads', [])
    # total_entities = len(roads) + len(objects)
    
    # # Create the recarray with the entity_dtype
    # entities = np.zeros(total_entities, dtype=entity_dtype()).view(np.recarray)
    
    # # Process objects
    # current_idx = 0
    # for obj in objects:
    #     # Set basic entity data
    #     obj_type = 1 if obj.get('type') == 'vehicle' or obj.get('type') == 1 else 1
    #     entities[current_idx].type = obj_type
    #     entities[current_idx].road_object_id = obj.get('id', 0)
    #     entities[current_idx].road_point_id = obj.get('road_point_id', 0)
    #     entities[current_idx].array_size = 91
        
    #     # Allocate memory for trajectory arrays
    #     # Note: In a real implementation, you'd need to manage this memory properly
    #     # and ensure it's freed when no longer needed
    #     traj_x = np.zeros(91, dtype=np.float32)
    #     traj_y = np.zeros(91, dtype=np.float32)
    #     traj_z = np.zeros(91, dtype=np.float32)
    #     traj_vx = np.zeros(91, dtype=np.float32)
    #     traj_vy = np.zeros(91, dtype=np.float32)
    #     traj_vz = np.zeros(91, dtype=np.float32)
    #     traj_heading = np.zeros(91, dtype=np.float32)
    #     traj_valid = np.zeros(91, dtype=np.int32)
        
    #     # Fill trajectory arrays
    #     positions = obj.get('position', [])
    #     for i in range(91):
    #         pos = positions[i] if i < len(positions) else {'x': 0.0, 'y': 0.0, 'z': 0.0}
    #         traj_x[i] = float(pos.get('x', 0.0))
    #         traj_y[i] = float(pos.get('y', 0.0))
    #         traj_z[i] = float(pos.get('z', 0.0))
        
    #     velocities = obj.get('velocity', [])
    #     for i in range(91):
    #         vel = velocities[i] if i < len(velocities) else {'x': 0.0, 'y': 0.0}
    #         traj_vx[i] = float(vel.get('x', 0.0))
    #         traj_vy[i] = float(vel.get('y', 0.0))
    #         # vz is unused, already zeros
        
    #     headings = obj.get('heading', [])
    #     for i in range(91):
    #         traj_heading[i] = float(headings[i]) if i < len(headings) else 0.0
        
    #     valids = obj.get('valid', [])
    #     for i in range(91):
    #         traj_valid[i] = int(valids[i]) if i < len(valids) else 0
        
    #     # Store pointers to arrays in the entity
    #     # Note: This assumes the arrays won't be garbage collected
    #     entities[current_idx].traj_x = traj_x.ctypes.data
    #     entities[current_idx].traj_y = traj_y.ctypes.data
    #     entities[current_idx].traj_z = traj_z.ctypes.data
    #     entities[current_idx].traj_vx = traj_vx.ctypes.data
    #     entities[current_idx].traj_vy = traj_vy.ctypes.data
    #     entities[current_idx].traj_vz = traj_vz.ctypes.data
    #     entities[current_idx].traj_heading = traj_heading.ctypes.data
    #     entities[current_idx].traj_valid = traj_valid.ctypes.data
        
    #     # Set scalar fields
    #     entities[current_idx].width = float(obj.get('width', 0.0))
    #     entities[current_idx].length = float(obj.get('length', 0.0))
    #     entities[current_idx].height = float(obj.get('height', 0.0))
        
    #     goal_pos = obj.get('goalPosition', {'x': 0, 'y': 0, 'z': 0})
    #     entities[current_idx].goal_position_x = float(goal_pos.get('x', 0.0))
    #     entities[current_idx].goal_position_y = float(goal_pos.get('y', 0.0))
    #     entities[current_idx].goal_position_z = float(goal_pos.get('z', 0.0))
        
    #     # Set current state (first position in trajectory)
    #     if len(positions) > 0:
    #         pos = positions[0]
    #         entities[current_idx].x = float(pos.get('x', 0.0))
    #         entities[current_idx].y = float(pos.get('y', 0.0))
    #         entities[current_idx].z = float(pos.get('z', 0.0))
        
    #     if len(velocities) > 0:
    #         vel = velocities[0]
    #         entities[current_idx].vx = float(vel.get('x', 0.0))
    #         entities[current_idx].vy = float(vel.get('y', 0.0))
    #         entities[current_idx].vz = 0.0
        
    #     if len(headings) > 0:
    #         entities[current_idx].heading = float(headings[0])
        
    #     entities[current_idx].valid = 1  # Assume valid
    #     entities[current_idx].collision_state = 0  # Assume no collision
        
    #     current_idx += 1
    
    # # Process roads
    # for road in roads:
    #     geometry = road.get('geometry', [])
    #     size = len(geometry)
        
    #     # Map road types
    #     road_type = road.get('map_element_id', 0)
    #     if road_type >= 0 and road_type <= 3:
    #         road_type = 4
    #     elif road_type >= 5 and road_type <= 13:
    #         road_type = 5
    #     elif road_type >= 14 and road_type <= 16:
    #         road_type = 6
    #     elif road_type == 17:
    #         road_type = 7
    #     elif road_type == 18:
    #         road_type = 8
    #     elif road_type == 19:
    #         road_type = 9
    #     elif road_type == 20:
    #         road_type = 10
        
    #     entities[current_idx].type = road_type
    #     entities[current_idx].road_object_id = 0
    #     entities[current_idx].road_point_id = road.get('id', 0)
    #     entities[current_idx].array_size = size
        
    #     # Allocate memory for geometry arrays
    #     geom_x = np.zeros(size, dtype=np.float32)
    #     geom_y = np.zeros(size, dtype=np.float32)
    #     geom_z = np.zeros(size, dtype=np.float32)
        
    #     # Fill geometry arrays
    #     for i, point in enumerate(geometry):
    #         geom_x[i] = float(point.get('x', 0.0))
    #         geom_y[i] = float(point.get('y', 0.0))
    #         geom_z[i] = float(point.get('z', 0.0))
        
    #     # Store pointers to arrays
    #     entities[current_idx].traj_x = geom_x.ctypes.data
    #     entities[current_idx].traj_y = geom_y.ctypes.data
    #     entities[current_idx].traj_z = geom_z.ctypes.data
        
    #     # Set scalar fields
    #     entities[current_idx].width = float(road.get('width', 0.0))
    #     entities[current_idx].length = float(road.get('length', 0.0))
    #     entities[current_idx].height = float(road.get('height', 0.0))
        
    #     goal_pos = road.get('goalPosition', {'x': 0, 'y': 0, 'z': 0})
    #     entities[current_idx].goal_position_x = float(goal_pos.get('x', 0.0))
    #     entities[current_idx].goal_position_y = float(goal_pos.get('y', 0.0))
    #     entities[current_idx].goal_position_z = float(goal_pos.get('z', 0.0))

    #     current_idx += 1
    entities = np.zeros(1, dtype=entity_dtype())
    return entities


if __name__ == '__main__':
        load_map('resources/tfrecord-00000-of-01000_325.json', 'map.bin')

