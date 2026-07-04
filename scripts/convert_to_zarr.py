import os
import pickle
import numpy as np
import zarr

def convert_pkl_to_zarr(pkl_path, zarr_path):
    with open(pkl_path, "rb") as f:
        dataset = pickle.load(f)
    
    all_joint_positions = []
    all_hand_positions = []
    all_cube_positions = []
    all_actions = []
    episode_ends = []
    
    current_step = 0
    for trial in dataset:
        for obs in trial["observations"]:
            all_joint_positions.append(obs["joint_positions"])
            all_hand_positions.append(obs["hand_position"])
            all_cube_positions.append(obs["cube_position"])
        
        for act in trial["actions"]:
            all_actions.append(act)
            
        current_step += len(trial["actions"])
        episode_ends.append(current_step)
        
    joint_positions = np.array(all_joint_positions, dtype=np.float32)
    hand_positions = np.array(all_hand_positions, dtype=np.float32)
    cube_positions = np.array(all_cube_positions, dtype=np.float32)
    actions = np.array(all_actions, dtype=np.float32)
    episode_ends = np.array(episode_ends, dtype=np.int32)
    
    obs_concat = np.hstack([joint_positions, hand_positions, cube_positions])
    
    root = zarr.open_group(zarr_path, mode='w')
    data_group = root.create_group('data')
    meta_group = root.create_group('meta')
    
    obs_arr = data_group.create_array(name='obs', shape=obs_concat.shape, chunks=(100, obs_concat.shape[1]), dtype='float32')
    obs_arr[:] = obs_concat
    
    act_arr = data_group.create_array(name='action', shape=actions.shape, chunks=(100, actions.shape[1]), dtype='float32')
    act_arr[:] = actions
    
    meta_arr = meta_group.create_array(name='episode_ends', shape=episode_ends.shape, chunks=(100,), dtype='int32')
    meta_arr[:] = episode_ends
    
    print(f"Converted dataset saved to Zarr format at: {zarr_path}")

if __name__ == "__main__":
    convert_pkl_to_zarr("data/panda_push_expert_demos.pkl", "data/panda_push_expert_demos.zarr")