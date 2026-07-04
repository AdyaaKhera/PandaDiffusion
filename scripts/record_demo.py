import sys
import os
import time
import random
import numpy as np
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.panda_env import PandaEnv

def generate_expert_trajectory(env, trial_idx):
    obs = env.reset()
    
    target_cube_x = random.uniform(-0.15, 0.15)
    target_cube_y = random.uniform(-0.15, 0.15)
    
    env.sim.setObjectPosition(env.cube, [target_cube_x, target_cube_y, 0.73], -1)
    
    env.sim.step()
    obs = env.get_observation()
    
    start_pos = obs["hand_position"]
    cube_pos = obs["cube_position"]
    
    target_pos = [cube_pos[0], cube_pos[1], cube_pos[2] + 0.12]
    
    trajectory_data = {
        "observations": [],
        "actions": []
    }
    
    steps = 40
    for i in range(steps):
        alpha = i / float(steps)
        action = [
            (1 - alpha) * start_pos[0] + alpha * target_pos[0],
            (1 - alpha) * start_pos[1] + alpha * target_pos[1],
            (1 - alpha) * start_pos[2] + alpha * target_pos[2]
        ]
        
        trajectory_data["observations"].append(obs)
        trajectory_data["actions"].append(np.array(action, dtype=np.float32))
        
        obs, _, _, _ = env.step(action)
        time.sleep(0.005)
        
    print(f"→ Trial {trial_idx} recorded successfully. Cube position: [{cube_pos[0]:.2f}, {cube_pos[1]:.2f}]")
    return trajectory_data

def main():
    env = PandaEnv(target_object_name="Cuboid")
    
    num_trials = 50
    dataset = []
    
    print(f"Starting randomized expert demonstration collection for {num_trials} trials...")
    
    for tx in range(num_trials):
        trial_data = generate_expert_trajectory(env, tx)
        dataset.append(trial_data)
        
    os.makedirs("data", exist_ok=True)
    output_path = "data/panda_push_expert_demos.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(dataset, f)
        
    print(f"\nCollection complete! Randomized dataset saved to: {output_path}")
    env.close()

if __name__ == "__main__":
    main()