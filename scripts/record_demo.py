import sys
import os
import time
import numpy as np
import pickle # Used to serialize and store our dataset locally

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.panda_env import PandaEnv

def generate_expert_trajectory(env, trial_idx):
    """Runs a single demonstration trial and collects observation/action pairs."""
    obs = env.reset()
    
    start_pos = obs["hand_position"]
    cube_pos = obs["cube_position"]
    
    # Define a clean approach trajectory targeting 12cm above the cube
    target_pos = [cube_pos[0], cube_pos[1], cube_pos[2] + 0.12]
    
    trajectory_data = {
        "observations": [],
        "actions": []
    }
    
    steps = 40
    for i in range(steps):
        # Linear interpolation to find the intermediate action step
        alpha = i / float(steps)
        action = [
            (1 - alpha) * start_pos[0] + alpha * target_pos[0],
            (1 - alpha) * start_pos[1] + alpha * target_pos[1],
            (1 - alpha) * start_pos[2] + alpha * target_pos[2]
        ]
        
        # Save current state before executing the action
        trajectory_data["observations"].append(obs)
        trajectory_data["actions"].append(np.array(action, dtype=np.float32))
        
        # Step the environment forward
        obs, _, _, _ = env.step(action)
        time.sleep(0.01) # Rapid collection speed
        
    print(f"→ Trial {trial_idx} recorded successfully ({steps} steps).")
    return trajectory_data

def main():
    env = PandaEnv(target_object_name="Cuboid")
    
    num_trials = 10 # Let's start with a small batch of 10 demonstrations to test the pipeline
    dataset = []
    
    print(f"Starting expert demonstration collection for {num_trials} trials...")
    
    for tx in range(num_trials):
        trial_data = generate_expert_trajectory(env, tx)
        dataset.append(trial_data)
        
    # Ensure our data folder directory exists securely
    os.makedirs("data", exist_ok=True)
    
    # Save the data payload out to a local file structure
    output_path = "data/panda_push_expert_demos.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(dataset, f)
        
    print(f"\nCollection complete! Dataset successfully saved to: {output_path}")
    env.close()

if __name__ == "__main__":
    main()