import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.panda_env import PandaEnv

def main():
    env = PandaEnv(target_object_name="Cuboid")
    obs = env.reset()
    
    start_pos = obs["hand_position"]
    cube_pos = obs["cube_position"]
    
    print(f"\nStarting Hand Position: {start_pos}")
    print(f"Target Cube Position: {cube_pos}")
    print("\nMoving towards the cube gradually...")

    approach_target = [cube_pos[0], cube_pos[1], cube_pos[2] + 0.1] 
    
    steps = 50
    for i in range(steps):
        alpha = i / float(steps)
        current_action = [
            (1 - alpha) * start_pos[0] + alpha * approach_target[0],
            (1 - alpha) * start_pos[1] + alpha * approach_target[1],
            (1 - alpha) * start_pos[2] + alpha * approach_target[2]
        ]
        
        obs, reward, done, info = env.step(current_action)
        
        print(f"Step {i+1}/{steps} | Current Hand XYZ: {obs['hand_position']}")
        time.sleep(0.05) 
        
    print("\nArrived above target object!")
    env.close()

if __name__ == "__main__":
    main()