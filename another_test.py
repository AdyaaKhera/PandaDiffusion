import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.panda_env import PandaEnv

def main():
    print("Initializing environment...")
    env = PandaEnv(target_object_name="Cuboid")
    
    print("\nResetting environment...")
    obs = env.reset()
    
    print("\n--- Current Observation Snapshot ---")
    print(f"Joint Positions (7): {obs['joint_positions']}")
    print(f"Hand Position (X, Y, Z): {obs['hand_position']}")
    print(f"Cube Position (X, Y, Z): {obs['cube_position']}")
    print("------------------------------------")
    
    env.close()
    print("\nTest complete!")

if __name__ == "__main__":
    main()