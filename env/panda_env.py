from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np

class PandaEnv:
    def __init__(self, target_object_name="Cuboid"):

        self.client = RemoteAPIClient()
        self.sim = self.client.require("sim")
        
        self.robot = self.sim.getObject("/Franka")
        
        self.joints = []
        objects = self.sim.getObjectsInTree(self.robot)
        for obj in objects:
            alias = self.sim.getObjectAlias(obj)
            if alias == "joint":
                self.joints.append(obj)
            elif alias == "connection":
                self.end_effector = obj
                
        try:
            self.cube = self.sim.getObject(f"/{target_object_name}")
        except Exception:

            self.cube = self.sim.getObject(target_object_name)

        print(f"PandaEnv initialized successfully. Found {len(self.joints)} joints and target '{target_object_name}'.")

    def get_observation(self):
        """
        Gathers positions from the simulation environment.
        Returns a dictionary matching the data structure needed for training.
        """
        joint_positions = [self.sim.getJointPosition(j) for j in self.joints]
        hand_position = self.sim.getObjectPosition(self.end_effector, -1)
        cube_position = self.sim.getObjectPosition(self.cube, -1)
        
        return {
            "joint_positions": np.array(joint_positions, dtype=np.float32),
            "hand_position": np.array(hand_position, dtype=np.float32),
            "cube_position": np.array(cube_position, dtype=np.float32)
        }

    def reset(self):
        self.sim.stopSimulation()
        while self.sim.getSimulationState() != self.sim.simulation_stopped:
            pass
        self.sim.startSimulation()
        return self.get_observation()

    def close(self):
        self.sim.stopSimulation()