from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np

class PandaEnv:
    def __init__(self, target_object_name="Cuboid"):
        self.client = RemoteAPIClient()
        self.sim = self.client.require("sim")
        self.simIK = self.client.require("simIK") # Load the IK plugin
        
        # Track main persistent handles
        self.robot = self.sim.getObject("/Franka")
        self.joints = []
        self.end_effector = None
        
        # Dynamically discover permanent components
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

        # Initialize placeholder properties for dynamic IK elements
        self.ik_env = None
        self.ik_group = None
        self.ik_target = None
        self.ik_element = None

        print(f"PandaEnv base structures initialized. Target: '{target_object_name}'.")

    def _setup_ik(self):
        """Internal helper to build the IK chain fresh whenever simulation starts."""
        self.ik_env = self.simIK.createEnvironment()
        self.ik_group = self.simIK.createGroup(self.ik_env)
        
        # Generate the dummy frame dynamically in the active simulation session
        self.ik_target = self.sim.createDummy(0.01)
        self.sim.setObjectAlias(self.ik_target, "IK_Target")
        
        # Configure the active IK link
        self.ik_element = self.simIK.addElementFromScene(
            self.ik_env, self.ik_group, self.robot, self.end_effector, self.ik_target, self.simIK.handle_all
        )
        
        # Set calculation constraints
        self.simIK.setGroupCalculation(self.ik_env, self.ik_group, self.simIK.method_pseudo_inverse, 0.1, 99)

    def get_observation(self):
        joint_positions = [self.sim.getJointPosition(j) for j in self.joints]
        hand_position = self.sim.getObjectPosition(self.end_effector, -1)
        cube_position = self.sim.getObjectPosition(self.cube, -1)
        
        return {
            "joint_positions": np.array(joint_positions, dtype=np.float32),
            "hand_position": np.array(hand_position, dtype=np.float32),
            "cube_position": np.array(cube_position, dtype=np.float32)
        }

    def step(self, action):
            # Convert any NumPy floats back into basic Python floats for the ZeroMQ API
            target_xyz = [float(x) for x in action]
            
            # Safely move the runtime target frame
            self.sim.setObjectPosition(self.ik_target, target_xyz, -1)
            
            # Execute IK solver step
            self.simIK.handleGroup(self.ik_env, self.ik_group, {"applyToScene": True})
            
            # Advance physics clock
            self.sim.step()
            
            return self.get_observation(), 0.0, False, {}

    def reset(self):
        # Stop and wait for reset sequence to clear
        self.sim.stopSimulation()
        while self.sim.getSimulationState() != self.sim.simulation_stopped:
            pass
            
        # Re-initialize the physical state
        self.sim.startSimulation()
        
        # CRITICAL: Build the temporary IK elements fresh for this session block
        self._setup_ik()
        
        # Snap our target dummy frame directly onto the starting hand position
        initial_hand = self.sim.getObjectPosition(self.end_effector, -1)
        self.sim.setObjectPosition(self.ik_target, initial_hand, -1)
        
        return self.get_observation()

    def close(self):
        self.sim.stopSimulation()