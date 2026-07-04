from coppeliasim_zmqremoteapi_client import RemoteAPIClient

client = RemoteAPIClient()
sim = client.require('sim')

joint = sim.getObject('/Franka/joint[0]')

angle = sim.getJointPosition(joint)

print("Joint angle:", angle)