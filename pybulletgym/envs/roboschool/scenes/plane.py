import pybullet_data

from .scene_bases import Scene
import pybullet


class PlaneScene(Scene):
    multiplayer = False

    def __init__(self, bullet_client, gravity, timestep, frame_skip):
        Scene.__init__(self, bullet_client, gravity, timestep, frame_skip)
        bullet_client.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        plane_id = bullet_client.loadURDF("plane.urdf")
        bullet_client.changeDynamics(plane_id, -1, lateralFriction=100000000000000, spinningFriction=100000000000, rollingFriction=0.1, restitution=0.9)

    def episode_restart(self, bullet_client):
        Scene.episode_restart(self, bullet_client)