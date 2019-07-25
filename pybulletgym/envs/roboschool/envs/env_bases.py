import gym, gym.spaces, gym.utils, gym.utils.seeding
import numpy as np
import pybullet
from pybullet_envs.bullet import bullet_client

from pkg_resources import parse_version

from tkinter import *

class BaseBulletEnv(gym.Env):
    """
    Base class for Bullet physics simulation environments in a Scene.
    These environments create single-player scenes and behave like normal Gym environments, if
    you don't use multiplayer.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 60
    }

    def __init__(self, robot, render=False):
        self.scene = None
        self.physicsClientId = -1
        self.ownsPhysicsClient = 0
        self.camera = Camera()
        self.isRender = render
        self.robot = robot
        self._seed()
        self._cam_dist = 3
        self._cam_yaw = 0
        self._cam_pitch = -30
        self._render_width = 320
        self._render_height = 240

        self.action_space = robot.action_space
        self.observation_space = robot.observation_space

        self.tk_root = Tk()
        self.HUD = HUD(self.tk_root)

    def configure(self, args):
        self.robot.args = args

    def _seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        self.robot.np_random = self.np_random  # use the same np_randomizer for robot as for env
        return [seed]

    def _reset(self):
        if self.physicsClientId < 0:
            self.ownsPhysicsClient = True

            if self.isRender:
                self._p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
            else:
                self._p = bullet_client.BulletClient()

            self.physicsClientId = self._p._client
            #TODO this disables gui, doe we want it
            self._p.configureDebugVisualizer(pybullet.COV_ENABLE_GUI,0)


        if self.scene is None:
            self.scene = self.create_single_player_scene(self._p)
        if not self.scene.multiplayer and self.ownsPhysicsClient:
            self.scene.episode_restart(self._p)

        self.robot.scene = self.scene

        self.frame = 0
        self.done = 0
        self.reward = 0
        dump = 0
        s = self.robot.reset(self._p)
        self.potential = self.robot.calc_potential()
        return s

    def _render(self, mode, close=False):
        if mode == "human":
            self.isRender = True
            # render HUD interface
            self.tk_root.update()
        if mode != "rgb_array":
            return np.array([])

        base_pos = [0,0,0]
        if hasattr(self,'robot'):
            if hasattr(self.robot,'body_xyz'):
                base_pos = self.robot.body_xyz

        view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=self._cam_dist,
            yaw=self._cam_yaw,
            pitch=self._cam_pitch,
            roll=0,
            upAxisIndex=2)
        proj_matrix = self._p.computeProjectionMatrixFOV(
            fov=60, aspect=float(self._render_width)/self._render_height,
            nearVal=0.1, farVal=100.0)
        (_, _, px, _, _) = self._p.getCameraImage(
            width=self._render_width, height=self._render_height, viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL
        )
        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def _close(self):
        if self.ownsPhysicsClient:
            if self.physicsClientId >= 0:
                self._p.disconnect()
        self.physicsClientId = -1

    def update_HUD(self, values):
        self.HUD.append_values(values)
        self.tk_root.update()

    # backwards compatibility for gym >= v0.9.x
    # for extension of this class.
    def step(self, *args, **kwargs):
        return self._step(*args, **kwargs)

    if parse_version(gym.__version__)>=parse_version('0.9.6'):
        close = _close
        render = _render
        reset = _reset
        seed = _seed


class Camera:
    def __init__(self):
        pass

    def move_and_look_at(self, i, j, k, x, y, z):
        lookat = [x, y, z]
        distance = 10
        yaw = 10
        self._p.resetDebugVisualizerCamera(distance, yaw, -20, lookat)


class HUD(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.parent = parent

    def init_lines(self, line_labels):
        self.npoints = 100
        self.lines = []
        self.parent.wm_title("HUD")
        canvas_height = 20
        window_height = (canvas_height + 2) * len(line_labels)
        self.parent.wm_geometry("{}x{}".format(self.npoints, window_height))
        h = self.parent.winfo_height()
        self.canvases = []
        for i in range(len(line_labels)):
            canvas = Canvas(self, height=canvas_height, background="white")
            self.canvases.append(canvas)
            canvas.bind("<Configure>", self.on_resize)
            self.lines.append([0 for x in range(self.npoints)])
            canvas.create_line((0, 0, 0, 0), tag="Line{}".format(i), fill='darkblue', width=1)
            #self.Line1 = [0 for x in range(self.npoints)]
            canvas.grid(sticky="news")
            self.grid_rowconfigure(0, weight=1)
            self.grid_columnconfigure(0, weight=1)
            self.grid(sticky="news")
        self.parent.grid_rowconfigure(0, weight=1)
        self.parent.grid_columnconfigure(0, weight=1)

    def on_resize(self, event):
        self.replot()

    def append_values(self, values):
        """
        Update the cached data lists with new values.
        """
        if len(values) != len(self.lines):
            print("Error values for HUD {} and number of lines {} do not match.".format(len(values), len(self.lines)))
        for i in range(len(values)):
            self.lines[i].append(values[i])
            self.lines[i] = self.lines[i][-1 * self.npoints:]

        #self.Line1.append(x)
        #self.Line1 = self.Line1[-1 * self.npoints:]
        self.replot()
        return

    def replot(self):
        """
        Update the canvas graph lines from the cached data lists.
        The lines are scaled to match the canvas size as the window may
        be resized by the user.
        """
        maxs = []
        coords = []
        max_all = 200.0
        for i in range(len(self.lines)):
            canvas = self.canvases[i]
            w = canvas.winfo_width()
            h = canvas.winfo_height() - 2
            maxs.append(max(self.lines[i]) + 1e-5)
            coords_i = []
            for n in range(0, self.npoints):
                x = (w * n) / self.npoints
                coords_i.append(x)
                #coords_i.append(h - ((h * (self.lines[i][n] + (h/2) )) / max_all))
                # first go to center, then add value scaled with max
                coords_i.append((h/2) + 1 - ((h/2) * (self.lines[i][n])))
            canvas.coords("Line{}".format(i), *coords_i)

        #max_X = max(self.Line1) + 1e-5

        #coordsX = []
        #for n in range(0, self.npoints):
        #    x = (w * n) / self.npoints
        #    coordsX.append(x)
        #    coordsX.append(h - ((h * (self.Line1[n]+100)) / max_all))
        #self.canvas.coords('X', *coordsX)