import gym, gym.spaces, gym.utils, gym.utils.seeding
import numpy as np
import pybullet
from pybullet_envs.bullet import bullet_client

from pkg_resources import parse_version

import threading
import time
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

        self.step_count = 0

        self.action_space = robot.action_space
        self.observation_space = robot.observation_space

        self.tk_root = None
        self.hud_active = False
        self.hud_lines = []

        if self.isRender:
            self.enable_HUD()

    def configure(self, args):
        self.robot.args = args

    def HUD(self,state, a ,done):
        pass #backwards compatibility

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
            # TODO this disables gui, do we want it
            self._p.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)

        if self.scene is None:
            self.scene = self.create_single_player_scene(self._p)
        if not self.scene.multiplayer and self.ownsPhysicsClient:
            self.scene.episode_restart(self._p)

        self.robot.scene = self.scene

        self.step_count = 0
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
        if mode != "rgb_array":
            return np.array([])

        base_pos = [0, 0, 0]
        if hasattr(self, 'robot'):
            if hasattr(self.robot, 'body_xyz'):
                base_pos = self.robot.body_xyz

        view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=self._cam_dist,
            yaw=self._cam_yaw,
            pitch=self._cam_pitch,
            roll=0,
            upAxisIndex=2)
        proj_matrix = self._p.computeProjectionMatrixFOV(
            fov=60, aspect=float(self._render_width) / self._render_height,
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

    def start_HUD(self):
        self.hud_active = True
        self.tk_root = Tk()
        self.hud = HUD(self.tk_root)
        self.hud.add_lines(self.hud_lines)
        while (True):
            self.hud.replot()
            self.tk_root.update()
            time.sleep(0.01)

    # backwards compatibility for gym >= v0.9.x
    # for extension of this class.
    def step(self, *args, **kwargs):
        return self._step(*args, **kwargs)

    if parse_version(gym.__version__) >= parse_version('0.9.6'):
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


class LineGraph():
    def __init__(self, label, n_points, min_value, max_value):
        self.label = label
        self.n_points = 200 #todo n_points
        self.values = [0] * self.n_points
        self.canvas = None
        self.min_value = min_value
        self.max_value = max_value
        self.value_range = self.max_value - self.min_value

    def append_value(self, value):
        """
        Update the cached data lists with new values.
        """
        self.values.append(value)
        self.values = self.values[-1 * self.n_points:]

    def replot(self):
        """
        Update the canvas graph line from the cached data lists.
        The line is scaled to match the canvas size as the window may
        be resized by the user.
        """
        # only plot if canvas was set and value is set
        if self.canvas and self.value_range:
            w = self.canvas.winfo_width()
            # we lose 2 pixels for the border
            h = self.canvas.winfo_height() - 3
            coords = []
            for i in range(0, self.n_points):
                x = (w * i) / self.n_points
                coords.append(x)
                # first go to center, then add value scaled with max
                #coords.append((h / 2) + 1 - ((h / 2 - 2) * (self.values[i])))

                # coords are from top to bottom. add +1 pixel to make sure that all pixels are visible
                coords.append(h + 1- ((h/self.value_range) * (self.values[i] - self.min_value)))
            #return coords
            self.canvas.coords("Line", *coords)

    def get_label(self):
        return self.label

    def get_n_points(self):
        return self.n_points

    def set_canvas(self, canvas):
        self.canvas = canvas

    def get_min_value(self):
        return self.min_value

    def get_max_value(self):
        return self.max_value

class HUD(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.parent = parent
        self.lines = []
        self.parent.wm_title("HUD")
        self.canvas_height = 20
        self.parent.resizable(width=False, height=False)
        self.grid(sticky="news")
        self.parent.grid_rowconfigure(0, weight=1)
        self.parent.grid_columnconfigure(0, weight=1)
        self.resizeWindow()

    def resizeWindow(self):
        window_height = (self.canvas_height + 2) * len(self.lines)
        longest_line = 0
        for line in self.lines:
            longest_line = max(longest_line, line.get_n_points())
        self.parent.wm_geometry("{}x{}".format(longest_line + 200, window_height))

    def add_line(self, line):
        self.lines.append(line)
        Label(self, text=line.get_label()).grid(row=len(self.lines), column=0)
        Label(self, text="{}:{}".format(line.get_min_value(), line.get_max_value())).grid(row=len(self.lines), column=1)
        canvas = Canvas(self, height=self.canvas_height, width=line.get_n_points(), background="white")
        canvas.create_line((0, 0, 0, 0), tag="Line", fill='darkblue', width=1)
        canvas.grid(row=len(self.lines), column=2)
        line.set_canvas(canvas)
        self.resizeWindow()

    def add_lines(self, lines):
        for line in lines:
            self.add_line(line)

    def on_resize(self, event):
        self.replot()

    def replot(self):
        i = 0
        for line in self.lines:
            line.replot()
            #line.canvas.coords("Line", *coords)
            i += 1