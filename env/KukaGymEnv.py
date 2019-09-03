# coding=utf-8

from env.kuka import Kuka
import random
import os
from gym import spaces
import time
import pybullet as p
from env import kuka
import numpy as np
import pybullet_data
import glob
from pkg_resources import parse_version
import gym
from gym.utils import seeding


class KukaDiverseObjectEnv(Kuka):
    """Class for Kuka environment with diverse objects.

    In each episode some objects are chosen from a set of 1000 diverse objects.
    These 1000 objects are split 90/10 into a train and test set.
    """

    def __init__(self,
                 urdfRoot=pybullet_data.getDataPath(),
                 actionRepeat=500,
                 isEnableSelfCollision=True,
                 renders=False,
                 isDiscrete=False,
                 maxSteps=8,
                 dv=0.06,
                 removeHeightHack=False,
                 blockRandom=0.3,
                 cameraRandom=0,
                 width=128,
                 height=128,
                 numObjects=5,
                 isTest=False):
        """Initializes the KukaDiverseObjectEnv.

        Args:
          urdfRoot: The diretory from which to load environment URDF's.
          actionRepeat: The number of simulation steps to apply for each action.
          isEnableSelfCollision: If true, enable self-collision.
          renders: If true, render the bullet GUI.
          isDiscrete: If true, the action space is discrete. If False, the
            action space is continuous.
          maxSteps: The maximum number of actions per episode.
          dv: The velocity along each dimension for each action.
          removeHeightHack: If false, there is a "height hack" where the gripper
            automatically moves down for each action. If true, the environment is
            harder and the policy chooses the height displacement.
          blockRandom: A float between 0 and 1 indicated block randomness. 0 is
            deterministic.
          cameraRandom: A float between 0 and 1 indicating camera placement
            randomness. 0 is deterministic.
          width: The image width.
          height: The observation image height.
          numObjects: The number of objects in the bin.
          isTest: If true, use the test set of objects. If false, use the train
            set of objects.
        """

        self._isDiscrete = isDiscrete
        self._timeStep = 1. / 240.
        self._urdfRoot = urdfRoot
        self._actionRepeat = actionRepeat
        self._isEnableSelfCollision = isEnableSelfCollision
        self._observation = []
        self._envStepCounter = 0
        self._renders = renders
        self._maxSteps = maxSteps
        self.terminated = 0
        self._cam_dist = 1.3
        self._cam_yaw = 180
        self._cam_pitch = -40
        self._dv = dv
        self._p = p
        self._removeHeightHack = removeHeightHack
        self._blockRandom = blockRandom
        self._cameraRandom = cameraRandom
        self._width = width
        self._height = height
        self._numObjects = numObjects
        self._isTest = isTest
        self.observation_space = spaces.Box(low=0, high=255, shape=(self._width, self._height, 3), dtype=np.uint32)

        if self._renders:
            self.cid = p.connect(p.SHARED_MEMORY)
            if (self.cid < 0):
                self.cid = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(1.3, 180, -41, [0.52, -0.2, -0.33])
        else:
            self.cid = p.connect(p.DIRECT)
        self.seed()

        if (self._isDiscrete):
            if self._removeHeightHack:
                self.action_space = spaces.Discrete(9)
            else:
                self.action_space = spaces.Discrete(7)
        else:
            self.action_space = spaces.Box(low=-1, high=1, shape=(3,))  # dx, dy, da
            if self._removeHeightHack:
                self.action_space = spaces.Box(low=-1,
                                               high=1,
                                               shape=(4,))  # dx, dy, dz, da

    def _reset(self):
        """Environment reset called at the beginning of an episode.
        """
        # Set the camera settings.
        look = [0.6, 0.4, 0.34]
        # look = [0.23, 0.2, 0.54]
        distance = 1.
        # pitch = -56 + self._cameraRandom * np.random.uniform(-3, 3)
        pitch = -50
        yaw = -185
        roll = 10
        # yaw = 245 + self._cameraRandom * np.random.uniform(-3, 3)
        # roll = 0
        self._view_matrix = p.computeViewMatrixFromYawPitchRoll(
            look, distance, yaw, pitch, roll, 2)
        fov = 20. + self._cameraRandom * np.random.uniform(-2, 2)
        aspect = self._width / self._height
        near = 0.01
        far = 10
        self._proj_matrix = p.computeProjectionMatrixFOV(
            fov, aspect, near, far)
        self.current_a = 0
        self._attempted_grasp = False
        self._env_step = 0
        self.terminated = 0
        self.out_of_range = 0
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=300)
        p.setTimeStep(self._timeStep)
        p.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"), [0, 0, -1])

        p.loadURDF(os.path.join(self._urdfRoot, "table/table.urdf"), 0.5000000, 0.00000, -.820000, 0.000000, 0.000000,
                   0.0, 1.0)

        p.setGravity(0, 0, -10)
        self._kuka = kuka.Kuka(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
        self._envStepCounter = 0
        p.stepSimulation()

        # Choose the objects in the bin.
        urdfList = self._get_random_object(
            self._numObjects, self._isTest)
        self._objectUids = self._randomly_place_objects(urdfList)
        self._observation = self._get_observation()
        return np.array(self._observation)

    def get_full_state(self):
        full_state = []

        for uid in self._objectUids:
            pos, ori = p.getBasePositionAndOrientation(uid)
            full_state.extend(pos)
        full_state.extend( self._kuka.getObservation())
        full_state = np.array(full_state).flatten()

        # full_state.shape = (15,)
        return full_state

    def demo_policy(self):
        fs = self.get_full_state()
        # 随便选一个物体
        fu_state = {'gripper.x': fs[9],
                    'gripper.y': fs[10],
                    'gripper.z': fs[11],
                    'object1.x': fs[0],
                    'object1.y': fs[1],
                    'object1.z': fs[2]}

        action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        # move to up of object
        dx = (fu_state['object1.x'] - fu_state['gripper.x'])
        dy = (fu_state['object1.y'] - fu_state['gripper.y'])
        dz = (fu_state['object1.z'] - fu_state['gripper.z']) + 0.01  # 比物体稍稍高一点点
        action[0] = np.clip(dx, -0.05, 0.05)
        action[1] = np.clip(dy, -0.05, 0.05)
        action[2] = np.clip(dz, -0.05, 0.05)
        action[3] = 0.0
        return action

    def _randomly_place_objects(self, urdfList):
        """Randomly places the objects in the bin.

        Args:
          urdfList: The list of urdf files to place in the bin.

        Returns:
          The list of object unique ID's.
        """

        # Randomize positions of each object urdf.
        objectUids = []
        for urdf_name in urdfList:
            xpos = 0.4 + self._blockRandom * random.random()
            ypos = self._blockRandom * (random.random() - .5)
            angle = np.pi / 2 + self._blockRandom * np.pi * random.random()
            orn = p.getQuaternionFromEuler([0, 0, angle])
            urdf_path = os.path.join(self._urdfRoot, urdf_name)
            uid = p.loadURDF(urdf_path, [xpos, ypos, .05],
                             [orn[0], orn[1], orn[2], orn[3]])

            objectUids.append(uid)
            # Let each object fall to the tray individual, to prevent object
            # intersection.
            for _ in range(500):
                p.stepSimulation()
        return objectUids

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_observation(self):
        """Return the observation as an image.
        """
        img_arr = p.getCameraImage(width=self._width,
                                   height=self._height,
                                   viewMatrix=self._view_matrix,
                                   projectionMatrix=self._proj_matrix)
        rgb = img_arr[2]
        np_img_arr = np.reshape(rgb, (self._height, self._width, 4))
        return np_img_arr[:, :, :3]

    def _step(self, action):
        """Environment step.

        Args:
          action: 5-vector parameterizing XYZ offset, vertical angle offset
          (radians), and grasp angle (radians).
        Returns:
          observation: Next observation.
          reward: Float of the per-step reward as a result of taking the action.
          done: Bool of whether or not the episode has ended.
          debug: Dictionary of extra information provided by environment.
        """
        dv = self._dv  # velocity per physics step. dv = 1
        if self._isDiscrete:
            # Static type assertion for integers.
            assert isinstance(action, int)
            if self._removeHeightHack:
                dx = [0, -dv, dv, 0, 0, 0, 0, 0, 0][action]
                dy = [0, 0, 0, -dv, dv, 0, 0, 0, 0][action]
                dz = [0, 0, 0, 0, 0, -dv, dv, 0, 0][action]
                da = [0, 0, 0, 0, 0, 0, 0, -0.25, 0.25][action]
            else:
                dx = [0, -dv, dv, 0, 0, 0, 0][action]
                dy = [0, 0, 0, -dv, dv, 0, 0][action]
                dz = -dv
                da = [0, 0, 0, 0, 0, -0.25, 0.25][action]
        else:
            act_scale = np.array([0.05, 0.05, 0.05, np.radians(90)])
            action = action * act_scale
            dx = dv * action[0]
            dy = dv * action[1]
            if self._removeHeightHack:
                dz = dv * action[2]
                da = action[3]
            else:
                dz = -dv
                da = action[2]

        return self._step_continuous([dx, dy, dz, da, 0.3])

    def _step_continuous(self, action):
        """Applies a continuous velocity-control action.

        Args:
          action: dx, dy, dz, da, finger_angel
          (radians), and grasp angle (radians).
        Returns:
          observation: Next observation.
          reward: Float of the per-step reward as a result of taking the action.
          done: Bool of whether or not the episode has ended.
          debug: Dictionary of extra information provided by environment.
        """
        # Perform commanded action.
        self._env_step += 1
        state = p.getLinkState(self._kuka.kukaUid, self._kuka.kukaEndEffectorIndex)
        current_EndEffectorPos = state[0]
        self._kuka.endEffectorPos[0] = current_EndEffectorPos[0] + action[0]

        self._kuka.endEffectorPos[1] = current_EndEffectorPos[1] + action[1]

        self._kuka.endEffectorPos[2] = current_EndEffectorPos[2] + action[2] + 0.02  # gravity offset

        self.current_a += action[3]  # angel
        for _ in range(self._actionRepeat):
            self._kuka.applyAction(self._kuka.endEffectorPos, self.current_a, 0.3)
            p.stepSimulation()
            if self._renders:
                # time.sleep(self._timeStep)
                pass
            if self._termination():
                break
        # If we are close to the bin, attempt grasp.
        state = p.getLinkState(self._kuka.kukaUid,
                               self._kuka.kukaEndEffectorIndex)
        end_effector_pos = state[0]
        if end_effector_pos[2] <= 0.1:
            finger_angle = 0.3
            for _ in range(500):
                # grasp_action
                self._kuka.applyAction(end_effector_pos, self.current_a, fingerAngle=finger_angle)
                p.stepSimulation()
                if self._renders:
                    time.sleep(self._timeStep)
                finger_angle -= 0.3 / 100.
                if finger_angle < 0:
                    finger_angle = 0
            # up the gripple a little and grasp
            end_effector_pos = np.array(end_effector_pos)
            for _ in range(400):
                end_effector_pos[2] = end_effector_pos[2] + 0.001

                self._kuka.applyAction(end_effector_pos, da=self.current_a, fingerAngle=finger_angle)
                p.stepSimulation()
                if self._renders:
                    time.sleep(self._timeStep)
                finger_angle -= 0.3 / 100.
                if finger_angle < 0:
                    finger_angle = 0

            self._attempted_grasp = True
        observation = self._get_observation()
        done = self._termination()
        reward = self._reward()
        debug = {
            'grasp_success': self._graspSuccess
        }
        return observation, reward, done, debug

    def _reward(self):
        """
        gripper out of range   -1
        Prone to object bonus  - dis_to_nearest_object
        grasp success          +2
        attempted_grasp_but_not_success 0
        """
        state = p.getLinkState(self._kuka.kukaUid,
                               self._kuka.kukaEndEffectorIndex)
        end_effector_pos = state[0]

        # out_of_range
        out_of_range = False
        if (end_effector_pos[0] < 0.537 - 0.25) or (end_effector_pos[0] > 0.537 + 0.25):
            out_of_range = True
        elif (end_effector_pos[1] < -0.25) or (end_effector_pos[1] > 0.25):
            out_of_range = True
        elif (end_effector_pos[2] < -0.) or (end_effector_pos[2] > 0.5 + 0.3):
            out_of_range = True
        if out_of_range:
            return -1

        self._graspSuccess = 0
        dis_list = []
        for uid in self._objectUids:
            pos, _ = p.getBasePositionAndOrientation(uid)
            dx = pos[0] - end_effector_pos[0]
            dy = pos[1] - end_effector_pos[1]
            dz = pos[2] - end_effector_pos[2]
            dis_ = np.sqrt( dx **2 + dy**2 + dz**2)
            dis_list.append( [uid, dis_]) #算一个距离 append一个距离

            # If any block is above height, return reward.
            if pos[2] > 0.05:
                self._graspSuccess += 1
                reward = 2
                return reward
        dis_list = np.array(dis_list)
        min_dis_from_EndEffector_to_object = np.min(dis_list[:, 1])
        prone_to_object_bonus = 0.2 - min_dis_from_EndEffector_to_object

        if self._attempted_grasp:
            reward = 0
        else:
            reward = prone_to_object_bonus

        return reward

    def _termination(self):
        """Terminates the episode if we have tried to grasp or if we are above
        maxSteps steps or gripper out of range.
        """
        return self._attempted_grasp or (self._env_step >= self._maxSteps) or self.out_of_range

    def _get_random_object(self, num_objects, test):
        """Randomly choose an object urdf from the random_urdfs directory.

        Args:
          num_objects:
            Number of graspable objects.

        Returns:
          A list of urdf filenames.
        """
        if test:
            urdf_pattern = os.path.join(self._urdfRoot, 'random_urdfs/*0/*.urdf')
        else:
            urdf_pattern = os.path.join(self._urdfRoot, 'random_urdfs/*[^0]/*.urdf')
        found_object_directories = glob.glob(urdf_pattern)
        total_num_objects = len(found_object_directories)
        selected_objects = np.random.choice(np.arange(total_num_objects),
                                            num_objects)
        selected_objects_filenames = []
        for object_index in selected_objects:
            selected_objects_filenames += [found_object_directories[object_index]]
        return selected_objects_filenames

    if parse_version(gym.__version__) >= parse_version('0.9.6'):
        reset = _reset

        step = _step