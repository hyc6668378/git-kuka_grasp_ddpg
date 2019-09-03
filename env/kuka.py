import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import pybullet as p
import numpy as np
import copy
import math
import pybullet_data


class Kuka:

    def __init__(self, urdfRootPath=pybullet_data.getDataPath(), timeStep=0.01):
        self.urdfRootPath = urdfRootPath
        self.timeStep = timeStep
        self.maxVelocity = 1.35
        self.maxForce = 600.
        self.fingerAForce = 2
        self.fingerBForce = 2.5
        self.fingerTipForce = 2
        self.useInverseKinematics = 1
        self.useSimulation = 1
        self.useNullSpace = 0
        self.useOrientation = 1
        self.kukaEndEffectorIndex = 6
        self.kukaGripperIndex = 7
        # lower limits for null space
        self.ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
        # upper limits for null space
        self.ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
        # joint ranges for null space
        self.jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]
        # restposes for null space
        self.rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
        # joint damping coefficents
        self.jd = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001]
        self.reset()

    def reset(self):
        objects = p.loadSDF(os.path.join(self.urdfRootPath, "kuka_iiwa/kuka_with_gripper2.sdf"))
        self.kukaUid = objects[0]
        # for i in range (p.getNumJoints(self.kukaUid)):
        #  print(p.getJointInfo(self.kukaUid,i))
        p.resetBasePositionAndOrientation(self.kukaUid, [-0.100000, 0.000000, 0.070000],
                                          [0.000000, 0.000000, 0.000000, 1.000000])
        # self.jointPositions = [0.006418, 0.413184, -0.011401, -1.589317, 0.005379, 1.137684, -0.006539, 0.000048,
        #                        -0.299912, 0.000000, -0.000043, 0.299960, 0.000000, -0.000200]
        self.jointPositions = [0.0052822753, 0.63178448, -0.009966143,
                               -1.7201607, 0.008282072, 0.7897110, -0.00860167,
                               0.000048, -0.299912, 0.000000, -0.000043, 0.299960, 0.000000, -0.000200
                               ]

        self.numJoints = p.getNumJoints(self.kukaUid)
        for jointIndex in range(self.numJoints):
            p.resetJointState(self.kukaUid, jointIndex, self.jointPositions[jointIndex])
            p.setJointMotorControl2(self.kukaUid, jointIndex, p.POSITION_CONTROL,
                                    targetPosition=self.jointPositions[jointIndex], force=self.maxForce)

        self.trayUid = p.loadURDF(os.path.join(self.urdfRootPath, "tray/tray.urdf"), 0.640000, 0.075000, -0.190000,
                                  0.000000, 0.000000, 1.000000, 0.000000)
        self.endEffectorPos = [0.537, 0.0, 0.5]
        self.endEffectorAngle = 0
        self.current_endEffectorAngle = 0

        self.motorNames = []
        self.motorIndices = []

        for i in range(self.numJoints):
            jointInfo = p.getJointInfo(self.kukaUid, i)
            qIndex = jointInfo[3]
            if qIndex > -1:
                # print("motorname")
                # print(jointInfo[1])
                self.motorNames.append(str(jointInfo[1]))
                self.motorIndices.append(i)

    def getActionDimension(self):
        if (self.useInverseKinematics):
            return len(self.motorIndices)
        return 6  # position x,y,z and roll/pitch/yaw euler angles of end effector

    def getObservationDimension(self):
        return len(self.getObservation())

    def getObservation(self, gr=True):
        observation = []
        if gr == True:
            state = p.getLinkState(self.kukaUid, self.kukaGripperIndex)
        else:
            state = p.getLinkState(self.kukaUid, self.kukaEndEffectorIndex)

        pos = state[0]
        orn = state[1]
        euler = p.getEulerFromQuaternion(orn)

        observation.extend(list(pos))
        observation.extend(list(euler))

        return observation

    def applyAction(self, pos, da, fingerAngle=0.3):

        # print ("self.numJoints")
        # print (self.numJoints)
        if (self.useInverseKinematics):
            self.current_endEffectorAngle = da
            self.current_endEffectorAngle = np.clip(self.current_endEffectorAngle, a_min=np.radians(-90),
                                                    a_max=np.radians(90))  # constrain in [-pi/2, pi/2]

            orn = p.getQuaternionFromEuler([0, -math.pi, 0])  # -math.pi,yaw])
            if (self.useNullSpace == 1):
                if (self.useOrientation == 1):
                    jointPoses = p.calculateInverseKinematics(self.kukaUid, self.kukaEndEffectorIndex, pos, orn,
                                                              self.ll, self.ul, self.jr, self.rp)
                else:
                    jointPoses = p.calculateInverseKinematics(self.kukaUid, self.kukaEndEffectorIndex, pos,
                                                              lowerLimits=self.ll, upperLimits=self.ul,
                                                              jointRanges=self.jr, restPoses=self.rp)
            else:
                if (self.useOrientation == 1):
                    jointPoses = p.calculateInverseKinematics(self.kukaUid, self.kukaEndEffectorIndex, pos, orn,
                                                              jointDamping=self.jd)
                else:
                    jointPoses = p.calculateInverseKinematics(self.kukaUid, self.kukaEndEffectorIndex, pos)

            if (self.useSimulation):
                for i in range(self.kukaEndEffectorIndex + 1):
                    p.setJointMotorControl2(bodyUniqueId=self.kukaUid, jointIndex=i, controlMode=p.POSITION_CONTROL,
                                            maxVelocity=self.maxVelocity, targetVelocity=0,
                                            targetPosition=jointPoses[i], force=self.maxForce, positionGain=0.03,
                                            velocityGain=1)
            else:
                # reset the joint state (ignoring all dynamics, not recommended to use during simulation)
                for i in range(self.kukaEndEffectorIndex + 1):  # self.numJoints
                    p.resetJointState(self.kukaUid, i, jointPoses[i])
            # fingers
            p.setJointMotorControl2(self.kukaUid, 7, p.POSITION_CONTROL, targetPosition=self.current_endEffectorAngle,
                                    force=self.maxForce)
            p.setJointMotorControl2(self.kukaUid, 8, p.POSITION_CONTROL, targetPosition=-fingerAngle,
                                    force=self.fingerAForce)
            p.setJointMotorControl2(self.kukaUid, 11, p.POSITION_CONTROL, targetPosition=fingerAngle,
                                    force=self.fingerBForce)

            p.setJointMotorControl2(self.kukaUid, 10, p.POSITION_CONTROL, targetPosition=0, force=self.fingerTipForce)
            p.setJointMotorControl2(self.kukaUid, 13, p.POSITION_CONTROL, targetPosition=0, force=self.fingerTipForce)

        else:
            for action in range(len(motorCommands)):
                motor = self.motorIndices[action]
                p.setJointMotorControl2(self.kukaUid, motor, p.POSITION_CONTROL, targetPosition=motorCommands[action],
                                        force=self.maxForce)

