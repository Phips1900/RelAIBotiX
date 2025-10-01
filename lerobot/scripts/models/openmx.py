import numpy as np
from roboticstoolbox.robot.Robot import Robot
from spatialmath import SE3
from math import pi


class OpenManipulatorX(Robot):
    """
    Class that imports the OpenMANIPULATOR-X URDF model

    ``OpenManipulatorX()`` loads the robot model from a URDF file located at:
    /home/iasuser/GodsonInverseKinematics/hardware-interface-fm/panda_mujoco/open_manipulator_x/omx_arm.urdf.xacro

    It uses the ERobot class, which supports URDF, meshes, and modern robot features.
    """

    def __init__(self):

        links, name, urdf_string, urdf_filepath = self.URDF_read(
            "/home/iasuser/GodsonInverseKinematics/hardware-interface-fm/panda_mujoco/open_manipulator-main/open_manipulator_description/urdf/omx/omx_arm.urdf.xacro"
        )

        self.xml_path = "/home/iasuser/GodsonInverseKinematics/hardware-interface-fm/panda_mujoco/open_manipulator_x/scene.xml"

        super().__init__(
            links,
            name=name.upper(),
            manufacturer="ROBOTIS",
            gripper_links=None,
            urdf_string=urdf_string,
            urdf_filepath=urdf_filepath,
        )

        # Define common configurations
        dof = self.n
        self.qz = np.zeros(dof)
        self.qn = np.linspace(0.1, 0.5, dof)

        self.addconfiguration("qz", self.qz)
        self.addconfiguration("qn", self.qn)

        # Initial joint positions and control values
        self.qpos = np.zeros(14)  # Update based on model's full state vector
        self.ctrl = np.zeros(dof)

        # Optional tool transformation offset (to match end-effector frame)
        self.tool = SE3.Tz(0.038)  # Update as per end-effector definition


if __name__ == "__main__":
    robot = OpenManipulatorX()
    print(robot)
    print(robot.ets())
