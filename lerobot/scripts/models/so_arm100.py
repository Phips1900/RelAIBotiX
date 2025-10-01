import numpy as np
from roboticstoolbox.robot.Robot import Robot
from spatialmath import SE3


class SOArm(Robot):
    """
    Class that imports the SO_ARM100 URDF model

    ``SOArm100()`` loads the robot model from a URDF file located at:

    It uses the ERobot class, which supports URDF, meshes, and more modern robot features.
    """

    def __init__(self):

        links, name, urdf_string, urdf_filepath = self.URDF_read(
            "/home/rl/Desktop/lerobot-icra26/scripts/so_arm100.urdf"
            # "/home/iasuser/GodsonInverseKinematics/hardware-interface-fm/panda_mujoco/trs_so_arm100/Generated_urdf_sr100/so_arm100_no_rotation.urdf"
        )

        self.xml_path = "/home/iasuser/GodsonInverseKinematics/hardware-interface-fm/panda_mujoco/trs_so_arm100/scene.xml"

        super().__init__(
            links,
            name=name,
            manufacturer="SO_ARM100",
            urdf_string=urdf_string,
            urdf_filepath=urdf_filepath,
        )

        # Optional: Define common joint configurations (adjust based on DOF)
        dof = self.n
        self.qz = np.zeros(dof)
        self.qn = np.linspace(0.1, 0.5, dof)  # Example nominal config
        print(dof)

        # self.grippers[0].tool = SE3(0, 0, 0.1034)
        self.gripper_max_speed = 0.5
        self.gripper_range = 0.5

        self.addconfiguration("qz", self.qz)
        self.addconfiguration("qn", self.qn)

        # Initial joint positions (qpos) and control values (ctrl)
        # self.qpos = [
        #     1.7, -1.63, 1.51, 1.63,
        #     1.51, 1.3, 4.97794548e-08,
        #     3.90000000e-01, -1.76430372e-17, 1.26473448e-02,
        #     1.00000000e+00, -1.24372044e-15, 4.29022815e-16
        # ]
        self.qpos = np.zeros(13)
        self.ctrl = [
            0, -1.63, 1.51, 1.63,
            1.51, 0
        ]
        # self.ctrl = [
        #     0,0,0,0,0,0
        # ]

        # Optional: Set tool frame or gripper link if applicable
        # self.tool = SE3(0, 0, 0.1)  # Adjust this if the robot has a defined tool frame

        self.t_offset = SE3()
        self.o_offset = SE3.RPY([-3.1415/2, -0.0, 3.1415/2])  # For so_arm100.urdf
        # self.o_offset = SE3.RPY([-3.08240417, 0.01940791, -1.50855497])  # For so_arm100_no_rotation.urdf
        # self.o_offset = SE3.RPY([3.09514304, 0.00290751, 1.761197]) # For so101_new_calib.urdf


if __name__ == "__main__":  # pragma: no cover

    r = SOArm()
    print(r)

    print("\nDefined joint configuration (qz):", r.qz)

    # If there's a gripper chain or link (optional)
    if hasattr(r, "grippers") and r.grippers:
        for link in r.grippers[0].links:
            print(link)
