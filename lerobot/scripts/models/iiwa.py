import numpy as np
from roboticstoolbox.robot.Robot import Robot
from spatialmath import SE3


class KukaIIWA7(Robot):
    """
    Class that imports the KUKA IIWA 7 URDF model.

    ``KukaIIWA7()`` loads the robot model from a URDF file located at:
    /home/iasuser/GodsonInverseKinematics/kuka_iiwa/urdf/iiwa7.urdf
    """

    def __init__(self):
        urdf_path = "/home/iasuser/GodsonInverseKinematics/kuka_iiwa/urdf/iiwa7.urdf"
        # urdf_path = "/home/iasuser/GodsonInverseKinematics/hardware-interface-fm/panda_mujoco/kuka_iiwa_14/iiwa14_urdf_generated.urdf"

        links, name, urdf_string, urdf_filepath = self.URDF_read(urdf_path)

        self.xml_path = "/home/iasuser/GodsonInverseKinematics/kuka_iiwa/scene.xml"

        super().__init__(
            links,
            name=name,
            manufacturer="KUKA",
            urdf_string=urdf_string,
            urdf_filepath=urdf_filepath,
        )

        self.xml_path = "/home/iasuser/GodsonInverseKinematics/hardware-interface-fm/panda_mujoco/kuka_iiwa_14/scene.xml"

        # Set joint configurations
        dof = self.n
        self.qz = np.zeros(dof)
        self.qn = np.linspace(0.1, 0.5, dof)

        self.addconfiguration("qz", self.qz)
        self.addconfiguration("qn", self.qn)

        # self.qpos = [
        #     1.63, 0, 1.19, 1.49, 0, -1.72,
        #     0.244, 3.90000000e-01, -1.76430372e-17, 1.26473448e-02,
        #     1.00000000e+00, -1.24372044e-15, 4.29022815e-16, 2.55008890e-17,
        #     0.39, 0, 0.018, 0, 0, 0, 0, 0
        # ]
        self.qpos = np.zeros(22)
        # self.ctrl = [
        #     1.63, 0, 1.19, 1.49, 0, -1.72,
        #     0.244, 0
        # ]
        self.ctrl = [
            0.208, 0.628, -0.386, -1.32, -0.0593, 0.963, 0.2, 0
        ]


if __name__ == "__main__":  # pragma: no cover
    r = KukaIIWA7()
    print(r)

    print("\nDefined joint configuration (qz):", r.qz)
    print(dir(r))

    if hasattr(r, "grippers") and r.grippers:
        for link in r.grippers[0].links:
            print(link)