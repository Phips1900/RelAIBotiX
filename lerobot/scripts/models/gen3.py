#!/usr/bin/env python

import numpy as np
from roboticstoolbox.robot.Robot import Robot


class KinovaGen3(Robot):
    """
    Class that imports a KinovaGen3 URDF model

    ``KinovaGen3()`` is a class which imports a KinovaGen3 robot definition
    from a URDF file.  The model describes its kinematic and graphical
    characteristics.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.URDF.KinovaGen3()
        >>> print(robot)

    Defined joint configurations are:

    - qz, zero joint angle configuration, 'L' shaped configuration
    - qr, vertical 'READY' configuration
    - qs, arm is stretched out in the x-direction
    - qn, arm is at a nominal non-singular configuration

    """

    def __init__(self):

        links, name, urdf_string, urdf_filepath = self.URDF_read(
            "kortex_description/robots/gen3.xacro"
        )

        super().__init__(
            links,
            name=name,
            manufacturer="Kinova",
            urdf_string=urdf_string,
            urdf_filepath=urdf_filepath,
            # gripper_links=elinks[9]
        )

        self.xml_path = "/home/iasuser/GodsonInverseKinematics/hardware-interface-fm/panda_mujoco/kinova_gen3/scene.xml"

        # self.qdlim = np.array([
        # 2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100, 3.0, 3.0])

        self.qpos = [
            0, 0.63, -0.2, 1.08,
            0.02, 1.53, 0, 4.97794548e-08,
            1.61628711e-07, 3.90000000e-01, -1.76430372e-17, 1.26473448e-02,
            1.00000000e+00, -1.24372044e-15, 0, 0, 0, 0, 0, 0, 0, 0
        ]
        # self.qpos = np.zeros(22)
        # self.ctrl = np.zeros(8)
        self.ctrl = [
            0, 0.63, -0.2, 1.08,
            0.02, 1.53, 0, 0
        ]

        self.qr = np.array([np.pi, -0.3, 0, -1.6, 0, -1.0, np.pi / 2])
        self.qz = np.zeros(7)

        self.addconfiguration("qr", self.qr)
        self.addconfiguration("qz", self.qz)


if __name__ == "__main__":  # pragma nocover

    robot = KinovaGen3()
    print(robot)
