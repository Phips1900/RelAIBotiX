#!/usr/bin/env python

import numpy as np
from roboticstoolbox.robot.Robot import Robot
from spatialmath import SE3


class Panda(Robot):
    """
    Class that imports a Panda URDF model

    ``Panda()`` is a class which imports a Franka-Emika Panda robot definition
    from a URDF file.  The model describes its kinematic and graphical
    characteristics.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.URDF.Panda()
        >>> print(robot)

    Defined joint configurations are:

    - qz, zero joint angle configuration, 'L' shaped configuration
    - qr, vertical 'READY' configuration
    - qs, arm is stretched out in the x-direction
    - qn, arm is at a nominal non-singular configuration

    """

    def __init__(self):

        links, name, urdf_string, urdf_filepath = self.URDF_read(
            "franka_description/robots/panda_arm_hand.urdf.xacro"
            # "/home/iasuser/GodsonInverseKinematics/hardware-interface-fm/panda_mujoco/franka_emika_panda/Generated_urdf_panda/panda_urdf_generated.urdf"
        )

        self.xml_path = "/home/iasuser/GodsonInverseKinematics/hardware-interface-fm/panda_mujoco/franka_emika_panda/demo_scene.xml"

        super().__init__(
            links,
            name=name,
            manufacturer="Franka Emika",
            gripper_links=links[9],
            urdf_string=urdf_string,
            urdf_filepath=urdf_filepath,
        )

        self.grippers[0].tool = SE3(0, 0, 0.1034)

        self.qdlim = np.array(
            [2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100, 3.0, 3.0]
        )

        self.qr = np.array([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])
        self.qz = np.zeros(7)

        self.addconfiguration("qr", self.qr)
        self.addconfiguration("qz", self.qz)

        self.qpos = [
            7.5e-02, -7.97688089e-01, -4.66516989e-02, -2.33454747e+00,
            -3.33191676e-02, 1.52918672e+00, 8.26202457e-01, 4.97794548e-08,
            1.61628711e-07, 3.90000000e-01, -1.76430372e-17, 1.26473448e-02,
            1.00000000e+00, -1.24372044e-15, 4.29022815e-16, 2.55008890e-17
        ]
        # self.qpos = np.zeros(16)
        self.ctrl = [
            0.075, -0.88, -0.04643195, -2.32839035, -0.03337264,
            1.53047642, 0.82637249, 0.0
        ]

        self.t_offset = SE3()
        self.o_offset = SE3()


if __name__ == "__main__":  # pragma nocover

    r = Panda()
    print(r)
    print(dir(r))

    r.qz

    for link in r.grippers[0].links:
        print(link)
