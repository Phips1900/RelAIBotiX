import mujoco
from scipy.spatial.transform import Rotation

from model import Model
from spatialmath import SE3
import roboticstoolbox as rtb
import numpy as np
from spatialmath.base import rpy2jac


class Kinematics(Model):
    def __init__(self, path):
        super().__init__(path)
        self.model_name = self.get_model_name()
        self.dof = self.get_dof()
        self.robot_model = self.get_model()
        self.model = mujoco.MjModel.from_xml_path(path)
        self.data = mujoco.MjData(self.model)
        print("Motion Controller model: ", self.robot_model)

    def inverse_kinematics_nr(self, *args, **kwargs):
        """
        Wrapper for the Newton Raphson inverse kinematics solver
        This method computes the joint configuration that achieves the target end-effector pose. It wraps 'ikine_NR()' from the RoboticsToolbox
        :param args:
        :param kwargs:
        :return:
        """
        sol = self.robot_model.ikine_NR(*args, **kwargs)
        return sol

    def inverse_kinematics_gn(self, *args, **kwargs):
        """
        Wrapper for the Newton Raphson inverse kinematics solver
        This method computes the joint configuration that achieves the target end-effector pose. It wraps 'ikine_GN()' from the RoboticsToolbox
        :param args:
        :param kwargs:
        :return:
        """
        sol = self.robot_model.ikine_GN(*args, **kwargs)
        return sol

    def inverse_kinematics_lm(self, *args, **kwargs):
        """
        Wrapper for the Levenberg-Marquadt inverse kinematics solver
        This method computes the joint configuration that achieves the target end-effector pose. It wraps 'ikine_LM()' from the RoboticsToolbox
        :param args:
        :param kwargs:
        :return:
        """
        sol = self.robot_model.ikine_LM(*args, **kwargs)
        return sol

    def inverse_kinematics_jacobian(self, T_target, q_curr, tol_pos, tol_ori, speed_factor, dt, max_joint_delta):
        T_curr, _ = self.forward_kinematics(q_curr)

        v = T_target.t - T_curr.t  # 3D translation error

        # Orientation error in RPY space
        rpy_goal = T_target.rpy(unit='rad')
        rpy_curr = T_curr.rpy(unit='rad')
        rpy_diff = (rpy_goal - rpy_curr + np.pi) % (2 * np.pi) - np.pi  # Euler angle difference

        # Exit condition
        if np.linalg.norm(v) < tol_pos and np.linalg.norm(rpy_diff) < tol_ori:
            return q_curr

        # Combine as RPY twist
        twist_rpy = np.concatenate((v, rpy_diff)) * speed_factor

        # Compute Jacobian and augment with RPY mapping
        Ai = rpy2jac(rpy_curr)  # Maps omega to rpy
        J0 = self.robot_model.jacob0(q_curr)
        Jaug = np.eye(6)
        Jaug[3:, 3:] = np.linalg.pinv(Ai)
        J_rpy = Jaug @ J0

        # Introduce damping factor to jacobian pseudoinverse to prevent erratic movement beyond workspace limit
        damp_factor = 0.01
        J_T = J_rpy.T

        # Compute condition number of the Jacobian
        cond = np.linalg.cond(J_rpy)
        if cond > 1e3:
            print(f"Joint Limit Exceeded: High Jacobian condition number: {cond:.2e}")

        # Damped Least Squares pseudoinverse
        qd = J_T @ np.linalg.inv(J_rpy @ J_T + damp_factor ** 2 * np.eye(6)) @ twist_rpy

        # Clip joint velocity using max_joint_delta
        max_dq = max_joint_delta / dt
        qd = np.clip(qd, -max_dq, max_dq)

        # Integrate joint velocity
        q_next = q_curr + qd * dt

        # Optional joint limit clamp
        if hasattr(self.robot_model, "qlim") and self.robot_model.qlim is not None:
            q_next = np.clip(q_next, self.robot_model.qlim[0], self.robot_model.qlim[1])

        return q_next

    def forward_kinematics(self, *joint_values):
        """
        Calculates the forward kinematics of the robot for the given join values.

        Accepts either individual joint values as separate arguments or a single iterable (list, tuple, numpy array) containing all joint values.
        The number of joint values must match the robot's degrees of freedom (DOF).

        :param joint_values: Joint angles of each robot joint. Either:
                - A list of joint values, e.g., forward_kinematics(0.1, 0.2, ..., 0.7)
                - A single iterable containing joint values, e.g. output from the rtb method fkine()
        :return:
                - A 7-element numpy array [x, y, z, qx, qy, qz, qw]
        """
        s = self.robot_model.fkine(joint_values)

        # Introduce offset based on the robot model
        t_offset = SE3()
        o_offset = SE3.RPY([-3.1415 / 2, -0.0, 3.1415 / 2])
        s = s * o_offset.inv() * t_offset

        # Extract position from the translation part (last column, first 3 rows)
        pos = s.A[:3, 3]

        # Extract rotation matrix (top-left 3x3)
        rot_matrix = s.A[:3, :3]

        # Convert rotation matrix to quaternion
        rot = Rotation.from_matrix(rot_matrix)
        quat = rot.as_quat()  # [qx, qy, qz, qw]

        return np.concatenate([pos, quat])


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Kinematics utility")
    parser.add_argument(
        "xml_path",
        type=str,
        help="Path to the MuJoCo XML model file"
    )
    parser.add_argument(
        "--q",
        nargs="+",
        type=float,
        help="Joint values to compute forward kinematics for"
    )
    args = parser.parse_args()

    kine = Kinematics(path=args.xml_path)

    if args.q is not None:
        print(kine.forward_kinematics(*args.q))
    else:
        print("No joint values provided. Use --q to pass joint values.")
        sys.exit(0)