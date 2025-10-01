#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from kinematics import Kinematics
from scipy.spatial.transform import Rotation as R
from spatialmath import SE3
from scipy.spatial.transform import Rotation

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat
from contextlib import nullcontext

import os
import numpy as np
import mujoco
import cv2  # <-- used for blur

from lerobot.cameras import (  # noqa: F401
    CameraConfig,  # noqa: F401
)
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.policies.factory import make_policy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so100_follower,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
# Ensure Mujoco sim follower type is registered (import side-effect)
from lerobot.robots import so101_follower_mj  # noqa: F401

from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_so100_leader,
    homunculus,
    koch_leader,
    make_teleoperator_from_config,
    so100_leader,
    so101_leader,
)
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop
from lerobot.utils.control_utils import (
    init_keyboard_listener,
    is_headless,
    predict_action,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
    log_say,
)
from lerobot.utils.visualization_utils import _init_rerun, log_rerun_data

# -------------------------
# Project-root discovery and relative paths
# -------------------------

def _find_project_root(project_dir_name: str = "lerobot-icra26") -> Path:
    """
    Locate the project root directory. Priority:
    1) Environment variable LEROBOT_PROJECT_ROOT if set.
    2) The nearest ancestor directory named 'project_dir_name'.
    3) A parent directory that contains 'lerobot/src/lerobot'.
    4) Fallback to the directory two levels above this file.
    """
    env_root = os.environ.get("LEROBOT_PROJECT_ROOT", "").strip()
    if env_root:
        p = Path(env_root).resolve()
        if p.exists():
            return p

    here = Path(__file__).resolve()
    # Look for a parent directory named 'lerobot-icra26'
    for parent in [here] + list(here.parents):
        if parent.name == project_dir_name:
            return parent
        candidate = parent / project_dir_name
        if candidate.exists():
            return candidate

    # Look for a layout containing 'lerobot/src/lerobot'
    for parent in here.parents:
        if (parent / "lerobot" / "src" / "lerobot").exists():
            return parent

    # Fallback
    return here.parent.parent

PROJECT_ROOT = _find_project_root()

# Path to the MJCF scene relative to the discovered project root
MJCF_SCENE_PATH = PROJECT_ROOT / "lerobot" / "src" / "lerobot" / "robots" / "so101_follower_mj" / "trs_so_arm100" / "scene.xml"

# -------------------------
# Your Logger implementation (verbatim)
# -------------------------
import h5py


class Logger:
    def __init__(self, filename, buffer_size=1000, feature_names=None,
                 feature_dtype='f8', label_dtype='i', ts_dtype='f8'):
        self.filename = filename
        self.buffer_size = max(1, int(buffer_size))
        self.feature_names_init = feature_names
        self.feature_dtype = np.dtype(feature_dtype)
        self.label_dtype = np.dtype(label_dtype)
        self.ts_dtype = np.dtype(ts_dtype)
        self.file = h5py.File(filename, 'a', libver='latest')
        self._feature_names = None
        self.feature_count = None
        self.initialized = False
        self._buffer = {'timestamps': [], 'features': [], 'labels': []}
        self._prepare_datasets()

        self.features = 0
        self.timestamp = 0
        self.label = 0

    @property
    def feature_names(self):
        if self._feature_names is None and 'features' in self.file:
            names = self.file['features'].attrs.get('feature_names')
            if names is not None:
                self._feature_names = [
                    n.decode('utf-8') if isinstance(n, bytes) else str(n)
                    for n in names
                ]
        return self._feature_names

    def _prepare_datasets(self):
        if all(name in self.file for name in ['timestamps', 'features', 'labels']):
            ds_feat = self.file['features']
            self.feature_count = ds_feat.shape[1]
            self._feature_names = self.feature_names
            self.initialized = True

    def _init_datasets(self, features_example):
        self.feature_count = len(features_example)
        if self.feature_names_init and len(self.feature_names_init) == self.feature_count:
            names = self.feature_names_init
        else:
            names = [f'feature_{i}' for i in range(self.feature_count)]
            print("WARNING: feature do not fit")
        self._feature_names = names

        ts_chunk = (min(self.buffer_size * 5, 16384),)
        ft_chunk = (min(self.buffer_size * 5, max(1, 1024 * 1024 // (self.feature_count * self.feature_dtype.itemsize))),
                    self.feature_count)
        lb_chunk = (min(self.buffer_size * 5, 16384),)

        self.file.create_dataset(
            'timestamps', shape=(0,), maxshape=(None,),
            dtype=self.ts_dtype, chunks=ts_chunk, compression='gzip'
        )
        ds_feat = self.file.create_dataset(
            'features', shape=(0, self.feature_count), maxshape=(None, self.feature_count),
            dtype=self.feature_dtype, chunks=ft_chunk, compression='gzip'
        )
        ds_feat.attrs['feature_names'] = [n.encode('utf-8') for n in names]
        self.file.create_dataset(
            'labels', shape=(0,), maxshape=(None,),
            dtype=self.label_dtype, chunks=lb_chunk, compression='gzip'
        )
        self.initialized = True

    def log(self):
        if self.file is None:
            raise ValueError("File is closed.")
        features_array = np.asarray(self.features, dtype=self.feature_dtype)
        if not self.initialized:
            self._init_datasets(features_array)

        if features_array.shape != (self.feature_count,):
            raise ValueError(f"Features must have shape ({self.feature_count},) but got {features_array.shape}.")

        self._buffer['timestamps'].append(self.ts_dtype.type(self.timestamp))
        self._buffer['features'].append(features_array)
        self._buffer['labels'].append(self.label_dtype.type(self.label))

        if len(self._buffer['timestamps']) >= self.buffer_size:
            self.flush()

    def flush(self):
        if self.file is None or not self.initialized or not self._buffer['timestamps']:
            return

        start = self.file['timestamps'].shape[0]
        n = len(self._buffer['timestamps'])
        new_size = start + n

        self.file['timestamps'].resize(new_size, axis=0)
        self.file['features'].resize(new_size, axis=0)
        self.file['labels'].resize(new_size, axis=0)

        self.file['timestamps'][start:new_size] = np.array(self._buffer['timestamps'], dtype=self.ts_dtype)
        self.file['features'][start:new_size, :] = np.stack(self._buffer['features'], axis=0)
        self.file['labels'][start:new_size] = np.array(self._buffer['labels'], dtype=self.label_dtype)

        self._buffer = {'timestamps': [], 'features': [], 'labels': []}
        self.file.flush()

    def close(self):
        if self.file:
            self.flush()
            self.file.close()
            self.file = None
            self.initialized = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# -------------------------
# MuJoCo cube randomization (re-used)
# -------------------------

def randomize_cube_xy_in_sim(
    robot: Robot,
    body_name: str = "cube_body",
    x_range: tuple[float, float] = (0.18, 0.30),
    y_range: tuple[float, float] = (-0.15, 0.15),
    z: float = 0.015,
    settle_steps: int = 5,
    avoid_target_xy: tuple[float, float] | None = None,
    min_dist_from_target: float = 0.0,
    max_tries: int = 100,
) -> tuple[float, float, float] | None:
    model = getattr(robot, "model", None)
    data = getattr(robot, "data", None)
    if model is None or data is None:
        logging.warning("Randomize skipped: robot model/data not available (robot type: %s).", getattr(robot, "name", type(robot).__name__))
        return None

    b_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if b_id < 0:
        logging.warning("Randomize skipped: body '%s' not found in model.", body_name)
        return None

    j_adr = int(model.body_jntadr[b_id])
    j_num = int(model.body_jntnum[b_id])
    if j_num < 1:
        logging.warning("Randomize skipped: body '%s' has no joints (expected a free joint).", body_name)
        return None

    j_id = j_adr
    j_type = int(model.jnt_type[j_id])
    if j_type != int(mujoco.mjtJoint.mjJNT_FREE):
        logging.warning("Randomize skipped: first joint of body '%s' is not FREE.", body_name)
        return None

    x = y = None
    need_avoid = avoid_target_xy is not None and min_dist_from_target > 0.0
    min_d2 = min_dist_from_target * min_dist_from_target
    for _ in range(max_tries):
        x_try = float(np.random.uniform(*x_range))
        y_try = float(np.random.uniform(*y_range))
        if need_avoid:
            dx = x_try - float(avoid_target_xy[0])
            dy = y_try - float(avoid_target_xy[1])
            if (dx * dx + dy * dy) < min_d2:
                continue
        x, y = x_try, y_try
        break

    if x is None or y is None:
        logging.warning(
            "Randomize skipped: could not sample a position >= %.3f m away from target %s within %d tries.",
            min_dist_from_target, avoid_target_xy, max_tries
        )
        return None

    qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
    qadr = int(model.jnt_qposadr[j_id])
    data.qpos[qadr : qadr + 7] = [x, y, float(z), qw, qx, qy, qz]

    mujoco.mj_forward(model, data)
    logging.info("Randomized cube '%s' to (x=%.3f, y=%.3f, z=%.3f).", body_name, x, y, z)
    return (x, y, float(z))


# -------------------------
# Helpers to access joint state by NAME (explicit, no looping)
# -------------------------

def _require_joint_id(model, name: str) -> int:
    j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if j_id < 0:
        raise ValueError(f"Joint named '{name}' not found in model.")
    return j_id

def read_hinge_pos_vel_by_names(model, data, joint_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    pos = []
    vel = []
    for nm in joint_names:
        j_id = _require_joint_id(model, nm)
        qadr = int(model.jnt_qposadr[j_id])
        dadr = int(model.jnt_dofadr[j_id])
        pos.append(float(data.qpos[qadr]))
        vel.append(float(data.qvel[dadr]))
    return np.asarray(pos, dtype=np.float64), np.asarray(vel, dtype=np.float64)

def read_gripper_state_from_jaw(model, data, jaw_joint_name: str = "Jaw") -> float:
    j_id = _require_joint_id(model, jaw_joint_name)
    qadr = int(model.jnt_qposadr[j_id])
    qpos = float(data.qpos[qadr])

    if hasattr(model, "jnt_range"):
        rmin = float(model.jnt_range[j_id, 0])
        rmax = float(model.jnt_range[j_id, 1])
        if rmax > rmin:
            val = (qpos - rmin) / (rmax - rmin)
            val = float(np.clip(val, 0.0, 1.0))
        else:
            val = 0.0
    else:
        val = float(np.clip(qpos, 0.0, 1.0))

    invert = False
    if invert:
        val = 1.0 - val
    return min(val * 2, 1.0)

def read_free_body_pose7(model, data, body_name: str = "cube_body") -> np.ndarray | None:
    b_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if b_id < 0:
        return None
    j_adr = int(model.body_jntadr[b_id])
    j_num = int(model.body_jntnum[b_id])
    if j_num < 1:
        return None
    j_id = j_adr
    if int(model.jnt_type[j_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
        return None
    qadr = int(model.jnt_qposadr[j_id])
    x, y, z = data.qpos[qadr : qadr + 3]
    qw, qx, qy, qz = data.qpos[qadr + 3 : qadr + 7]
    return np.array([x, y, z, qx, qy, qz, qw], dtype=np.float64)

# Use the project-relative MJCF scene path for kinematics
xml_path = str(MJCF_SCENE_PATH)
kine = Kinematics(path=xml_path)

def fake_eef_from_joints_5(joint_pos_first5: np.ndarray) -> np.ndarray:
    return kine.forward_kinematics(joint_pos_first5)


def build_logger_features_vector(
    model,
    data,
    joint_names_order: list[str],
    cube_pose7: np.ndarray | None,
) -> np.ndarray:
    joint_pos, joint_vel = read_hinge_pos_vel_by_names(model, data, joint_names_order)
    eef7 = fake_eef_from_joints_5(joint_pos[:5])
    gripper_state = read_gripper_state_from_jaw(model, data, jaw_joint_name="Jaw")
    c7 = cube_pose7 if cube_pose7 is not None else np.zeros(7, dtype=np.float64)

    feats = np.concatenate([
        eef7,                 # 7
        joint_pos,            # 5
        joint_vel,            # 5
        np.array([gripper_state], dtype=np.float64),  # 1
        c7,                   # 7
    ], axis=0)
    return feats


# -------------------------
# Fault injection helpers
# -------------------------

@dataclass
class FaultEpisodeSpec:
    kind: str  # "noise", "blur", "brightness"
    strength: float  # in [0,1]
    # convenience triple for logging (only the chosen kind is nonzero)
    strengths_triple: tuple[float, float, float]  # (noise_s, blur_s, brightness_s)

def _choose_fault_for_episode(rng: np.random.Generator | None = None) -> FaultEpisodeSpec:
    rng = rng or np.random.default_rng()
    kind = rng.choice(["noise", "blur", "brightness"])
    s = float(rng.uniform(0.0, 1.0))
    triple = {
        "noise": (s, 0.0, 0.0),
        "blur": (0.0, s, 0.0),
        "brightness": (0.0, 0.0, s),
    }[kind]
    return FaultEpisodeSpec(kind=kind, strength=s, strengths_triple=triple)

def _is_rgb_image_like(x: np.ndarray) -> bool:
    if not isinstance(x, np.ndarray):
        return False
    if x.ndim == 3 and x.shape[2] == 3:
        return True
    if x.ndim == 3 and x.shape[0] == 3:
        return True
    return False

def _to_hwc(img: np.ndarray) -> tuple[np.ndarray, bool]:
    if img.ndim == 3 and img.shape[0] == 3 and img.shape[-1] != 3:
        # CHW -> HWC
        return np.transpose(img, (1, 2, 0)), True
    return img, False

def _from_hwc(img: np.ndarray, was_chw: bool) -> np.ndarray:
    return np.transpose(img, (2, 0, 1)) if was_chw else img

def _add_gaussian_noise(img: np.ndarray, s: float) -> np.ndarray:
    # s in [0,1], sigma scales with dtype range
    if img.dtype == np.uint8:
        sigma = s * 200
        noise = np.random.normal(0.0, sigma, img.shape).astype(np.float32)
        out = img.astype(np.float32) + noise
        return np.clip(out, 0, 255).astype(np.uint8)
    else:
        # assume float in [0,1]
        sigma = s * 0.2
        noise = np.random.normal(0.0, sigma, img.shape).astype(np.float32)
        out = img.astype(np.float32) + noise
        return np.clip(out, 0.0, 1.0).astype(img.dtype)

def _gaussian_blur(img: np.ndarray, s: float) -> np.ndarray:
    # s in [0,1], kernel up to 11, sigma up to ~3
    k = 1 + 2 * int(round(s * 10))  # 1,3,5,7,9,11
    k = max(1, k)
    if k == 1:
        return img
    sigma = s * 20.0
    return cv2.GaussianBlur(img, (k, k), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_DEFAULT)

def _adjust_brightness(img: np.ndarray, s: float) -> np.ndarray:
    # s in [0,1], factor in [0.0, 2.0] around 1.0
    max_delta = 1
    sign = -1.0 if np.random.rand() < 0.5 else 1.0
    factor = 1.0 + sign * s * max_delta
    if img.dtype == np.uint8:
        out = img.astype(np.float32) * factor
        return np.clip(out, 0, 255).astype(np.uint8)
    else:
        out = img.astype(np.float32) * factor
        return np.clip(out, 0.0, 1.0).astype(img.dtype)

def _apply_fault_to_image(img: np.ndarray, spec: FaultEpisodeSpec) -> np.ndarray:
    hwc, was_chw = _to_hwc(img)
    if spec.kind == "noise":
        hwc = _add_gaussian_noise(hwc, spec.strength)
    elif spec.kind == "blur":
        hwc = _gaussian_blur(hwc, spec.strength)
    elif spec.kind == "brightness":
        hwc = _adjust_brightness(hwc, spec.strength)
    else:
        pass
    return _from_hwc(hwc, was_chw)

def _apply_faults_to_observation_inplace(observation: dict, spec: FaultEpisodeSpec):
    # Traverse dict recursively and apply to all image-like arrays
    def recurse(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(v, dict):
                    recurse(v)
                elif isinstance(v, np.ndarray) and _is_rgb_image_like(v):
                    node[k] = _apply_fault_to_image(v, spec)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                if isinstance(v, dict):
                    recurse(v)
                elif isinstance(v, np.ndarray) and _is_rgb_image_like(v):
                    node[i] = _apply_fault_to_image(v, spec)

    recurse(observation)

def build_logger_features_vector_with_fault(
    model,
    data,
    joint_names_order: list[str],
    cube_pose7: np.ndarray | None,
    fault_strengths_triple: tuple[float, float, float],
) -> np.ndarray:
    base = build_logger_features_vector(model, data, joint_names_order, cube_pose7)
    f = np.array(list(fault_strengths_triple), dtype=np.float64)
    return np.concatenate([base, f], axis=0)


@dataclass
class InferenceDatasetConfig:
    repo_id: str
    single_task: str
    root: str | Path | None = None
    fps: int = 30
    episode_time_s: int | float = 20
    reset_time_s: int | float = 5
    num_episodes: int = 10
    video: bool = True
    push_to_hub: bool = False
    private: bool = False
    tags: list[str] | None = None
    num_image_writer_processes: int = 0
    num_image_writer_threads_per_camera: int = 4
    video_encoding_batch_size: int = 1
    # NEW: first n episodes recorded without any image faults
    n_normal_episodes: int = 0

    def __post_init__(self):
        if self.single_task is None:
            raise ValueError("You need to provide a task in `single_task`.")


@dataclass
class UserLoggerConfig:
    enabled: bool = True
    filename: str = "lerobot_mj0_faults.h5"
    buffer_size: int = 1_000_000


@dataclass
class InferenceConfig:
    robot: RobotConfig
    dataset: InferenceDatasetConfig
    teleop: TeleoperatorConfig | None = None
    policy: PreTrainedConfig | None = None
    display_data: bool = False
    play_sounds: bool = True
    resume: bool = False
    logger: UserLoggerConfig = UserLoggerConfig()

    def __post_init__(self):
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

        if self.teleop is None and self.policy is None:
            raise ValueError("Choose a policy, a teleoperator or both to control the robot")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


@safe_stop_image_writer
def inference_loop_with_fault(
    robot: Robot,
    events: dict,
    fps: int,
    dataset: LeRobotDataset | None = None,
    teleop: Teleoperator | list[Teleoperator] | None = None,
    policy: PreTrainedPolicy | None = None,
    control_time_s: int | None = None,
    single_task: str | None = None,
    display_data: bool = False,
    logger_obj: Logger | None = None,
    label_value: int = -1,
    joint_names_order: list[str] | None = None,
    fault_spec: FaultEpisodeSpec | None = None,
):
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")

    teleop_arm = teleop_keyboard = None
    if isinstance(teleop, list):
        teleop_keyboard = next((t for t in teleop if isinstance(t, KeyboardTeleop)), None)
        teleop_arm = next(
            (
                t
                for t in teleop
                if isinstance(
                    t,
                    (
                        so100_leader.SO100Leader,
                        so101_leader.SO101Leader,
                        koch_leader.KochLeader,
                    ),
                )
            ),
            None,
        )

        if not (teleop_arm and teleop_keyboard and len(teleop) == 2 and robot.name == "lekiwi_client"):
            raise ValueError(
                "For multi-teleop, the list must contain exactly one KeyboardTeleop and one arm teleoperator. Currently only supported for LeKiwi robot."
            )

    if policy is not None:
        policy.reset()

    # Use simulation time for episode timing if available, otherwise fallback to wall time
    timestamp = 0.0
    model = getattr(robot, "model", None)
    data = getattr(robot, "data", None)
    if data is not None and hasattr(data, "time"):
        use_sim_time = True
        start_episode_t_sim = float(data.time)
        start_episode_t_wall = None
    else:
        use_sim_time = False
        start_episode_t_wall = time.perf_counter()
        start_episode_t_sim = None

    # normalize triple for logger
    fault_triple = (0.0, 0.0, 0.0) if fault_spec is None else fault_spec.strengths_triple

    while timestamp < control_time_s:
        start_loop_t_wall = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        observation = robot.get_observation()

        # Apply image faults before policy sees the observation
        if fault_spec is not None:
            _apply_faults_to_observation_inplace(observation, fault_spec)

        observation_frame = None
        if policy is not None or dataset is not None:
            if dataset is not None:
                observation_frame = build_dataset_frame(dataset.features, observation, prefix="observation")

        if policy is not None:
            action_values = predict_action(
                observation_frame,
                policy,
                get_safe_torch_device(policy.config.device),
                policy.config.use_amp,
                task=single_task,
                robot_type=robot.robot_type,
            )
            action = {key: action_values[i].item() for i, key in enumerate(robot.action_features)}
        elif policy is None and isinstance(teleop, Teleoperator):
            action = teleop.get_action()
        elif policy is None and isinstance(teleop, list):
            arm_action = teleop_arm.get_action()
            arm_action = {f"arm_{k}": v for k, v in arm_action.items()}
            keyboard_action = teleop_keyboard.get_action()
            base_action = robot._from_keyboard_to_base_action(keyboard_action)
            action = {**arm_action, **base_action} if len(base_action) > 0 else arm_action
        else:
            logging.info(
                "No policy or teleoperator provided, skipping action generation."
                "This is likely to happen when resetting the environment without a teleop device."
            )
            dt_s_wall = time.perf_counter() - start_loop_t_wall
            #busy_wait(1 / fps - dt_s_wall)
            timestamp = time.perf_counter() - start_episode_t_wall
            continue

        sent_action = robot.send_action(action)

        if dataset is not None and False:
            action_frame = build_dataset_frame(dataset.features, sent_action, prefix="action")
            frame = {}
            if observation_frame is not None:
                frame.update(observation_frame)
            frame.update(action_frame)
            dataset.add_frame(frame, task=single_task)

        if display_data:
            log_rerun_data(observation, action)

        # Custom HDF5 logging at control rate
        if logger_obj is not None and model is not None and data is not None:
            sim_time = float(getattr(data, "time", time.perf_counter()))
            cube_pose7 = read_free_body_pose7(model, data, body_name="cube_body")
            feats = build_logger_features_vector_with_fault(
                model=model,
                data=data,
                joint_names_order=joint_names_order or [],
                cube_pose7=cube_pose7,
                fault_strengths_triple=fault_triple,
            )
            logger_obj.timestamp = sim_time
            logger_obj.features = feats
            logger_obj.label = int(label_value)  # -1 for reset, episode index for recording
            logger_obj.log()

        dt_s_wall = time.perf_counter() - start_loop_t_wall
        busy_wait(1 / fps - dt_s_wall)

        # Update elapsed time using simulation time if available
        if use_sim_time:
            timestamp = float(data.time) - start_episode_t_sim
        else:
            timestamp = time.perf_counter() - start_episode_t_wall


# -------------------------
# Reset helpers (no teleoperation)
# -------------------------

def _set_joint_positions_hard(model, data, joint_map: dict[str, float]):
    # Force-set qpos/vel each tick and zero controls to avoid actuators pulling away.
    for name, q in joint_map.items():
        j_id = _require_joint_id(model, name)
        qadr = int(model.jnt_qposadr[j_id])
        dadr = int(model.jnt_dofadr[j_id])
        data.qpos[qadr] = float(q)
        data.qvel[dadr] = 0.0
    # Best-effort: zero all controls so built-in servos don't fight the set pose.
    if hasattr(data, "ctrl") and data.ctrl is not None and data.ctrl.size > 0:
        data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)


def reset_robot_to_joint_positions_and_wait(
    robot: Robot,
    events: dict,
    fps: int,
    control_time_s: float,
    joint_names_order: list[str],
    joint_values_arm5_and_jaw: list[float],
    logger_obj: Logger | None = None,
):
    model = getattr(robot, "model", None)
    data = getattr(robot, "data", None)
    if model is None or data is None:
        logging.warning("reset_robot_to_joint_positions_and_wait: robot model/data not found; skipping reset.")
        # Just wait for the reset duration to keep flow consistent
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < control_time_s and not events["exit_early"]:
            busy_wait(1 / fps)
        events["exit_early"] = False
        return

    # Expected: 5 arm joints + 1 gripper (Jaw)
    if len(joint_values_arm5_and_jaw) != 6:
        raise ValueError("Expected 6 joint values: [Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw]")

    joint_map = {
        joint_names_order[0]: joint_values_arm5_and_jaw[0],  # Rotation
        joint_names_order[1]: joint_values_arm5_and_jaw[1],  # Pitch
        joint_names_order[2]: joint_values_arm5_and_jaw[2],  # Elbow
        joint_names_order[3]: joint_values_arm5_and_jaw[3],  # Wrist_Pitch
        joint_names_order[4]: joint_values_arm5_and_jaw[4],  # Wrist_Roll
        "Jaw": joint_values_arm5_and_jaw[5],                 # Jaw
    }

    # Log once the intended and current positions
    try:
        cur_pos, _ = read_hinge_pos_vel_by_names(model, data, joint_names_order)
        logging.info("Reset: current arm qpos=%s; target=%s", np.round(cur_pos, 4), np.round(joint_values_arm5_and_jaw[:5], 4))
    except Exception:
        pass

    # Use wall time for reset window and hard-set the joints at every tick.
    start_wall_t = time.perf_counter()
    fault_triple = (0.0, 0.0, 0.0)

    while (time.perf_counter() - start_wall_t) < control_time_s and not events["exit_early"]:
        loop_t0 = time.perf_counter()

        # Force the target pose every tick (prevents any actuator from moving away)
        _set_joint_positions_hard(model, data, joint_map)

        # Custom logging during reset (label = -1)
        if logger_obj is not None:
            cube_pose7 = read_free_body_pose7(model, data, body_name="cube_body")
            feats = build_logger_features_vector_with_fault(
                model=model,
                data=data,
                joint_names_order=joint_names_order,
                cube_pose7=cube_pose7,
                fault_strengths_triple=fault_triple,
            )
            # Use wall time for timestamp here (sim time won't advance without mj_step)
            logger_obj.timestamp = time.perf_counter()
            logger_obj.features = feats
            logger_obj.label = -1
            logger_obj.log()

        # maintain control rate
        busy_wait(1.0 / fps - (time.perf_counter() - loop_t0))

    # One final set to ensure pose before starting next episode
    _set_joint_positions_hard(model, data, joint_map)
    events["exit_early"] = False


@parser.wrap()
def run_inference(cfg: InferenceConfig) -> LeRobotDataset:
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        _init_rerun(session_name="inference_with_fault")

    robot = make_robot_from_config(cfg.robot)
    # No teleop: we won't create or connect any teleoperator

    # Create dataset to store observations/actions/videos if desired
    action_features = hw_to_dataset_features(robot.action_features, "action", cfg.dataset.video)
    obs_features = hw_to_dataset_features(robot.observation_features, "observation", cfg.dataset.video)
    dataset_features = {**action_features, **obs_features}

    if cfg.resume:
        dataset = LeRobotDataset(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
        )
        if hasattr(robot, "cameras") and len(robot.cameras) > 0:
            dataset.start_image_writer(
                num_processes=cfg.dataset.num_image_writer_processes,
                num_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
            )
        sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, dataset_features)
    else:
        sanity_check_dataset_name(cfg.dataset.repo_id, cfg.policy)
        dataset = LeRobotDataset.create(
            cfg.dataset.repo_id,
            cfg.dataset.fps,
            root=cfg.dataset.root,
            robot_type=robot.name,
            features=dataset_features,
            use_videos=cfg.dataset.video,
            image_writer_processes=cfg.dataset.num_image_writer_processes,
            image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
        )

    policy = None if cfg.policy is None else make_policy(cfg.policy, ds_meta=dataset.meta)

    robot.connect()

    listener, events = init_keyboard_listener()

    manager_ctx = VideoEncodingManager(dataset) if cfg.dataset.video else nullcontext()

    # Prepare custom logger if enabled
    my_logger = None
    if cfg.logger.enabled:
        feature_names = [
            "x","y","z","ox","oy","oz","ow",
            "joint_pos_1","joint_pos_2","joint_pos_3","joint_pos_4","joint_pos_5",
            "joint_vel_1","joint_vel_2","joint_vel_3","joint_vel_4","joint_vel_5",
            "gripper_state",
            "cx","cy","cz","cox","coy","coz","cow",
            # appended fault strengths
            "noise_strength","blur_strength","brightness_strength",
        ]
        my_logger = Logger(cfg.logger.filename, buffer_size=cfg.logger.buffer_size, feature_names=feature_names)
        logging.info("Custom logger initialized: %s", cfg.logger.filename)

    with manager_ctx:
        model = getattr(robot, "model", None)
        data = getattr(robot, "data", None)
        if model is None or data is None:
            logging.warning("Robot model/data not found; custom logger features will be zeros if enabled.")

        joint_names_order = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]
        if model is not None:
            for nm in joint_names_order + ["Jaw"]:
                _require_joint_id(model, nm)

        recorded_episodes = 0

        # Initial random placement before episode 0 (reset window)
        placed = randomize_cube_xy_in_sim(
            robot=robot,
            body_name="cube_body",
            x_range=(0.18, 0.30),
            y_range=(-0.15, 0.15),
            z=0.015,
            settle_steps=5,
            avoid_target_xy=(0.2, -0.15),
            min_dist_from_target=0.05,
            max_tries=100,
        )
        if placed is not None:
            logging.info("[Init] Cube randomized to x=%.3f, y=%.3f, z=%.3f.", *placed)
        else:
            logging.warning("[Init] Cube randomization failed.")

        # Reset joint targets (no teleop)
        reset_joint_values = [0.0036, -3.14, 3.09, 1.29, -1.69, -0.193]

        # Initial reset phase (label = -1), BEFORE episode 0, no faults applied
        if cfg.dataset.reset_time_s > 0:
            log_say("Initial reset: resetting robot to predefined joint configuration", cfg.play_sounds)
            reset_robot_to_joint_positions_and_wait(
                robot=robot,
                events=events,
                fps=cfg.dataset.fps,
                control_time_s=cfg.dataset.reset_time_s,
                joint_names_order=joint_names_order,
                joint_values_arm5_and_jaw=reset_joint_values,
                logger_obj=my_logger,
            )

        while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
            # Per-episode cube randomization
            placed = randomize_cube_xy_in_sim(
                robot=robot,
                body_name="cube_body",
                x_range=(0.18, 0.30),
                y_range=(-0.15, 0.15),
                z=0.015,
                settle_steps=5,
                avoid_target_xy=(0.2, -0.15),
                min_dist_from_target=0.05,
                max_tries=100,
            )
            if placed is not None:
                logging.info("Cube randomized to x=%.3f, y=%.3f, z=%.3f before episode %d.", placed[0], placed[1], placed[2], recorded_episodes)
            else:
                logging.warning("Cube randomization failed before episode %d.", recorded_episodes)

            # Choose this episode's fault (first n_normal_episodes run without faults)
            if recorded_episodes < cfg.dataset.n_normal_episodes:
                fault_spec = None
                logging.info("Episode %d fault: none (normal episode)", recorded_episodes)
                say_msg = f"Running inference episode {recorded_episodes} (no fault)"
            else:
                fault_spec = _choose_fault_for_episode()
                logging.info(
                    "Episode %d fault: kind=%s, strength=%.3f (noise=%.3f, blur=%.3f, brightness=%.3f)",
                    recorded_episodes, fault_spec.kind, fault_spec.strength,
                    fault_spec.strengths_triple[0], fault_spec.strengths_triple[1], fault_spec.strengths_triple[2],
                )
                say_msg = f"Running inference episode {recorded_episodes} (fault={fault_spec.kind}, s={fault_spec.strength:.2f})"

            log_say(say_msg, cfg.play_sounds)

            # Recording phase (label = episode index)
            inference_loop_with_fault(
                robot=robot,
                events=events,
                fps=cfg.dataset.fps,
                teleop=None,  # no teleop at all
                policy=policy,
                dataset=dataset,
                control_time_s=cfg.dataset.episode_time_s,
                single_task=cfg.dataset.single_task,
                display_data=cfg.display_data,
                logger_obj=my_logger,
                label_value=recorded_episodes,
                joint_names_order=joint_names_order,
                fault_spec=fault_spec,
            )

            # Reset phase after episode (label = -1), no faults
            should_do_reset_window = not events["stop_recording"] and (
                (recorded_episodes < cfg.dataset.num_episodes - 1) or events["rerecord_episode"]
            )
            if should_do_reset_window and cfg.dataset.reset_time_s > 0:
                log_say("Reset the environment (predefined joint configuration)", cfg.play_sounds)
                reset_robot_to_joint_positions_and_wait(
                    robot=robot,
                    events=events,
                    fps=cfg.dataset.fps,
                    control_time_s=cfg.dataset.reset_time_s,
                    joint_names_order=joint_names_order,
                    joint_values_arm5_and_jaw=reset_joint_values,
                    logger_obj=my_logger,
                )

            if events["rerecord_episode"]:
                log_say("Re-record episode", cfg.play_sounds)
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue

            #dataset.save_episode()
            recorded_episodes += 1

    log_say("Stop inference", cfg.play_sounds, blocking=True)

    if my_logger is not None:
        my_logger.close()
        logging.info("Custom logger closed: %s", cfg.logger.filename)

    robot.disconnect()

    if not is_headless() and listener is not None:
        listener.stop()

    if cfg.dataset.push_to_hub:
        dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)

    log_say("Exiting", cfg.play_sounds)
    return dataset


def main():
    run_inference()


if __name__ == "__main__":
    import sys
    # Example defaults; adjust paths/ports to your setup
    if len(sys.argv) == 1:
        sys.argv += [
            "--robot.type=so101_follower_mj",
            "--robot.id=my_sim_follower",
            "--robot.launch_viewer=true",
            "--robot.use_degrees=false",
            '--robot.cameras={"hand_eye": {"type": "mujoco", "name": "handeye", "width": 640, "height": 480, "fps": 30}}',
            "--display_data=true",
            "--dataset.single_task=Grab the blue cube and place it on the red target",
            "--dataset.push_to_hub=false",
            "--dataset.fps=30",
            "--dataset.episode_time_s=10",
            "--dataset.reset_time_s=2",
            "--dataset.num_episodes=200",
            "--dataset.n_normal_episodes=100",
            "--logger.enabled=true",
            "--logger.buffer_size=1000",
            "--policy.path=act_mj2/checkpoints/100000/pretrained_model",
            "--dataset.repo_id=org_or_project/eval_act_mj2_faults",
            "--logger.filename=eval_act_mj2_faults.h5",
        ]
    main()