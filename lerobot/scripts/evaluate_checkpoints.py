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

import sys
import subprocess
from pathlib import Path
from typing import Optional


# ------------- CONFIG YOU MAY WANT TO TWEAK -------------
# Path to the inference script (this file assumes both are in the same dir)
INFERENCE_SCRIPT = Path(__file__).with_name("inference_with_fault.py")

# Base directory where checkpoints live (folders with 6-digit or 7-digit step names)
# Use a relative or neutral path for open source. For example: "checkpoints"
BASE_CHECKPOINT_DIR = Path("act_mj2/checkpoints")  # TODO: set to your neutral checkpoints path

# Subdirectory under each step folder that contains the pretrained model
PRETRAINED_SUBDIR = "pretrained_model"

# Dataset repo ID prefix; we will append the step in k
# Example for Hugging Face: "org_or_project/eval_vqbet_mj2_faults"
DATASET_REPO_PREFIX = "org_or_project/eval_act_mj2_faults"  # TODO: replace with neutral org/project

# Name prefix for logger files; we append the step in k, e.g., "eval_vqbet_mj2_faults60.h5"
LOGGER_FILE_PREFIX = "eval_act_mj2_faults"

# Steps (in k) to evaluate
STEPS_K = [100]

# Evaluation settings
NUM_EPISODES = 20
FPS = 240
EPISODE_TIME_S = 6
RESET_TIME_S = 2
N_NORMAL_EPISODES = 20  # 0 means faults are applied in all episodes
DISPLAY_DATA = False
LAUNCH_VIEWER = True

# Camera config (as JSON string for the CLI)
CAMERA_JSON = '{"hand_eye": {"type": "mujoco", "name": "handeye", "width": 640, "height": 480, "fps": 30}}'

# Single task name
SINGLE_TASK = "Grab the blue cube and place it on the red target"  # TODO: change if task text might identify your project

# Whether to skip a checkpoint run if the H5 logger already exists
SKIP_IF_LOG_EXISTS = False
# --------------------------------------------------------


def stepk_to_padded_steps(step_k: int) -> str:
    """
    Convert a step in thousands to a 7-digit zero-padded absolute step count.
    Example: 60 -> "0060000", 800 -> "0800000".
    Note: kept for compatibility, not used for lookup anymore.
    """
    abs_steps = step_k * 1000
    return f"{abs_steps:07d}"


def find_step_dir(base_dir: Path, step_k: int) -> Optional[Path]:
    """
    Find a checkpoint directory for the given step_k, trying multiple naming conventions:
    - 7-digit zero-padded absolute steps (e.g., 0060000, 0100000, 0800000)
    - 6-digit zero-padded absolute steps (e.g., 060000, 100000, 800000)
    - No padding (e.g., 60000, 100000, 800000)
    Returns the first existing path found, or None if not found.
    """
    abs_steps = step_k * 1000
    candidates = [
        f"{abs_steps:07d}",  # 7-digit
        f"{abs_steps:06d}",  # 6-digit
        str(abs_steps),      # no padding
    ]
    for name in candidates:
        path = base_dir / name
        if path.exists():
            return path
    return None


def main():
    if not INFERENCE_SCRIPT.exists():
        print(f"ERROR: inference script not found at {INFERENCE_SCRIPT}")
        sys.exit(1)

    for step_k in STEPS_K:
        step_dir = find_step_dir(BASE_CHECKPOINT_DIR, step_k)
        if step_dir is None:
            print(f"[{step_k}k] SKIP: no checkpoint dir found under {BASE_CHECKPOINT_DIR} for {step_k}k")
            continue

        policy_path = step_dir / PRETRAINED_SUBDIR
        if not policy_path.exists():
            print(f"[{step_k}k] SKIP: policy path does not exist: {policy_path}")
            continue

        dataset_repo_id = f"{DATASET_REPO_PREFIX}{step_k}"
        logger_filename = f"{LOGGER_FILE_PREFIX}{step_k}.h5"
        # Use a neutral, generic robot ID
        robot_id = f"sim_follower_{step_k}k"

        if SKIP_IF_LOG_EXISTS and Path(logger_filename).exists():
            print(f"[{step_k}k] SKIP: logger file already exists: {logger_filename}")
            continue

        print(f"\n=== Evaluating checkpoint {step_k}k ===")
        print(f"Checkpoint dir: {step_dir}")
        print(f"Policy path: {policy_path}")
        print(f"Dataset repo id: {dataset_repo_id}")
        print(f"Logger file: {logger_filename}\n")

        # Build CLI args for this run (use list form to avoid shell quoting issues)
        cmd = [
            sys.executable, str(INFERENCE_SCRIPT),
            f"--robot.type=so101_follower_mj",
            f"--robot.id={robot_id}",
            f"--robot.launch_viewer={'true' if LAUNCH_VIEWER else 'false'}",
            f"--robot.use_degrees=false",
            f"--robot.cameras={CAMERA_JSON}",
            f"--display_data={'true' if DISPLAY_DATA else 'false'}",
            f"--dataset.single_task={SINGLE_TASK}",
            f"--dataset.push_to_hub=false",
            f"--dataset.fps={FPS}",
            f"--dataset.episode_time_s={EPISODE_TIME_S}",
            f"--dataset.reset_time_s={RESET_TIME_S}",
            f"--dataset.num_episodes={NUM_EPISODES}",
            f"--dataset.n_normal_episodes={N_NORMAL_EPISODES}",
            f"--logger.enabled=true",
            f"--logger.buffer_size=1000",
            f"--policy.path={policy_path}",
            f"--dataset.repo_id={dataset_repo_id}",
            f"--logger.filename={logger_filename}",
        ]

        try:
            # Let the output stream directly to console for visibility
            res = subprocess.run(cmd, check=False)
            if res.returncode != 0:
                print(f"[{step_k}k] ERROR: process returned code {res.returncode}")
            else:
                print(f"[{step_k}k] DONE")
        except KeyboardInterrupt:
            print("\nInterrupted by user. Stopping.")
            break
        except Exception as e:
            print(f"[{step_k}k] ERROR: {e}")

    print("\nAll requested checkpoints processed.")


if __name__ == "__main__":
    main()