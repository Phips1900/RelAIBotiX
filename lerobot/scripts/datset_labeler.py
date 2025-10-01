#!/usr/bin/env python3
# label_mujoco_timeline.py
# Interactive MuJoCo trajectory labeler with live drag preview and object pose support.
#
# Defaults are set so you can run it without CLI args. Adjust constants below if needed.

import argparse
import math
import time
from typing import List, Tuple, Optional

import h5py
import numpy as np

import mujoco
import mujoco.viewer

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.widgets import SpanSelector

# =========================
# Defaults (edit here)
# =========================
DEFAULT_H5 = ""
DEFAULT_MODEL_XML = ""

DEFAULT_JOINT_COLS = ["joint_pos_1", "joint_pos_2", "joint_pos_3", "joint_pos_4", "joint_pos_5"]
DEFAULT_GRIPPER_COL = "gripper_state"
DEFAULT_OBJECT_POSE_COLS = ["cx", "cy", "cz", "cox", "coy", "coz", "cow"]

DEFAULT_JOINT_QPOS_IDX = [0, 1, 2, 3, 4]
DEFAULT_GRIPPER_QPOS_IDX = [5]  # single-DOF gripper

DEFAULT_FPS = 30.0

# Attempt to find cube in model automatically; you can override names here if needed.
OBJECT_FREEJOINT_NAME_CANDIDATES = ["cube_freejoint", "Cube_freejoint", "box_freejoint", "object_freejoint"]
OBJECT_MOCAP_BODY_NAME_CANDIDATES = ["cube", "Cube", "box", "Box", "object", "Object"]

# Class feature column name to write (stored as numeric values 0..4, -1 for unlabeled)
CLASS_LABEL_FEATURE = "class_label"

CLASS_NAMES = ["move", "pick", "carry", "place", "reset"]
# Colors: index 0 is for -1 (unlabeled), 1..5 correspond to classes 0..4
CLASS_COLORS = [
    "#bdbdbd",  # -1 unlabeled
    "#1f77b4",  # 0 move
    "#2ca02c",  # 1 pick
    "#ff7f0e",  # 2 carry
    "#d62728",  # 3 place
    "#9467bd",  # 4 reset
]


def load_h5(h5_path: str):
    with h5py.File(h5_path, "r") as f:
        timestamps = f["timestamps"][:]
        features = f["features"][:]
        labels = f["labels"][:]  # episode id or -1 between episodes
        raw_names = f["features"].attrs.get("feature_names", [])
        feature_names = [
            n.decode("utf-8") if isinstance(n, (bytes, bytearray)) else str(n)
            for n in raw_names
        ]
    return timestamps, features, labels, feature_names


def ensure_class_label_feature(h5_path: str) -> int:
    """
    Ensure 'class_label' exists as a feature column. Returns its column index.
    If it doesn't exist, append it initialized to -1.
    """
    with h5py.File(h5_path, "r+") as f:
        feats = f["features"]
        raw_names = feats.attrs.get("feature_names", [])
        names = [
            n.decode("utf-8") if isinstance(n, (bytes, bytearray)) else str(n)
            for n in raw_names
        ]
        if CLASS_LABEL_FEATURE in names:
            return names.index(CLASS_LABEL_FEATURE)

    # Add new column at the end, preserving dataset properties
    with h5py.File(h5_path, "r+") as f:
        feats = f["features"]
        N, F = feats.shape
        dtype = feats.dtype
        feats_new = f.create_dataset(
            "features_new",
            shape=(N, F + 1),
            dtype=dtype,
            chunks=feats.chunks,
            compression=feats.compression,
            compression_opts=feats.compression_opts,
            shuffle=feats.shuffle,
            fletcher32=feats.fletcher32,
        )
        chunk_rows = max(1, (feats.chunks[0] if feats.chunks else min(N, 100000)))
        for i0 in range(0, N, chunk_rows):
            i1 = min(N, i0 + chunk_rows)
            block = feats[i0:i1, :]
            feats_new[i0:i1, :F] = block
            feats_new[i0:i1, F] = np.asarray(-1, dtype=dtype)

        names = [
            n.decode("utf-8") if isinstance(n, (bytes, bytearray)) else str(n)
            for n in feats.attrs.get("feature_names", [])
        ] + [CLASS_LABEL_FEATURE]
        feats_new.attrs["feature_names"] = np.array(
            names, dtype=h5py.string_dtype(encoding="utf-8")
        )
        del f["features"]
        f.move("features_new", "features")
        return F


def extract_episodes(ep_ids: np.ndarray) -> List[Tuple[int, int, int]]:
    """
    Return list of (start_idx, end_idx_exclusive, episode_id) for contiguous regions with labels != -1.
    """
    episodes = []
    N = len(ep_ids)
    i = 0
    while i < N:
        if ep_ids[i] == -1:
            i += 1
            continue
        ep = int(ep_ids[i])
        start = i
        while i < N and ep_ids[i] == ep:
            i += 1
        end = i
        episodes.append((start, end, ep))
    return episodes


def _joint_id_from_qpos_index(model: mujoco.MjModel, qpos_index: int) -> Optional[int]:
    # Find joint whose qpos starts at qpos_index
    for j in range(model.njnt):
        if int(model.jnt_qposadr[j]) == qpos_index:
            return j
    return None


def _find_named_joint(model: mujoco.MjModel, name: str) -> Optional[int]:
    try:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    except mujoco.Error:
        return None


def _find_named_body(model: mujoco.MjModel, name: str) -> Optional[int]:
    try:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    except mujoco.Error:
        return None


def _normalize_quat_wxyz(q: np.ndarray) -> np.ndarray:
    # Normalize quaternion [w, x, y, z]; if tiny, fall back to identity
    n = float(np.linalg.norm(q))
    if n < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / n


class Labeler:
    def __init__(
        self,
        h5_path: str = DEFAULT_H5,
        model_xml: str = DEFAULT_MODEL_XML,
        joint_cols: List[str] = DEFAULT_JOINT_COLS,
        gripper_col: Optional[str] = DEFAULT_GRIPPER_COL,
        object_pose_cols: Optional[List[str]] = DEFAULT_OBJECT_POSE_COLS,
        joint_qpos_idx: List[int] = DEFAULT_JOINT_QPOS_IDX,
        gripper_qpos_idx: Optional[List[int]] = DEFAULT_GRIPPER_QPOS_IDX,
        fps: float = DEFAULT_FPS,
        invert_gripper: bool = False,
    ):
        self.h5_path = h5_path
        self.viewer = None  # will be set in run()

        # Ensure we have class_label column
        self.class_col_idx = ensure_class_label_feature(h5_path)

        # Load arrays into memory
        self.timestamps, self.features, self.ep_ids, self.feature_names = load_h5(h5_path)
        self.N, self.F = self.features.shape
        # Editable class array (int16 with -1 unlabeled)
        cls_vals = self.features[:, self.class_col_idx]
        self.class_arr = np.rint(cls_vals).astype(np.int16, copy=True)

        # Resolve feature indices
        name_to_idx = {n: i for i, n in enumerate(self.feature_names)}
        try:
            self.joint_cols_idx = [name_to_idx[name] for name in joint_cols]
        except KeyError as e:
            raise KeyError(f"Feature name not found: {e}. Available: {self.feature_names}")

        self.gripper_col_idx = name_to_idx[gripper_col] if gripper_col is not None else None

        self.object_pose_idx = None
        if object_pose_cols is not None:
            ok = all(n in name_to_idx for n in object_pose_cols)
            if ok:
                self.object_pose_idx = [name_to_idx[n] for n in object_pose_cols]
            else:
                missing = [n for n in object_pose_cols if n not in name_to_idx]
                print(f"[WARN] Object pose columns missing in features: {missing} — object pose will not be updated.")

        # Episodes to label (skip ep_ids == -1 spans)
        self.episodes = extract_episodes(self.ep_ids)
        if not self.episodes:
            raise ValueError("No episodes found (labels == -1 everywhere?).")

        # MuJoCo setup
        self.model = mujoco.MjModel.from_xml_path(model_xml)
        self.data = mujoco.MjData(self.model)

        # Arm mapping
        self.joint_qpos_idx = joint_qpos_idx
        if len(self.joint_qpos_idx) != len(self.joint_cols_idx):
            raise ValueError("Mismatch: joint_cols and joint_qpos_idx must have same length.")

        # Gripper mapping
        self.gripper_qpos_idx = gripper_qpos_idx or []
        self.invert_gripper = invert_gripper  # matches your logging function default False

        # Object binding (freejoint or mocap)
        self.obj_mode = None  # 'freejoint' or 'mocap'
        self.obj_qpos_adr = None
        self.obj_mocap_id = None
        self._bind_object()

        # UI state
        self.cur_ep = 0
        self.cur_global_idx = self.episodes[self.cur_ep][0]
        self.sel_local_range: Optional[Tuple[int, int]] = None
        self.play = False
        self.last_time = time.time()
        self.fps = fps
        self.history = []  # stack of (g0, g1, old_values)

        # Span dragging state
        self._prev_span_extents: Optional[Tuple[float, float]] = None

        # Timeline plot
        self.fig, self.ax = plt.subplots(figsize=(12, 2.8))
        self.fig.canvas.manager.set_window_title("Episode timeline (labeling)")

        # Colors and scale
        self.cmap = ListedColormap(CLASS_COLORS)
        boundaries = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        self.norm = BoundaryNorm(boundaries, self.cmap.N, clip=True)

        self.im = None
        self.cursor_line = None
        self.span = None
        self.ep_text = None  # on-plot text label for episode

        self._build_timeline_plot()
        self._connect_events()

    # ----- Object binding -----
    def _bind_object(self):
        if self.object_pose_idx is None:
            print("[INFO] No object pose features configured — skipping object pose binding.")
            return

        # Try freejoint by name candidates
        for nm in OBJECT_FREEJOINT_NAME_CANDIDATES:
            j_id = _find_named_joint(self.model, nm)
            if j_id is not None:
                if int(self.model.jnt_type[j_id]) == int(mujoco.mjtJoint.mjJNT_FREE):
                    self.obj_mode = "freejoint"
                    self.obj_qpos_adr = int(self.model.jnt_qposadr[j_id])
                    print(f"[INFO] Bound object pose to freejoint '{nm}' at qpos adr {self.obj_qpos_adr}.")
                    return
        # Try mocap body by name candidates
        for nm in OBJECT_MOCAP_BODY_NAME_CANDIDATES:
            b_id = _find_named_body(self.model, nm)
            if b_id is not None:
                mocap_id = int(self.model.body_mocapid[b_id])
                if mocap_id != -1:
                    self.obj_mode = "mocap"
                    self.obj_mocap_id = mocap_id
                    print(f"[INFO] Bound object pose to mocap body '{nm}' with mocap id {mocap_id}.")
                    return

        print("[WARN] Could not bind object pose (no matching freejoint or mocap body). The cube will not be updated.")

    # ----- Episode helpers -----
    def _episode_slice(self, ep_idx: int) -> Tuple[int, int, int]:
        return self.episodes[ep_idx]

    def _local_idx(self, global_idx: int) -> int:
        start, end, _ = self._episode_slice(self.cur_ep)
        return int(np.clip(global_idx - start, 0, end - start - 1))

    def _global_idx_from_local(self, local_idx: int) -> int:
        start, end, _ = self._episode_slice(self.cur_ep)
        return int(np.clip(start + local_idx, start, end - 1))

    # ----- Plot / UI -----
    def _build_timeline_plot(self):
        self.ax.clear()
        start, end, ep_id = self._episode_slice(self.cur_ep)
        L = end - start
        ep_classes = self.class_arr[start:end]

        self.im = self.ax.imshow(
            ep_classes[np.newaxis, :],
            aspect="auto",
            interpolation="nearest",
            cmap=self.cmap,
            norm=self.norm,
            extent=(0, L, 0, 1),
        )
        self.ax.set_yticks([])
        self.ax.set_xlim(0, L)
        self.ax.set_xlabel("Frames (within current episode)")
        self.ax.set_title("Drag to select (live preview). Keys: 0..4 label, backspace clear, left/right step, space play, n/p ep, k save, u undo")

        # Cursor line
        cur_local = self._local_idx(self.cur_global_idx)
        self.cursor_line = self.ax.axvline(cur_local, color="k", lw=1)

        # Episode text
        if self.ep_text is not None:
            self.ep_text.remove()
        self.ep_text = self.ax.text(
            0.01, 1.25, f"Labeling Episode {ep_id} (frames {start}..{end-1})",
            transform=self.ax.transAxes, fontsize=11, fontweight="bold", va="bottom"
        )

        # Span selector with live on-move callback
        if self.span is not None:
            self.span.disconnect_events()
        self.span = SpanSelector(
            self.ax,
            onselect=self._on_span_select,
            direction="horizontal",
            useblit=True,
            interactive=True,
            drag_from_anywhere=True,
            button=1,
            onmove_callback=self._on_span_move,  # live callback
        )
        self._prev_span_extents = None

        self.fig.canvas.draw_idle()

    def _connect_events(self):
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        start, end, _ = self._episode_slice(self.cur_ep)
        L = end - start
        local = int(np.clip(round(event.xdata), 0, L - 1))
        self.cur_global_idx = self._global_idx_from_local(local)
        self.sel_local_range = None
        self._refresh_cursor()
        self._render_current()

    def _on_span_move(self, vmin: float, vmax: float):
        # Live-update sim to the edge being dragged.
        start, end, _ = self._episode_slice(self.cur_ep)
        L = end - start
        if L <= 0:
            return

        # Decide which edge moved relative to last move
        if self._prev_span_extents is None:
            moved_edge = vmax
        else:
            pmin, pmax = self._prev_span_extents
            dmin = abs(vmin - pmin)
            dmax = abs(vmax - pmax)
            moved_edge = vmax if dmax >= dmin else vmin

        local = int(np.clip(round(moved_edge), 0, L - 1))
        self.cur_global_idx = self._global_idx_from_local(local)
        self._refresh_cursor()
        self._render_current()

        self._prev_span_extents = (vmin, vmax)

    def _on_span_select(self, xmin, xmax):
        start, end, _ = self._episode_slice(self.cur_ep)
        L = end - start
        i0 = int(np.clip(math.floor(min(xmin, xmax)), 0, L - 1))
        i1 = int(np.clip(math.ceil(max(xmin, xmax)), 0, L))
        if i1 <= i0:
            i1 = i0 + 1
        self.sel_local_range = (i0, i1)
        self._prev_span_extents = None
        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        key = event.key or ""

        # Label assignment
        if key in list("01234"):
            cls = int(key)
            self._apply_label(cls)
            return

        # Navigation
        step = 1
        if "shift" in key:
            step = 10
        if "ctrl" in key:
            step = 50

        if key.endswith("right"):
            self._step(step); return
        if key.endswith("left"):
            self._step(-step); return

        if key == " ":
            self.play = not self.play
            self.last_time = time.time()
            return

        if key == "n":
            self._change_episode(+1); return
        if key == "p":
            self._change_episode(-1); return

        if key == "k":  # save to H5 (avoid 's' which saves PNG in Matplotlib)
            self._save_to_h5(); return

        if key == "u":
            self._undo(); return

        if key == "backspace":
            self._apply_label(-1); return

    def _step(self, delta_local: int):
        start, end, _ = self._episode_slice(self.cur_ep)
        cur_local = self._local_idx(self.cur_global_idx)
        new_local = int(np.clip(cur_local + delta_local, 0, end - start - 1))
        self.cur_global_idx = self._global_idx_from_local(new_local)
        self.sel_local_range = None
        self._refresh_cursor()
        self._render_current()

    def _change_episode(self, delta: int):
        new_ep = int(np.clip(self.cur_ep + delta, 0, len(self.episodes) - 1))
        if new_ep == self.cur_ep:
            return
        self.cur_ep = new_ep
        start, end, _ = self._episode_slice(self.cur_ep)
        self.cur_global_idx = start
        self.sel_local_range = None
        self._build_timeline_plot()
        self._render_current()

    def _apply_label(self, cls: int):
        start, end, _ = self._episode_slice(self.cur_ep)
        if self.sel_local_range is not None:
            i0_local, i1_local = self.sel_local_range
        else:
            cur_local = self._local_idx(self.cur_global_idx)
            i0_local, i1_local = cur_local, cur_local + 1

        g0 = self._global_idx_from_local(i0_local)
        g1 = self._global_idx_from_local(i1_local - 1) + 1

        # Confine to current episode bounds
        g0 = max(g0, start)
        g1 = min(g1, end)
        if g1 <= g0:
            return

        old_vals = self.class_arr[g0:g1].copy()
        self.history.append((g0, g1, old_vals))
        self.class_arr[g0:g1] = cls

        # Update timeline stripe
        ep_classes = self.class_arr[start:end]
        self.im.set_data(ep_classes[np.newaxis, :])
        self.fig.canvas.draw_idle()

    def _undo(self):
        if not self.history:
            return
        g0, g1, old_vals = self.history.pop()
        self.class_arr[g0:g1] = old_vals
        start, end, _ = self._episode_slice(self.cur_ep)
        if not (g1 <= start or g0 >= end):
            ep_classes = self.class_arr[start:end]
            self.im.set_data(ep_classes[np.newaxis, :])
            self.fig.canvas.draw_idle()

    def _refresh_cursor(self):
        cur_local = self._local_idx(self.cur_global_idx)
        self.cursor_line.set_xdata([cur_local, cur_local])
        self.fig.canvas.draw_idle()

    # ----- Simulation rendering -----
    def _render_current(self):
        i = self.cur_global_idx
        # Reset qpos/qvel
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0

        # Arm joints
        for j, f_idx in enumerate(self.joint_cols_idx):
            q_idx = self.joint_qpos_idx[j]
            self.data.qpos[q_idx] = float(self.features[i, f_idx])

        # Gripper: invert your logging mapping (g in [0,1] -> qpos in [rmin, rmin+0.5*(rmax-rmin)])
        if self.gripper_col_idx is not None and self.gripper_qpos_idx:
            g = float(self.features[i, self.gripper_col_idx])
            g = float(np.clip(g, 0.0, 1.0))
            if self.invert_gripper:
                g = 1.0 - g
            for q_idx in self.gripper_qpos_idx:
                j_id = _joint_id_from_qpos_index(self.model, q_idx)
                if j_id is None:
                    continue
                rmin = float(self.model.jnt_range[j_id, 0])
                rmax = float(self.model.jnt_range[j_id, 1])
                # Protect against invalid ranges
                if not (rmax > rmin):
                    q = 0.0
                else:
                    q = rmin + 0.5 * g * (rmax - rmin)
                self.data.qpos[q_idx] = q

        # Object pose (cube): features are [cx,cy,cz, cox,coy,coz,cow] with quat xyzw; convert to wxyz
        if self.object_pose_idx is not None and self.obj_mode is not None:
            cx, cy, cz, cox, coy, coz, cow = [float(self.features[i, k]) for k in self.object_pose_idx]
            pos = np.array([cx, cy, cz], dtype=float)
            quat_wxyz = _normalize_quat_wxyz(np.array([cow, cox, coy, coz], dtype=float))  # reorder to wxyz
            if self.obj_mode == "freejoint" and self.obj_qpos_adr is not None:
                a = self.obj_qpos_adr
                self.data.qpos[a:a+3] = pos
                self.data.qpos[a+3:a+7] = quat_wxyz
            elif self.obj_mode == "mocap" and self.obj_mocap_id is not None:
                m = self.obj_mocap_id
                self.data.mocap_pos[m, :] = pos
                self.data.mocap_quat[m, :] = quat_wxyz

        mujoco.mj_forward(self.model, self.data)

    def _save_to_h5(self):
        # Persist to features[:, class_col]
        with h5py.File(self.h5_path, "r+") as f:
            feats = f["features"]
            dtype = feats.dtype
            to_write = self.class_arr.astype(dtype, copy=False)
            feats[:, self.class_col_idx] = to_write
            # keep local array in sync
            self.features[:, self.class_col_idx] = to_write
        print(f"Saved '{CLASS_LABEL_FEATURE}' to H5 (column {self.class_col_idx}).")

    def run(self):
        # Launch viewer and main loop
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            self.viewer = viewer
            self._render_current()
            viewer.sync()
            while viewer.is_running() and plt.fignum_exists(self.fig.number):
                if self.play:
                    now = time.time()
                    dt = now - self.last_time
                    self.last_time = now
                    frames = int(dt * self.fps)
                    if frames >= 1:
                        self._step(frames)

                # Render current state
                self._render_current()
                viewer.sync()
                plt.pause(1.0 / 120.0)
        self.viewer = None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--h5", default=DEFAULT_H5, help="Path to HDF5 file")
    p.add_argument("--model", default=DEFAULT_MODEL_XML, help="Path to MuJoCo XML model")
    p.add_argument("--fps", type=float, default=DEFAULT_FPS)
    # You can still override mappings via CLI if needed:
    p.add_argument("--joint-cols", nargs="+", default=DEFAULT_JOINT_COLS)
    p.add_argument("--gripper-col", default=DEFAULT_GRIPPER_COL)
    p.add_argument("--object-pose-cols", nargs="+", default=DEFAULT_OBJECT_POSE_COLS)
    p.add_argument("--joint-qpos-idx", nargs="+", type=int, default=DEFAULT_JOINT_QPOS_IDX)
    p.add_argument("--gripper-qpos-idx", nargs="*", type=int, default=DEFAULT_GRIPPER_QPOS_IDX)
    p.add_argument("--invert-gripper", action="store_true", help="Invert gripper semantics (rare)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    labeler = Labeler(
        h5_path=args.h5,
        model_xml=args.model,
        joint_cols=args.joint_cols,
        gripper_col=args.gripper_col,
        object_pose_cols=args.object_pose_cols,
        joint_qpos_idx=args.joint_qpos_idx,
        gripper_qpos_idx=args.gripper_qpos_idx,
        fps=args.fps,
        invert_gripper=args.invert_gripper,
    )
    labeler.run()