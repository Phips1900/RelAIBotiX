from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import re
import h5py
import json


@dataclass
class ComponentMetrics:
    active: bool
    metrics: Dict[str, float]


@dataclass
class SkillSegment:
    idx: int
    name: str
    start_idx: int
    end_idx: int
    t_start: float
    t_end: float
    duration: float
    components: Dict[str, ComponentMetrics]


@dataclass
class RunTrace:
    run_id: int
    start_idx: int
    end_idx: int
    t_start: float
    t_end: float
    duration: float
    skill_sequence: List[str]
    goal_pos: Tuple[float, float, float]
    final_pos: Tuple[float, float, float]
    pos_error_norm: float
    success: bool
    segments: List[SkillSegment]


#SKILL_ID_TO_NAME = {0: "Init", 1: "Move", 2: "Pick", 3: "Carry", 4: "Place"}
# SKILL_ID_TO_NAME = {0: "Move", 1: "Pick", 2: "Carry", 3: "Place", 4: "Init"}
#SKILL_ID_TO_NAME = {0: "Move", 1: "Pick", 2: "Carry", 3: "Place", 4: "Open", 5: "Drag", 6: "Use_Switch"}
SKILL_ID_TO_NAME = {0: "Move", 1: "Pick", 2: "Carry", 3: "Place", 4: "Stack"}

class BehavioralAnalyzer:
    def __init__(
            self,
            *,
            # success metric
            pos_success_tol: float = 0.02,  # meters (XY success check handled elsewhere)

            # generic “nonzero” cutoff for duty()
            eps_abs: float = 1e-3,

            # generic activity fallbacks (used for non-velocity signals if needed)
            dc_thr: float = 0.2,  # duty >= 20%
            rms_thr: float = 1e-2,  # generic RMS threshold
            range_thr: float = 1e-2,  # generic range threshold

            # velocity-based activity
            eps_vel: float = 0.03,  # rad/s considered "moving"
            rms_vel_thr: float = 0.05,  # rad/s episode-level RMS for active

            # position-based activity
            range_pos_thr: float = 0.02,  # rad (~1.1°) span within episode
            eps_step_pos: float = 0.001,  # rad per sample (≈10 ms at 100 Hz)

            # torque/effort amplitude cutoff
            eps_effort: float = 0.1,

            # gripper logic (state in [0,1], 0=closed, 1=open)
            gripper_close_thresh: float = 0.05,  # <=5% ⇒ considered closed
            gripper_step_thr: float = 0.05,  # >5% change ⇒ transition open↔closed
            gripper_transition_mid: float = 0.5,  # boundary to count transitions

            # velocity band edges and multipliers
            vel_band_bins: tuple = (0.0, 0.5, 1.0, float("inf")),  # [0..0.5], (0.5..1.0], (1.0..inf)
            vel_band_multipliers: tuple = (1.0, 2.0, 5.0),  # (low, med, high)

            # effort band edges and multipliers (optional effort integration)
            eff_band_bins: tuple = (0.0, 0.2, 0.6, float("inf")),
            eff_band_multipliers: tuple = (1.0, 1.25, 1.75),

            # effort integration mode and strength
            default_effort_mode: str = "modifier",  # "off" | "modifier" | "override"
            effort_beta: float = 1.0,

    ):
        # store
        self.pos_success_tol = pos_success_tol

        self.eps_abs = eps_abs
        self.dc_thr = dc_thr
        self.rms_thr = rms_thr
        self.range_thr = range_thr

        self.eps_vel = eps_vel
        self.rms_vel_thr = rms_vel_thr

        self.range_pos_thr = range_pos_thr
        self.eps_step_pos = eps_step_pos

        self.eps_effort = eps_effort

        self.gripper_close_thresh = gripper_close_thresh
        self.gripper_step_thr = gripper_step_thr
        self.gripper_transition_mid = gripper_transition_mid

        self.vel_band_bins = vel_band_bins
        self.vel_band_multipliers = vel_band_multipliers

        self.eff_band_bins = eff_band_bins
        self.eff_band_multipliers = eff_band_multipliers
        self.default_effort_mode = default_effort_mode
        self.effort_beta = effort_beta

    def analyze(
            self,
            *,
            features: pd.DataFrame,
            feature_names: List[str],
            labels: np.ndarray,
            timestamps: np.ndarray,
            trials_csv: Optional[pd.DataFrame] = None,
            episode_labels: Optional[np.ndarray] = None
    ) -> List[RunTrace]:
        # build component mapping from feature_names (auto)
        component_cols = self._infer_component_cols(feature_names)

        # detect runs using last-4 rule
        run_bounds = self._detect_runs(labels, episode_labels=episode_labels)

        traces: List[RunTrace] = []

        feat_names = list(feature_names)

        for run_id, (i0, i1) in enumerate(run_bounds):
            # per-run slice
            X_run = features.iloc[i0:i1 + 1].to_numpy(dtype=float)
            # segment segments via run-length on labels slice
            segments = self._segment_skills(labels[i0:i1 + 1], timestamps[i0:i1 + 1], offset=i0)

            # per-episode component metrics
            for se in segments:
                se.components = self._component_metrics(features, component_cols, timestamps, se.start_idx, se.end_idx)

            # success check
            tol = self.pos_success_tol
            res = self.compute_run_success(run_id, X_run, feat_names, trials_csv, tol)
            success = bool(res["success"])
            pos_err = float(res["pos_err"])
            goal = (float(res["goal"][0]), float(res["goal"][1]), float("nan"))
            final = (float(res["final"][0]), float(res["final"][1]), float("nan"))

            traces.append(RunTrace(
                run_id=run_id,
                start_idx=i0, end_idx=i1,
                t_start=float(timestamps[i0]), t_end=float(timestamps[i1]),
                duration=float(timestamps[i1] - timestamps[i0]),
                skill_sequence=[se.name for se in segments],
                goal_pos=goal, final_pos=final,
                pos_error_norm=pos_err, success=success,
                segments=segments
            ))
        return traces

    # ---- helpers ----
    def _infer_component_cols(self, names: List[str]) -> Dict[str, Dict[str, int]]:
        """
        Auto-maps names like joint_pos_1, joint_vel_1, ..., gripper_state -> component/property indices.
        Produces: {"j1":{"pos": idx, "vel": idx2}, ..., "gripper":{"state": idx}}
        """
        comp: Dict[str, Dict[str, int]] = {}

        pat_primary = re.compile(r"^joint_(pos|vel|eff)_(\d+)$")
        pat_eff_aliases = re.compile(r"^joint_(torque|effort|tau)_(\d+)$")

        for idx, n in enumerate(names):
            s = n.lower()
            # joints
            m = re.match(r"joint_(pos|vel|eff)_(\d+)$", s)
            if m:
                prop = m.group(1)  # 'pos' or 'vel'
                jid = f"j{int(m.group(2))}"
                comp.setdefault(jid, {})[prop] = idx
                continue
            # gripper
            if s == "gripper_state":
                comp.setdefault("gripper", {})["state"] = idx
        return comp

    def _get_col_index(self, names, key):
        try:
            return names.index(key)
        except ValueError:
            return None

    def _success_from_csv(self, trials_csv, run_id, tol):
        if trials_csv is None or run_id not in trials_csv.index:
            return None
        goal = tuple(trials_csv.loc[run_id, ["x_plan", "y_plan"]].astype(float))
        final = tuple(trials_csv.loc[run_id, ["x_real", "y_real"]].astype(float))
        dx, dy = float(goal[0] - final[0]), float(goal[1] - final[1])
        max_abs_xy = max(abs(dx), abs(dy))
        return {
            "success": (max_abs_xy <= tol),
            "pos_err": max_abs_xy,
            "goal": goal,
            "final": final,
            "source": "csv",
        }

    def _success_from_cxcy(self, X_run, feat_names, tol, target_xy=(0.20, -0.15)):
        if X_run is None or len(X_run) == 0 or not feat_names:
            return None
        i_cx = self._get_col_index(feat_names, "cx")
        i_cy = self._get_col_index(feat_names, "cy")
        if i_cx is None or i_cy is None:
            return None
        cx_last = float(X_run[-1, i_cx])
        cy_last = float(X_run[-1, i_cy])
        gx, gy = map(float, target_xy)
        dx, dy = gx - cx_last, gy - cy_last
        max_abs_xy = max(abs(dx), abs(dy))
        return {
            "success": (max_abs_xy <= tol),
            "pos_err": max_abs_xy,
            "goal": (gx, gy),
            "final": (cx_last, cy_last),
            "source": "features_cxcy_last",
        }

    def compute_run_success(self, run_id, X_run, feat_names, trials_csv, tol):
        """
        Returns dict with keys: success (bool), pos_err (float), goal (xy), final (xy), source (str)
        If neither source is available, returns success=False and pos_err=nan with source='none'.
        """
        out = self._success_from_csv(trials_csv, run_id, tol)
        if out is not None:
            return out

        out = self._success_from_cxcy(X_run, feat_names, tol)
        if out is not None:
            return out

        # fallback: nothing available
        return {
            "success": False,  # or None if you prefer tri-state
            "pos_err": float("nan"),
            "goal": (float("nan"), float("nan")),
            "final": (float("nan"), float("nan")),
            "source": "none",
        }

    def _detect_runs_from_episode_labels(self, ep_labels: np.ndarray) -> List[Tuple[int, int]]:
        """
        Treat ep_labels as per-sample run/episode IDs (e.g., 0,1,2,... or any integers).
        Returns contiguous bounds (start_idx, end_idx) for each constant-ID block,
        skipping invalid markers (NaN / -1).
        """

        lab = np.asarray(ep_labels).reshape(-1)
        n = lab.size
        bounds: List[Tuple[int, int]] = []
        i = 0

        is_valid = np.isfinite(lab)

        # mark invalids (NaN) and ignore negative IDs like -1
        def valid_idx(k):
            return (k < n) and is_valid[k] and (lab[k] >= 0)

        while i < n:
            # find next valid start
            while i < n and not valid_idx(i):
                i += 1
            if i >= n:
                break
            s = i
            curr = lab[i]
            i += 1
            # advance while same episode id and valid
            while i < n and valid_idx(i) and lab[i] == curr:
                i += 1
            e = i - 1
            bounds.append((int(s), int(e)))
        return bounds

    def _detect_runs(self, skills: np.ndarray, episode_labels: Optional[np.array]) -> List[Tuple[int, int]]:
        """
        Start at first 0/1 after previous run, end at the LAST 4 before the next 0/1 (or EOF).
        """
        # --- Prefer explicit episode labels if provided ---
        if episode_labels is not None:
            return self._detect_runs_from_episode_labels(episode_labels)


        STARTS = {0, 1}
        n = len(skills);
        bounds = []
        i = 0
        while i < n:
            # find next start
            while i < n and skills[i] not in STARTS:
                i += 1
            if i >= n: break
            s = i;
            i += 1
            last4 = None
            while i < n:
                if skills[i] == 4:
                    last4 = i
                    i += 1
                    while i < n and skills[i] == 4:
                        last4 = i;
                        i += 1
                    # walk until next start or EOF
                    while i < n and skills[i] not in STARTS:
                        i += 1
                    bounds.append((s, last4))
                    break
                else:
                    i += 1
            else:
                # no '4' found before EOF
                bounds.append((s, n - 1))
                break
        return bounds

    def _segment_skills(self, sk_slice: np.ndarray, t_slice: np.ndarray, offset: int) -> List[SkillSegment]:
        eps: List[SkillSegment] = []
        start = 0;
        idx = 0
        for i in range(1, len(sk_slice) + 1):
            if i == len(sk_slice) or sk_slice[i] != sk_slice[start]:
                s_id = int(sk_slice[start])
                name = SKILL_ID_TO_NAME.get(s_id, str(s_id))
                eps.append(SkillSegment(
                    idx=idx, name=name,
                    start_idx=offset + start, end_idx=offset + i - 1,
                    t_start=float(t_slice[start]), t_end=float(t_slice[i - 1]),
                    duration=float(t_slice[i - 1] - t_slice[start]),
                    components={}
                ))
                idx += 1;
                start = i

        return eps

    # def _component_metrics(
    #         self,
    #         feats: pd.DataFrame,
    #         comp_cols: Dict[str, Dict[str, int]],
    #         ts: np.ndarray,  # NEW: timestamps array (shape N,)
    #         i0: int,
    #         i1: int
    # ) -> Dict[str, ComponentMetrics]:
    #     seg = feats.iloc[i0:i1 + 1]
    #     tseg = ts[i0:i1 + 1]
    #     out: Dict[str, ComponentMetrics] = {}
    #
    #     # helper: sum dt where mask at interval start is True
    #     def active_seconds(mask: np.ndarray, t: np.ndarray) -> float:
    #         if mask.size <= 1:
    #             return 0.0
    #         dt = np.diff(t)
    #         return float(np.sum(dt[mask[:-1]]))
    #
    #     for comp, props in comp_cols.items():
    #         m: Dict[str, float] = {}
    #         active_any = False
    #         comp_mask = np.zeros(len(seg), dtype=bool)  # per-sample activity mask (OR of all props)
    #
    #         for prop, col in props.items():
    #             arr = seg.iloc[:, col].to_numpy(dtype=float)
    #             arr = arr[np.isfinite(arr)]
    #             if arr.size == 0:
    #                 continue
    #
    #             prop_l = prop.lower()
    #             comp_l = comp.lower()
    #
    #             # --- common aggregates ---
    #             mean = float(np.mean(arr))
    #             std = float(np.std(arr))
    #             rms = float(np.sqrt(np.mean(arr ** 2)))
    #             mx = float(np.max(arr))
    #             rng = float(np.max(arr) - np.min(arr))
    #             duty = float(np.mean(np.abs(arr) > getattr(self, "eps_abs", 1e-3)))
    #
    #             m[f"{comp}.{prop}_mean"] = mean
    #             m[f"{comp}.{prop}_std"] = std
    #             m[f"{comp}.{prop}_rms"] = rms
    #             m[f"{comp}.{prop}_max"] = mx
    #             m[f"{comp}.{prop}_range"] = rng
    #             m[f"{comp}.{prop}_duty"] = duty
    #
    #             # build a per-sample activity mask for this property
    #             prop_mask = np.zeros_like(comp_mask)
    #
    #             # --- velocity-specific extras & mask ---
    #             if "vel" in prop_l:
    #                 absmax = float(np.max(np.abs(arr)))
    #                 m[f"{comp}.{prop}_absmax"] = absmax  # magnitude peak
    #                 m[f"{comp}.{prop}_max"] = mx
    #
    #                 eps_vel = self.eps_vel
    #                 rms_vel_thr = self.rms_vel_thr  # rad/s
    #                 dc_thr = self.dc_thr
    #
    #                 duty_vel = float(np.mean(np.abs(arr) > eps_vel))
    #                 m[f"{comp}.{prop}_duty_vel"] = duty_vel
    #
    #                 # velocity mask: above eps_vel
    #                 prop_mask = (np.abs(arr) > eps_vel)
    #
    #                 # episode-level activation
    #                 active_any = active_any or (duty_vel >= dc_thr) or (rms >= rms_vel_thr)
    #
    #                 # time in velocity bands (per episode)
    #                 bt = self._band_times_from_velocity(arr, tseg)
    #                 m[f"{comp}.{prop}_time_low_sec"] = bt["time_low_sec"]
    #                 m[f"{comp}.{prop}_time_med_sec"] = bt["time_med_sec"]
    #                 m[f"{comp}.{prop}_time_high_sec"] = bt["time_high_sec"]
    #                 m[f"{comp}.{prop}_moving_time_sec"] = bt["moving_time_sec"]
    #
    #                 # fractions w.r.t. episode duration (guard div-by-zero)
    #                 mt = bt["moving_time_sec"]
    #                 # ep_dur = float(tseg[-1] - tseg[0]) if tseg[-1] > tseg[0] else 0.0
    #                 if mt > 0:
    #                     m[f"{comp}.{prop}_frac_low"] = bt["time_low_sec"] / mt
    #                     m[f"{comp}.{prop}_frac_med"] = bt["time_med_sec"] / mt
    #                     m[f"{comp}.{prop}_frac_high"] = bt["time_high_sec"] / mt
    #                 else:
    #                     m[f"{comp}.{prop}_frac_low"] = 0.0
    #                     m[f"{comp}.{prop}_frac_med"] = 0.0
    #                     m[f"{comp}.{prop}_frac_high"] = 0.0
    #
    #             # --- position: range criterion + per-sample step mask ---
    #             elif "pos" in prop_l:
    #                 range_pos_thr = getattr(self, "range_pos_thr", 0.02)  # rad
    #                 step_thr = getattr(self, "eps_step_pos", 0.001)  # rad / sample (≈10ms at 100Hz)
    #
    #                 # per-sample |Δpos|
    #                 steps = np.r_[0.0, np.abs(np.diff(arr))]
    #                 prop_mask = (steps > step_thr)
    #
    #                 active_any = active_any or (rng >= range_pos_thr)
    #
    #             # --- torque/effort: simple amplitude/rms mask ---
    #             elif prop_l in ("eff", "tau", "torque", "effort"):
    #                 eps_eff = getattr(self, "eps_effort", getattr(self, "eps_abs", 1e-3))
    #                 prop_mask = (np.abs(arr) > eps_eff)
    #                 active_any = active_any or (rms >= getattr(self, "rms_thr", 1e-2))
    #
    #                 # Effort band exposure (optional, used later for modifiers)
    #                 bt = self._band_times_from_effort(arr, tseg)
    #                 m[f"{comp}.{prop}_time_low_sec"] = bt["time_low_sec"]
    #                 m[f"{comp}.{prop}_time_med_sec"] = bt["time_med_sec"]
    #                 m[f"{comp}.{prop}_time_high_sec"] = bt["time_high_sec"]
    #                 m[f"{comp}.{prop}_active_time_sec"] = bt["active_time_sec"]
    #
    #                 at = bt["active_time_sec"]
    #                 if at > 0:
    #                     m[f"{comp}.{prop}_frac_low"] = bt["time_low_sec"] / at
    #                     m[f"{comp}.{prop}_frac_med"] = bt["time_med_sec"] / at
    #                     m[f"{comp}.{prop}_frac_high"] = bt["time_high_sec"] / at
    #                 else:
    #                     m[f"{comp}.{prop}_frac_low"] = 0.0
    #                     m[f"{comp}.{prop}_frac_med"] = 0.0
    #                     m[f"{comp}.{prop}_frac_high"] = 0.0
    #                 prop_mask = (np.abs(arr) > eps_eff)
    #                 active_any = active_any or (rms >= getattr(self, "rms_thr", 1e-2))
    #
    #             # --- special: gripper state in [0,1] (0=closed, 1=open) ---
    #             if comp_l == "gripper" and prop_l in ("state", "pos"):
    #                 state = np.clip(arr, 0.0, 1.0)
    #                 close_thr = getattr(self, "gripper_close_thresh", 0.05)
    #                 step_thr = getattr(self, "gripper_step_thr", 0.05)
    #                 mid = getattr(self, "gripper_transition_mid", 0.5)
    #
    #                 closed_mask = (state <= close_thr)
    #                 # transitions: open<->closed edges when |Δstate| > step_thr
    #                 trans = np.abs(np.diff(state)) > step_thr
    #                 trans_mask = np.r_[False, trans]
    #
    #                 # gripper "used" whenever closed OR transitioning
    #                 gripper_mask = closed_mask | trans_mask
    #                 prop_mask = gripper_mask
    #
    #                 closed_frac = float(np.mean(closed_mask))
    #                 transitions = int(np.sum((state[:-1] > mid) != (state[1:] > mid)))
    #
    #                 m[f"{comp}.{prop}_closed_frac"] = closed_frac
    #                 m[f"{comp}.{prop}_transitions"] = transitions
    #
    #                 # episode-level activation logic for gripper
    #                 gripper_used = (closed_frac > 0.0) or (transitions >= 1)
    #                 active_any = active_any or gripper_used
    #
    #             # combine into component-level mask
    #             comp_mask |= prop_mask
    #
    #             # generic fallback (for properties that didn't hit a branch)
    #             if not (comp_l == "gripper" and prop_l in ("state", "pos")) and prop_l not in ("vel", "pos", "tau",
    #                                                                                            "torque", "effort"):
    #                 active_any = active_any or (
    #                         (duty >= getattr(self, "dc_thr", 0.2)) or
    #                         (rms >= getattr(self, "rms_thr", 1e-2)) or
    #                         (rng >= getattr(self, "range_thr", 1e-2))
    #                 )
    #
    #         # --- active time (seconds) & fraction of episode ---
    #         if tseg.size >= 2:
    #             comp_active_sec = active_seconds(comp_mask, tseg)
    #             ep_dur = float(tseg[-1] - tseg[0]) if tseg[-1] > tseg[0] else 0.0
    #             comp_active_frac = (comp_active_sec / ep_dur) if ep_dur > 0 else 0.0
    #         else:
    #             comp_active_sec = 0.0
    #             comp_active_frac = 0.0
    #
    #         # If component isn't active by episode-level rule, force time to 0 (your requirement)
    #         if not active_any:
    #             comp_active_sec = 0.0
    #             comp_active_frac = 0.0
    #
    #         m[f"{comp}.active_time_sec"] = comp_active_sec
    #         m[f"{comp}.active_fraction"] = comp_active_frac
    #
    #         out[comp] = ComponentMetrics(active=bool(active_any), metrics=m)
    #
    #     return out

    def _component_metrics(
            self,
            feats: pd.DataFrame,
            comp_cols: Dict[str, Dict[str, int]],
            ts: np.ndarray,
            i0: int,
            i1: int
    ) -> Dict[str, ComponentMetrics]:
        """
        Compute per-component metrics for segment [i0:i1] (inclusive).
        - comp_cols maps component -> dict of property -> column index, e.g. {"j1":{"pos":i,"vel":j,"eff":k}}
        - All activity masks are built on the base timeline (ts[i0:i1+1]) as per-interval masks of length len(ts_seg)-1.
        """
        import numpy as np

        seg = feats.iloc[i0:i1 + 1]
        tseg = np.asarray(ts[i0:i1 + 1], dtype=float)
        out: Dict[str, ComponentMetrics] = {}

        if tseg.size <= 1:
            return out  # nothing to do

        dt_base = np.diff(tseg)  # length len(tseg)-1
        ep_dur = float(tseg[-1] - tseg[0]) if tseg[-1] > tseg[0] else 0.0

        for comp, props in comp_cols.items():
            m: Dict[str, float] = {}
            active_any = False

            # Per-component activity mask on BASE timeline
            comp_mask = np.zeros(len(dt_base), dtype=bool)

            for prop, col in props.items():
                prop_l = str(prop).lower()
                comp_l = str(comp).lower()

                # Slice the series from feats on this segment
                arr = seg.iloc[:, col].to_numpy(dtype=float)
                # Keep raw arrays (don't shrink!) so masks align with tseg
                # Basic aggregates (guard non-finite safely)
                arr_finite = arr[np.isfinite(arr)]
                if arr_finite.size == 0:
                    continue

                mean = float(np.mean(arr_finite))
                std = float(np.std(arr_finite))
                rms = float(np.sqrt(np.mean(arr_finite ** 2)))
                mx = float(np.max(arr_finite))
                rng = float(np.max(arr_finite) - np.min(arr_finite))
                duty = float(np.mean(np.abs(arr_finite) > getattr(self, "eps_abs", 1e-3)))

                m[f"{comp}.{prop}_mean"] = mean
                m[f"{comp}.{prop}_std"] = std
                m[f"{comp}.{prop}_rms"] = rms
                m[f"{comp}.{prop}_max"] = mx
                m[f"{comp}.{prop}_range"] = rng
                m[f"{comp}.{prop}_duty"] = duty

                # ---- Velocity branch (native timeline = base) ----
                if "vel" in prop_l:
                    eps_vel = getattr(self, "eps_vel", 1e-3)
                    rms_vel_thr = getattr(self, "rms_vel_thr", 1e-2)
                    dc_thr = getattr(self, "dc_thr", 0.2)

                    # per-interval mask on base timeline
                    vel_mask = self._active_mask_from_series(arr, tseg, eps_vel)
                    comp_mask |= vel_mask

                    # episode-level activation
                    duty_vel = float(np.mean(np.abs(arr_finite) > eps_vel))
                    m[f"{comp}.{prop}_duty_vel"] = duty_vel
                    active_any = active_any or (duty_vel >= dc_thr) or (rms >= rms_vel_thr)

                    # velocity band times (on base timeline)
                    vbt = self._band_times_from_velocity(arr, tseg)
                    m[f"{comp}.{prop}_time_low_sec"] = vbt["time_low_sec"]
                    m[f"{comp}.{prop}_time_med_sec"] = vbt["time_med_sec"]
                    m[f"{comp}.{prop}_time_high_sec"] = vbt["time_high_sec"]
                    m[f"{comp}.{prop}_moving_time_sec"] = vbt["moving_time_sec"]

                    mt = vbt["moving_time_sec"]
                    if mt > 0.0:
                        m[f"{comp}.{prop}_frac_low"] = vbt["time_low_sec"] / mt
                        m[f"{comp}.{prop}_frac_med"] = vbt["time_med_sec"] / mt
                        m[f"{comp}.{prop}_frac_high"] = vbt["time_high_sec"] / mt
                    else:
                        m[f"{comp}.{prop}_frac_low"] = 0.0
                        m[f"{comp}.{prop}_frac_med"] = 0.0
                        m[f"{comp}.{prop}_frac_high"] = 0.0

                # ---- Position branch (range + step activity) ----
                elif "pos" in prop_l:
                    range_pos_thr = getattr(self, "range_pos_thr", 0.02)  # rad
                    step_thr = getattr(self, "eps_step_pos", 0.001)  # rad / sample

                    steps = np.r_[0.0, np.abs(np.diff(arr))]  # per-sample
                    # convert to per-interval by looking at left endpoint step
                    pos_mask = steps[:-1] > step_thr
                    comp_mask |= pos_mask

                    active_any = active_any or (rng >= range_pos_thr)

                # ---- Effort/Torque branch (may be different timeline) ----
                elif prop_l in ("eff", "tau", "torque", "effort"):
                    eps_eff = getattr(self, "eps_effort", getattr(self, "eps_abs", 1e-3))

                    # effort may be downsampled → get aligned ts specific to this column
                    eff_vals, eff_ts = self._slice_series_with_ts(feats, ts, i0, i1, col)

                    # per-interval mask on its own timeline
                    eff_mask_local = self._active_mask_from_series(eff_vals, eff_ts, eps_eff)
                    # turn mask into (t0,t1) intervals on eff_ts
                    eff_intervals = self._boolean_runs_to_intervals(eff_mask_local, eff_ts)
                    # project onto base timeline → per-interval mask length == len(dt_base)
                    eff_mask_on_base = self._intervals_to_base_mask(eff_intervals, tseg)
                    comp_mask |= eff_mask_on_base

                    # Effort band exposure (computed on effort timeline ONLY)
                    ebt = self._band_times_from_effort(eff_vals, eff_ts)
                    m[f"{comp}.{prop}_time_low_sec"] = ebt["time_low_sec"]
                    m[f"{comp}.{prop}_time_med_sec"] = ebt["time_med_sec"]
                    m[f"{comp}.{prop}_time_high_sec"] = ebt["time_high_sec"]
                    m[f"{comp}.{prop}_active_time_sec"] = ebt["active_time_sec"]

                    at = ebt["active_time_sec"]
                    if at > 0.0:
                        m[f"{comp}.{prop}_frac_low"] = ebt["time_low_sec"] / at
                        m[f"{comp}.{prop}_frac_med"] = ebt["time_med_sec"] / at
                        m[f"{comp}.{prop}_frac_high"] = ebt["time_high_sec"] / at
                    else:
                        m[f"{comp}.{prop}_frac_low"] = 0.0
                        m[f"{comp}.{prop}_frac_med"] = 0.0
                        m[f"{comp}.{prop}_frac_high"] = 0.0

                    # episode-level activation
                    active_any = active_any or (rms >= getattr(self, "rms_thr", 1e-2))

                # ---- Gripper branch (state ∈ [0,1]) ----
                if comp_l == "gripper" and prop_l in ("state", "pos"):
                    state = np.clip(arr, 0.0, 1.0)
                    close_thr = getattr(self, "gripper_close_thresh", 0.05)
                    step_thr = getattr(self, "gripper_step_thr", 0.05)
                    mid = getattr(self, "gripper_transition_mid", 0.5)

                    closed_left = state[:-1] <= close_thr  # per-interval
                    trans = np.abs(np.diff(state)) > step_thr  # per-interval
                    gripper_mask = closed_left | trans
                    comp_mask |= gripper_mask

                    closed_frac = float(np.mean(state <= close_thr))
                    transitions = int(np.sum((state[:-1] > mid) != (state[1:] > mid)))

                    m[f"{comp}.{prop}_closed_frac"] = closed_frac
                    m[f"{comp}.{prop}_transitions"] = transitions

                    gripper_used = (closed_frac > 0.0) or (transitions >= 1)
                    active_any = active_any or gripper_used

                # ---- Generic fallback: episode-level "active" heuristics only ----
                if not (("vel" in prop_l) or ("pos" in prop_l) or (prop_l in ("eff", "tau", "torque", "effort")) or (
                        comp_l == "gripper" and prop_l in ("state", "pos"))):
                    active_any = active_any or (
                            (duty >= getattr(self, "dc_thr", 0.2)) or
                            (rms >= getattr(self, "rms_thr", 1e-2)) or
                            (rng >= getattr(self, "range_thr", 1e-2))
                    )

            # --- Active time and fraction for this component (on base timeline) ---
            comp_active_sec = float(np.sum(dt_base[comp_mask])) if dt_base.size else 0.0
            comp_active_frac = (comp_active_sec / ep_dur) if ep_dur > 0.0 else 0.0

            # If "inactive" per episode-level logic, zero-out the activity summary (as per your requirement)
            if not active_any:
                comp_active_sec = 0.0
                comp_active_frac = 0.0

            m[f"{comp}.active_time_sec"] = comp_active_sec
            m[f"{comp}.active_fraction"] = comp_active_frac

            out[comp] = ComponentMetrics(active=bool(active_any), metrics=m)

        return out

    def summarize_timings(self, traces):
        """
        Returns:
          - global_totals: dict with 'total_run_time_sec' and 'n_runs'
          - skill_time:   DataFrame [skill, total_time_sec, n_episodes, avg_time_per_episode_sec, avg_time_per_run_sec]
          - comp_active:  DataFrame [component, total_active_time_sec]
          - comp_active_by_skill: DataFrame [skill, component, total_active_time_sec]
        """
        runs_df, segments_df, components_df = self.to_frames(traces)

        # ---- totals
        total_run_time = float(runs_df["duration"].sum())
        n_runs = int(len(runs_df))
        global_totals = {"total_run_time_sec": total_run_time, "n_runs": n_runs}

        # ---- skill totals and averages
        skill_group = segments_df.groupby("skill", as_index=False)["duration"].agg(total_time_sec="sum",
                                                                                   n_episodes="count")
        skill_group["avg_time_per_episode_sec"] = skill_group["total_time_sec"] / skill_group["n_episodes"]

        # per-run time spent in each skill → average across runs
        per_run_skill = segments_df.groupby(["run_id", "skill"], as_index=False)["duration"].sum()
        avg_per_run = per_run_skill.groupby("skill", as_index=False)["duration"].mean().rename(
            columns={"duration": "avg_time_per_run_sec"})
        skill_time = skill_group.merge(avg_per_run, on="skill", how="left")

        # ---- component active time totals (across all episodes/runs)
        if "active_time_sec" in components_df.columns:
            comp_active = components_df.groupby("component", as_index=False)["active_time_sec"].sum() \
                .rename(columns={"active_time_sec": "total_active_time_sec"})
            comp_active_by_skill = components_df.groupby(["skill", "component"], as_index=False)[
                "active_time_sec"].sum() \
                .rename(columns={"active_time_sec": "total_active_time_sec"})
        else:
            # if not present, return empties
            comp_active = pd.DataFrame(columns=["component", "total_active_time_sec"])
            comp_active_by_skill = pd.DataFrame(columns=["skill", "component", "total_active_time_sec"])

        return global_totals, skill_time, comp_active, comp_active_by_skill

    def to_frames(self, traces):

        # ---- runs_df
        runs_df = pd.DataFrame([dict(
            run_id=r.run_id,
            start_idx=r.start_idx, end_idx=r.end_idx,
            t_start=r.t_start, t_end=r.t_end,
            duration=r.duration,
            skill_sequence=r.skill_sequence,
            goal_x=r.goal_pos[0], goal_y=r.goal_pos[1],
            final_x=r.final_pos[0], final_y=r.final_pos[1],
            pos_error_norm=r.pos_error_norm, success=r.success
        ) for r in traces])

        # ---- segments_df
        episodes_rows = []
        for r in traces:
            for se in r.segments:
                episodes_rows.append(dict(
                    run_id=r.run_id, ep_idx=se.idx, skill=se.name,
                    start_idx=se.start_idx, end_idx=se.end_idx,
                    t_start=se.t_start, t_end=se.t_end, duration=se.duration
                ))
        segments_df = pd.DataFrame(episodes_rows)

        # ---- components_df (normalized active time columns)
        comp_rows = []
        for r in traces:
            for se in r.segments:
                for comp, cm in se.components.items():
                    row = dict(
                        run_id=r.run_id, ep_idx=se.idx, skill=se.name,
                        component=comp, active=cm.active,
                    )
                    # normalized keys (so you don't need to parse metric dict keys later)
                    row["active_time_sec"] = cm.metrics.get(f"{comp}.active_time_sec", 0.0)
                    row["active_fraction"] = cm.metrics.get(f"{comp}.active_fraction", 0.0)
                    # optional convenience: velocity peaks if present
                    row["vel_absmax"] = cm.metrics.get(f"{comp}.vel_absmax", None)
                    row["vel_max"] = cm.metrics.get(f"{comp}.vel_max", None)
                    row["vel_rms"] = cm.metrics.get(f"{comp}.vel_rms", None)
                    # keep the rest of the metrics (wide) if you like:
                    wide = {k: v for k, v in cm.metrics.items() if
                            not k.endswith(".active_time_sec") and not k.endswith(".active_fraction")}
                    row.update(wide)
                    comp_rows.append(row)
        components_df = pd.DataFrame(comp_rows)

        return runs_df, segments_df, components_df

    def _band_times_from_velocity(self, vel: np.ndarray, ts: np.ndarray):
        """
        vel: 1D velocity array over the episode (rad/s)
        ts:  1D timestamps over the episode (sec)
        Returns dict with seconds spent in each band: low/med/high.
        """
        if vel.size <= 1 or ts.size <= 1:
            return dict(time_low_sec=0.0, time_med_sec=0.0, time_high_sec=0.0)

        speed = np.abs(vel)
        # speed = np.where(speed > self.eps_vel, speed, 0.0)  # suppress tiny noise
        dt = np.diff(ts)
        s = speed[:-1]  # align with dt (interval starts)

        b0, b1, b2, b3 = self.vel_band_bins
        move_mask = (s > self.eps_vel)

        low_mask = move_mask & (s <= b1)
        med_mask = move_mask & (s > b1) & (s <= b2)
        high_mask = move_mask & (s > b2)

        t_low = float(np.sum(dt[low_mask]))
        t_med = float(np.sum(dt[med_mask]))
        t_high = float(np.sum(dt[high_mask]))
        t_move = t_low + t_med + t_high
        return dict(time_low_sec=t_low, time_med_sec=t_med, time_high_sec=t_high, moving_time_sec=t_move)

    def _band_times_from_effort(self, eff: np.ndarray, ts: np.ndarray):
        """
        Effort bands over the episode/segment. Returns:
          dict(time_low_sec, time_med_sec, time_high_sec, active_time_sec)

        Robust to length mismatches: if len(eff) != len(ts), we construct a uniform
        dt over [ts[0], ts[-1]] with length len(eff)-1 so the masks and dt align.
        """
        eff = np.asarray(eff).reshape(-1)
        ts = np.asarray(ts).reshape(-1)

        if eff.size <= 1 or ts.size <= 1:
            return dict(time_low_sec=0.0, time_med_sec=0.0, time_high_sec=0.0, active_time_sec=0.0)

        # Build dt aligned to the effort series
        if eff.size == ts.size:
            dt = np.diff(ts)
        else:
            # Fallback: uniform spacing over the segment duration
            total = float(ts[-1] - ts[0])
            steps = max(int(eff.size) - 1, 1)
            dt = np.full(steps, total / steps, dtype=float)

        a = np.abs(eff)
        a0 = a[:-1]  # align with dt

        b0, b1, b2, b3 = self.eff_band_bins
        act_mask = (a0 > self.eps_effort)

        low_mask = act_mask & (a0 >= b0) & (a0 < b1)
        med_mask = act_mask & (a0 >= b1) & (a0 < b2)
        high_mask = act_mask & (a0 >= b2) & (a0 < b3)

        # IMPORTANT: masks and dt now have identical length
        t_low = float(np.sum(dt[low_mask]))
        t_med = float(np.sum(dt[med_mask]))
        t_high = float(np.sum(dt[high_mask]))
        t_act = t_low + t_med + t_high

        return dict(time_low_sec=t_low, time_med_sec=t_med, time_high_sec=t_high, active_time_sec=t_act)

    def _slice_series_with_ts(self, features, timestamps, idx0, idx1, col_idx):
        """Return (series, ts_for_series) over [idx0:idx1] inclusive.
        If the column has a different sample count, generate a matching ts."""
        series = features.iloc[idx0:idx1 + 1, col_idx].to_numpy().reshape(-1)
        ts_seg = np.asarray(timestamps[idx0:idx1 + 1], dtype=float).reshape(-1)
        if series.shape[0] != ts_seg.shape[0]:
            if series.shape[0] >= 2:
                ts_aligned = np.linspace(ts_seg[0], ts_seg[-1], num=series.shape[0])
            else:
                ts_aligned = ts_seg[:series.shape[0]]
        else:
            ts_aligned = ts_seg
        return series, ts_aligned

    def _boolean_runs_to_intervals(self, mask_dt: np.ndarray, ts: np.ndarray):
        """mask_dt is length len(ts)-1. Return list of (t_start, t_end) where mask is True."""
        if mask_dt.size == 0:
            return []
        x = mask_dt.astype(np.int8)
        d = np.diff(np.r_[0, x, 0])
        starts = np.where(d == 1)[0]
        ends = np.where(d == -1)[0]
        # each True run covers dt cells [k ... e-1] -> time [ts[k], ts[e]]
        intervals = [(float(ts[k]), float(ts[e])) for k, e in zip(starts, ends)]
        return intervals

    def _intervals_to_base_mask(self, intervals, base_ts: np.ndarray):
        """Project (t_start, t_end) intervals onto base timeline.
        Returns mask of length len(base_ts)-1 where any overlap marks True."""
        n = max(len(base_ts) - 1, 0)
        out = np.zeros(n, dtype=bool)
        if n == 0 or not intervals:
            return out
        for t0, t1 in intervals:
            if t1 <= base_ts[0] or t0 >= base_ts[-1]:
                continue
            # indices of base bins [base_ts[i], base_ts[i+1])
            i0 = max(0, int(np.searchsorted(base_ts, t0, side="right") - 1))
            i1 = min(n, int(np.searchsorted(base_ts, t1, side="left")))
            if i1 > i0:
                out[i0:i1] = True
        return out

    def _active_mask_from_series(self, values: np.ndarray, ts: np.ndarray, eps: float):
        """Build per-dt active mask: True when |values| > eps (length len(ts)-1)."""
        if values.size <= 1 or ts.size <= 1:
            return np.zeros(0, dtype=bool)
        v0 = np.abs(values[:-1])
        return v0 > eps

    def summarize(self, traces) -> Dict[str, pd.DataFrame]:
        """
        Returns a dict of DataFrames:
          - sequences:        frequency of skill sequences (count, percent)
          - overall:          one-row overall summary (totals, success rate)
          - skill_time:       per-skill total/avg time
          - comp_usage:       per-skill × component usage/time
          - joint_velocity:   per-skill × joint velocity peaks (absmax totals, avg max)
        """
        runs_df, segments_df, components_df = self.to_frames(traces)

        # --- 1) Skill-sequence frequencies ---
        seq_series = runs_df["skill_sequence"].apply(lambda seq: " > ".join(seq))
        seq_counts = seq_series.value_counts().rename_axis("sequence").reset_index(name="count")
        n_runs = len(runs_df)
        seq_counts["percent"] = 100.0 * seq_counts["count"] / max(n_runs, 1)

        # --- 2) Overall totals & success rate ---
        overall = pd.DataFrame([{
            "n_runs": n_runs,
            "total_run_time_sec": float(runs_df["duration"].sum()),
            "success_rate_percent": 100.0 * float(runs_df["success"].mean()) if n_runs else 0.0,
        }])

        # --- 3) Per-skill total/average time ---
        skill_time = (
            segments_df.groupby("skill", as_index=False)["duration"]
            .agg(total_time_sec="sum", n_episodes="count")
        )
        skill_time["avg_time_per_episode_sec"] = skill_time["total_time_sec"] / skill_time["n_episodes"]
        # per-run avg time in each skill (some runs may lack a skill)
        per_run_skill = segments_df.groupby(["run_id", "skill"], as_index=False)["duration"].sum()
        avg_per_run = (
            per_run_skill.groupby("skill", as_index=False)["duration"]
            .mean().rename(columns={"duration": "avg_time_per_run_sec"})
        )
        skill_time = skill_time.merge(avg_per_run, on="skill", how="left")

        # --- 4) Per-skill × component usage/time ---
        # n_episodes per skill to compute active %
        n_eps_per_skill = segments_df.groupby("skill", as_index=False)["ep_idx"].count() \
            .rename(columns={"ep_idx": "n_episodes_skill"})

        # count episodes where component is active (per skill)
        # components_df has rows per (run_id, ep_idx, skill, component)
        active_counts = (
            components_df.groupby(["skill", "component"], as_index=False)["active"]
            .sum().rename(columns={"active": "n_active_episodes"})
        )
        # total & avg active time per skill×component
        time_aggs = (
            components_df.groupby(["skill", "component"], as_index=False)["active_time_sec"]
            .agg(total_active_time_sec="sum", avg_active_time_sec="mean")
        )
        comp_usage = (
            active_counts.merge(time_aggs, on=["skill", "component"], how="outer")
            .merge(n_eps_per_skill, on="skill", how="left")
            .fillna({"n_active_episodes": 0, "total_active_time_sec": 0.0, "avg_active_time_sec": 0.0})
        )
        comp_usage["active_pct_episodes"] = 100.0 * comp_usage["n_active_episodes"] / comp_usage[
            "n_episodes_skill"].clip(lower=1)

        # --- 5) Per-skill × joint velocity peaks ---
        # Keep only joint components (j1..j7). If you name differently, adjust the startswith.
        joints_only = components_df[components_df["component"].str.startswith("j")]
        # robust: some columns might be missing if you didn't store them
        for col in ["vel_absmax", "vel_max"]:
            if col not in joints_only.columns:
                joints_only[col] = pd.NA

        joint_velocity = (
            joints_only.groupby(["skill", "component"], as_index=False)
            .agg(
                total_vel_absmax=("vel_absmax", "sum"),
                avg_vel_max=("vel_max", "mean"),
                avg_vel_absmax=("vel_absmax", "mean"),
                max_vel=("vel_max", "max"),
                max_abs_vel=("vel_absmax", "max"),
            )
        )

        # --- Velocity bands (per skill × joint) ---
        joints_only = components_df[components_df["component"].str.startswith("j")].copy()

        def pick_cols(df, suffix):
            cols = [c for c in df.columns if c.endswith(suffix)]
            return cols[0] if cols else None

        # consolidate band columns (they may be stored per-prop "vel")
        def col_or_nan(df, suffix):
            # Try the standard key name as we emitted in _component_metrics
            # We stored per component as f"{comp}.vel_time_low_sec", but in components_df
            # we flattened keys, so they will be columns named like "j1.vel_time_low_sec".
            cols = [c for c in df.columns if c.endswith(suffix)]
            if not cols:
                return None
            # Sum across any duplicates (shouldn't happen; one vel per joint), but safe:
            df[suffix] = df[cols].sum(axis=1)
            return suffix

        # low_key = col_or_nan(joints_only, ".vel_time_low_sec")
        low_col = col_or_nan(joints_only, ".vel_time_low_sec")
        med_col = col_or_nan(joints_only, ".vel_time_med_sec")
        high_col = col_or_nan(joints_only, ".vel_time_high_sec")
        mov_col = col_or_nan(joints_only, ".vel_moving_time_sec")

        # Aggregate total band times per skill×joint
        if all(c is not None for c in (low_col, med_col, high_col, mov_col)):
            band_aggs = (
                joints_only.groupby(["skill", "component"], as_index=False)[[low_col, med_col, high_col, mov_col]]
                .sum()
                .rename(columns={
                    low_col: "time_low_sec",
                    med_col: "time_med_sec",
                    high_col: "time_high_sec",
                    mov_col: "moving_time_sec",
                })
            )
            mt = band_aggs["moving_time_sec"].replace(0, np.nan)
            band_aggs["frac_low_of_moving"] = band_aggs["time_low_sec"] / mt
            band_aggs["frac_med_of_moving"] = band_aggs["time_med_sec"] / mt
            band_aggs["frac_high_of_moving"] = band_aggs["time_high_sec"] / mt
            band_aggs[["frac_low_of_moving", "frac_med_of_moving", "frac_high_of_moving"]] = \
                band_aggs[["frac_low_of_moving", "frac_med_of_moving", "frac_high_of_moving"]].fillna(0.0)
        else:
            band_aggs = pd.DataFrame(columns=[
                "skill", "component", "time_low_sec", "time_med_sec", "time_high_sec",
                "moving_time_sec", "frac_low_of_moving", "frac_med_of_moving", "frac_high_of_moving"
            ])

        # --- Effort bands (optional) ---
        joints_only = components_df[components_df["component"].str.startswith("j")].copy()

        def col_or_nan(df, suffix):
            cols = [c for c in df.columns if c.endswith(suffix)]
            if not cols:
                return None
            df[suffix] = df[cols].sum(axis=1)
            return suffix

        eff_low_col = col_or_nan(joints_only, ".eff_time_low_sec")
        eff_med_col = col_or_nan(joints_only, ".eff_time_med_sec")
        eff_high_col = col_or_nan(joints_only, ".eff_time_high_sec")
        eff_act_col = col_or_nan(joints_only, ".eff_active_time_sec")

        if all(c is not None for c in (eff_low_col, eff_med_col, eff_high_col, eff_act_col)):
            effort_bands = (
                joints_only.groupby(["skill", "component"], as_index=False)[
                        [eff_low_col, eff_med_col, eff_high_col, eff_act_col]]
                .sum()
                .rename(columns={
                    eff_low_col: "eff_time_low_sec",
                    eff_med_col: "eff_time_med_sec",
                    eff_high_col: "eff_time_high_sec",
                    eff_act_col: "eff_active_time_sec",
                })
            )
            eat = effort_bands["eff_active_time_sec"].replace(0, np.nan)
            effort_bands["eff_frac_low"] = effort_bands["eff_time_low_sec"] / eat
            effort_bands["eff_frac_med"] = effort_bands["eff_time_med_sec"] / eat
            effort_bands["eff_frac_high"] = effort_bands["eff_time_high_sec"] / eat
            effort_bands[["eff_frac_low", "eff_frac_med", "eff_frac_high"]] = effort_bands[
                    ["eff_frac_low", "eff_frac_med", "eff_frac_high"]].fillna(0.0)
        else:
            effort_bands = pd.DataFrame(columns=[
                    "skill", "component", "eff_time_low_sec", "eff_time_med_sec", "eff_time_high_sec",
                    "eff_active_time_sec", "eff_frac_low", "eff_frac_med", "eff_frac_high"
            ])

        return {
            "sequences": seq_counts,
            "overall": overall,
            "skill_time": skill_time,
            "comp_usage": comp_usage,
            "joint_velocity": joint_velocity,
            "velocity_bands": band_aggs,
            "effort_bands": effort_bands
        }

    # def assess_failure_from_bands(self, traces):
    #     """
    #     Compute P_fail(skill, component) from band exposure.
    #     base_prob_per_minute: dict mapping component -> p0 per minute (e.g., {"j1":1e-3, "joint":1e-3, "default":1e-3})
    #     Uses self.vel_band_multipliers = (m_low, m_med, m_high).
    #     Returns DataFrame with columns:
    #       [skill, component, p0_per_min, lambda0_per_s, time_low_sec, time_med_sec, time_high_sec,
    #        m_low, m_med, m_high, hazard, p_fail]
    #     """
    #
    #     summary = self.summarize(traces)
    #     bands = summary["velocity_bands"].copy()
    #     if bands.empty:
    #         return pd.DataFrame(columns=[
    #             "skill", "component", "p0_per_min", "lambda0_per_s", "time_low_sec", "time_med_sec", "time_high_sec",
    #             "m_low", "m_med", "m_high", "hazard", "p_fail"
    #         ])
    #
    #     m_low, m_med, m_high = self.vel_band_multipliers
    #
    #     def base_p0_for(comp: str):
    #         if comp in self.base_prob_per_minute:
    #             return float(self.base_prob_per_minute[comp])
    #         # allow a generic 'joint' default for all j*
    #         if comp.startswith("j") and "joint" in base_prob_per_minute:
    #             return float(self.base_prob_per_minute["joint"])
    #         if "default" in self.base_prob_per_minute:
    #             return float(self.base_prob_per_minute["default"])
    #         raise ValueError(
    #             f"No base failure probability provided for component '{comp}' and no 'joint'/'default' fallback.")
    #
    #     rows = []
    #     for _, row in bands.iterrows():
    #         comp = row["component"]
    #         p0 = base_p0_for(comp)
    #         # per-second hazard (Poisson assumption)
    #         lambda0 = -np.log(max(1.0 - p0, 1e-12)) / 60.0
    #
    #         tL = float(row["time_low_sec"])
    #         tM = float(row["time_med_sec"])
    #         tH = float(row["time_high_sec"])
    #
    #         hazard = lambda0 * (m_low * tL + m_med * tM + m_high * tH)
    #         p_fail = 1.0 - np.exp(-hazard)
    #
    #         rows.append(dict(
    #             skill=row["skill"], component=comp,
    #             p0_per_min=p0, lambda0_per_s=lambda0,
    #             time_low_sec=tL, time_med_sec=tM, time_high_sec=tH,
    #             m_low=m_low, m_med=m_med, m_high=m_high,
    #             hazard=hazard, p_fail=p_fail
    #         ))
    #
    #     df = pd.DataFrame(rows)
    #     return df

    # def assess_failure_from_bands(self, traces, effort_mode: Optional[str] = None, beta: Optional[float] = None):
    #     """
    #     Compute P_fail(skill, component) using velocity bands, optionally modulated by effort.
    #     Parameters:
    #         effort_mode:
    #             - "off":      ignore effort completely (legacy behavior)
    #             - "modifier": use velocity band times ONLY; multiply hazard by EffortFactor
    #             - "override": active time = union(vel_active, eff_active); rescale velocity fractions to union
    #         beta:
    #             Strength of effort factor blending in "modifier"/"override":
    #                 EffortFactor = (1 - beta) + beta * ( Σ_j eff_frac_j * m_eff_j )
    #
    #     Returns DataFrame with columns:
    #         [skill, component, p0_per_min, lambda0_per_s,
    #          time_low_sec, time_med_sec, time_high_sec,
    #          m_low, m_med, m_high, eff_mode, eff_factor, union_active_time_sec,
    #          hazard, p_fail]
    #     """
    #     mode = (effort_mode or getattr(self, "default_effort_mode", "off")).lower()
    #     beta = getattr(self, "effort_beta", 1.0) if beta is None else float(beta)
    #
    #     summary = self.summarize(traces)
    #     bands = summary.get("velocity_bands", pd.DataFrame()).copy()
    #     if bands.empty:
    #         return pd.DataFrame(columns=[
    #             "skill","component","p0_per_min","lambda0_per_s",
    #             "time_low_sec","time_med_sec","time_high_sec",
    #             "m_low","m_med","m_high","eff_mode","eff_factor","union_active_time_sec",
    #             "hazard","p_fail"
    #         ])
    #
    #     etab = summary.get("effort_bands", pd.DataFrame()).copy()
    #     if etab.empty:
    #         etab = pd.DataFrame(columns=[
    #             "skill","component","eff_time_low_sec","eff_time_med_sec","eff_time_high_sec",
    #             "eff_active_time_sec","eff_frac_low","eff_frac_med","eff_frac_high"
    #         ])
    #         etab["skill"] = bands["skill"]
    #         etab["component"] = bands["component"]
    #         etab = etab.fillna(0.0)
    #
    #     tab = pd.merge(bands, etab, on=["skill","component"], how="left").fillna(0.0)
    #
    #     mL, mM, mH = self.vel_band_multipliers
    #     eL, eM, eH = getattr(self, "eff_band_multipliers", (1.0,1.0,1.0))
    #
    #     rows = []
    #     for _, r in tab.iterrows():
    #         comp = r["component"]
    #
    #         # base probability lookup with fallbacks
    #         p0 = 0.0
    #         if hasattr(self, "base_prob_per_minute"):
    #             if comp in self.base_prob_per_minute:
    #                 p0 = float(self.base_prob_per_minute[comp])
    #             elif comp.startswith("j") and ("joint" in self.base_prob_per_minute):
    #                 p0 = float(self.base_prob_per_minute["joint"])
    #             else:
    #                 p0 = float(self.base_prob_per_minute.get("default", 0.0))
    #         if p0 <= 0.0:
    #             # skip if no base rate available
    #             continue
    #
    #         lambda0 = -np.log(max(1.0 - p0, 1e-12)) / 60.0
    #
    #         # velocity exposure (no effort added here)
    #         tL = float(r.get("time_low_sec", 0.0))
    #         tM = float(r.get("time_med_sec", 0.0))
    #         tH = float(r.get("time_high_sec", 0.0))
    #         tV = float(r.get("moving_time_sec", 0.0))
    #
    #         if tV <= 0.0:
    #             rows.append(dict(
    #                 skill=r["skill"], component=comp, p0_per_min=p0, lambda0_per_s=lambda0,
    #                 time_low_sec=0.0, time_med_sec=0.0, time_high_sec=0.0,
    #                 m_low=mL, m_med=mM, m_high=mH,
    #                 eff_mode=mode, eff_factor=1.0, union_active_time_sec=max(tV, float(r.get("eff_active_time_sec", 0.0))),
    #                 hazard=0.0, p_fail=0.0
    #             ))
    #             continue
    #
    #         hazard_vel = lambda0 * (mL * tL + mM * tM + mH * tH)
    #
    #         # EffortFactor from fractions (dimensionless)
    #         eff_factor = 1.0
    #         if mode in ("modifier", "override"):
    #             eat = float(r.get("eff_active_time_sec", 0.0))
    #             if eat > 0.0:
    #                 efL = float(r.get("eff_frac_low", 0.0))
    #                 efM = float(r.get("eff_frac_med", 0.0))
    #                 efH = float(r.get("eff_frac_high", 0.0))
    #                 e_avg = (efL*eL) + (efM*eM) + (efH*eH)
    #                 eff_factor = (1.0 - beta) + beta * e_avg
    #             else:
    #                 eff_factor = 1.0
    #
    #         union_active = float(max(tV, float(r.get("eff_active_time_sec", 0.0))))
    #         if mode == "override" and union_active > 0 and tV > 0:
    #             # rescale velocity fractions to union time (no double counting)
    #             fL = tL / tV
    #             fM = tM / tV
    #             fH = tH / tV
    #             tL_u = fL * union_active
    #             tM_u = fM * union_active
    #             tH_u = fH * union_active
    #             hazard = lambda0 * (mL*tL_u + mM*tM_u + mH*tH_u) * eff_factor
    #         elif mode == "modifier":
    #             hazard = hazard_vel * eff_factor
    #         else:
    #             hazard = hazard_vel
    #             union_active = tV
    #
    #         p_fail = 1.0 - np.exp(-hazard)
    #
    #         rows.append(dict(
    #             skill=r["skill"], component=comp, p0_per_min=p0, lambda0_per_s=lambda0,
    #             time_low_sec=tL, time_med_sec=tM, time_high_sec=tH,
    #             m_low=mL, m_med=mM, m_high=mH,
    #             eff_mode=mode, eff_factor=eff_factor, union_active_time_sec=union_active,
    #             hazard=hazard, p_fail=p_fail
    #         ))
    #
    #     return pd.DataFrame(rows)

    # def assess_failure_for_gripper(self, traces):
    #     """Compute gripper p_fail(skill) using active_time_sec only."""
    #     runs_df, segments_df, components_df = self.to_frames(traces)
    #
    #     g = components_df[components_df["component"] == "gripper"]
    #     if g.empty:
    #         return pd.DataFrame(
    #             columns=["skill", "component", "active_time_sec", "p0_per_min", "lambda0_per_s", "hazard", "p_fail"])
    #
    #     agg = g.groupby(["skill", "component"], as_index=False)["active_time_sec"].sum()
    #
    #     p0 = self.base_prob_per_minute.get("gripper")
    #     if p0 is None:
    #         raise ValueError("No base probability for 'gripper' (and no 'default'). Provide it in the JSON.")
    #
    #     lambda0 = -np.log(max(1.0 - p0, 1e-12)) / 60.0
    #     agg["p0_per_min"] = p0
    #     agg["lambda0_per_s"] = lambda0
    #     agg["hazard"] = lambda0 * agg["active_time_sec"]
    #     agg["p_fail"] = 1.0 - np.exp(-agg["hazard"])
    #     return agg

    def _avg_skill_time_from_summary(self, summary) -> dict:
        import pandas as pd
        st = summary.get("skill_time", pd.DataFrame())
        if not isinstance(st, pd.DataFrame) or "skill" not in st.columns:
            raise KeyError("summary['skill_time'] missing or malformed.")
        # prefer explicit per-episode avg
        col = "avg_time_per_episode_sec"
        if col not in st.columns:
            raise KeyError(f"'skill_time' must contain '{col}'.")
        return dict(zip(st["skill"], st[col].astype(float)))

    def assess_failure_from_bands(self, traces, effort_mode: str = "modifier", beta: float = 1.0):
        """
        Exposure = avg_time_per_episode(skill) from summary['skill_time'].
        That average is allocated into velocity bands using velocity_bands' FRACTIONS.
        Effort handled via 'modifier' (default) or 'override' as a multiplicative factor from effort FRACTIONS.

        Returns DataFrame:
          [skill, component, t_avg_skill_sec,
           vel_frac_low, vel_frac_med, vel_frac_high,
           eff_frac_low, eff_frac_med, eff_frac_high,
           p0_per_min, lambda0_per_s, m_low, m_med, m_high,
           eff_mode, eff_factor, hazard, p_fail]
        """
        import numpy as np
        import pandas as pd

        mode = (effort_mode or getattr(self, "default_effort_mode", "modifier")).lower()
        beta = getattr(self, "effort_beta", 1.0) if beta is None else float(beta)

        summary = self.summarize(traces)

        # 1) avg per-episode skill times (aligns with PRISM)
        avg_skill_time = self._avg_skill_time_from_summary(summary)  # {skill: seconds}

        # 2) velocity band FRACTIONS (already computed by summarize)
        bands = summary.get("velocity_bands", pd.DataFrame()).copy()
        if bands.empty:
            return pd.DataFrame(columns=[
                "skill", "component", "t_avg_skill_sec",
                "vel_frac_low", "vel_frac_med", "vel_frac_high",
                "eff_frac_low", "eff_frac_med", "eff_frac_high",
                "p0_per_min", "lambda0_per_s", "m_low", "m_med", "m_high",
                "eff_mode", "eff_factor", "hazard", "p_fail"
            ])

        # 3) effort FRACTIONS (optional)
        effort = summary.get("effort_bands", pd.DataFrame()).copy()
        if effort.empty:
            effort = pd.DataFrame({"skill": bands["skill"], "component": bands["component"],
                                   "eff_frac_low": 0.0, "eff_frac_med": 0.0, "eff_frac_high": 0.0})
        else:
            # ensure columns exist
            for c in ("eff_frac_low", "eff_frac_med", "eff_frac_high"):
                if c not in effort.columns:
                    effort[c] = 0.0

        # Merge velocity & effort fractions
        need_cols = ["skill", "component",
                     "frac_low_of_moving", "frac_med_of_moving", "frac_high_of_moving"]
        for c in need_cols:
            if c not in bands.columns:
                bands[c] = 0.0

        tab = pd.merge(
            bands[["skill", "component", "frac_low_of_moving", "frac_med_of_moving", "frac_high_of_moving"]],
            effort[["skill", "component", "eff_frac_low", "eff_frac_med", "eff_frac_high"]],
            on=["skill", "component"], how="left"
        ).fillna(0.0)

        mL, mM, mH = self.vel_band_multipliers
        eL, eM, eH = getattr(self, "eff_band_multipliers", (1.0, 1.0, 1.0))

        rows = []
        for _, r in tab.iterrows():
            skill = r["skill"];
            comp = r["component"]

            # base probability lookup (per-minute)
            p0 = 0.0
            if hasattr(self, "base_prob_per_minute"):
                if comp in self.base_prob_per_minute:
                    p0 = float(self.base_prob_per_minute[comp])
                elif comp.startswith("j") and ("joint" in self.base_prob_per_minute):
                    p0 = float(self.base_prob_per_minute["joint"])
                else:
                    p0 = float(self.base_prob_per_minute.get("default", 0.0))
            if p0 <= 0.0:
                continue

            lambda0 = -np.log(max(1.0 - p0, 1e-12)) / 60.0  # per-second

            # avg skill time (per-episode)
            t_avg = float(avg_skill_time.get(skill, 0.0))
            if t_avg <= 0.0:
                rows.append(dict(
                    skill=skill, component=comp, t_avg_skill_sec=0.0,
                    vel_frac_low=0.0, vel_frac_med=0.0, vel_frac_high=0.0,
                    eff_frac_low=float(r.get("eff_frac_low", 0.0)),
                    eff_frac_med=float(r.get("eff_frac_med", 0.0)),
                    eff_frac_high=float(r.get("eff_frac_high", 0.0)),
                    p0_per_min=p0, lambda0_per_s=lambda0,
                    m_low=mL, m_med=mM, m_high=mH,
                    eff_mode=mode, eff_factor=1.0,
                    hazard=0.0, p_fail=0.0
                ))
                continue

            # velocity fractions (already normalized per summarize)
            fL = float(r["frac_low_of_moving"])
            fM = float(r["frac_med_of_moving"])
            fH = float(r["frac_high_of_moving"])

            # safety: clip and renormalize small drifts
            vec = np.array([max(0.0, fL), max(0.0, fM), max(0.0, fH)], dtype=float)
            s = vec.sum()
            if s > 0:
                vec /= s
            fL, fM, fH = vec.tolist()

            # allocate avg skill time into bands
            tL = t_avg * fL
            tM = t_avg * fM
            tH = t_avg * fH

            hazard_vel = lambda0 * (mL * tL + mM * tM + mH * tH)

            # effort factor from fractions
            efL = float(r.get("eff_frac_low", 0.0))
            efM = float(r.get("eff_frac_med", 0.0))
            efH = float(r.get("eff_frac_high", 0.0))

            eff_factor = 1.0
            if mode in ("modifier", "override"):
                sum_eff = efL + efM + efH
                if sum_eff > 1e-9:
                    # normalize fractions just in case
                    efL, efM, efH = efL / sum_eff, efM / sum_eff, efH / sum_eff
                    e_avg = (efL * eL) + (efM * eM) + (efH * eH)
                    eff_factor = (1.0 - beta) + beta * e_avg
                else:
                    # no effort data -> neutral factor
                    eff_factor = 1.0

            # 'override' keeps the same semantics (multiplicative on the per-episode allocation)
            hazard = hazard_vel * (eff_factor if mode in ("modifier", "override") else 1.0)
            p_fail = 1.0 - np.exp(-hazard)

            rows.append(dict(
                skill=skill, component=comp, t_avg_skill_sec=t_avg,
                vel_frac_low=fL, vel_frac_med=fM, vel_frac_high=fH,
                eff_frac_low=efL, eff_frac_med=efM, eff_frac_high=efH,
                p0_per_min=p0, lambda0_per_s=lambda0,
                m_low=mL, m_med=mM, m_high=mH,
                eff_mode=mode, eff_factor=eff_factor,
                hazard=hazard, p_fail=p_fail
            ))

        return pd.DataFrame(rows)

    def assess_failure_for_gripper(self, traces):
        """
        P_fail(skill, 'gripper') using avg per-episode gripper active time from summary['comp_usage'].
        This aligns FT exposure with PRISM rewards (per-episode averages).
        """
        import numpy as np
        import pandas as pd

        summary = self.summarize(traces)
        cu = summary.get("comp_usage", pd.DataFrame()).copy()
        if cu.empty:
            return pd.DataFrame(columns=[
                "skill", "component", "avg_active_time_per_episode_sec",
                "p0_per_min", "lambda0_per_s", "hazard", "p_fail"
            ])

        g = cu[cu["component"] == "gripper"]
        if g.empty:
            return pd.DataFrame(columns=[
                "skill", "component", "avg_active_time_per_episode_sec",
                "p0_per_min", "lambda0_per_s", "hazard", "p_fail"
            ])

        # comp_usage already gives per-episode mean active time
        out = g[["skill", "component", "avg_active_time_sec"]].rename(
            columns={"avg_active_time_sec": "avg_active_time_per_episode_sec"}
        ).copy()

        # base rate
        p0 = None
        if hasattr(self, "base_prob_per_minute"):
            p0 = self.base_prob_per_minute.get("gripper", self.base_prob_per_minute.get("default"))
        if p0 is None:
            raise ValueError("No base probability for 'gripper'. Provide it in base_prob_per_minute.")

        lambda0 = -np.log(max(1.0 - float(p0), 1e-12)) / 60.0
        out["p0_per_min"] = float(p0)
        out["lambda0_per_s"] = lambda0
        out["hazard"] = lambda0 * out["avg_active_time_per_episode_sec"].astype(float)
        out["p_fail"] = 1.0 - np.exp(-out["hazard"])
        return out[["skill", "component", "avg_active_time_per_episode_sec", "p0_per_min", "lambda0_per_s", "hazard",
                    "p_fail"]]

    def load_base_probs_from_json(self, path: str, *, add_defaults: bool = True):
        """
        Reads a JSON like franka_config.json and returns a dict of per-minute base probabilities:
            {"j1": p0, ..., "j7": p0, "gripper": p0, "controller": p0, "power_supply": p0,
             "sensors": p0, "camera": p0, "joint": median_joint_p0, "default": median_joint_p0}
        Also saves:
            self.hw_config   -> full parsed JSON
            self.redundancy  -> {component -> bool}
            self.base_prob_per_minute -> the returned dict
        """
        with open(path, "r") as f:
            cfg = json.load(f)
        comps = cfg.get("components", {})

        base = {}
        redundancy = {}

        # Joints: "Joint_1".."Joint_7" -> "j1".."j7"
        for i in range(1, 8):
            k = f"Joint_{i}"
            if k in comps:
                spec = comps[k]
                if "failure_probability" in spec:
                    base[f"j{i}"] = float(spec["failure_probability"])
                redundancy[f"j{i}"] = bool(spec.get("redundancy", False))

        # Other components with canonical keys
        mapping = {
            "Gripper": "gripper",
            "Controller": "controller",
            "Power_Supply": "power_supply",
            "Sensors": "sensors",
            "Camera": "camera",
        }
        for json_key, canon in mapping.items():
            if json_key in comps:
                spec = comps[json_key]
                if "failure_probability" in spec:
                    base[canon] = float(spec["failure_probability"])
                redundancy[canon] = bool(spec.get("redundancy", False))

        # Keep everything for FT logic
        self.hw_config = cfg
        self.redundancy = redundancy

        # Save for reuse
        self.base_prob_per_minute = base
        return base


# # 1) Load H5 pieces
# with h5py.File("/Users/Phips1900/PhD/Research/RelAIBotiX/datasets/franka_test_100.h5", "r") as f:
#     features_arr = f["features"][()]  # (N, 22)
#     labels = f["labels"][()]  # (N,)
#     timestamps = f["timestamps"][()]  # (N,)
#     feat_names = [n.decode() if hasattr(n, "decode") else str(n)
#                   for n in f["features"].attrs["feature_names"]]
#
# features_df = pd.DataFrame(features_arr)
#
# # 2) Load CSV with goal/final per run
# trials_csv = pd.read_csv(
#     "/Users/Phips1900/PhD/Research/RelAIBotiX/datasets/franka_trial_summary_100.csv")  # must have goal_x/y/z, final_x/y/z
#
# # 3) Analyze
# an = BehavioralAnalyzer(pos_success_tol=0.02)  # tune eps_abs/dc_thr/rms_thr/range_thr later if needed
# traces: List[RunTrace] = an.analyze(
#     features=features_df,
#     feature_names=feat_names,
#     labels=labels,
#     timestamps=timestamps,
#     trials_csv=trials_csv
# )
#
# print(len(traces), "runs")
# print(traces[0].skill_sequence)
# print(traces[3].success, traces[3].pos_error_norm)
# for se in traces[8].segments:
#     print(se.idx, se.name, se.duration)
#
# totals, skill_time, comp_active, comp_active_by_skill = an.summarize_timings(traces)
#
# print(totals)                 # {'total_run_time_sec': ..., 'n_runs': 10}
# print(skill_time.head())      # per-skill totals + averages
# print(comp_active.sort_values("total_active_time_sec", ascending=False).head())
# print(comp_active_by_skill.head())
#
# summ = an.summarize(traces)
#
# print(summ["sequences"])      # counts + %
# print(summ["overall"])        # n_runs, total time, success %
# print(summ["skill_time"])     # per-skill totals/averages
# print(summ["comp_usage"]      # active % and times per skill×component
#       .sort_values(["skill","active_pct_episodes"], ascending=[True, False])
#       .head(20))
# print(summ["joint_velocity"]  # velocity peaks per joint×skill
#       .sort_values(["skill","total_vel_absmax"], ascending=[True, False])
#       .head(20))
#
# base_p = an.load_base_probs_from_json("/Users/Phips1900/PhD/Research/RelAIBotiX/config_files/franka_config.json")
#
# joint_failure_table = an.assess_failure_from_bands(traces)
# print(joint_failure_table.sort_values("p_fail", ascending=False).head(10))
# gripper_table = an.assess_failure_for_gripper(traces)
#
# ft = joint_failure_table.copy()
# num_cols = ft.select_dtypes(include="number").columns
# ft[num_cols] = ft[num_cols].round(10)   # or fewer decimals
# ft.to_csv("failure_table.csv", index=False)

