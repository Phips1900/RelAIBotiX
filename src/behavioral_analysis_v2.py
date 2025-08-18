from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import re
import h5py


@dataclass
class ComponentMetrics:
    active: bool
    metrics: Dict[str, float]  # e.g. {"j3.tau_p95": 9.1, "gripper.pos_duty": 0.66}


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


SKILL_ID_TO_NAME = {0: "Init", 1: "Move", 2: "Pick", 3: "Carry", 4: "Place"}


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

    def analyze(
            self,
            *,
            features: pd.DataFrame,
            feature_names: List[str],
            labels: np.ndarray,
            timestamps: np.ndarray,
            trials_csv: pd.DataFrame
    ) -> List[RunTrace]:
        # build component mapping from feature_names (auto)
        component_cols = self._infer_component_cols(feature_names)

        # detect runs using last-4 rule
        run_bounds = self._detect_runs(labels)

        traces: List[RunTrace] = []

        for run_id, (i0, i1) in enumerate(run_bounds):
            # segment segments via run-length on labels slice
            segments = self._segment_skills(labels[i0:i1 + 1], timestamps[i0:i1 + 1], offset=i0)

            # per-episode component metrics
            for se in segments:
                se.components = self._component_metrics(features, component_cols, timestamps, se.start_idx, se.end_idx)

            # success from CSV
            goal = tuple(trials_csv.loc[run_id, ["x_plan", "y_plan", ]].astype(float))
            final = tuple(trials_csv.loc[run_id, ["x_real", "y_real", ]].astype(float))
            dx = float(goal[0] - final[0])
            dy = float(goal[1] - final[1])
            max_abs_xy = max(abs(dx), abs(dy))
            success = (max_abs_xy <= self.pos_success_tol)
            pos_err = max_abs_xy
            #pos_err = float(np.linalg.norm(np.array(final) - np.array(goal)))
            #success = pos_err <= self.pos_success_tol

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
        for idx, n in enumerate(names):
            s = n.lower()
            # joints
            m = re.match(r"joint_(pos|vel)_(\d+)$", s)
            if m:
                prop = m.group(1)  # 'pos' or 'vel'
                jid = f"j{int(m.group(2))}"
                comp.setdefault(jid, {})[prop] = idx
                continue
            # gripper
            if s == "gripper_state":
                comp.setdefault("gripper", {})["state"] = idx
        return comp

    def _detect_runs(self, skills: np.ndarray) -> List[Tuple[int, int]]:
        """
        Start at first 0/1 after previous run, end at the LAST 4 before the next 0/1 (or EOF).
        """
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

    def _component_metrics(
            self,
            feats: pd.DataFrame,
            comp_cols: Dict[str, Dict[str, int]],
            ts: np.ndarray,  # NEW: timestamps array (shape N,)
            i0: int,
            i1: int
    ) -> Dict[str, ComponentMetrics]:
        seg = feats.iloc[i0:i1 + 1]
        tseg = ts[i0:i1 + 1]
        out: Dict[str, ComponentMetrics] = {}

        # helper: sum dt where mask at interval start is True
        def active_seconds(mask: np.ndarray, t: np.ndarray) -> float:
            if mask.size <= 1:
                return 0.0
            dt = np.diff(t)
            return float(np.sum(dt[mask[:-1]]))

        for comp, props in comp_cols.items():
            m: Dict[str, float] = {}
            active_any = False
            comp_mask = np.zeros(len(seg), dtype=bool)  # per-sample activity mask (OR of all props)

            for prop, col in props.items():
                arr = seg.iloc[:, col].to_numpy(dtype=float)
                arr = arr[np.isfinite(arr)]
                if arr.size == 0:
                    continue

                prop_l = prop.lower()
                comp_l = comp.lower()

                # --- common aggregates ---
                mean = float(np.mean(arr))
                std = float(np.std(arr))
                rms = float(np.sqrt(np.mean(arr ** 2)))
                mx = float(np.max(arr))
                rng = float(np.max(arr) - np.min(arr))
                duty = float(np.mean(np.abs(arr) > getattr(self, "eps_abs", 1e-3)))

                m[f"{comp}.{prop}_mean"] = mean
                m[f"{comp}.{prop}_std"] = std
                m[f"{comp}.{prop}_rms"] = rms
                m[f"{comp}.{prop}_max"] = mx
                m[f"{comp}.{prop}_range"] = rng
                m[f"{comp}.{prop}_duty"] = duty

                # build a per-sample activity mask for this property
                prop_mask = np.zeros_like(comp_mask)

                # --- velocity-specific extras & mask ---
                if "vel" in prop_l:
                    absmax = float(np.max(np.abs(arr)))
                    m[f"{comp}.{prop}_absmax"] = absmax  # magnitude peak

                    eps_vel = getattr(self, "eps_vel", 0.03)  # rad/s
                    rms_vel_thr = getattr(self, "rms_vel_thr", 0.05)  # rad/s
                    dc_thr = getattr(self, "dc_thr", 0.2)

                    duty_vel = float(np.mean(np.abs(arr) > eps_vel))
                    m[f"{comp}.{prop}_duty_vel"] = duty_vel

                    # velocity mask: above eps_vel
                    prop_mask = (np.abs(arr) > eps_vel)

                    # episode-level activation
                    active_any = active_any or (duty_vel >= dc_thr) or (rms >= rms_vel_thr)

                # --- position: range criterion + per-sample step mask ---
                elif "pos" in prop_l:
                    range_pos_thr = getattr(self, "range_pos_thr", 0.02)  # rad
                    step_thr = getattr(self, "eps_step_pos", 0.001)  # rad / sample (≈10ms at 100Hz)

                    # per-sample |Δpos|
                    steps = np.r_[0.0, np.abs(np.diff(arr))]
                    prop_mask = (steps > step_thr)

                    active_any = active_any or (rng >= range_pos_thr)

                # --- torque/effort: simple amplitude/rms mask ---
                elif prop_l in ("tau", "torque", "effort"):
                    eps_eff = getattr(self, "eps_effort", getattr(self, "eps_abs", 1e-3))
                    prop_mask = (np.abs(arr) > eps_eff)
                    active_any = active_any or (rms >= getattr(self, "rms_thr", 1e-2))

                # --- special: gripper state in [0,1] (0=closed, 1=open) ---
                if comp_l == "gripper" and prop_l in ("state", "pos"):
                    state = np.clip(arr, 0.0, 1.0)
                    close_thr = getattr(self, "gripper_close_thresh", 0.05)
                    step_thr = getattr(self, "gripper_step_thr", 0.05)
                    mid = getattr(self, "gripper_transition_mid", 0.5)

                    closed_mask = (state <= close_thr)
                    # transitions: open<->closed edges when |Δstate| > step_thr
                    trans = np.abs(np.diff(state)) > step_thr
                    trans_mask = np.r_[False, trans]

                    # gripper "used" whenever closed OR transitioning
                    gripper_mask = closed_mask | trans_mask
                    prop_mask = gripper_mask

                    closed_frac = float(np.mean(closed_mask))
                    transitions = int(np.sum((state[:-1] > mid) != (state[1:] > mid)))

                    m[f"{comp}.{prop}_closed_frac"] = closed_frac
                    m[f"{comp}.{prop}_transitions"] = transitions

                    # episode-level activation logic for gripper
                    gripper_used = (closed_frac > 0.0) or (transitions >= 1)
                    active_any = active_any or gripper_used

                # combine into component-level mask
                comp_mask |= prop_mask

                # generic fallback (for properties that didn't hit a branch)
                if not (comp_l == "gripper" and prop_l in ("state", "pos")) and prop_l not in ("vel", "pos", "tau",
                                                                                               "torque", "effort"):
                    active_any = active_any or (
                            (duty >= getattr(self, "dc_thr", 0.2)) or
                            (rms >= getattr(self, "rms_thr", 1e-2)) or
                            (rng >= getattr(self, "range_thr", 1e-2))
                    )

            # --- active time (seconds) & fraction of episode ---
            if tseg.size >= 2:
                comp_active_sec = active_seconds(comp_mask, tseg)
                ep_dur = float(tseg[-1] - tseg[0]) if tseg[-1] > tseg[0] else 0.0
                comp_active_frac = (comp_active_sec / ep_dur) if ep_dur > 0 else 0.0
            else:
                comp_active_sec = 0.0
                comp_active_frac = 0.0

            # If component isn't active by episode-level rule, force time to 0 (your requirement)
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

        return {
            "sequences": seq_counts,
            "overall": overall,
            "skill_time": skill_time,
            "comp_usage": comp_usage,
            "joint_velocity": joint_velocity,
        }


# 1) Load H5 pieces
with h5py.File("/Users/Phips1900/PhD/Research/RelAIBotiX/datasets/franka_test_100.h5", "r") as f:
    features_arr = f["features"][()]  # (N, 22)
    labels = f["labels"][()]  # (N,)
    timestamps = f["timestamps"][()]  # (N,)
    feat_names = [n.decode() if hasattr(n, "decode") else str(n)
                  for n in f["features"].attrs["feature_names"]]

features_df = pd.DataFrame(features_arr)

# 2) Load CSV with goal/final per run
trials_csv = pd.read_csv(
    "/Users/Phips1900/PhD/Research/RelAIBotiX/datasets/franka_trial_summary_100.csv")  # must have goal_x/y/z, final_x/y/z

# 3) Analyze
an = BehavioralAnalyzer(pos_success_tol=0.02)  # tune eps_abs/dc_thr/rms_thr/range_thr later if needed
traces: List[RunTrace] = an.analyze(
    features=features_df,
    feature_names=feat_names,
    labels=labels,
    timestamps=timestamps,
    trials_csv=trials_csv
)

print(len(traces), "runs")
print(traces[0].skill_sequence)
print(traces[3].success, traces[3].pos_error_norm)
for se in traces[8].segments:
    print(se.idx, se.name, se.duration)

totals, skill_time, comp_active, comp_active_by_skill = an.summarize_timings(traces)

print(totals)                 # {'total_run_time_sec': ..., 'n_runs': 10}
print(skill_time.head())      # per-skill totals + averages
print(comp_active.sort_values("total_active_time_sec", ascending=False).head())
print(comp_active_by_skill.head())

summ = an.summarize(traces)

print(summ["sequences"])      # counts + %
print(summ["overall"])        # n_runs, total time, success %
print(summ["skill_time"])     # per-skill totals/averages
print(summ["comp_usage"]      # active % and times per skill×component
      .sort_values(["skill","active_pct_episodes"], ascending=[True, False])
      .head(20))
print(summ["joint_velocity"]  # velocity peaks per joint×skill
      .sort_values(["skill","total_vel_absmax"], ascending=[True, False])
      .head(20))

