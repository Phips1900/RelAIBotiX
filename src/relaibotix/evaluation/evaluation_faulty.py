#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluation_faulty.py

Evaluate success on normal and faulty episodes.

- Episodes are taken from the per-sample "labels" (episode ids).
- Success is computed from the last (cx, cy) sample vs target (0.20, -0.15)
  with tolerance 0.02 using max-abs error in XY.
- Faulty episodes are those where any of:
    noise_strength, blur_strength, brightness_strength
  is > 0 (episode-wise max).
  Subtypes:
    - noise_only      (noise>0, blur==0, brightness==0)
    - blur_only       (blur>0, noise==0, brightness==0)
    - brightness_only (brightness>0, noise==0, blur==0)
    - combined        (at least two > 0)
- Produces:
    1) runs.csv  : per-episode table (id, sizes, success, strengths, group, etc.)
    2) report.pdf: summary tables + joint velocity distribution plots by group
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import re

# ----------------------- CONFIG -----------------------
H5_PATH = "/Users/Phips1900/PhD/Research/RelAIBotiX/datasets/IL/difussion/eval_diffusion_mj2_faults_.h5"
OUT_DIR = "/Users/Phips1900/PhD/Research/RelAIBotiX/artifacts/reports/diffusion/fault"  # folder for runs.csv and report.pdf
TOL = 0.02
TARGET_XY = (0.20, -0.15)


# ------------------------------------------------------


# -------------------- utilities -----------------------

def get_col_index(names: List[str], key: str) -> Optional[int]:
    try:
        return names.index(key)
    except ValueError:
        return None


def success_from_cxcy(X_run: np.ndarray, feat_names: List[str],
                      tol: float, target_xy: Tuple[float, float]):
    """
    Compute success for a single episode from the LAST cx, cy sample.
    Returns dict {success, pos_err, goal, final, source} or None if cx/cy missing.
    """
    if X_run is None or len(X_run) == 0 or not feat_names:
        return None
    i_cx = get_col_index(feat_names, "cx")
    i_cy = get_col_index(feat_names, "cy")
    if i_cx is None or i_cy is None:
        return None

    cx_last = float(X_run[-1, i_cx])
    cy_last = float(X_run[-1, i_cy])
    gx, gy = map(float, target_xy)
    dx, dy = gx - cx_last, gy - cy_last
    max_abs_xy = max(abs(dx), abs(dy))
    return {
        "success": (max_abs_xy <= tol),
        "pos_err": float(max_abs_xy),
        "goal": (gx, gy),
        "final": (cx_last, cy_last),
        "source": "features_cxcy_last",
    }


def contiguous_episode_bounds(ep_labels: np.ndarray) -> List[Tuple[int, int, int]]:
    """
    Turn per-sample episode labels into (run_id, start_idx, end_idx) for each contiguous block
    with a valid (finite, >=0) episode id.
    """
    ep = np.asarray(ep_labels).reshape(-1)
    n = ep.size
    bounds: List[Tuple[int, int, int]] = []
    i = 0

    def valid(k):
        return (k < n) and np.isfinite(ep[k]) and (ep[k] >= 0)

    while i < n:
        while i < n and not valid(i):
            i += 1
        if i >= n:
            break
        s = i
        rid = int(ep[i])
        i += 1
        while i < n and valid(i) and int(ep[i]) == rid:
            i += 1
        e = i - 1
        bounds.append((rid, s, e))
    return bounds


def classify_perturbation_group(strengths: Dict[str, float]) -> str:
    """
    strengths keys: 'noise', 'blur', 'brightness' (episode-wise max)
    Returns one of: 'normal', 'noise_only', 'blur_only', 'brightness_only', 'combined'
    """
    nz = {k: (v > 0.0) for k, v in strengths.items()}
    nnz = sum(1 for v in nz.values() if v)
    if nnz == 0:
        return "normal"
    if nnz >= 2:
        return "combined"
    if nz.get("noise", False):
        return "noise_only"
    if nz.get("blur", False):
        return "blur_only"
    if nz.get("brightness", False):
        return "brightness_only"
    return "combined"


def gather_joint_velocity_columns(feat_names: List[str]) -> List[int]:
    """
    Return column indices for joint_vel_1, joint_vel_2, ...
    """
    cols = []
    for i, n in enumerate(feat_names):
        if re.fullmatch(r"joint_vel_\d+", n):
            cols.append(i)
    return cols


def plot_joint_velocity_distributions(pdf: PdfPages,
                                      feat_names: List[str],
                                      episodes_by_group: Dict[str, List[np.ndarray]],
                                      title_prefix: str = "Joint velocity |v| distributions"):
    """
    For each group, collect all |joint_vel_k| samples across its episodes and
    plot a boxplot per joint (one page per group).
    """
    jcols = gather_joint_velocity_columns(feat_names)
    jnames = [feat_names[i] for i in jcols]
    if not jcols:
        # nothing to plot
        fig, ax = plt.subplots(figsize=(10, 1.5))
        ax.text(0.5, 0.5, "No joint_vel_* columns found.", ha="center", va="center")
        ax.axis("off")
        pdf.savefig(fig)
        plt.close(fig)
        return

    for group, episodes in episodes_by_group.items():
        per_joint_vals = []
        for j_idx in jcols:
            vals = []
            for X_run in episodes:
                if X_run.size == 0:
                    continue
                v = np.abs(X_run[:, j_idx])
                v = v[np.isfinite(v)]
                if v.size:
                    vals.append(v)
            if vals:
                per_joint_vals.append(np.concatenate(vals, axis=0))
            else:
                per_joint_vals.append(np.array([], dtype=float))

        fig, ax = plt.subplots(figsize=(max(10, 1.0 * len(jcols)), 6))
        data = [v if v.size > 0 else np.array([np.nan]) for v in per_joint_vals]
        ax.boxplot(data, showfliers=False)
        ax.set_title(f"{title_prefix} — {group}")
        ax.set_xticks(np.arange(1, len(jnames) + 1))
        ax.set_xticklabels(jnames, rotation=45, ha="right")
        ax.set_ylabel("|velocity| (rad/s)")
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def dataframe_page_to_pdf(pdf: PdfPages, df: pd.DataFrame, title: str, max_rows: int = 40):
    """
    Render a small/medium dataframe as text on a PDF page (paginate if long).
    """
    if df.empty:
        fig, ax = plt.subplots(figsize=(8.5, 1.5))
        ax.text(0.5, 0.5, f"{title}\n(empty)", ha="center", va="center")
        ax.axis("off")
        pdf.savefig(fig)
        plt.close(fig)
        return

    start = 0
    while start < len(df):
        end = min(start + max_rows, len(df))
        chunk = df.iloc[start:end]
        fig, ax = plt.subplots(figsize=(11, 8.5))  # landscape-ish
        ax.axis("off")
        ax.set_title(title if start == 0 else f"{title} (cont.)", loc="left", fontsize=12, pad=12)
        text = chunk.to_string(index=False)
        ax.text(0.01, 0.98, text, fontsize=8, va="top", family="monospace")
        pdf.savefig(fig)
        plt.close(fig)
        start = end


def summarize_joint_velocities(feat_names: List[str],
                               episodes_by_group: Dict[str, List[np.ndarray]],
                               groups: List[str] = ("normal", "noise_only", "blur_only",
                                                    "brightness_only")) -> pd.DataFrame:
    """
    Returns a tidy table with per-group, per-joint mean|v| and max|v|.
    Columns: [group, joint, mean_abs_vel, max_abs_vel, n_samples]
    """
    jcols = gather_joint_velocity_columns(feat_names)
    jnames = [feat_names[i] for i in jcols]
    rows = []
    for g in groups:
        runs = episodes_by_group.get(g, [])
        if not runs:
            for jn in jnames:
                rows.append(dict(group=g, joint=jn, mean_abs_vel=np.nan, max_abs_vel=np.nan, n_samples=0))
            continue

        for j_idx, jn in zip(jcols, jnames):
            vals = []
            for X in runs:
                if X.size == 0:
                    continue
                v = np.abs(X[:, j_idx])
                v = v[np.isfinite(v)]
                if v.size:
                    vals.append(v)
            if vals:
                v_all = np.concatenate(vals, axis=0)
                rows.append(dict(group=g, joint=jn,
                                 mean_abs_vel=float(np.mean(v_all)),
                                 max_abs_vel=float(np.max(v_all)),
                                 n_samples=int(v_all.size)))
            else:
                rows.append(dict(group=g, joint=jn, mean_abs_vel=np.nan, max_abs_vel=np.nan, n_samples=0))
    return pd.DataFrame(rows)


def plot_joint_velocity_summary(stats: pd.DataFrame,
                                out_png_path: Path,
                                pdf: Optional[PdfPages] = None,
                                title: str = "Per-joint velocity summary (|v|): groups in one plot"):
    """
    One figure, two panels: left = mean |v|, right = max |v|, grouped by joint with bars per condition.
    Saves PNG and optionally appends to an open PdfPages.
    """
    if stats.empty:
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.axis("off")
        ax.text(0.5, 0.5, "No joint velocity columns found.", ha="center", va="center")
        if pdf: pdf.savefig(fig)
        fig.savefig(out_png_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    groups = ["normal", "noise_only", "blur_only", "brightness_only"]
    stats = stats[stats["group"].isin(groups)].copy()
    joints = list(stats["joint"].dropna().unique())
    if not joints:
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.axis("off")
        ax.text(0.5, 0.5, "No joints in stats.", ha="center", va="center")
        if pdf: pdf.savefig(fig)
        fig.savefig(out_png_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    mean_mat = np.zeros((len(groups), len(joints)), dtype=float) * np.nan
    max_mat = np.zeros_like(mean_mat)
    for gi, g in enumerate(groups):
        sg = stats[stats["group"] == g]
        mrow = sg.set_index("joint")["mean_abs_vel"].reindex(joints)
        xrow = sg.set_index("joint")["max_abs_vel"].reindex(joints)
        mean_mat[gi, :] = mrow.to_numpy()
        max_mat[gi, :] = xrow.to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(max(12, 1.2 * len(joints)), 5), sharey=False)
    bar_w = 0.18
    x = np.arange(len(joints))
    offsets = np.linspace(-1.5 * bar_w, 1.5 * bar_w, num=len(groups))

    def _bars(ax, mat, ylabel, title_suffix):
        for gi, g in enumerate(groups):
            ax.bar(x + offsets[gi], mat[gi, :], width=bar_w, label=g.replace("_", " "))
        ax.set_xticks(x)
        ax.set_xticklabels(joints, rotation=40, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title_suffix}")
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax.legend(frameon=False, fontsize=9)

    _bars(axes[0], mean_mat, r"mean $|v|$ (rad/s)", "Mean")
    _bars(axes[1], max_mat, r"max $|v|$ (rad/s)", "Max")

    fig.suptitle(title, y=1.02, fontsize=12)
    fig.tight_layout()
    if pdf:
        pdf.savefig(fig)
    fig.savefig(out_png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------- main --------------------------

def run_evaluation_faulty(h5_path: str,
                          out_dir: str,
                          tol: float = 0.02,
                          target_xy: Tuple[float, float] = (0.20, -0.15)):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_csv = out_dir / "reliabotix_eval.csv"
    report_pdf = out_dir / "reliabotix_eval.pdf"

    # ---- Load data
    with h5py.File(h5_path, "r") as f:
        features = f["features"][()]  # (N, D)
        episode_labels = f["labels"][()]  # (N,)
        timestamps = f["timestamps"][()]  # (N,)
        raw_names = f["features"].attrs["feature_names"]
        feat_names = [n.decode() if hasattr(n, "decode") else str(n) for n in raw_names]

    N, D = features.shape
    print(f"Loaded: features={features.shape}, labels={episode_labels.shape}, timestamps={timestamps.shape}")

    # ---- Column indices for perturbations and cx/cy
    i_noise = get_col_index(feat_names, "noise_strength")
    i_blur = get_col_index(feat_names, "blur_strength")
    i_brightness = get_col_index(feat_names, "brightness_strength")

    # ---- Episode bounds
    bounds = contiguous_episode_bounds(episode_labels)
    if not bounds:
        raise RuntimeError("No valid episodes found in labels (need non-negative, contiguous ids).")
    print(f"Detected {len(bounds)} episodes.")

    # ---- Per-episode loop
    rows = []
    groups_to_Xruns: Dict[str, List[np.ndarray]] = {g: [] for g in
                                                    ["normal", "noise_only", "blur_only", "brightness_only",
                                                     "combined"]}
    groups_to_Xruns["faulty_any"] = []

    for (run_id, i0, i1) in bounds:
        X_run = features[i0:i1 + 1, :]
        # strengths: use episode-wise MAX (robust if values vary during the episode)
        noise = float(np.nanmax(X_run[:, i_noise])) if i_noise is not None else float("nan")
        blur = float(np.nanmax(X_run[:, i_blur])) if i_blur is not None else float("nan")
        bright = float(np.nanmax(X_run[:, i_brightness])) if i_brightness is not None else float("nan")

        strengths = {
            "noise": 0.0 if np.isnan(noise) else noise,
            "blur": 0.0 if np.isnan(blur) else blur,
            "brightness": 0.0 if np.isnan(bright) else bright,
        }
        group = classify_perturbation_group(strengths)

        # success from last (cx, cy)
        sres = success_from_cxcy(X_run, feat_names, tol=tol, target_xy=target_xy)
        if sres is None:
            success = False
            pos_err = float("nan")
            goal_xy = (float("nan"), float("nan"))
            final_xy = (float("nan"), float("nan"))
        else:
            success = bool(sres["success"])
            pos_err = float(sres["pos_err"])
            goal_xy = tuple(sres["goal"])
            final_xy = tuple(sres["final"])

        n_samples = int(X_run.shape[0])
        dur_sec = float(timestamps[i1] - timestamps[i0]) if i1 > i0 else 0.0

        rows.append(dict(
            run_id=int(run_id),
            start_idx=int(i0), end_idx=int(i1),
            n_samples=n_samples, duration_sec=dur_sec,
            success=success, pos_err=pos_err,
            goal_x=goal_xy[0], goal_y=goal_xy[1],
            final_x=final_xy[0], final_y=final_xy[1],
            noise_strength_max=strengths["noise"],
            blur_strength_max=strengths["blur"],
            brightness_strength_max=strengths["brightness"],
            group=group
        ))

        # collect episodes per plotting group
        if group in groups_to_Xruns:
            groups_to_Xruns[group].append(X_run)
        if group != "normal":
            groups_to_Xruns["faulty_any"].append(X_run)

    runs_df = pd.DataFrame(rows).sort_values("run_id").reset_index(drop=True)

    # ---- Compute success rates
    def succ_rate(df):
        if df.empty:
            return np.nan
        return float(np.mean(df["success"].astype(float)))

    subsets = {
        "normal": runs_df[runs_df["group"] == "normal"],
        "faulty_any": runs_df[runs_df["group"] != "normal"],
        "noise_only": runs_df[runs_df["group"] == "noise_only"],
        "blur_only": runs_df[runs_df["group"] == "blur_only"],
        "brightness_only": runs_df[runs_df["group"] == "brightness_only"],
        "combined": runs_df[runs_df["group"] == "combined"],
    }

    summary_rows = []
    for name, df in subsets.items():
        summary_rows.append(dict(
            subset=name,
            n_episodes=int(len(df)),
            success_rate=float("nan") if df.empty else round(100.0 * succ_rate(df), 2),
            mean_pos_err=float(df["pos_err"].mean()) if not df.empty else float("nan"),
            median_pos_err=float(df["pos_err"].median()) if not df.empty else float("nan"),
            avg_duration_sec=float(df["duration_sec"].mean()) if not df.empty else float("nan"),
            avg_n_samples=float(df["n_samples"].mean()) if not df.empty else float("nan"),
        ))
    summary_df = pd.DataFrame(summary_rows).sort_values("subset").reset_index(drop=True)

    # ---- Save CSV
    runs_df.to_csv(runs_csv, index=False)
    print(f"Wrote per-episode table → {runs_csv}")

    # ---- Build PDF
    with PdfPages(report_pdf) as pdf:
        # Cover page
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.text(0.02, 0.95, "Evaluation Report — Normal vs Faulty Episodes", fontsize=16, weight="bold", va="top")
        ax.text(0.02, 0.88, f"H5: {h5_path}", fontsize=10, va="top")
        ax.text(0.02, 0.84, f"Episodes: {len(runs_df)}", fontsize=10, va="top")
        ax.text(0.02, 0.80, f"Success tolerance (XY max-abs): {tol} m; Target: {target_xy}", fontsize=10, va="top")
        ax.text(0.02, 0.76, "Faulty flags: noise_strength, blur_strength, brightness_strength (episode max > 0)",
                fontsize=10, va="top")
        pdf.savefig(fig);
        plt.close(fig)

        # Summary table
        dataframe_page_to_pdf(pdf, summary_df, "Summary — success rates & episode stats", max_rows=40)

        # Sample of runs_df (head + tail)
        head = runs_df.head(30).copy()
        tail = runs_df.tail(30).copy()
        dataframe_page_to_pdf(pdf, head, "Per-episode (HEAD)")
        dataframe_page_to_pdf(pdf, tail, "Per-episode (TAIL)")

        # Joint velocity distributions per group
        groups_for_plots = ["normal", "noise_only", "blur_only", "brightness_only", "combined"]
        episodes_by_group = {g: groups_to_Xruns.get(g, []) for g in groups_for_plots}
        plot_joint_velocity_distributions(pdf, feat_names, episodes_by_group)

        stats = summarize_joint_velocities(feat_names, episodes_by_group)
        vel_png = out_dir / "vel_summary.png"
        plot_joint_velocity_summary(stats, vel_png, pdf=pdf,
                                    title="Per-joint |velocity|: nominal vs. noise/blur/brightness")

    print(f"Wrote PDF report → {report_pdf}")
