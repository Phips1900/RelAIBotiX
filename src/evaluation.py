# evaluation.py
from typing import Dict, List, Optional, Union
from pathlib import Path
import numpy as np
import pandas as pd
import copy
import re
import matplotlib.pyplot as plt

from reliability_models import HybridReliabilityModel, MarkovChain, FaultTree
from relaibotix_helper import *


# ----- Pull per-skill FT (top-event) probabilities -----
def skill_failure_probs_from_fts(hybrid_model: HybridReliabilityModel) -> Dict[str, float]:
    """
    Returns {skill: p_fail_top_event}. Assumes your hybrid_solver already solved the FTs and
    wrote probs back onto the FaultTree objects. If not, you can re-build ft_dict and solve again.
    """
    out = {}
    for ft in hybrid_model.get_fault_trees():
        out[ft.get_skill()] = float(ft.get_top_event_failure_prob())
    return out


# ----- Task success rate from analyzer summary -----
def task_success_rate_from_summary(summary: Dict[str, pd.DataFrame]) -> float:
    if "overall" not in summary or "success_rate_percent" not in summary["overall"].columns:
        return 0.0
    return float(summary["overall"]["success_rate_percent"].iloc[0])


def plot_skill_failures_separate(
    ft_failure: Dict[str, float],
    absorb_failure: Dict[str, float],
    out_ft: Union[str, Path],
    out_absorb: Union[str, Path],
    *,
    order: Optional[list] = None,
    title_ft: str = "Per-skill failure probability (FT top-event)",
    title_abs: str = "Per-skill failure probability (Absorbing / DTMC)",
) -> tuple[Path, Path]:
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    # Common skill order
    all_skills = set(ft_failure) | set(absorb_failure)
    if order:
        skills = [s for s in order if s in all_skills] + [s for s in all_skills if s not in order]
    else:
        skills = sorted(all_skills)

    # FT-only
    ft_vals = [float(ft_failure.get(s, np.nan)) for s in skills]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(skills, ft_vals)
    ax.set_ylabel("P(failure)")
    ax.set_title(title_ft)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    #ax.set_xticklabels(skills, rotation=20, ha="right")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()
    out_ft = Path(out_ft); fig.savefig(out_ft, bbox_inches="tight", dpi=150); plt.close(fig)

    # Absorbing-only
    ab_vals = [float(absorb_failure.get(s, np.nan)) for s in skills]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(skills, ab_vals)
    ax.set_ylabel("P(failure)")
    ax.set_title(title_abs)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    # ax.set_xticklabels(skills, rotation=20, ha="right")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()
    out_absorb = Path(out_absorb); fig.savefig(out_absorb, bbox_inches="tight", dpi=150); plt.close(fig)

    return out_ft, out_absorb


def plot_sensitivity_outcomes_spider_failure(
        base_system_failure: float,
        sens_df: pd.DataFrame,
        outpath: Union[str, Path],
        *,
        top_k: Optional[int] = None,  # show only the top-k (by |delta_failure|)
        title: str = "Sensitivity analysis (system failure probability)",
) -> Path:
    df = sens_df.copy()
    # order by absolute impact so the most relevant are readable on the spider
    df["abs_delta"] = df["delta_failure"].abs()
    df = df.sort_values("abs_delta", ascending=False)
    if top_k is not None and 0 < top_k < len(df):
        df = df.iloc[:top_k]

    labels = df["component"].tolist()
    vals = df["new_system_failure"].astype(float).to_numpy()

    if len(labels) == 0:
        out = Path(outpath);
        out.parent.mkdir(parents=True, exist_ok=True)
        # nothing to plot; return an empty file path expectation
        return out

    # angles & closed loop
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    vals_loop = np.r_[vals, vals[:1]]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # baseline ring
    baseline = float(base_system_failure)
    ax.plot([0, 0], [0, 0], alpha=0)  # force autoscale init
    ax.plot(angles, [baseline] * (len(labels) + 1), linestyle="--", linewidth=2, label="baseline")

    # perturbed curve
    ax.plot(angles, vals_loop, linewidth=2, label="×10 component p")
    ax.fill(angles, vals_loop, alpha=0.2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f"{l}\n{v:.2e}" for l, v in zip(labels, vals)], fontsize=9)
    ax.set_title(title, y=1.08)
    ax.grid(True)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))

    # ensure the radial scale includes both baseline and max value
    rmax = max(np.max(vals) * 1.05, baseline * 1.05)
    ax.set_ylim(0, rmax)

    fig.tight_layout()
    outpath = Path(outpath)
    fig.savefig(outpath, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return outpath


def sensitivity_analysis(
        failure_table: pd.DataFrame,
        redundancy_flags: Dict[str, bool],
        summary: Dict[str, pd.DataFrame],
        *,
        factor: float = 10.0,
        components: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    One-at-a-time sensitivity on BASIC EVENTS:
      - Builds a fresh HRM (FTs + MC) for the BASE case.
      - For each component, multiplies its p_fail in the failure_table by `factor` (clipped to 1),
        rebuilds HRM, recomputes system FAILURE probability via your wrapper.
    Returns DataFrame with:
      ['component','factor','base_system_failure','new_system_failure','delta_failure']
    """

    # --- Base model (fresh) ---
    base_fts = build_fault_trees_from_failure_table_basic(failure_table, redundancy_flags, copies=2)
    base_mc = build_dtmc_from_summary(summary, failure_per_skill=None, done_alpha=1.0)
    base_hrm = HybridReliabilityModel("base")
    base_hrm.add_markov_chain(base_mc)
    for ft in base_fts: base_hrm.add_fault_tree(ft)
    base_fail, _, _ = base_hrm.compute_system_reliability(ft_dict=create_ft_dict(base_hrm))

    # Components list
    if components is None:
        comps = sorted(failure_table["component"].astype(str).unique().tolist())
    else:
        comps = list(components)

    rows = []
    for comp in comps:
        # mutate table for this component
        mut_df = failure_table.copy()
        mask = mut_df["component"].astype(str).eq(comp)
        if not mask.any():
            # no rows for this comp in the table; skip
            rows.append({
                "component": comp,
                "factor": float(factor),
                "base_system_failure": float(base_fail),
                "new_system_failure": float(base_fail),
                "delta_failure": 0.0,
            })
            continue
        mut_df.loc[mask, "p_fail"] = np.clip(mut_df.loc[mask, "p_fail"].astype(float) * factor, 0.0, 1.0)

        # rebuild HRM from mutated table
        fts2 = build_fault_trees_from_failure_table_basic(mut_df, redundancy_flags, copies=2)
        mc2 = build_dtmc_from_summary(summary, failure_per_skill=None, done_alpha=1.0)
        hrm2 = HybridReliabilityModel(f"sens_{comp}")
        hrm2.add_markov_chain(mc2)
        for ft in fts2: hrm2.add_fault_tree(ft)

        new_fail, _, _ = hrm2.compute_system_reliability(ft_dict=create_ft_dict(hrm2))
        rows.append({
            "component": comp,
            "factor": float(factor),
            "base_system_failure": float(base_fail),
            "new_system_failure": float(new_fail),
            "delta_failure": float(new_fail - base_fail),
        })

    return (pd.DataFrame(rows)
            .sort_values("delta_failure", ascending=False)
            .reset_index(drop=True))


# ----- JSON payload for PDF -----
def write_report_json(
        name: str,
        system_failure_prob: float,
        task_success_rate_percent: float,
        base_probs: Dict[str, float],
        skill_pf: Dict[str, float],
        outpath: Path,
) -> Path:
    payload = {
        "name": name,
        "system_failure_prob": float(system_failure_prob),
        "task_success_rate_percent": float(task_success_rate_percent),
        "components": [{"name": k, "failure_prob": float(v)} for k, v in base_probs.items()],
        "skills": [{"name": s, "skill_failure_prob": float(p)} for s, p in skill_pf.items()],
    }
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w") as f:
        import json;
        json.dump(payload, f, indent=2)
    return outpath
