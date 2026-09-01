"""This module provides helper functions for RelAIbotiX."""
from typing import Dict, Union, Any, List, Optional
import math
import h5py
import numpy as np
from pathlib import Path
import pandas as pd
from omegaconf import OmegaConf, DictConfig
from relaibotix.reliability.reliability_models import HybridReliabilityModel, FaultTree, MarkovChain
from relaibotix.reliability.graph import create_ft_graph

# --- constants ---------------------------------------------------------------
ALWAYS_ACTIVE = ["Controller", "Power_Supply", "Sensors", "Camera"]  # JSON keys


def _normalize_failure_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize to columns: ['skill','component','p_fail'] and drop empties."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["skill", "component", "p_fail"])
    cols = {c.lower(): c for c in df.columns}
    skill_col = cols.get("skill")
    comp_col = cols.get("component")
    p_col = (cols.get("p_fail") or cols.get("pfail") or
             cols.get("prob_fail") or cols.get("failure_prob") or cols.get("failure_probability"))
    if not (skill_col and comp_col and p_col):
        raise ValueError("Input df must have columns for skill, component, and p_fail.")
    out = (df.rename(columns={skill_col: "skill", comp_col: "component", p_col: "p_fail"})
           [["skill", "component", "p_fail"]].copy())
    out = out.dropna(subset=["skill", "component", "p_fail"])
    out["skill"] = out["skill"].astype(str)
    out["component"] = out["component"].astype(str)
    out["p_fail"] = out["p_fail"].astype(float)
    return out


def combine_failure_tables(joint_failure_table: pd.DataFrame,
                           gripper_table: pd.DataFrame) -> pd.DataFrame:
    """
    Stack joints + gripper into one tidy table:
      columns: ['skill','component','p_fail']
    If duplicates exist for the same (skill,component), keep the MAX (conservative).
    """
    jf = _normalize_failure_cols(joint_failure_table)
    gf = _normalize_failure_cols(gripper_table)
    if jf.empty and gf.empty:
        return pd.DataFrame(columns=["skill", "component", "p_fail"])
    combined = pd.concat([jf, gf], ignore_index=True)
    combined = (combined.groupby(["skill", "component"], as_index=False)["p_fail"].max())
    return combined


def _p_per_min_to_lambda_per_s(p_per_min: float) -> float:
    """Exact per-minute prob -> per-second hazard λ: λ = -ln(1-p)/60."""
    p = max(min(float(p_per_min), 1.0 - 1e-15), 0.0)
    return -math.log(1.0 - p) / 60.0


def augment_failure_table_with_always_active(
        failure_table: pd.DataFrame,
        summary: dict,
        base_p_per_min: dict,
        components_to_add=("Controller", "Power_Supply", "Sensors", "Camera"),
        name_map=None,
) -> pd.DataFrame:
    """
    Adds rows (skill × component) for always-active components with:
      p_fail(skill) = 1 - exp(-λ * total_time_in_skill)
    Uses summary['skill_time'] with columns ['skill','total_time_sec'].
    name_map can map JSON names to failure_table names if needed.
    """
    name_map = name_map or {}
    if "skill_time" not in summary:
        raise ValueError("summary['skill_time'] missing. Run summarize() first.")
    st = summary["skill_time"][["skill", "avg_time_per_episode_sec"]].copy()

    rows = []
    for _, r in st.iterrows():
        skill = r["skill"];
        # T = float(r["total_time_sec"])
        T = float(r["avg_time_per_episode_sec"])
        for comp_json in components_to_add:
            comp_ft = name_map.get(comp_json, comp_json)
            if comp_json not in base_p_per_min:
                continue
            lam = _p_per_min_to_lambda_per_s(base_p_per_min[comp_json])
            p = 1.0 - math.exp(-lam * T)
            rows.append({"skill": skill, "component": comp_ft, "p_fail": float(p)})

    if not rows:
        return failure_table.copy()

    extra = pd.DataFrame(rows)
    combined = (pd.concat([failure_table, extra], ignore_index=True)
                .groupby(["skill", "component"], as_index=False)["p_fail"]
                .max())  # conservative on duplicates
    return combined


# --- Redundancy expansion: per-component, ungrouped (Controller_1/_2, …) ----
def expand_basic_redundancy(
        basic_events: dict[str, float],
        redundancy_flags: dict[str, bool],
        *,
        copies: int = 2
) -> tuple[dict[str, float], dict[str, list[str]]]:
    """
    For each component with redundancy=True, replace single basic event with
    <copies> clones and return an AND group mapping for the FT builder.
    """
    expanded: dict[str, float] = {}
    groups: dict[str, list[str]] = {}
    for comp, p in basic_events.items():
        if redundancy_flags.get(comp, False):
            members = [f"{comp}_{i + 1}" for i in range(copies)]
            for m in members: expanded[m] = float(p)
            groups[comp] = members
        else:
            expanded[comp] = float(p)
    return expanded, groups


def build_fault_trees_from_failure_table_basic(
        failure_table: pd.DataFrame,  # ['skill','component','p_fail']
        redundancy_flags: dict[str, bool],
        copies: int = 2
) -> list:
    fts = []
    for skill, sub in failure_table.groupby("skill"):
        basic = {row["component"]: float(row["p_fail"]) for _, row in sub.iterrows()}
        expanded_basic, groups = expand_basic_redundancy(basic, redundancy_flags, copies=copies)
        ft = FaultTree(name=f"FT_{skill}", top_event=f"{skill}_failure", skill=skill)
        ft.add_basic_events(expanded_basic)
        if groups:
            ft.add_gates(redundancy=True, redundant_components=groups)
        else:
            ft.add_gates(redundancy=False)
        fts.append(ft)
    return fts


# --- Build DTMC from analyzer summary (symbolic or numeric) ------------------
def build_dtmc_from_summary(
        summary: dict,
        *,
        failure_per_skill: Optional[Dict[str, float]] = None,
        done_alpha: float = 1.0,
        canonical_order: Optional[List[str]] = None
):
    mc = MarkovChain("RelAIBotiX-DTMC")
    mc.build_from_analyzer(summary, failure_per_skill=failure_per_skill,
                           done_alpha=done_alpha, canonical_order=canonical_order)
    return mc


def create_ft_dict(hybrid_model: Any) -> Dict[str, List[Any]]:
    """
    Creates a dictionary of fault trees from the hybrid model.

    Args:
        hybrid_model (Any): An object that contains a 'fault_trees' attribute (iterable of fault tree objects).
                            Each fault tree is expected to have a 'name' attribute.

    Returns:
        Dict[str, List[Any]]: A dictionary mapping fault tree names to a list [fault_tree, fault_tree_graph].
    """
    ft_dict: Dict[str, List[Any]] = {}
    for ft in hybrid_model.fault_trees:
        # Assumes create_ft_graph is defined elsewhere and returns the fault tree graph for ft.
        ft_graph = create_ft_graph(ft)
        ft_dict[ft.name] = [ft, ft_graph]
    return ft_dict


def perform_sensitivity_analysis(hybrid_model: Any,
                                 robotic_system: Any,
                                 sensitivity_analysis_data: Dict[str, float]
                                 ) -> Dict[str, float]:
    """
    Performs sensitivity analysis on the hybrid reliability model for each component in the robotic system.

    Args:
        hybrid_model (Any): The initial hybrid reliability model.
        robotic_system (Any): The robotic system containing components and skills.
        sensitivity_analysis_data (Dict[str, float]): A dictionary to store computed system reliability
            for each component, keyed by component name.

    Returns:
        Dict[str, float]: The updated sensitivity analysis data mapping each component's name to its
                          computed system reliability.
    """
    for component in robotic_system.components:
        # Create new states based on the current skills of the robotic system.
        new_states = create_skill_list(robotic_system.get_skills())

        # Create a new hybrid model and associated Markov chain for the component.
        hybrid_model_new = HybridReliabilityModel(component.name)
        mc_new = MarkovChain(component.name)
        mc_new.auto_create_mc(states=new_states, done_state=True, repeat_info=1)
        hybrid_model_new.add_markov_chain(mc_new)

        # Create fault trees based on the new hybrid model.
        hybrid_model_new = create_fault_trees(robotic_system, hybrid_model_new)

        # Update the basic event probability for the component in each fault tree.
        for ft in hybrid_model_new.fault_trees:
            if component.name in ft.basic_events:
                old_prob = ft.basic_events[component.name]
                new_prob = old_prob * 10.0
                ft.basic_events[component.name] = new_prob

        # Create a fault tree dictionary and compute system reliability.
        new_fts = create_ft_dict(hybrid_model_new)
        new_system_reliability, new_absorption_prob, new_absorption_time = (hybrid_model_new.compute_system_reliability(
            ft_dict=new_fts,
            repeat_dict={'done': 0.1, 'object_detection': 0.9}
        ))

        # Store the computed reliability in the sensitivity analysis data.
        sensitivity_analysis_data[component.name] = new_system_reliability

    return sensitivity_analysis_data


def compare_predictions(y_true: np.ndarray, y_pred: np.ndarray, class_names=None):
    """
    Quick comparison of ground-truth vs predicted labels.
    Returns a dict with accuracy, per-class accuracy, confusion matrix, and label order.

    - y_true, y_pred: 1D integer arrays of same length (or longer; extra tail is ignored)
    - class_names: optional list/tuple mapping label->name in the same order as 'labels'
    """
    if y_true is None:
        raise ValueError("y_true is None — nothing to compare against.")
    n = min(len(y_true), len(y_pred))
    yt = np.asarray(y_true[:n], dtype=int)
    yp = np.asarray(y_pred[:n], dtype=int)

    labels = np.unique(np.concatenate([np.unique(yt), np.unique(yp)]))
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    K = len(labels)

    # confusion matrix
    cm = np.zeros((K, K), dtype=int)
    for t, p in zip(yt, yp):
        cm[label_to_idx[t], label_to_idx[p]] += 1

    # overall accuracy
    acc = (cm.diagonal().sum() / max(1, cm.sum())).item()

    # per-class accuracy (a.k.a. recall)
    support = cm.sum(axis=1)  # true instances per class
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class_acc = np.where(support > 0, cm.diagonal() / support, np.nan)

    # pretty names if provided
    if class_names is not None and len(class_names) >= labels.max() + 1:
        names = [class_names[l] for l in labels]
    else:
        names = [str(l) for l in labels]

    # brief printout
    print(f"[compare] N={n}, overall accuracy = {acc:.3f}")
    for name, a, sup in zip(names, per_class_acc, support):
        a_str = "nan" if np.isnan(a) else f"{a:.3f}"
        print(f"  - {name:>10s}: acc={a_str} (support={int(sup)})")

    return {
        "accuracy": float(acc),
        "per_class_accuracy": per_class_acc,
        "support": support,
        "labels": labels,
        "class_names": names,
        "confusion_matrix": cm,
    }


def filter_short_segments(y_pred: np.ndarray, *, min_len: int = 10, context: int = 1) -> np.ndarray:
    """
    Replace short segments (< min_len) with the surrounding majority label
    when the majority label before and after the segment MATCH.

    Parameters
    ----------
    y_pred : 1D int array of labels
    min_len : minimum allowed length of a segment
    context : number of samples to look before/after the segment

    Returns
    -------
    np.ndarray : corrected labels (same shape as y_pred)
    """
    y = np.asarray(y_pred, dtype=int)
    N = y.size
    if N == 0:
        return y

    # find segment boundaries
    starts = np.flatnonzero(np.r_[True, y[1:] != y[:-1]])
    ends   = np.r_[starts[1:], N]

    out = y.copy()
    for s, e in zip(starts, ends):
        if (e - s) >= min_len:
            continue

        left  = y[max(0, s - context): s]
        right = y[e: min(N, e + context)]

        if left.size == 0 or right.size == 0:
            continue

        # majority labels in context windows
        lmaj = np.bincount(left).argmax()
        rmaj = np.bincount(right).argmax()

        # overwrite only if both sides agree
        if lmaj == rmaj:
            out[s:e] = lmaj

    return out
