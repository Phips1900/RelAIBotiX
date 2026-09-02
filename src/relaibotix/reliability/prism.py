# relaibotix/reliability/relaibotix_prism.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, List
import pandas as pd
import numpy as np
from decimal import Decimal, getcontext
from collections import defaultdict


def _normalize_row_terms(terms, precision):
    """
    terms: list[(to_state_name, prob_float)]
    returns: list[(to_state_name, prob_float)] whose probs sum EXACTLY to 1.0
    """
    # keep enough working precision, then quantize to 10^-precision
    getcontext().prec = precision + 8
    quantum = Decimal(1).scaleb(-precision)  # 10^-precision

    # keep positive terms; if empty, caller will add a self loop
    dec_terms = [(to, Decimal(str(p))) for (to, p) in terms if p > 0.0]
    if not dec_terms:
        return []

    # quantize all except we’ll recompute one to close the row
    dec_terms = [(to, d.quantize(quantum)) for (to, d) in dec_terms]

    # if everything rounded to zero, keep a single 1.0
    if sum(d for _, d in dec_terms) == 0:
        t0 = dec_terms[0][0]
        return [(t0, 1.0)]

    # adjust the last term to make the sum exactly 1.0
    if len(dec_terms) == 1:
        adj = Decimal(1)
        dec_terms[0] = (dec_terms[0][0], adj)
    else:
        rest = sum(d for _, d in dec_terms[:-1])
        last = Decimal(1) - rest
        if last < 0:
            # if due to rounding the rest exceeded 1, push correction into the largest term
            k = max(range(len(dec_terms)), key=lambda i: dec_terms[i][1])
            rest2 = sum(dec_terms[i][1] for i in range(len(dec_terms)) if i != k)
            last = Decimal(1) - rest2
            dec_terms[k] = (dec_terms[k][0], max(Decimal(0), last))
        else:
            dec_terms[-1] = (dec_terms[-1][0], last)

    # back to floats
    return [(to, float(d)) for (to, d) in dec_terms]


def export_prism_from_mc(
    mc,
    *,
    model_name: str = "RelAIBotiX",
    precision: int = 16,
    include_rewards: bool = True,
) -> Tuple[str, Dict]:
    """
    Build a PRISM DTMC text from a MarkovChain that already contains numeric transitions.
    No init distribution: start state is s=0.
    """
    states: List[str] = list(mc.get_states())                # e.g., ["Init","Move","Pick","Carry","Place"]
    absorbing: List[str] = list(mc.get_absorbing_states())   # e.g., ["Pick_failure", ... , "done"?]

    # Deduplicate while preserving order
    seen = set()
    all_states: List[str] = []
    for s in states + absorbing:
        if s not in seen:
            seen.add(s)
            all_states.append(s)

    if not all_states:
        raise ValueError("No states provided by the MarkovChain.")

    # Index map; start state is index 0
    idx = {s: i for i, s in enumerate(all_states)}
    init_id = 0  # <-- required: s=0

    # Numeric transitions
    trans: Dict[str, Dict[str, float]] = mc.get_transitions()

    # Begin PRISM
    lines = []
    lines.append("dtmc\n")
    lines.append(f"module {model_name}\n")
    lines.append(f"  s : [0..{len(all_states)-1}] init {init_id};\n\n")

    # One row per state; if row missing or empty -> self-loop
    for frm in all_states:
        frm_id = idx[frm]
        out = trans.get(frm, {}) or {}
        terms = [(to, float(p)) for to, p in out.items() if (to in idx and p > 0.0)]

        # Normalize/round row to sum exactly 1.0; if nothing -> self-loop
        terms = _normalize_row_terms(terms, precision)
        if not terms:
            terms = [(frm, 1.0)]

        rhs = " + ".join(f"{p:.{precision}g} : (s'={idx[to]})" for to, p in terms)
        lines.append(f"  [] s={frm_id} -> {rhs};\n")

    lines.append("endmodule\n\n")

    # ---------- Labels ----------
    # label_name -> set(state_ids)
    label_map = defaultdict(set)

    # Per-state labels (skills and absorbing) EXCEPT a label literally named "init" (reserved)
    for s in all_states:
        label_name = s
        if label_name.lower() == "init":
            # skip or rename; choose to skip to avoid any collision with PRISM reserved word
            continue
        label_map[label_name].add(idx[s])

    # Combined 'failure' label for any state that ends with '_failure'
    fail_states = [s for s in all_states if s.endswith("_failure")]
    if fail_states:
        for s in fail_states:
            label_map["failure"].add(idx[s])

    # Combined 'done' label if a state is exactly "done" (case-insensitive)
    done_states = [s for s in all_states if s.lower() == "done"]
    if done_states:
        for s in done_states:
            label_map["done"].add(idx[s])

    # Write labels (do NOT lowercase names globally to avoid creating "init")
    for name, idset in label_map.items():
        if not idset:
            continue
        ids_str = " | ".join(f"s={i}" for i in sorted(idset))
        lines.append(f'label "{name}" = {ids_str};\n')

    # Optional reward structure
    if include_rewards:
        lines.append("\nrewards \"steps\"\n")
        lines.append("  true : 1;\n")
        lines.append("endrewards\n")

    text = "".join(lines)
    meta = {
        "state_index": idx,
        "init_state": init_id,
        "states": all_states,
        "fail_states": fail_states,
        "done_states": done_states,
    }
    return text, meta


def write_prism_and_props(
    mc,
    out_basename: Path | str,
    *,
    model_name: str = "RelAIBotiX",
    precision: int = 16,
) -> Tuple[Path, Path]:
    """Write <basename>.pm and <basename>.pctl from MarkovChain with s=0 start and requested properties."""
    out_basename = Path(out_basename)
    prism_txt, meta = export_prism_from_mc(mc, model_name=model_name, precision=precision)
    prism_path = out_basename.with_suffix(".pm")
    props_path = out_basename.with_suffix(".pctl")

    # Ensure directory exists and write model
    prism_path.parent.mkdir(parents=True, exist_ok=True)
    prism_path.write_text(prism_txt)

    # Properties:
    #  1) P=? [ F "failure" ]
    #  2) P=? [ (! "failure") U "done" ]  (only if 'done' exists)
    #  3) P=? [ F "<each_specific_failure_label>" ] for every *_failure state
    lines: List[str] = []

    # 1) Probability to eventually hit ANY failure
    lines.append('P=? [ F "failure" ]')

    # 2) Probability to reach done before any failure (until)
    if meta["done_states"]:
        lines.append('P=? [ (! "failure") U "done" ]')

    # 3) Probability to eventually hit each specific failure state
    for fs in meta["fail_states"]:
        # Use the exact state name as the label we wrote above
        lines.append(f'P=? [ F "{fs}" ]')

    props_path.write_text("\n".join(lines) + "\n")
    return prism_path, props_path


def _ordered_states_no_done(mc) -> Tuple[List[str], List[str], List[str]]:
    """Return (all_states_no_done, failure_states, done_states)."""
    states: List[str] = list(mc.get_states())
    absorbing: List[str] = list(mc.get_absorbing_states())
    # Preserve order, dedupe
    seen = set()
    all_states: List[str] = []
    for s in states + absorbing:
        if s not in seen:
            seen.add(s)
            all_states.append(s)
    done_states = [s for s in all_states if s.lower() == "done"]
    keep_states = [s for s in all_states if s not in done_states]
    failure_states = [s for s in keep_states if s.endswith("_failure")]
    return keep_states, failure_states, done_states


def export_prism_no_done(
    mc,
    *,
    model_name: str = "RelAIBotiX_NoDone",
    precision: int = 12,
    include_rewards: bool = True,
    state_time_seconds: Optional[Dict[str, float]] = None,
    start_state: Optional[str] = None,  # if None, first in list is the start (s=0)
) -> Tuple[str, Dict]:
    """
    Export a PRISM DTMC WITHOUT 'done' state. Any probability mass to 'done' is
    redirected to the start state (s=0). Adds a per-state 'time' reward (seconds)
    for all transient (non-absorbing) states, if provided in `state_time_seconds`.
    """
    all_states, fail_states, done_states = _ordered_states_no_done(mc)
    if not all_states:
        raise ValueError("No states available to export (after removing 'done').")

    # Reorder so the desired start state is first (s=0)
    if start_state:
        # match by exact name if possible, else case-insensitive
        pick = None
        if start_state in all_states:
            pick = start_state
        else:
            low = start_state.lower()
            for s in all_states:
                if s.lower() == low:
                    pick = s
                    break
        if pick and all_states[0] != pick:
            all_states = [pick] + [s for s in all_states if s != pick]

    idx = {s: i for i, s in enumerate(all_states)}
    start_name = all_states[0]  # s=0
    start_id = 0

    # Transitions
    raw_trans: Dict[str, Dict[str, float]] = mc.get_transitions()

    # Build PRISM
    lines = []
    lines.append("dtmc\n")
    lines.append(f"module {model_name}\n")
    lines.append(f"  s : [0..{len(all_states)-1}] init {start_id};\n\n")

    for frm in all_states:
        out = dict(raw_trans.get(frm, {}) or {})
        # Remove any transitions to 'done' and accumulate probability to redirect
        redirect_p = 0.0
        if done_states:
            for d in done_states:
                redirect_p += float(out.pop(d, 0.0))
        # If frm not in raw_trans, we still write a self-loop later

        # Remove transitions to states we dropped (defensive)
        out = {to: float(p) for to, p in out.items() if to in idx and p > 0.0}

        # Redirect prob mass to start state
        if redirect_p > 0.0:
            out[start_name] = out.get(start_name, 0.0) + redirect_p

        # Normalize row; if empty, self-loop
        terms = _normalize_row_terms(list(out.items()), precision)
        if not terms:
            terms = [(frm, 1.0)]

        rhs = " + ".join(f"{p:.{precision}g} : (s'={idx[to]})" for to, p in terms)
        lines.append(f"  [] s={idx[frm]} -> {rhs};\n")

    lines.append("endmodule\n\n")

    # ---------- Labels ----------
    label_map = defaultdict(set)

    RESERVED_LABELS = {"init"}  # avoid PRISM reserved label names
    def safe_label(name: str) -> str:
        return f"{name}_state" if name.lower() in RESERVED_LABELS else name

    # Individual state labels (except 'done' which is removed)
    for s in all_states:
        lab = safe_label(s)
        label_map[lab].add(idx[s])

    # Combined 'failure' label
    if fail_states:
        for s in fail_states:
            label_map["failure"].add(idx[s])

    # Write labels
    for name, idset in label_map.items():
        if not idset:
            continue
        ids_str = " | ".join(f"s={i}" for i in sorted(idset))
        lines.append(f'label "{name}" = {ids_str};\n')

    # ---------- Rewards (per-state time in seconds) ----------
    if include_rewards:
        # Only for transient states (non-absorbing, i.e., not *_failure)
        lines.append('\nrewards "time"\n')
        times = state_time_seconds or {}
        for s in all_states:
            if s in fail_states:
                continue
            val = float(times.get(s, 0.0))  # 0.0 if not provided
            if val != 0.0:
                lines.append(f"  s={idx[s]} : {val};\n")
        lines.append("endrewards\n")

    text = "".join(lines)
    meta = {
        "state_index": idx,
        "states": all_states,
        "start_state": start_name,
        "fail_states": fail_states,
        "removed_done_states": done_states,
    }
    return text, meta


def write_prism_no_done_and_props(
    mc,
    out_basename: Path | str,
    *,
    model_name: str = "RelAIBotiX_NoDone",
    precision: int = 12,
    state_time_seconds: Optional[Dict[str, float]] = None,
    start_state: Optional[str] = None,
) -> Tuple[Path, Path]:
    """
    Write <basename>.pm and <basename>.pctl with 'done' removed and 'time' reward.
    Properties include:
      - P=? [ F "failure" ]                 (probability of ever failing)
      - R{"time"}=? [ F "failure" ]         (expected time to failure; uses 'time' rewards)
    """
    out_basename = Path(out_basename)
    prism_txt, meta = export_prism_no_done(
        mc,
        model_name=model_name,
        precision=precision,
        include_rewards=True,
        state_time_seconds=state_time_seconds,
        start_state=start_state,
    )
    pm_path = out_basename.with_suffix(".pm")
    pctl_path = out_basename.with_suffix(".pctl")

    pm_path.parent.mkdir(parents=True, exist_ok=True)
    pm_path.write_text(prism_txt)

    lines: List[str] = []
    lines.append('P=? [ F "failure" ]')
    lines.append('R{"time"}=? [ F "failure" ]')  # mean time to failure (seconds)

    pctl_path.write_text("\n".join(lines) + "\n")
    return pm_path, pctl_path
