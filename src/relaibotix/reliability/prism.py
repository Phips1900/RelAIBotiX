# relaibotix/reliability/relaibotix_prism.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, List
import pandas as pd
import numpy as np
from decimal import Decimal, getcontext
from collections import defaultdict


def _round(p: float, prec: int) -> float:
    # keep rows summing to 1 after rounding (handled later)
    return float(round(float(p), prec))


def _normalize_pi0_distribution(pi0, state_names):
    """
    Accepts:
      - dict: {state_name -> p} or {state_index -> p}
      - list/tuple/np.ndarray: length == len(state_names)
      - pandas Series/DataFrame (1 column)
    Returns: dict {state_name -> float}
    """
    if pi0 is None:
        return None

    # dict with names or indices
    if isinstance(pi0, dict):
        out = {}
        for k, v in pi0.items():
            if isinstance(k, (int, np.integer)):
                if k < 0 or k >= len(state_names):
                    continue
                out[state_names[int(k)]] = float(v)
            else:
                k = str(k)
                if k in state_names:
                    out[k] = float(v)
        return out

    # pandas
    if isinstance(pi0, pd.DataFrame):
        if pi0.shape[1] != 1:
            raise ValueError("pi0 DataFrame must have exactly one column.")
        arr = pi0.iloc[:, 0].to_numpy()
    elif isinstance(pi0, pd.Series):
        arr = pi0.to_numpy()
    else:
        # numpy/list/tuple
        arr = np.asarray(pi0)

    arr = np.asarray(arr, dtype=float).reshape(-1)
    if arr.size != len(state_names):
        raise ValueError(f"pi0 length {arr.size} != number of states {len(state_names)}.")
    return {state_names[i]: float(arr[i]) for i in range(len(state_names))}


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
    precision: int = 12,
    include_rewards: bool = True,
) -> Tuple[str, Dict]:
    """
    Build a PRISM DTMC text from a MarkovChain that already contains numeric transitions.
    Returns (prism_text, meta) where meta carries the state index mapping and labels.
    """
    states: List[str] = list(mc.get_states())                # e.g., ["Init","Move","Pick","Carry","Place"]
    absorbing: List[str] = list(mc.get_absorbing_states())   # e.g., ["Init_failure", ... , "done"?]
    # Deduplicate, preserve order
    seen = set()
    all_states = []
    for s in states + absorbing:
        if s not in seen:
            seen.add(s)
            all_states.append(s)

    # Add a dummy initial chooser state
    INIT_NAME = "__init__"
    all_states_with_init = all_states + [INIT_NAME]

    # indexes
    idx = {s: i for i, s in enumerate(all_states_with_init)}
    init_id = idx[INIT_NAME]

    # transitions (already numeric after hybrid solve)
    trans: Dict[str, Dict[str, float]] = mc.get_transitions()

    # initial distribution (after hybrid solve)
    raw_pi0 = getattr(mc, "pi0", None)
    pi0 = _normalize_pi0_distribution(raw_pi0, all_states)  # use ALL states (skills + absorbing)
    if pi0 is None or sum(pi0.values()) <= 0.0:
        # fallback to first “regular” state if available
        first = states[0] if states else all_states[0]
        pi0 = {first: 1.0}

    # Build PRISM
    lines = []
    lines.append("dtmc\n")
    lines.append(f"module {model_name}\n")
    lines.append(f"  s : [0..{len(all_states_with_init)-1}] init {init_id};\n\n")

    # init transitions
    init_terms = []
    total = 0.0
    for st, p in pi0.items():
        if st not in idx:
            continue
        pr = _round(p, precision)
        total += pr
        init_terms.append(f"{pr} : (s'={idx[st]})")
    # normalize slight rounding drift
    if init_terms and abs(total - 1.0) > 1e-12:
        # adjust the largest term
        parts = [t.split(":")[0].strip() for t in init_terms]
        vals = [float(p) for p in parts]
        k = max(range(len(vals)), key=lambda i: vals[i])
        delta = 1.0 - sum(vals)
        vals[k] += delta
        init_terms = [f"{vals[i]} : (s'={init_terms[i].split('=')[-1]}" for i in range(len(vals))]  # (safe enough)
    lines.append(f"  [] s={init_id} -> " + " + ".join(init_terms) + ";\n\n")

    # --- regular rows ---
    for frm, out in trans.items():
        frm_id = idx[frm]
        terms = [(to, float(p)) for to, p in out.items() if to in idx and p > 0.0]
        terms = _normalize_row_terms(terms, precision)
        if not terms:
            # absorbing/self-loop
            terms = [(frm, 1.0)]
        rhs = " + ".join(f"{p:.{precision}g} : (s'={idx[to]})" for to, p in terms)
        lines.append(f"  [] s={frm_id} -> {rhs};\n")

    lines.append("endmodule\n\n")

    # --- Labels (unique by name) ---
    # Build a map: label_name -> set(state_ids)
    label_map = defaultdict(set)

    # Per-skill labels
    for s in states:
        label_map[s].add(idx[s])

    # Per-failure labels
    fail_states = [s for s in all_states if s.endswith("_failure")]
    for s in fail_states:
        label_map[s].add(idx[s])

    # Combined labels
    if fail_states:
        for s in fail_states:
            label_map["failure"].add(idx[s])

    done_states = [s for s in all_states if s.lower() == "done"]
    if done_states:
        for s in done_states:
            label_map["done"].add(idx[s])

    # Optional: normalize names to a single case to avoid 'Done'/'done' duplicates
    # (comment out if you rely on case)
    normalized_label_map = defaultdict(set)
    for name, idset in label_map.items():
        normalized_label_map[name.lower()] |= set(idset)

    # Write exactly once per label name
    for name, idset in normalized_label_map.items():
        if not idset:
            continue
        ids_str = " | ".join(f"s={i}" for i in sorted(idset))
        lines.append(f'label "{name}" = {ids_str};\n')

    # Simple reward: 1 per step (so you can ask expected steps to absorb)
    if include_rewards:
        lines.append("\nrewards \"steps\"\n")
        lines.append("  true : 1;\n")
        lines.append("endrewards\n")

    text = "".join(lines)
    meta = {
        "state_index": idx,
        "init_state": init_id,
        "states": all_states_with_init,
        "fail_states": fail_states,
        "done_states": done_states,
    }
    return text, meta


def write_prism_and_props(
    mc,
    out_basename: Path | str,
    *,
    model_name: str = "RelAIBotiX",
    precision: int = 12,
) -> Tuple[Path, Path]:
    """Write <basename>.pm and <basename>.pctl from MarkovChain."""
    out_basename = Path(out_basename)
    prism_txt, meta = export_prism_from_mc(mc, model_name=model_name, precision=precision)
    prism_path = out_basename.with_suffix(".pm")
    props_path = out_basename.with_suffix(".pctl")

    # model
    prism_path.parent.mkdir(parents=True, exist_ok=True)
    prism_path.write_text(prism_txt)

    # properties (success/failure; expected steps to absorption)
    lines = []
    if meta["fail_states"]:
        lines.append('P=? [ F "failure" ]')  # probability to ever hit ANY failure
    if meta["done_states"]:
        lines.append('P=? [ F "done" ]')     # probability to finish successfully
    lines.append('R{"steps"}=? [ F ("failure" | "done") ]')  # expected number of steps to absorb
    props_path.write_text("\n".join(lines) + "\n")
    return prism_path, props_path
