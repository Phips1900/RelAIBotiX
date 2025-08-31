"""This module provides the classes HybridReliabilityModel, MarkovChain, and FaultTree."""
from __future__ import annotations
from typing import Optional, List, Tuple, Dict, Union, Iterable
from solver import *
import numpy as np
import pandas as pd
import math
import copy


class HybridReliabilityModel:
    """
    Orchestrates: Analyzer -> Fault Trees (per skill) -> DTMC -> external solver.
    """

    def __init__(self, name: str):
        self.name = name
        self.markov_chain: Optional[MarkovChain] = None
        self.fault_trees: List[FaultTree] = []

        # (optional) keep references for audit
        self.summary: Optional[Dict[str, pd.DataFrame]] = None
        self.failure_table: Optional[pd.DataFrame] = None
        self.redundancy_map: Dict[str, Dict[str, List[str]]] = {}  # {"Pick":{"arm":["j1","j2"]}, ...}

    # -------------------- housekeeping --------------------

    def clear(self) -> None:
        self.name = ""
        self.markov_chain = None
        self.fault_trees.clear()
        self.summary = None
        self.failure_table = None
        self.redundancy_map = {}

    def get_name(self) -> str:
        return self.name

    def set_name(self, name: str) -> bool:
        self.name = name
        return True

    # -------------------- attach components --------------------

    def add_markov_chain(self, markov_chain: MarkovChain) -> bool:
        self.markov_chain = markov_chain
        return True

    def remove_markov_chain(self) -> bool:
        self.markov_chain = None
        return True

    def add_fault_tree(self, fault_tree: FaultTree) -> bool:
        self.fault_trees.append(fault_tree)
        return True

    def remove_fault_tree(self, fault_tree: FaultTree) -> bool:
        if fault_tree not in self.fault_trees:
            return False
        self.fault_trees.remove(fault_tree)
        return True

    def get_markov_chain(self) -> Optional[MarkovChain]:
        return self.markov_chain

    def get_fault_trees(self) -> List[FaultTree]:
        return self.fault_trees

    # -------------------- build from analyzer outputs --------------------

    def build_fault_trees_from_failure_table(
            self,
            failure_table: pd.DataFrame,
            *,
            redundancy_map: Optional[Dict[str, Dict[str, List[str]]]] = None
    ) -> None:
        """
        Create one FaultTree per skill from a tidy table with columns:
           ['skill','component','p_fail']
        Redundancy_map groups components into AND gates per skill, e.g.:
           {"Pick":{"arm":["j1","j2","j3"]}, "Place":{"control":["controller","power_supply"]}}
        """
        self.failure_table = failure_table.copy()
        if redundancy_map:
            self.redundancy_map = redundancy_map

        self.fault_trees.clear()

        for skill, sub in failure_table.groupby("skill"):
            # active components = rows present for this skill
            basic = {row["component"]: float(row["p_fail"]) for _, row in sub.iterrows()}
            ft = FaultTree(name=f"FT_{skill}", top_event=f"{skill}_failure", skill=skill)

            # build with redundancy groups if provided for this skill
            groups = (redundancy_map or {}).get(skill, {})
            use_redundancy = len(groups) > 0
            ft.auto_create_ft(basic_events=basic, redundancy=use_redundancy, redundant_components=groups)
            self.fault_trees.append(ft)

    def build_markov_chain_from_summary(
            self,
            summary: Dict[str, pd.DataFrame],
            *,
            failure_per_skill: Optional[Dict[str, float]] = None,  # if None, build symbolic DTMC (late binding)
            done_alpha: float = 1.0,
            canonical_order: Optional[List[str]] = None
    ) -> None:
        """
        Build the DTMC using your analyzer summary (sequences & counts).
        If failure_per_skill is None, DTMC stores symbolic transitions.
        """
        self.summary = summary
        if self.markov_chain is None:
            self.markov_chain = MarkovChain("RelAIBotiX-DTMC")

        self.markov_chain.build_from_analyzer(
            summary=summary,
            failure_per_skill=failure_per_skill,  # may be None (symbolic)
            canonical_order=canonical_order,
            done_alpha=done_alpha
        )

    # -------------------- FT evaluation and MC binding --------------------

    def evaluate_fault_trees(self) -> Dict[str, float]:
        """
        Evaluate each FT to get per-skill top-event probabilities.
        Returns: {"Init": p, "Move": p, ..., "Place": p}
        """
        skill_p: Dict[str, float] = {}
        for ft in self.fault_trees:
            p_top = ft.evaluate(set_cache=True)
            # ft.top_event is "<skill>_failure"
            skill = ft.get_skill() if ft.get_skill() else ft.get_top_event().replace("_failure", "")
            skill_p[skill] = p_top
        return skill_p

    def compile_mc_with_ft(self, *, failure_map: Optional[Dict[str, float]] = None) -> None:
        """
        Late-binding path: substitute FT results into a symbolic DTMC.
        failure_map may be either:
          - {"Init": pI, "Move": pM, ...}  (skill names)
          - or {"Init_failure": pI, "Move_failure": pM, ...} (top-event tokens)
        """
        if self.markov_chain is None:
            raise ValueError("Markov chain not set.")

        if failure_map is None:
            # build from our FTs if not provided
            skill_p = self.evaluate_fault_trees()
        else:
            # normalize keys to "<skill>_failure"
            skill_p = {}
            for k, v in failure_map.items():
                if k.endswith("_failure"):
                    skill = k[:-8]
                else:
                    skill = k
                skill_p[skill] = float(v)

        # transform to tokens for DTMC compiler
        tokens = {f"{skill}_failure": p for skill, p in skill_p.items()}
        self.markov_chain.compile_with_failures(tokens)

    # -------------------- solver interop --------------------

    def as_solver_ft_dict(self) -> Dict[str, Dict]:
        """
        Export a simple dict the external solver can consume.
        Shape:
          {
            "<skill>_failure": {
               "basic_events": {"j1": p, "controller": p, ...},
               "gates": {... full gate dict ...}
            },
            ...
          }
        """
        out: Dict[str, Dict] = {}
        for ft in self.fault_trees:
            exp = ft.export()
            out[ft.get_top_event()] = {
                "basic_events": exp["basic_events"],
                "gates": exp["gates"],
                "skill": exp["skill"],
                "name": exp["name"],
            }
        return out

    def compute_system_reliability(
            self,
            *,
            repeat_dict: Optional[Dict] = None,
            ft_dict=None,
            use_numeric_mc: bool = True,
    ) -> Tuple[float, List[float], List[float]]:
        """
        Adaptor for solver.py.
        Expects solver to expose `hybrid_solver(ft_dict, mc_object, repeat_dict)`.

        Returns:
           system_reliability, absorption_prob_row, absorption_time_row
        """
        if self.markov_chain is None:
            raise ValueError("Cannot compute reliability: Markov chain is not set.")

        absorption_prob, absorption_time = hybrid_solver(
            ft_dict=ft_dict,
            mc_object=self.markov_chain,
        )

        # Your solver seems to return row vectors; keep your original post-processing:
        num_cols = absorption_prob.shape[1] - 1
        absorption_prob_row = absorption_prob[0, 0:num_cols]
        system_reliability = float(absorption_prob_row.sum())
        return system_reliability, list(absorption_prob_row), list(absorption_time[0])


class MarkovChain:
    """
    MarkovChain class
    """

    def __init__(self, name: str):
        """constructor"""
        self.name = name
        # core topology
        self.states: List[str] = []
        self.absorbing_states: List[str] = []
        self.edges: Dict[str, List[str]] = {}
        # transitions
        self.transitions: Dict[str, Dict[str, Union[float, str]]] = {}
        self.state_order: List[str] = []  # states + absorbing states order for P
        self.P: Optional[np.ndarray] = None  # numeric transition matrix
        self.pi0: Optional[np.ndarray] = None  # initial distribution over state_order (absorbing entries = 0)
        # self.transition_matrix = []

    # --------------------- basic mutators ---------------------

    def clear(self) -> None:
        self.states.clear()
        self.absorbing_states.clear()
        self.edges.clear()
        self.transitions.clear()
        self.state_order.clear()
        self.P = None
        self.pi0 = None

    def add_states(self, states: List[str]) -> bool:
        self.states = list(states)
        return True

    def add_absorbing_states(self, add_done: bool = True) -> bool:
        self.absorbing_states = [f"{s}_failure" for s in self.states]
        if add_done:
            self.absorbing_states.append("done")
        return True

    def compile_with_failures(self, failure_map: Dict[str, float]) -> np.ndarray:
        """
        Substitute {'Move_failure': p, ...} into self.transitions and build numeric P.
        Supports forms:
          'x'                  -> variable
          '1 - x'              -> survival
          '(1 - x)'            -> survival
          'k*(1 - x)' or '(1 - x)*k'  -> weighted survival (k numeric)
        """
        self.state_order = list(self.states) + list(self.absorbing_states)
        idx = {s: i for i, s in enumerate(self.state_order)}
        n = len(self.state_order)
        P = np.zeros((n, n), dtype=float)

        def eval_expr(expr: Union[str, float]) -> float:
            if isinstance(expr, (int, float)):
                return float(expr)
            s = str(expr).strip().replace(" ", "")
            # k*(1-x) or (1-x)*k
            if "*" in s and "(1-" in s:
                a, b = s.split("*", 1)

                def surv(part):
                    if part.startswith("(1-") and part.endswith(")"):
                        var = part[3:-1]
                        return 1.0 - float(failure_map[var])
                    return None

                # try both orders
                k = None;
                surv_val = None
                try:
                    k = float(a);
                    surv_val = surv(b)
                except:
                    pass
                if k is None or surv_val is None:
                    try:
                        k = float(b);
                        surv_val = surv(a)
                    except:
                        pass
                if k is not None and surv_val is not None:
                    return k * surv_val
            # (1-x)
            if s.startswith("(1-") and s.endswith(")"):
                var = s[3:-1]
                return 1.0 - float(failure_map[var])
            # 1-x
            if s.startswith("1-"):
                var = s[2:]
                return 1.0 - float(failure_map[var])
            # plain var
            return float(failure_map[s])

        for src, dsts in self.transitions.items():
            i = idx[src]
            for d, expr in dsts.items():
                j = idx[d]
                P[i, j] = eval_expr(expr)
            if src in self.absorbing_states:
                P[i, :] = 0.0
                P[i, i] = 1.0

        # normalize tiny drift and validate
        rowsum = P.sum(axis=1)
        for i, s in enumerate(self.state_order):
            if s in self.absorbing_states:
                P[i, :] = 0.0;
                P[i, i] = 1.0
            elif rowsum[i] > 0:
                P[i, :] /= rowsum[i]
        self.P = P
        self._validate_rows()
        return P

    def export_numeric(self):
        """
        Returns numeric inputs for the solver:
          P        : np.ndarray transition matrix
          states   : list[str] order of rows/cols in P
          absorbing: list[str] absorbing states (subset of states)
          pi0      : np.ndarray initial distribution over 'states'
        """
        if self.P is None or self.pi0 is None:
            raise RuntimeError("Build/compile the DTMC first (P, pi0 missing).")
        return self.P, list(self.state_order), list(self.absorbing_states), self.pi0

    # --------------------- analyzer integration ---------------------

    @staticmethod
    def _most_frequent_sequence(summary_sequences: pd.DataFrame) -> List[str]:
        """
        Get the token order from the most frequent sequence string, e.g. "Init > Move > Pick > Carry > Place".
        """
        if summary_sequences.empty:
            return []
        seq = summary_sequences.sort_values("count", ascending=False).iloc[0]["sequence"]
        return [tok.strip() for tok in str(seq).split(" > ") if tok.strip()]

    @staticmethod
    def _start_skill_mixture(summary: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Initial distribution over starting states from analyzer summary.
        Returns dict like {"Init": 0.59, "Move": 0.41}.
        """
        seq_df = summary["sequences"]  # columns: sequence, count, ...
        starts = seq_df["sequence"].str.split(" > ").str[0]
        # sum counts per start token
        start_counts = seq_df.assign(start=starts).groupby("start")["count"].sum()
        R = int(summary["overall"].iloc[0]["n_runs"])
        if R <= 0:
            return {}
        return {k: float(v) / R for k, v in start_counts.items()}

    @staticmethod
    def _place_exit_weights(summary: Dict[str, pd.DataFrame], *, done_alpha: float = 1.0) -> Dict[str, float]:
        """
        Where to go after Place if Place succeeds:
          weights over {"Done", <starting skills>}, summing to 1.
        done_alpha=1.0 implements 1/(R+1) smoothing for the Done branch.
        """
        seq_df = summary["sequences"]
        starts = seq_df["sequence"].str.split(" > ").str[0]
        start_counts = seq_df.assign(start=starts).groupby("start")["count"].sum()
        R = int(summary["overall"].iloc[0]["n_runs"])
        # Denominator ensures the weights sum to 1 when sum(counts)=R
        Z = float(R + done_alpha)
        w = {"Done": float(done_alpha) / Z}
        for start, cnt in start_counts.items():
            w[start] = float(cnt) / Z
        # No extra normalization needed: (done_alpha + sum(cnt))/Z == 1
        return w

    def build_from_analyzer(
            self,
            summary: Dict[str, pd.DataFrame],
            failure_per_skill: Dict[str, float] = None,
            *,
            canonical_order: Optional[List[str]] = None,
            done_alpha: float = 1.0
    ) -> None:
        """
        Auto-build the DTMC from analyzer summaries and per-skill failure probabilities.

        summary: output of `summarize(...)`:
           - summary["sequences"]: columns ["sequence","count",...]
           - summary["overall"]:   one row with "n_runs"
        failure_per_skill: e.g., {"Init": p_fail_init, "Move": p_fail_move, ..., "Place": p_fail_place}

        canonical_order:
           If provided, use that exact linear order for the task (e.g., ["Init","Move","Pick","Carry","Place"]).
           Otherwise, we infer the order from the most frequent sequence.

        done_alpha:
           Dirichlet/Laplace smoothing mass for 'Done' in the Place row (1/(R+1)).
        """
        self.clear()

        # 1) States & absorbing
        if canonical_order is None:
            canonical_order = self._most_frequent_sequence(summary["sequences"])
        self.add_states(canonical_order)
        self.add_absorbing_states(add_done=True)

        # 2) Initial distribution (Init vs Move mix, etc.)
        start_mix = self._start_skill_mixture(summary)
        self._build_pi0(start_mix)

        # (optional) keep adjacency for inspection
        self._build_edges_for_linear_flow(start_mix_keys=list(start_mix.keys()))

        # 3) Place/last-skill exit weights (Done + restart mix)
        place_weights = self._place_exit_weights(summary, done_alpha=done_alpha)

        # 4) Build transitions
        if failure_per_skill is None:
            # symbolic: strings like "Move_failure", "w*(1 - Place_failure)"
            self._build_symbolic_transitions(place_weights)
            self.P = None  # no numeric matrix yet; call compile_with_failures later
        else:
            # numeric: bind FT probabilities now
            self._build_numeric_transitions(failure_per_skill, place_weights)
            # 5) Compile numeric matrix & validate
            self._compile_numeric()
            self._validate_rows()

    # --------------------- internal helpers ---------------------

    def _build_pi0(self, start_mix: Dict[str, float]) -> None:
        """Build initial distribution over (states + absorbing), allocating mass only to observed start states."""
        self.state_order = list(self.states) + list(self.absorbing_states)
        idx = {s: i for i, s in enumerate(self.state_order)}
        pi = np.zeros(len(self.state_order), dtype=float)
        for s, p in start_mix.items():
            if s in idx:  # only states
                pi[idx[s]] = float(p)
        # normalize if needed
        s = pi.sum()
        if s > 0:
            pi /= s
        self.pi0 = pi

    def _build_edges_for_linear_flow(self, start_mix_keys: List[str]) -> None:
        """Adjacency lists (lists only)."""
        self.edges.clear()
        n = len(self.states)
        for i, s in enumerate(self.states):
            self.edges[s] = []
            # next op in chain (if any)
            if i + 1 < n:
                self.edges[s].append(self.states[i + 1])
            # its own failure absorbing
            self.edges[s].append(f"{s}_failure")

        # Last op (e.g., Place): add 'done' and also arcs back to observed start states
        if self.states:
            last = self.states[-1]
            if "done" in self.absorbing_states:
                self.edges[last].append("done")
            for st in start_mix_keys:
                if st in self.states:
                    self.edges[last].append(st)

        # absorbing self-loops
        for a in self.absorbing_states:
            self.edges[a] = [a]

    def _build_numeric_transitions(
            self,
            failure_per_skill: Dict[str, float],
            place_weights: Dict[str, float],
    ) -> None:
        """
        Fill `self.transitions` with numeric probabilities (still stored as dicts) and keep a parallel view for P.
        """
        self.transitions.clear()

        # Precompute convenience
        last = self.states[-1] if self.states else None
        for i, s in enumerate(self.states):
            self.transitions[s] = {}
            p_fail = float(failure_per_skill.get(s, 0.0))
            p_surv = max(0.0, 1.0 - p_fail)

            # failure arc
            self.transitions[s][f"{s}_failure"] = p_fail

            if s != last:
                # linear forward when we survive
                nxt = self.states[i + 1]
                self.transitions[s][nxt] = p_surv
            else:
                # last state's exit split by empirical mixture (Done + start skills)
                for dest, w in place_weights.items():
                    if dest == "Done" and "done" in self.absorbing_states:
                        self.transitions[s]["done"] = p_surv * float(w)
                    elif dest in self.states:
                        self.transitions[s][dest] = p_surv * float(w)
                # (If a weight points to a state not in `self.states`, ignore silently.)

        # absorbing rows
        for a in self.absorbing_states:
            self.transitions[a] = {a: 1.0}

    def _compile_numeric(self) -> None:
        """Create the numeric P matrix from `self.transitions` in the fixed `state_order`."""
        if not self.state_order:
            self.state_order = list(self.states) + list(self.absorbing_states)
        idx = {s: i for i, s in enumerate(self.state_order)}
        n = len(self.state_order)
        P = np.zeros((n, n), dtype=float)
        for src, dsts in self.transitions.items():
            i = idx[src]
            for d, prob in dsts.items():
                P[i, idx[d]] = float(prob)
            # ensure absorbing identity
            if src in self.absorbing_states:
                P[i, :] = 0.0
                P[i, i] = 1.0
        self.P = P

    def _validate_rows(self, tol: float = 1e-9) -> None:
        if self.P is None:
            return
        rowsum = self.P.sum(axis=1)
        bad = np.where(np.abs(rowsum - 1.0) > tol)[0]
        if bad.size:
            msg = "; ".join([f"{self.state_order[i]} sum={rowsum[i]:.6f}" for i in bad.tolist()])
            raise ValueError(f"DTMC row(s) not stochastic: {msg}")

    def _build_symbolic_transitions(self, place_weights: Dict[str, float]) -> None:
        """Store transitions as strings with tokens like 'Move_failure' or 'w*(1 - Place_failure)'."""
        self.transitions.clear()
        last = self.states[-1] if self.states else None
        for i, s in enumerate(self.states):
            self.transitions[s] = {}
            fail_tok = f"{s}_failure"
            surv_tok = f"(1 - {fail_tok})"

            # failure arc
            self.transitions[s][fail_tok] = fail_tok

            if s != last:
                nxt = self.states[i + 1]
                self.transitions[s][nxt] = surv_tok
            else:
                # split survival over Done + start skills with numeric weights
                for dest, w in place_weights.items():
                    if dest == "Done" and "done" in self.absorbing_states:
                        self.transitions[s]["done"] = f"{w}*{surv_tok}"
                    elif dest in self.states:
                        self.transitions[s][dest] = f"{w}*{surv_tok}"

        for a in self.absorbing_states:
            self.transitions[a] = {a: 1.0}

    # --------------------- convenience accessors ---------------------

    def get_states(self) -> List[str]:
        return self.states

    def get_absorbing_states(self) -> List[str]:
        return self.absorbing_states

    def get_edges(self) -> Dict[str, List[str]]:
        return self.edges

    def get_transitions(self) -> Dict[str, Dict[str, Union[float, str]]]:
        return self.transitions

    def transition_matrix(self) -> np.ndarray:
        if self.P is None:
            raise RuntimeError("Call build_from_analyzer(...) first.")
        return self.P

    def initial_distribution(self) -> np.ndarray:
        if self.pi0 is None:
            raise RuntimeError("Call build_from_analyzer(...) first.")
        return self.pi0


class FaultTree:
    """
    Per-skill Fault Tree.
    Top event is always an OR gate.
    Children of the Top OR are:
      - Basic events (component failures), and/or
      - AND gates representing redundancy groups (e.g., loss_of_arm = AND(j1, j2, ...)).
    """

    def __init__(self, name: str, top_event: str = "", skill: str = ""):
        self.name: str = name
        self.top_event: str = top_event
        self.skill: str = skill

        # basic_events: { "j1": p, "controller": p, ... }
        self.basic_events: Dict[str, float] = {}

        # gates structure:
        # { "<top_event>": {"OR": ["j1", "loss_of_arm", ...]},
        #   "loss_of_arm": {"AND": ["j1", "j2"]}, ... }
        self.gates: Dict[str, Dict[str, List[str]]] = {}

        self.top_event_failure_prob: float = 0.0  # cache if you evaluate in-place

    # ------------------- housekeeping -------------------

    def clear_ft(self) -> None:
        self.name = ""
        self.top_event = ""
        self.skill = ""
        self.basic_events.clear()
        self.gates.clear()
        self.top_event_failure_prob = 0.0

    def set_top_event(self, top_event: str) -> bool:
        self.top_event = top_event
        return True

    def get_top_event(self) -> str:
        return self.top_event

    def set_skill(self, skill: str) -> bool:
        self.skill = skill
        return True

    def get_skill(self) -> str:
        return self.skill

    def set_top_event_failure_prob(self, p: float) -> bool:
        self.top_event_failure_prob = float(p)
        return True

    def get_top_event_failure_prob(self) -> float:
        return self.top_event_failure_prob

    # ------------------- basic events -------------------

    def add_single_basic_event(self, basic_event: str, prob: float) -> bool:
        self.basic_events[basic_event] = float(prob)
        return True

    def add_basic_events(self, basic_events: Dict[str, float]) -> bool:
        self.basic_events = {k: float(v) for k, v in basic_events.items()}
        return True

    def get_basic_events(self) -> Dict[str, float]:
        return self.basic_events

    def remove_basic_event(self, basic_event: str) -> bool:
        if basic_event not in self.basic_events:
            return False
        del self.basic_events[basic_event]
        return True

    # ------------------- gates -------------------

    def _ensure_top_or(self) -> None:
        if self.top_event == "":
            raise ValueError("Top event not set.")
        if self.top_event not in self.gates:
            self.gates[self.top_event] = {}
        if "OR" not in self.gates[self.top_event]:
            self.gates[self.top_event]["OR"] = []

    def add_redundant_group(self, group_name: str, members: Iterable[str]) -> str:
        """
        Create an AND gate 'loss_of_<group_name>' with the given members (basic event names).
        Returns the created gate name so you can OR it under the top event.
        """
        gate = f"loss_of_{group_name}"
        self.gates[gate] = {"AND": list(members)}
        return gate

    def add_single_gate(self, gate_name: str, gate_type: str, children: Iterable[str]) -> bool:
        """
        Add an arbitrary gate (useful if you pre-build a structure).
        If gate_name != top_event, you must also attach it under the top OR separately.
        """
        gate_type = gate_type.upper()
        if gate_type not in ("OR", "AND"):
            raise ValueError("gate_type must be 'OR' or 'AND'.")
        self.gates[gate_name] = {gate_type: list(children)}
        return True

    def add_gates(self, redundancy: bool = False, redundant_components: Optional[Dict[str, List[str]]] = None) -> bool:
        """
        Build Top OR:
          - If no redundancy: OR over all basic events
          - With redundancy: OR over (AND-gate per group) + any remaining basic events
        Does NOT delete from self.basic_events (no side effects).
        """
        redundant_components = redundant_components or {}
        self._ensure_top_or()

        # Clear any existing children under Top OR, we rebuild
        self.gates[self.top_event]["OR"] = []

        # Track which basics are already consumed by redundancy groups (for reporting only)
        consumed: set = set()

        if redundancy and len(redundant_components) > 0:
            # Create one AND gate per redundancy group and OR it under top
            for group_name, members in redundant_components.items():
                gate = self.add_redundant_group(group_name, members)
                self.gates[self.top_event]["OR"].append(gate)
                consumed.update(members)

            # Add any basic events not covered by groups directly under OR
            for be in self.basic_events:
                if be not in consumed:
                    self.gates[self.top_event]["OR"].append(be)

        else:
            # simple OR over all basics
            self.gates[self.top_event]["OR"].extend(list(self.basic_events.keys()))
        return True

    def get_gates(self) -> Dict[str, Dict[str, List[str]]]:
        return self.gates

    def remove_gate(self, gate_name: str) -> bool:
        if gate_name not in self.gates:
            return False
        del self.gates[gate_name]
        # also remove it from Top OR if present
        if self.top_event in self.gates and "OR" in self.gates[self.top_event]:
            self.gates[self.top_event]["OR"] = [
                c for c in self.gates[self.top_event]["OR"] if c != gate_name
            ]
        return True

    # ------------------- auto-create -------------------

    def auto_create_ft(
            self,
            basic_events: Dict[str, float],
            top_event: str = "",
            skill: str = "",
            redundancy: bool = False,
            redundant_components: Optional[Dict[str, List[str]]] = None,
    ) -> bool:
        """
        Build a FT in one call.
        - top_event: if not set already, required here
        - skill: optional label
        - basic_events: {component -> failure probability} for this skill (active components only)
        - redundancy: if True, use redundant_components to put AND sub-gates under Top OR
        - redundant_components: {"group_name": ["j1","j2"], "power": ["power_supply","controller"], ...}
        """
        if not self.top_event and not top_event:
            raise ValueError("No top_event provided.")
        if top_event:
            self.set_top_event(top_event)
        if skill:
            self.set_skill(skill)

        self.add_basic_events(basic_events)
        self.add_gates(redundancy=redundancy, redundant_components=redundant_components)
        return True

    # ------------------- validation & evaluation -------------------

    def validate(self) -> None:
        """
        Check that top OR exists and that all references point to either:
          - a basic event in self.basic_events, or
          - a gate in self.gates with a valid AND/OR list.
        Raises ValueError on problems.
        """
        self._ensure_top_or()

        # DFS over top OR children
        def _is_leaf(x: str) -> bool:
            return x in self.basic_events

        def _is_gate(x: str) -> bool:
            return x in self.gates and any(k in self.gates[x] for k in ("OR", "AND"))

        for child in self.gates[self.top_event]["OR"]:
            if not (_is_leaf(child) or _is_gate(child)):
                raise ValueError(f"Unknown node referenced under top OR: {child}")
            if _is_gate(child):
                kinds = list(self.gates[child].keys())
                if len(kinds) != 1 or kinds[0] not in ("OR", "AND"):
                    raise ValueError(f"Gate {child} must have exactly one of OR/AND.")
                if not isinstance(self.gates[child][kinds[0]], list):
                    raise ValueError(f"Gate {child} children must be a list.")

    def evaluate(self, set_cache: bool = True) -> float:
        """
        Exact evaluation under independence:
          P(AND(children)) = ∏ p(child)
          P(OR(children))  = 1 - ∏ (1 - p(child))
        Children may be basic events or sub-gates (recursively).
        """
        self.validate()

        def p_node(node: str) -> float:
            # leaf
            if node in self.basic_events:
                return float(self.basic_events[node])
            # gate
            gate = self.gates.get(node, {})
            if "AND" in gate:
                vals = [p_node(c) for c in gate["AND"]]
                out = 1.0
                for v in vals:
                    out *= float(v)
                return out
            if "OR" in gate:
                vals = [p_node(c) for c in gate["OR"]]
                prod = 1.0
                for v in vals:
                    prod *= (1.0 - float(v))
                return 1.0 - prod
            # unreachable if validate() passed
            return 0.0

        p_top = p_node(self.top_event)
        if set_cache:
            self.top_event_failure_prob = float(p_top)
        return float(p_top)

    # ------------------- export helpers -------------------

    def export(self) -> Dict[str, dict]:
        """Return a copy of the structure for solvers/logging."""
        return {
            "name": self.name,
            "top_event": self.top_event,
            "skill": self.skill,
            "basic_events": copy.deepcopy(self.basic_events),
            "gates": copy.deepcopy(self.gates),
            "top_event_failure_prob": float(self.top_event_failure_prob),
        }
