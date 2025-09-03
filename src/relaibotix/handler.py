from relaibotix.behavioral.behavioral_analysis_v2 import *
from relaibotix.evaluation.evaluation import *
from relaibotix.evaluation.pdf_handler import *
import pandas as pd
import h5py
from typing import Dict, Union, Any, List, Optional
from pathlib import Path

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
    "/Users/Phips1900/PhD/Research/RelAIBotiX/datasets/franka_trial_summary_100.csv")

# 3) Analyze
an = BehavioralAnalyzer(pos_success_tol=0.02)  # tune eps_abs/dc_thr/rms_thr/range_thr later if needed
traces: List[RunTrace] = an.analyze(
    features=features_df,
    feature_names=feat_names,
    labels=labels,
    timestamps=timestamps,
    trials_csv=trials_csv
)

summary = an.summarize(traces)

base_p = an.load_base_probs_from_json("/Users/Phips1900/PhD/Research/RelAIBotiX/config_files/robots/franka_config.json")

joint_failure_table = an.assess_failure_from_bands(traces)
gripper_table = an.assess_failure_for_gripper(traces)
failure_table = combine_failure_tables(joint_failure_table, gripper_table)

base_p_per_min = getattr(an, "base_prob_per_minute", {})
redundancy_flags = getattr(an, "redundancy", {})

# 2) Add always-active components (controller, power_supply, sensors, camera)
failure_table_aug = augment_failure_table_with_always_active(
    failure_table=failure_table,
    summary=summary,
    base_p_per_min=base_p_per_min,
    components_to_add=["controller","power_supply","sensors","camera"]
)

# 3) Build FaultTrees (per-skill) with basic redundancy expansion (Controller_1/_2, ...)
fts = build_fault_trees_from_failure_table_basic(
    failure_table=failure_table_aug,
    redundancy_flags=redundancy_flags,
    copies=2
)
hybrid_model = HybridReliabilityModel('RelAIBotiX-hybrid')
for ft in fts:
    hybrid_model.add_fault_tree(ft)
hybrid_model.failure_table = failure_table_aug
hybrid_model.redundancy_map = redundancy_flags

ft_dict = create_ft_dict(hybrid_model)

# 4) Build DTMC from summary (symbolic; bind later inside solver, or bind now if you prefer)
mc = build_dtmc_from_summary(summary, failure_per_skill=None, done_alpha=1.0)
hybrid_model.add_markov_chain(mc)


system_reliability, absorption_prob, absorption_time = hybrid_model.compute_system_reliability(ft_dict=ft_dict)

print(system_reliability)
print(absorption_prob)

out_dir = Path("/Users/Phips1900/PhD/Research/RelAIBotiX/artifacts/reports/new")
out_dir.mkdir(parents=True, exist_ok=True)

# 1) Per-skill failure probs (from solved FTs)
skill_pf = skill_failure_probs_from_fts(hybrid_model)

# 2) Task success rate from analyzer summary
task_success = task_success_rate_from_summary(summary)


fail_abs_states = [s for s in mc.get_absorbing_states() if s.endswith("_failure")]
skills = [s[:-8] for s in fail_abs_states]  # strip "_failure"

probs = absorption_prob
if len(probs) == len(fail_abs_states) + 1:
    probs = probs[:len(fail_abs_states)]  # drop 'done' column if present

assert len(probs) == len(skills), f"length mismatch: {len(probs)} vs {len(skills)}"

skill_absorb_fail = dict(zip(skills, map(float, probs)))

# Sensitivity
sens_df = sensitivity_analysis(failure_table=failure_table_aug, redundancy_flags=redundancy_flags, summary=summary,
                               factor=10.0)

sens_df.to_csv(out_dir / "sensitivity.csv", index=False)

plots = []

# Plots
ft_png, ab_png = plot_skill_failures_separate(
    ft_failure=skill_pf,
    absorb_failure=skill_absorb_fail,
    out_ft=out_dir / "skills_ft.png",
    out_absorb=out_dir / "skills_absorbing.png",
    order=skills,  # keeps bars aligned to your DTMC order
)
plots.append(ft_png)
plots.append(ab_png)

p_2 = plot_sensitivity_outcomes_spider_failure(
    base_system_failure=system_reliability,
    sens_df=sens_df,
    outpath=out_dir / "sensitivity_spider.png",
)
plots.append(p_2)

# 5) JSON payload and PDF
report_json = write_report_json(
    name="RelAIBotiX",
    system_failure_prob=system_reliability,
    task_success_rate_percent=task_success,
    base_probs=base_p_per_min,
    skill_pf=skill_pf,
    outpath=out_dir / "report.json",
)
create_pdf_from_json_and_plots(str(report_json), [str(p) for p in plots], filename=str(out_dir / "report.pdf"))
print(f"[OK] Wrote evaluation to {out_dir.resolve()}")
