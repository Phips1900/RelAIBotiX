import argparse

from relaibotix.behavioral.behavioral_analysis_v2 import *
from relaibotix.evaluation.evaluation import *
from relaibotix.evaluation.pdf_handler import *
from relaibotix.reliability.prism import *


def run_relaibotix(h5_path: str,
                   checkpoint_path: str,
                   config_path: str,
                   out_dir: str = "artifacts/reports",
                   out_dir_prism: str = "artifacts/prism", ):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir_prism = Path(out_dir_prism)
    out_dir_prism.mkdir(parents=True, exist_ok=True)

    run_inference(h5_path=h5_path,
                  checkpoint_path=checkpoint_path,
                  model_type="cnn_transformer",
                  window_size=500,
                  feature_columns=[0, 1, 2, 3, 4, 5, 6, 17],
                  num_classes=5,
                  batch_size=64,
                  device="mps",
                  out_labels_name="labels_pred",
                  episode_ids_ds="labels",
                  overwrite=True,
                  stride=1,
                  )

    with h5py.File(h5_path, "r") as f:
        features_arr = f["features"][()]  # (N, 22)
        episode_labels = f["labels"][()]  # (N,)
        labels_pred = f["labels_pred"][()]  # (N,)
        timestamps = f["timestamps"][()]  # (N,)
        feat_names = [n.decode() if hasattr(n, "decode") else str(n)
                      for n in f["features"].attrs["feature_names"]]
    features_df = pd.DataFrame(features_arr)
    labels_pred_filter = filter_short_segments(labels_pred)
    an = BehavioralAnalyzer()
    traces: List[RunTrace] = an.analyze(
        features=features_df,
        feature_names=feat_names,
        labels=labels_pred_filter,
        timestamps=timestamps,
        episode_labels=episode_labels
    )
    summary = an.summarize(traces)
    base_p = an.load_base_probs_from_json(config_path)
    joint_failure_table = an.assess_failure_from_bands(traces)
    gripper_table = an.assess_failure_for_gripper(traces)
    failure_table = combine_failure_tables(joint_failure_table, gripper_table)

    base_p_per_min = getattr(an, "base_prob_per_minute", {})
    redundancy_flags = getattr(an, "redundancy", {})

    failure_table_aug = augment_failure_table_with_always_active(
        failure_table=failure_table,
        summary=summary,
        base_p_per_min=base_p_per_min,
        components_to_add=["controller", "power_supply", "sensors", "camera"]
    )

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
    mc = build_dtmc_from_summary(summary, failure_per_skill=None, done_alpha=1.0)
    hybrid_model.add_markov_chain(mc)

    system_reliability, absorption_prob, absorption_time = hybrid_model.compute_system_reliability(ft_dict=ft_dict)

    mc = hybrid_model.get_markov_chain()
    prism_model, prism_props = write_prism_and_props(
        mc,
        out_basename=out_dir_prism / "prism_model",
        model_name="RelAIBotiX",
        precision=12,
    )
    print(f"[PRISM] wrote {prism_model} and {prism_props}")

    skill_time = summary.get("skill_time", {})
    state_time_seconds = {
        str(r.skill): float(r.avg_time_per_episode_sec)
        for r in skill_time.itertuples(index=False)
    }

    write_prism_no_done_and_props(
        mc,
        out_basename=out_dir_prism / "prism_model_no_done",
        state_time_seconds=state_time_seconds,
    )

    skill_pf = skill_failure_probs_from_fts(hybrid_model)

    fail_abs_states = [s for s in mc.get_absorbing_states() if s.endswith("_failure")]
    skills = [s[:-8] for s in fail_abs_states]  # strip "_failure"

    probs = absorption_prob
    if len(probs) == len(fail_abs_states) + 1:
        probs = probs[:len(fail_abs_states)]  # drop 'done' column if present

    assert len(probs) == len(skills), f"length mismatch: {len(probs)} vs {len(skills)}"

    skill_absorb_fail = dict(zip(skills, map(float, probs)))

    sens_df = sensitivity_analysis(failure_table=failure_table_aug, redundancy_flags=redundancy_flags, summary=summary,
                                   factor=10.0)

    sens_df.to_csv(out_dir / "sensitivity.csv", index=False)
    plots = []
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
    vel_plots = plot_velocity_bands_per_skill(velocity_bands=summary.get("velocity_bands", {}), outpath=out_dir)
    eff_plots = plot_effort_bands_per_skill(effort_bands=summary.get("effort_bands", {}), outpath=out_dir)

    for plot in vel_plots:
        plots.append(plot)

    if eff_plots:
        for plot in eff_plots:
            plots.append(plot)

    report_json = write_report_json_extended(
        name="RelAIBotiX - Reliability Report",
        system_failure_prob=system_reliability,
        base_probs=base_p_per_min,
        skill_pf=skill_pf,
        summary=summary,
        outpath=out_dir / "reliability_report.json",
    )

    create_pdf_extended(str(report_json), [str(p) for p in plots], filename=str(out_dir / "reliability_report.pdf"))


def run_skill_detector(h5_path: str, checkpoint_path: str,
                       out_dir: str = "artifacts/skill_detector",
                       ):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_inference(h5_path=h5_path,
                  checkpoint_path=checkpoint_path,
                  model_type="cnn_transformer",
                  window_size=500,
                  feature_columns=[0, 1, 2, 3, 4, 5, 6, 17],
                  num_classes=5,
                  batch_size=64,
                  device="mps",
                  out_labels_name="labels_pred",
                  episode_ids_ds="labels",
                  overwrite=True,
                  stride=1,
                  )
    with h5py.File(h5_path, "r") as f:
        labels_pred = f["labels_pred"][()]  # (N,)

    labels_pred_filter = filter_short_segments(labels_pred)
    labels_df = pd.DataFrame(labels_pred_filter)
    labels_df.to_csv(out_dir / "labels_pred.csv", index=False)


def _cli_relaibotix():
    parser = argparse.ArgumentParser(description="Run RelAIBotiX pipeline")
    parser.add_argument("--h5", required=True, help="Path to H5 dataset")
    parser.add_argument("--ckpt", help="Path to checkpoint file",
                        default="artifacts/checkpoints/skill_detector.ckpt")
    parser.add_argument("--config", help="Path to robot config JSON",
                        default="config_files/robots/so_arm_config.json")
    parser.add_argument("--output", default="artifacts/reports", help="Report output directory")
    parser.add_argument("--prism", default="artifacts/prism", help="PRISM output directory")
    args = parser.parse_args()

    run_relaibotix(
        h5_path=args.h5,
        checkpoint_path=args.ckpt,
        config_path=args.config,
        out_dir=args.output,
        out_dir_prism=args.prism
    )


def _cli_skill_detector():
    parser = argparse.ArgumentParser(description="Run skill detector only")
    parser.add_argument("--h5", required=True, help="Path to H5 dataset")
    parser.add_argument("--ckpt", help="Path to checkpoint file",
                        default="artifacts/checkpoints/skill_detector.ckpt")
    parser.add_argument("--output", default="artifacts/skill_detector", help="Output directory")
    args = parser.parse_args()

    run_skill_detector(
        h5_path=args.h5,
        checkpoint_path=args.ckpt,
        out_dir=args.output,
    )
