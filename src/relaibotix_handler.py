from json_handler import *
from behavioral_analysis import *
from pdf_handler import *
from relaibotix_helper import *
import argparse


def relaibotix(config_file: str,
               dataset_file: str,
               json_output: str,
               spider_chart_output: str,
               pdf_output: str):
    """
    Runs the reliability assessment using provided input and output files.

    Args:
        config_file (str): Path to the robot configuration JSON file.
        dataset_file (str): Path to the dataset (.npy file).
        json_output (str): Path to the output JSON file to write the robotic system data.
        spider_chart_output (str): Path to save the generated spider chart image.
        pdf_output (str): Path to save the generated PDF report.
    """
    # Load Franka config file
    robotic_data = read_json(config_file)
    component_names = robotic_data['components']
    robotic_name = robotic_data['robot']
    robotic_type = robotic_data['robot_type']

    # Create and populate the robotic system object
    robotic_system = RoboticSystem(robotic_name, robot_type=robotic_type)
    for component, reliability_data in component_names.items():
        failure_probability = reliability_data['failure_probability']
        redundancy = reliability_data['redundancy']
        robotic_system.add_component(Component(component, failure_prob=failure_probability, redundancy=redundancy))

    # Load dataset
    robot_data = load_data(dataset_file)

    # Create behavioral analysis profile and detect skills
    behavioral_profile = BehavioralAnalysis(name=robotic_system.get_name(),
                                            time_series=robot_data,
                                            robot_type=robotic_system.get_robot_type())
    behavioral_profile.detect_skill_sequence()
    skills = behavioral_profile.get_skill_sequence()
    add_skills(skills, robotic_system)

    # Determine active components and add them to skills
    behavioral_profile.analyze_active_components()
    active_components = behavioral_profile.get_active_components()
    add_components_to_skill(robotic_system, active_components)

    # Extract and classify properties, then add them to the system
    behavioral_profile.extract_properties()
    extracted_properties = behavioral_profile.get_extracted_properties()
    classified_properties = classify_property_dict(extracted_properties)
    if classified_properties:
        add_properties(classified_properties, robotic_system)

    # Create a hybrid reliability model and associated Markov chain
    hybrid_model = HybridReliabilityModel('Franka_hybrid')
    mc = MarkovChain('Franka')
    states = create_skill_list(robotic_system.get_skills())
    mc.auto_create_mc(states=states, done_state=True, repeat_info=1)
    hybrid_model.add_markov_chain(mc)

    # Create fault trees and compute system reliability
    hybrid_model = create_fault_trees(robotic_system, hybrid_model)
    ft_dict = create_ft_dict(hybrid_model)
    system_reliability, absorption_prob, absorption_time = hybrid_model.compute_system_reliability(
        ft_dict=ft_dict,
        repeat_dict={'done': 0.1, 'object_detection': 0.9}
    )
    robotic_system.set_system_failure_prob(float(system_reliability))
    add_skill_failure_prob(hybrid_model, robotic_system)

    # Write the updated robotic system to a JSON file
    write_json(robotic_system, json_output)

    # Perform sensitivity analysis and create a spider chart
    sensitivity_analysis_data = {'None': system_reliability}
    sensitivity_analysis_data = perform_sensitivity_analysis(hybrid_model, robotic_system, sensitivity_analysis_data)
    create_custom_spider_chart(sensitivity_analysis_data, title='System failure probability',
                               save_path=spider_chart_output)

    # Create PDF report from JSON and plots
    plot_files = [spider_chart_output]
    create_pdf_from_json_and_plots(json_output, plot_files, filename=pdf_output)

    print("Processing complete. Output files generated:")
    print(f"  JSON: {json_output}")
    print(f"  Spider Chart: {spider_chart_output}")
    print(f"  PDF Report: {pdf_output}")


def main():
    parser = argparse.ArgumentParser(description="Run RelAIBotiX.")
    parser.add_argument("--config", required=True, help="Path to the robot configuration JSON file")
    parser.add_argument("--dataset", required=True, help="Path to the dataset (.npy file)")
    parser.add_argument("--json_output", required=True, help="Path to the output JSON file")
    parser.add_argument("--spider_chart", required=True, help="Path to save the spider chart image")
    parser.add_argument("--pdf_output", required=True, help="Path to save the PDF report")

    args = parser.parse_args()

    relaibotix(
        config_file=args.config,
        dataset_file=args.dataset,
        json_output=args.json_output,
        spider_chart_output=args.spider_chart,
        pdf_output=args.pdf_output
    )


if __name__ == "__main__":
    main()
