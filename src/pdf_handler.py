import json
from fpdf import FPDF
from PIL import Image
import os


def create_pdf_from_json_and_plots(json_file, plot_files, filename='output.pdf'):
    """
    Creates a PDF that includes information from a JSON file and actual plot images.

    Parameters:
    - json_file (str): Path to the JSON file containing the data.
    - plot_files (list): List of paths to plot image files to include in the PDF.
    - filename (str): The filename for the generated PDF.

    Returns:
    - None: Saves the PDF to the specified file.
    """
    # Load data from JSON file
    with open(json_file, 'r') as file:
        data = json.load(file)

    # Create instance of FPDF class
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Add a page
    pdf.add_page()

    # Set title
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, f"System Report: {data['name']}", ln=True, align='C')
    pdf.ln(10)  # Line break

    # Add System Failure Probability
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'System Failure Probability:', ln=True)
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f"{data['system_failure_prob']:.2E}", ln=True)

    pdf.ln(10)  # Line break

    # Add Components Information
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Components:', ln=True)

    pdf.set_font('Arial', '', 12)
    for component in data['components']:
        pdf.cell(0, 10, f"Component: {component['name']}", ln=True)
        pdf.cell(0, 10, f"  Failure Probability: {component['failure_prob']:.2E}", ln=True)
        """
        if component['properties']:
            pdf.cell(0, 10, '  Properties:', ln=True)
            for prop_name, prop_values in component['properties'].items():
                pdf.cell(0, 10, f"    {prop_name.capitalize()}:", ln=True)
                for task, value in prop_values.items():
                    pdf.cell(0, 10, f"      {task}: {value}", ln=True)
        if component['skills']:
            pdf.cell(0, 10, '  Skills:', ln=True)
            for skill in component['skills']:
                pdf.cell(0, 10, f"    {skill}", ln=True)
        """
        pdf.ln(5)  # Line break between components

    pdf.ln(10)  # Line break before skills section

    # Add Skills Information
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Skills:', ln=True)

    pdf.set_font('Arial', '', 12)
    for skill in data['skills']:
        pdf.cell(0, 10, f"Skill: {skill['name']}", ln=True)
        pdf.cell(0, 10, f"  Failure Probability: {skill['skill_failure_prob']:.2E}", ln=True)
        # pdf.cell(0, 10, '  Components:', ln=True)
        """
        for component in skill['components']:
            pdf.cell(0, 10, f"    {component}", ln=True)
        if skill['failure_modes']:
            pdf.cell(0, 10, '  Failure Modes:', ln=True)
            for mode in skill['failure_modes']:
                pdf.cell(0, 10, f"    {mode}", ln=True)
        """
        pdf.ln(5)  # Line break between skills

    pdf.ln(10)  # Line break before plots

    # Insert each plot into the PDF
    for plot_file in plot_files:
        if os.path.exists(plot_file):
            pdf.add_page()  # Start each plot on a new page
            pdf.image(plot_file, x=10, y=20, w=pdf.w - 20)
            pdf.ln(10)  # Line break after the plot
        else:
            print(f"Warning: {plot_file} does not exist and will be skipped.")

    # Save the pdf with name .pdf
    pdf.output(filename)
