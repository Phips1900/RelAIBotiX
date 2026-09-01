"""
Description: This file contains the functions to create the reliability report as pdf.
"""
import os
import json
from fpdf import FPDF
from typing import List, Sequence


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
    pdf.set_font('Arial', '', 14)
    pdf.cell(0, 10, f"{data['system_failure_prob']:.2E}", ln=True)

    pdf.ln(10)  # Line break

    # Add Components Information
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Components:', ln=True)

    pdf.set_font('Arial', '', 12)
    for component in data['components']:
        pdf.cell(0, 10, f"Component: {component['name']}", ln=True)
        pdf.cell(0, 10, f"  Failure Probability: {component['failure_prob']:.2E}", ln=True)
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


def _sec_to_str(s):
    try: s = float(s)
    except: return str(s)
    m, sec = divmod(s, 60.0)
    h, m = divmod(m, 60.0)
    if h >= 1: return f"{int(h)}h {int(m)}m {sec:04.1f}s"
    if m >= 1: return f"{int(m)}m {sec:04.1f}s"
    return f"{sec:.2f}s"


def _table(pdf: FPDF, rows, col_w, line_h=8):
    for row in rows:
        for w, txt in zip(col_w, row):
            pdf.cell(w, line_h, txt, border=1)
        pdf.ln(line_h)


def create_pdf_extended(json_file: str, plot_files, filename: str = "output.pdf"):
    with open(json_file, "r") as f:
        data = json.load(f)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Header
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, f"System Report: {data.get('name','')}", ln=True, align="C")
    pdf.ln(4)

    # Top stats
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, f"System failure probability: {data.get('system_failure_prob', 0.0):.2E}", ln=True)
    ov = data.get("overall", {})
    pdf.cell(0, 8, f"Runs: {ov.get('n_runs',0)}", ln=True)
    pdf.cell(0, 8, f"Total run time: {_sec_to_str(ov.get('total_run_time_sec',0.0))}", ln=True)
    pdf.ln(3)

    # --- Per-skill timing table (unchanged, still useful) ---
    st = data.get("skill_time", [])
    if st:
        pdf.set_font("Arial", "B", 13)
        pdf.cell(0, 9, "Per-skill timing", ln=True)
        pdf.set_font("Arial", "", 11)
        rows = [["Skill", "Total time", "Avg/episode", "Avg/run"]]
        for r in st:
            rows.append([
                str(r.get("skill","")),
                _sec_to_str(r.get("total_time_sec",0.0)),
                _sec_to_str(r.get("avg_time_per_episode_sec",0.0)),
                _sec_to_str(r.get("avg_time_per_run_sec",0.0)),
            ])
        _table(pdf, rows, col_w=[45, 45, 45, 45], line_h=8)
        pdf.ln(2)

    # --- Component usage (ALL rows; trimmed columns) ---
    cu = data.get("component_usage", [])
    if cu:
        pdf.set_font("Arial", "B", 13)
        pdf.cell(0, 9, "Component usage", ln=True)
        pdf.set_font("Arial", "", 11)

        # columns: Skill | Component | Active% | Active time (sum) | Active time (avg)
        rows = [["Skill","Component","Active%","Active time (sum)","Active time (avg)"]]
        for r in cu:
            rows.append([
                str(r.get("skill","")),
                str(r.get("component","")),
                f"{float(r.get('active_pct_episodes',0.0)):.1f}%",
                _sec_to_str(r.get("total_active_time_sec",0.0)),
                _sec_to_str(r.get("avg_active_time_sec",0.0)),
            ])
        # widths sum ~ 185 (fits A4 portrait with default margins)
        _table(pdf, rows, col_w=[30, 60, 20, 37, 38], line_h=8)
        pdf.ln(2)

    # --- Velocity bands (ALL rows; trimmed columns) ---
    vb = data.get("velocity_bands", [])
    if vb:
        pdf.set_font("Arial", "B", 13)
        pdf.cell(0, 9, "Joint velocity bands", ln=True)
        pdf.set_font("Arial", "", 11)

        # columns: Skill | Joint | Low | Med | High
        rows = [["Skill","Joint","Low","Med","High"]]
        for r in vb:
            rows.append([
                str(r.get("skill","")),
                str(r.get("component","")),
                _sec_to_str(r.get("time_low_sec",0.0)),
                _sec_to_str(r.get("time_med_sec",0.0)),
                _sec_to_str(r.get("time_high_sec",0.0)),
            ])
        _table(pdf, rows, col_w=[40, 35, 35, 35, 35], line_h=8)

    # Plots (unchanged)
    for plot_file in plot_files:
        if os.path.exists(plot_file):
            pdf.add_page()
            pdf.image(plot_file, x=10, y=20, w=pdf.w - 20)

    pdf.output(filename)
    return filename
