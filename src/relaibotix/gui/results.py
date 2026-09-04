"""Qt result dashboard backed by publication output files."""

from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


_BLUE = "#2563eb"
_TEAL = "#0f9f8f"
_ORANGE = "#e8792e"
_GRID = "#d0d5dd"


class PlotPanel(QWidget):
    """One embedded Matplotlib figure with native navigation and save controls."""

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self.figure = Figure(figsize=(5.0, 3.6), constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)
        self.message("Run an analysis to generate this plot.")

    def axes(self):
        self.figure.clear()
        axes = self.figure.add_subplot(111)
        axes.grid(axis="y", color=_GRID, linewidth=0.7, alpha=0.65)
        axes.set_axisbelow(True)
        return axes

    def message(self, text: str) -> None:
        axes = self.axes()
        axes.text(0.5, 0.5, text, ha="center", va="center", transform=axes.transAxes)
        axes.set_axis_off()
        self.canvas.draw_idle()


class ResultsView(QTabWidget):
    """Overview and four publication-oriented plots for one completed run."""

    def __init__(self) -> None:
        super().__init__()
        self.summary = QLabel("Validate an HDF5 file or run an analysis to begin.")
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        overview = QWidget()
        overview_layout = QVBoxLayout(overview)
        overview_layout.addWidget(self.summary)
        overview_layout.addStretch()

        self.joint_travel = PlotPanel()
        self.skill_duration = PlotPanel()
        behavior = QSplitter(Qt.Orientation.Horizontal)
        behavior.addWidget(self.joint_travel)
        behavior.addWidget(self.skill_duration)
        behavior.setSizes((500, 500))

        self.skill_failure = PlotPanel()
        self.sensitivity = PlotPanel()
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)

        self.addTab(overview, "Overview")
        self.addTab(behavior, "Behavior")
        self.addTab(self.skill_failure, "Reliability")
        self.addTab(self.sensitivity, "Sensitivity")
        self.addTab(self.log, "Details")

    def load_results(self, output_directory: str | Path) -> None:
        output = Path(output_directory)
        behavior = json.loads((output / "behavior" / "behavior.json").read_text())
        reliability = json.loads(
            (output / "reliability" / "reliability.json").read_text()
        )
        sensitivity_path = output / "reliability" / "sensitivity.csv"
        sensitivity = (
            pd.read_csv(sensitivity_path) if sensitivity_path.is_file() else pd.DataFrame()
        )

        segments = pd.DataFrame(behavior["segments"])
        joints = pd.DataFrame(behavior["joint_summary"])
        skills = pd.DataFrame(behavior["skill_summary"])
        component_failures = pd.DataFrame(reliability["component_failures"])
        skill_failure = pd.DataFrame(reliability["skill_probabilities"])
        distance_units = self._distance_units(component_failures)
        self._overview(output, segments, joints, reliability, sensitivity, distance_units)
        self._joint_travel(joints, distance_units)
        self._skill_duration(skills)
        self._skill_failure(skill_failure)
        self._sensitivity(sensitivity)

    def _overview(
        self,
        output: Path,
        segments: pd.DataFrame,
        joints: pd.DataFrame,
        reliability: dict[str, object],
        sensitivity: pd.DataFrame,
        distance_units: dict[str, str],
    ) -> None:
        runs = segments["episode_key"].astype(str).nunique()
        total_time = float(segments["duration"].sum())
        travel_by_unit: dict[str, float] = {}
        for row in joints.itertuples(index=False):
            unit = distance_units.get(str(row.joint))
            if unit:
                travel_by_unit[unit] = (
                    travel_by_unit.get(unit, 0.0) + float(row.total_traveled_distance)
                )
        travel_text = "; ".join(
            f"{value / runs:.2f} {unit}" for unit, value in sorted(travel_by_unit.items())
        ) or "not available"
        failure = float(reliability["dtmc"]["failure_probability"])
        mttf = float(reliability["repeated_run_mttf"]["hours"])
        critical = "not calculated"
        if not sensitivity.empty:
            critical = " and ".join(sensitivity.head(2)["component"].astype(str))
        solver_status = []
        for solver in ("prism", "storm"):
            solver_file = output / "reliability" / f"{solver}.json"
            if solver_file.is_file():
                payload = json.loads(solver_file.read_text())
                mode = "exact" if payload.get("exact") else "approximate"
                solver_status.append(f"{solver.upper()} {payload.get('version') or ''} ({mode}) ✓")
        solvers = ", ".join(solver_status) or "Internal solver"
        self.summary.setText(
            "<h3>Analysis complete</h3>"
            f"<p><b>Runs:</b> {runs}<br>"
            f"<b>Average time per run:</b> {total_time / runs:.2f} s<br>"
            f"<b>Average component travel per run:</b> {travel_text}<br>"
            f"<b>Failure probability per run:</b> {failure:.4e}<br>"
            f"<b>Repeated-operation MTTF:</b> {mttf:,.0f} h<br>"
            f"<b>Most influential components:</b> {html.escape(critical)}<br>"
            f"<b>Solver verification:</b> {html.escape(solvers)}</p>"
            "<p>Use the plot toolbar to zoom, inspect, or save a figure.</p>"
        )

    @staticmethod
    def _distance_units(component_failures: pd.DataFrame) -> dict[str, str]:
        if component_failures.empty or "distance_unit" not in component_failures:
            return {}
        result: dict[str, str] = {}
        for row in component_failures[["component", "distance_unit"]].itertuples(index=False):
            if pd.notna(row.distance_unit) and str(row.distance_unit).strip():
                result[str(row.component)] = str(row.distance_unit)
        return result

    def _joint_travel(self, joints: pd.DataFrame, distance_units: dict[str, str]) -> None:
        if joints.empty or not distance_units:
            self.joint_travel.message("No joint-distance data are available.")
            return
        values = (
            joints.groupby("joint", as_index=False)["total_traveled_distance"]
            .sum()
        )
        values["unit"] = values["joint"].astype(str).map(distance_units)
        values = values.dropna(subset=["unit"])
        units = sorted(values["unit"].unique())
        self.joint_travel.figure.clear()
        for index, unit in enumerate(units, start=1):
            axes = self.joint_travel.figure.add_subplot(1, len(units), index)
            subset = values[values["unit"] == unit].sort_values("total_traveled_distance")
            axes.barh(
                subset["joint"].astype(str).str.replace("_", " "),
                subset["total_traveled_distance"],
                color=_BLUE,
            )
            axes.set_title(f"Travel ({unit})")
            axes.set_xlabel(f"Traveled distance ({unit})")
            axes.grid(axis="x", color=_GRID, linewidth=0.7, alpha=0.65)
            axes.set_axisbelow(True)
        self.joint_travel.figure.suptitle("Cumulative traveled distance by component")
        self.joint_travel.canvas.draw_idle()

    def _skill_duration(self, skills: pd.DataFrame) -> None:
        if skills.empty:
            self.skill_duration.message("No skill-duration data are available.")
            return
        values = skills.sort_values("skill_id")
        axes = self.skill_duration.axes()
        axes.bar(values["skill"].astype(str), values["total_duration"], color=_TEAL)
        axes.set_title("Total recorded duration by skill")
        axes.set_ylabel("Duration (s)")
        axes.tick_params(axis="x", rotation=25)
        self.skill_duration.canvas.draw_idle()

    def _skill_failure(self, values: pd.DataFrame) -> None:
        if values.empty:
            self.skill_failure.message("No per-skill reliability data are available.")
            return
        values = values.sort_values("bdd_probability", ascending=False)
        axes = self.skill_failure.axes()
        axes.bar(values["skill"].astype(str), values["bdd_probability"], color=_ORANGE)
        logarithmic = (values["bdd_probability"] > 0.0).all()
        if logarithmic:
            axes.set_yscale("log")
        axes.set_title("Modeled failure probability per skill execution")
        axes.set_ylabel("Failure probability" + (" (log scale)" if logarithmic else ""))
        axes.tick_params(axis="x", rotation=25)
        self.skill_failure.canvas.draw_idle()

    def _sensitivity(self, values: pd.DataFrame) -> None:
        if values.empty:
            self.sensitivity.message("Sensitivity analysis was not requested for this run.")
            return
        values = values.sort_values("absolute_system_probability_change")
        axes = self.sensitivity.axes()
        axes.barh(
            values["component"].astype(str).str.replace("_", " "),
            values["absolute_system_probability_change"],
            color=_BLUE,
        )
        factor = float(values["requested_factor"].iloc[0])
        axes.set_title(f"Component sensitivity to a ×{factor:g} base-probability change")
        axes.set_xlabel("Absolute change in system failure probability")
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 2))
        axes.xaxis.set_major_formatter(formatter)
        axes.grid(axis="x", color=_GRID, linewidth=0.7, alpha=0.65)
        axes.grid(axis="y", visible=False)
        self.sensitivity.canvas.draw_idle()
