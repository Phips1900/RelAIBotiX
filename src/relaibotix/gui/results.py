"""Qt result dashboard backed by publication output files."""

from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd
from matplotlib import colormaps
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.ticker import ScalarFormatter
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QSplitter,
    QTableView,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


_BLUE = "#2563eb"
_TEAL = "#0f9f8f"
_ORANGE = "#e8792e"
_GRID = "#d0d5dd"
_EXPOSURE_COLORS = ("#22a06b", "#f5b700", "#d64545")


class DataFrameModel(QAbstractTableModel):
    """Small sortable adapter for analysis result tables."""

    def __init__(self, frame: pd.DataFrame | None = None) -> None:
        super().__init__()
        self.frame = (frame if frame is not None else pd.DataFrame()).copy()

    def rowCount(self, parent=QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self.frame)

    def columnCount(self, parent=QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self.frame.columns)

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        value = self.frame.iat[index.row(), index.column()]
        if role == Qt.ItemDataRole.DisplayRole:
            if pd.isna(value):
                return ""
            if isinstance(value, float):
                return f"{value:.6g}" if value == 0.0 or abs(value) >= 1e-3 else f"{value:.4e}"
            return str(value)
        if role == Qt.ItemDataRole.TextAlignmentRole and isinstance(value, (int, float)):
            return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        return None

    def headerData(self, section: int, orientation, role=Qt.ItemDataRole.DisplayRole):  # noqa: N802
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return str(self.frame.columns[section]).replace("_", " ")
        return str(section + 1)

    def sort(self, column: int, order=Qt.SortOrder.AscendingOrder) -> None:
        if self.frame.empty:
            return
        self.layoutAboutToBeChanged.emit()
        self.frame = self.frame.sort_values(
            self.frame.columns[column],
            ascending=order == Qt.SortOrder.AscendingOrder,
            kind="stable",
        ).reset_index(drop=True)
        self.layoutChanged.emit()


class DataTable(QTableView):
    def __init__(self) -> None:
        super().__init__()
        self.setAlternatingRowColors(True)
        self.setSortingEnabled(True)
        self.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)

    def set_frame(self, frame: pd.DataFrame) -> None:
        self.setModel(DataFrameModel(frame))
        self.resizeColumnsToContents()


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
    """Interactive plots and sortable tables for one completed run."""

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

        self.velocity_exposure = PlotPanel()
        self.effort_exposure = PlotPanel()
        exposure = QSplitter(Qt.Orientation.Horizontal)
        exposure.addWidget(self.velocity_exposure)
        exposure.addWidget(self.effort_exposure)
        exposure.setSizes((500, 500))

        self.episode_box = QComboBox()
        self.episode_box.currentTextChanged.connect(self._timeline)
        self.timeline = PlotPanel()
        timeline = QWidget()
        timeline_layout = QVBoxLayout(timeline)
        timeline_controls = QHBoxLayout()
        timeline_controls.addWidget(QLabel("Recorded run"))
        timeline_controls.addWidget(self.episode_box, 1)
        timeline_layout.addLayout(timeline_controls)
        timeline_layout.addWidget(self.timeline, 1)

        behavior_tabs = QTabWidget()
        behavior_tabs.addTab(behavior, "Summary")
        behavior_tabs.addTab(exposure, "Velocity and effort")
        behavior_tabs.addTab(timeline, "Skill timeline")

        self.skill_failure = PlotPanel()
        self.sensitivity = PlotPanel()
        self.tables = QTabWidget()
        self.skill_table = DataTable()
        self.joint_table = DataTable()
        self.skill_reliability_table = DataTable()
        self.component_table = DataTable()
        self.sensitivity_table = DataTable()
        self.tables.addTab(self.skill_table, "Skills")
        self.tables.addTab(self.joint_table, "Components")
        self.tables.addTab(self.skill_reliability_table, "Skill reliability")
        self.tables.addTab(self.component_table, "Reliability details")
        self.tables.addTab(self.sensitivity_table, "Sensitivity")
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self._segments = pd.DataFrame()

        self.addTab(overview, "Overview")
        self.addTab(behavior_tabs, "Behavior")
        self.addTab(self.skill_failure, "Reliability")
        self.addTab(self.sensitivity, "Sensitivity")
        self.addTab(self.tables, "Tables")
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
        self._exposure(joints, "velocity")
        self._exposure(joints, "effort")
        self._load_timeline(segments)
        self._skill_failure(skill_failure)
        self._sensitivity(sensitivity)
        self.skill_table.set_frame(skills)
        self.joint_table.set_frame(joints)
        self.skill_reliability_table.set_frame(skill_failure)
        self.component_table.set_frame(component_failures)
        self.sensitivity_table.set_frame(sensitivity)

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

    def _exposure(self, joints: pd.DataFrame, kind: str) -> None:
        panel = self.velocity_exposure if kind == "velocity" else self.effort_exposure
        columns = [f"{kind}_time_low", f"{kind}_time_medium", f"{kind}_time_high"]
        if joints.empty or not set(columns).issubset(joints.columns):
            panel.message(f"No {kind} exposure data are available.")
            return
        values = joints.groupby("joint", as_index=False)[columns].sum()
        values["total"] = values[columns].sum(axis=1)
        values = values[values["total"] > 0.0].sort_values("total")
        if values.empty:
            panel.message(f"No measured {kind} exposure is available for this robot configuration.")
            return
        axes = panel.axes()
        left = pd.Series(0.0, index=values.index)
        labels = ("Low", "Medium", "High")
        for column, label, color in zip(columns, labels, _EXPOSURE_COLORS):
            axes.barh(
                values["joint"].astype(str).str.replace("_", " "),
                values[column],
                left=left,
                label=label,
                color=color,
            )
            left = left + values[column]
        axes.set_title(f"Time in {kind} exposure bands")
        axes.set_xlabel("Cumulative exposure time (s)")
        axes.grid(axis="x", color=_GRID, linewidth=0.7, alpha=0.65)
        axes.grid(axis="y", visible=False)
        axes.legend(loc="lower right")
        panel.canvas.draw_idle()

    def _load_timeline(self, segments: pd.DataFrame) -> None:
        self._segments = segments.copy()
        self.episode_box.blockSignals(True)
        self.episode_box.clear()
        if not segments.empty:
            self.episode_box.addItems(segments["episode_key"].astype(str).drop_duplicates().tolist())
        self.episode_box.blockSignals(False)
        self._timeline(self.episode_box.currentText())

    def _timeline(self, episode_key: str) -> None:
        if self._segments.empty or not episode_key:
            self.timeline.message("No episode timeline is available.")
            return
        values = self._segments[
            self._segments["episode_key"].astype(str) == str(episode_key)
        ].sort_values("segment_index")
        start = float(values["start_time"].min())
        axes = self.timeline.axes()
        color_map = colormaps["tab10"]
        legend: dict[str, object] = {}
        for row in values.itertuples(index=False):
            left = float(row.start_time) - start
            duration = max(float(row.duration), 1e-9)
            color = color_map(int(row.skill_id) % 10)
            axes.broken_barh([(left, duration)], (0, 8), facecolors=color)
            legend.setdefault(str(row.skill), color)
            if duration >= max(float(values["duration"].sum()) * 0.04, 0.2):
                axes.text(
                    left + duration / 2.0,
                    4,
                    str(row.skill),
                    ha="center",
                    va="center",
                    fontsize=8,
                    clip_on=True,
                )
        axes.set_ylim(0, 8)
        axes.set_yticks([])
        axes.set_xlabel("Time from start of recorded run (s)")
        axes.set_title(f"Skill sequence — {episode_key}")
        axes.grid(axis="x", color=_GRID, linewidth=0.7, alpha=0.65)
        axes.legend(
            handles=[Patch(facecolor=color, label=name) for name, color in legend.items()],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=min(5, len(legend)),
        )
        self.timeline.canvas.draw_idle()

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
