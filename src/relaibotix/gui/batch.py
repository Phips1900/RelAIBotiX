"""Native publication-experiment batch screen."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from matplotlib import colormaps
from PySide6.QtCore import QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .results import DataTable, PlotPanel, _GRID


class BatchPage(QWidget):
    """Configure, execute, and inspect a publication experiment manifest."""

    run_requested = Signal(list)

    def __init__(self) -> None:
        super().__init__()
        root_path = Path(__file__).resolve().parents[3]
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        configuration = QGroupBox("Publication experiment set")
        form = QFormLayout(configuration)
        self.manifest_path = QLineEdit(str(root_path / "configs" / "experiments" / "paper.json"))
        manifest_row = QHBoxLayout()
        manifest_row.addWidget(self.manifest_path)
        manifest_button = QPushButton("Browse…")
        manifest_button.setMaximumWidth(100)
        manifest_button.clicked.connect(self._browse_manifest)
        manifest_row.addWidget(manifest_button)
        form.addRow("Manifest", manifest_row)

        self.output_path = QLineEdit(str(root_path / "artifacts" / "paper_validation_gui"))
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_path)
        output_button = QPushButton("Browse…")
        output_button.setMaximumWidth(100)
        output_button.clicked.connect(self._browse_output)
        output_row.addWidget(output_button)
        form.addRow("Output folder", output_row)

        self.include_optional = QCheckBox("Include optional experiments such as SO-ARM MuJoCo")
        self.include_optional.setChecked(True)
        form.addRow("Scope", self.include_optional)

        self.prism = QCheckBox("PRISM exact")
        self.storm = QCheckBox("STORM exact")
        solver_row = QHBoxLayout()
        solver_row.addWidget(self.prism)
        solver_row.addWidget(self.storm)
        solver_row.addStretch()
        form.addRow("Verification", solver_row)
        self.prism_path = QLineEdit("prism")
        self.storm_path = QLineEdit("storm")
        form.addRow("PRISM executable", self.prism_path)
        form.addRow("STORM executable", self.storm_path)
        layout.addWidget(configuration)

        actions = QHBoxLayout()
        note = QLabel("This workflow uses the stored, reviewed detector predictions declared by the legacy manifest.")
        note.setWordWrap(True)
        note.setStyleSheet("color: #667085;")
        actions.addWidget(note, 1)
        self.run_button = QPushButton("Run experiment set")
        self.run_button.clicked.connect(self._request_run)
        actions.addWidget(self.run_button)
        layout.addLayout(actions)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        self.summary = QLabel("Select a manifest and run the experiment set.")
        self.summary.setWordWrap(True)
        summary_page = QWidget()
        summary_layout = QVBoxLayout(summary_page)
        summary_layout.addWidget(self.summary)
        summary_layout.addStretch()

        self.failure_plot = PlotPanel()
        self.mttf_plot = PlotPanel()
        comparison = QSplitter()
        comparison.addWidget(self.failure_plot)
        comparison.addWidget(self.mttf_plot)
        comparison.setSizes((550, 550))

        self.table = DataTable()
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.results = QTabWidget()
        self.results.addTab(summary_page, "Overview")
        self.results.addTab(comparison, "Policy comparison")
        self.results.addTab(self.table, "Paper table")
        self.results.addTab(self.log, "Details")
        layout.addWidget(self.results, 1)

        open_output = QPushButton("Open batch output folder")
        open_output.clicked.connect(self._open_output)
        layout.addWidget(open_output)

    def _request_run(self) -> None:
        manifest = Path(self.manifest_path.text())
        if not manifest.is_file():
            QMessageBox.warning(self, "Missing manifest", "Please select an existing experiment manifest.")
            return
        if not self.output_path.text().strip():
            QMessageBox.warning(self, "Missing output", "Please select an output folder.")
            return
        arguments = [
            "experiments", "run", str(manifest), "--output", self.output_path.text()
        ]
        if not self.include_optional.isChecked():
            arguments.append("--exclude-optional")
        if self.prism.isChecked():
            arguments.extend(("--prism", "--prism-executable", self.prism_path.text()))
        if self.storm.isChecked():
            arguments.extend(("--storm", "--storm-executable", self.storm_path.text()))
        self.set_running(True)
        self.log.clear()
        self.results.setCurrentWidget(self.log)
        self.run_requested.emit(arguments)

    def set_running(self, running: bool) -> None:
        self.run_button.setEnabled(not running)
        self.progress.setRange(0, 0 if running else 1)
        if not running:
            self.progress.setValue(1)

    def append_log(self, text: str) -> None:
        self.log.appendPlainText(text)

    def finished(self, exit_code: int) -> None:
        self.set_running(False)
        if exit_code != 0:
            self.summary.setText("<h3>Experiment set did not complete</h3><p>Review Details for the cause.</p>")
            self.results.setCurrentWidget(self.log)
            return
        try:
            self.load_results(self.output_path.text())
        except Exception as error:
            self.summary.setText(
                f"<h3>Experiment set complete</h3><p>The combined dashboard could not be loaded: {error}</p>"
            )
        self.results.setCurrentIndex(0)

    def failed(self, detail: str) -> None:
        self.set_running(False)
        self.log.appendPlainText(detail)
        self.summary.setText("<h3>Experiment set failed</h3><p>Review Details for the cause.</p>")
        self.results.setCurrentWidget(self.log)

    def load_results(self, output_directory: str | Path) -> None:
        output = Path(output_directory)
        frame = pd.read_csv(output / "paper_results.csv")
        self.table.set_frame(frame)
        included = int((frame["scope"] == "included").sum())
        optional = int((frame["scope"] == "optional").sum())
        self.summary.setText(
            "<h3>Experiment set complete</h3>"
            f"<p><b>Experiments:</b> {len(frame)}<br>"
            f"<b>Included:</b> {included}<br>"
            f"<b>Optional:</b> {optional}</p>"
            "<p>CSV, Markdown, LaTeX, and provenance outputs are available in the batch output folder.</p>"
        )
        self._comparison_plot(
            self.failure_plot,
            frame,
            "system_failure_probability_per_run",
            "Failure probability per run",
            logarithmic=True,
        )
        self._comparison_plot(
            self.mttf_plot,
            frame,
            "mttf_h",
            "Repeated-operation MTTF (h)",
            logarithmic=False,
        )

    @staticmethod
    def _comparison_plot(
        panel: PlotPanel,
        frame: pd.DataFrame,
        column: str,
        title: str,
        *,
        logarithmic: bool,
    ) -> None:
        values = frame.copy()
        values["label"] = values["task"].astype(str) + " — " + values["policy"].astype(str)
        values = values.sort_values(column)
        tasks = values["task"].astype(str).drop_duplicates().tolist()
        colors = {task: colormaps["tab10"](index % 10) for index, task in enumerate(tasks)}
        axes = panel.axes()
        axes.barh(
            values["label"],
            values[column],
            color=[colors[str(task)] for task in values["task"]],
        )
        if logarithmic and (values[column] > 0.0).all():
            axes.set_xscale("log")
        axes.set_title(title)
        axes.set_xlabel(title + (" (log scale)" if logarithmic else ""))
        axes.grid(axis="x", color=_GRID, linewidth=0.7, alpha=0.65)
        axes.grid(axis="y", visible=False)
        panel.canvas.draw_idle()

    def _browse_manifest(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select experiment manifest", "", "JSON files (*.json)")
        if path:
            self.manifest_path.setText(path)

    def _browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select batch output folder")
        if path:
            self.output_path.setText(path)

    def _open_output(self) -> None:
        path = Path(self.output_path.text())
        if path.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))
        else:
            QMessageBox.information(self, "Output folder", "The batch output folder does not exist yet.")
