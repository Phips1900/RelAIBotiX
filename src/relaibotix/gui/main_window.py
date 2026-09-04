"""Native RelAIBotiX analysis window."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import html
import io
import json
import os
from pathlib import Path
import sys
import traceback

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QUrl, Signal, Slot
from PySide6.QtGui import QDesktopServices, QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


class _SignalWriter(io.TextIOBase):
    def __init__(self, signal: Signal) -> None:
        super().__init__()
        self.signal = signal

    def write(self, text: str) -> int:
        if text.strip():
            self.signal.emit(text.rstrip())
        return len(text)


class _WorkerSignals(QObject):
    log = Signal(str)
    finished = Signal(int)
    failed = Signal(str)


class _CommandWorker(QRunnable):
    """Run the shared CLI orchestration in a background Qt thread."""

    def __init__(self, arguments: list[str]) -> None:
        super().__init__()
        self.arguments = arguments
        self.signals = _WorkerSignals()

    @Slot()
    def run(self) -> None:
        from relaibotix.cli import main

        writer = _SignalWriter(self.signals.log)
        try:
            with redirect_stdout(writer), redirect_stderr(writer):
                exit_code = main(self.arguments)
        except BaseException as error:
            detail = "".join(traceback.format_exception_only(type(error), error)).strip()
            self.signals.failed.emit(detail)
            return
        self.signals.finished.emit(exit_code)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("RelAIBotiX")
        self.resize(1050, 760)
        self.thread_pool = QThreadPool.globalInstance()
        self._active_worker: _CommandWorker | None = None
        self._busy_buttons: list[QPushButton] = []
        self._build_ui()
        self._load_robot_configs()
        self._load_detectors()

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(14)

        title = QLabel("RelAIBotiX")
        title.setFont(QFont(title.font().family(), 22, QFont.Weight.DemiBold))
        subtitle = QLabel("Behavioral and reliability analysis for robot learning policies")
        subtitle.setStyleSheet("color: #667085;")
        root.addWidget(title)
        root.addWidget(subtitle)

        root.addWidget(self._input_group())
        root.addWidget(self._skill_group())
        root.addWidget(self._analysis_group())

        actions = QHBoxLayout()
        self.validate_button = QPushButton("Validate HDF5")
        self.run_button = QPushButton("Run complete analysis")
        self.run_button.setDefault(True)
        self.validate_button.clicked.connect(self._validate)
        self.run_button.clicked.connect(self._run_analysis)
        self._busy_buttons = [self.validate_button, self.run_button]
        actions.addStretch()
        actions.addWidget(self.validate_button)
        actions.addWidget(self.run_button)
        root.addLayout(actions)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        root.addWidget(self.progress)

        self.results = QTabWidget()
        self.summary = QLabel("Validate an HDF5 file or run an analysis to begin.")
        self.summary.setWordWrap(True)
        self.summary.setAlignment(self.summary.alignment())
        summary_page = QWidget()
        summary_layout = QVBoxLayout(summary_page)
        summary_layout.addWidget(self.summary)
        summary_layout.addStretch()
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.results.addTab(summary_page, "Overview")
        self.results.addTab(self.log, "Details")
        root.addWidget(self.results, 1)

        open_output = QPushButton("Open output folder")
        open_output.clicked.connect(self._open_output)
        root.addWidget(open_output)
        self.setCentralWidget(central)

    def _input_group(self) -> QGroupBox:
        group = QGroupBox("1. Input data")
        form = QFormLayout(group)
        self.input_path = QLineEdit()
        input_row = QHBoxLayout()
        input_row.addWidget(self.input_path)
        browse_input = QPushButton("Browse…")
        browse_input.setMaximumWidth(100)
        browse_input.clicked.connect(self._browse_input)
        input_row.addWidget(browse_input)
        form.addRow("HDF5 file", input_row)

        self.config_box = QComboBox()
        self.config_box.setEditable(True)
        config_row = QHBoxLayout()
        config_row.addWidget(self.config_box)
        browse_config = QPushButton("Browse…")
        browse_config.setMaximumWidth(100)
        browse_config.clicked.connect(self._browse_config)
        config_row.addWidget(browse_config)
        form.addRow("Robot configuration", config_row)

        self.output_path = QLineEdit(str(Path.cwd() / "artifacts" / "gui_run"))
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_path)
        browse_output = QPushButton("Browse…")
        browse_output.setMaximumWidth(100)
        browse_output.clicked.connect(self._browse_output)
        output_row.addWidget(browse_output)
        form.addRow("Output folder", output_row)
        return group

    def _skill_group(self) -> QGroupBox:
        group = QGroupBox("2. Skill detection")
        form = QFormLayout(group)
        self.detector_box = QComboBox()
        self.detector_box.currentIndexChanged.connect(self._detector_changed)
        form.addRow("Pretrained detector", self.detector_box)

        self.legacy_predictions = QCheckBox(
            "Use stored predictions (legacy paper reproduction only)"
        )
        self.legacy_predictions.toggled.connect(self._legacy_changed)
        form.addRow("Legacy data", self.legacy_predictions)

        self.checkpoint_root = QLineEdit(
            os.environ.get("RELAIBOTIX_CHECKPOINT_ROOT", "artifacts/checkpoints")
        )
        root_row = QHBoxLayout()
        root_row.addWidget(self.checkpoint_root)
        self.checkpoint_root_button = QPushButton("Browse…")
        self.checkpoint_root_button.setMaximumWidth(100)
        self.checkpoint_root_button.clicked.connect(self._browse_checkpoint_root)
        root_row.addWidget(self.checkpoint_root_button)
        form.addRow("Checkpoint folder", root_row)

        self.checkpoint_path = QLineEdit()
        checkpoint_row = QHBoxLayout()
        checkpoint_row.addWidget(self.checkpoint_path)
        self.checkpoint_button = QPushButton("Browse…")
        self.checkpoint_button.setMaximumWidth(100)
        self.checkpoint_button.clicked.connect(self._browse_checkpoint)
        checkpoint_row.addWidget(self.checkpoint_button)
        form.addRow("Custom checkpoint", checkpoint_row)

        self.modality_box = QComboBox()
        self.modality_box.addItems(("auto", "timeseries", "camera", "hybrid"))
        self.modality_box.currentTextChanged.connect(self._modality_changed)
        self.device_box = QComboBox()
        self.device_box.addItems(("auto", "cpu", "cuda", "mps"))
        row = QHBoxLayout()
        row.addWidget(QLabel("Modality"))
        row.addWidget(self.modality_box)
        row.addSpacing(20)
        row.addWidget(QLabel("Device"))
        row.addWidget(self.device_box)
        form.addRow("Inference", row)

        self.video_path = QLineEdit()
        video_row = QHBoxLayout()
        video_row.addWidget(self.video_path)
        self.video_button = QPushButton("Browse…")
        self.video_button.setMaximumWidth(100)
        self.video_button.clicked.connect(self._browse_video)
        video_row.addWidget(self.video_button)
        form.addRow("Aligned video folder", video_row)
        self._modality_changed()
        return group

    def _analysis_group(self) -> QGroupBox:
        group = QGroupBox("3. Reliability analysis")
        form = QFormLayout(group)
        self.sensitivity = QCheckBox("Run component sensitivity analysis")
        self.sensitivity.setChecked(True)
        self.sensitivity_factor = QDoubleSpinBox()
        self.sensitivity_factor.setRange(1.01, 1000.0)
        self.sensitivity_factor.setValue(10.0)
        row = QHBoxLayout()
        row.addWidget(self.sensitivity)
        row.addWidget(QLabel("Factor"))
        row.addWidget(self.sensitivity_factor)
        row.addStretch()
        form.addRow("Sensitivity", row)

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
        return group

    def _load_robot_configs(self) -> None:
        roots = (
            Path(__file__).resolve().parents[3] / "configs" / "robots",
            Path(sys.prefix) / "share" / "relaibotix" / "configs" / "robots",
        )
        seen: set[Path] = set()
        for root in roots:
            for path in sorted(root.glob("*.json")):
                resolved = path.resolve()
                if resolved not in seen:
                    self.config_box.addItem(path.stem.replace("_", " ").title(), str(resolved))
                    seen.add(resolved)

    def _load_detectors(self) -> None:
        self.detector_box.addItem("Automatically select", None)
        try:
            from relaibotix.skilldetector import load_registry

            for detector in load_registry().detectors.values():
                label = f"{detector.detector_id} — {detector.case_study}, {detector.modality}"
                self.detector_box.addItem(label, detector.detector_id)
        except Exception:
            pass
        self.detector_box.addItem("Custom checkpoint", "__custom__")
        self._detector_changed()

    def _detector_changed(self) -> None:
        custom = (
            self.detector_box.isEnabled()
            and self.detector_box.currentData() == "__custom__"
        )
        self.checkpoint_path.setEnabled(custom)
        self.checkpoint_button.setEnabled(custom)
        self.checkpoint_root.setEnabled(self.detector_box.isEnabled() and not custom)
        self.checkpoint_root_button.setEnabled(self.detector_box.isEnabled() and not custom)

    def _modality_changed(self) -> None:
        needs_video = (
            self.modality_box.isEnabled()
            and self.modality_box.currentText() in {"camera", "hybrid"}
        )
        self.video_path.setEnabled(needs_video)
        self.video_button.setEnabled(needs_video)

    def _legacy_changed(self) -> None:
        enabled = not self.legacy_predictions.isChecked()
        self.detector_box.setEnabled(enabled)
        self.modality_box.setEnabled(enabled)
        self.device_box.setEnabled(enabled)
        self._detector_changed()
        self._modality_changed()

    def _selected_config(self) -> str:
        typed = self.config_box.currentText().strip()
        if Path(typed).is_file():
            return typed
        data = self.config_box.currentData()
        return str(data) if data else typed

    def _require_paths(self, *, output: bool = False) -> bool:
        missing = []
        if not Path(self.input_path.text()).is_file():
            missing.append("an existing HDF5 file")
        if not Path(self._selected_config()).is_file():
            missing.append("an existing robot configuration")
        if output and not self.output_path.text().strip():
            missing.append("an output folder")
        if missing:
            QMessageBox.warning(self, "Missing input", "Please select " + " and ".join(missing) + ".")
            return False
        return True

    def _validate(self) -> None:
        if not self._require_paths():
            return
        self._start_worker([
            "h5", "validate", self.input_path.text(), "--config", self._selected_config()
        ], validation=True)

    def _run_analysis(self) -> None:
        if not self._require_paths(output=True):
            return
        arguments = [
            "run", self.input_path.text(), "--config", self._selected_config(),
            "--output", self.output_path.text(), "--modality", self.modality_box.currentText(),
            "--device", self.device_box.currentText(),
        ]
        legacy = self.legacy_predictions.isChecked()
        detector = self.detector_box.currentData()
        if legacy:
            from relaibotix.data import inspect_h5

            if not inspect_h5(self.input_path.text()).skill_labels:
                QMessageBox.warning(
                    self,
                    "No stored predictions",
                    "This HDF5 file does not contain detector predictions for legacy reproduction.",
                )
                return
            arguments.append("--legacy-existing-predictions")
        elif detector == "__custom__":
            checkpoint = self.checkpoint_path.text().strip()
            if not Path(checkpoint).is_file():
                QMessageBox.warning(self, "Missing checkpoint", "Please select an existing checkpoint file.")
                return
            arguments.extend(("--checkpoint", checkpoint))
        else:
            try:
                from relaibotix.skilldetector import (
                    load_registry,
                    resolve_checkpoint,
                    select_detector,
                )

                selected = select_detector(
                    load_registry(),
                    self.input_path.text(),
                    detector_id=str(detector) if detector else None,
                    modality=None if self.modality_box.currentText() == "auto" else self.modality_box.currentText(),
                )
                resolve_checkpoint(selected, self.checkpoint_root.text())
                arguments.extend((
                    "--detector", selected.detector_id,
                    "--checkpoint-root", self.checkpoint_root.text(),
                ))
            except Exception as error:
                QMessageBox.warning(
                    self,
                    "No compatible detector",
                    f"{error}\n\nFor an old paper dataset, enable legacy paper reproduction. "
                    "New datasets require a compatible pretrained detector.",
                )
                return
        if self.sensitivity.isChecked():
            arguments.extend(("--sensitivity", str(self.sensitivity_factor.value())))
        if self.prism.isChecked():
            arguments.extend(("--prism", "--prism-executable", self.prism_path.text()))
        if self.storm.isChecked():
            arguments.extend(("--storm", "--storm-executable", self.storm_path.text()))
        if self.video_path.isEnabled():
            video = self.video_path.text().strip()
            if not Path(video).is_dir():
                QMessageBox.warning(self, "Missing video", "Please select the aligned video folder.")
                return
            arguments.extend(("--lerobot-root", video))
        self._start_worker(arguments, validation=False)

    def _start_worker(self, arguments: list[str], *, validation: bool) -> None:
        self.log.clear()
        self.summary.setText("Validation in progress…" if validation else "Analysis in progress…")
        self.results.setCurrentIndex(1)
        self.progress.setRange(0, 0)
        for button in self._busy_buttons:
            button.setEnabled(False)
        worker = _CommandWorker(arguments)
        worker.signals.log.connect(self.log.appendPlainText)
        worker.signals.failed.connect(self._failed)
        worker.signals.finished.connect(
            self._validation_finished if validation else self._analysis_finished
        )
        self._active_worker = worker
        self.thread_pool.start(worker)

    def _finish_busy(self) -> None:
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        for button in self._busy_buttons:
            button.setEnabled(True)
        self._active_worker = None

    def _validation_finished(self, exit_code: int) -> None:
        self._finish_busy()
        if exit_code == 0:
            self.summary.setText("<h3>HDF5 validation passed</h3><p>The file is compatible with the selected robot configuration. Review warnings in Details.</p>")
        else:
            self.summary.setText("<h3>HDF5 validation failed</h3><p>Review the reported errors in Details.</p>")
        self.results.setCurrentIndex(0)

    def _analysis_finished(self, exit_code: int) -> None:
        self._finish_busy()
        if exit_code != 0:
            self.summary.setText("<h3>Analysis did not complete</h3><p>Review Details for the cause.</p>")
            self.results.setCurrentIndex(0)
            return
        try:
            reliability = json.loads(
                (Path(self.output_path.text()) / "reliability" / "reliability.json").read_text()
            )
            behavior = json.loads(
                (Path(self.output_path.text()) / "behavior" / "behavior.json").read_text()
            )
            runs = len({str(row["episode_key"]) for row in behavior["segments"]})
            total_time = sum(float(row["duration"]) for row in behavior["segments"])
            failure = float(reliability["dtmc"]["failure_probability"])
            mttf = float(reliability["repeated_run_mttf"]["hours"])
            self.summary.setText(
                "<h3>Analysis complete</h3>"
                f"<p><b>Runs:</b> {runs}<br>"
                f"<b>Average time per run:</b> {total_time / runs:.2f} s<br>"
                f"<b>Failure probability per run:</b> {failure:.4e}<br>"
                f"<b>Repeated-operation MTTF:</b> {mttf:,.0f} h</p>"
                "<p>Detailed CSV, JSON, PRISM, and STORM files are available in the output folder.</p>"
            )
        except Exception as error:
            self.summary.setText(
                "<h3>Analysis complete</h3><p>Results were written, but the overview could not be loaded: "
                + html.escape(str(error)) + "</p>"
            )
        self.results.setCurrentIndex(0)

    def _failed(self, detail: str) -> None:
        self._finish_busy()
        self.log.appendPlainText(detail)
        self.summary.setText("<h3>Operation failed</h3><p>Review Details for the cause.</p>")
        self.results.setCurrentIndex(1)

    def _browse_input(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select HDF5 file", "", "HDF5 files (*.h5 *.hdf5)")
        if path:
            self.input_path.setText(path)

    def _browse_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select robot configuration", "", "JSON files (*.json)")
        if path:
            self.config_box.setEditText(path)

    def _browse_checkpoint(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select detector checkpoint")
        if path:
            self.checkpoint_path.setText(path)

    def _browse_checkpoint_root(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select checkpoint folder")
        if path:
            self.checkpoint_root.setText(path)

    def _browse_video(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select aligned video folder")
        if path:
            self.video_path.setText(path)

    def _browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select output folder")
        if path:
            self.output_path.setText(path)

    def _open_output(self) -> None:
        path = Path(self.output_path.text())
        if path.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))
        else:
            QMessageBox.information(self, "Output folder", "The output folder does not exist yet.")
