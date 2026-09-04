"""Qt application entry point."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from .main_window import MainWindow


def launch_gui() -> int:
    application = QApplication.instance() or QApplication(sys.argv)
    application.setApplicationName("RelAIBotiX")
    application.setOrganizationName("RelAIBotiX")
    window = MainWindow()
    window.show()
    return application.exec()
