"""Optional native desktop interface for RelAIBotiX."""


def launch_gui() -> int:
    """Launch the PySide6 application without importing Qt in CLI-only use."""

    from .app import launch_gui as _launch_gui

    return _launch_gui()
