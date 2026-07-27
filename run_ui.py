"""Entry point for the StardewBot desktop widget.

Run with::

    python run_ui.py

The window is borderless and centered on the primary screen. Use an
``--no-on-top`` flag or right-click the widget to unpin. Closing the
window releases the microphone and exits cleanly.
"""
from __future__ import annotations

import argparse
import signal
import sys
from typing import Optional

from PySide6.QtCore import QCoreApplication, QTimer
from PySide6.QtWidgets import QApplication

from src.ui import StardewWidgetWindow


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--size",
        type=int,
        default=180,
        help="Square widget size in pixels (default: 180).",
    )
    parser.add_argument(
        "--no-on-top",
        action="store_true",
        help="Disable 'always on top' window flag.",
    )
    parser.add_argument(
        "--no-pin-toggle",
        action="store_true",
        help="Disable right-click 'Pin/Unpin' menu.",
    )
    return parser.parse_args(argv)


def _install_sigint_handler(app: QApplication, window: "StardewWidgetWindow") -> None:
    """Route Ctrl+C to Qt's closeEvent path so the pipeline thread is
    joined and the microphone released before the process exits.

    Without this, SIGINT lands in the middle of the QTimer-driven
    ``_tick`` callback (the only Python bytecode boundary inside Qt's
    mostly-C++ event loop) and surfaces as an uncaught
    ``KeyboardInterrupt``, leaving the worker thread running and the
    audio stream held open.
    """
    def _handler(signum, frame):
        # Close the widget first so ``closeEvent``'s release path
        # (``request_stop`` -> ``QThread.wait`` -> mic stream.close``)
        # runs before the event loop tears down.
        try:
            window.close()
        except Exception:
            pass
        app.quit()

    signal.signal(signal.SIGINT, _handler)
    # SIGTERM is best-effort: some Windows shells swallow it, but
    # installing it costs nothing and gives the same clean shutdown
    # path on POSIX ``kill`` (which defaults to SIGTERM) and Windows
    # ``taskkill`` (which on success delivers SIGTERM).
    try:
        signal.signal(signal.SIGTERM, _handler)
    except (AttributeError, ValueError):
        # SIGTERM isn't available on every platform — some Windows
        # builds raise ``ValueError`` when you try to install a handler,
        # and exotic embedded builds may drop the constant entirely
        # (``AttributeError``). Silently skip rather than blocking the
        # SIGINT path above.
        pass

    # On Windows SIGINT is delivered only at the next Python bytecode
    # boundary. ``app.exec()`` typically sits in C++ until a QTimer
    # fires, so ``_handler`` would not run for seconds. Spin a 100 ms
    # no-op timer that nudges the interpreter often enough to dispatch
    # the signal.
    if sys.platform == "win32":
        keepalive = QTimer()
        keepalive.setInterval(100)
        keepalive.timeout.connect(lambda: None)
        keepalive.start()


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    QCoreApplication.setOrganizationName("StardewBot")
    QCoreApplication.setApplicationName("StardewBotWidget")
    app = QApplication.instance() or QApplication(sys.argv)
    window = StardewWidgetWindow(
        size=args.size,
        on_top=not args.no_on_top,
        always_on_top_toggle=not args.no_pin_toggle,
    )
    _install_sigint_handler(app, window)
    window.show()
    window.start()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
