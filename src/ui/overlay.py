"""
PyQt6 transparent, always-on-top, click-through caption overlay.

Design
------
- Frameless window with fully transparent background.
- Text rendered on a semi-transparent black pill/rounded rectangle.
- WindowStaysOnTopHint → always drawn above other windows.
- WA_TransparentForMouseEvents → mouse clicks fall through to the app beneath.
- On Linux/X11: Qt.WindowType.X11BypassWindowManagerHint prevents the WM from
  decorating or managing the window, which is necessary to stay above fullscreen games.
- Position: horizontally centred, pinned near the bottom of the screen.

Thread safety
-------------
`OverlayWindow.show_text()` is safe to call from any thread via Qt's
`QMetaObject.invokeMethod` with a `QueuedConnection`.
"""

from __future__ import annotations

import platform

from PyQt6.QtCore import (
    Qt,
    QTimer,
    pyqtSignal,
    pyqtSlot,
)
from PyQt6.QtGui import (
    QColor,
    QFont,
    QPainter,
    QPainterPath,
)
from PyQt6.QtWidgets import QApplication, QLabel, QWidget


# ─────────────────────────────────────────────────────────────────────────────
# OverlayWindow
# ─────────────────────────────────────────────────────────────────────────────

class OverlayWindow(QWidget):
    """
    Transparent caption overlay window.

    Parameters
    ----------
    font_size : int
        Caption font size in points.
    display_duration_ms : int
        How long each caption stays on screen before fading (ms).
    max_chars : int
        Maximum characters to display at once (older text is trimmed).
    """

    # Class-level signal: emitting from any thread routes to _set_text in the UI thread.
    _show_text_signal = pyqtSignal(str)

    # Styling constants
    _BG_COLOR = QColor(0, 0, 0, 180)           # RGBA – semi-transparent black
    _TEXT_COLOR = QColor(255, 255, 255, 255)    # white
    _CORNER_RADIUS = 12
    _H_PADDING = 20
    _V_PADDING = 10
    _BOTTOM_MARGIN = 80                         # px from screen bottom

    def __init__(
        self,
        font_size: int = 22,
        display_duration_ms: int = 4000,
        max_chars: int = 200,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._display_duration_ms = display_duration_ms
        self._max_chars = max_chars
        self._current_text = ""

        self._setup_window()
        self._setup_label(font_size)
        self._setup_timer()
        # Connect the cross-thread signal to the UI-thread slot
        self._show_text_signal.connect(self._set_text)

    # ──────────────────────────────────────────────────────────────────
    # Setup

    def _setup_window(self) -> None:
        flags = (
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool                    # no taskbar entry
        )
        # On Linux/X11 bypass the window manager to stay above fullscreen
        if platform.system() == "Linux":
            flags |= Qt.WindowType.X11BypassWindowManagerHint

        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    def _setup_label(self, font_size: int) -> None:
        self._label = QLabel("", self)
        self._label.setWordWrap(True)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        font = QFont("Arial", font_size, QFont.Weight.Bold)
        self._label.setFont(font)

        # Transparent background on the label itself; painting done in paintEvent
        self._label.setStyleSheet("background: transparent; color: rgba(255,255,255,255);")

    def _setup_timer(self) -> None:
        """Timer that clears the caption after display_duration_ms."""
        self._clear_timer = QTimer(self)
        self._clear_timer.setSingleShot(True)
        self._clear_timer.timeout.connect(self._clear_text)

    # ──────────────────────────────────────────────────────────────────
    # Public API (thread-safe)

    def show_text(self, text: str) -> None:
        """
        Display *text* in the overlay.
        Safe to call from any thread – emits a queued signal to the UI thread.
        """
        self._show_text_signal.emit(text)

    # ──────────────────────────────────────────────────────────────────
    # Slots (UI thread only)

    @pyqtSlot(str)
    def _set_text(self, text: str) -> None:
        # Append to existing text with a space, trim to max_chars
        combined = (self._current_text + " " + text).strip()
        if len(combined) > self._max_chars:
            combined = combined[-self._max_chars:]
        self._current_text = combined

        self._label.setText(self._current_text)
        self._reposition()
        self.show()

        # Restart the clear timer
        self._clear_timer.start(self._display_duration_ms)

    @pyqtSlot()
    def _clear_text(self) -> None:
        self._current_text = ""
        self._label.setText("")
        self.hide()

    # ──────────────────────────────────────────────────────────────────
    # Layout

    def _reposition(self) -> None:
        """Resize and re-centre the window based on the current text."""
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        geo = screen.geometry()

        self._label.adjustSize()
        w = min(self._label.sizeHint().width() + self._H_PADDING * 2, geo.width() - 40)
        h = self._label.sizeHint().height() + self._V_PADDING * 2

        x = (geo.width() - w) // 2
        y = geo.height() - h - self._BOTTOM_MARGIN

        self.setGeometry(x, y, w, h)
        self._label.setGeometry(
            self._H_PADDING, self._V_PADDING,
            w - self._H_PADDING * 2, h - self._V_PADDING * 2,
        )

    # ──────────────────────────────────────────────────────────────────
    # Paint – rounded background pill

    def paintEvent(self, event: object) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        path = QPainterPath()
        path.addRoundedRect(
            0, 0, self.width(), self.height(),
            self._CORNER_RADIUS, self._CORNER_RADIUS,
        )
        painter.fillPath(path, self._BG_COLOR)
        painter.end()
