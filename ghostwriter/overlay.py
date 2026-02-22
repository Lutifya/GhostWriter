"""Transparent, always-on-top subtitle overlay built with Tkinter.

The overlay renders a semi-transparent black box containing white subtitle
text.  It floats above all other windows and auto-hides after
``display_duration`` milliseconds of inactivity.

Thread safety
~~~~~~~~~~~~~
:py:meth:`SubtitleOverlay.show_text` may be called from any thread; it
schedules the actual Tk update on the main thread via ``root.after(0, …)``.
:py:meth:`SubtitleOverlay.run` blocks and must be called from the main thread.
"""

import tkinter as tk
from typing import Optional, Tuple


class SubtitleOverlay:
    """Tkinter-based transparent subtitle overlay.

    Args:
        font: Tkinter font tuple, e.g. ``("Arial", 24, "bold")``.
        fg_color: Subtitle text colour.
        bg_color: Background colour of the subtitle box.
        alpha: Overall window opacity in ``[0.0, 1.0]``.
        max_lines: Maximum number of recent lines kept on screen.
        display_duration: Milliseconds before the overlay auto-hides.
        position: ``"bottom"``, ``"top"``, or ``"center"``.
    """

    DEFAULT_FONT: Tuple = ("Arial", 24, "bold")
    DEFAULT_FG_COLOR: str = "white"
    DEFAULT_BG_COLOR: str = "black"
    DEFAULT_ALPHA: float = 0.85
    DEFAULT_MAX_LINES: int = 2
    DEFAULT_DISPLAY_DURATION: int = 5_000

    def __init__(
        self,
        font: Tuple = DEFAULT_FONT,
        fg_color: str = DEFAULT_FG_COLOR,
        bg_color: str = DEFAULT_BG_COLOR,
        alpha: float = DEFAULT_ALPHA,
        max_lines: int = DEFAULT_MAX_LINES,
        display_duration: int = DEFAULT_DISPLAY_DURATION,
        position: str = "bottom",
    ) -> None:
        self.font = font
        self.fg_color = fg_color
        self.bg_color = bg_color
        self.alpha = alpha
        self.max_lines = max_lines
        self.display_duration = display_duration
        self.position = position

        self._root: Optional[tk.Tk] = None
        self._label: Optional[tk.Label] = None
        self._hide_job: Optional[str] = None
        self._lines: list = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show_text(self, text: str) -> None:
        """Display *text* as a subtitle line.

        Thread-safe: may be called from any thread.

        Args:
            text: The transcribed sentence to display.
        """
        if self._root is None:
            return

        self._lines.append(text)
        if len(self._lines) > self.max_lines:
            self._lines = self._lines[-self.max_lines :]

        display_text = "\n".join(self._lines)
        self._root.after(0, lambda t=display_text: self._update_display(t))

    def run(self) -> None:
        """Initialise and start the Tkinter main loop.

        This method blocks until :py:meth:`close` is called (or the window
        is destroyed).  Must be called from the main thread.
        """
        self._setup()
        self._root.mainloop()

    def close(self) -> None:
        """Quit the Tkinter main loop and destroy the window."""
        if self._root is not None:
            self._root.quit()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _setup(self) -> None:
        """Create and configure the Tkinter window."""
        self._root = tk.Tk()
        self._root.title("GhostWriter")

        # Remove title bar / window decorations
        self._root.overrideredirect(True)
        # Stay above all other windows
        self._root.attributes("-topmost", True)
        # Semi-transparent window
        self._root.attributes("-alpha", self.alpha)
        self._root.configure(bg=self.bg_color)

        # On platforms that support per-colour transparency make the
        # background colour fully transparent (keeps only the text visible).
        try:
            self._root.attributes("-transparentcolor", self.bg_color)
        except tk.TclError:
            pass

        screen_w = self._root.winfo_screenwidth()

        self._label = tk.Label(
            self._root,
            text="",
            font=self.font,
            fg=self.fg_color,
            bg=self.bg_color,
            wraplength=screen_w - 100,
            justify="center",
            padx=12,
            pady=8,
        )
        self._label.pack()

        # Start hidden; appear only when there is text to show
        self._root.withdraw()

    def _position_window(self) -> None:
        """Place the overlay window at the configured screen position."""
        self._root.update_idletasks()
        screen_w = self._root.winfo_screenwidth()
        screen_h = self._root.winfo_screenheight()
        win_w = self._root.winfo_reqwidth()
        win_h = self._root.winfo_reqheight()

        x = (screen_w - win_w) // 2

        if self.position == "bottom":
            y = screen_h - win_h - 100
        elif self.position == "top":
            y = 50
        else:
            y = (screen_h - win_h) // 2

        self._root.geometry(f"+{x}+{y}")

    def _update_display(self, text: str) -> None:
        """Update the label and make the window visible.

        Must be called from the Tkinter main thread (use ``root.after``).
        """
        if self._label is None or self._root is None:
            return

        self._label.configure(text=text)
        self._root.deiconify()
        self._position_window()

        # Reset the auto-hide timer
        if self._hide_job is not None:
            self._root.after_cancel(self._hide_job)
        self._hide_job = self._root.after(self.display_duration, self._hide)

    def _hide(self) -> None:
        """Hide the overlay and clear the line buffer."""
        if self._root is not None:
            self._root.withdraw()
        self._lines = []
        self._hide_job = None
