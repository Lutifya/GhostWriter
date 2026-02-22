"""Tests for ghostwriter.overlay.

Tkinter requires a display, so we mock the ``tkinter`` module entirely to
keep these tests runnable in headless CI environments.
"""

import sys
from types import ModuleType
from unittest.mock import MagicMock, call, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tk_mock() -> ModuleType:
    """Return a minimal tkinter stub."""
    tk_mock = MagicMock(name="tkinter")

    # tk.TclError must be a real exception class so ``except tk.TclError`` works
    class TclError(Exception):
        pass

    tk_mock.TclError = TclError

    root = MagicMock(name="Tk_instance")
    root.winfo_screenwidth.return_value = 1920
    root.winfo_screenheight.return_value = 1080
    root.winfo_reqwidth.return_value = 600
    root.winfo_reqheight.return_value = 60
    tk_mock.Tk.return_value = root

    label = MagicMock(name="Label_instance")
    tk_mock.Label.return_value = label

    return tk_mock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tk_mock():
    mock = _make_tk_mock()
    with patch.dict(sys.modules, {"tkinter": mock}):
        # Reload the overlay module so it picks up the mock
        import importlib

        import ghostwriter.overlay as overlay_mod

        importlib.reload(overlay_mod)
        yield mock, overlay_mod.SubtitleOverlay
    # Restore original module after test
    import importlib

    import ghostwriter.overlay as overlay_mod

    importlib.reload(overlay_mod)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def test_default_init(tk_mock):
    _, SubtitleOverlay = tk_mock
    overlay = SubtitleOverlay()
    assert overlay.alpha == 0.85
    assert overlay.max_lines == 2
    assert overlay.display_duration == 5_000
    assert overlay.position == "bottom"
    assert overlay._root is None
    assert overlay._label is None
    assert overlay._lines == []


def test_custom_init(tk_mock):
    _, SubtitleOverlay = tk_mock
    overlay = SubtitleOverlay(
        font=("Helvetica", 18, "bold"),
        fg_color="yellow",
        bg_color="#000000",
        alpha=0.9,
        max_lines=3,
        display_duration=3_000,
        position="top",
    )
    assert overlay.font == ("Helvetica", 18, "bold")
    assert overlay.fg_color == "yellow"
    assert overlay.alpha == 0.9
    assert overlay.max_lines == 3
    assert overlay.position == "top"


# ---------------------------------------------------------------------------
# show_text
# ---------------------------------------------------------------------------


def test_show_text_no_root_is_noop(tk_mock):
    _, SubtitleOverlay = tk_mock
    overlay = SubtitleOverlay()
    # Must not raise even before _setup() is called
    overlay.show_text("Hello")


def test_show_text_schedules_after(tk_mock):
    tk_mod, SubtitleOverlay = tk_mock
    overlay = SubtitleOverlay()
    overlay._root = tk_mod.Tk.return_value

    overlay.show_text("Hello world")
    overlay._root.after.assert_called_once()
    args = overlay._root.after.call_args[0]
    assert args[0] == 0  # delay=0 for immediate scheduling


def test_show_text_respects_max_lines(tk_mock):
    tk_mod, SubtitleOverlay = tk_mock
    overlay = SubtitleOverlay(max_lines=2)
    overlay._root = tk_mod.Tk.return_value

    for word in ("One", "Two", "Three"):
        overlay.show_text(word)

    assert len(overlay._lines) == 2
    assert overlay._lines == ["Two", "Three"]


# ---------------------------------------------------------------------------
# _update_display
# ---------------------------------------------------------------------------


def test_update_display_shows_text(tk_mock):
    tk_mod, SubtitleOverlay = tk_mock
    overlay = SubtitleOverlay()
    overlay._root = tk_mod.Tk.return_value
    overlay._label = tk_mod.Label.return_value

    overlay._update_display("Test subtitle")

    overlay._label.configure.assert_called_once_with(text="Test subtitle")
    overlay._root.deiconify.assert_called_once()


def test_update_display_cancels_previous_hide(tk_mock):
    tk_mod, SubtitleOverlay = tk_mock
    overlay = SubtitleOverlay()
    overlay._root = tk_mod.Tk.return_value
    overlay._label = tk_mod.Label.return_value
    overlay._hide_job = "job-123"

    overlay._update_display("Text")

    overlay._root.after_cancel.assert_called_once_with("job-123")


def test_update_display_sets_new_hide_job(tk_mock):
    tk_mod, SubtitleOverlay = tk_mock
    overlay = SubtitleOverlay(display_duration=4_000)
    overlay._root = tk_mod.Tk.return_value
    overlay._label = tk_mod.Label.return_value
    tk_mod.Tk.return_value.after.return_value = "new-job"

    overlay._update_display("Text")

    # The second after() call is the hide timer
    calls = overlay._root.after.call_args_list
    hide_call = calls[-1]
    assert hide_call[0][0] == 4_000  # display_duration


# ---------------------------------------------------------------------------
# _hide
# ---------------------------------------------------------------------------


def test_hide_withdraws_window(tk_mock):
    tk_mod, SubtitleOverlay = tk_mock
    overlay = SubtitleOverlay()
    overlay._root = tk_mod.Tk.return_value
    overlay._lines = ["line1", "line2"]

    overlay._hide()

    overlay._root.withdraw.assert_called_once()
    assert overlay._lines == []
    assert overlay._hide_job is None


# ---------------------------------------------------------------------------
# _setup – position variants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "position,expected_y",
    [
        ("bottom", 1080 - 60 - 100),  # screen_h - win_h - 100
        ("top", 50),
        ("center", (1080 - 60) // 2),
    ],
)
def test_position_window(tk_mock, position, expected_y):
    tk_mod, SubtitleOverlay = tk_mock
    root = tk_mod.Tk.return_value

    overlay = SubtitleOverlay(position=position)
    overlay._root = root
    overlay._position_window()

    expected_x = (1920 - 600) // 2
    root.geometry.assert_called_with(f"+{expected_x}+{expected_y}")


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------


def test_close_calls_quit(tk_mock):
    tk_mod, SubtitleOverlay = tk_mock
    overlay = SubtitleOverlay()
    overlay._root = tk_mod.Tk.return_value
    overlay.close()
    overlay._root.quit.assert_called_once()


def test_close_without_root_is_noop(tk_mock):
    _, SubtitleOverlay = tk_mock
    overlay = SubtitleOverlay()
    overlay.close()  # Must not raise
