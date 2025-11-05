"""Ensure the desktop UI module can be imported without side effects."""


def test_import() -> None:
    """Importing the Tkinter UI should not raise errors."""

    __import__("src.ui_tk")
