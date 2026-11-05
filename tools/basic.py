"""Vivienne's first three tools. Gemini reads each docstring to decide when to call it."""
import subprocess
from datetime import datetime
from pathlib import Path

NOTES = Path.home() / "vivienne_notes.txt"


def get_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%A, %d %B %Y, %H:%M")


def take_note(text: str) -> str:
    """Save a note for the user. Use when they say note, remind me, or remember."""
    text = text.strip()
    if not text:
        return "There was nothing to note."
    with NOTES.open("a", encoding="utf-8") as f:
        f.write(f"[{datetime.now():%Y-%m-%d %H:%M}] {text}\n")
    return "Note saved."


def open_app(name: str) -> str:
    """Open a Mac application by name, for example Safari, Notes or Spotify."""
    name = name.strip()
    if not name or name.startswith("-"):
        return "That isn't a valid app name."
    try:
        r = subprocess.run(["open", "-a", name], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return f"Couldn't open {name}."
    return f"Opened {name}." if r.returncode == 0 else f"Couldn't open {name}."
