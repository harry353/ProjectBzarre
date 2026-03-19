"""Project-wide logging helper with colored severity labels."""

from __future__ import annotations

import builtins
import re
import sys

# ANSI reset code applied after every colorized label.
COLOR_RESET = "\033[0m"
# Mapping from severity label text to its ANSI foreground color code.
COLOR_MAP = {
    "OK": "\033[32m",
    "INFO": "\033[34m",
    "WARN": "\033[33m",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
    "SKIP": "\033[96m",
}

# Pre-compiled regex that matches bracketed severity labels such as [OK] or [ERROR].
_LABEL_PATTERN = re.compile(r"\[(OK|INFO|WARN|WARNING|ERROR|SKIP)\]")
# Reference to the original built-in print, saved before any monkey-patching.
_original_print = builtins.print
# Guard flag to prevent installing the colored print wrapper more than once.
_installed = False


def _colorize_text(value: str) -> str:
    # Replace every bracketed severity label in the string with its ANSI-colored equivalent.
    def repl(match: re.Match[str]) -> str:
        label = match.group(1)
        color = COLOR_MAP.get(label)
        return f"[{color}{label}{COLOR_RESET}]" if color else match.group(0)

    return _LABEL_PATTERN.sub(repl, value)


def _colored_print(*args, **kwargs):
    # Intercept print calls and colorize any string arguments before forwarding.
    colored_args = [
        _colorize_text(str(arg)) if isinstance(arg, str) else arg for arg in args
    ]
    _original_print(*colored_args, **kwargs)


def enable_colored_logging() -> None:
    """Install a print wrapper that colorizes severity labels globally."""
    global _installed
    if _installed:
        return
    # Replace the built-in print globally so all modules benefit automatically.
    builtins.print = _colored_print  # type: ignore[assignment]
    _installed = True
