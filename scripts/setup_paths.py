"""Shared setup: ensure results directories exist.

Imported by every pipeline script so a clean checkout (no results/)
can run without manual mkdir. Idempotent.
"""
import os


def ensure_dirs():
    for d in ("results/tables", "results/figures", "results/models"):
        os.makedirs(d, exist_ok=True)
