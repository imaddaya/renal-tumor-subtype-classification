"""Portable inference paths; importing this module needs no ML runtime."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def models_dir():
    return project_path(os.environ.get("RCC_MODELS_DIR", "Models"))


def macenko_reference_path():
    return project_path(
        os.environ.get(
            "RCC_MACENKO_REFERENCE", "data/train/c/DHMC_0040/p_2688_3808.jpg"
        )
    )


def cors_origins():
    return [
        value.strip().rstrip("/")
        for value in os.environ.get(
            "RCC_ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
        ).split(",")
        if value.strip()
    ]
