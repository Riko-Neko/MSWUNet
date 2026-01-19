# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ROOT = Path(__file__).resolve().parents[*]
sys.path.insert(0, str(ROOT))

from config.settings import Settings
"""

import os


def _b(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(int(default))).lower() in {"1", "true", "yes"}


class Settings:
    DEBUG = _b("DEBUG", False)
    PROD = _b("PROD", False)
