"""
prompt_opt — Self-improving prompt optimization pipeline.

Public surface (import from here for convenience):
    from prompt_opt import Config, settings
"""

from prompt_opt.config import Settings, settings  # noqa: F401

__version__ = "0.1.0"
__all__ = ["Settings", "settings"]
