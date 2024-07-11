from pathlib import Path

__version__ = "1.0.0"


keys_cfg_path = Path(__file__).resolve().parent.parent.parent / "keys.cfg"


__all__ = [
    "keys_cfg_path",
    "__version__",
]
