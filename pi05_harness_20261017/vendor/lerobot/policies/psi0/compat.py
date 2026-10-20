"""Load the canonical Psi0 implementation without copying its source code."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType


def load_psi_module(source_dir: str | Path, module: str) -> ModuleType:
    """Import ``psi.<module>`` from a checked Psi0 repository."""

    root = Path(source_dir).expanduser().resolve()
    package_root = root / "src"
    package = package_root / "psi"
    if not package.is_dir():
        raise FileNotFoundError(
            f"Psi0 source package was not found at {package}. "
            "Set --policy.psi0_source_dir to the /cs/Psi0 checkout."
        )
    package_root_str = str(package_root)
    if package_root_str not in sys.path:
        sys.path.insert(0, package_root_str)
    return importlib.import_module(f"psi.{module}")
