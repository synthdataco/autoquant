"""Every script under scripts/ imports against the pinned engine, so a name the engine drops fails here."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = sorted((Path(__file__).resolve().parents[1] / "scripts").glob("*.py"))


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.name)
def test_script_imports(script: Path, monkeypatch):
    spec = importlib.util.spec_from_file_location(script.stem, script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Registered as `python <script>` would be: dataclasses resolve annotations through sys.modules.
    monkeypatch.setitem(sys.modules, script.stem, module)
    spec.loader.exec_module(module)
