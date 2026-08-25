"""MEIA 包内维护命令入口测试。"""

from pathlib import Path
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "module_name",
    (
        "meia.benchmark_large_system",
        "meia.check_public_docs",
        "meia.generate_default_style",
        "meia.regenerate_visualization_example",
    ),
)
def test_maintenance_module_exposes_help(module_name):
    result = subprocess.run(
        [sys.executable, "-m", module_name, "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
