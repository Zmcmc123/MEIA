"""Command-line contract for the deterministic large-system benchmark."""

import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_COMMAND = [sys.executable, "-m", "meia.benchmark_large_system"]


def test_large_system_benchmark_reports_machine_readable_metrics():
    result = subprocess.run(
        [
            *BENCHMARK_COMMAND,
            "--nx",
            "2",
            "--water-layers",
            "1",
            "--skip-2d",
            "--json",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["source_atoms"] > 0
    assert payload["atom_instances"] == payload["source_atoms"]
    assert payload["timings_s"]["topology"] >= 0.0
    assert payload["timings_s"]["figure3d"] >= 0.0
    assert payload["figure3d_json_bytes"] > 0
    assert payload["manual_2d_recommended"] is False
    assert payload["timings_s"]["render2d"] is None
    assert payload["preview_png_bytes"] is None


def test_large_system_benchmark_rejects_nonpositive_grid():
    result = subprocess.run(
        [*BENCHMARK_COMMAND, "--nx", "0", "--skip-2d"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert result.returncode != 0
    assert "--nx must be at least 1" in result.stderr
