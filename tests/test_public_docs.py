"""公开 Markdown 资源完整性检查的命令行契约。"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_checker(root: Path, *files: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "meia.check_public_docs",
            "--root",
            str(root),
            "--files",
            *files,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_checker_accepts_existing_relative_image_and_external_link(tmp_path):
    docs = tmp_path / "docs"
    image = docs / "images" / "figure.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"\xff\xd8example\xff\xd9")
    tutorial = docs / "tutorial.md"
    tutorial.write_text(
        "![Figure](images/figure.jpg)\n[GitHub](https://github.com/)\n",
        encoding="utf-8",
    )

    result = _run_checker(tmp_path, "docs/tutorial.md")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "checked 1 Markdown file" in result.stdout


def test_checker_reports_missing_local_target(tmp_path):
    tutorial = tmp_path / "tutorial.md"
    tutorial.write_text("![Missing](images/missing.jpg)\n", encoding="utf-8")

    result = _run_checker(tmp_path, "tutorial.md")

    assert result.returncode == 1
    assert "tutorial.md:1" in result.stdout
    assert "missing local target: images/missing.jpg" in result.stdout


def test_checker_rejects_invalid_jpeg_bytes(tmp_path):
    image = tmp_path / "broken.jpg"
    image.write_text("not a JPEG", encoding="utf-8")
    tutorial = tmp_path / "tutorial.md"
    tutorial.write_text("![Broken](broken.jpg)\n", encoding="utf-8")

    result = _run_checker(tmp_path, "tutorial.md")

    assert result.returncode == 1
    assert "invalid JPEG data: broken.jpg" in result.stdout


def test_checker_reports_case_mismatch_even_on_case_insensitive_filesystems(tmp_path):
    image = tmp_path / "Figure.jpg"
    image.write_bytes(b"\xff\xd8example\xff\xd9")
    tutorial = tmp_path / "tutorial.md"
    tutorial.write_text("![Figure](figure.jpg)\n", encoding="utf-8")

    result = _run_checker(tmp_path, "tutorial.md")

    assert result.returncode == 1
    assert "path case mismatch: figure.jpg" in result.stdout


def test_root_license_matches_auxiliary_license_copy():
    """GitHub 检测入口与 license 目录中的分发副本必须保持一致。"""

    assert (PROJECT_ROOT / "LICENSE.md").read_bytes() == (
        PROJECT_ROOT / "license" / "LICENSE"
    ).read_bytes()
