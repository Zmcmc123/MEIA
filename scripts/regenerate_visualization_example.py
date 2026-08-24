#!/usr/bin/env python3
"""使用受哈希约束的原始构型确定性再生 v7 工作区与 SVG。"""

from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_MANIFEST_PATH = (
    PROJECT_ROOT / "examples" / "co2_h2o_color_strength.manifest.json"
).resolve()
REFERENCE_INPUT_SIGNATURE = {
    "content_sha256": (
        "187ee6a6d1c5bffc2b55a8ea254f0dc86c82a1f56743fcdad55504488b399d5f"
    ),
    "atom_count": 225,
    "symbols_sha256": (
        "4555e45eecaefcfc308f5c386b206bf0b01e398dddfeefa57645f258ffc90a2d"
    ),
}
REFERENCE_CREATED_AT = "2026-08-23T00:00:00+08:00"
REFERENCE_VIEW_ROTATION = "-90z,-90x"
REFERENCE_WORKSPACE = "examples/co2_h2o_color_strength.meia.json"
REFERENCE_OUTPUT = "examples/co2_h2o_color_strength.svg"
REFERENCE_NAME = "co2-h2o-color-strength-reference"
MANIFEST_FIELDS = {
    "schema_version",
    "source_date_epoch",
    "created_at",
    "input",
    "view_rotation",
    "color_strengths",
    "workspace",
    "workspace_sha256",
    "output",
    "output_sha256",
}


def _project_path(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"manifest 的 {label} 必须是非空仓库相对路径")
    path = (PROJECT_ROOT / value).resolve()
    if PROJECT_ROOT not in path.parents:
        raise ValueError(f"manifest 的 {label} 必须位于项目目录内")
    return path


def _load_manifest(path: Path) -> dict:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取生成清单：{exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 2:
        raise ValueError("仅支持 schema_version=2 的示例生成清单")
    missing = sorted(MANIFEST_FIELDS - set(manifest))
    unknown = sorted(set(manifest) - MANIFEST_FIELDS)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"缺少字段 {', '.join(missing)}")
        if unknown:
            details.append(f"包含未知字段 {', '.join(unknown)}")
        raise ValueError("manifest schema 2 字段不精确：" + "；".join(details))
    return manifest


def _validate_identity_bound_manifest(path: Path, manifest: dict) -> bool:
    """仅对已授权提交案例锁定代码内的身份与写入路径。"""
    if path.resolve() != REFERENCE_MANIFEST_PATH:
        return False
    expected = {
        "input": REFERENCE_INPUT_SIGNATURE,
        "created_at": REFERENCE_CREATED_AT,
        "view_rotation": REFERENCE_VIEW_ROTATION,
        "workspace": REFERENCE_WORKSPACE,
        "output": REFERENCE_OUTPUT,
    }
    mismatches = [
        label
        for label, expected_value in expected.items()
        if manifest.get(label) != expected_value
    ]
    if mismatches:
        raise ValueError(
            "提交参考案例的身份绑定不匹配："
            + "、".join(mismatches)
        )
    return True


def _workspace_name(workspace_path: Path, is_reference_case: bool) -> str:
    if is_reference_case:
        return REFERENCE_NAME
    suffix = ".meia.json"
    filename = workspace_path.name
    stem = (
        filename[: -len(suffix)]
        if filename.endswith(suffix)
        else workspace_path.stem
    )
    return f"{stem or 'visualization'}-reference"


def _signature_mapping(content: bytes, atoms) -> dict:
    symbols = "\0".join(atoms.get_chemical_symbols()).encode("utf-8")
    return {
        "content_sha256": sha256(content).hexdigest(),
        "atom_count": len(atoms),
        "symbols_sha256": sha256(symbols).hexdigest(),
    }


def _source_stat_identity(stat_result) -> tuple[int, int, int]:
    """返回能捕获内容替换或就地改写的文件系统身份。"""
    return (
        stat_result.st_size,
        stat_result.st_mtime_ns,
        stat_result.st_ino,
    )


def _verify_source_unchanged(
    path: Path,
    expected_stat_identity: tuple[int, int, int],
    expected_sha256: str,
) -> None:
    """在任何仓库产物写入前重新校验外部源文件。"""
    stat_before = _source_stat_identity(path.stat())
    content = path.read_bytes()
    stat_after = _source_stat_identity(path.stat())
    if (
        stat_before != expected_stat_identity
        or stat_after != expected_stat_identity
        or sha256(content).hexdigest() != expected_sha256
    ):
        raise ValueError("输入构型在生成期间发生变化，已停止且未写入产物")


def _manifest_bytes(manifest: dict) -> bytes:
    return (
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="校验输入哈希，并确定性再生 MEIA v7 工作区与 SVG。"
    )
    parser.add_argument("--input", required=True, help="原始构型文件路径")
    parser.add_argument("--manifest", required=True, help="仓库内生成清单路径")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--check",
        action="store_true",
        help="生成到内存并核对仓库内现有输出，不写文件",
    )
    action.add_argument(
        "--overwrite",
        action="store_true",
        help="哈希全部匹配后覆盖清单指定的生成产物",
    )
    args = parser.parse_args()

    manifest_path = _project_path(args.manifest, "manifest")
    manifest = _load_manifest(manifest_path)
    is_reference_case = _validate_identity_bound_manifest(
        manifest_path,
        manifest,
    )
    source_date_epoch = manifest.get("source_date_epoch")
    if isinstance(source_date_epoch, bool) or not isinstance(source_date_epoch, int):
        raise ValueError("manifest 的 source_date_epoch 必须是整数")

    # Matplotlib 在导入时读取这两个变量。固定 SVG 日期可使输出逐字节复现。
    os.environ["SOURCE_DATE_EPOCH"] = str(source_date_epoch)
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "meia-mpl-cache"),
    )
    sys.path.insert(0, str(PROJECT_ROOT))

    import matplotlib.pyplot as plt

    from ase.utils import rotate
    import numpy as np

    from meia import __version__
    from meia.atom_styles import AtomColorStrength
    from meia.export import export_figure
    from meia.io import read_structure
    from meia.pipeline import render_atoms
    from meia.presets import (
        PresetKind,
        PresetMetadata,
        SCHEMA_VERSION,
        SnapshotStructure,
        WorkspaceSnapshot,
        load_default_style,
        workspace_snapshot_to_json,
    )
    from meia.sidebar import initialize_visual_state
    from meia.view_state import rotation_matrix_to_camera
    from meia.visual_state import ViewSettings, resolve_render_context

    input_path = Path(args.input).resolve()
    source_stat_identity = _source_stat_identity(input_path.stat())
    content = input_path.read_bytes()
    if _source_stat_identity(input_path.stat()) != source_stat_identity:
        raise ValueError("输入构型在读取期间发生变化，已停止")
    atoms = read_structure(str(input_path))

    expected_input = manifest.get("input")
    if not isinstance(expected_input, dict):
        raise ValueError("manifest 缺少 input 构型身份")
    expected_signature = {
        key: expected_input.get(key)
        for key in ("content_sha256", "atom_count", "symbols_sha256")
    }
    actual_signature = _signature_mapping(content, atoms)
    if actual_signature != expected_signature:
        raise ValueError(
            "输入构型与生成清单不匹配："
            f"期望 {expected_signature}，实际 {actual_signature}"
        )

    workspace_path = _project_path(manifest.get("workspace"), "workspace")
    output_path = _project_path(manifest.get("output"), "output")
    if len({manifest_path, workspace_path, output_path}) != 3:
        raise ValueError("manifest、workspace 与 output 必须是三个不同路径")

    view_rotation = manifest.get("view_rotation")
    color_strengths = manifest.get("color_strengths")
    if not isinstance(color_strengths, list):
        raise ValueError("manifest 的 color_strengths 必须是数组")
    default = load_default_style()
    state = initialize_visual_state(atoms, default)
    state = replace(
        state,
        style=replace(
            state.style,
            view=ViewSettings(
                view_rotation,
                rotation_matrix_to_camera(
                    np.asarray(rotate(view_rotation), dtype=float)
                ),
            ),
        ),
        atom_selection=replace(
            state.atom_selection,
            color_strengths=tuple(
                AtomColorStrength(
                    item["atom_index"],
                    item["atom_symbol"],
                    item["strength"],
                )
                for item in color_strengths
            ),
        ),
    )
    snapshot = WorkspaceSnapshot(
        metadata=PresetMetadata(
            SCHEMA_VERSION,
            PresetKind.WORKSPACE_SNAPSHOT,
            _workspace_name(workspace_path, is_reference_case),
            manifest["created_at"],
            __version__,
        ),
        structure=SnapshotStructure.from_atoms(atoms, input_path.name),
        state=state,
    )
    workspace_bytes = (workspace_snapshot_to_json(snapshot) + "\n").encode(
        "utf-8"
    )
    context = resolve_render_context(atoms, state)
    config = context.config
    figure = render_atoms(
        atoms,
        config,
        render_context=context,
    )
    try:
        data = export_figure(figure, state.style.export.format, config)
    finally:
        plt.close(figure)
    if data is None:
        raise RuntimeError("示例生成未返回图像字节")

    _verify_source_unchanged(
        input_path,
        source_stat_identity,
        actual_signature["content_sha256"],
    )

    actual_workspace_sha256 = sha256(workspace_bytes).hexdigest()
    actual_output_sha256 = sha256(data).hexdigest()

    if args.check:
        failures = []
        for label, expected, actual in (
            (
                "workspace_sha256",
                manifest.get("workspace_sha256"),
                actual_workspace_sha256,
            ),
            (
                "output_sha256",
                manifest.get("output_sha256"),
                actual_output_sha256,
            ),
        ):
            if expected != actual:
                failures.append(f"{label} 期望 {expected}，实际 {actual}")
        for label, path, expected_bytes in (
            ("workspace", workspace_path, workspace_bytes),
            ("output", output_path, data),
        ):
            if not path.is_file() or path.read_bytes() != expected_bytes:
                failures.append(f"{label} 产物与可再生结果不一致：{path}")
        if failures:
            raise ValueError("可再生检查失败：" + "；".join(failures))
        print(
            "验证通过："
            f"{workspace_path} ({actual_workspace_sha256})；"
            f"{output_path} ({actual_output_sha256})"
        )
        return 0

    workspace_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    workspace_path.write_bytes(workspace_bytes)
    output_path.write_bytes(data)
    manifest["workspace_sha256"] = actual_workspace_sha256
    manifest["output_sha256"] = actual_output_sha256
    manifest_path.write_bytes(_manifest_bytes(manifest))
    print(
        "已再生："
        f"{workspace_path} ({actual_workspace_sha256})；"
        f"{output_path} ({actual_output_sha256})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
