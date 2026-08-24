"""
批量处理脚本（独立运行，不依赖 Streamlit）。

用法：
    python -m meia.batch <input_dir> [options]

示例：
    # 批量导出 SVG
    python -m meia.batch /path/to/structures -o /path/to/output

    # 指定格式和旋转角度
    python -m meia.batch /path/to/structures -o /path/to/output -f png --rotation 90x

    # 不绘制化学键
    python -m meia.batch /path/to/structures -o /path/to/output --no-bonds
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import os
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.utils import rotate

from .config import RenderConfig
from .export import export_figure
from .io import (
    SUPPORTED_STRUCTURE_EXTENSIONS,
    is_supported_structure_filename,
    read_structure,
)
from .pipeline import plan_output_paths, render_atoms, render_file
from .presets import (
    PresetError,
    StylePreset,
    WorkspaceSnapshot,
    apply_style_preset,
    load_default_style,
    parse_preset,
)
from .view_state import rotation_matrix_to_camera
from .size_profiles import RadiusMode, replace_active_bond_width
from .visual_state import VisualizationState, resolve_render_context


# 支持的构型文件扩展名
SUPPORTED_EXTENSIONS = SUPPORTED_STRUCTURE_EXTENSIONS


def _style_preset_with_overrides(
    preset: StylePreset,
    *,
    output_format: str | None = None,
    rotation: str | None = None,
    draw_bonds: bool | None = None,
    bond_cutoff: float | None = None,
    radius_scale: float | None = None,
    bond_width_ratio: float | None = None,
    dpi: int | None = None,
    transparent: bool | None = None,
) -> StylePreset:
    """只覆盖显式给出的命令行参数，其余继承内置 v7 风格。"""
    style = preset.style
    view = style.view
    size_profiles = style.size_profiles
    bonds = style.bonds
    defaults = bonds.defaults
    export = style.export

    if rotation is not None:
        matrix = np.asarray(rotate(rotation), dtype=float)
        view = replace(
            view,
            rotation=rotation,
            camera=rotation_matrix_to_camera(matrix),
        )
    if radius_scale is not None:
        if size_profiles.active_mode is RadiusMode.COVALENT:
            size_profiles = replace(
                size_profiles,
                covalent=replace(
                    size_profiles.covalent,
                    global_scale=radius_scale,
                ),
            )
        else:
            size_profiles = replace(
                size_profiles,
                uniform=replace(
                    size_profiles.uniform,
                    global_scale=radius_scale,
                ),
            )
    if draw_bonds is not None:
        bonds = replace(bonds, draw_bonds=draw_bonds)
    if bond_cutoff is not None:
        defaults = replace(defaults, bond_cutoff=bond_cutoff)
    if bond_width_ratio is not None:
        size_profiles = replace_active_bond_width(
            size_profiles,
            bond_width_ratio,
        )
    bonds = replace(bonds, defaults=defaults)
    if output_format is not None:
        export = replace(export, format=output_format)
    if dpi is not None:
        export = replace(export, dpi=dpi)
    if transparent is not None:
        export = replace(export, transparent=transparent)

    return replace(
        preset,
        style=replace(
            style,
            view=view,
            size_profiles=size_profiles,
            bonds=bonds,
            export=export,
        ),
    )


def find_structure_files(input_dir: str) -> List[str]:
    """扫描目录，返回所有支持的构型文件路径。"""
    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise ValueError(f"输入路径不是目录：{input_dir}")

    files = []
    for f in sorted(input_path.iterdir()):
        if f.is_file() and is_supported_structure_filename(f.name):
            files.append(str(f))
    return files


def batch_process(
    input_dir: str,
    output_dir: str,
    config: Optional[RenderConfig] = None,
    draw_bonds: bool = True,
    output_format: str = "svg",
    visualization_preset: Optional[StylePreset] = None,
    *,
    overwrite: bool = False,
) -> List[Optional[str]]:
    """批量处理目录中的构型文件。

    Parameters
    ----------
    input_dir : str
        输入目录，包含构型文件
    output_dir : str
        输出目录
    config : RenderConfig, optional
        渲染参数；为 None 且无预设时读取内置 v7 默认风格
    draw_bonds : bool
        是否绘制化学键
    output_format : str
        输出格式：svg, png, pdf

    Returns
    -------
    List[Optional[str]]
        成功导出的文件路径列表；失败项为 None
    """
    if isinstance(visualization_preset, WorkspaceSnapshot):
        raise PresetError(
            "批处理只接受通用风格预设；工作状态快照包含单一具体结构，"
            "不能应用到输入目录。",
            message_key="preset.batch_style_only",
        )
    if visualization_preset is not None and not isinstance(
        visualization_preset, StylePreset
    ):
        raise PresetError(
            "批处理只接受 StylePreset 通用风格预设",
            message_key="preset.batch_style_only",
        )
    if visualization_preset is None and config is None:
        visualization_preset = _style_preset_with_overrides(
            load_default_style(),
            output_format=output_format,
            draw_bonds=draw_bonds,
        )
    if isinstance(visualization_preset, StylePreset):
        portable_state = VisualizationState(style=visualization_preset.style)
        config = resolve_render_context(Atoms(), portable_state).config
        output_format = visualization_preset.style.export.format
        draw_bonds = visualization_preset.style.bonds.draw_bonds
    elif config is None:
        config = RenderConfig()

    files = find_structure_files(input_dir)
    if not files:
        print(f"⚠️  在 {input_dir} 中未找到支持的构型文件")
        print(f"   支持的格式：{', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        return []

    output_paths = plan_output_paths(
        files,
        output_dir,
        output_format,
        overwrite=overwrite,
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"📂 输入目录：{input_dir}")
    print(f"📁 输出目录：{output_dir}")
    print(f"🔢 找到 {len(files)} 个文件")
    print(f"🎨 格式：{output_format} | 化学键：{'是' if draw_bonds else '否'}")
    print(f"🔄 旋转：{config.rotation}")
    print("-" * 50)

    results = []
    for i, (filepath, out_path) in enumerate(zip(files, output_paths), 1):
        basename = os.path.basename(filepath)
        try:
            active_config = config
            active_draw_bonds = draw_bonds
            active_format = output_format
            if isinstance(visualization_preset, StylePreset):
                atoms = read_structure(filepath)
                state = apply_style_preset(
                    VisualizationState(style=visualization_preset.style),
                    visualization_preset,
                    atoms,
                )
                context = resolve_render_context(atoms, state)
                active_config = context.config
                active_draw_bonds = context.bond_settings.draw_bonds
                active_format = visualization_preset.style.export.format
                fig = render_atoms(atoms, active_config, render_context=context)
            else:
                fig = render_file(filepath, active_config, draw_bonds=active_draw_bonds)
            out_name = os.path.basename(out_path)
            export_figure(
                fig,
                active_format,
                active_config,
                out_path,
                overwrite=overwrite,
            )
            plt.close(fig)
            print(f"  [{i}/{len(files)}] ✅ {basename} → {out_name}")
            results.append(out_path)
        except Exception as e:
            print(f"  [{i}/{len(files)}] ❌ {basename} — {e}")
            results.append(None)

    print("-" * 50)
    success = sum(1 for r in results if r is not None)
    failed = len(results) - success
    print(f"✅ 成功：{success}  ❌ 失败：{failed}")
    return results


def main():
    """命令行入口。"""
    parser = argparse.ArgumentParser(
        description="批量原子构型可视化 — 从构型文件生成带化学键的 2D 投影图",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python -m meia.batch ./structures -o ./output
  python -m meia.batch ./structures -o ./output -f png --rotation 90x
  python -m meia.batch ./structures -o ./output --no-bonds --cutoff 1.2
        """,
    )
    parser.add_argument("input_dir", help="输入目录（包含构型文件）")
    parser.add_argument("-o", "--output", required=True, help="输出目录")
    parser.add_argument(
        "-f", "--format", default=None, choices=["svg", "png", "pdf"],
        help="输出格式（默认继承内置风格）",
    )
    parser.add_argument("--rotation", default=None, help="旋转角度（默认继承内置风格）")
    parser.add_argument("--no-bonds", action="store_true", help="不绘制化学键")
    parser.add_argument("--cutoff", type=float, default=None, help="成键容差（默认继承内置风格）")
    parser.add_argument("--radius-scale", type=float, default=None, help="全局原子半径倍率（默认继承内置风格）")
    parser.add_argument("--bond-width", type=float, default=None, help="键宽比例（默认继承内置风格）")
    parser.add_argument("--dpi", type=int, default=None, help="PNG 分辨率（默认继承内置风格）")
    parser.add_argument("--no-transparent", action="store_true", help="不使用透明背景")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="明确允许覆盖已存在的同名导出文件",
    )
    parser.add_argument(
        "--preset",
        help=(
            "通用风格预设 JSON；提供后其参数优先于其他渲染选项。"
            "只接受 v7 StylePreset，不接受工作状态快照"
        ),
    )

    args = parser.parse_args()

    config = None
    visualization_preset = None
    if args.preset:
        try:
            parsed = parse_preset(Path(args.preset).read_bytes())
            if isinstance(parsed, WorkspaceSnapshot):
                raise PresetError(
                    "批处理不接受工作状态快照，请导出通用风格预设。",
                    message_key="preset.batch_style_only",
                )
            if not isinstance(parsed, StylePreset):
                raise PresetError(
                    "批处理只接受 v7 StylePreset 通用风格预设",
                    message_key="preset.batch_style_only",
                )
            visualization_preset = parsed
        except (OSError, PresetError) as exc:
            parser.error(f"通用风格预设无法读取：{exc}")
    else:
        try:
            visualization_preset = _style_preset_with_overrides(
                load_default_style(),
                output_format=args.format,
                rotation=args.rotation,
                draw_bonds=False if args.no_bonds else None,
                bond_cutoff=args.cutoff,
                radius_scale=args.radius_scale,
                bond_width_ratio=args.bond_width,
                dpi=args.dpi,
                transparent=False if args.no_transparent else None,
            )
        except (PresetError, TypeError, ValueError) as exc:
            parser.error(f"批处理参数无效：{exc}")

    if isinstance(visualization_preset, StylePreset):
        active_format = visualization_preset.style.export.format
        active_draw_bonds = visualization_preset.style.bonds.draw_bonds
    else:
        active_format = args.format or "svg"
        active_draw_bonds = not args.no_bonds

    results = batch_process(
        input_dir=args.input_dir,
        output_dir=args.output,
        config=config,
        draw_bonds=active_draw_bonds,
        output_format=active_format,
        visualization_preset=visualization_preset,
        overwrite=args.overwrite,
    )

    # 退出码
    if any(r is None for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
