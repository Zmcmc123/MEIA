#!/usr/bin/env python3
"""确定性生成 MEIA 内置 v7 通用风格。"""

from __future__ import annotations

import argparse
import colorsys
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ase.data import chemical_symbols  # noqa: E402
from ase.data.colors import jmol_colors  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402

from meia import __version__  # noqa: E402
from meia.bond_rules import BondStrokeStyle  # noqa: E402
from meia.hydrogen_bonds import HydrogenBondSettings  # noqa: E402
from meia.presets import (  # noqa: E402
    PresetKind,
    PresetMetadata,
    SCHEMA_VERSION,
    StylePreset,
    parse_preset,
    style_preset_to_json,
)
from meia.periodic_display import CellPeriodicSettings  # noqa: E402
from meia.visual_state import (  # noqa: E402
    AtomCellSettings,
    BondModuleSettings,
    ExportSettings,
    PairRuleDefaults,
    PortableStyle,
    ViewSettings,
)


ANCHOR_COLORS = {
    "H": "#E6E6E5",
    "C": "#3F4F6A",
    "O": "#E5A6A6",
    "Si": "#5386C6",
    "Ca": "#9ECC91",
}


def soften_jmol(rgb):
    """把 Jmol 基础色调整到当前论文插图的柔和风格。"""
    hue, lightness, saturation = colorsys.rgb_to_hls(*map(float, rgb))
    softened_lightness = min(0.78, max(0.38, 0.62 * lightness + 0.24))
    softened_saturation = min(0.58, 0.58 * saturation)
    return colorsys.hls_to_rgb(hue, softened_lightness, softened_saturation)


def _jmol_base_color(atomic_number: int):
    if atomic_number < len(jmol_colors):
        return jmol_colors[atomic_number]
    # ASE 3.22 只提供到 Mt (Z=109)。110–118 按同族上一周期
    # Pt–Rn 的 Jmol 色回退，保留周期表语义且结果可复现。
    fallback = atomic_number - 32
    return jmol_colors[fallback]


def complete_palette() -> dict[str, str]:
    colors = {
        symbol: to_hex(soften_jmol(_jmol_base_color(index))).upper()
        for index, symbol in enumerate(chemical_symbols[1:119], start=1)
    }
    colors.update(ANCHOR_COLORS)
    return colors


def build_default_style() -> StylePreset:
    style = PortableStyle(
        view=ViewSettings(),
        atom_cell=AtomCellSettings(
            outline_width=0.5,
            element_colors=complete_palette(),
        ),
        bonds=BondModuleSettings(
            draw_bonds=True,
            style=BondStrokeStyle(0.25, "#231815"),
            defaults=PairRuleDefaults(
                bond_cutoff=1.0,
                long_distance_threshold_angstrom=2.0,
                pair_distance_multipliers=(("H", "O", 1.20),),
            ),
            hydrogen_bonds=HydrogenBondSettings(True, 2.5, 120.0),
            pair_rules=(),
        ),
        cell_periodic=CellPeriodicSettings(
            show_unit_cell=2,
            unwrap_bonded_groups=True,
        ),
        export=ExportSettings("svg", 600, True),
    )
    return StylePreset(
        metadata=PresetMetadata(
            schema_version=SCHEMA_VERSION,
            preset_kind=PresetKind.STYLE,
            name="MEIA default paper style",
            created_at="2026-08-21T00:00:00+08:00",
            meia_version=__version__,
        ),
        style=style,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="生成完整周期表的 MEIA v7 默认风格 JSON。"
    )
    parser.add_argument("--output", required=True, help="显式输出文件路径")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="明确允许覆盖已有输出文件",
    )
    args = parser.parse_args()

    output = Path(args.output).expanduser().resolve()
    if output.exists() and not args.overwrite:
        parser.error(f"输出已存在：{output}；如需覆盖请显式传入 --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = style_preset_to_json(build_default_style()) + "\n"
    parsed = parse_preset(payload)
    if not isinstance(parsed, StylePreset):
        raise RuntimeError("生成结果不是 v7 StylePreset")
    output.write_text(payload, encoding="utf-8")
    print(f"已生成：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
