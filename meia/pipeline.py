"""
便捷管线模块。

一行调用完成从结构文件到渲染图的全流程。
"""

from pathlib import Path
from typing import Optional, List, Sequence
import unicodedata
import matplotlib.pyplot as plt
from ase import Atoms

from .config import RenderConfig
from .bond_rules import BondSettings
from .export import export_figure
from .io import read_structure
from .i18n import LocalizedError
from .view import render_2d


class OutputCollisionError(LocalizedError):
    """多个输入会映射到同一个输出路径。"""


def plan_output_paths(
    filepaths: Sequence[str],
    output_dir: str,
    output_format: str,
    *,
    overwrite: bool = False,
) -> List[str]:
    """在渲染前一次性验证批处理输出，避免同名或旧文件被覆盖。"""
    suffix = output_format.lower().lstrip(".")
    targets = [Path(output_dir) / f"{Path(path).stem}.{suffix}" for path in filepaths]
    owners: dict[str, tuple[Path, list[str]]] = {}
    for source, target in zip(filepaths, targets):
        resolved = target.resolve(strict=False)
        # macOS 常见卷不区分大小写，并可能归一化 Unicode 文件名。
        # 为了保证批处理在任何目标卷上都不静默覆盖，
        # 预检使用保守的 NFC + casefold 键。
        collision_key = unicodedata.normalize("NFC", str(resolved)).casefold()
        if collision_key not in owners:
            owners[collision_key] = (target, [])
        owners[collision_key][1].append(str(source))
    collisions = [
        (target, sources)
        for target, sources in owners.values()
        if len(sources) > 1
    ]
    if collisions:
        details = "; ".join(
            f"{target.name} <- {', '.join(Path(item).name for item in sources)}"
            for target, sources in sorted(collisions, key=lambda item: str(item[0]))
        )
        raise OutputCollisionError(
            f"批处理输出同名冲突：{details}",
            message_key="export.output_collision",
            message_params={"details": details},
        )
    existing = [target for target in targets if target.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"输出文件已存在：{names}")
    return [str(path) for path in targets]


def render_atoms(
    atoms: Atoms,
    config: Optional[RenderConfig] = None,
    draw_bonds: bool = True,
    bond_settings: Optional[BondSettings] = None,
    fig=None,
    ax=None,
    render_context=None,
):
    """从 Atoms 对象渲染到 Figure。"""
    if render_context is not None:
        config = render_context.config
        bond_settings = render_context.bond_settings
        draw_bonds = bond_settings.draw_bonds
    if config is None:
        config = RenderConfig()
    return render_2d(
        atoms,
        config,
        draw_bonds=draw_bonds,
        bond_settings=bond_settings,
        render_context=render_context,
        fig=fig,
        ax=ax,
    )


def render_file(
    filepath: str,
    config: Optional[RenderConfig] = None,
    draw_bonds: bool = True,
    output_path: Optional[str] = None,
    output_format: str = "svg",
    bond_settings: Optional[BondSettings] = None,
    *,
    overwrite: bool = False,
):
    """从文件渲染并可选导出。"""
    atoms = read_structure(filepath)
    fig = render_atoms(
        atoms,
        config,
        draw_bonds=draw_bonds,
        bond_settings=bond_settings,
    )

    if output_path:
        export_figure(
            fig,
            output_format,
            config,
            output_path,
            overwrite=overwrite,
        )

    return fig


def render_batch(
    filepaths: List[str],
    output_dir: str,
    config: Optional[RenderConfig] = None,
    draw_bonds: bool = True,
    output_format: str = "svg",
    bond_settings: Optional[BondSettings] = None,
    *,
    overwrite: bool = False,
):
    """批量渲染。"""
    import os
    output_paths = plan_output_paths(
        filepaths,
        output_dir,
        output_format,
        overwrite=overwrite,
    )
    os.makedirs(output_dir, exist_ok=True)

    if config is None:
        config = RenderConfig()

    results = []
    for filepath, out_path in zip(filepaths, output_paths):
        try:
            atoms = read_structure(filepath)
            fig = render_atoms(
                atoms,
                config,
                draw_bonds=draw_bonds,
                bond_settings=bond_settings,
            )

            export_figure(
                fig,
                output_format,
                config,
                out_path,
                overwrite=overwrite,
            )
            results.append(out_path)

            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            results.append(None)

    return results
