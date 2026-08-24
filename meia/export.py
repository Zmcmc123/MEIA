"""
导出模块。

将 Matplotlib Figure 导出为 SVG / PNG / PDF。
"""

import io
from pathlib import Path
from typing import Mapping, Optional
from xml.etree import ElementTree as ET
import matplotlib.pyplot as plt

from .config import RenderConfig
from .i18n import LocalizedError


SVG_NAMESPACE = "http://www.w3.org/2000/svg"
XLINK_NAMESPACE = "http://www.w3.org/1999/xlink"
BOND_ROLES = (
    "cap-a",
    "cap-b",
    "rect-a",
    "rect-b",
    "outline-1",
    "outline-2",
)


class SVGGroupingError(ValueError):
    """MEIA 的 SVG 语义键组不完整或不一致。"""


class ExportGenerationError(LocalizedError):
    """图像在编码或写入边界上导出失败。"""


def postprocess_meia_svg(
    svg_bytes: bytes,
    bond_manifest: Optional[Mapping[str, Mapping]] = None,
    atom_manifest: Optional[Mapping[str, Mapping]] = None,
    hydrogen_bond_manifest: Optional[Mapping[str, Mapping]] = None,
) -> bytes:
    """合并共价键组，并注释已有的原子与氢键包装器。"""
    if not bond_manifest and not atom_manifest and not hydrogen_bond_manifest:
        return svg_bytes

    try:
        root = ET.fromstring(svg_bytes)
    except ET.ParseError as exc:
        raise SVGGroupingError(f"Matplotlib SVG 无法解析：{exc}") from exc

    ET.register_namespace("", SVG_NAMESPACE)
    ET.register_namespace("xlink", XLINK_NAMESPACE)

    if atom_manifest:
        atom_wrappers: dict[str, ET.Element] = {}
        for node in root.iter():
            atom_id = node.attrib.get("id")
            if atom_id not in atom_manifest:
                continue
            if atom_id in atom_wrappers:
                raise SVGGroupingError(f"原子实例 {atom_id} 的包装器重复")
            atom_wrappers[atom_id] = node
        for atom_id, metadata in atom_manifest.items():
            wrapper = atom_wrappers.get(atom_id)
            if wrapper is None:
                raise SVGGroupingError(f"原子实例 {atom_id} 缺少 SVG 包装器")
            wrapper.set(
                "data-meia-source-atom-index",
                str(metadata["source_atom_index"]),
            )
            wrapper.set(
                "data-meia-replica-translation",
                ",".join(
                    str(value) for value in metadata["replica_translation"]
                ),
            )
            wrapper.set(
                "data-meia-image-shift",
                ",".join(str(value) for value in metadata["image_shift"]),
            )

    if hydrogen_bond_manifest:
        hydrogen_wrappers: dict[str, ET.Element] = {}
        for node in root.iter():
            instance_id = node.attrib.get("id")
            if instance_id not in hydrogen_bond_manifest:
                continue
            if instance_id in hydrogen_wrappers:
                raise SVGGroupingError(f"氢键实例 {instance_id} 的包装器重复")
            hydrogen_wrappers[instance_id] = node
        for instance_id, metadata in hydrogen_bond_manifest.items():
            wrapper = hydrogen_wrappers.get(instance_id)
            if wrapper is None:
                raise SVGGroupingError(f"氢键实例 {instance_id} 缺少 SVG 包装器")
            wrapper.set("data-donor-oxygen", str(metadata["donor_oxygen"]))
            wrapper.set("data-hydrogen", str(metadata["hydrogen"]))
            wrapper.set("data-acceptor-oxygen", str(metadata["acceptor_oxygen"]))
            for attribute, key in (
                ("data-donor-oxygen-image-shift", "donor_oxygen_image_shift"),
                ("data-hydrogen-image-shift", "hydrogen_image_shift"),
                ("data-acceptor-oxygen-image-shift", "acceptor_oxygen_image_shift"),
            ):
                wrapper.set(
                    attribute,
                    ",".join(str(value) for value in metadata[key]),
                )

    if not bond_manifest:
        return ET.tostring(root, encoding="utf-8", xml_declaration=True)

    role_wrappers: dict[str, dict[str, tuple[ET.Element, ET.Element]]] = {}
    for parent in root.iter():
        for child in list(parent):
            gid = child.attrib.get("id", "")
            if "__" not in gid:
                continue
            bond_id, role = gid.rsplit("__", 1)
            if bond_id not in bond_manifest or role not in BOND_ROLES:
                continue
            roles = role_wrappers.setdefault(bond_id, {})
            if role in roles:
                raise SVGGroupingError(f"键组 {bond_id} 的 {role} 对象重复")
            roles[role] = (parent, child)

    prepared: dict[str, tuple[ET.Element, list[ET.Element], ET.Element]] = {}
    for bond_id, metadata in bond_manifest.items():
        roles = role_wrappers.get(bond_id, {})
        missing = [role for role in BOND_ROLES if role not in roles]
        if missing:
            raise SVGGroupingError(
                f"键组 {bond_id} 缺少对象：{', '.join(missing)}"
            )
        parents = {id(roles[role][0]): roles[role][0] for role in BOND_ROLES}
        if len(parents) != 1:
            raise SVGGroupingError(f"键组 {bond_id} 的对象不在同一个 SVG 图层")
        parent = next(iter(parents.values()))
        wrappers = [roles[role][1] for role in BOND_ROLES]

        group = ET.Element(
            f"{{{SVG_NAMESPACE}}}g",
            {
                "id": bond_id,
                "data-atom-a": str(metadata["atom_i"]),
                "data-atom-b": str(metadata["atom_j"]),
                "data-atom-a-image-shift": ",".join(
                    str(value) for value in metadata["atom_i_image_shift"]
                ),
                "data-atom-b-image-shift": ",".join(
                    str(value) for value in metadata["atom_j_image_shift"]
                ),
                "data-elements": "-".join(metadata["elements"]),
                "data-periodic-offset": ",".join(
                    str(value) for value in metadata["periodic_offset"]
                ),
            },
        )
        for role, wrapper in zip(BOND_ROLES, wrappers):
            visible_children = list(wrapper)
            if len(visible_children) != 1:
                raise SVGGroupingError(
                    f"键组 {bond_id} 的 {role} 应包含一个可见对象，"
                    f"实际为 {len(visible_children)} 个"
                )
            visible = visible_children[0]
            for attribute, value in wrapper.attrib.items():
                if attribute != "id" and attribute not in visible.attrib:
                    visible.set(attribute, value)
            visible.set("data-role", role)
            group.append(visible)
        prepared[bond_id] = (parent, wrappers, group)

    by_parent: dict[int, tuple[ET.Element, dict[ET.Element, str], dict[str, ET.Element]]] = {}
    for bond_id, (parent, wrappers, group) in prepared.items():
        parent_key = id(parent)
        if parent_key not in by_parent:
            by_parent[parent_key] = (parent, {}, {})
        _, wrapper_to_bond, groups = by_parent[parent_key]
        for wrapper in wrappers:
            wrapper_to_bond[wrapper] = bond_id
        groups[bond_id] = group

    for parent, wrapper_to_bond, groups in by_parent.values():
        first_wrapper = {
            bond_id: min(
                (wrapper for wrapper, owner in wrapper_to_bond.items() if owner == bond_id),
                key=lambda wrapper: list(parent).index(wrapper),
            )
            for bond_id in groups
        }
        rebuilt = []
        for child in list(parent):
            bond_id = wrapper_to_bond.get(child)
            if bond_id is None:
                rebuilt.append(child)
            elif first_wrapper[bond_id] is child:
                rebuilt.append(groups[bond_id])
        parent[:] = rebuilt

    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _export_figure_unchecked(
    fig: plt.Figure,
    format: str = "svg",
    config: Optional[RenderConfig] = None,
    filepath: Optional[str] = None,
    *,
    overwrite: bool = False,
) -> Optional[bytes]:
    """导出 Figure 为文件或字节流。

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        要导出的 Figure
    format : str
        导出格式：'svg', 'png', 'pdf'
    config : RenderConfig, optional
        渲染参数（用于 dpi 和 transparent）
    filepath : str, optional
        文件路径；为 None 则返回字节流

    Returns
    -------
    bytes or None
        filepath 为 None 时返回 bytes，否则返回 None
    """
    if config is None:
        config = RenderConfig()

    if filepath is not None and Path(filepath).exists() and not overwrite:
        raise FileExistsError(f"输出文件已存在：{filepath}")

    format = format.lower()
    dpi = config.dpi if format == "png" else None

    if format == "svg":
        buf = io.BytesIO()
        fig.savefig(
            buf,
            format="svg",
            bbox_inches="tight",
            transparent=config.transparent,
        )
        data = postprocess_meia_svg(
            buf.getvalue(),
            getattr(fig, "_meia_bond_manifest", None),
            getattr(fig, "_meia_atom_manifest", None),
            getattr(fig, "_meia_hydrogen_bond_manifest", None),
        )
        if filepath:
            Path(filepath).write_bytes(data)
            return None
        return data

    if filepath:
        fig.savefig(
            filepath,
            format=format,
            dpi=dpi,
            bbox_inches="tight",
            transparent=config.transparent,
        )
        return None
    else:
        buf = io.BytesIO()
        fig.savefig(
            buf,
            format=format,
            dpi=dpi,
            bbox_inches="tight",
            transparent=config.transparent,
        )
        buf.seek(0)
        return buf.read()


def export_figure(
    fig: plt.Figure,
    format: str = "svg",
    config: Optional[RenderConfig] = None,
    filepath: Optional[str] = None,
    *,
    overwrite: bool = False,
) -> Optional[bytes]:
    """导出 Figure，并把可预期的编码/写入失败转换为稳定双语诊断。"""
    try:
        return _export_figure_unchecked(
            fig,
            format,
            config,
            filepath,
            overwrite=overwrite,
        )
    except (ExportGenerationError, FileExistsError):
        raise
    except Exception as exc:
        raise ExportGenerationError(
            f"导出 {format} 失败：{type(exc).__name__}: {exc}",
            message_key="export.generation_error",
            message_params={
                "format": str(format).upper(),
                "path": filepath or "memory",
                "error_type": type(exc).__name__,
            },
        ) from exc


def export_svg(fig: plt.Figure, config: Optional[RenderConfig] = None,
               filepath: Optional[str] = None, *,
               overwrite: bool = False) -> Optional[bytes]:
    """便捷方法：导出 SVG。"""
    return export_figure(fig, "svg", config, filepath, overwrite=overwrite)


def export_png(fig: plt.Figure, config: Optional[RenderConfig] = None,
               filepath: Optional[str] = None, *,
               overwrite: bool = False) -> Optional[bytes]:
    """便捷方法：导出 PNG。"""
    return export_figure(fig, "png", config, filepath, overwrite=overwrite)


def export_pdf(fig: plt.Figure, config: Optional[RenderConfig] = None,
               filepath: Optional[str] = None, *,
               overwrite: bool = False) -> Optional[bytes]:
    """便捷方法：导出 PDF。"""
    return export_figure(fig, "pdf", config, filepath, overwrite=overwrite)
