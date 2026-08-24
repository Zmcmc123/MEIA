"""原子构型文件读取的统一入口。"""

from __future__ import annotations

from os import PathLike
from pathlib import Path
import re
from typing import Union

from ase import Atoms
from ase.io import read

from .i18n import LocalizedError


LAMMPS_DATA_EXTENSIONS = {".data", ".lmp"}
SUPPORTED_STRUCTURE_EXTENSIONS = {
    ".vasp", ".cif", ".xyz", ".lmp", ".data", ".pdb", ".mol", ".sdf",
}
EXTENSIONLESS_VASP_FILENAMES = {"poscar", "contcar"}


class StructureReadError(LocalizedError):
    """构型文件无法在 ASE 读取边界上解析。"""


def is_supported_structure_filename(filename: str) -> bool:
    """检查用户可见文件名，同时支持无后缀 POSCAR/CONTCAR。"""
    path = Path(filename)
    return (
        path.name.lower() in EXTENSIONLESS_VASP_FILENAMES
        or path.suffix.lower() in SUPPORTED_STRUCTURE_EXTENSIONS
    )


def read_structure(filepath: Union[str, PathLike]) -> Atoms:
    """通过 ASE 读取构型，并为 LAMMPS data 扩展名显式指定格式。"""
    path = Path(filepath)
    try:
        if path.suffix.lower() in LAMMPS_DATA_EXTENSIONS:
            atom_style = _detect_lammps_atom_style(path)
            return read(path, format="lammps-data", style=atom_style)
        return read(path)
    except StructureReadError:
        raise
    except Exception as exc:
        raise StructureReadError(
            f"无法读取构型 {path}：{type(exc).__name__}: {exc}",
            message_key="file.structure_read_error",
            message_params={
                "filename": str(path),
                "error_type": type(exc).__name__,
                "detail": str(exc),
            },
        ) from exc


def _detect_lammps_atom_style(path: Path) -> str:
    """识别 LAMMPS data 的 Atoms 风格，兼容 ASE 3.22 不自动识别的情况。"""
    lines = path.read_text(errors="replace").splitlines()
    supported_styles = {"atomic", "full", "charge", "molecular", "bond", "angle"}

    for index, line in enumerate(lines):
        match = re.match(r"^\s*Atoms\s*(?:#\s*([A-Za-z]+))?\s*$", line)
        if match is None:
            continue

        declared_style = (match.group(1) or "").lower()
        if declared_style in supported_styles:
            return declared_style

        for data_line in lines[index + 1:]:
            fields = data_line.split("#", 1)[0].split()
            if not fields:
                continue
            if not fields[0].lstrip("+-").isdigit():
                break

            field_count = len(fields)
            if field_count in {5, 8}:
                return "atomic"
            if field_count in {7, 10}:
                return "full"
            if field_count in {6, 9}:
                # charge: id type q x y z; molecular/bond/angle: id mol type x y z
                charge_or_type = fields[2].lower()
                if any(marker in charge_or_type for marker in (".", "e")):
                    return "charge"
                return "molecular"
            break

    # ASE 的历史默认为 full；无法识别时保持该行为，交由 ASE 报错。
    return "full"
