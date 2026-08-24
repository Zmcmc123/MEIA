"""
化学键识别模块。

基于 ASE NeighborList，使用共价半径之和 × 容差因子作为成键判据。
容差因子可调，并支持按元素对类型过滤。

注意：natural_cutoffs 返回的是每个原子的共价半径本身（不是 0.5 倍），
NeighborList 中 pair(i,j) 的成键距离阈值 = cutoffs[i] + cutoffs[j] = r_i + r_j。
因此 bond_cutoff=1.0 表示"距离 ≤ 共价半径之和"才成键。
"""

from dataclasses import dataclass
from typing import List, Optional, Set, Tuple
import numpy as np
from ase import Atoms
from ase.neighborlist import NeighborList, natural_cutoffs

from .config import RenderConfig


@dataclass
class Bond:
    """一根化学键。"""
    i: int   # 原子 A 索引
    j: int   # 原子 B 索引
    offset: Tuple[int, int, int] = (0, 0, 0)  # B 相对 A 的周期晶胞偏移


def find_bonds(
    atoms: Atoms,
    config: RenderConfig,
    allowed_pairs: Optional[Set[Tuple[str, str]]] = None,
) -> List[Bond]:
    """识别化学键。

    Parameters
    ----------
    atoms : Atoms
        ASE Atoms 对象
    config : RenderConfig
        渲染参数（使用 bond_cutoff）
    allowed_pairs : set of (str, str) tuples, optional
        允许的元素对（排序后），如 {('Ca','O'), ('O','Si')}
        为 None 则允许所有元素对

    Returns
    -------
    List[Bond]
        化学键列表
    """
    cutoffs = [c * config.bond_cutoff for c in natural_cutoffs(atoms)]
    # 成键阈值必须严格由共价半径与 bond_cutoff 决定；ASE 默认 skin=0.3 Å
    # 仅适合邻居表更新缓冲，会把阈值之外的原子也返回为候选邻居。
    nl = NeighborList(
        cutoffs,
        skin=0.0,
        self_interaction=False,
        bothways=True,
    )
    nl.update(atoms)

    symbols = atoms.get_chemical_symbols()
    bonds = []

    for i in range(len(atoms)):
        indices, offsets = nl.get_neighbors(i)
        for j, offset in zip(indices, offsets):
            if j > i:
                if allowed_pairs is not None:
                    pair = tuple(sorted([symbols[i], symbols[j]]))
                    if pair not in allowed_pairs:
                        continue
                bonds.append(
                    Bond(
                        i=i,
                        j=int(j),
                        offset=tuple(int(value) for value in offset),
                    )
                )

    return bonds


def find_bonds_with_exclusions(
    atoms: Atoms,
    config: RenderConfig,
    exclude: Optional[List[tuple]] = None,
    include: Optional[List[tuple]] = None,
    allowed_pairs: Optional[Set[Tuple[str, str]]] = None,
) -> List[Bond]:
    """识别化学键，支持手动排除和额外包含。"""
    bonds = find_bonds(atoms, config, allowed_pairs=allowed_pairs)

    if exclude:
        exclude_set = {tuple(sorted(p)) for p in exclude}
        bonds = [b for b in bonds if tuple(sorted((b.i, b.j))) not in exclude_set]

    if include:
        existing = {tuple(sorted((b.i, b.j))) for b in bonds}
        for i, j in include:
            key = tuple(sorted((i, j)))
            if key not in existing:
                bonds.append(Bond(i=key[0], j=key[1]))
                existing.add(key)

    return bonds
