"""成键判据与周期边界回归测试。"""

from ase import Atoms

from meia.bonds import find_bonds
from meia.bond_rules import BondPairRule, BondSettings, resolve_bonds
from meia.config import RenderConfig
from meia.geometry import compute_bond_geometries
from meia.periodic_display import CellPeriodicSettings, build_periodic_display
from meia.projection import project_periodic_display


def test_bond_cutoff_does_not_include_neighborlist_skin():
    """bond_cutoff=1.0 时，1.10 Å 的 H-H 不应越过 0.62 Å 共价阈值成键。"""
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [1.10, 0.0, 0.0]])

    bonds = find_bonds(atoms, RenderConfig(bond_cutoff=1.0), allowed_pairs=None)

    assert bonds == []


def test_default_allowed_pairs_accepts_ca_o():
    """默认元素对过滤应保留按字典序规范化后的 Ca-O 键。"""
    atoms = Atoms("CaO", positions=[[0.0, 0.0, 0.0], [2.30, 0.0, 0.0]])
    config = RenderConfig(bond_cutoff=1.0)

    bonds = find_bonds(
        atoms,
        config,
        allowed_pairs=config.get_effective_allowed_pairs(),
    )

    assert [(bond.i, bond.j) for bond in bonds] == [(0, 1)]


def test_periodic_bond_is_detected_and_rendered_between_unwrapped_instances():
    """跨晶胞匹配应连接两个实际显示实例，不得形成悬空半键。"""
    atoms = Atoms(
        "H2",
        positions=[[0.10, 0.0, 0.0], [9.60, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    config = RenderConfig(
        bond_cutoff=1.0,
        rotation="0x,0y,0z",
        show_unit_cell=0,
    )

    bonds = find_bonds(atoms, config, allowed_pairs=None)

    assert len(bonds) == 1
    assert bonds[0].offset == (-1, 0, 0)

    settings = BondSettings(
        pair_rules=(BondPairRule("H", "H", 0.4, 0.6),),
    )
    resolution = resolve_bonds(atoms, settings)
    display = build_periodic_display(
        atoms,
        resolution.matched,
        CellPeriodicSettings(show_unit_cell=0),
    )
    projection = project_periodic_display(atoms, display, config, frozenset())
    geometries = compute_bond_geometries(
        display.bond_instances,
        projection,
        config,
    )

    assert len(geometries) == 1
    assert geometries[0].atom_i_image_shift == (0, 0, 0)
    assert geometries[0].atom_j_image_shift == (-1, 0, 0)
