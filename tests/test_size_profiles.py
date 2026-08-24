"""原子半径与键宽双尺寸档案测试。"""

from dataclasses import replace

import pytest
from ase import Atoms
from ase.data import atomic_numbers, covalent_radii

from meia.bond_rules import BondPairRule
from meia.i18n import I18n, Locale, LocalizedError
from meia.size_profiles import (
    CovalentSizeProfile,
    RadiusMode,
    SizeProfileSettings,
    UniformSizeProfile,
    apply_size_profile_edits,
    replace_active_bond_width,
    resolve_active_bond_width,
    resolve_display_radii,
)


def test_nonpositive_size_profile_has_exact_english_diagnostic():
    with pytest.raises(LocalizedError) as captured:
        CovalentSizeProfile(global_scale=0.0)

    assert captured.value.message_key == "atom.value_positive"
    assert I18n(Locale.EN).error_text(captured.value, "atom.apply_failed") == (
        "Atom size value for covalent.global_scale must be greater than 0; "
        "received 0.0."
    )
from meia.visual_state import (
    BondModuleSettings,
    PortableStyle,
    VisualizationState,
    resolve_render_context,
)


def test_default_profiles_preserve_covalent_visuals_and_use_uniform_option_c():
    settings = SizeProfileSettings()

    assert settings.active_mode is RadiusMode.COVALENT
    assert settings.covalent.global_scale == 0.6
    assert settings.covalent.bond_width_ratio == 0.45
    assert settings.uniform.global_scale == 1.0
    assert settings.uniform.reference_radius_angstrom == 0.35
    assert settings.uniform.bond_width_ratio == 0.45
    assert resolve_display_radii(settings, ["H", "O"]) == pytest.approx(
        [covalent_radii[atomic_numbers["H"]] * 0.6,
         covalent_radii[atomic_numbers["O"]] * 0.6]
    )


def test_profiles_resolve_independently_and_freeze_override_inputs():
    covalent_overrides = {"O": 0.8}
    uniform_overrides = {"O": 0.42}
    settings = SizeProfileSettings(
        covalent=CovalentSizeProfile(
            global_scale=0.5,
            reference_overrides_angstrom=covalent_overrides,
            bond_width_ratio=0.3,
        ),
        uniform=UniformSizeProfile(
            global_scale=2.0,
            reference_radius_angstrom=0.2,
            reference_overrides_angstrom=uniform_overrides,
            bond_width_ratio=0.7,
        ),
    )
    covalent_overrides["O"] = 99.0
    uniform_overrides["O"] = 99.0

    assert resolve_display_radii(
        settings, ["H", "O"], mode=RadiusMode.COVALENT
    ) == pytest.approx([covalent_radii[1] * 0.5, 0.4])
    assert resolve_display_radii(
        settings, ["H", "O"], mode=RadiusMode.UNIFORM
    ) == pytest.approx([0.4, 0.84])
    with pytest.raises(TypeError):
        settings.uniform.reference_overrides_angstrom["H"] = 0.3


def test_uniform_base_can_decrease_repeatedly_without_creating_false_overrides():
    settings = SizeProfileSettings()
    for reference_radius in (0.50, 0.40, 0.30):
        settings = apply_size_profile_edits(
            settings,
            mode=RadiusMode.UNIFORM,
            global_scale=1.0,
            uniform_reference_radius_angstrom=reference_radius,
            submitted_display_radii_angstrom={},
        )
        assert dict(settings.uniform.reference_overrides_angstrom) == {}
        assert resolve_display_radii(settings, ["H", "O"]) == pytest.approx(
            [reference_radius, reference_radius]
        )


def test_explicit_uniform_override_survives_base_change_and_scales_proportionally():
    settings = apply_size_profile_edits(
        SizeProfileSettings(),
        mode=RadiusMode.UNIFORM,
        global_scale=1.0,
        uniform_reference_radius_angstrom=0.35,
        submitted_display_radii_angstrom={"O": 0.28},
    )
    settings = apply_size_profile_edits(
        settings,
        mode=RadiusMode.UNIFORM,
        global_scale=1.0,
        uniform_reference_radius_angstrom=0.25,
        submitted_display_radii_angstrom={},
    )
    assert resolve_display_radii(settings, ["H", "O"]) == pytest.approx(
        [0.25, 0.28]
    )

    scaled = apply_size_profile_edits(
        settings,
        mode=RadiusMode.UNIFORM,
        global_scale=0.5,
        uniform_reference_radius_angstrom=0.25,
        submitted_display_radii_angstrom={},
    )
    assert resolve_display_radii(scaled, ["H", "O"]) == pytest.approx(
        [0.125, 0.14]
    )


def test_submitting_default_radius_removes_an_existing_element_override():
    settings = SizeProfileSettings(
        active_mode=RadiusMode.UNIFORM,
        uniform=UniformSizeProfile(
            reference_radius_angstrom=0.35,
            reference_overrides_angstrom={"O": 0.28},
        ),
    )

    edited = apply_size_profile_edits(
        settings,
        mode=RadiusMode.UNIFORM,
        global_scale=1.0,
        uniform_reference_radius_angstrom=0.30,
        submitted_display_radii_angstrom={"O": 0.30},
    )

    assert dict(edited.uniform.reference_overrides_angstrom) == {}


def test_switch_and_bond_width_edit_preserve_the_inactive_profile():
    settings = SizeProfileSettings(
        covalent=CovalentSizeProfile(global_scale=0.7, bond_width_ratio=0.32),
        uniform=UniformSizeProfile(
            global_scale=1.1,
            reference_radius_angstrom=0.3,
            bond_width_ratio=0.58,
        ),
    )
    uniform = apply_size_profile_edits(
        settings,
        mode=RadiusMode.UNIFORM,
        global_scale=1.1,
        uniform_reference_radius_angstrom=0.3,
        submitted_display_radii_angstrom={},
    )
    assert uniform.covalent == settings.covalent
    assert resolve_active_bond_width(uniform) == 0.58

    narrowed = replace_active_bond_width(uniform, 0.41)
    assert narrowed.uniform.bond_width_ratio == 0.41
    assert narrowed.covalent.bond_width_ratio == 0.32
    assert settings.uniform.bond_width_ratio == 0.58


@pytest.mark.parametrize(
    "factory",
    [
        lambda: CovalentSizeProfile(global_scale=0),
        lambda: CovalentSizeProfile(bond_width_ratio=float("nan")),
        lambda: CovalentSizeProfile(reference_overrides_angstrom={"X": 0.2}),
        lambda: UniformSizeProfile(global_scale=True),
        lambda: UniformSizeProfile(reference_radius_angstrom=-0.1),
        lambda: UniformSizeProfile(reference_overrides_angstrom={"O": float("inf")}),
        lambda: SizeProfileSettings(active_mode="ionic"),
    ],
)
def test_invalid_size_profile_values_are_rejected(factory):
    with pytest.raises((TypeError, ValueError)):
        factory()


def test_dataclass_replace_keeps_profile_value_semantics():
    settings = SizeProfileSettings()
    updated = replace(
        settings,
        covalent=replace(settings.covalent, global_scale=0.8),
    )
    assert settings.covalent.global_scale == 0.6
    assert updated.covalent.global_scale == 0.8


def test_render_context_uses_active_profile_for_both_radii_and_bond_width():
    atoms = Atoms("HO", positions=[[0, 0, 0], [0.9, 0, 0]])
    profiles = SizeProfileSettings(
        active_mode=RadiusMode.UNIFORM,
        uniform=UniformSizeProfile(
            global_scale=0.8,
            reference_radius_angstrom=0.30,
            reference_overrides_angstrom={"O": 0.25},
            bond_width_ratio=0.62,
        ),
    )
    state = VisualizationState(
        style=PortableStyle(
            size_profiles=profiles,
            bonds=BondModuleSettings(
                pair_rules=(BondPairRule("H", "O", 0.8, 1.2),)
            ),
        )
    )

    context = resolve_render_context(atoms, state)

    assert context.config.get_atom_radii(["H", "O"]) == pytest.approx(
        [0.24, 0.20]
    )
    assert context.config.bond_width_ratio == 0.62
    assert context.bond_settings.style.width_ratio == 0.62
