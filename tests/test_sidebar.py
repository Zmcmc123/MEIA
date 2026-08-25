"""Streamlit 侧边栏表单的显式提交边界测试。"""

from contextlib import nullcontext
from dataclasses import replace
import re

import pytest

from ase import Atoms

from meia.sidebar import (
    AtomFormSubmission,
    BondFormSubmission,
    ATOM_SELECTION_DRAFT_REVISION_KEY,
    initialize_visual_state,
    load_visual_state,
    render_atom_cell_form,
    render_cell_periodic_form,
    render_bond_form,
    render_export_form,
    render_atom_selection_form,
    store_visual_state,
)
from meia.size_profiles import (
    CovalentSizeProfile,
    RadiusMode as ProfileRadiusMode,
    SizeProfileSettings,
    UniformSizeProfile,
)
from meia.bond_rules import BondStrokeStyle
from meia.visual_state import (
    AtomCellSettings,
    ExportSettings,
    PortableStyle,
    VisualizationState,
)
import meia.sidebar as sidebar_module


def _default_atom_cell():
    return AtomCellSettings()


def _default_profiles():
    return SizeProfileSettings()


def test_atom_form_explains_buffered_dual_profile_application(monkeypatch):
    atoms = Atoms("HO", positions=[[0, 0, 0], [1, 0, 0]])
    current = _default_atom_cell()
    profiles = SizeProfileSettings()
    fake = FakeStreamlit(
        keyed_session=True,
        values={"meia_atom_cell_radius_mode": ProfileRadiusMode.UNIFORM.value},
    )
    monkeypatch.setattr("meia.sidebar.st", fake)

    result = render_atom_cell_form(
        current,
        profiles,
        atoms,
        current.element_colors,
    )

    assert result is None
    assert profiles.active_mode is ProfileRadiusMode.COVALENT
    assert (
        "共价半径与相等半径分别保存一套原子大小和化学键粗细的默认参数；"
        "切换方案后，点击“应用原子设置”即可整套应用对应参数。"
    ) in fake.captions


def test_atom_form_applies_target_profile_and_atom_settings_together(monkeypatch):
    atoms = Atoms("HO", positions=[[0, 0, 0], [1, 0, 0]])
    current = _default_atom_cell()
    profiles = SizeProfileSettings(
        uniform=UniformSizeProfile(
            global_scale=1.0,
            reference_radius_angstrom=0.35,
            bond_width_ratio=0.61,
        )
    )
    fake = FakeStreamlit(
        submitted={"应用原子设置"},
        keyed_session=True,
        values={
            "meia_atom_cell_radius_mode": ProfileRadiusMode.UNIFORM.value,
            "meia_atom_cell_uniform_global_scale": 1.0,
            "meia_atom_cell_uniform_reference_radius": 0.30,
        },
    )
    monkeypatch.setattr("meia.sidebar.st", fake)

    result = render_atom_cell_form(
        current,
        profiles,
        atoms,
        current.element_colors,
    )

    assert isinstance(result, AtomFormSubmission)
    assert result.size_profiles.active_mode is ProfileRadiusMode.UNIFORM
    assert result.size_profiles.uniform.reference_radius_angstrom == 0.30
    assert result.size_profiles.uniform.bond_width_ratio == 0.61
    assert result.atom_cell.outline_width == current.outline_width


def test_restore_colors_does_not_activate_an_unapplied_profile(monkeypatch):
    atoms = Atoms("O", positions=[[0, 0, 0]])
    default = _default_atom_cell()
    current = replace(
        default,
        element_colors={**dict(default.element_colors), "O": "#336699"},
    )
    profiles = SizeProfileSettings()
    fake = FakeStreamlit(
        submitted={"恢复默认元素配色"},
        keyed_session=True,
        values={"meia_atom_cell_radius_mode": ProfileRadiusMode.UNIFORM.value},
    )
    monkeypatch.setattr("meia.sidebar.st", fake)

    result = render_atom_cell_form(
        current,
        profiles,
        atoms,
        default.element_colors,
    )

    assert isinstance(result, AtomFormSubmission)
    assert result.size_profiles is profiles
    assert dict(result.atom_cell.element_colors) == dict(default.element_colors)


def test_bond_form_updates_only_the_active_profile_width(monkeypatch):
    fake = FakeStreamlit(
        submitted={"应用化学键设置"},
        values={"统一键宽比例": 0.30},
    )
    monkeypatch.setattr("meia.sidebar.st", fake)
    profiles = SizeProfileSettings(
        active_mode=ProfileRadiusMode.UNIFORM,
        uniform=UniformSizeProfile(bond_width_ratio=0.55),
    )

    result = render_bond_form(
        _bond_settings(),
        profiles,
        Atoms("HO", positions=[[0, 0, 0], [1, 0, 0]]),
    )

    assert isinstance(result, BondFormSubmission)
    assert result.size_profiles.uniform.bond_width_ratio == 0.30
    assert result.size_profiles.covalent.bond_width_ratio == 0.45
from meia.atom_styles import (
    AtomColorOverride,
    AtomColorStrength,
    AtomSelectionSettings,
    HiddenAtom,
)
from meia.bond_rules import (
    AtomBondOverride,
    BondPairRule,
    BondStyle,
    OverrideVisibility,
)
from meia.hydrogen_bonds import HydrogenBondSettings
from meia.i18n import I18n, Locale
from meia.visual_state import BondModuleSettings
from meia.periodic_display import CellPeriodicSettings, PeriodicRange


class FakeStreamlit:
    def __init__(self, *, submitted=(), values=None, keyed_session=False, session_state=None):
        self.submitted = set(submitted)
        self.values = dict(values or {})
        self.keyed_session = keyed_session
        self.session_state = dict(session_state or {})
        self.form_calls = []
        self.errors = []
        self.warnings = []
        self.widget_labels = []
        self.selectbox_options = {}
        self.plotly_figures = []
        self.captions = []
        self.widget_kwargs = {}
        self.formatted_options = {}

    def _widget_value(self, label, default, kwargs):
        """可选地模拟 Streamlit 的 keyed widget 首次默认值和重跑优先级。"""
        key = kwargs.get("key")
        if self.keyed_session and key is not None:
            if key not in self.session_state:
                self.session_state[key] = self.values.get(
                    key,
                    self.values.get(label, default),
                )
            return self.session_state[key]
        return self.values.get(label, default)

    def form(self, key, clear_on_submit=False):
        self.form_calls.append((key, clear_on_submit))
        return nullcontext()

    def selectbox(self, label, options, index=0, **_kwargs):
        self.widget_labels.append(label)
        self.selectbox_options[label] = tuple(options)
        self.widget_kwargs[label] = dict(_kwargs)
        return self._widget_value(label, options[index], _kwargs)

    def text_input(self, label, value="", **_kwargs):
        self.widget_labels.append(label)
        self.widget_kwargs[label] = dict(_kwargs)
        return self._widget_value(label, value, _kwargs)

    def slider(self, label, minimum, maximum, value, step, **_kwargs):
        self.widget_labels.append(label)
        self.widget_kwargs[label] = {
            **dict(_kwargs),
            "min_value": minimum,
            "max_value": maximum,
            "step": step,
        }
        return self._widget_value(label, value, _kwargs)

    def number_input(self, label, *args, value=None, **_kwargs):
        self.widget_labels.append(label)
        if value is None:
            value = args[2]
        self.widget_kwargs[label] = dict(_kwargs)
        return self._widget_value(label, value, _kwargs)

    def checkbox(self, label, value=False, **_kwargs):
        self.widget_labels.append(label)
        return self._widget_value(label, value, _kwargs)

    def color_picker(self, label, value, **_kwargs):
        self.widget_labels.append(label)
        return self.values.get(label, value)

    def form_submit_button(self, label, **_kwargs):
        return label in self.submitted

    def multiselect(self, label, options, **_kwargs):
        self.widget_labels.append(label)
        self.widget_kwargs[label] = dict(_kwargs)
        format_func = _kwargs.get("format_func")
        if format_func is not None:
            self.formatted_options[label] = tuple(
                format_func(option) for option in options
            )
        key = _kwargs.get("key")
        if key in self.session_state:
            return self.session_state[key]
        return self.values.get(label, _kwargs.get("default", []))

    def markdown(self, _value, **_kwargs):
        return None

    def caption(self, value, **_kwargs):
        self.captions.append(value)
        return None

    def plotly_chart(self, figure, **_kwargs):
        self.plotly_figures.append(figure)
        return None

    def error(self, message):
        self.errors.append(message)

    def warning(self, message):
        self.warnings.append(message)


class StrictBoundsFakeStreamlit(FakeStreamlit):
    """模拟 Streamlit 对初始值越过控件边界时的拒绝行为。"""

    @staticmethod
    def _require_value_in_bounds(value, minimum, maximum):
        if not minimum <= value <= maximum:
            raise ValueError(f"widget value {value} outside [{minimum}, {maximum}]")

    def slider(self, label, minimum, maximum, value, step, **kwargs):
        self._require_value_in_bounds(value, minimum, maximum)
        return super().slider(label, minimum, maximum, value, step, **kwargs)

    def number_input(self, label, *args, value=None, **kwargs):
        if value is None:
            value = args[2]
        minimum = kwargs.get("min_value", args[0] if args else None)
        maximum = kwargs.get("max_value", args[1] if len(args) > 1 else None)
        widget_value = value
        key = kwargs.get("key")
        if self.keyed_session and key in self.session_state:
            widget_value = self.session_state[key]
        self._require_value_in_bounds(widget_value, minimum, maximum)
        return super().number_input(label, *args, value=value, **kwargs)


def test_sidebar_no_longer_exposes_removed_view_form():
    """侧栏视角已迁回主页面，旧入口必须删除以免状态分叉。"""
    assert not hasattr(sidebar_module, "render_view_form")
    assert not hasattr(sidebar_module, "VIEW_PRESETS")


def _render_atom_form(atoms, current, fake, profiles=None):
    import meia.sidebar as sidebar

    original = sidebar.st
    sidebar.st = fake
    try:
        return sidebar.render_atom_cell_form(
            current,
            profiles or _default_profiles(),
            atoms,
            current.element_colors,
        )
    finally:
        sidebar.st = original


def test_atom_form_shows_only_present_elements_and_does_not_apply_drafts(monkeypatch):
    atoms = Atoms("HOH", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    current = _default_atom_cell()
    fake = FakeStreamlit(
        values={"半径方案": ProfileRadiusMode.UNIFORM.value, "全局半径缩放": 1.0},
        keyed_session=True,
    )
    monkeypatch.setattr("meia.sidebar.st", fake)

    profiles = _default_profiles()
    result = render_atom_cell_form(
        current, profiles, atoms, current.element_colors
    )

    assert result is None
    assert fake.widget_labels[:2] == ["半径方案", "全局半径缩放"]
    assert "H 最终显示半径 / Å" in fake.widget_labels
    assert "O 最终显示半径 / Å" in fake.widget_labels
    assert "C 最终显示半径 / Å" not in fake.widget_labels
    assert fake.widget_labels.index("H 最终显示半径 / Å") < fake.widget_labels.index(
        "轮廓粗细"
    )
    kwargs = fake.widget_kwargs["H 最终显示半径 / Å"]
    assert kwargs["min_value"] == 0.01
    assert kwargs["max_value"] == 5.0
    assert kwargs["step"] == 0.01
    assert kwargs["format"] == "%.2f"
    assert fake.selectbox_options["半径方案"] == tuple(
        mode.value for mode in ProfileRadiusMode
    )
    assert fake.widget_kwargs["全局半径缩放"]["min_value"] == 0.1
    assert fake.widget_kwargs["全局半径缩放"]["max_value"] == 1.5
    assert fake.widget_kwargs["全局半径缩放"]["step"] == 0.05
    assert "统一基础半径 / Å" in fake.widget_labels
    assert fake.session_state["meia_atom_cell_radius_mode"] == ProfileRadiusMode.UNIFORM.value
    assert profiles.active_mode is ProfileRadiusMode.COVALENT
    assert fake.form_calls == [("meia_atom_cell_form", False)]


@pytest.mark.parametrize(
    (
        "scale",
        "uniform_reference",
        "override",
        "expected_scale_bound",
        "expected_uniform_bound",
        "expected_radius_bound",
    ),
    [
        (2.0, 10.0, 12.0, (0.1, 2.0), (0.1, 10.0), (0.01, 24.0)),
        (0.05, 0.05, 0.1, (0.05, 1.5), (0.05, 5.0), (0.005, 5.0)),
    ],
)
def test_atom_form_safely_mounts_legal_imported_radii_outside_preferred_ui_range(
    monkeypatch,
    scale,
    uniform_reference,
    override,
    expected_scale_bound,
    expected_uniform_bound,
    expected_radius_bound,
):
    """严格 v7 允许正有限值，侧栏必须能承接超出常用编辑范围的已应用状态。"""
    atoms = Atoms("O", positions=[[0, 0, 0]])
    current = _default_atom_cell()
    profiles = SizeProfileSettings(
        active_mode=ProfileRadiusMode.UNIFORM,
        uniform=UniformSizeProfile(
            global_scale=scale,
            reference_radius_angstrom=uniform_reference,
            reference_overrides_angstrom={"O": override},
        ),
    )
    fake = StrictBoundsFakeStreamlit(keyed_session=True)
    monkeypatch.setattr("meia.sidebar.st", fake)

    assert render_atom_cell_form(
        current, profiles, atoms, current.element_colors
    ) is None
    scale_kwargs = fake.widget_kwargs["全局半径缩放"]
    uniform_kwargs = fake.widget_kwargs["统一基础半径 / Å"]
    radius_kwargs = fake.widget_kwargs["O 最终显示半径 / Å"]
    assert (
        scale_kwargs["min_value"],
        scale_kwargs["max_value"],
    ) == pytest.approx(expected_scale_bound)
    assert (
        uniform_kwargs["min_value"],
        uniform_kwargs["max_value"],
    ) == pytest.approx(expected_uniform_bound)
    assert (
        radius_kwargs["min_value"],
        radius_kwargs["max_value"],
    ) == pytest.approx(expected_radius_bound)


def test_atom_form_keeps_large_element_draft_in_bounds_after_scale_rerun(
    monkeypatch,
):
    """改倍率触发重跑时，旧元素草稿不得越过新计算值对应的控件边界。"""
    atoms = Atoms("O", positions=[[0, 0, 0]])
    current = _default_atom_cell()
    profiles = SizeProfileSettings(
        active_mode=ProfileRadiusMode.UNIFORM,
        uniform=UniformSizeProfile(
            global_scale=2.0,
            reference_radius_angstrom=10.0,
            reference_overrides_angstrom={"O": 12.0},
        ),
    )
    fake = StrictBoundsFakeStreamlit(keyed_session=True)
    monkeypatch.setattr("meia.sidebar.st", fake)

    assert render_atom_cell_form(
        current, profiles, atoms, current.element_colors
    ) is None
    assert fake.session_state["meia_atom_cell_uniform_display_radius_O"] == 24.0

    fake.session_state["meia_atom_cell_uniform_global_scale"] = 0.1
    assert render_atom_cell_form(
        current, profiles, atoms, current.element_colors
    ) is None
    radius_kwargs = fake.widget_kwargs["O 最终显示半径 / Å"]
    assert fake.session_state["meia_atom_cell_uniform_display_radius_O"] == (
        pytest.approx(1.2)
    )
    assert radius_kwargs["max_value"] == pytest.approx(5.0)


def test_radius_draft_keys_preserve_profiles_across_mode_reruns(monkeypatch):
    atoms = Atoms("HO", positions=[[0, 0, 0], [1, 0, 0]])
    current = _default_atom_cell()
    profiles = _default_profiles()
    fake = FakeStreamlit(
        keyed_session=True,
        values={
            "meia_atom_cell_radius_mode": ProfileRadiusMode.UNIFORM.value,
            "meia_atom_cell_uniform_global_scale": 1.0,
            "meia_atom_cell_uniform_display_radius_H": 0.8,
            "meia_atom_cell_uniform_display_radius_O": 1.2,
        },
        session_state={
            "meia_atom_cell_covalent_display_radius_H": 0.2,
            "meia_atom_cell_covalent_display_radius_O": 0.4,
        },
    )
    monkeypatch.setattr("meia.sidebar.st", fake)

    assert render_atom_cell_form(
        current, profiles, atoms, current.element_colors
    ) is None
    assert fake.session_state["meia_atom_cell_uniform_display_radius_H"] == 0.8
    assert fake.session_state["meia_atom_cell_uniform_display_radius_O"] == 1.2
    assert fake.session_state["meia_atom_cell_covalent_display_radius_H"] == pytest.approx(0.2)
    assert fake.session_state["meia_atom_cell_covalent_display_radius_O"] == pytest.approx(0.4)

    fake.session_state["meia_atom_cell_radius_mode"] = ProfileRadiusMode.COVALENT.value
    assert render_atom_cell_form(
        current, profiles, atoms, current.element_colors
    ) is None
    assert fake.session_state["meia_atom_cell_covalent_display_radius_H"] == 0.2
    assert profiles.active_mode is ProfileRadiusMode.COVALENT


def test_atom_form_global_scale_change_has_no_element_overrides(monkeypatch):
    atoms = Atoms("HO", positions=[[0, 0, 0], [1, 0, 0]])
    current = _default_atom_cell()
    fake = FakeStreamlit(
        keyed_session=True,
    )
    monkeypatch.setattr("meia.sidebar.st", fake)
    profiles = _default_profiles()

    assert render_atom_cell_form(
        current, profiles, atoms, current.element_colors
    ) is None
    fake.session_state["meia_atom_cell_covalent_global_scale"] = 0.8
    fake.submitted.add("应用原子设置")

    result = render_atom_cell_form(
        current, profiles, atoms, current.element_colors
    )

    assert result is not None
    assert result.size_profiles.covalent.global_scale == 0.8
    assert dict(result.size_profiles.covalent.reference_overrides_angstrom) == {}
    assert dict(result.size_profiles.uniform.reference_overrides_angstrom) == {}


def test_atom_form_element_edit_writes_only_active_profile(monkeypatch):
    atoms = Atoms("HO", positions=[[0, 0, 0], [1, 0, 0]])
    current = _default_atom_cell()
    fake = FakeStreamlit(
        submitted={"应用原子设置"},
        keyed_session=True,
        values={"meia_atom_cell_covalent_display_radius_O": 0.7},
    )
    monkeypatch.setattr("meia.sidebar.st", fake)

    result = render_atom_cell_form(
        current, _default_profiles(), atoms, current.element_colors
    )

    assert result is not None
    assert result.size_profiles.active_mode is ProfileRadiusMode.COVALENT
    assert result.size_profiles.covalent.reference_overrides_angstrom["O"] == (
        pytest.approx(0.7 / 0.6)
    )
    assert "H" not in result.size_profiles.covalent.reference_overrides_angstrom
    assert dict(result.size_profiles.uniform.reference_overrides_angstrom) == {}


def test_atom_form_scale_and_edited_element_use_absolute_precedence(monkeypatch):
    atoms = Atoms("HO", positions=[[0, 0, 0], [1, 0, 0]])
    current = _default_atom_cell()
    fake = FakeStreamlit(
        keyed_session=True,
    )
    monkeypatch.setattr("meia.sidebar.st", fake)
    profiles = _default_profiles()

    assert render_atom_cell_form(
        current, profiles, atoms, current.element_colors
    ) is None
    fake.session_state["meia_atom_cell_covalent_global_scale"] = 0.8
    fake.session_state["meia_atom_cell_covalent_display_radius_O"] = 0.7
    fake.submitted.add("应用原子设置")

    result = render_atom_cell_form(
        current, profiles, atoms, current.element_colors
    )

    assert result is not None
    assert result.size_profiles.covalent.global_scale == 0.8
    assert result.size_profiles.covalent.reference_overrides_angstrom["O"] == (
        pytest.approx(0.875)
    )
    assert "H" not in result.size_profiles.covalent.reference_overrides_angstrom


def test_atom_form_scale_only_resubmit_does_not_create_stale_overrides(monkeypatch):
    """已应用的全局倍率改变必须重置旧控件草稿，不得在下次提交时写入伪覆盖。"""
    atoms = Atoms("HO", positions=[[0, 0, 0], [1, 0, 0]])
    current = _default_atom_cell()
    fake = FakeStreamlit(
        submitted={"应用原子设置"},
        keyed_session=True,
        values={"meia_atom_cell_covalent_global_scale": 0.8},
    )
    monkeypatch.setattr("meia.sidebar.st", fake)

    profiles = _default_profiles()
    first = render_atom_cell_form(
        current, profiles, atoms, current.element_colors
    )
    assert first is not None
    assert first.size_profiles.covalent.global_scale == 0.8
    assert dict(first.size_profiles.covalent.reference_overrides_angstrom) == {}

    second = render_atom_cell_form(
        first.atom_cell,
        first.size_profiles,
        atoms,
        first.atom_cell.element_colors,
    )

    assert second is not None
    assert second.size_profiles.covalent.global_scale == 0.8
    assert dict(second.size_profiles.covalent.reference_overrides_angstrom) == {}
    assert fake.session_state["meia_atom_cell_covalent_display_radius_H"] == pytest.approx(
        0.31 * 0.8
    )
    assert fake.session_state["meia_atom_cell_covalent_display_radius_O"] == pytest.approx(
        0.66 * 0.8
    )


def test_atom_form_uniform_base_change_does_not_turn_unchanged_fields_into_overrides(
    monkeypatch,
):
    """切换到相等半径并只改基础半径时，未编辑元素应继承新基值。"""
    atoms = Atoms("HO", positions=[[0, 0, 0], [1, 0, 0]])
    current = _default_atom_cell()
    fake = FakeStreamlit(
        submitted={"应用原子设置"},
        keyed_session=True,
        values={
            "meia_atom_cell_radius_mode": ProfileRadiusMode.UNIFORM.value,
            "meia_atom_cell_uniform_reference_radius": 1.2,
        },
    )
    monkeypatch.setattr("meia.sidebar.st", fake)

    result = render_atom_cell_form(
        current, _default_profiles(), atoms, current.element_colors
    )

    assert result is not None
    assert result.size_profiles.active_mode is ProfileRadiusMode.UNIFORM
    assert result.size_profiles.uniform.reference_radius_angstrom == 1.2
    assert dict(result.size_profiles.uniform.reference_overrides_angstrom) == {}


def test_atom_form_uniform_switch_preserves_covalent_profile(monkeypatch):
    atoms = Atoms("HO", positions=[[0, 0, 0], [1, 0, 0]])
    current = _default_atom_cell()
    profiles = SizeProfileSettings(
        covalent=CovalentSizeProfile(
            reference_overrides_angstrom={"O": 0.8}
        ),
        uniform=UniformSizeProfile(
            reference_overrides_angstrom={"O": 1.4}
        ),
    )
    fake = FakeStreamlit(
        submitted={"应用原子设置"},
        keyed_session=True,
        values={
            "meia_atom_cell_radius_mode": ProfileRadiusMode.UNIFORM.value,
            "meia_atom_cell_uniform_global_scale": 0.6,
            "meia_atom_cell_uniform_reference_radius": 1.2,
            "meia_atom_cell_uniform_display_radius_H": 0.72,
            "meia_atom_cell_uniform_display_radius_O": 0.84,
        },
    )
    monkeypatch.setattr("meia.sidebar.st", fake)

    result = render_atom_cell_form(
        current, profiles, atoms, current.element_colors
    )

    assert result is not None
    assert dict(result.size_profiles.covalent.reference_overrides_angstrom) == {
        "O": 0.8
    }
    assert dict(result.size_profiles.uniform.reference_overrides_angstrom) == {
        "O": 1.4
    }
    assert result.size_profiles.uniform.reference_radius_angstrom == 1.2


def test_atom_form_invalid_candidate_is_atomic(monkeypatch):
    atoms = Atoms("HO", positions=[[0, 0, 0], [1, 0, 0]])
    current = _default_atom_cell()
    fake = FakeStreamlit(
        submitted={"应用原子设置"},
        keyed_session=True,
        session_state={"meia_atom_cell_covalent_display_radius_O": -1},
    )
    monkeypatch.setattr("meia.sidebar.st", fake)

    result = render_atom_cell_form(
        current, _default_profiles(), atoms, current.element_colors
    )

    assert result is None
    assert len(fake.errors) == 1
    assert fake.errors == ["O 显示半径必须大于 0 Å；收到 -1。"]
    assert current == _default_atom_cell()


def test_english_atom_validation_keeps_value_without_han(monkeypatch):
    atoms = Atoms("HO", positions=[[0, 0, 0], [1, 0, 0]])
    current = _default_atom_cell()
    fake = FakeStreamlit(
        submitted={"Apply Atom Settings"},
        keyed_session=True,
        session_state={"meia_atom_cell_covalent_display_radius_O": -1},
    )
    monkeypatch.setattr("meia.sidebar.st", fake)

    assert render_atom_cell_form(
        current,
        _default_profiles(),
        atoms,
        current.element_colors,
        I18n(Locale.EN),
    ) is None
    assert len(fake.errors) == 1
    assert "-1" in fake.errors[0]
    assert not re.search(r"[\u3400-\u9fff]", fake.errors[0])


def test_atom_form_restore_colors_only_restores_palette(monkeypatch):
    atoms = Atoms("O", positions=[[0, 0, 0]])
    default = _default_atom_cell()
    current = replace(
        default,
        element_colors={**dict(default.element_colors), "O": "#336699"},
    )
    profiles = SizeProfileSettings(
        covalent=CovalentSizeProfile(global_scale=0.8)
    )
    fake = FakeStreamlit(
        submitted={"恢复默认元素配色"},
        keyed_session=True,
        values={
            "meia_atom_cell_covalent_global_scale": 0.5,
            "轮廓粗细": 0.9,
        },
    )
    monkeypatch.setattr("meia.sidebar.st", fake)

    result = render_atom_cell_form(
        current, profiles, atoms, default.element_colors
    )

    assert result is not None
    assert result.size_profiles is profiles
    assert result.atom_cell.outline_width == current.outline_width
    assert dict(result.atom_cell.element_colors) == dict(default.element_colors)


def test_cell_periodic_form_normalizes_non_pbc_and_reports_instance_count(
    monkeypatch,
):
    """非 PBC 轴即使有草稿输入，应仍以 [0,1) 原子化应用。"""
    atoms = Atoms(
        "H2",
        positions=[[0, 0, 0], [1, 0, 0]],
        cell=[5, 5, 5],
        pbc=[True, False, True],
    )
    fake = FakeStreamlit(
        submitted={"应用晶胞与周期性设置"},
        values={
            "a 起点": -1,
            "a 终点": 2,
            "b 起点": -2,
            "b 终点": 3,
            "c 起点": 0,
            "c 终点": 2,
        },
    )
    monkeypatch.setattr("meia.sidebar.st", fake)

    result = render_cell_periodic_form(CellPeriodicSettings(), atoms)

    assert result is not None
    assert result.a == PeriodicRange(-1, 2)
    assert result.b == PeriodicRange(0, 1)
    assert result.c == PeriodicRange(0, 2)
    assert fake.widget_kwargs["b 起点"]["disabled"] is True
    assert fake.widget_kwargs["b 终点"]["disabled"] is True
    assert any("非 PBC" in caption and "b" in caption for caption in fake.captions)
    assert "当前周期数：a=3，b=1，c=2" in fake.captions
    assert "预计显示 12 个原子实例" in fake.captions


def test_cell_periodic_form_rejects_over_limit_atomically(monkeypatch):
    """超出 50,000 实例时不得返回部分周期状态。"""
    atoms = Atoms(
        "H" * 1001,
        positions=[[0, 0, 0]] * 1001,
        cell=[5, 5, 5],
        pbc=True,
    )
    fake = FakeStreamlit(
        submitted={"应用晶胞与周期性设置"},
        values={
            "a 起点": 0,
            "a 终点": 5,
            "b 起点": 0,
            "b 终点": 5,
            "c 起点": 0,
            "c 终点": 2,
        },
    )
    monkeypatch.setattr("meia.sidebar.st", fake)

    result = render_cell_periodic_form(CellPeriodicSettings(), atoms)

    assert result is None
    assert "预计显示 50,050 个原子实例" in fake.captions
    assert fake.errors == [
        "周期显示将生成 50,050 个原子实例，超过 50,000 个的上限。"
    ]


def test_atom_cell_restore_returns_full_default_palette(monkeypatch):
    atoms = Atoms("O", positions=[[0, 0, 0]])
    default = _default_atom_cell()
    edited = replace(
        default,
        element_colors={**dict(default.element_colors), "O": "#336699"},
    )
    fake = FakeStreamlit(submitted={"恢复默认元素配色"})
    monkeypatch.setattr("meia.sidebar.st", fake)

    result = render_atom_cell_form(
        edited, _default_profiles(), atoms, default.element_colors
    )

    assert result is not None
    assert dict(result.atom_cell.element_colors) == dict(default.element_colors)


def test_export_form_is_buffered_and_returns_complete_settings(monkeypatch):
    current = ExportSettings()
    fake = FakeStreamlit(
        submitted={"应用导出设置"},
        values={"导出格式": "PNG", "PNG 分辨率 (DPI)": 900, "透明背景": False},
    )
    monkeypatch.setattr("meia.sidebar.st", fake)

    result = render_export_form(current)

    assert result == ExportSettings("png", 900, False)
    assert fake.form_calls == [("meia_export_form", False)]
    assert fake.session_state == {}


def test_visual_state_session_seam_round_trips_typed_state():
    session = {}
    state = VisualizationState(style=PortableStyle(atom_cell=_default_atom_cell()))

    store_visual_state(session, state)

    assert load_visual_state(session) is state


def _bond_settings() -> BondModuleSettings:
    return BondModuleSettings(
        draw_bonds=True,
        style=BondStrokeStyle(0.25, "#231815"),
        pair_rules=(
            BondPairRule("H", "O", 0.0, 1.2, enabled=True),
            BondPairRule("Ca", "O", 2.1, 2.8, enabled=False),
            BondPairRule("Si", "O", 1.4, 1.9, enabled=True),
        ),
    )


def test_bond_form_returns_none_until_submitted(monkeypatch):
    fake = FakeStreamlit(values={"显示普通化学键": False})
    monkeypatch.setattr("meia.sidebar.st", fake)

    result = render_bond_form(
        _bond_settings(),
        _default_profiles(),
        Atoms("HO", positions=[[0, 0, 0], [1, 0, 0]]),
    )

    assert result is None
    assert fake.form_calls == [("meia_bond_form", False)]


def test_bond_form_shows_only_matched_rules_and_current_elements_for_addition(
    monkeypatch,
):
    """匹配规则保持精简，但新增入口只提供当前构型元素。"""
    fake = FakeStreamlit()
    monkeypatch.setattr("meia.sidebar.st", fake)

    render_bond_form(
        _bond_settings(),
        _default_profiles(),
        Atoms("HO", positions=[[0, 0, 0], [1, 0, 0]]),
    )

    assert len(fake.plotly_figures) == 1
    table = fake.plotly_figures[0].data[0]
    assert list(table.cells.values[0]) == ["H–O"]
    assert list(table.cells.values[4]) == [1]
    assert "显示 H–O" in fake.widget_labels
    assert not any("Ca–O" in label for label in fake.widget_labels)
    assert not any("O–Si" in label for label in fake.widget_labels)
    assert "新增该元素对" in fake.widget_labels
    assert fake.selectbox_options["元素 A"] == ("H", "O")
    assert fake.selectbox_options["元素 B"] == ("H", "O")
    assert "删除元素对" not in fake.widget_labels


def test_bond_form_adds_a_current_structure_element_pair(monkeypatch):
    """新增元素对应保留已有规则并追加规范化的默认规则。"""
    fake = FakeStreamlit(
        submitted={"应用化学键设置"},
        values={
            "新增该元素对": True,
            "元素 A": "Ca",
            "元素 B": "O",
        },
    )
    monkeypatch.setattr("meia.sidebar.st", fake)
    current = BondModuleSettings(
        pair_rules=(BondPairRule("H", "O", 0.0, 1.2),),
    )

    result = render_bond_form(
        current,
        _default_profiles(),
        Atoms("HOCa", positions=[[0, 0, 0], [1, 0, 0], [5, 0, 0]]),
    )

    assert result is not None
    by_pair = {rule.pair: rule for rule in result.bonds.pair_rules}
    assert set(by_pair) == {("H", "O"), ("Ca", "O")}
    assert by_pair[("Ca", "O")].min_distance == 0.0
    assert by_pair[("Ca", "O")].enabled is True
    assert by_pair[("Ca", "O")].participates_in_periodic_unwrap is True


def test_bond_form_submit_returns_total_global_state_and_keeps_zero_match_rule(
    monkeypatch,
):
    fake = FakeStreamlit(
        submitted={"应用化学键设置"},
        values={
            "显示普通化学键": False,
            "统一键宽比例": 0.30,
            "统一描边粗细": 0.10,
        },
    )
    monkeypatch.setattr("meia.sidebar.st", fake)

    result = render_bond_form(
        _bond_settings(),
        _default_profiles(),
        Atoms("HO", positions=[[0, 0, 0], [1, 0, 0]]),
    )

    assert result is not None
    assert result.bonds.draw_bonds is False
    assert result.size_profiles.covalent.bond_width_ratio == 0.30
    assert result.bonds.style.stroke_width == 0.10
    by_pair = {rule.pair: rule for rule in result.bonds.pair_rules}
    assert set(by_pair) == {("H", "O"), ("Ca", "O"), ("O", "Si")}
    assert by_pair[("Ca", "O")] == _bond_settings().pair_rules[1]
    assert by_pair[("O", "Si")] == _bond_settings().pair_rules[2]


def test_bond_form_shows_independent_topology_and_hydrogen_controls(
    monkeypatch,
):
    """显示开关、周期整理和氢键阈值必须是彼此独立的可见控件。"""
    fake = FakeStreamlit()
    monkeypatch.setattr("meia.sidebar.st", fake)
    current = BondModuleSettings(
        pair_rules=(
            BondPairRule(
                "H",
                "O",
                0.0,
                1.2,
                enabled=True,
                participates_in_periodic_unwrap=False,
            ),
        ),
        hydrogen_bonds=HydrogenBondSettings(True, 2.5, 120.0),
    )

    render_bond_form(
        current,
        _default_profiles(),
        Atoms("HO", positions=[[0, 0, 0], [1, 0, 0]]),
    )

    table = fake.plotly_figures[0].data[0]
    assert list(table.header.values) == [
        "元素对",
        "显示",
        "周期整理",
        "距离范围 / Å",
        "当前匹配",
    ]
    assert "参与 H–O 周期整理" in fake.widget_labels
    assert "显示氢键" in fake.widget_labels
    assert "H···O 最大距离 / Å" in fake.widget_labels
    assert "O–H···O 最小夹角 / °" in fake.widget_labels


def test_bond_form_submits_pair_and_hydrogen_state_atomically(monkeypatch):
    """键控控件先采纳 current，重跑提交时再原子化采纳会话草稿。"""
    fake = FakeStreamlit(keyed_session=True)
    monkeypatch.setattr("meia.sidebar.st", fake)
    current = BondModuleSettings(
        pair_rules=(
            BondPairRule(
                "H",
                "O",
                0.0,
                1.2,
                participates_in_periodic_unwrap=False,
            ),
        ),
        hydrogen_bonds=HydrogenBondSettings(True, 2.5, 120.0),
    )

    assert render_bond_form(
        current,
        _default_profiles(),
        Atoms("HO", positions=[[0, 0, 0], [1, 0, 0]]),
    ) is None
    assert {
        key: fake.session_state[key]
        for key in (
            "meia_bond_form_H_O_periodic",
            "meia_bond_form_hydrogen_draw",
            "meia_bond_form_hydrogen_maximum",
            "meia_bond_form_hydrogen_angle",
        )
    } == {
        "meia_bond_form_H_O_periodic": False,
        "meia_bond_form_hydrogen_draw": True,
        "meia_bond_form_hydrogen_maximum": 2.5,
        "meia_bond_form_hydrogen_angle": 120.0,
    }

    fake.session_state.update(
        {
            "meia_bond_form_H_O_enabled": False,
            "meia_bond_form_H_O_periodic": True,
            "meia_bond_form_hydrogen_draw": False,
            "meia_bond_form_hydrogen_maximum": 2.3,
            "meia_bond_form_hydrogen_angle": 135.0,
        }
    )
    fake.submitted.add("应用化学键设置")
    result = render_bond_form(
        current,
        _default_profiles(),
        Atoms("HO", positions=[[0, 0, 0], [1, 0, 0]]),
    )

    assert result is not None
    rule = result.bonds.pair_rules[0]
    assert rule.enabled is False
    assert rule.participates_in_periodic_unwrap is True
    assert result.bonds.hydrogen_bonds == HydrogenBondSettings(False, 2.3, 135.0)


def test_bond_form_rejects_invalid_distance_without_returning_partial_state(
    monkeypatch,
):
    fake = FakeStreamlit(
        submitted={"应用化学键设置"},
        values={"H–O 最大距离 / Å": -0.1},
    )
    monkeypatch.setattr("meia.sidebar.st", fake)

    result = render_bond_form(
        _bond_settings(),
        _default_profiles(),
        Atoms("HO", positions=[[0, 0, 0], [1, 0, 0]]),
    )

    assert result is None
    assert fake.errors


def test_english_bond_validation_keeps_distance_without_han(monkeypatch):
    fake = FakeStreamlit(
        submitted={"Apply Bond Settings"},
        values={"H–O Maximum Distance / Å": -0.1},
    )
    monkeypatch.setattr("meia.sidebar.st", fake)

    assert render_bond_form(
        _bond_settings(),
        _default_profiles(),
        Atoms("HO", positions=[[0, 0, 0], [1, 0, 0]]),
        I18n(Locale.EN),
    ) is None
    assert "-0.1" in fake.errors[0]
    assert not re.search(r"[\u3400-\u9fff]", fake.errors[0])


def test_atom_selection_form_uses_deterministic_union_then_invert_algorithm(
    monkeypatch,
):
    atoms = Atoms(
        symbols=["H", "O", "Ca", "C"],
        positions=[[0, 0, 0], [1, 0, 0], [5, 0, 0], [8, 0, 0]],
    )
    fake = FakeStreamlit(
        submitted={"应用原子操作"},
        values={
            "当前选择（可搜索）": [0],
            "按序号加入选择": "2",
            "按元素加入选择": ["Ca"],
            "反选最终集合": True,
        },
    )
    monkeypatch.setattr("meia.sidebar.st", fake)

    result = render_atom_selection_form(
        AtomSelectionSettings(),
        atoms,
        (("H", "O"), ("Ca", "O")),
    )

    assert result is not None
    assert result.selected_atom_indices == (3,)
    assert result.color_overrides == ()
    assert result.color_strengths == ()
    assert result.bond_overrides == ()


def test_atom_selection_form_keeps_full_searchable_list_below_large_threshold(
    monkeypatch,
):
    atoms = Atoms(symbols=["H"] * 999, positions=[[0, 0, 0]] * 999)
    fake = FakeStreamlit()
    monkeypatch.setattr("meia.sidebar.st", fake)

    assert render_atom_selection_form(AtomSelectionSettings(), atoms, ()) is None

    assert len(fake.formatted_options["当前选择（可搜索）"]) == 999


def test_atom_selection_form_pages_atom_options_at_large_threshold(monkeypatch):
    atoms = Atoms(symbols=["H"] * 1000, positions=[[0, 0, 0]] * 1000)
    current = AtomSelectionSettings(selected_atom_indices=(0, 500, 999))
    fake = FakeStreamlit()
    monkeypatch.setattr("meia.sidebar.st", fake)

    assert render_atom_selection_form(current, atoms, ()) is None

    assert "当前选择（可搜索）" not in fake.formatted_options
    assert max(len(options) for options in fake.formatted_options.values()) <= 200
    assert len(fake.formatted_options["当前页原子（最多 200 个）"]) == 200
    assert "当前已选择 3 个原子。" in fake.captions
    assert any("H #1" in caption and "H #1000" in caption for caption in fake.captions)


def test_large_atom_selection_combines_page_index_and_element_union(monkeypatch):
    symbols = ["H"] * 997 + ["O", "Ca", "C"]
    atoms = Atoms(symbols=symbols, positions=[[0, 0, 0]] * 1000)
    current = AtomSelectionSettings(selected_atom_indices=(900,))
    fake = FakeStreamlit(
        submitted={"应用原子操作"},
        values={
            "原子页码（共 5 页）": 1,
            "当前页原子（最多 200 个）": [5],
            "当前页操作": "add",
            "按序号加入选择": "250",
            "按元素加入选择": ["Ca"],
        },
    )
    monkeypatch.setattr("meia.sidebar.st", fake)

    result = render_atom_selection_form(current, atoms, ())

    assert result is not None
    assert result.selected_atom_indices == (5, 249, 900, 998)


def test_atom_selection_clear_has_highest_precedence(monkeypatch):
    atoms = Atoms("HO", positions=[[0, 0, 0], [1, 0, 0]])
    fake = FakeStreamlit(
        submitted={"应用原子操作"},
        values={
            "当前选择（可搜索）": [0, 1],
            "反选最终集合": True,
            "清空选择": True,
        },
    )
    monkeypatch.setattr("meia.sidebar.st", fake)

    result = render_atom_selection_form(
        AtomSelectionSettings(), atoms, (("H", "O"),)
    )

    assert result is not None
    assert result.selected_atom_indices == ()


def test_atom_selection_form_does_not_offer_or_apply_subject_emphasis(monkeypatch):
    atoms = Atoms("HH", positions=[[0, 0, 0], [1, 0, 0]])
    current = AtomSelectionSettings(
        selected_atom_indices=(0,),
        default_color_strength=0.65,
    )
    fake = FakeStreamlit(
        submitted={"应用原子操作"},
        values={
            "当前选择（可搜索）": [0],
            "强调当前选区为主体": True,
            "背景色彩强度": 30,
        },
    )
    monkeypatch.setattr("meia.sidebar.st", fake)

    result = render_atom_selection_form(
        current,
        atoms,
        (),
    )

    assert result is not None
    assert result.selected_atom_indices == (0,)
    assert result.default_color_strength == pytest.approx(0.65)
    assert result.color_strengths == ()
    assert "强调当前选区为主体" not in fake.widget_labels
    assert "背景色彩强度" not in fake.widget_labels


def test_atom_selection_reset_revision_uses_fresh_widget_identity(monkeypatch):
    """还原后选择表单应换用新 key，隔离浏览器仍持有的旧草稿。"""
    atoms = Atoms("HO", positions=[[0, 0, 0], [1, 0, 0]])
    fake = FakeStreamlit(
        keyed_session=True,
        session_state={
            ATOM_SELECTION_DRAFT_REVISION_KEY: 1,
            "meia_atom_selection_range": "1",
        },
    )
    monkeypatch.setattr("meia.sidebar.st", fake)

    assert (
        render_atom_selection_form(
            AtomSelectionSettings(), atoms, (("H", "O"),)
        )
        is None
    )

    assert fake.widget_kwargs["按序号加入选择"]["key"] == (
        "meia_atom_selection_range__reset_1"
    )
    assert fake.session_state["meia_atom_selection_range__reset_1"] == ""
    assert fake.session_state["meia_atom_selection_range"] == "1"


def test_atom_selection_pending_widget_state_is_not_combined_with_default(
    monkeypatch,
):
    """3D 选择写入 widget state 后，不应再传 default 触发 Streamlit 警告。"""
    atoms = Atoms("HO", positions=[[0, 0, 0], [1, 0, 0]])
    fake = FakeStreamlit(submitted={"应用原子操作"})
    fake.session_state["meia_atom_selection_indices"] = [1]
    monkeypatch.setattr("meia.sidebar.st", fake)

    result = render_atom_selection_form(
        AtomSelectionSettings(selected_atom_indices=(0,)),
        atoms,
        (("H", "O"),),
    )

    assert result is not None
    assert result.selected_atom_indices == (1,)
    assert "default" not in fake.widget_kwargs["当前选择（可搜索）"]


def test_atom_selection_unchecked_operations_preserve_existing_records(monkeypatch):
    atoms = Atoms("HO", positions=[[0, 0, 0], [1, 0, 0]])
    current = AtomSelectionSettings(
        selected_atom_indices=(0,),
        color_overrides=(AtomColorOverride(0, "H", "#336699"),),
        color_strengths=(AtomColorStrength(0, "H", 0.3),),
        bond_overrides=(
            AtomBondOverride(0, "H", "H", "O", OverrideVisibility.HIDE),
        ),
    )
    fake = FakeStreamlit(
        submitted={"应用原子操作"},
        values={"当前选择（可搜索）": [1]},
    )
    monkeypatch.setattr("meia.sidebar.st", fake)

    result = render_atom_selection_form(current, atoms, (("H", "O"),))

    assert result is not None
    assert result.selected_atom_indices == (1,)
    assert result.color_overrides == current.color_overrides
    assert result.color_strengths == current.color_strengths
    assert result.bond_overrides == current.bond_overrides


def test_atom_selection_pair_operation_updates_only_compatible_atoms(monkeypatch):
    atoms = Atoms(
        symbols=["H", "O", "Ca"],
        positions=[[0, 0, 0], [1, 0, 0], [5, 0, 0]],
    )
    fake = FakeStreamlit(
        submitted={"应用原子操作"},
        values={
            "当前选择（可搜索）": [0, 1, 2],
            "H–O 化学键规则": "hide",
        },
    )
    monkeypatch.setattr("meia.sidebar.st", fake)

    result = render_atom_selection_form(
        AtomSelectionSettings(), atoms, (("H", "O"), ("Ca", "O"))
    )

    assert result is not None
    assert {(item.atom_index, item.pair) for item in result.bond_overrides} == {
        (0, ("H", "O")),
        (1, ("H", "O")),
    }


def test_atom_selection_form_restores_hidden_atom_and_sets_hydrogen_rule(
    monkeypatch,
):
    """恢复原子可见性和设置氢键例外必须在同一次原子操作中提交。"""
    atoms = Atoms("HO", positions=[[0, 0, 0], [1, 0, 0]])
    current = AtomSelectionSettings(
        selected_atom_indices=(0,),
        hidden_atoms=(HiddenAtom(0, "H"),),
    )
    fake = FakeStreamlit(
        submitted={"应用原子操作"},
        values={
            "修改原子显示状态": True,
            "原子显示操作": "show",
            "氢键规则": "hide",
        },
    )
    monkeypatch.setattr("meia.sidebar.st", fake)

    result = render_atom_selection_form(current, atoms, (("H", "O"),))

    assert result is not None
    assert result.hidden_atoms == ()
    assert result.hydrogen_bond_overrides[0].atom_index == 0
    assert (
        result.hydrogen_bond_overrides[0].visibility
        is OverrideVisibility.HIDE
    )
    assert fake.formatted_options["当前选择（可搜索）"] == (
        "H #1（已隐藏）",
        "O #2",
    )
