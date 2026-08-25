"""MEIA Streamlit 交互界面。"""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import os
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components
from ase import Atoms

from meia import __version__
from meia.atom_styles import replace_selected_indices
from meia.brand import (
    DEFAULT_EXPORT_STEM,
    LOCALE_STORAGE_KEY,
    PRODUCT_FULL_NAME,
    PRODUCT_NAME,
    STYLE_JSON_SUFFIX,
    WORKSPACE_JSON_SUFFIX,
)
from meia.bond_rules import normalize_element_pair
from meia.display_complexity import measure_display_complexity
from meia.export import export_figure
from meia.i18n import I18n, Locale, LocalizedError
from meia.io import StructureReadError, is_supported_structure_filename, read_structure
from meia.locale_state import (
    APP_LOCALE_KEY,
    APP_LOCALE_SOURCE_KEY,
    APP_LOCALE_WIDGET_KEY,
    initialize_locale,
    load_locale,
    locale_cookie_markup,
    set_manual_locale,
)
from meia.presets import (
    PresetError,
    StylePreset,
    WorkspaceSnapshot,
    apply_style_preset,
    load_default_style,
    parse_preset,
    style_preset_to_json,
    workspace_snapshot_to_json,
)
from meia.preview import (
    PREVIEW_CSS_HEIGHT,
    PREVIEW_CSS_WIDTH,
    preview_image_html,
    render_preview_png,
)
from meia.preview_state import (
    PreviewArtifact,
    PreviewKey,
    PreviewStatus,
    preview_status,
    should_render_preview,
)
from meia.sidebar import (
    ATOM_SELECTION_DRAFT_REVISION_KEY,
    atom_selection_draft_widget_key,
    initialize_visual_state,
    load_visual_state,
    matched_bond_pairs,
    render_atom_cell_form,
    render_atom_selection_form,
    render_bond_form,
    render_cell_periodic_form,
    render_export_form,
    store_visual_state,
)
from meia.view import render_2d
from meia.viewer import atom_viewer, create_3d_figure
from meia.view_state import (
    AppliedViewState,
    ViewerEventError,
    accept_apply_camera_event,
    accept_atom_selection_batch_event,
    accept_atom_selection_event,
    camera_for_lattice_axis,
    camera_to_rotation_matrix,
    load_applied_view_state,
    store_applied_view_state,
    update_applied_view,
)
from meia.visual_state import (
    PortableStyle,
    VisualizationState,
    apply_camera_only,
    replace_atom_and_size_profiles,
    replace_atom_selection,
    replace_bonds_and_size_profiles,
    replace_cell_periodic,
    replace_export,
    reset_visual_modules_from_style,
    resolve_render_context,
)
from meia.workspace import (
    ActiveWorkspace,
    PendingSnapshot,
    activate_upload,
    build_style_preset,
    build_workspace_snapshot,
    confirm_pending_snapshot,
    stage_snapshot,
)


ACTIVE_WORKSPACE_KEY = "meia_active_workspace"
LAST_UPLOAD_SHA256_KEY = "meia_last_seen_upload_sha256"
VISUAL_STRUCTURE_ID_KEY = "meia_visual_state_structure_id"
PENDING_SNAPSHOT_KEY = "meia_pending_workspace_snapshot"
PENDING_SNAPSHOT_HASH_KEY = "meia_pending_workspace_snapshot_sha256"
HANDLED_SNAPSHOT_HASH_KEY = "meia_handled_workspace_snapshot_sha256"
SNAPSHOT_CONFIRMATION_KEY = "meia_snapshot_overwrite_confirmed"
SNAPSHOT_CONFIRMATION_RESET_KEY = "meia_reset_snapshot_confirmation"
RESET_STYLE_BASELINE_KEY = "meia_reset_style_baseline"
PREVIEW_ARTIFACT_KEY = "meia_2d_preview_artifact"
THREE_D_INTERACTION_CAPTION = I18n(Locale.ZH_CN).text(
    "viewer.interaction_caption"
)
LOCALE_DRAFT_SNAPSHOT_KEY = "meia_locale_draft_snapshot"
LOCALE_DRAFT_PREFIXES = (
    "meia_atom_cell_",
    "meia_bond_form_",
    "meia_cell_periodic_",
    "meia_atom_selection_",
    "meia_export_form_",
)
LOCALE_DRAFT_KEYS = ("meia_preset_name",)


def _initialize_i18n() -> I18n:
    """从 Cookie 或请求语言初始化与可视化状态隔离的界面语言。"""
    current = initialize_locale(
        st.session_state,
        stored_locale=st.context.cookies.get(LOCALE_STORAGE_KEY),
        accept_language=st.context.headers.get("Accept-Language"),
    )
    if st.session_state.get(APP_LOCALE_SOURCE_KEY) == "manual":
        components.html(locale_cookie_markup(current), height=0, width=0)
    return I18n(current)


def _render_locale_selector(i18n: I18n) -> I18n:
    """在侧栏顶部渲染始终可见的手动语言开关。"""
    current = load_locale(st.session_state)
    if current is None:
        raise RuntimeError("locale must be initialized before rendering its selector")
    selected = st.sidebar.radio(
        i18n.text("locale.selector.label"),
        options=(Locale.ZH_CN, Locale.EN),
        index=0 if current is Locale.ZH_CN else 1,
        format_func=lambda locale: i18n.text(
            "locale.option.zh" if locale is Locale.ZH_CN else "locale.option.en"
        ),
        horizontal=True,
        key=APP_LOCALE_WIDGET_KEY,
        label_visibility="collapsed",
    )
    selected_locale = Locale(selected)
    if selected_locale is not current:
        st.session_state[LOCALE_DRAFT_SNAPSHOT_KEY] = {
            key: value
            for key, value in st.session_state.items()
            if key.startswith(LOCALE_DRAFT_PREFIXES) or key in LOCALE_DRAFT_KEYS
        }
        set_manual_locale(st.session_state, selected_locale)
        st.rerun()
        return I18n(selected_locale)
    draft_snapshot = st.session_state.pop(LOCALE_DRAFT_SNAPSHOT_KEY, None)
    if isinstance(draft_snapshot, dict):
        for key, value in draft_snapshot.items():
            st.session_state[key] = value
    return i18n


def _reset_snapshot_confirmation_for_payload(
    session_state,
    payload_sha256: str,
) -> bool:
    """新快照内容必须重新得到覆盖确认。"""
    changed = session_state.get(PENDING_SNAPSHOT_HASH_KEY) != payload_sha256
    if changed:
        session_state[SNAPSHOT_CONFIRMATION_KEY] = False
    return changed


def _consume_snapshot_confirmation_reset(session_state) -> None:
    """在复选框实例化前清理上次取消或成功导入的状态。"""
    if session_state.pop(SNAPSHOT_CONFIRMATION_RESET_KEY, False):
        session_state[SNAPSHOT_CONFIRMATION_KEY] = False


def _read_uploaded_structure(uploaded) -> tuple[bytes, Atoms]:
    uploaded_bytes = uploaded.getvalue()
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded.name}") as tmp:
            tmp.write(uploaded_bytes)
            tmp_path = tmp.name
        try:
            atoms = read_structure(tmp_path)
        except StructureReadError as exc:
            cause = exc.__cause__ or exc
            raise StructureReadError(
                f"unable to read uploaded structure {uploaded.name}: {cause}",
                message_key="file.structure_read_error",
                message_params={
                    "filename": uploaded.name,
                    "error_type": type(cause).__name__,
                    "detail": str(cause),
                },
            ) from exc
    finally:
        if tmp_path is not None and os.path.exists(tmp_path):
            os.unlink(tmp_path)
    return uploaded_bytes, atoms


RESET_WIDGET_PREFIXES = (
    "meia_atom_cell_",
    "meia_bond_form_",
    "meia_cell_periodic_",
    "meia_atom_selection_",
)

RESET_WIDGET_KEYS = (
    "meia_processed_selection_event_id",
    "meia_pending_atom_selection_indices",
)
RESET_WIDGET_REINITIALIZE_KEY = "meia_reset_widget_reinitialize"


def _clear_reset_scoped_widgets(session_state) -> None:
    """只清理还原模块草稿与选择事件，保留视角、导出和快照状态。"""
    for key in list(session_state):
        if key.startswith(RESET_WIDGET_PREFIXES) or key in RESET_WIDGET_KEYS:
            del session_state[key]


def _consume_reset_widget_reinitialize(session_state) -> bool:
    """在还原后的下一次运行开头再次清理浏览器回传的旧控件值。"""
    if not session_state.pop(RESET_WIDGET_REINITIALIZE_KEY, False):
        return False
    _clear_reset_scoped_widgets(session_state)
    return True


def _advance_atom_selection_draft_revision(session_state) -> int:
    """切换选择表单控件身份，避免浏览器恢复已经作废的旧草稿。"""
    previous = session_state.get(ATOM_SELECTION_DRAFT_REVISION_KEY, 0)
    if not isinstance(previous, int) or previous < 0:
        previous = 0
    revision = previous + 1
    session_state[ATOM_SELECTION_DRAFT_REVISION_KEY] = revision
    return revision


def _clear_buffered_visual_widgets() -> None:
    prefixes = (
        "meia_view_form_",
        "meia_atom_cell_",
        "meia_bond_form_",
        "meia_cell_periodic_",
        "meia_atom_selection_",
        "meia_export_form_",
    )
    for key in list(st.session_state):
        if key.startswith(prefixes):
            del st.session_state[key]
    _advance_atom_selection_draft_revision(st.session_state)


def _active_workspace_from_upload(uploaded) -> tuple[ActiveWorkspace | None, bool]:
    current = st.session_state.get(ACTIVE_WORKSPACE_KEY)
    if current is not None and not isinstance(current, ActiveWorkspace):
        current = None
        st.session_state.pop(ACTIVE_WORKSPACE_KEY, None)
    if uploaded is None:
        return current, False
    if not is_supported_structure_filename(uploaded.name):
        raise LocalizedError(
            "unsupported structure filename or extension",
            message_key="file.unsupported_filename",
            message_params={"filename": uploaded.name},
        )

    payload = uploaded.getvalue()
    upload_sha256 = sha256(payload).hexdigest()
    last_seen = st.session_state.get(LAST_UPLOAD_SHA256_KEY)
    if current is not None and upload_sha256 == last_seen:
        return current, False

    payload, atoms = _read_uploaded_structure(uploaded)
    active, observed_sha256, replaced_structure = activate_upload(
        current,
        last_seen,
        payload,
        uploaded.name,
        atoms,
    )
    st.session_state[ACTIVE_WORKSPACE_KEY] = active
    st.session_state[LAST_UPLOAD_SHA256_KEY] = observed_sha256
    return active, replaced_structure


def _reset_visual_state_for_structure(
    active: ActiveWorkspace | None,
    default_style: StylePreset,
    *,
    force: bool = False,
):
    structure_id = active.structure_id if active is not None else "no-structure"
    if (
        force
        or st.session_state.get(VISUAL_STRUCTURE_ID_KEY) != structure_id
        or "meia_visual_state" not in st.session_state
    ):
        atoms = active.atoms if active is not None else Atoms()
        state = initialize_visual_state(atoms, default_style)
        store_visual_state(st.session_state, state)
        st.session_state[VISUAL_STRUCTURE_ID_KEY] = structure_id
        st.session_state["meia_processed_selection_event_id"] = None
        _clear_buffered_visual_widgets()
        return state
    return load_visual_state(st.session_state)


def _style_import_notice(i18n: I18n | None = None) -> str:
    return (i18n or I18n(Locale.ZH_CN)).text("file.style_applied")


def _commit_confirmed_snapshot(
    session_state,
    pending: PendingSnapshot,
    *,
    confirmed: bool,
) -> tuple[ActiveWorkspace, VisualizationState]:
    """先完成快照规范化与校验，成功后才同时替换会话状态。"""
    activated, normalized_state = confirm_pending_snapshot(
        pending,
        confirmed=confirmed,
    )
    session_state[ACTIVE_WORKSPACE_KEY] = activated
    store_visual_state(session_state, normalized_state)
    session_state[VISUAL_STRUCTURE_ID_KEY] = activated.structure_id
    return activated, normalized_state


def _periodic_diagnostic_notice(
    render_context,
    i18n: I18n | None = None,
) -> str | None:
    i18n = i18n or I18n(Locale.ZH_CN)
    diagnostics = render_context.periodic_display.diagnostics
    if not diagnostics:
        return None
    affected_atoms = {
        atom_index
        for diagnostic in diagnostics
        for atom_index in diagnostic.atom_indices
    }
    pairs = sorted(
        {
            normalize_element_pair(*pair)
            for diagnostic in diagnostics
            for pair in diagnostic.conflicting_element_pairs
        }
    )
    pair_text = i18n.text("common.list_separator").join(
        f"{element_a}–{element_b}" for element_a, element_b in pairs
    )
    pair_clause = (
        i18n.text("periodic.diagnostic_pairs", pairs=pair_text)
        if pair_text
        else ""
    )
    return i18n.text(
        "periodic.diagnostic",
        conflicts=len(diagnostics),
        atoms=len(affected_atoms),
        pair_clause=pair_clause,
    )


def _render_json_imports(
    active: ActiveWorkspace | None,
    visual_state,
    style_upload,
    snapshot_upload,
    container,
    i18n: I18n | None = None,
) -> None:
    i18n = i18n or I18n(Locale.ZH_CN)
    with container:
        parsed_style = None
        if style_upload is not None:
            try:
                parsed_style = parse_preset(style_upload.getvalue())
                if not isinstance(parsed_style, StylePreset):
                    raise PresetError(
                        "workspace snapshot supplied to the style field",
                        message_key="preset.workspace_in_style_slot",
                    )
            except PresetError as exc:
                st.error(i18n.error_text(exc, "file.style_invalid"))
                return

        parsed_snapshot = None
        if snapshot_upload is not None:
            try:
                parsed_snapshot = parse_preset(snapshot_upload.getvalue())
                if not isinstance(parsed_snapshot, WorkspaceSnapshot):
                    raise PresetError(
                        "style preset supplied to the workspace snapshot field",
                        message_key="preset.style_in_workspace_slot",
                    )
            except PresetError as exc:
                st.error(i18n.error_text(exc, "file.snapshot_invalid"))
                return

        _consume_snapshot_confirmation_reset(st.session_state)
        with st.form("meia_style_import_form", clear_on_submit=False):
            apply_style = st.form_submit_button(
                i18n.text("file.apply_style"),
                type="primary",
                disabled=active is None or parsed_style is None,
            )
        if apply_style:
            try:
                if active is None:
                    raise PresetError(
                        "a structure is required before applying a style",
                        message_key="preset.structure_required",
                    )
                if isinstance(parsed_style, StylePreset):
                    updated = apply_style_preset(
                        visual_state,
                        parsed_style,
                        active.atoms,
                    )
                else:
                    raise PresetError(
                        "workspace snapshot supplied to the style field",
                        message_key="preset.workspace_in_style_slot",
                    )
            except (PresetError, TypeError, ValueError) as exc:
                st.error(i18n.error_text(exc, "file.style_not_applied"))
            else:
                store_visual_state(st.session_state, updated)
                st.session_state[RESET_STYLE_BASELINE_KEY] = parsed_style.style
                st.session_state[VISUAL_STRUCTURE_ID_KEY] = active.structure_id
                _clear_buffered_visual_widgets()
                st.session_state["meia_notice"] = _style_import_notice(i18n)
                st.rerun()

        pending_snapshot = st.session_state.get(PENDING_SNAPSHOT_KEY)
        if not isinstance(pending_snapshot, PendingSnapshot):
            pending_snapshot = None
            st.session_state.pop(PENDING_SNAPSHOT_KEY, None)
        if snapshot_upload is None:
            pending_snapshot = None
            st.session_state.pop(PENDING_SNAPSHOT_KEY, None)
            st.session_state.pop(PENDING_SNAPSHOT_HASH_KEY, None)
            st.session_state.pop(HANDLED_SNAPSHOT_HASH_KEY, None)
            st.session_state[SNAPSHOT_CONFIRMATION_KEY] = False
        else:
            snapshot_payload = snapshot_upload.getvalue()
            snapshot_sha256 = sha256(snapshot_payload).hexdigest()
            _reset_snapshot_confirmation_for_payload(
                st.session_state,
                snapshot_sha256,
            )
            already_handled = (
                snapshot_sha256
                == st.session_state.get(HANDLED_SNAPSHOT_HASH_KEY)
            )
            staged_sha256 = st.session_state.get(PENDING_SNAPSHOT_HASH_KEY)
            if not already_handled and (
                pending_snapshot is None or staged_sha256 != snapshot_sha256
            ):
                pending_snapshot = stage_snapshot(parsed_snapshot)
                st.session_state[PENDING_SNAPSHOT_KEY] = pending_snapshot
                st.session_state[PENDING_SNAPSHOT_HASH_KEY] = snapshot_sha256
            elif already_handled:
                pending_snapshot = None

        if pending_snapshot is not None:
            st.info(
                i18n.text(
                    "file.snapshot_pending",
                    filename=snapshot_upload.name,
                    source=pending_snapshot.source_name,
                    count=pending_snapshot.atom_count,
                )
            )
            st.warning(i18n.text("file.snapshot_warning"))

        with st.form("meia_snapshot_import_form", clear_on_submit=False):
            if SNAPSHOT_CONFIRMATION_KEY not in st.session_state:
                st.session_state[SNAPSHOT_CONFIRMATION_KEY] = False
            confirmed = st.checkbox(
                i18n.text("file.snapshot_confirm"),
                disabled=pending_snapshot is None,
                key=SNAPSHOT_CONFIRMATION_KEY,
            )
            apply_snapshot = st.form_submit_button(
                i18n.text("file.apply_snapshot"),
                type="primary",
                disabled=pending_snapshot is None,
            )
            cancel_snapshot = st.form_submit_button(
                i18n.text("file.cancel_snapshot"),
                disabled=pending_snapshot is None,
            )
        if cancel_snapshot:
            staged_sha256 = st.session_state.get(PENDING_SNAPSHOT_HASH_KEY)
            if staged_sha256 is not None:
                st.session_state[HANDLED_SNAPSHOT_HASH_KEY] = staged_sha256
            st.session_state.pop(PENDING_SNAPSHOT_KEY, None)
            st.session_state.pop(PENDING_SNAPSHOT_HASH_KEY, None)
            st.session_state[SNAPSHOT_CONFIRMATION_RESET_KEY] = True
            st.session_state["meia_notice"] = i18n.text("file.snapshot_cancelled")
            st.rerun()
        if apply_snapshot:
            try:
                activated, _normalized_state = _commit_confirmed_snapshot(
                    st.session_state,
                    pending_snapshot,
                    confirmed=confirmed,
                )
            except (PresetError, TypeError, ValueError) as exc:
                st.error(i18n.error_text(exc, "file.snapshot_not_applied"))
            else:
                st.session_state["meia_processed_selection_event_id"] = None
                staged_sha256 = st.session_state.get(PENDING_SNAPSHOT_HASH_KEY)
                if staged_sha256 is not None:
                    st.session_state[HANDLED_SNAPSHOT_HASH_KEY] = staged_sha256
                st.session_state.pop(PENDING_SNAPSHOT_KEY, None)
                st.session_state.pop(PENDING_SNAPSHOT_HASH_KEY, None)
                st.session_state[SNAPSHOT_CONFIRMATION_RESET_KEY] = True
                _clear_buffered_visual_widgets()
                st.session_state["meia_notice"] = (
                    i18n.text("file.snapshot_applied")
                )
                st.rerun()


def _initialize_applied_view(structure_id: str, visual_state) -> AppliedViewState:
    active_view = visual_state.style.view
    control_token = (
        structure_id,
        active_view.rotation,
        active_view.camera.eye,
        active_view.camera.up,
        active_view.camera.center,
    )
    required = {
        "meia_applied_camera",
        "meia_applied_rotation_matrix",
        "meia_processed_viewer_event_id",
        "meia_view_revision",
    }
    if (
        st.session_state.get("meia_view_control_token") != control_token
        or not required.issubset(st.session_state)
    ):
        revision = "state:" + sha256(repr(control_token).encode("utf-8")).hexdigest()[:16]
        applied = AppliedViewState(
            camera=active_view.camera,
            rotation_matrix=camera_to_rotation_matrix(active_view.camera),
            event_id=None,
            view_revision=revision,
        )
        st.session_state["meia_view_control_token"] = control_token
        store_applied_view_state(st.session_state, applied)
        return applied
    return load_applied_view_state(st.session_state)


def _render_global_forms(
    active,
    visual_state,
    default_style,
    i18n: I18n | None = None,
):
    i18n = i18n or I18n(Locale.ZH_CN)
    atoms = active.atoms if active is not None else Atoms()
    with st.sidebar:
        st.divider()
        st.header(i18n.text("sidebar.parameters"))

        with st.expander(i18n.text("sidebar.atoms"), expanded=False):
            submitted_atom_cell = render_atom_cell_form(
                visual_state.style.atom_cell,
                visual_state.style.size_profiles,
                atoms,
                default_style.style.atom_cell.element_colors,
                i18n,
            )
            if submitted_atom_cell is not None:
                visual_state = replace_atom_and_size_profiles(
                    visual_state,
                    submitted_atom_cell.atom_cell,
                    submitted_atom_cell.size_profiles,
                )
                store_visual_state(st.session_state, visual_state)
                # 全局倍率或基础半径会联动所有元素的绝对半径字段；立即重跑，
                # 让 keyed 控件从刚应用的类型化状态重新初始化。
                st.rerun()

        with st.expander(i18n.text("sidebar.bonds"), expanded=False):
            submitted_bonds = render_bond_form(
                visual_state.style.bonds,
                visual_state.style.size_profiles,
                atoms,
                i18n,
            )
            if submitted_bonds is not None:
                visual_state = replace_bonds_and_size_profiles(
                    visual_state,
                    submitted_bonds.bonds,
                    submitted_bonds.size_profiles,
                )
                store_visual_state(st.session_state, visual_state)

        with st.expander(i18n.text("sidebar.periodic"), expanded=False):
            submitted_cell_periodic = render_cell_periodic_form(
                visual_state.style.cell_periodic,
                atoms,
                i18n,
            )
            if submitted_cell_periodic is not None:
                visual_state = replace_cell_periodic(
                    visual_state,
                    submitted_cell_periodic,
                )
                store_visual_state(st.session_state, visual_state)

        if active is not None:
            with st.expander(i18n.text("sidebar.selection"), expanded=False):
                submitted_selection = render_atom_selection_form(
                    visual_state.atom_selection,
                    atoms,
                    matched_bond_pairs(visual_state.style.bonds, atoms),
                    i18n,
                )
                if submitted_selection is not None:
                    visual_state = replace_atom_selection(
                        visual_state,
                        submitted_selection,
                    )
                    store_visual_state(st.session_state, visual_state)

        with st.expander(i18n.text("sidebar.export"), expanded=False):
            submitted_export = render_export_form(visual_state.style.export, i18n)
            if submitted_export is not None:
                visual_state = replace_export(visual_state, submitted_export)
                store_visual_state(st.session_state, visual_state)
            export_name = st.text_input(
                i18n.text("export.name"),
                value=DEFAULT_EXPORT_STEM,
                key="meia_preset_name",
            )
            download_container = st.container()

        reset_clicked = st.button(
            i18n.text("sidebar.reset"),
            disabled=active is None,
            use_container_width=True,
            key="meia_reset_visual_modules",
        )
        if reset_clicked:
            baseline = st.session_state.get(RESET_STYLE_BASELINE_KEY)
            if not isinstance(baseline, PortableStyle):
                baseline = default_style.style
            try:
                reset_state = reset_visual_modules_from_style(
                    visual_state,
                    baseline,
                    atoms,
                )
            except (TypeError, ValueError) as exc:
                st.error(i18n.error_text(exc, "sidebar.reset_failed"))
            else:
                store_visual_state(st.session_state, reset_state)
                _clear_reset_scoped_widgets(st.session_state)
                _advance_atom_selection_draft_revision(st.session_state)
                st.session_state[RESET_WIDGET_REINITIALIZE_KEY] = True
                st.session_state["meia_notice"] = i18n.text("sidebar.reset_done")
                st.rerun()

    return visual_state, export_name, download_container


def _render_export_downloads(
    container,
    preview_artifact,
    current_preview_key,
    active,
    state,
    export_name,
    i18n: I18n | None = None,
) -> None:
    """发布 JSON，并且只在图像与当前状态一致时允许下载。"""
    i18n = i18n or I18n(Locale.ZH_CN)
    with container:
        safe_name = export_name.strip() or DEFAULT_EXPORT_STEM
        try:
            style_preset = build_style_preset(
                state,
                safe_name,
                __version__,
            )
            workspace_snapshot = build_workspace_snapshot(
                active.atoms,
                active.source_name,
                state,
                safe_name,
                __version__,
            )
            style_bytes = style_preset_to_json(style_preset).encode("utf-8")
            workspace_bytes = workspace_snapshot_to_json(
                workspace_snapshot
            ).encode("utf-8")
        except Exception as exc:
            st.error(i18n.error_text(exc, "export.generation_failed"))
            return

        stem = os.path.splitext(active.source_name)[0]
        artifact_is_current = (
            isinstance(preview_artifact, PreviewArtifact)
            and preview_artifact.key == current_preview_key
        )
        if artifact_is_current:
            export_format = preview_artifact.export_format
            st.download_button(
                label=i18n.text(
                    "export.download_image",
                    format=export_format.upper(),
                ),
                data=preview_artifact.export_bytes,
                file_name=f"{stem}_meia.{export_format}",
                mime={
                    "svg": "image/svg+xml",
                    "png": "image/png",
                    "pdf": "application/pdf",
                }.get(export_format, "application/octet-stream"),
            )
        else:
            st.caption(i18n.text("export.image_unavailable"))
        st.download_button(
            label=i18n.text("export.download_style"),
            data=style_bytes,
            file_name=f"{safe_name}{STYLE_JSON_SUFFIX}",
            mime="application/json",
        )
        st.download_button(
            label=i18n.text("export.download_workspace"),
            data=workspace_bytes,
            file_name=f"{safe_name}{WORKSPACE_JSON_SUFFIX}",
            mime="application/json",
        )


def _apply_viewer_event(
    raw_event,
    active,
    visual_state,
    applied_view,
    i18n: I18n | None = None,
) -> None:
    i18n = i18n or I18n(Locale.ZH_CN)
    if not isinstance(raw_event, dict):
        return
    try:
        if raw_event.get("event_type") == "select_atom":
            selection = accept_atom_selection_event(
                raw_event,
                current_structure_id=active.structure_id,
                processed_event_id=st.session_state.get(
                    "meia_processed_selection_event_id"
                ),
                atom_symbols=active.atoms.get_chemical_symbols(),
            )
            if selection is None:
                return
            updated = replace_selected_indices(
                visual_state.atom_selection,
                (selection.atom_index,),
                len(active.atoms),
            )
            visual_state = replace_atom_selection(visual_state, updated)
            store_visual_state(st.session_state, visual_state)
            st.session_state["meia_processed_selection_event_id"] = selection.event_id
            st.session_state["meia_pending_atom_selection_indices"] = (
                updated.selected_atom_indices
            )
            st.rerun()

        if raw_event.get("event_type") == "select_atoms":
            selection = accept_atom_selection_batch_event(
                raw_event,
                current_structure_id=active.structure_id,
                processed_event_id=st.session_state.get(
                    "meia_processed_selection_event_id"
                ),
                atom_count=len(active.atoms),
            )
            if selection is None:
                return
            updated = replace_selected_indices(
                visual_state.atom_selection,
                selection.atom_indices,
                len(active.atoms),
            )
            visual_state = replace_atom_selection(visual_state, updated)
            store_visual_state(st.session_state, visual_state)
            st.session_state["meia_processed_selection_event_id"] = selection.event_id
            st.session_state["meia_pending_atom_selection_indices"] = (
                updated.selected_atom_indices
            )
            st.rerun()

        accepted_camera = accept_apply_camera_event(
            raw_event,
            current_structure_id=active.structure_id,
            processed_event_id=applied_view.event_id,
        )
        if accepted_camera is None:
            return
        updated_view = update_applied_view(applied_view, accepted_camera)
        store_applied_view_state(st.session_state, updated_view)
        visual_state = apply_camera_only(visual_state, accepted_camera.camera)
        store_visual_state(st.session_state, visual_state)
        st.session_state["meia_view_control_token"] = (
            active.structure_id,
            visual_state.style.view.rotation,
            visual_state.style.view.camera.eye,
            visual_state.style.view.camera.up,
            visual_state.style.view.camera.center,
        )
        st.rerun()
    except ViewerEventError as exc:
        st.error(i18n.error_text(exc, "viewer.event_failed"))


def _render_main_title(i18n: I18n) -> None:
    if i18n.locale is Locale.ZH_CN:
        st.title(i18n.text("app.title.zh"))
        return
    st.markdown(
        f'<h1><strong>{PRODUCT_NAME}</strong> '
        '<span style="font-size:0.55em;font-weight:500">'
        f'- {PRODUCT_FULL_NAME}'
        '</span></h1>',
        unsafe_allow_html=True,
    )


def main() -> None:
    page_locale = load_locale(st.session_state)
    page_title = (
        I18n(page_locale).text("app.page_title")
        if page_locale is not None
        else "MEIA"
    )
    st.set_page_config(page_title=page_title, page_icon="⚛", layout="wide")
    i18n = _initialize_i18n()
    i18n = _render_locale_selector(i18n)
    _consume_reset_widget_reinitialize(st.session_state)
    pending_selection = st.session_state.pop(
        "meia_pending_atom_selection_indices", None
    )
    if pending_selection is not None:
        revision = st.session_state.get(ATOM_SELECTION_DRAFT_REVISION_KEY, 0)
        if not isinstance(revision, int) or revision < 0:
            revision = 0
        selection_widget_key = atom_selection_draft_widget_key(
            "meia_atom_selection_indices",
            revision,
        )
        st.session_state[selection_widget_key] = list(pending_selection)

    _render_main_title(i18n)
    st.caption(i18n.text("app.subtitle"))
    st.caption(i18n.text("app.author"))
    notice = st.session_state.pop("meia_notice", None)
    if notice:
        st.success(notice)

    try:
        default_style = load_default_style()
    except PresetError as exc:
        st.error(i18n.error_text(exc, "file.default_style_failed"))
        return

    file_module = st.sidebar.expander(
        i18n.text("file.module"),
        expanded=st.session_state.get(ACTIVE_WORKSPACE_KEY) is None,
    )
    with file_module:
        structure_upload = st.file_uploader(
            i18n.text("file.upload_structure"),
            help=i18n.text("file.upload_structure_help"),
            key="meia_structure_upload",
        )
        style_upload = st.file_uploader(
            i18n.text("file.import_style"),
            type=["json"],
            key="meia_style_upload",
        )
        st.caption(i18n.text("file.default_style_path_hint"))
        snapshot_upload = st.file_uploader(
            i18n.text("file.import_snapshot"),
            type=["json"],
            key="meia_snapshot_upload",
        )

    try:
        active, replaced_structure = _active_workspace_from_upload(structure_upload)
    except Exception as exc:
        st.error(i18n.error_text(exc, "file.structure_read_failed"))
        return

    visual_state = _reset_visual_state_for_structure(
        active,
        default_style,
        force=replaced_structure,
    )
    _render_json_imports(
        active,
        visual_state,
        style_upload,
        snapshot_upload,
        file_module,
        i18n,
    )
    visual_state, export_name, download_container = _render_global_forms(
        active,
        visual_state,
        default_style,
        i18n,
    )

    if active is None:
        with st.sidebar:
            st.caption(i18n.text("file.empty_sidebar_help"))
        st.info(i18n.text("file.start_prompt"))
        return

    atoms = active.atoms
    st.success(
        i18n.text(
            "structure.current",
            source=active.source_name,
            count=len(atoms),
        )
    )
    render_context = resolve_render_context(atoms, visual_state)
    diagnostic_notice = _periodic_diagnostic_notice(render_context, i18n)
    if diagnostic_notice is not None:
        st.warning(diagnostic_notice)
    if not any(
        instance.source_atom_index not in render_context.hidden_atom_indices
        for instance in render_context.periodic_display.atom_instances
    ):
        st.caption(i18n.text("structure.no_visible_atoms"))
    applied_view = _initialize_applied_view(active.structure_id, visual_state)
    component_revision = f"{active.structure_id}:{applied_view.view_revision}"
    selected_indices = visual_state.atom_selection.selected_atom_indices
    output_config = replace(
        render_context.config,
        rotation_matrix=applied_view.rotation_matrix,
    )
    output_context = replace(render_context, config=output_config)
    complexity = measure_display_complexity(len(atoms), output_context)
    st.subheader(i18n.text("viewer.title"))
    st.caption(i18n.text("viewer.interaction_caption"))
    figure_3d = create_3d_figure(
        atoms,
        output_config,
        camera=applied_view.camera,
        uirevision=component_revision,
        selected_atom_indices=selected_indices,
        render_context=output_context,
        figure_messages=i18n.bundle("figure3d"),
    )
    axis_cameras = {
        axis: camera_for_lattice_axis(atoms.cell.array, axis)
        for axis in ("a", "b", "c")
    }
    raw_event = atom_viewer(
        figure=figure_3d,
        structure_id=active.structure_id,
        view_revision=component_revision,
        applied_camera=applied_view.camera,
        locale=i18n.locale,
        messages=i18n.bundle("viewer"),
        axis_cameras=axis_cameras,
        selected_atom_index=None,
        selected_atom_indices=selected_indices,
        batch_selection_enabled=True,
        extreme_3d_interaction=complexity.extreme_3d_interaction,
        style_dirty=False,
        key="meia_3d_viewer",
    )
    _apply_viewer_event(raw_event, active, visual_state, applied_view, i18n)

    st.subheader(i18n.text("preview.title"))
    current_preview_key = PreviewKey.build(
        active.structure_id,
        visual_state,
        applied_view.rotation_matrix,
    )
    preview_artifact = st.session_state.get(PREVIEW_ARTIFACT_KEY)
    if (
        not isinstance(preview_artifact, PreviewArtifact)
        or preview_artifact.key.structure_id != active.structure_id
    ):
        preview_artifact = None
        st.session_state.pop(PREVIEW_ARTIFACT_KEY, None)
    status = preview_status(preview_artifact, current_preview_key)

    refresh_requested = False
    if complexity.manual_2d_recommended:
        st.caption(
            i18n.text(
                "preview.complexity",
                source=complexity.source_atom_count,
                instances=complexity.atom_instance_count,
                artists=complexity.estimated_2d_artist_count,
            )
        )
        refresh_requested = st.button(
            i18n.text("preview.refresh"),
            key="meia_refresh_2d_preview",
            use_container_width=True,
        )

    if should_render_preview(
        complexity,
        status,
        refresh_requested=refresh_requested,
    ):
        figure_2d = None
        try:
            with st.spinner(i18n.text("preview.rendering")):
                figure_2d = render_2d(
                    atoms,
                    output_config,
                    render_context=output_context,
                )
                preview_bytes = render_preview_png(
                    figure_2d,
                    transparent=output_config.transparent,
                )
                export_format = visual_state.style.export.format
                export_bytes = export_figure(
                    figure_2d,
                    export_format,
                    output_config,
                )
                if not preview_bytes or not export_bytes:
                    raise ValueError("preview or export image data was not generated")
                preview_artifact = PreviewArtifact(
                    key=current_preview_key,
                    preview_png=preview_bytes,
                    export_format=export_format,
                    export_bytes=export_bytes,
                )
                st.session_state[PREVIEW_ARTIFACT_KEY] = preview_artifact
                status = PreviewStatus.CURRENT
        except Exception as exc:
            st.error(i18n.error_text(exc, "export.generation_failed"))
        finally:
            if figure_2d is not None:
                plt.close(figure_2d)

    if status is PreviewStatus.MISSING:
        st.caption(i18n.text("preview.missing"))
    elif status is PreviewStatus.STALE:
        st.caption(i18n.text("preview.stale"))
    else:
        st.caption(i18n.text("preview.current"))

    if isinstance(preview_artifact, PreviewArtifact):
        st.markdown(
            preview_image_html(
                preview_artifact.preview_png,
                alt_text=i18n.text("preview.alt"),
            ),
            unsafe_allow_html=True,
        )
        st.caption(
            i18n.text(
                "preview.pixel_density",
                width=PREVIEW_CSS_WIDTH,
                height=PREVIEW_CSS_HEIGHT,
            )
        )

    _render_export_downloads(
        download_container,
        preview_artifact,
        current_preview_key,
        active,
        visual_state,
        export_name,
        i18n,
    )


if __name__ == "__main__":
    main()
