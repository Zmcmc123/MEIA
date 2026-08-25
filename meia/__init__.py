"""MEIA — Molecular and Extended-system Illustration Assistant。"""

from .brand import MEIA_VERSION
from .i18n import I18n, Locale, LocalizedError


__version__ = MEIA_VERSION

from .config import RenderConfig
from .display_complexity import (
    EXTREME_3D_ATOM_THRESHOLD,
    LARGE_3D_ATOM_THRESHOLD,
    MANUAL_2D_ARTIST_THRESHOLD,
    DisplayComplexity,
    measure_display_complexity,
)
from .projection import project_atoms, ProjectionResult
from .bonds import find_bonds, Bond
from .bond_rules import (
    AtomBondOverride,
    BondPairRule,
    BondOverrideConflict,
    BondResolution,
    BondRuleError,
    BondSettings,
    BondStrokeStyle,
    BondStyle,
    OverrideVisibility,
    ResolvedBond,
    count_drawable_bonds,
    find_override_conflicts,
    initialize_bond_settings,
    normalize_element_pair,
    resolve_bonds,
    validate_bond_settings,
)
from .bond_state import (
    AppliedBondState,
    apply_bond_draft,
    initialize_bond_state,
    load_bond_state,
    reset_bond_state_for_structure,
    set_bond_draft,
    store_bond_state,
)
from .bond_segments import BondSurfaceSegment, clip_bond_to_spheres
from .hydrogen_bonds import (
    HYDROGEN_BOND_MAX_DISTANCE,
    HYDROGEN_BOND_MIN_ANGLE,
    HydrogenBond,
    HydrogenBondGeometry,
    HydrogenBondSettings,
    compute_hydrogen_bond_geometries,
    resolve_hydrogen_bonds,
)
from .atom_styles import (
    AtomHydrogenBondOverride,
    AtomColorOverride,
    AtomColorStrength,
    AtomSelectionOperation,
    AtomSelectionSettings,
    HiddenAtom,
    apply_atom_selection_operation,
    atom_color_override_mapping,
    compact_color_strengths,
    color_strength_mapping,
    replace_selected_indices,
    resolved_color_strengths,
    validate_atom_selection_settings,
)
from .periodic_display import (
    MAX_PERIODIC_ATOM_INSTANCES,
    CellPeriodicSettings,
    LatticeShift,
    PeriodicRange,
    estimate_periodic_atom_instances,
    normalize_periodic_settings,
)
from .selection_paging import (
    ATOM_SELECTION_PAGE_SIZE,
    LARGE_SELECTION_THRESHOLD,
    AtomSelectionPage,
    apply_page_selection,
    selection_page,
)
from .geometry import compute_bond_geometries, BondGeometry
from .renderer import render
from .export import (
    SVGGroupingError,
    export_figure,
    export_pdf,
    export_png,
    export_svg,
    postprocess_meia_svg,
)
from .pipeline import (
    OutputCollisionError,
    plan_output_paths,
    render_atoms,
    render_file,
    render_batch,
)
from .view import camera_to_rotation_matrix, render_2d
from .view_state import (
    AcceptedCameraEvent,
    AppliedViewState,
    ApplyCameraEvent,
    AtomSelectionBatchEvent,
    AtomSelectionEvent,
    CameraState,
    CameraValidationError,
    ViewerEventError,
    accept_apply_camera_event,
    accept_atom_selection_batch_event,
    accept_atom_selection_event,
    camera_for_lattice_axis,
    initial_applied_view,
    load_applied_view_state,
    parse_apply_camera_event,
    parse_atom_selection_batch_event,
    parse_atom_selection_event,
    rotation_matrix_to_camera,
    structure_id_from_bytes,
    store_applied_view_state,
    update_applied_view,
)
from .preview import (
    PREVIEW_CSS_HEIGHT,
    PREVIEW_CSS_WIDTH,
    PREVIEW_PIXEL_HEIGHT,
    PREVIEW_PIXEL_WIDTH,
    preview_image_html,
    render_preview_png,
)
from .visual_state import (
    AtomCellSettings as AppliedAtomCellSettings,
    BondModuleSettings,
    ExportSettings as AppliedExportSettings,
    PairRuleDefaults,
    PortableStyle,
    RenderContext,
    ViewSettings,
    VisualizationState,
    apply_camera_only,
    apply_portable_style,
    merge_pair_rules_for_structure,
    merge_portable_style_for_structure,
    replace_atom_and_size_profiles,
    replace_atom_cell,
    replace_atom_selection,
    replace_bonds,
    replace_bonds_and_size_profiles,
    replace_cell_periodic,
    replace_export,
    replace_view,
    resolve_render_context,
)
from .render_topology import (
    RenderTopology,
    TopologyCacheEntry,
    TopologyKey,
    build_render_topology,
    compose_render_context,
    topology_key,
)
from .size_profiles import (
    CovalentSizeProfile,
    RadiusMode,
    SizeProfileSettings,
    UniformSizeProfile,
    apply_size_profile_edits,
    replace_active_bond_width,
    resolve_active_bond_width,
    resolve_display_radii,
)
from .presets import (
    PresetError,
    SCHEMA_VERSION,
    PresetKind,
    PresetMetadata,
    SnapshotStructure,
    StylePreset,
    WorkspaceSnapshot,
    apply_style_preset,
    apply_workspace_snapshot,
    load_default_style,
    parse_preset,
    style_preset_to_json,
    workspace_snapshot_to_json,
    visual_state_fingerprint,
)
from .preview_state import (
    PreviewArtifact,
    PreviewKey,
    PreviewStatus,
    preview_status,
    should_render_preview,
)
from .viewer import atom_viewer, create_3d_figure
from .workspace import (
    ActiveWorkspace,
    PendingSnapshot,
    activate_snapshot,
    activate_upload,
    build_style_preset,
    build_workspace_snapshot,
    canonical_structure_bytes,
    confirm_pending_snapshot,
    stage_snapshot,
    structure_identity,
)
from .batch import batch_process
from .io import read_structure
