# MEIA Large-System Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep large periodic slab/water structures responsive during 3D navigation and atom styling while preserving full scientific state and full-fidelity 2D/export output.

**Architecture:** Add a deterministic display-complexity policy and a session-local 2D artifact lifecycle so expensive Matplotlib work leaves the live 3D path. Represent dense background color strength compactly in memory, batch Plotly zoom updates, use sparse selection overlays, page large atom selectors, and reuse structure topology across style-only changes.

**Tech Stack:** Python 3.10, ASE, NumPy, Matplotlib, Plotly 5.24.1, Streamlit 1.38.0, JavaScript ES modules, Node.js test runner, Vite.

**Spec:** `docs/SPEC.large-system-optimization.md`

## Global Constraints

- Work only on `codex/large-system-optimization`; do not merge `main`.
- Keep MEIA version `0.11.0` and strict schema v7 throughout this branch.
- Do not add runtime dependencies or force-upgrade the frontend lock file.
- Do not modify input `ase.Atoms`; all render and benchmark operations use copies or read-only access.
- Do not downsample, truncate, or silently hide data in final 2D/SVG/PNG/PDF output.
- Interactive layer simplification is ephemeral, applies only at 20,000 or more displayed atom instances, and restores automatically after input becomes idle.
- Preserve `zh-CN` and `en` catalog parity and keep language outside visualization JSON.
- Use a writable temporary `MPLCONFIGDIR` for every Matplotlib verification.
- Every production behavior follows a witnessed RED → GREEN cycle.

---

### Task 1: Display-complexity policy

**Files:**
- Create: `meia/display_complexity.py`
- Create: `tests/test_display_complexity.py`
- Modify: `meia/__init__.py`

**Interfaces:**
- Consumes: `meia.visual_state.RenderContext` and source atom count.
- Produces: `DisplayComplexity.from_counts(...)`, `measure_display_complexity(...)`, `MANUAL_2D_ARTIST_THRESHOLD`, `LARGE_3D_ATOM_THRESHOLD`, and `EXTREME_3D_ATOM_THRESHOLD`.

- [ ] **Step 1: Write threshold tests before the module exists**

```python
from meia.display_complexity import DisplayComplexity


def test_display_complexity_uses_render_artist_and_instance_thresholds():
    ordinary = DisplayComplexity.from_counts(900, 900, 500, 200)
    assert ordinary.estimated_2d_artist_count == 4100
    assert ordinary.manual_2d_recommended is False
    assert ordinary.large_3d_interaction is False

    large_2d = DisplayComplexity.from_counts(900, 900, 650, 200)
    assert large_2d.estimated_2d_artist_count == 5000
    assert large_2d.manual_2d_recommended is True

    large_3d = DisplayComplexity.from_counts(1000, 5000, 0, 0)
    assert large_3d.large_3d_interaction is True
    assert large_3d.extreme_3d_interaction is False

    extreme = DisplayComplexity.from_counts(1000, 20000, 0, 0)
    assert extreme.extreme_3d_interaction is True
```

- [ ] **Step 2: Run the new tests and verify RED**

Run: `python -m pytest tests/test_display_complexity.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'meia.display_complexity'`.

- [ ] **Step 3: Implement the immutable complexity model**

```python
MANUAL_2D_ARTIST_THRESHOLD = 5_000
LARGE_3D_ATOM_THRESHOLD = 5_000
EXTREME_3D_ATOM_THRESHOLD = 20_000


@dataclass(frozen=True)
class DisplayComplexity:
    source_atom_count: int
    atom_instance_count: int
    visible_bond_instance_count: int
    hydrogen_bond_instance_count: int

    @property
    def estimated_2d_artist_count(self) -> int:
        return (
            self.atom_instance_count
            + 6 * self.visible_bond_instance_count
            + self.hydrogen_bond_instance_count
        )

    @property
    def manual_2d_recommended(self) -> bool:
        return self.estimated_2d_artist_count >= MANUAL_2D_ARTIST_THRESHOLD
```

`measure_display_complexity` must count only visible atom instances and visible bond instances after hidden-atom filtering. Validate every count as a non-negative integer so a malformed context fails explicitly.

- [ ] **Step 4: Add a real RenderContext integration test**

Build an `Atoms("HOH")` context with `resolve_render_context`, call `measure_display_complexity`, and assert literal source/instance counts. The test must mutate neither positions nor cell.

- [ ] **Step 5: Run focused tests and package compilation**

Run:

```bash
python -m pytest tests/test_display_complexity.py tests/test_visual_state.py -q
python -m compileall -q meia tests
```

Expected: all selected tests pass and both commands exit 0.

- [ ] **Step 6: Commit Task 1**

```bash
git add meia/display_complexity.py meia/__init__.py tests/test_display_complexity.py
git commit -m "feat: classify large display workloads"
```

---

### Task 2: Session-local 2D preview artifacts and explicit large-system refresh

**Files:**
- Create: `meia/preview_state.py`
- Create: `tests/test_preview_state.py`
- Modify: `meia/presets.py`
- Modify: `meia/preview.py`
- Modify: `app.py`
- Modify: `tests/test_app_and_batch.py`
- Modify: `meia/locales/zh-CN.json`
- Modify: `meia/locales/en.json`

**Interfaces:**
- Consumes: `DisplayComplexity`, `VisualizationState`, structure ID, applied rotation matrix, existing `render_2d`, `render_preview_png`, and `export_figure`.
- Produces: `visual_state_fingerprint(state) -> str`, `PreviewKey`, `PreviewArtifact`, `preview_status(...)`, and `should_render_preview(...)`.

- [ ] **Step 1: Write fingerprint and cache-state tests**

```python
def test_preview_key_changes_for_style_or_camera_but_not_object_identity():
    state = VisualizationState()
    same_value = VisualizationState()
    camera = np.eye(3)
    assert PreviewKey.build("structure-a", state, camera) == PreviewKey.build(
        "structure-a", same_value, camera.copy()
    )
    changed = replace_atom_selection(
        state,
        AtomSelectionSettings(
            color_strengths=(AtomColorStrength(0, "H", 0.3),)
        ),
    )
    assert PreviewKey.build("structure-a", changed, camera) != PreviewKey.build(
        "structure-a", state, camera
    )


def test_preview_cache_distinguishes_missing_current_and_stale():
    current = PreviewKey("structure-a", "state-a", "camera-a")
    old = PreviewKey("structure-a", "state-old", "camera-a")
    artifact = PreviewArtifact(old, b"png", "svg", b"svg")
    assert preview_status(None, current) is PreviewStatus.MISSING
    assert preview_status(artifact, current) is PreviewStatus.STALE
    assert preview_status(replace(artifact, key=current), current) is PreviewStatus.CURRENT
```

The production change caught is accidental reuse of a preview after a style, structure, camera, format, DPI, or transparency change.

- [ ] **Step 2: Run preview-state tests and verify RED**

Run: `python -m pytest tests/test_preview_state.py -q`

Expected: import fails because `meia.preview_state` and `visual_state_fingerprint` do not exist.

- [ ] **Step 3: Expose deterministic visualization-state encoding**

Refactor the existing private preset mappings without changing JSON output:

```python
def visual_state_fingerprint(state: VisualizationState) -> str:
    payload = {
        "style": _style_sections(state.style),
        "atom_selection": _atom_selection_mapping(state.atom_selection),
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(canonical).hexdigest()
```

Add a regression assertion that `style_preset_to_json` and `workspace_snapshot_to_json` remain byte-for-byte unchanged for an existing fixed fixture.

- [ ] **Step 4: Implement preview lifecycle types and policy**

```python
class PreviewStatus(str, Enum):
    MISSING = "missing"
    CURRENT = "current"
    STALE = "stale"


@dataclass(frozen=True)
class PreviewKey:
    structure_id: str
    visual_state_sha256: str
    camera_sha256: str


@dataclass(frozen=True)
class PreviewArtifact:
    key: PreviewKey
    preview_png: bytes
    export_format: str
    export_bytes: bytes
```

`should_render_preview` returns true for every missing/stale small-system preview and only for an explicit refresh click in manual mode. It must never return true solely because a stale large-system artifact exists.

- [ ] **Step 5: Write AppTest behavior for a large system before changing `app.py`**

Patch `measure_display_complexity` to return a manual-2D workload and count `render_2d` calls. Assert:

```python
app_module.main()
assert captured["render_2d_calls"] == 0
assert "生成 2D 预览" in app_module.st.buttons
assert any("尚未生成" in text for text in app_module.st.captions)
```

Add a second test where the refresh button returns true and assert exactly one `render_2d` call, one `render_preview_png` call, and one `plt.close` path.

- [ ] **Step 6: Run the AppTest cases and verify RED**

Run:

```bash
python -m pytest tests/test_app_and_batch.py::test_large_system_skips_automatic_2d_preview tests/test_app_and_batch.py::test_large_system_refresh_generates_one_current_artifact -q
```

Expected: the first test observes one current unconditional `render_2d` call; the second cannot find the explicit refresh control.

- [ ] **Step 7: Refactor the page rendering path**

After `create_3d_figure`, compute `DisplayComplexity` and `PreviewKey`. For manual workloads:

1. show source atom count, display instance count, and artist estimate;
2. render a bilingual refresh button;
3. show current/stale artifact status;
4. call the full renderer only when `should_render_preview` is true;
5. cache only bytes and key in `st.session_state`;
6. close every newly created Figure in a `finally` block.

Keep automatic behavior below the threshold. `_render_export_downloads` must always provide style and workspace JSON; it provides the image download only when the artifact key matches the current key. A stale image must not be downloadable under the current filename.

- [ ] **Step 8: Add both locale catalogs and catalog tests**

Add matched keys for complexity summary, missing preview, stale preview, current preview, refresh button, rendering spinner, and unavailable image export. Run:

```bash
python -m pytest tests/test_i18n.py tests/test_preview.py tests/test_preview_state.py tests/test_app_and_batch.py -q
```

- [ ] **Step 9: Commit Task 2**

```bash
git add app.py meia/presets.py meia/preview.py meia/preview_state.py meia/locales/zh-CN.json meia/locales/en.json tests/test_preview_state.py tests/test_app_and_batch.py tests/test_i18n.py
git commit -m "feat: defer expensive large-system previews"
```

---

### Task 3: Compact runtime color-strength profile and subject emphasis

**Files:**
- Modify: `meia/atom_styles.py`
- Modify: `meia/config.py`
- Modify: `meia/visual_state.py`
- Modify: `meia/presets.py`
- Modify: `meia/sidebar.py`
- Modify: `meia/viewer.py`
- Modify: `tests/test_atom_styles.py`
- Modify: `tests/test_presets.py`
- Modify: `tests/test_sidebar.py`
- Modify: `tests/test_visual_state.py`
- Modify: `tests/test_app_and_batch.py`
- Modify: `meia/locales/zh-CN.json`
- Modify: `meia/locales/en.json`

**Interfaces:**
- Consumes: existing `AtomColorStrength`, `AtomSelectionSettings`, v7 `color_strengths` JSON, and current selected source indices.
- Produces: `default_color_strength`, compact exceptions, `resolved_color_strengths(...)`, `compact_color_strengths(...)`, and `emphasize_subject(...)`.

- [ ] **Step 1: Write compact profile tests**

```python
def test_subject_emphasis_keeps_subject_selection_and_compacts_background():
    atoms = Atoms(symbols=["H"] * 5000)
    current = AtomSelectionSettings(selected_atom_indices=(3, 7))
    updated = emphasize_subject(atoms, current, 0.30)
    assert updated.selected_atom_indices == (3, 7)
    assert updated.default_color_strength == pytest.approx(0.30)
    assert [(item.atom_index, item.strength) for item in updated.color_strengths] == [
        (3, 1.0),
        (7, 1.0),
    ]
    values = resolved_color_strengths(updated, len(atoms))
    assert values[3] == pytest.approx(1.0)
    assert values[7] == pytest.approx(1.0)
    assert values[4999] == pytest.approx(0.30)
```

Add a test showing that an ordinary strength operation removes an exception when the selected value equals the profile default, and retains a `1.0` exception when the default is below 1.0.

- [ ] **Step 2: Run the new tests and verify RED**

Run: `python -m pytest tests/test_atom_styles.py -q`

Expected: `AtomSelectionSettings` rejects `default_color_strength` or `emphasize_subject` is missing.

- [ ] **Step 3: Implement compact in-memory semantics**

Append `default_color_strength: float = 1.0` to `AtomSelectionSettings`. Change canonicalization to remove records equal to that default, not records equal to `1.0` unconditionally.

```python
def resolved_color_strengths(settings, atom_count):
    values = np.full(atom_count, settings.default_color_strength, dtype=float)
    for item in settings.color_strengths:
        values[item.atom_index] = item.strength
    return values


def emphasize_subject(atoms, settings, background_strength):
    value = normalize_color_strength(background_strength)
    symbols = atoms.get_chemical_symbols()
    subject = settings.selected_atom_indices
    return replace(
        settings,
        default_color_strength=value,
        color_strengths=tuple(
            AtomColorStrength(index, symbols[index], 1.0)
            for index in subject
            if value != 1.0
        ),
    )
```

Preserve color overrides, bond overrides, hidden atoms, and hydrogen-bond overrides.

- [ ] **Step 4: Keep schema v7 byte semantics**

Change `_atom_selection_mapping` to receive the snapshot symbols and expand resolved strengths into the existing list of non-1.0 records. On workspace parse, call `compact_color_strengths` after atom identities are known. Use the most frequent resolved value as the runtime default, breaking ties in favor of `1.0`, and store only values different from that default.

Keep `visual_state_fingerprint` independent of the expanded v7 list: its internal runtime mapping includes `default_color_strength` and the compact exceptions. Add a test proving that compact and expanded states with the same resolved strengths produce the same fingerprint after normalization.

Write a v7 round-trip test with 5000 background atoms and two subjects. Assert:

- serialized root still has exactly the existing atom-selection keys;
- `schema_version == 7`;
- JSON expands 4998 background records;
- parsed runtime state has two exceptions and the same 5000 resolved values.

- [ ] **Step 5: Wire the profile through RenderConfig and 2D/3D**

Add `atom_default_color_strength` to `RenderConfig`. `get_atom_color_strengths` fills the NumPy array with that default before applying exceptions. `resolve_render_context` passes both fields.

In `create_3d_figure`, replace scalar `marker.line.color` with `config.get_atom_outline_colors(...)` indexed by source atom identity so 30% background outlines blend toward white while 100% subjects remain black. Keep line width exactly `1.0`.

- [ ] **Step 6: Add the subject-emphasis sidebar operation**

Inside the existing atom selection form, add an explicitly enabled operation with a 0–100% background strength slider. It applies to the current subject selection without inverting it. Empty subject selection must show a localized error and return no partial state.

Write `FakeStreamlit` tests asserting that a 5000-atom form returns two selected subject indices, default 0.3, and two exceptions rather than 4998 records.

- [ ] **Step 7: Run strength, preset, rendering, and sidebar tests**

Run:

```bash
python -m pytest tests/test_atom_styles.py tests/test_presets.py tests/test_sidebar.py tests/test_visual_state.py tests/test_app_and_batch.py -q
python -m compileall -q app.py meia tests
```

- [ ] **Step 8: Commit Task 3**

```bash
git add meia/atom_styles.py meia/config.py meia/visual_state.py meia/presets.py meia/sidebar.py meia/viewer.py meia/locales/zh-CN.json meia/locales/en.json tests/test_atom_styles.py tests/test_presets.py tests/test_sidebar.py tests/test_visual_state.py tests/test_app_and_batch.py
git commit -m "feat: add compact subject emphasis"
```

---

### Task 4: Paged atom selection for large structures

**Files:**
- Create: `meia/selection_paging.py`
- Create: `tests/test_selection_paging.py`
- Modify: `meia/sidebar.py`
- Modify: `tests/test_sidebar.py`
- Modify: `meia/locales/zh-CN.json`
- Modify: `meia/locales/en.json`

**Interfaces:**
- Consumes: atom count, current selected source indices, one-based page number, page selections, and action code.
- Produces: `AtomSelectionPage`, `selection_page(...)`, and `apply_page_selection(...)`.

- [ ] **Step 1: Write paging behavior tests**

```python
def test_page_selection_adds_and_removes_without_touching_other_pages():
    page = selection_page(atom_count=2500, page_number=2, page_size=200)
    assert page.indices[0] == 200
    assert page.indices[-1] == 399
    assert page.page_count == 13
    assert apply_page_selection((5, 205, 900), (210, 211), "add") == (
        5, 205, 210, 211, 900
    )
    assert apply_page_selection((5, 205, 210, 900), (205, 210), "remove") == (
        5, 900
    )
```

Add validation tests for page 0, page 14, page size 0, and an index outside the current page.

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_selection_paging.py -q`

Expected: module import fails.

- [ ] **Step 3: Implement the pure page model**

Use `LARGE_SELECTION_THRESHOLD = 1_000` and `ATOM_SELECTION_PAGE_SIZE = 200`. Return frozen tuples and validate all public arguments before set operations.

- [ ] **Step 4: Write sidebar tests before changing the form**

For 999 atoms assert the full searchable multiselect still contains 999 options. For 1000 atoms assert no widget contains more than 200 atom options, and assert the selected summary count is present. Submit page action `add`, an index expression, and an element selection together; assert deterministic union semantics.

- [ ] **Step 5: Verify the sidebar tests fail for the current all-atom widget**

Run the new test names in `tests/test_sidebar.py`. Expected: the 1000-atom widget records 1000 options.

- [ ] **Step 6: Integrate paging without changing small-system behavior**

Below 1000 atoms retain the existing multiselect. At and above 1000:

- show page number and at most 200 formatted atom choices;
- show `add` and `remove` actions;
- preserve selection outside the active page;
- combine the page result with the existing index, element, invert, and clear operations in that order;
- keep clear as the highest-precedence operation;
- show a localized selection count and a summary of at most 20 source atom labels.

- [ ] **Step 7: Run focused sidebar and localization checks**

Run:

```bash
python -m pytest tests/test_selection_paging.py tests/test_sidebar.py tests/test_i18n.py -q
python -m compileall -q meia tests
```

- [ ] **Step 8: Commit Task 4**

```bash
git add meia/selection_paging.py meia/sidebar.py meia/locales/zh-CN.json meia/locales/en.json tests/test_selection_paging.py tests/test_sidebar.py tests/test_i18n.py
git commit -m "feat: page atom selection for large structures"
```

---

### Task 5: Batched 3D zoom updates, sparse selection overlay, and interaction LOD

**Files:**
- Create: `meia/components/atom_viewer/frontend/src/selection-spatial-index.mjs`
- Create: `meia/components/atom_viewer/frontend/src/selection-spatial-index.test.mjs`
- Create: `meia/components/atom_viewer/frontend/src/idle-scheduler.mjs`
- Create: `meia/components/atom_viewer/frontend/src/idle-scheduler.test.mjs`
- Modify: `meia/viewer.py`
- Modify: `meia/components/atom_viewer/__init__.py`
- Modify: `meia/components/atom_viewer/frontend/src/viewer-style.mjs`
- Modify: `meia/components/atom_viewer/frontend/src/viewer-style.test.mjs`
- Modify: `meia/components/atom_viewer/frontend/src/selection-state.mjs`
- Modify: `meia/components/atom_viewer/frontend/src/selection-state.test.mjs`
- Modify: `meia/components/atom_viewer/frontend/src/main.mjs`
- Modify: `meia/components/atom_viewer/frontend/src/frame-scheduler.test.mjs`
- Modify: `tests/test_app_and_batch.py`

**Interfaces:**
- Consumes: Plotly trace metadata, atom coordinates/customdata, draft source indices, zoom scale, camera/aspect ratio, and `extreme_3d_interaction` from Python.
- Produces: `plotlyCombinedTraceUpdate(...)`, `sparseSelectionTraceUpdate(...)`, `SelectionSpatialIndex`, and `createIdleTask(...)`.

- [ ] **Step 1: Write a combined-update test**

```javascript
test("one zoom update carries every trace property in one Plotly call", () => {
  const result = plotlyCombinedTraceUpdate([
    {meta: {meia_role: "atoms", meia_base_marker_sizes: [6, 8]}},
    {meta: {meia_role: "bond_outlines", meia_base_line_width: 5}},
    {meta: {meia_role: "bonds", meia_base_line_width: 4}},
    {meta: {meia_role: "hydrogen_bonds", meia_base_line_width: 3}},
  ], 1.5, camera, {x: 2, y: 2, z: 2})
  assert.deepEqual(result.traceIndices, [0, 1, 2, 3])
  assert.deepEqual(result.dataUpdate["marker.size"][0], [9, 12])
  assert.deepEqual(result.dataUpdate["line.width"], [undefined, 7.5, 6, 4.5])
})
```

The mutation caught is reintroducing one awaited `Plotly.update` per trace.

- [ ] **Step 2: Write sparse-selection tests**

Given an atom trace with source indices `[0, 1, 0, 1]` and selected sources `[0]`, assert the update contains exactly two x/y/z/customdata entries and two nonzero marker sizes. Assert clearing selection returns empty arrays, not four transparent markers.

- [ ] **Step 3: Write spatial-index and idle-scheduler tests**

Use literal points spanning several 36 px cells. Assert nearest lookup returns the closest front-most identity, a small rectangle visits only intersecting cells, and a full-window rectangle returns every canonical source index. With a fake timer, schedule three idle tasks and assert only the final callback executes at 80 ms.

- [ ] **Step 4: Run frontend tests and verify RED**

Run: `npm test`

Expected: imports for the three new APIs fail.

- [ ] **Step 5: Implement combined style and sparse-selection builders**

`plotlyCombinedTraceUpdate` must align each property array with `traceIndices` using `undefined` for traces to which the property does not apply. It returns one `layoutUpdate` preserving camera and manual aspect ratio.

`sparseSelectionTraceUpdate` scans the atom trace once, retains every periodic replica whose source index is selected, and emits marker sizes already scaled for the current zoom. Selection color remains `rgba(255,213,79,0.55)` and no fixed-size outer ring is introduced.

- [ ] **Step 6: Implement spatial indexing and trailing idle synchronization**

`SelectionSpatialIndex` stores projected atoms by integer `(floor(x / 36), floor(y / 36))` keys. Click lookup searches only cells touched by the hit radius. Rectangle lookup visits intersecting cells and canonicalizes source indices.

`createIdleTask` clears the previous timer, schedules the newest callback, catches errors through the supplied handler, and exposes `cancel()` for component rerender cleanup.

- [ ] **Step 7: Pass the extreme-workload flag through Python**

Add `extreme_3d_interaction: bool = False` to `render_atom_viewer`. Move complexity measurement before the `atom_viewer` call in `app.py`, pass the measured value, and validate it before forwarding to the component.

- [ ] **Step 8: Integrate a single Plotly update per zoom frame**

Replace the trace loop in `syncViewerTraceStyles` with one `Plotly.update`. The scene aspect ratio changes immediately; trace sizes use the latest-frame scheduler and receive a final 80 ms idle sync.

On local selection changes, update only the sparse selection trace. On camera/zoom changes while selection mode is active, refresh projected positions and rebuild its spatial index.

- [ ] **Step 9: Add extreme interaction LOD**

When `extreme_3d_interaction` is true, pointer/wheel camera input temporarily sets only `bond_outlines` and `hydrogen_bonds` traces to `visible: false`. A 120 ms idle task restores their prior visibility. Component rerender, pointer cancel, and errors must restore or cancel pending state. The Python figure and all exported state remain unchanged.

- [ ] **Step 10: Verify frontend integration and production build**

Run:

```bash
npm test
npm run build
```

Expected: all tests pass, Vite exits 0, and the generated `dist` assets are updated.

- [ ] **Step 11: Run Python seams and commit Task 5**

Run: `python -m pytest tests/test_app_and_batch.py tests/test_visual_state.py -q`

Then commit:

```bash
git add meia/viewer.py meia/components/atom_viewer/__init__.py meia/components/atom_viewer/frontend/src meia/components/atom_viewer/frontend/dist tests/test_app_and_batch.py
git commit -m "perf: batch large viewer interactions"
```

---

### Task 6: Structure-topology reuse across style-only changes

**Files:**
- Create: `meia/render_topology.py`
- Create: `tests/test_render_topology.py`
- Modify: `meia/visual_state.py`
- Modify: `app.py`
- Modify: `tests/test_visual_state.py`
- Modify: `tests/test_app_and_batch.py`

**Interfaces:**
- Consumes: structure ID, bond module settings, PBC settings, atom bond/hydrogen-bond visibility exceptions, hidden atom identities, and hydrogen-bond geometry settings.
- Produces: `RenderTopology`, `TopologyKey`, `build_render_topology(...)`, `compose_render_context(...)`, and `TopologyCacheEntry`.

- [ ] **Step 1: Write topology-key mutation tests**

Create one structure/state and assert the key remains equal after changing only:

- element colors;
- atom color strength profile;
- atom radius and bond width;
- export format/DPI/transparency;
- camera orientation.

Assert the key changes after modifying a pair distance, PBC range, unwrap participation, hidden atom, atom bond override, hydrogen-bond distance, or hydrogen-bond angle.

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_render_topology.py -q`

Expected: import fails because `meia.render_topology` does not exist.

- [ ] **Step 3: Extract topology construction without behavior changes**

Move bond resolution, topology-bond filtering, periodic display construction, and hydrogen-bond candidate resolution into `build_render_topology`. Keep `resolve_render_context(atoms, state)` as a compatibility wrapper:

```python
def resolve_render_context(atoms, state):
    topology = build_render_topology(atoms, state)
    return compose_render_context(atoms, state, topology)
```

`compose_render_context` validates that the topology key matches the current topology-affecting state before reuse. A mismatch raises `ValueError` instead of silently rendering stale bonds.

- [ ] **Step 4: Write a reuse test with observed neighbor-list calls**

Wrap the real topology builder with a counting function in an app integration test. Apply a color-only change and rerun the page; assert the topology build count remains 1 while `RenderContext.config.get_atom_colors` changes. Apply a PBC range change and assert the count becomes 2.

- [ ] **Step 5: Add a one-entry session-local cache**

Store `TopologyCacheEntry(key, topology)` under a MEIA-prefixed session key. Do not use global `st.cache_data` for `Atoms`. Replace the entry atomically only after a successful topology build. Structure replacement and snapshot replacement naturally miss because `structure_id` is part of the key.

- [ ] **Step 6: Run topology, visual-state, PBC, and app tests**

Run:

```bash
python -m pytest tests/test_render_topology.py tests/test_visual_state.py tests/test_periodic_display.py tests/test_app_and_batch.py -q
python -m compileall -q app.py meia tests
```

- [ ] **Step 7: Commit Task 6**

```bash
git add meia/render_topology.py meia/visual_state.py app.py tests/test_render_topology.py tests/test_visual_state.py tests/test_app_and_batch.py
git commit -m "perf: reuse structure render topology"
```

---

### Task 7: Reproducible benchmark, documentation, and final verification

**Files:**
- Create: `scripts/benchmark_large_system.py`
- Create: `tests/test_large_system_benchmark.py`
- Modify: `docs/PROGRESS.md`
- Modify: `docs/SPEC.md`
- Modify: `README.md`
- Modify: `README.en.md`

**Interfaces:**
- Consumes: public MEIA rendering APIs and the new complexity/topology/preview policies.
- Produces: a deterministic CLI benchmark with JSON and human-readable output.

- [ ] **Step 1: Write benchmark CLI behavior tests**

Run the benchmark module with `--nx 2 --water-layers 1 --skip-2d --json` in a subprocess. Assert exit 0 and literal keys:

```python
assert payload["source_atoms"] > 0
assert payload["atom_instances"] == payload["source_atoms"]
assert payload["timings_s"]["topology"] >= 0.0
assert payload["timings_s"]["figure3d"] >= 0.0
assert payload["figure3d_json_bytes"] > 0
assert payload["manual_2d_recommended"] is False
```

Add an invalid `--nx 0` case that exits nonzero with a concise diagnostic.

- [ ] **Step 2: Run the benchmark tests and verify RED**

Run: `python -m pytest tests/test_large_system_benchmark.py -q`

Expected: subprocess fails because the script does not exist.

- [ ] **Step 3: Implement the benchmark CLI**

Generate a square three-layer slab plus deterministic water grids entirely in memory. Support `--nx`, `--water-layers`, `--repeat-a`, `--repeat-b`, `--repeat-c`, `--skip-2d`, and `--json`. Report source/instance/bond/hydrogen counts, topology/context/3D/JSON/2D/PNG timings, JSON/PNG sizes, peak RSS, and complexity flags. Close every Matplotlib Figure.

- [ ] **Step 4: Capture before/after comparable measurements**

Run these exact cases from a fresh process with a fresh writable Matplotlib cache:

```bash
python scripts/benchmark_large_system.py --nx 10 --json
python scripts/benchmark_large_system.py --nx 20 --json
python scripts/benchmark_large_system.py --nx 30 --json
python scripts/benchmark_large_system.py --nx 30 --repeat-a 2 --repeat-b 2 --skip-2d --json
```

Record results in `docs/PROGRESS.md` with machine/dependency context. Do not claim browser FPS from backend timings.

- [ ] **Step 5: Update public behavior documentation**

Document automatic versus explicit 2D preview, stale-preview/export protection, subject emphasis, large-selection paging, visible instance counts, interaction-only layer simplification, and the 50,000-instance hard cap in both README languages and `docs/SPEC.md`. State that final exports always use full data.

- [ ] **Step 6: Run a fresh real-browser workflow**

Start Streamlit in a fresh process and verify with both `examples/CONTCAR` and a benchmark-generated structure:

1. small-case 2D remains automatic;
2. large-case 3D appears before manual 2D generation;
3. selection-off and selection-on trackpad zoom retain camera and scale atoms/bonds together;
4. click and box selection do not jump the camera;
5. confirm selection updates the sidebar without automatic large 2D rendering;
6. subject emphasis fades 3D fills, bonds, hydrogen bonds, and outlines;
7. manual 2D refresh enables a current image download;
8. changing style marks the artifact stale and removes the old image download;
9. SVG and workspace snapshot export and re-import successfully.

Capture console errors and actual interaction timings. Treat missing browser access as an unverified scope, not a pass.

- [ ] **Step 7: Run the full repository checks**

Run:

```bash
python -m compileall -q app.py meia scripts tests
python scripts/check_public_docs.py
python -m pytest -W error::FutureWarning -q
cd meia/components/atom_viewer/frontend
npm ci
npm test
npm run build
```

Read every exit code and test count. Do not proceed to the final commit if any command fails.

- [ ] **Step 8: Inspect scoped changes and commit Task 7**

Run `git diff --check`, inspect `git status --short`, and stage only benchmark/documentation/generated frontend assets that belong to this feature.

```bash
git add scripts/benchmark_large_system.py tests/test_large_system_benchmark.py docs/PROGRESS.md docs/SPEC.md README.md README.en.md
git commit -m "docs: validate large-system workflow"
```

- [ ] **Step 9: Keep the branch isolated for owner testing**

Report the branch name, worktree path, commit list, benchmark table, browser evidence, tests, remaining risks, and npm audit warning. Do not merge, rebase, force-push, or delete the worktree. Push the feature branch only after the project owner explicitly requests remote publication.

## Plan self-review result

- Every requirement in `docs/SPEC.large-system-optimization.md` is assigned to Tasks 1–7 except Matplotlib collection batching.
- Collection batching remains outside this implementation because exact depth ordering and SVG element grouping are release requirements; Task 7 establishes the post-optimization benchmark needed to decide whether a separate renderer plan is justified.
- All schema-facing work explicitly exports and imports schema v7; no v8 field is introduced.
- Function and type names are consistent across producing and consuming tasks.
- All behavior-changing tasks include a witnessed failing test before production edits.
