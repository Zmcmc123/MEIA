import assert from "node:assert/strict"
import test from "node:test"

import {
  camerasEqual,
  cloneCameraForPlotly,
  isUserInitiatedRelayout,
  makeApplyCameraEvent,
  mergeRelayoutCamera,
  normalizeCamera,
  cameraApplyButtonState,
  viewerIsDirty,
} from "./camera-state.mjs"


async function loadCameraControls() {
  return import("./camera-controls.mjs")
}


test("normalizeCamera fills every Plotly camera field", () => {
  const camera = normalizeCamera({ eye: { x: 2, y: 0, z: 0 } })

  assert.deepEqual(camera.center, { x: 0, y: 0, z: 0 })
  assert.deepEqual(camera.up, { x: 0, y: 0, z: 1 })
  assert.deepEqual(camera.projection, { type: "orthographic" })
})


test("mergeRelayoutCamera accepts nested scene.camera", () => {
  const initial = normalizeCamera({})
  const merged = mergeRelayoutCamera(initial, {
    "scene.camera": { eye: { x: 0, y: 2, z: 0 } },
  })

  assert.deepEqual(merged.eye, { x: 0, y: 2, z: 0 })
  assert.deepEqual(merged.up, initial.up)
})


test("mergeRelayoutCamera accepts dotted camera keys", () => {
  const initial = normalizeCamera({})
  const merged = mergeRelayoutCamera(initial, {
    "scene.camera.eye.x": -2,
    "scene.camera.up.y": 1,
  })

  assert.equal(merged.eye.x, -2)
  assert.equal(merged.up.y, 1)
})


test("camerasEqual uses a numeric tolerance", () => {
  const camera = normalizeCamera({})
  const close = structuredClone(camera)
  close.eye.x += 5e-10

  assert.equal(camerasEqual(camera, close), true)
  close.eye.x += 5e-7
  assert.equal(camerasEqual(camera, close), false)
})


test("camerasEqual ignores orthographic zoom and pan", () => {
  const applied = normalizeCamera({
    eye: { x: 1, y: 1, z: 1 },
    center: { x: 0, y: 0, z: 0 },
  })
  const zoomed = normalizeCamera({
    eye: { x: 2, y: 2, z: 2 },
    center: { x: 0, y: 0, z: 0 },
  })
  const panned = normalizeCamera({
    eye: { x: 1, y: 1, z: 1 },
    center: { x: 0.3, y: -0.2, z: 0.1 },
  })

  assert.equal(camerasEqual(applied, zoomed), true)
  assert.equal(camerasEqual(applied, panned), true)
})


test("viewer dirty state depends only on camera orientation", () => {
  const camera = normalizeCamera({})

  assert.equal(viewerIsDirty(camera, camera, false), false)
  assert.equal(viewerIsDirty(camera, camera, true), false)
  assert.equal(
    viewerIsDirty(camera, normalizeCamera({eye: {x: 2, y: 0, z: 0}}), false),
    true,
  )
})


test("camera apply button returns semantic camera-only states", () => {
  const applied = normalizeCamera({})
  const dirty = normalizeCamera({ eye: { x: 2, y: 0, z: 0 } })

  assert.deepEqual(cameraApplyButtonState(applied, applied, false), {
    disabled: true,
    state: "applied",
  })
  assert.deepEqual(cameraApplyButtonState(dirty, applied, false), {
    disabled: false,
    state: "dirty",
  })
  assert.deepEqual(cameraApplyButtonState(dirty, applied, true), {
    disabled: true,
    state: "waiting",
  })
})


test("apply camera event carries camera identity only", () => {
  const event = makeApplyCameraEvent("structure-a", "event-1", normalizeCamera({}))

  assert.deepEqual(Object.keys(event).sort(), [
    "camera",
    "event_id",
    "event_type",
    "structure_id",
  ])
  assert.equal(event.event_type, "apply_camera")
})


test("normalizeCamera rejects perspective and non-finite values", () => {
  assert.throws(
    () => normalizeCamera({ projection: { type: "perspective" } }),
    /orthographic/,
  )
  assert.throws(() => normalizeCamera({ eye: { x: NaN } }), /finite/)
  assert.throws(() => normalizeCamera({ eye: { x: Infinity } }), /finite/)
})


test("mergeRelayoutCamera rejects null camera groups", () => {
  const initial = normalizeCamera({})

  assert.throws(
    () => mergeRelayoutCamera(initial, { "scene.camera": { eye: null } }),
    /eye must be an object/,
  )
})


test("unrelated relayout keys do not change the draft camera", () => {
  const initial = normalizeCamera({ eye: { x: -1, y: 2, z: 3 } })
  const merged = mergeRelayoutCamera(initial, { "scene.aspectmode": "data" })

  assert.deepEqual(merged, initial)
})


test("programmatic relayout is ignored outside a recent user interaction", () => {
  assert.equal(isUserInitiatedRelayout(Number.NEGATIVE_INFINITY, 5000), false)
  assert.equal(isUserInitiatedRelayout(1000, 5000), false)
  assert.equal(isUserInitiatedRelayout(4500, 5000), true)
})


test("Plotly camera mutations cannot change AtomViewer state", () => {
  const stateCamera = normalizeCamera({ up: { x: 0, y: 0, z: -1 } })
  const plotlyCamera = cloneCameraForPlotly(stateCamera)

  plotlyCamera.up.z = 1

  assert.equal(stateCamera.up.z, -1)
  assert.notEqual(plotlyCamera, stateCamera)
})


test("angle step defaults to 90 degrees and rejects values outside 0.1-90", async () => {
  const { normalizeAngleStep } = await loadCameraControls()

  assert.equal(normalizeAngleStep(""), 90)
  assert.equal(normalizeAngleStep("5.5"), 5.5)
  assert.throws(() => normalizeAngleStep("0"), /0.1.*90/)
  assert.throws(() => normalizeAngleStep("90.1"), /0.1.*90/)
})


test("screen-local arrow controls orbit a camera by exactly 90 degrees", async () => {
  const { orbitCamera } = await loadCameraControls()
  const initial = normalizeCamera({
    eye: { x: 0, y: 0, z: 2 },
    up: { x: 0, y: 1, z: 0 },
  })

  assert.deepEqual(orbitCamera(initial, "left", 90).eye, { x: -2, y: 0, z: 0 })
  assert.deepEqual(orbitCamera(initial, "right", 90).eye, { x: 2, y: 0, z: 0 })
  assert.deepEqual(orbitCamera(initial, "up", 90).eye, { x: 0, y: 2, z: 0 })
  assert.deepEqual(orbitCamera(initial, "down", 90).eye, { x: 0, y: -2, z: 0 })
})


test("arrow controls preserve eye distance and a valid orthogonal up vector", async () => {
  const { orbitCamera } = await loadCameraControls()
  const initial = normalizeCamera({
    eye: { x: 1.25, y: 1.25, z: 1.25 },
    up: { x: 0, y: 0, z: 1 },
  })

  const rotated = orbitCamera(initial, "up", 17.5)
  const eye = Object.values(rotated.eye)
  const up = Object.values(rotated.up)
  const dot = eye.reduce((sum, value, index) => sum + value * up[index], 0)

  assert.ok(Math.abs(Math.hypot(...eye) - Math.hypot(1.25, 1.25, 1.25)) < 1e-12)
  assert.ok(Math.abs(Math.hypot(...up) - 1) < 1e-12)
  assert.ok(Math.abs(dot) < 1e-12)
})


test("axis preset selection clones and validates the requested draft camera", async () => {
  const { cameraFromAxisPreset } = await loadCameraControls()
  const presets = {
    a: normalizeCamera({ eye: { x: 2, y: 0, z: 0 } }),
    b: normalizeCamera({ eye: { x: 0, y: 2, z: 0 } }),
    c: normalizeCamera({ eye: { x: 0, y: 0, z: 2 }, up: { x: 0, y: 1, z: 0 } }),
  }

  const selected = cameraFromAxisPreset(presets, "b")
  selected.eye.y = 9

  assert.equal(presets.b.eye.y, 2)
  assert.throws(() => cameraFromAxisPreset(presets, "d"), /unknown axis/)
})
