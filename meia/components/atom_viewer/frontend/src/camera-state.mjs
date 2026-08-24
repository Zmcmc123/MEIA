export const DEFAULT_CAMERA = Object.freeze({
  eye: Object.freeze({ x: 1.25, y: 1.25, z: 1.25 }),
  up: Object.freeze({ x: 0, y: 0, z: 1 }),
  center: Object.freeze({ x: 0, y: 0, z: 0 }),
  projection: Object.freeze({ type: "orthographic" }),
})

const VECTOR_GROUPS = ["eye", "up", "center"]
const AXES = ["x", "y", "z"]
const USER_RELAYOUT_WINDOW_MS = 2000


function normalizeVector(value, fallback, name) {
  const source = value ?? {}
  if (typeof source !== "object" || Array.isArray(source)) {
    throw new Error(`camera ${name} must be an object`)
  }

  const result = {}
  for (const axis of AXES) {
    const component = source[axis] ?? fallback[axis]
    if (typeof component !== "number" || !Number.isFinite(component)) {
      throw new Error(`camera ${name}.${axis} must be finite`)
    }
    result[axis] = component
  }
  return result
}


export function normalizeCamera(value = {}) {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("camera must be an object")
  }

  const projection = value.projection?.type ?? value.projection ?? "orthographic"
  if (projection !== "orthographic") {
    throw new Error("camera must be orthographic")
  }
  return {
    eye: normalizeVector(value.eye, DEFAULT_CAMERA.eye, "eye"),
    up: normalizeVector(value.up, DEFAULT_CAMERA.up, "up"),
    center: normalizeVector(value.center, DEFAULT_CAMERA.center, "center"),
    projection: { type: "orthographic" },
  }
}


export function cloneCameraForPlotly(value) {
  return normalizeCamera(value)
}


export function mergeRelayoutCamera(previous, update) {
  if (update === null || typeof update !== "object" || Array.isArray(update)) {
    throw new Error("relayout update must be an object")
  }

  const base = normalizeCamera(previous)
  const nested = update["scene.camera"]
  const merged = structuredClone(base)

  if (nested !== undefined) {
    if (nested === null || typeof nested !== "object" || Array.isArray(nested)) {
      throw new Error("scene.camera must be an object")
    }
    for (const group of VECTOR_GROUPS) {
      if (nested[group] !== undefined) {
        const incoming = nested[group]
        if (incoming === null || typeof incoming !== "object" || Array.isArray(incoming)) {
          throw new Error(`scene.camera.${group} must be an object`)
        }
        merged[group] = { ...merged[group], ...incoming }
      }
    }
    if (nested.projection !== undefined) {
      if (nested.projection === null) {
        throw new Error("scene.camera.projection must not be null")
      }
      merged.projection = nested.projection
    }
  }

  for (const [key, component] of Object.entries(update)) {
    const match = key.match(/^scene\.camera\.(eye|up|center)\.(x|y|z)$/)
    if (match) {
      merged[match[1]][match[2]] = component
    }
    if (key === "scene.camera.projection.type") {
      merged.projection = { type: component }
    }
  }
  return normalizeCamera(merged)
}


function cameraOrientation(value) {
  const camera = normalizeCamera(value)
  const eye = AXES.map(axis => camera.eye[axis])
  const up = AXES.map(axis => camera.up[axis])
  const eyeNorm = Math.hypot(...eye)
  if (eyeNorm <= Number.EPSILON) {
    throw new Error("camera eye vector must be non-zero")
  }
  const view = eye.map(component => -component / eyeNorm)
  const right = [
    view[1] * up[2] - view[2] * up[1],
    view[2] * up[0] - view[0] * up[2],
    view[0] * up[1] - view[1] * up[0],
  ]
  const rightNorm = Math.hypot(...right)
  if (rightNorm <= Number.EPSILON) {
    throw new Error("camera up must not be parallel to view")
  }
  const normalizedRight = right.map(component => component / rightNorm)
  const correctedUp = [
    normalizedRight[1] * view[2] - normalizedRight[2] * view[1],
    normalizedRight[2] * view[0] - normalizedRight[0] * view[2],
    normalizedRight[0] * view[1] - normalizedRight[1] * view[0],
  ]
  return [...view, ...normalizedRight, ...correctedUp]
}


export function camerasEqual(a, b, tolerance = 1e-9) {
  const left = normalizeCamera(a)
  const right = normalizeCamera(b)
  if (left.projection.type !== right.projection.type) {
    return false
  }
  const leftOrientation = cameraOrientation(left)
  const rightOrientation = cameraOrientation(right)
  return leftOrientation.every(
    (component, index) => Math.abs(component - rightOrientation[index]) <= tolerance,
  )
}


export function viewerIsDirty(draftCamera, appliedCamera, _styleDirty = false) {
  return !camerasEqual(draftCamera, appliedCamera)
}


export function cameraApplyButtonState(draftCamera, appliedCamera, waiting) {
  const dirty = viewerIsDirty(draftCamera, appliedCamera)
  return {
    disabled: !dirty || Boolean(waiting),
    state: waiting ? "waiting" : (dirty ? "dirty" : "applied"),
  }
}


export function makeApplyCameraEvent(structureId, eventId, camera) {
  if (typeof structureId !== "string" || structureId.trim().length === 0) {
    throw new Error("structureId must be a non-empty string")
  }
  if (typeof eventId !== "string" || eventId.trim().length === 0) {
    throw new Error("eventId must be a non-empty string")
  }
  return {
    event_type: "apply_camera",
    event_id: eventId,
    structure_id: structureId,
    camera: normalizeCamera(camera),
  }
}


export function isUserInitiatedRelayout(
  lastInteractionAt,
  now,
  windowMs = USER_RELAYOUT_WINDOW_MS,
) {
  return (
    Number.isFinite(lastInteractionAt)
    && Number.isFinite(now)
    && now >= lastInteractionAt
    && now - lastInteractionAt <= windowMs
  )
}
