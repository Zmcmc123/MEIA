import { cloneCameraForPlotly, normalizeCamera } from "./camera-state.mjs"


const DEFAULT_ANGLE_STEP = 90
const MIN_ANGLE_STEP = 0.1
const MAX_ANGLE_STEP = 90
const DIRECTIONS = new Set(["left", "right", "up", "down"])
const AXES = ["x", "y", "z"]


function toVector(group) {
  return AXES.map(axis => group[axis])
}


function toGroup(vector) {
  const cleaned = vector.map(value => Math.abs(value) < 1e-12 ? 0 : value)
  return Object.fromEntries(AXES.map((axis, index) => [axis, cleaned[index]]))
}


function dot(a, b) {
  return a.reduce((sum, value, index) => sum + value * b[index], 0)
}


function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ]
}


function normalizeVector(vector, name) {
  const length = Math.hypot(...vector)
  if (!Number.isFinite(length) || length <= Number.EPSILON) {
    throw new Error(`${name} must be a non-zero finite vector`)
  }
  return vector.map(value => value / length)
}


function rotateVector(vector, axis, angleDegrees) {
  const radians = angleDegrees * Math.PI / 180
  const cosine = Math.cos(radians)
  const sine = Math.sin(radians)
  const axisCrossVector = cross(axis, vector)
  const axisDotVector = dot(axis, vector)
  return vector.map((value, index) => (
    value * cosine
    + axisCrossVector[index] * sine
    + axis[index] * axisDotVector * (1 - cosine)
  ))
}


export function normalizeAngleStep(value) {
  if (value === "" || value === null || value === undefined) {
    return DEFAULT_ANGLE_STEP
  }
  const result = typeof value === "number" ? value : Number(value)
  if (
    !Number.isFinite(result)
    || result < MIN_ANGLE_STEP
    || result > MAX_ANGLE_STEP
  ) {
    throw new Error(
      `angle step must be between ${MIN_ANGLE_STEP}° and ${MAX_ANGLE_STEP}°`,
    )
  }
  return result
}


export function orbitCamera(value, direction, angleDegrees) {
  if (!DIRECTIONS.has(direction)) {
    throw new Error(`unknown orbit direction: ${direction}`)
  }
  const angle = normalizeAngleStep(angleDegrees)
  const camera = normalizeCamera(value)
  const eye = toVector(camera.eye)
  const rawUp = toVector(camera.up)
  const view = normalizeVector(eye.map(component => -component), "camera view")
  const right = normalizeVector(cross(view, rawUp), "camera right")
  const correctedUp = normalizeVector(cross(right, view), "camera up")

  const horizontal = direction === "left" || direction === "right"
  const axis = horizontal ? correctedUp : right
  const signedAngle = (
    direction === "left" || direction === "up" ? -angle : angle
  )
  const rotatedEye = rotateVector(eye, axis, signedAngle)
  const rotatedUp = normalizeVector(
    rotateVector(correctedUp, axis, signedAngle),
    "rotated camera up",
  )

  return normalizeCamera({
    ...camera,
    eye: toGroup(rotatedEye),
    up: toGroup(rotatedUp),
  })
}


export function cameraFromAxisPreset(presets, axis) {
  if (!presets || typeof presets !== "object" || !(axis in presets)) {
    throw new Error(`unknown axis preset: ${axis}`)
  }
  return cloneCameraForPlotly(presets[axis])
}
