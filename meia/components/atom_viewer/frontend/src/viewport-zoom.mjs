const AXES = ["x", "y", "z"]
const WHEEL_ZOOM_FACTOR = 1.1
const MIN_ASPECT_COMPONENT = 1e-4
const MAX_ASPECT_COMPONENT = 1e4


function finitePositive(value, label) {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    throw new Error(`${label} must be a positive finite number`)
  }
  return value
}


export function normalizeAspectRatio(value) {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("scene aspect ratio must be an object")
  }
  return Object.fromEntries(AXES.map(axis => [
    axis,
    finitePositive(value[axis], `scene aspect ratio ${axis}`),
  ]))
}


export function aspectRatiosEqual(left, right, tolerance = 1e-9) {
  const a = normalizeAspectRatio(left)
  const b = normalizeAspectRatio(right)
  return AXES.every(axis => Math.abs(a[axis] - b[axis]) <= tolerance)
}


export function aspectRatioZoomScale(baseValue, currentValue) {
  const base = normalizeAspectRatio(baseValue)
  const current = normalizeAspectRatio(currentValue)
  const ratios = AXES.map(axis => current[axis] / base[axis])
  return Math.cbrt(ratios.reduce((product, ratio) => product * ratio, 1))
}


export function mergeRelayoutAspectRatio(currentValue, update) {
  const current = normalizeAspectRatio(currentValue)
  if (update === null || typeof update !== "object" || Array.isArray(update)) {
    return current
  }
  const nested = update["scene.aspectratio"]
  const merged = nested === undefined
    ? {...current}
    : {...current, ...nested}
  for (const axis of AXES) {
    const dotted = update[`scene.aspectratio.${axis}`]
    if (dotted !== undefined) {
      merged[axis] = dotted
    }
  }
  return normalizeAspectRatio(merged)
}


export function zoomAspectRatioForWheel(value, deltaY) {
  if (typeof deltaY !== "number" || !Number.isFinite(deltaY)) {
    throw new Error("wheel delta must be finite")
  }
  const aspectRatio = normalizeAspectRatio(value)
  if (deltaY === 0) {
    return aspectRatio
  }
  const requestedFactor = deltaY < 0
    ? WHEEL_ZOOM_FACTOR
    : 1 / WHEEL_ZOOM_FACTOR
  const minimum = Math.min(...Object.values(aspectRatio))
  const maximum = Math.max(...Object.values(aspectRatio))
  const factor = Math.max(
    MIN_ASPECT_COMPONENT / minimum,
    Math.min(MAX_ASPECT_COMPONENT / maximum, requestedFactor),
  )
  return Object.fromEntries(
    AXES.map(axis => [axis, aspectRatio[axis] * factor]),
  )
}
