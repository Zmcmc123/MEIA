import { normalizeCamera } from "./camera-state.mjs"
import { normalizeAspectRatio } from "./viewport-zoom.mjs"


const MIN_VISIBLE_MARKER_SIZE = 1
const MAX_MARKER_SIZE = 180
const MIN_LINE_WIDTH = 0.5
const MAX_LINE_WIDTH = 40
const SELECTED_COLOR = "rgba(255,213,79,0.55)"
const TRANSPARENT_COLOR = "rgba(0,0,0,0)"


function clamp(value, minimum, maximum) {
  return Math.max(minimum, Math.min(maximum, value))
}


function finitePositive(value, label) {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    throw new Error(`${label} must be a positive finite number`)
  }
  return value
}


function scaleMarkerSize(baseSize, zoomScale, visible) {
  const base = finitePositive(baseSize, "base marker size")
  if (!visible) {
    return 0
  }
  return clamp(base * zoomScale, MIN_VISIBLE_MARKER_SIZE, MAX_MARKER_SIZE)
}


export function viewerTraceStyleAtZoom(trace, zoomScale, selectedIndices = []) {
  const scale = finitePositive(zoomScale, "zoom scale")
  const role = trace?.meta?.meia_role
  const baseMarkerSizes = trace?.meta?.meia_base_marker_sizes
  if (role === "atoms") {
    if (!Array.isArray(baseMarkerSizes)) {
      throw new Error("atom trace is missing base marker sizes")
    }
    return {
      "marker.size": baseMarkerSizes.map(size => scaleMarkerSize(size, scale, true)),
    }
  }
  if (role === "selection") {
    if (!Array.isArray(baseMarkerSizes)) {
      throw new Error("selection trace is missing base marker sizes")
    }
    const sourceAtomIndices = trace?.meta?.meia_source_atom_indices
    if (
      !Array.isArray(sourceAtomIndices)
      || sourceAtomIndices.length !== baseMarkerSizes.length
      || sourceAtomIndices.some(index => !Number.isInteger(index) || index < 0)
    ) {
      throw new Error("selection trace is missing source atom indices")
    }
    const selected = new Set(selectedIndices)
    return {
      "marker.size": baseMarkerSizes.map(
        (size, index) => scaleMarkerSize(
          size,
          scale,
          selected.has(sourceAtomIndices[index]),
        ),
      ),
      "marker.color": baseMarkerSizes.map(
        (_size, index) => selected.has(sourceAtomIndices[index])
          ? SELECTED_COLOR
          : TRANSPARENT_COLOR,
      ),
    }
  }
  if (["bonds", "bond_outlines", "hydrogen_bonds"].includes(role)) {
    const baseLineWidth = finitePositive(
      trace?.meta?.meia_base_line_width,
      "base line width",
    )
    return {
      "line.width": clamp(baseLineWidth * scale, MIN_LINE_WIDTH, MAX_LINE_WIDTH),
    }
  }
  return null
}


export function plotlyUpdateForSingleTrace(update) {
  if (update === null || typeof update !== "object" || Array.isArray(update)) {
    throw new Error("Plotly trace update must be an object")
  }
  return Object.fromEntries(
    Object.entries(update).map(([property, value]) => [property, [value]]),
  )
}


export function plotlyAtomicUpdateForSingleTrace(update, camera, aspectRatio) {
  return {
    dataUpdate: plotlyUpdateForSingleTrace(update),
    layoutUpdate: {
      "scene.camera": normalizeCamera(camera),
      "scene.aspectratio": normalizeAspectRatio(aspectRatio),
      "scene.aspectmode": "manual",
    },
  }
}
