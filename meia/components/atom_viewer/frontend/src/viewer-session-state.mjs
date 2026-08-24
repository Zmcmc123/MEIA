import { normalizeCamera } from "./camera-state.mjs"
import { normalizeAngleStep } from "./camera-controls.mjs"
import { normalizeAtomIndices } from "./selection-state.mjs"
import { normalizeAspectRatio } from "./viewport-zoom.mjs"


const STORAGE_KEY = "meia.viewer.session.v1"


function normalizeIdentity(value, name) {
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`${name} must be a non-empty string`)
  }
  return value
}


function normalizeState(value) {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("viewer session state must be an object")
  }
  return {
    draftCamera: normalizeCamera(value.draftCamera),
    baseAspectRatio: normalizeAspectRatio(value.baseAspectRatio),
    draftAspectRatio: normalizeAspectRatio(value.draftAspectRatio),
    selectionModeActive: Boolean(value.selectionModeActive),
    pythonSelectedIndices: normalizeAtomIndices(value.pythonSelectedIndices),
    draftSelectedIndices: normalizeAtomIndices(value.draftSelectedIndices),
    angleStep: normalizeAngleStep(value.angleStep),
  }
}


export function reconcileViewerSessionState(
  cachedState,
  nextPythonSelectedIndices,
  waitingForSelection = false,
) {
  const state = normalizeState(cachedState)
  const nextSelection = normalizeAtomIndices(nextPythonSelectedIndices)
  const pythonSelectionChanged = (
    state.pythonSelectedIndices.length !== nextSelection.length
    || state.pythonSelectedIndices.some(
      (atomIndex, position) => atomIndex !== nextSelection[position],
    )
  )
  return {
    ...state,
    pythonSelectedIndices: nextSelection,
    draftSelectedIndices: pythonSelectionChanged || waitingForSelection
      ? nextSelection
      : state.draftSelectedIndices,
  }
}


export function loadViewerSessionState(storage, structureId, viewRevision) {
  try {
    const payload = JSON.parse(storage.getItem(STORAGE_KEY))
    const expectedStructureId = normalizeIdentity(structureId, "structureId")
    const expectedViewRevision = normalizeIdentity(viewRevision, "viewRevision")
    if (
      payload?.version !== 1
      || payload.structureId !== expectedStructureId
      || payload.viewRevision !== expectedViewRevision
    ) {
      return null
    }
    return normalizeState(payload.state)
  } catch (_error) {
    return null
  }
}


export function saveViewerSessionState(
  storage,
  structureId,
  viewRevision,
  state,
) {
  try {
    const payload = {
      version: 1,
      structureId: normalizeIdentity(structureId, "structureId"),
      viewRevision: normalizeIdentity(viewRevision, "viewRevision"),
      state: normalizeState(state),
    }
    storage.setItem(STORAGE_KEY, JSON.stringify(payload))
  } catch (_error) {
    // Storage can be unavailable in privacy-restricted browser contexts.
  }
}
