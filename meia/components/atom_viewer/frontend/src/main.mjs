import Plotly from "plotly.js-dist-min"
import { Streamlit } from "streamlit-component-lib"

import {
  cameraApplyButtonState,
  cloneCameraForPlotly,
  isUserInitiatedRelayout,
  makeApplyCameraEvent,
  mergeRelayoutCamera,
  normalizeCamera,
} from "./camera-state.mjs"
import {
  cameraFromAxisPreset,
  normalizeAngleStep,
  orbitCamera,
} from "./camera-controls.mjs"
import {
  isPrimarySelectionPointer,
  shouldConsumeSelectionClick,
  shouldMarkCameraInteraction,
} from "./selection-interactions.mjs"
import {
  addAtomIndices,
  atomsInsideRectangle,
  atomSelectionFromPoint,
  makeAtomSelectionBatchEvent,
  makeAtomSelectionEvent,
  nearestAtomAtPoint,
  normalizeAtomIndices,
  projectAtomScreenPositions,
  toggleAtomIndex,
} from "./selection-state.mjs"
import {
  plotlyAtomicUpdateForSingleTrace,
  viewerTraceStyleAtZoom,
} from "./viewer-style.mjs"
import {
  aspectRatiosEqual,
  aspectRatioZoomScale,
  mergeRelayoutAspectRatio,
  normalizeAspectRatio,
  zoomAspectRatioForWheel,
} from "./viewport-zoom.mjs"
import {
  formatViewerMessage,
  normalizeViewerLocale,
  normalizeViewerMessages,
  replaceViewerMessages,
} from "./viewer-messages.mjs"
import {
  loadViewerSessionState,
  reconcileViewerSessionState,
  saveViewerSessionState,
} from "./viewer-session-state.mjs"
import "./styles.css"


const graph = document.querySelector("#viewer")
const viewerWrap = document.querySelector("#viewer-wrap")
const viewTools = document.querySelector("#view-tools")
const angleStepLabel = document.querySelector("#angle-step-label")
const orbitPad = document.querySelector("#orbit-pad")
const axisPresets = document.querySelector("#axis-presets")
const selectionTools = document.querySelector("#selection-tools")
const button = document.querySelector("#apply-camera")
const status = document.querySelector("#status")
const angleStepInput = document.querySelector("#angle-step")
const orbitButtons = [...document.querySelectorAll("[data-orbit]")]
const axisButtons = [...document.querySelectorAll("[data-axis]")]
const selectionModeButton = document.querySelector("#selection-mode")
const selectionHint = document.querySelector("#selection-hint")
const selectionCount = document.querySelector("#selection-count")
const clearSelectionButton = document.querySelector("#clear-selection")
const confirmSelectionButton = document.querySelector("#confirm-selection")
const selectionOverlay = document.querySelector("#selection-overlay")
const selectionBox = document.querySelector("#selection-box")

const CLICK_HIT_RADIUS = 18
const DRAG_THRESHOLD = 5
const DEFAULT_ANGLE_STEP = normalizeAngleStep(angleStepInput.value)

let structureId = null
let viewRevision = null
let appliedCamera = normalizeCamera({})
let draftCamera = cloneCameraForPlotly(appliedCamera)
let baseAspectRatio = null
let draftAspectRatio = null
let waitingForPython = false
let lastUserInteractionAt = Number.NEGATIVE_INFINITY
let axisCameras = null
let batchSelectionEnabled = false
let selectionModeActive = false
let pythonSelectedIndices = []
let draftSelectedIndices = []
let projectedAtoms = []
let selectionGesture = null
let waitingForSelection = false
let viewerLocale = null
let viewerMessages = null


function browserSessionStorage() {
  try {
    return globalThis.sessionStorage
  } catch (_error) {
    return null
  }
}


function persistViewerSession() {
  if (
    structureId === null
    || viewRevision === null
    || baseAspectRatio === null
    || draftAspectRatio === null
  ) {
    return
  }
  saveViewerSessionState(
    browserSessionStorage(),
    structureId,
    viewRevision,
    {
      draftCamera,
      baseAspectRatio,
      draftAspectRatio,
      selectionModeActive,
      pythonSelectedIndices,
      draftSelectedIndices,
      angleStep: angleStepInput.value,
    },
  )
}


function text(key, params = {}) {
  return formatViewerMessage(viewerMessages, key, params, viewerLocale)
}


function applyViewerMessages(locale, messages) {
  const translatedState = replaceViewerMessages(
    {
      draftCamera,
      draftAspectRatio,
      selectionModeActive,
      draftSelectedIndices,
      waitingForPython,
      waitingForSelection,
    },
    normalizeViewerLocale(locale),
    normalizeViewerMessages(messages),
  )
  viewerLocale = translatedState.locale
  viewerMessages = translatedState.messages
  document.documentElement.lang = viewerLocale
  viewTools.setAttribute("aria-label", text("controls.aria"))
  angleStepLabel.textContent = text("angle_step.label")
  orbitPad.setAttribute("aria-label", text("orbit.aria"))
  for (const control of orbitButtons) {
    control.setAttribute("aria-label", text(`orbit.${control.dataset.orbit}`))
  }
  axisPresets.setAttribute("aria-label", text("axis.aria"))
  selectionTools.setAttribute("aria-label", text("selection.tools_aria"))
  clearSelectionButton.textContent = text("selection.clear")
  graph.setAttribute("aria-label", text("canvas.aria"))
  selectionOverlay.setAttribute(
    "aria-label",
    text("selection.overlay_aria"),
  )
}


function setDirtyState() {
  const state = cameraApplyButtonState(
    draftCamera,
    appliedCamera,
    waitingForPython,
  )
  button.disabled = state.disabled
  button.textContent = text(
    state.state === "dirty" ? "camera.apply" : `camera.${state.state}`,
  )
  status.textContent = ""
  persistViewerSession()
}


function atomIndexSetsEqual(left, right) {
  return left.length === right.length
    && left.every((value, index) => value === right[index])
}


function selectionIsDirty() {
  return !atomIndexSetsEqual(draftSelectedIndices, pythonSelectedIndices)
}


function refreshProjectedAtoms() {
  if (!batchSelectionEnabled) {
    projectedAtoms = []
    return
  }
  try {
    projectedAtoms = projectAtomScreenPositions(
      graph,
      selectionOverlay.getBoundingClientRect(),
    )
  } catch (_error) {
    // Plotly 在 react 后的首帧才生成 cameraParams；下一次动画帧或
    // 用户开始选择时会再次投影，不向用户显示短暂的初始化错误。
    projectedAtoms = []
  }
}


async function syncViewerTraceStyles() {
  if (
    !Array.isArray(graph.data)
    || baseAspectRatio === null
    || draftAspectRatio === null
  ) {
    return
  }
  const zoomScale = aspectRatioZoomScale(baseAspectRatio, draftAspectRatio)
  for (let index = 0; index < graph.data.length; index += 1) {
    const update = viewerTraceStyleAtZoom(
      graph.data[index],
      zoomScale,
      draftSelectedIndices,
    )
    if (update !== null) {
      const atomicUpdate = plotlyAtomicUpdateForSingleTrace(
        update,
        draftCamera,
        draftAspectRatio,
      )
      await Plotly.update(
        graph,
        atomicUpdate.dataUpdate,
        atomicUpdate.layoutUpdate,
        [index],
      )
    }
  }
}


function updateSelectionControls() {
  if (!batchSelectionEnabled) {
    selectionModeActive = false
  }
  const dirty = selectionIsDirty()
  selectionModeButton.disabled = !batchSelectionEnabled || waitingForSelection
  selectionModeButton.setAttribute("aria-pressed", String(selectionModeActive))
  selectionModeButton.textContent = selectionModeActive
    ? text("selection.mode.on")
    : text("selection.mode.off")
  selectionOverlay.classList.toggle("active", selectionModeActive)
  viewerWrap.classList.toggle("selection-active", selectionModeActive)
  selectionOverlay.setAttribute("aria-hidden", String(!selectionModeActive))
  selectionHint.textContent = !batchSelectionEnabled
    ? text("selection.unavailable")
    : (selectionModeActive
      ? text("selection.hint.active")
      : text("selection.hint.inactive"))
  selectionCount.textContent = text("selection.count", {
    count: draftSelectedIndices.length,
    pending: dirty ? text("selection.pending") : "",
  })
  clearSelectionButton.disabled = (
    !batchSelectionEnabled
    || draftSelectedIndices.length === 0
    || waitingForSelection
  )
  confirmSelectionButton.disabled = (
    !batchSelectionEnabled || !dirty || waitingForSelection
  )
  confirmSelectionButton.textContent = waitingForSelection
    ? text("selection.confirming")
    : text("selection.confirm")
  persistViewerSession()
}


function pointerPosition(event) {
  const bounds = selectionOverlay.getBoundingClientRect()
  return {
    x: Math.max(0, Math.min(bounds.width, event.clientX - bounds.left)),
    y: Math.max(0, Math.min(bounds.height, event.clientY - bounds.top)),
  }
}


function showSelectionBox(start, end) {
  selectionBox.style.display = "block"
  selectionBox.setAttribute("x", String(Math.min(start.x, end.x)))
  selectionBox.setAttribute("y", String(Math.min(start.y, end.y)))
  selectionBox.setAttribute("width", String(Math.abs(end.x - start.x)))
  selectionBox.setAttribute("height", String(Math.abs(end.y - start.y)))
}


function hideSelectionBox() {
  selectionBox.style.display = "none"
  selectionBox.setAttribute("width", "0")
  selectionBox.setAttribute("height", "0")
}


function nextEventId() {
  if (typeof crypto.randomUUID === "function") {
    return crypto.randomUUID()
  }
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`
}


function markUserInteraction() {
  lastUserInteractionAt = performance.now()
}


function setToolControlsEnabled(enabled) {
  for (const control of [...orbitButtons, ...axisButtons]) {
    control.disabled = !enabled
  }
  angleStepInput.disabled = !enabled
}


async function showDraftCamera(nextCamera) {
  draftCamera = normalizeCamera(nextCamera)
  await Plotly.relayout(graph, {
    "scene.camera": cloneCameraForPlotly(draftCamera),
  })
  setDirtyState()
  requestAnimationFrame(refreshProjectedAtoms)
}


for (const eventName of ["pointerdown", "touchstart"]) {
  graph.addEventListener(eventName, event => {
    if (shouldMarkCameraInteraction(selectionModeActive, event)) {
      markUserInteraction()
    }
  }, {capture: true, passive: true})
}
// Plotly 的 canvas 在目标阶段同步发出 relayout；捕获阶段先记录滚动，
// 才能把触控板/滚轮产生的 aspectratio 更新识别为用户缩放。
graph.addEventListener("wheel", event => {
  if (shouldMarkCameraInteraction(selectionModeActive, event)) {
    markUserInteraction()
  }
}, {capture: true, passive: true})


async function showDraftAspectRatio(nextAspectRatio) {
  draftAspectRatio = normalizeAspectRatio(nextAspectRatio)
  await Plotly.relayout(graph, {
    "scene.aspectratio": draftAspectRatio,
    "scene.aspectmode": "manual",
  })
  await syncViewerTraceStyles()
  persistViewerSession()
  requestAnimationFrame(refreshProjectedAtoms)
}


viewerWrap.addEventListener("wheel", event => {
  if (!selectionModeActive) {
    return
  }
  event.preventDefault()
  event.stopPropagation()
  markUserInteraction()
  void showDraftAspectRatio(
    zoomAspectRatioForWheel(draftAspectRatio, event.deltaY),
  ).catch(error => {
    status.textContent = text("error.zoom", {detail: error.message})
  })
}, {capture: true, passive: false})


async function onRender(event) {
  try {
    const args = event.detail.args
    applyViewerMessages(args.locale, args.messages)
    const nextApplied = normalizeCamera(args.applied_camera)
    axisCameras = {
      a: cameraFromAxisPreset(args.axis_cameras, "a"),
      b: cameraFromAxisPreset(args.axis_cameras, "b"),
      c: cameraFromAxisPreset(args.axis_cameras, "c"),
    }
    const structureChanged = args.structure_id !== structureId
    const revisionChanged = args.view_revision !== viewRevision
    const nextPythonSelection = normalizeAtomIndices(
      args.selected_atom_indices ?? [],
    )
    const cachedState = structureChanged || revisionChanged
      ? loadViewerSessionState(
        browserSessionStorage(),
        args.structure_id,
        args.view_revision,
      )
      : null
    const restoredState = cachedState === null
      ? null
      : reconcileViewerSessionState(
        cachedState,
        nextPythonSelection,
        waitingForSelection,
      )
    if (structureChanged || revisionChanged) {
      if (restoredState === null) {
        draftCamera = cloneCameraForPlotly(nextApplied)
        baseAspectRatio = null
        draftAspectRatio = null
        selectionModeActive = false
        pythonSelectedIndices = []
        draftSelectedIndices = []
        angleStepInput.value = String(DEFAULT_ANGLE_STEP)
      } else {
        draftCamera = cloneCameraForPlotly(restoredState.draftCamera)
        baseAspectRatio = normalizeAspectRatio(restoredState.baseAspectRatio)
        draftAspectRatio = normalizeAspectRatio(restoredState.draftAspectRatio)
        selectionModeActive = restoredState.selectionModeActive
        pythonSelectedIndices = restoredState.pythonSelectedIndices
        draftSelectedIndices = restoredState.draftSelectedIndices
        angleStepInput.value = String(restoredState.angleStep)
      }
    }
    structureId = args.structure_id
    viewRevision = args.view_revision
    appliedCamera = nextApplied
    const pythonSelectionChanged = !atomIndexSetsEqual(
      nextPythonSelection,
      pythonSelectedIndices,
    )
    batchSelectionEnabled = Boolean(args.batch_selection_enabled)
    if (
      (restoredState === null && (structureChanged || revisionChanged))
      || pythonSelectionChanged
      || waitingForSelection
    ) {
      draftSelectedIndices = nextPythonSelection
    }
    pythonSelectedIndices = nextPythonSelection
    waitingForPython = false
    waitingForSelection = false

    const layout = structuredClone(args.figure.layout ?? {})
    layout.scene = {
      ...(layout.scene ?? {}),
      camera: cloneCameraForPlotly(draftCamera),
      uirevision: viewRevision,
    }
    if (draftAspectRatio !== null) {
      layout.scene.aspectmode = "manual"
      layout.scene.aspectratio = normalizeAspectRatio(draftAspectRatio)
    }
    const config = {
      displaylogo: false,
      responsive: true,
      scrollZoom: true,
      ...(args.figure.config ?? {}),
    }
    await Plotly.react(graph, args.figure.data ?? [], layout, config)
    const renderedAspectRatio = normalizeAspectRatio(
      graph?._fullLayout?.scene?.aspectratio,
    )
    draftAspectRatio = renderedAspectRatio
    if (baseAspectRatio === null) {
      baseAspectRatio = renderedAspectRatio
    }
    await syncViewerTraceStyles()
    graph.removeAllListeners("plotly_relayout")
    graph.removeAllListeners("plotly_click")
    graph.on("plotly_relayout", update => {
      if (!isUserInitiatedRelayout(lastUserInteractionAt, performance.now())) {
        return
      }
      try {
        const nextAspectRatio = mergeRelayoutAspectRatio(
          draftAspectRatio,
          update,
        )
        const zoomChanged = !aspectRatiosEqual(
          draftAspectRatio,
          nextAspectRatio,
        )
        draftCamera = mergeRelayoutCamera(draftCamera, update)
        draftAspectRatio = nextAspectRatio
        setDirtyState()
        if (zoomChanged) {
          void syncViewerTraceStyles()
        }
        requestAnimationFrame(refreshProjectedAtoms)
      } catch (error) {
        status.textContent = text("error.camera", {detail: error.message})
        button.disabled = true
      }
    })
    graph.on("plotly_click", clickEvent => {
      if (batchSelectionEnabled) {
        return
      }
      const selection = atomSelectionFromPoint(clickEvent?.points?.[0])
      if (selection === null) {
        return
      }
      Streamlit.setComponentValue(makeAtomSelectionEvent(
        structureId,
        selection.atomIndex,
        selection.atomSymbol,
        nextEventId(),
      ))
    })
    setDirtyState()
    setToolControlsEnabled(true)
    updateSelectionControls()
    requestAnimationFrame(refreshProjectedAtoms)
  } catch (error) {
    status.textContent = viewerMessages === null
      ? String(error.message)
      : text("error.load", {detail: error.message})
    button.disabled = true
    setToolControlsEnabled(false)
    batchSelectionEnabled = false
    updateSelectionControls()
  } finally {
    Streamlit.setFrameHeight()
  }
}


selectionModeButton.addEventListener("click", () => {
  if (selectionModeButton.disabled) {
    return
  }
  selectionModeActive = !selectionModeActive
  hideSelectionBox()
  selectionGesture = null
  refreshProjectedAtoms()
  updateSelectionControls()
})


viewerWrap.addEventListener("click", event => {
  if (!shouldConsumeSelectionClick(selectionModeActive, event)) {
    return
  }
  event.preventDefault()
  event.stopPropagation()
}, {capture: true})


clearSelectionButton.addEventListener("click", () => {
  if (clearSelectionButton.disabled) {
    return
  }
  draftSelectedIndices = []
  updateSelectionControls()
  void syncViewerTraceStyles()
})


confirmSelectionButton.addEventListener("click", () => {
  if (confirmSelectionButton.disabled) {
    return
  }
  waitingForSelection = true
  updateSelectionControls()
  Streamlit.setComponentValue(makeAtomSelectionBatchEvent(
    structureId,
    draftSelectedIndices,
    nextEventId(),
  ))
})


viewerWrap.addEventListener("pointerdown", event => {
  if (!isPrimarySelectionPointer(selectionModeActive, event)) {
    return
  }
  event.preventDefault()
  event.stopPropagation()
  refreshProjectedAtoms()
  const start = pointerPosition(event)
  selectionGesture = { pointerId: event.pointerId, start, end: start }
  viewerWrap.setPointerCapture?.(event.pointerId)
  showSelectionBox(start, start)
}, { capture: true })


viewerWrap.addEventListener("pointermove", event => {
  if (selectionGesture?.pointerId !== event.pointerId) {
    return
  }
  event.preventDefault()
  event.stopPropagation()
  const end = pointerPosition(event)
  selectionGesture.end = end
  showSelectionBox(selectionGesture.start, end)
}, { capture: true })


viewerWrap.addEventListener("pointerup", event => {
  if (selectionGesture?.pointerId !== event.pointerId) {
    return
  }
  event.preventDefault()
  event.stopPropagation()
  const start = selectionGesture.start
  const end = pointerPosition(event)
  const dragDistance = Math.hypot(end.x - start.x, end.y - start.y)
  selectionGesture = null
  viewerWrap.releasePointerCapture?.(event.pointerId)
  hideSelectionBox()
  if (dragDistance < DRAG_THRESHOLD) {
    const atom = nearestAtomAtPoint(projectedAtoms, end, CLICK_HIT_RADIUS)
    if (atom !== null) {
      draftSelectedIndices = toggleAtomIndex(
        draftSelectedIndices,
        atom.atomIndex,
      )
    }
  } else {
    draftSelectedIndices = addAtomIndices(
      draftSelectedIndices,
      atomsInsideRectangle(projectedAtoms, {
        x0: start.x,
        y0: start.y,
        x1: end.x,
        y1: end.y,
      }),
    )
  }
  updateSelectionControls()
  void syncViewerTraceStyles()
}, { capture: true })


viewerWrap.addEventListener("pointercancel", event => {
  if (selectionGesture?.pointerId !== event.pointerId) {
    return
  }
  event.stopPropagation()
  selectionGesture = null
  hideSelectionBox()
}, { capture: true })


new ResizeObserver(() => {
  requestAnimationFrame(refreshProjectedAtoms)
}).observe(graph)


for (const control of orbitButtons) {
  control.addEventListener("click", async () => {
    try {
      const step = normalizeAngleStep(angleStepInput.value)
      await showDraftCamera(orbitCamera(draftCamera, control.dataset.orbit, step))
    } catch (error) {
      status.textContent = text("error.adjust", {detail: error.message})
    }
  })
}


angleStepInput.addEventListener("change", persistViewerSession)


for (const control of axisButtons) {
  control.addEventListener("click", async () => {
    try {
      await showDraftCamera(cameraFromAxisPreset(axisCameras, control.dataset.axis))
    } catch (error) {
      status.textContent = text("error.adjust", {detail: error.message})
    }
  })
}


button.addEventListener("click", () => {
  if (button.disabled) {
    return
  }
  waitingForPython = true
  setDirtyState()
  Streamlit.setComponentValue(makeApplyCameraEvent(
    structureId,
    nextEventId(),
    draftCamera,
  ))
})

Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRender)
Streamlit.setComponentReady()
setToolControlsEnabled(false)
