export function isPrimarySelectionPointer(selectionModeActive, event) {
  return (
    Boolean(selectionModeActive)
    && event?.type === "pointerdown"
    && event.button === 0
  )
}


const SECONDARY_BUTTON = 2
const SECONDARY_BUTTON_MASK = 2


export function shouldBlockViewerGesture(event) {
  if (event?.type === "contextmenu") {
    return true
  }
  if (["pointermove", "mousemove"].includes(event?.type)) {
    return (event.buttons & SECONDARY_BUTTON_MASK) !== 0
  }
  if ([
    "pointerdown",
    "pointerup",
    "mousedown",
    "mouseup",
    "auxclick",
  ].includes(event?.type)) {
    return event.button === SECONDARY_BUTTON
  }
  return false
}


export function shouldProjectSelectionAtoms(
  batchSelectionEnabled,
  selectionModeActive,
) {
  return Boolean(batchSelectionEnabled) && Boolean(selectionModeActive)
}


export function shouldMarkCameraInteraction(selectionModeActive, event) {
  if (event?.type === "wheel") {
    return true
  }
  if (selectionModeActive) {
    return false
  }
  if (event?.type === "pointermove") {
    return (event.buttons & 1) !== 0
  }
  return [
    "pointerdown",
    "pointerup",
    "touchstart",
    "touchmove",
    "touchend",
  ].includes(event?.type)
}


export function shouldConsumeSelectionClick(selectionModeActive, event) {
  return Boolean(selectionModeActive) && event?.type === "click"
}
