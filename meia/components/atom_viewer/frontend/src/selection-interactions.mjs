export function isPrimarySelectionPointer(selectionModeActive, event) {
  return (
    Boolean(selectionModeActive)
    && event?.type === "pointerdown"
    && event.button === 0
  )
}


export function shouldMarkCameraInteraction(selectionModeActive, event) {
  if (event?.type === "wheel") {
    return true
  }
  if (selectionModeActive) {
    return false
  }
  return event?.type === "pointerdown" || event?.type === "touchstart"
}


export function shouldConsumeSelectionClick(selectionModeActive, event) {
  return Boolean(selectionModeActive) && event?.type === "click"
}
