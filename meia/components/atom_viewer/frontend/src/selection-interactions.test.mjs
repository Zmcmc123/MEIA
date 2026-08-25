import assert from "node:assert/strict"
import { readFile } from "node:fs/promises"
import test from "node:test"


test("selection capture accepts only a primary pointer gesture, never a wheel", async () => {
  const { isPrimarySelectionPointer } = await import("./selection-interactions.mjs")

  assert.equal(
    isPrimarySelectionPointer(true, {type: "pointerdown", button: 0}),
    true,
  )
  assert.equal(
    isPrimarySelectionPointer(true, {type: "wheel", button: 0}),
    false,
  )
  assert.equal(
    isPrimarySelectionPointer(true, {type: "pointerdown", button: 2}),
    false,
  )
  assert.equal(
    isPrimarySelectionPointer(false, {type: "pointerdown", button: 0}),
    false,
  )
})


test("viewer blocks secondary-button gestures but keeps primary input", async () => {
  const { shouldBlockViewerGesture } = await import("./selection-interactions.mjs")

  for (const event of [
    {type: "pointerdown", button: 2, buttons: 2},
    {type: "pointermove", button: -1, buttons: 2},
    {type: "pointerup", button: 2, buttons: 0},
    {type: "mousedown", button: 2, buttons: 2},
    {type: "mousemove", button: -1, buttons: 2},
    {type: "mouseup", button: 2, buttons: 0},
    {type: "auxclick", button: 2, buttons: 0},
    {type: "contextmenu", button: 2, buttons: 0},
  ]) {
    assert.equal(shouldBlockViewerGesture(event), true, event.type)
  }

  for (const event of [
    {type: "pointerdown", button: 0, buttons: 1},
    {type: "pointermove", button: -1, buttons: 1},
    {type: "pointerup", button: 0, buttons: 0},
    {type: "mousedown", button: 0, buttons: 1},
    {type: "mousemove", button: -1, buttons: 1},
    {type: "mouseup", button: 0, buttons: 0},
    {type: "wheel", button: 0, buttons: 0},
  ]) {
    assert.equal(shouldBlockViewerGesture(event), false, event.type)
  }
})


test("atom screen projection runs only while batch selection is active", async () => {
  const { shouldProjectSelectionAtoms } = await import("./selection-interactions.mjs")

  assert.equal(shouldProjectSelectionAtoms(true, true), true)
  assert.equal(shouldProjectSelectionAtoms(true, false), false)
  assert.equal(shouldProjectSelectionAtoms(false, true), false)
})


test("selection clicks never refresh the camera-interaction clock", async () => {
  const { shouldMarkCameraInteraction } = await import("./selection-interactions.mjs")

  assert.equal(
    shouldMarkCameraInteraction(false, {type: "pointerdown", button: 0}),
    true,
  )
  assert.equal(
    shouldMarkCameraInteraction(false, {type: "pointermove", buttons: 1}),
    true,
  )
  assert.equal(
    shouldMarkCameraInteraction(false, {type: "pointermove", buttons: 0}),
    false,
  )
  assert.equal(
    shouldMarkCameraInteraction(false, {type: "pointerup", button: 0}),
    true,
  )
  assert.equal(
    shouldMarkCameraInteraction(true, {type: "pointerdown", button: 0}),
    false,
  )
  assert.equal(
    shouldMarkCameraInteraction(true, {type: "click", button: 0}),
    false,
  )
  assert.equal(
    shouldMarkCameraInteraction(true, {type: "wheel"}),
    true,
  )
})


test("selection mode consumes the synthetic click before Plotly sees it", async () => {
  const { shouldConsumeSelectionClick } = await import("./selection-interactions.mjs")

  assert.equal(
    shouldConsumeSelectionClick(true, {type: "click", button: 0}),
    true,
  )
  assert.equal(
    shouldConsumeSelectionClick(false, {type: "click", button: 0}),
    false,
  )
  assert.equal(
    shouldConsumeSelectionClick(true, {type: "wheel"}),
    false,
  )
})


test("the viewer wires camera starts and selection clicks in capture phase", async () => {
  const mainSource = await readFile(new URL("./main.mjs", import.meta.url), "utf8")

  assert.match(
    mainSource,
    /graph\.addEventListener\(eventName,[\s\S]{0,320}shouldMarkCameraInteraction\(selectionModeActive, event\)[\s\S]{0,180}\{capture: true, passive: true\}/u,
  )
  assert.match(
    mainSource,
    /viewerWrap\.addEventListener\("click",[\s\S]{0,180}shouldConsumeSelectionClick\(selectionModeActive, event\)[\s\S]{0,180}event\.stopPropagation\(\)[\s\S]{0,80}\{capture: true\}/u,
  )
  assert.match(
    mainSource,
    /for \(const eventName of SECONDARY_GESTURE_EVENTS\)[\s\S]{0,300}shouldBlockViewerGesture\(event\)[\s\S]{0,160}event\.preventDefault\(\)[\s\S]{0,100}event\.stopPropagation\(\)[\s\S]{0,120}\{capture: true, passive: false\}/u,
  )
})
