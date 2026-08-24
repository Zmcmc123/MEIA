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


test("selection clicks never refresh the camera-interaction clock", async () => {
  const { shouldMarkCameraInteraction } = await import("./selection-interactions.mjs")

  assert.equal(
    shouldMarkCameraInteraction(false, {type: "pointerdown", button: 0}),
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
})
