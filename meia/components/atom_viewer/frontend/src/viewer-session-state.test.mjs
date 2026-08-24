import assert from "node:assert/strict"
import { readFile } from "node:fs/promises"
import test from "node:test"

import {
  loadViewerSessionState,
  reconcileViewerSessionState,
  saveViewerSessionState,
} from "./viewer-session-state.mjs"


function memoryStorage() {
  const values = new Map()
  return {
    getItem(key) {
      return values.get(key) ?? null
    },
    setItem(key, value) {
      values.set(key, value)
    },
  }
}


const state = {
  draftCamera: {
    eye: {x: 2, y: 0, z: 1},
    up: {x: 0, y: 0, z: 1},
    center: {x: 0, y: 0, z: 0},
    projection: {type: "orthographic"},
  },
  baseAspectRatio: {x: 2, y: 1, z: 1},
  draftAspectRatio: {x: 3, y: 1.5, z: 1.5},
  selectionModeActive: true,
  pythonSelectedIndices: [1],
  draftSelectedIndices: [1, 3],
  angleStep: 17.5,
}


test("viewer interaction state survives a component iframe reload", () => {
  const storage = memoryStorage()

  saveViewerSessionState(storage, "structure-a", "revision-a", state)

  assert.deepEqual(
    loadViewerSessionState(storage, "structure-a", "revision-a"),
    state,
  )
})


test("render reconciliation preserves every 3D draft when Python selection is unchanged", () => {
  assert.deepEqual(
    reconcileViewerSessionState(state, [1]),
    state,
  )
})


test("render reconciliation resets only selection drafts after an external selection", () => {
  assert.deepEqual(
    reconcileViewerSessionState(state, [2]),
    {
      ...state,
      pythonSelectedIndices: [2],
      draftSelectedIndices: [2],
    },
  )
})


test("viewer interaction state is isolated by structure and view revision", () => {
  const storage = memoryStorage()
  saveViewerSessionState(storage, "structure-a", "revision-a", state)

  assert.equal(
    loadViewerSessionState(storage, "structure-b", "revision-a"),
    null,
  )
  assert.equal(
    loadViewerSessionState(storage, "structure-a", "revision-b"),
    null,
  )
})


test("malformed or unavailable browser storage fails closed", () => {
  const malformed = {
    getItem() {
      return "not-json"
    },
  }
  const unavailable = {
    getItem() {
      throw new Error("blocked")
    },
    setItem() {
      throw new Error("blocked")
    },
  }

  assert.equal(
    loadViewerSessionState(malformed, "structure-a", "revision-a"),
    null,
  )
  assert.equal(
    loadViewerSessionState(unavailable, "structure-a", "revision-a"),
    null,
  )
  assert.doesNotThrow(
    () => saveViewerSessionState(
      unavailable,
      "structure-a",
      "revision-a",
      state,
    ),
  )
})


test("onRender is wired through cache loading and reconciliation before Plotly.react", async () => {
  const mainSource = await readFile(new URL("./main.mjs", import.meta.url), "utf8")

  assert.match(mainSource, /loadViewerSessionState\(/u)
  assert.match(mainSource, /reconcileViewerSessionState\(/u)
  assert.match(
    mainSource,
    /async function onRender[\s\S]*loadViewerSessionState\([\s\S]*reconcileViewerSessionState\([\s\S]*Plotly\.react\(/u,
  )
})
