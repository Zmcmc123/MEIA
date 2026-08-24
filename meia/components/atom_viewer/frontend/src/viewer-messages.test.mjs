import assert from "node:assert/strict"
import { readFile } from "node:fs/promises"
import test from "node:test"

import {
  formatViewerMessage,
  normalizeViewerMessages,
  replaceViewerMessages,
} from "./viewer-messages.mjs"


const messages = {
  "camera.apply": "Apply Current View",
  "selection.count.one": "Temporary selection: {count} atom{pending}",
  "selection.count.other": "Temporary selection: {count} atoms{pending}",
}


test("viewer messages reject a missing required key", () => {
  assert.throws(
    () => normalizeViewerMessages(messages),
    /missing viewer message/,
  )
})


test("viewer formatter selects English singular and plural templates", () => {
  const bundle = {
    "selection.count.one": "Temporary selection: {count} atom{pending}",
    "selection.count.other": "Temporary selection: {count} atoms{pending}",
  }

  assert.equal(
    formatViewerMessage(bundle, "selection.count", {count: 1, pending: ""}, "en"),
    "Temporary selection: 1 atom",
  )
  assert.equal(
    formatViewerMessage(bundle, "selection.count", {count: 2, pending: ""}, "en"),
    "Temporary selection: 2 atoms",
  )
})


test("message-only replacement preserves interaction state", () => {
  const camera = {eye: {x: 2, y: 1, z: 1}}
  const state = {
    draftCamera: camera,
    draftAspectRatio: {x: 2, y: 2, z: 2},
    selectionModeActive: true,
    draftSelectedIndices: [1, 3],
    waitingForPython: true,
    waitingForSelection: false,
    locale: "zh-CN",
    messages: {old: "旧"},
  }

  const updated = replaceViewerMessages(state, "en", {new: "new"})

  assert.equal(updated.locale, "en")
  assert.deepEqual(updated.messages, {new: "new"})
  assert.equal(updated.draftCamera, camera)
  assert.equal(updated.draftAspectRatio, state.draftAspectRatio)
  assert.equal(updated.draftSelectedIndices, state.draftSelectedIndices)
  assert.equal(updated.selectionModeActive, true)
  assert.equal(updated.waitingForPython, true)
  assert.equal(updated.waitingForSelection, false)
})


test("frontend production text has no hardcoded Han runtime strings", async () => {
  const html = await readFile(new URL("../index.html", import.meta.url), "utf8")
  const sources = await Promise.all([
    readFile(new URL("./main.mjs", import.meta.url), "utf8"),
    readFile(new URL("./camera-controls.mjs", import.meta.url), "utf8"),
  ])
  const runtimeSource = sources
    .map(source => source.replace(/^\s*\/\/.*$/gm, ""))
    .join("\n")

  assert.doesNotMatch(html, /[\u3400-\u9fff]/u)
  assert.doesNotMatch(runtimeSource, /[\u3400-\u9fff]/u)
})
