import assert from "node:assert/strict"
import { readFile } from "node:fs/promises"
import test from "node:test"


test("latest frame task coalesces queued work and never overlaps", async () => {
  const { createLatestFrameTask } = await import("./frame-scheduler.mjs")
  const frames = []
  const calls = []
  let releaseRunning
  const running = new Promise(resolve => {
    releaseRunning = resolve
  })
  const schedule = createLatestFrameTask(callback => frames.push(callback))

  schedule(async () => calls.push("discarded"))
  schedule(async () => {
    calls.push("first")
    await running
  })

  assert.equal(frames.length, 1)
  const firstFrame = frames.shift()()
  await Promise.resolve()
  assert.deepEqual(calls, ["first"])

  schedule(async () => calls.push("second-discarded"))
  schedule(async () => calls.push("second"))
  assert.equal(frames.length, 0)

  releaseRunning()
  await firstFrame
  assert.equal(frames.length, 1)
  await frames.shift()()
  assert.deepEqual(calls, ["first", "second"])
})


test("latest frame task reports failures and remains reusable", async () => {
  const { createLatestFrameTask } = await import("./frame-scheduler.mjs")
  const frames = []
  const failures = []
  const schedule = createLatestFrameTask(
    callback => frames.push(callback),
    error => failures.push(error.message),
  )

  schedule(async () => {
    throw new Error("render failed")
  })
  await frames.shift()()
  assert.deepEqual(failures, ["render failed"])

  schedule(async () => {})
  assert.equal(frames.length, 1)
  await frames.shift()()
})


test("viewer coalesces projection and trace-style work by animation frame", async () => {
  const mainSource = await readFile(new URL("./main.mjs", import.meta.url), "utf8")

  assert.match(
    mainSource,
    /const scheduleProjectedAtomsRefresh = createLatestFrameTask/u,
  )
  assert.match(
    mainSource,
    /const scheduleViewerTraceStyleSync = createLatestFrameTask/u,
  )
  assert.match(
    mainSource,
    /shouldProjectSelectionAtoms\(batchSelectionEnabled, selectionModeActive\)/u,
  )
  assert.doesNotMatch(
    mainSource,
    /requestAnimationFrame\(refreshProjectedAtoms\)/u,
  )
  assert.match(
    mainSource,
    /scheduleViewerTraceStyleSync\(syncViewerTraceStyles\)/u,
  )
})
