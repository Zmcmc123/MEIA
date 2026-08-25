import assert from "node:assert/strict"
import test from "node:test"

import { createIdleTask } from "./idle-scheduler.mjs"


test("idle task keeps only the final callback and runs it at 80 ms", async () => {
  const timers = []
  const cleared = new Set()
  const calls = []
  const schedule = createIdleTask({
    delayMs: 80,
    setTimer(callback, delay) {
      timers.push({callback, delay})
      return timers.length - 1
    },
    clearTimer(timerId) {
      cleared.add(timerId)
    },
  })

  schedule(() => calls.push("first"))
  schedule(() => calls.push("second"))
  schedule(() => calls.push("final"))

  assert.deepEqual(timers.map(timer => timer.delay), [80, 80, 80])
  assert.deepEqual([...cleared], [0, 1])
  await timers[2].callback()
  assert.deepEqual(calls, ["final"])
})


test("idle task reports errors and cancel clears pending work", async () => {
  const timers = []
  const cleared = []
  const errors = []
  const schedule = createIdleTask({
    delayMs: 120,
    setTimer(callback) {
      timers.push(callback)
      return timers.length - 1
    },
    clearTimer(timerId) {
      cleared.push(timerId)
    },
    onError(error) {
      errors.push(error.message)
    },
  })

  schedule(() => {
    throw new Error("restore failed")
  })
  await timers[0]()
  assert.deepEqual(errors, ["restore failed"])

  schedule(() => {})
  schedule.cancel()
  assert.deepEqual(cleared, [1])
})
