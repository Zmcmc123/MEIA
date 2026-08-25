export function createIdleTask({
  delayMs,
  setTimer = globalThis.setTimeout,
  clearTimer = globalThis.clearTimeout,
  onError = () => {},
}) {
  if (typeof delayMs !== "number" || !Number.isFinite(delayMs) || delayMs < 0) {
    throw new TypeError("delayMs must be a non-negative finite number")
  }
  if (
    typeof setTimer !== "function"
    || typeof clearTimer !== "function"
    || typeof onError !== "function"
  ) {
    throw new TypeError("idle scheduler callbacks must be functions")
  }
  let timerId = null

  const schedule = task => {
    if (typeof task !== "function") {
      throw new TypeError("idle task must be a function")
    }
    if (timerId !== null) {
      clearTimer(timerId)
    }
    timerId = setTimer(async () => {
      timerId = null
      try {
        await task()
      } catch (error) {
        onError(error)
      }
    }, delayMs)
  }
  schedule.cancel = () => {
    if (timerId !== null) {
      clearTimer(timerId)
      timerId = null
    }
  }
  return schedule
}
