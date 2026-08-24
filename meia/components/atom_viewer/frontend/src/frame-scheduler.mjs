export function createLatestFrameTask(requestFrame, onError = () => {}) {
  if (typeof requestFrame !== "function") {
    throw new TypeError("requestFrame must be a function")
  }
  if (typeof onError !== "function") {
    throw new TypeError("onError must be a function")
  }

  let latestTask = null
  let frameRequested = false
  let running = false

  function requestNextFrame() {
    if (frameRequested || running || latestTask === null) {
      return
    }
    frameRequested = true
    requestFrame(async () => {
      frameRequested = false
      running = true
      const task = latestTask
      latestTask = null
      try {
        await task()
      } catch (error) {
        onError(error)
      } finally {
        running = false
        requestNextFrame()
      }
    })
  }

  return task => {
    if (typeof task !== "function") {
      throw new TypeError("frame task must be a function")
    }
    latestTask = task
    requestNextFrame()
  }
}
