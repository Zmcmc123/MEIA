import assert from "node:assert/strict"
import test from "node:test"

import {
  aspectRatioZoomScale,
  mergeRelayoutAspectRatio,
  normalizeAspectRatio,
  zoomAspectRatioForWheel,
} from "./viewport-zoom.mjs"


test("orthographic zoom is measured from the scene span, not camera distance", () => {
  const base = normalizeAspectRatio({x: 1, y: 2, z: 4})
  const zoomed = normalizeAspectRatio({x: 1.5, y: 3, z: 6})

  assert.equal(aspectRatioZoomScale(base, zoomed), 1.5)
})


test("trackpad wheel zoom changes all three scene spans together", () => {
  const initial = normalizeAspectRatio({x: 1, y: 2, z: 4})

  assert.deepEqual(zoomAspectRatioForWheel(initial, -8), {
    x: 1.1,
    y: 2.2,
    z: 4.4,
  })
  assert.deepEqual(zoomAspectRatioForWheel(initial, 8), {
    x: 1 / 1.1,
    y: 2 / 1.1,
    z: 4 / 1.1,
  })
})


test("native Plotly relayout updates are merged into viewport zoom state", () => {
  const initial = normalizeAspectRatio({x: 1, y: 2, z: 4})

  assert.deepEqual(mergeRelayoutAspectRatio(initial, {
    "scene.aspectratio": {x: 1.2, y: 2.4, z: 4.8},
  }), {x: 1.2, y: 2.4, z: 4.8})
  assert.deepEqual(mergeRelayoutAspectRatio(initial, {
    "scene.aspectratio.x": 0.9,
    "scene.aspectratio.y": 1.8,
    "scene.aspectratio.z": 3.6,
  }), {x: 0.9, y: 1.8, z: 3.6})
})
