import assert from "node:assert/strict"
import test from "node:test"

import { SelectionSpatialIndex } from "./selection-spatial-index.mjs"


test("spatial index returns the closest front-most atom from touched cells", () => {
  const index = new SelectionSpatialIndex([
    {atomIndex: 2, atomSymbol: "O", x: 34, y: 34, depth: 0.2},
    {atomIndex: 4, atomSymbol: "C", x: 34, y: 34, depth: 0.1},
    {atomIndex: 8, atomSymbol: "Si", x: 200, y: 200, depth: 0.0},
  ])

  assert.equal(index.nearest({x: 35, y: 35}, 6)?.atomIndex, 4)
  assert.equal(index.lastVisitedCellCount, 4)
})


test("small and full rectangles visit bounded cells and canonicalize sources", () => {
  const index = new SelectionSpatialIndex([
    {atomIndex: 0, atomSymbol: "H", x: 10, y: 10, depth: 0.1},
    {atomIndex: 1, atomSymbol: "O", x: 50, y: 50, depth: 0.2},
    {atomIndex: 0, atomSymbol: "H", x: 90, y: 90, depth: 0.3},
    {atomIndex: 2, atomSymbol: "Si", x: 130, y: 130, depth: 0.4},
  ], 36)

  assert.deepEqual(
    index.insideRectangle({x0: 0, y0: 0, x1: 60, y1: 60}),
    [0, 1],
  )
  assert.equal(index.lastVisitedCellCount, 4)
  assert.deepEqual(
    index.insideRectangle({x0: 0, y0: 0, x1: 200, y1: 200}),
    [0, 1, 2],
  )
})
