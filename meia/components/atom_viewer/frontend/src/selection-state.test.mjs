import assert from "node:assert/strict"
import test from "node:test"

import {
  addAtomIndices,
  atomsInsideRectangle,
  atomSelectionFromPoint,
  makeAtomSelectionBatchEvent,
  makeAtomSelectionEvent,
  nearestAtomAtPoint,
  normalizeAtomIndices,
  projectAtomScreenPositions,
  toggleAtomIndex,
} from "./selection-state.mjs"


test("atom trace point is converted to a typed selection", () => {
  assert.deepEqual(
    atomSelectionFromPoint({
      data: { meta: { meia_role: "atoms" } },
      customdata: [2, "Si"],
    }),
    { atomIndex: 2, atomSymbol: "Si" },
  )
})


test("replica customdata selects the original atom identity", () => {
  assert.deepEqual(
    atomSelectionFromPoint({
      data: { meta: { meia_role: "atoms" } },
      customdata: [4, "O", [-1, 0, 0], [-1, 0, 0]],
    }),
    { atomIndex: 4, atomSymbol: "O" },
  )
})


test("non-atom traces and malformed customdata are ignored", () => {
  assert.equal(atomSelectionFromPoint({}), null)
  assert.equal(atomSelectionFromPoint({ customdata: [2, "Si"] }), null)
  const atomPoint = (customdata) => ({
    data: { meta: { meia_role: "atoms" } },
    customdata,
  })
  assert.equal(atomSelectionFromPoint(atomPoint([1])), null)
  assert.equal(atomSelectionFromPoint(atomPoint([true, "O"])), null)
  assert.equal(atomSelectionFromPoint(atomPoint([-1, "O"])), null)
  assert.equal(atomSelectionFromPoint(atomPoint([1, ""])), null)
  assert.equal(
    atomSelectionFromPoint({
      data: { meta: { meia_role: "bond" } },
      customdata: [1, "O"],
    }),
    null,
  )
})


test("selection event carries structure identity and zero-based atom index", () => {
  assert.deepEqual(
    makeAtomSelectionEvent("structure-a", 1, "O", "event-1"),
    {
      event_type: "select_atom",
      event_id: "event-1",
      structure_id: "structure-a",
      atom_index: 1,
      atom_symbol: "O",
    },
  )
})


test("selection event rejects invalid identities", () => {
  assert.throws(
    () => makeAtomSelectionEvent("", 1, "O", "event-1"),
    /structureId/,
  )
  assert.throws(
    () => makeAtomSelectionEvent("structure-a", 1, "O", ""),
    /eventId/,
  )
})


test("local atom selection is canonical and click toggles without an event", () => {
  assert.deepEqual(normalizeAtomIndices([3, 1, 3]), [1, 3])
  assert.deepEqual(toggleAtomIndex([1, 3], 1), [3])
  assert.deepEqual(toggleAtomIndex([1, 3], 2), [1, 2, 3])
  assert.deepEqual(addAtomIndices([1], [2, 0, 1]), [0, 1, 2])
  assert.throws(() => normalizeAtomIndices([true]), /atom index/)
})


test("atom centers inside a screen rectangle are selected inclusively", () => {
  const projected = [
    { atomIndex: 0, atomSymbol: "H", x: 40, y: 50, depth: 0.1 },
    { atomIndex: 1, atomSymbol: "O", x: 100, y: 100, depth: 0.2 },
    { atomIndex: 2, atomSymbol: "Si", x: 160, y: 130, depth: 0.3 },
  ]

  assert.deepEqual(
    atomsInsideRectangle(projected, { x0: 100, y0: 40, x1: 30, y1: 100 }),
    [0, 1],
  )
})


test("screen click selects only the nearest atom within the hit radius", () => {
  const projected = [
    { atomIndex: 2, atomSymbol: "O", x: 60, y: 50, depth: 0.2 },
    { atomIndex: 4, atomSymbol: "C", x: 52, y: 51, depth: 0.1 },
  ]

  assert.equal(
    nearestAtomAtPoint(projected, { x: 50, y: 50 }, 12)?.atomIndex,
    4,
  )
  assert.equal(nearestAtomAtPoint(projected, { x: 10, y: 10 }, 12), null)
})


test("Plotly camera matrices project atom trace centers into overlay pixels", () => {
  const identity = [
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ]
  const graph = {
    _fullLayout: {
      scene: {
        _scene: {
          container: {
            getBoundingClientRect: () => ({
              left: 10,
              top: 20,
              width: 200,
              height: 100,
            }),
          },
          glplot: {
            cameraParams: {
              model: identity,
              view: identity,
              projection: identity,
            },
          },
          traces: {
            atoms: {
              data: {
                meta: { meia_role: "atoms" },
                customdata: [[4, "O"], [7, "H"]],
              },
              dataPoints: [[0, 0, 0], [1, 1, 0]],
            },
          },
        },
      },
    },
  }

  assert.deepEqual(
    projectAtomScreenPositions(graph, { left: 0, top: 0 }),
    [
      { atomIndex: 4, atomSymbol: "O", x: 110, y: 70, depth: 0 },
      { atomIndex: 7, atomSymbol: "H", x: 210, y: 20, depth: 0 },
    ],
  )
})


test("batch confirmation event carries the complete canonical index set", () => {
  assert.deepEqual(
    makeAtomSelectionBatchEvent("structure-a", [3, 1, 3], "event-2"),
    {
      event_type: "select_atoms",
      event_id: "event-2",
      structure_id: "structure-a",
      atom_indices: [1, 3],
    },
  )
  assert.throws(
    () => makeAtomSelectionBatchEvent("structure-a", [false], "event-2"),
    /atom index/,
  )
})
