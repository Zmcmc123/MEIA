import assert from "node:assert/strict"
import test from "node:test"


async function loadViewerStyle() {
  return import("./viewer-style.mjs")
}


test("scene zoom scales atom diameter and bond width together", async () => {
  const { viewerTraceStyleAtZoom } = await loadViewerStyle()
  const zoom = 2
  assert.deepEqual(
    viewerTraceStyleAtZoom({
      meta: {
        meia_role: "atoms",
        meia_base_marker_sizes: [6, 10],
      },
    }, zoom),
    { "marker.size": [12, 20] },
  )
  assert.deepEqual(
    viewerTraceStyleAtZoom({
      meta: {
        meia_role: "bonds",
        meia_base_line_width: 4,
      },
    }, zoom),
    { "line.width": 8 },
  )
})


test("selection highlight stays on the atom instead of making an outer ring", async () => {
  const { viewerTraceStyleAtZoom } = await loadViewerStyle()

  assert.deepEqual(
    viewerTraceStyleAtZoom({
      meta: {
        meia_role: "selection",
        meia_base_marker_sizes: [6, 10, 8],
        meia_source_atom_indices: [0, 1, 2],
      },
    }, 1.5, [1]),
    {
      "marker.size": [0, 15, 0],
      "marker.color": [
        "rgba(0,0,0,0)",
        "rgba(255,213,79,0.55)",
        "rgba(0,0,0,0)",
      ],
    },
  )
})


test("selection trace highlights every replica of a selected source atom", async () => {
  const { viewerTraceStyleAtZoom } = await loadViewerStyle()

  assert.deepEqual(
    viewerTraceStyleAtZoom({
      meta: {
        meia_role: "selection",
        meia_base_marker_sizes: [10, 12, 10, 12],
        meia_source_atom_indices: [0, 1, 0, 1],
      },
    }, 1, [0]),
    {
      "marker.size": [10, 0, 10, 0],
      "marker.color": [
        "rgba(255,213,79,0.55)",
        "rgba(0,0,0,0)",
        "rgba(255,213,79,0.55)",
        "rgba(0,0,0,0)",
      ],
    },
  )
})


test("replica scene zoom scales atoms, covalent layers, and hydrogen colors", async () => {
  const {
    plotlyAtomicUpdateForSingleTrace,
    viewerTraceStyleAtZoom,
  } = await loadViewerStyle()
  const camera = {
    eye: {x: 0.5, y: 0.5, z: 0.5},
    up: {x: 0, y: 0, z: 1},
    center: {x: 0, y: 0, z: 0},
    projection: {type: "orthographic"},
  }
  const aspectRatio = {x: 2, y: 4, z: 8}
  const traces = [
    {
      meta: {
        meia_role: "atoms",
        meia_base_marker_sizes: [6, 8, 6, 8],
      },
    },
    {meta: {meia_role: "bond_outlines", meia_base_line_width: 5}},
    {meta: {meia_role: "bonds", meia_base_line_width: 4}},
    {meta: {meia_role: "hydrogen_bonds", meia_base_line_width: 3}},
    {meta: {meia_role: "hydrogen_bonds", meia_base_line_width: 3}},
  ]

  const atomicUpdates = traces.map(trace => plotlyAtomicUpdateForSingleTrace(
    viewerTraceStyleAtZoom(trace, 1.5),
    camera,
    aspectRatio,
  ))

  assert.deepEqual(
    atomicUpdates.map(update => update.dataUpdate),
    [
      {"marker.size": [[9, 12, 9, 12]]},
      {"line.width": [7.5]},
      {"line.width": [6]},
      {"line.width": [4.5]},
      {"line.width": [4.5]},
    ],
  )
  for (const update of atomicUpdates) {
    assert.deepEqual(update.layoutUpdate, {
      "scene.camera": camera,
      "scene.aspectratio": aspectRatio,
      "scene.aspectmode": "manual",
    })
  }
})


test("per-point marker arrays stay nested for a single Plotly trace", async () => {
  const { plotlyUpdateForSingleTrace } = await loadViewerStyle()

  assert.deepEqual(
    plotlyUpdateForSingleTrace({
      "marker.size": [6, 10],
      "line.width": 4,
    }),
    {
      "marker.size": [[6, 10]],
      "line.width": [4],
    },
  )
})


test("trace style updates preserve the current camera and scene span", async () => {
  const { plotlyAtomicUpdateForSingleTrace } = await loadViewerStyle()
  const camera = {
    eye: {x: 0.5, y: 0.5, z: 0.5},
    up: {x: 0, y: 0, z: 1},
    center: {x: 0, y: 0, z: 0},
    projection: {type: "orthographic"},
  }
  const aspectRatio = {x: 1.5, y: 3, z: 6}

  assert.deepEqual(
    plotlyAtomicUpdateForSingleTrace(
      {"marker.size": [12, 20]},
      camera,
      aspectRatio,
    ),
    {
      dataUpdate: {"marker.size": [[12, 20]]},
      layoutUpdate: {
        "scene.camera": camera,
        "scene.aspectratio": aspectRatio,
        "scene.aspectmode": "manual",
      },
    },
  )
})
