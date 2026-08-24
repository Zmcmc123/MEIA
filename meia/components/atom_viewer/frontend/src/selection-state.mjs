function nonEmptyString(value, name) {
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new Error(`${name} must be a non-empty string`)
  }
  return value
}


function transformVector(matrix, vector) {
  if (!Array.isArray(matrix) || matrix.length !== 16) {
    throw new Error("camera matrix must contain 16 numbers")
  }
  const output = [0, 0, 0, 0]
  for (let i = 0; i < 4; i += 1) {
    for (let j = 0; j < 4; j += 1) {
      output[j] += matrix[4 * i + j] * vector[i]
    }
  }
  if (!output.every(Number.isFinite)) {
    throw new Error("camera projection produced a non-finite coordinate")
  }
  return output
}


function projectPoint(cameraParams, point) {
  const modelPoint = transformVector(
    cameraParams?.model,
    [point[0], point[1], point[2], 1],
  )
  const viewPoint = transformVector(cameraParams?.view, modelPoint)
  return transformVector(cameraParams?.projection, viewPoint)
}


export function normalizeAtomIndices(value) {
  if (!Array.isArray(value)) {
    throw new Error("atom indices must be an array")
  }
  if (value.some(index => !Number.isInteger(index) || index < 0)) {
    throw new Error("atom index must be a non-negative integer")
  }
  return [...new Set(value)].sort((left, right) => left - right)
}


export function toggleAtomIndex(current, atomIndex) {
  const selected = new Set(normalizeAtomIndices(current))
  const [normalized] = normalizeAtomIndices([atomIndex])
  if (selected.has(normalized)) {
    selected.delete(normalized)
  } else {
    selected.add(normalized)
  }
  return normalizeAtomIndices([...selected])
}


export function addAtomIndices(current, additions) {
  return normalizeAtomIndices([
    ...normalizeAtomIndices(current),
    ...normalizeAtomIndices(additions),
  ])
}


export function atomsInsideRectangle(projectedAtoms, rectangle) {
  const left = Math.min(rectangle.x0, rectangle.x1)
  const right = Math.max(rectangle.x0, rectangle.x1)
  const top = Math.min(rectangle.y0, rectangle.y1)
  const bottom = Math.max(rectangle.y0, rectangle.y1)
  return normalizeAtomIndices(
    projectedAtoms
      .filter(atom => (
        atom.x >= left && atom.x <= right && atom.y >= top && atom.y <= bottom
      ))
      .map(atom => atom.atomIndex),
  )
}


export function nearestAtomAtPoint(projectedAtoms, point, hitRadius) {
  if (typeof hitRadius !== "number" || !Number.isFinite(hitRadius) || hitRadius < 0) {
    throw new Error("hit radius must be a non-negative finite number")
  }
  const maximumDistanceSquared = hitRadius * hitRadius
  let nearest = null
  let nearestDistanceSquared = maximumDistanceSquared
  for (const atom of projectedAtoms) {
    const distanceSquared = (atom.x - point.x) ** 2 + (atom.y - point.y) ** 2
    if (
      distanceSquared < nearestDistanceSquared
      || (
        distanceSquared === nearestDistanceSquared
        && nearest !== null
        && atom.depth < nearest.depth
      )
    ) {
      nearest = atom
      nearestDistanceSquared = distanceSquared
    }
  }
  return nearest
}


export function projectAtomScreenPositions(graph, overlayBounds) {
  const scene = graph?._fullLayout?.scene?._scene
  const atomTrace = Object.values(scene?.traces ?? {}).find(
    trace => trace?.data?.meta?.meia_role === "atoms",
  )
  if (!scene || !atomTrace) {
    throw new Error("Plotly atom scene is unavailable")
  }
  const points = atomTrace.dataPoints
  const identities = atomTrace.data.customdata
  if (!Array.isArray(points) || !Array.isArray(identities)) {
    throw new Error("Plotly atom trace is incomplete")
  }
  const sceneBounds = scene.container.getBoundingClientRect()
  const projected = []
  for (let index = 0; index < points.length; index += 1) {
    const selection = atomSelectionFromPoint({
      data: { meta: { meia_role: "atoms" } },
      customdata: identities[index],
    })
    if (selection === null) {
      throw new Error("Plotly atom identity is invalid")
    }
    const clip = projectPoint(scene.glplot?.cameraParams, points[index])
    if (Math.abs(clip[3]) <= Number.EPSILON) {
      continue
    }
    const normalizedX = clip[0] / clip[3]
    const normalizedY = clip[1] / clip[3]
    projected.push({
      atomIndex: selection.atomIndex,
      atomSymbol: selection.atomSymbol,
      x: sceneBounds.left - overlayBounds.left
        + (0.5 + 0.5 * normalizedX) * sceneBounds.width,
      y: sceneBounds.top - overlayBounds.top
        + (0.5 - 0.5 * normalizedY) * sceneBounds.height,
      depth: clip[2] / clip[3],
    })
  }
  return projected
}


export function atomSelectionFromPoint(point) {
  if (point?.data?.meta?.meia_role !== "atoms") {
    return null
  }
  const customdata = point?.customdata
  if (!Array.isArray(customdata) || customdata.length < 2) {
    return null
  }
  const [atomIndex, atomSymbol] = customdata
  if (!Number.isInteger(atomIndex) || atomIndex < 0) {
    return null
  }
  if (typeof atomSymbol !== "string" || atomSymbol.trim().length === 0) {
    return null
  }
  return { atomIndex, atomSymbol }
}


export function makeAtomSelectionEvent(
  structureId,
  atomIndex,
  atomSymbol,
  eventId,
) {
  nonEmptyString(structureId, "structureId")
  nonEmptyString(eventId, "eventId")
  if (!Number.isInteger(atomIndex) || atomIndex < 0) {
    throw new Error("atomIndex must be a non-negative integer")
  }
  nonEmptyString(atomSymbol, "atomSymbol")
  return {
    event_type: "select_atom",
    event_id: eventId,
    structure_id: structureId,
    atom_index: atomIndex,
    atom_symbol: atomSymbol,
  }
}


export function makeAtomSelectionBatchEvent(
  structureId,
  atomIndices,
  eventId,
) {
  nonEmptyString(structureId, "structureId")
  nonEmptyString(eventId, "eventId")
  return {
    event_type: "select_atoms",
    event_id: eventId,
    structure_id: structureId,
    atom_indices: normalizeAtomIndices(atomIndices),
  }
}
