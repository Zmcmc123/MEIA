import { normalizeAtomIndices } from "./selection-state.mjs"


function finitePoint(point) {
  if (
    typeof point?.x !== "number"
    || !Number.isFinite(point.x)
    || typeof point?.y !== "number"
    || !Number.isFinite(point.y)
  ) {
    throw new Error("selection point must contain finite x and y")
  }
  return point
}


export class SelectionSpatialIndex {
  constructor(projectedAtoms, cellSize = 36) {
    if (!Array.isArray(projectedAtoms)) {
      throw new Error("projected atoms must be an array")
    }
    if (typeof cellSize !== "number" || !Number.isFinite(cellSize) || cellSize <= 0) {
      throw new Error("selection cell size must be a positive finite number")
    }
    this.cellSize = cellSize
    this.cells = new Map()
    this.lastVisitedCellCount = 0
    for (const atom of projectedAtoms) {
      finitePoint(atom)
      if (!Number.isInteger(atom?.atomIndex) || atom.atomIndex < 0) {
        throw new Error("projected atom identity must be a non-negative integer")
      }
      const key = this.#keyFor(atom.x, atom.y)
      const values = this.cells.get(key) ?? []
      values.push(atom)
      this.cells.set(key, values)
    }
  }

  #keyFor(x, y) {
    return `${Math.floor(x / this.cellSize)},${Math.floor(y / this.cellSize)}`
  }

  #visitRectangle(left, top, right, bottom, visitor) {
    const startX = Math.floor(left / this.cellSize)
    const endX = Math.floor(right / this.cellSize)
    const startY = Math.floor(top / this.cellSize)
    const endY = Math.floor(bottom / this.cellSize)
    this.lastVisitedCellCount = 0
    for (let cellX = startX; cellX <= endX; cellX += 1) {
      for (let cellY = startY; cellY <= endY; cellY += 1) {
        this.lastVisitedCellCount += 1
        for (const atom of this.cells.get(`${cellX},${cellY}`) ?? []) {
          visitor(atom)
        }
      }
    }
  }

  nearest(point, hitRadius) {
    finitePoint(point)
    if (
      typeof hitRadius !== "number"
      || !Number.isFinite(hitRadius)
      || hitRadius < 0
    ) {
      throw new Error("hit radius must be a non-negative finite number")
    }
    const maximumDistanceSquared = hitRadius * hitRadius
    let nearest = null
    let nearestDistanceSquared = maximumDistanceSquared
    this.#visitRectangle(
      point.x - hitRadius,
      point.y - hitRadius,
      point.x + hitRadius,
      point.y + hitRadius,
      atom => {
        const distanceSquared = (atom.x - point.x) ** 2 + (atom.y - point.y) ** 2
        if (
          distanceSquared < nearestDistanceSquared
          || (
            distanceSquared === nearestDistanceSquared
            && (nearest === null || atom.depth < nearest.depth)
          )
        ) {
          nearest = atom
          nearestDistanceSquared = distanceSquared
        }
      },
    )
    return nearest
  }

  insideRectangle(rectangle) {
    const first = finitePoint({x: rectangle?.x0, y: rectangle?.y0})
    const second = finitePoint({x: rectangle?.x1, y: rectangle?.y1})
    const left = Math.min(first.x, second.x)
    const right = Math.max(first.x, second.x)
    const top = Math.min(first.y, second.y)
    const bottom = Math.max(first.y, second.y)
    const selected = []
    this.#visitRectangle(left, top, right, bottom, atom => {
      if (atom.x >= left && atom.x <= right && atom.y >= top && atom.y <= bottom) {
        selected.push(atom.atomIndex)
      }
    })
    return normalizeAtomIndices(selected)
  }
}
