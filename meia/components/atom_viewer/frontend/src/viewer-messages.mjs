const REQUIRED_VIEWER_MESSAGE_KEYS = Object.freeze([
  "camera.apply",
  "camera.applied",
  "camera.waiting",
  "controls.aria",
  "angle_step.label",
  "orbit.aria",
  "orbit.up",
  "orbit.left",
  "orbit.down",
  "orbit.right",
  "axis.aria",
  "selection.tools_aria",
  "selection.mode.on",
  "selection.mode.off",
  "selection.unavailable",
  "selection.hint.active",
  "selection.hint.inactive",
  "selection.count.one",
  "selection.count.other",
  "selection.pending",
  "selection.clear",
  "selection.confirm",
  "selection.confirming",
  "canvas.aria",
  "selection.overlay_aria",
  "error.zoom",
  "error.camera",
  "error.load",
  "error.adjust",
])


export function normalizeViewerLocale(value) {
  if (value === "zh-CN" || value === "en") {
    return value
  }
  throw new Error(`unsupported viewer locale: ${String(value)}`)
}


export function normalizeViewerMessages(value) {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("viewer messages must be an object")
  }
  const normalized = {}
  for (const key of REQUIRED_VIEWER_MESSAGE_KEYS) {
    const text = value[key]
    if (typeof text !== "string" || text.length === 0) {
      throw new Error(`missing viewer message: ${key}`)
    }
    normalized[key] = text
  }
  return Object.freeze(normalized)
}


export function formatViewerMessage(
  messages,
  key,
  params = {},
  locale = "zh-CN",
) {
  let resolvedKey = key
  if (!(resolvedKey in messages) && Object.hasOwn(params, "count")) {
    resolvedKey = (
      normalizeViewerLocale(locale) === "en" && params.count === 1
        ? `${key}.one`
        : `${key}.other`
    )
  }
  const template = messages[resolvedKey]
  if (typeof template !== "string") {
    throw new Error(`missing viewer message: ${resolvedKey}`)
  }
  return template.replace(/\{([A-Za-z0-9_]+)\}/g, (_match, name) => {
    if (!Object.hasOwn(params, name)) {
      throw new Error(`missing viewer message parameter: ${name}`)
    }
    return String(params[name])
  })
}


export function replaceViewerMessages(state, locale, messages) {
  if (state === null || typeof state !== "object" || Array.isArray(state)) {
    throw new Error("viewer state must be an object")
  }
  return {
    ...state,
    locale: normalizeViewerLocale(locale),
    messages,
  }
}
