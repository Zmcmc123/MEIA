const SUPPORTED_LOCALES = new Set(["zh-CN", "en"])


export function normalizeBrowserLocale(value) {
  if (typeof value !== "string" || value.trim() === "") {
    return "zh-CN"
  }
  return value.trim().toLowerCase().startsWith("zh") ? "zh-CN" : "en"
}


export function readStoredLocale(storage, storageKey) {
  try {
    const value = storage.getItem(storageKey)
    return SUPPORTED_LOCALES.has(value) ? value : null
  } catch (_error) {
    return null
  }
}


export function writeStoredLocale(storage, storageKey, locale) {
  try {
    storage.setItem(storageKey, locale)
  } catch (_error) {
    return false
  }
  return true
}


export function resolveLocalePreference(storedLocale, browserLocale) {
  if (SUPPORTED_LOCALES.has(storedLocale)) {
    return {locale: storedLocale, source: "stored"}
  }
  return {locale: normalizeBrowserLocale(browserLocale), source: "browser"}
}


async function initializeBrowserComponent() {
  const { Streamlit } = await import("streamlit-component-lib")
  let previousValue = null

  function emitPreference(event) {
    const args = event.detail.args ?? {}
    if (args.storage_key !== "meia.locale") {
      throw new Error("locale preference storage key is invalid")
    }
    const persistLocale = args.persist_locale
    if (persistLocale !== null && persistLocale !== undefined) {
      if (!SUPPORTED_LOCALES.has(persistLocale)) {
        throw new Error("locale preference persist_locale is invalid")
      }
      writeStoredLocale(window.localStorage, args.storage_key, persistLocale)
    }
    const preference = resolveLocalePreference(
      readStoredLocale(window.localStorage, args.storage_key),
      window.navigator?.language,
    )
    const serialized = JSON.stringify(preference)
    if (serialized !== previousValue) {
      previousValue = serialized
      Streamlit.setComponentValue(preference)
    }
    Streamlit.setFrameHeight(0)
  }

  Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, emitPreference)
  Streamlit.setComponentReady()
  Streamlit.setFrameHeight(0)
}


if (typeof window !== "undefined" && typeof document !== "undefined") {
  initializeBrowserComponent()
}
