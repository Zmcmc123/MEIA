import test from "node:test"
import assert from "node:assert/strict"

import {
  normalizeBrowserLocale,
  readStoredLocale,
  resolveLocalePreference,
  writeStoredLocale,
} from "./locale-preference.mjs"


test("Chinese browser variants normalize to zh-CN", () => {
  for (const value of ["zh", "zh-CN", "zh-SG", "zh-TW", "zh-HK"]) {
    assert.equal(normalizeBrowserLocale(value), "zh-CN")
  }
})

test("non-Chinese browser locales normalize to English", () => {
  assert.equal(normalizeBrowserLocale("en-US"), "en")
  assert.equal(normalizeBrowserLocale("de-DE"), "en")
})

test("missing or malformed browser locales use the Chinese fallback", () => {
  assert.equal(normalizeBrowserLocale(null), "zh-CN")
  assert.equal(normalizeBrowserLocale(""), "zh-CN")
})

test("stored manual preference wins over browser language", () => {
  assert.deepEqual(
    resolveLocalePreference("en", "zh-CN"),
    {locale: "en", source: "stored"},
  )
})

test("storage helpers tolerate unavailable storage", () => {
  const storage = {
    getItem() { throw new Error("blocked") },
    setItem() { throw new Error("blocked") },
  }
  assert.equal(readStoredLocale(storage, "meia.locale"), null)
  assert.equal(writeStoredLocale(storage, "meia.locale", "en"), false)
})
