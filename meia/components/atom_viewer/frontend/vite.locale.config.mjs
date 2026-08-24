import { defineConfig } from "vite"


export default defineConfig({
  root: "locale",
  base: "./",
  build: {
    outDir: "../dist/locale",
    emptyOutDir: true,
    sourcemap: false,
  },
})
