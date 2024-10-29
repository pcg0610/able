import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "path";
import tsconfigPaths from "vite-tsconfig-paths";

// https://vite.dev/config/
export default defineConfig({
  resolve: {
    alias: [
      { find: "@", replacement: resolve(__dirname, "src") },
      { find: "@pages", replacement: resolve(__dirname, "src/pages") },
      {
        find: "@components",
        replacement: resolve(__dirname, "src/components"),
      },
      {
        find: "@widgets",
        replacement: resolve(__dirname, "src/widgets"),
      },
      { find: "@shared", replacement: resolve(__dirname, "src/shared") },
      {
        find: "@icons",
        replacement: resolve(__dirname, "src/assets/icons"),
      },
      {
        find: "@images",
        replacement: resolve(__dirname, "src/assets/images"),
      },
    ],
  },
  plugins: [react(), tsconfigPaths()],
});
