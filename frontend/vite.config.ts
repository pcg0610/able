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
      { find: "@hooks", replacement: resolve(__dirname, "src/hooks") },
      {
        find: "@services",
        replacement: resolve(__dirname, "src/services"),
      },
      { find: "@shared", replacement: resolve(__dirname, "src/shared") },
      { find: "@stores", replacement: resolve(__dirname, "src/stores") },
      {
        find: "@icons",
        replacement: resolve(__dirname, "src/assets/icons"),
      },
      {
        find: "@images",
        replacement: resolve(__dirname, "src/assets/images"),
      },
      {
        find: "@fonts",
        replacement: resolve(__dirname, "src/assets/fonts"),
      },
    ],
  },
  plugins: [react(), tsconfigPaths()],
});
