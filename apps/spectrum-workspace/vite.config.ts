import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  // Use absolute paths for Tauri's tauri://localhost protocol
  // Tauri 2.0 serves bundled assets from tauri://localhost/ which requires absolute paths
  base: "/",
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
      "@components": path.resolve(__dirname, "./src/components"),
      "@features": path.resolve(__dirname, "./src/features"),
      "@hooks": path.resolve(__dirname, "./src/hooks"),
      "@lib": path.resolve(__dirname, "./src/lib"),
      "@stores": path.resolve(__dirname, "./src/stores"),
      "@types": path.resolve(__dirname, "./src/types"),
      "@utils": path.resolve(__dirname, "./src/utils"),
    },
  },
  // Vitest configuration
  test: {
    globals: true,
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    include: ["src/**/*.{test,spec}.{js,mjs,cjs,ts,mts,cts,jsx,tsx}"],
    coverage: {
      provider: "v8",
      reporter: ["text", "json", "html"],
    },
    alias: {
      // Mock monaco-editor in tests
      "monaco-editor": path.resolve(
        __dirname,
        "./src/test/__mocks__/monaco-editor.ts"
      ),
    },
  },
  server: {
    port: 16000,
    proxy: {
      "/api": {
        target: "http://localhost:16080",
        changeOrigin: true,
      },
      "/graphql": {
        target: "http://localhost:16080",
        changeOrigin: true,
      },
      "/ws": {
        target: "ws://localhost:16080",
        ws: true,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks: {
          "react-vendor": ["react", "react-dom", "react-router-dom"],
          "three-vendor": ["three", "@react-three/fiber", "@react-three/drei"],
          "radix-vendor": [
            "@radix-ui/react-dialog",
            "@radix-ui/react-dropdown-menu",
            "@radix-ui/react-tabs",
            "@radix-ui/react-tooltip",
          ],
          "monaco-vendor": ["@monaco-editor/react"],
        },
      },
    },
  },
  optimizeDeps: {
    include: ["react", "react-dom", "three"],
  },
});
