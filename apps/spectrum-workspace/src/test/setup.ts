/**
 * Vitest Test Setup
 *
 * Global test configuration and mocks for the NEURECTOMY IDE tests.
 */

import { vi, beforeAll, afterEach } from "vitest";
import "@testing-library/jest-dom/vitest";

// Mock Tauri APIs
vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(),
}));

vi.mock("@tauri-apps/api/event", () => ({
  listen: vi.fn(() => Promise.resolve(() => {})),
  emit: vi.fn(),
}));

vi.mock("@tauri-apps/plugin-shell", () => ({
  Command: {
    create: vi.fn(),
  },
}));

vi.mock("@tauri-apps/api/window", () => ({
  getCurrentWindow: vi.fn(() => ({
    listen: vi.fn(() => Promise.resolve(() => {})),
  })),
}));

// Mock fetch for API calls
global.fetch = vi.fn();

// Mock crypto for SigmaVault tests
Object.defineProperty(global, "crypto", {
  value: {
    subtle: {
      digest: vi.fn().mockResolvedValue(new ArrayBuffer(32)),
      importKey: vi.fn().mockResolvedValue({}),
      encrypt: vi.fn().mockResolvedValue(new ArrayBuffer(64)),
      decrypt: vi.fn().mockResolvedValue(new ArrayBuffer(32)),
    },
    getRandomValues: vi.fn((arr: Uint8Array) => {
      for (let i = 0; i < arr.length; i++) {
        arr[i] = Math.floor(Math.random() * 256);
      }
      return arr;
    }),
  },
});

// Mock TextEncoder/TextDecoder
global.TextEncoder = class TextEncoder {
  encode(str: string): Uint8Array {
    return new Uint8Array(Buffer.from(str, "utf-8"));
  }
};

global.TextDecoder = class TextDecoder {
  decode(arr: Uint8Array): string {
    return Buffer.from(arr).toString("utf-8");
  }
};

// Reset mocks after each test
afterEach(() => {
  vi.clearAllMocks();
});

// Setup before all tests
beforeAll(() => {
  // Suppress console.log during tests unless debugging
  if (!process.env.DEBUG) {
    vi.spyOn(console, "log").mockImplementation(() => {});
  }
});
