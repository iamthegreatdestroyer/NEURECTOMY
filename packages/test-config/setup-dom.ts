/**
 * @neurectomy/test-config - DOM Test Setup
 *
 * This file runs before all tests in jsdom environment packages.
 * Extends base setup with DOM-specific utilities and mocks.
 */

import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach, vi } from "vitest";

// Import base setup
import "./setup";

// ============================================================================
// DOM Cleanup
// ============================================================================

afterEach(() => {
  // Cleanup React Testing Library after each test
  cleanup();
});

// ============================================================================
// Browser API Mocks
// ============================================================================

// Mock window.matchMedia
Object.defineProperty(window, "matchMedia", {
  writable: true,
  value: vi.fn().mockImplementation((query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(), // deprecated
    removeListener: vi.fn(), // deprecated
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
});

// Mock window.ResizeObserver
class MockResizeObserver {
  observe = vi.fn();
  unobserve = vi.fn();
  disconnect = vi.fn();
}
Object.defineProperty(window, "ResizeObserver", {
  writable: true,
  value: MockResizeObserver,
});

// Mock window.IntersectionObserver
class MockIntersectionObserver {
  constructor(
    private callback: IntersectionObserverCallback,
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    _options?: IntersectionObserverInit
  ) {}
  root = null;
  rootMargin = "";
  thresholds = [];
  observe = vi.fn();
  unobserve = vi.fn();
  disconnect = vi.fn();
  takeRecords = vi.fn().mockReturnValue([]);

  // Utility to trigger intersection
  trigger(entries: Partial<IntersectionObserverEntry>[]) {
    this.callback(entries as IntersectionObserverEntry[], this);
  }
}
Object.defineProperty(window, "IntersectionObserver", {
  writable: true,
  value: MockIntersectionObserver,
});

// Mock window.scrollTo
Object.defineProperty(window, "scrollTo", {
  writable: true,
  value: vi.fn(),
});

// Mock element.scrollIntoView
Element.prototype.scrollIntoView = vi.fn();

// Mock window.getComputedStyle (basic implementation)
const originalGetComputedStyle = window.getComputedStyle;
window.getComputedStyle = (element: Element, pseudoElt?: string | null) => {
  try {
    return originalGetComputedStyle(element, pseudoElt);
  } catch {
    // Return a mock for elements not in DOM
    return {
      getPropertyValue: () => "",
    } as CSSStyleDeclaration;
  }
};

// ============================================================================
// Local Storage Mock
// ============================================================================

const localStorageMock = (() => {
  let store: Record<string, string> = {};
  return {
    getItem: vi.fn((key: string) => store[key] ?? null),
    setItem: vi.fn((key: string, value: string) => {
      store[key] = value;
    }),
    removeItem: vi.fn((key: string) => {
      delete store[key];
    }),
    clear: vi.fn(() => {
      store = {};
    }),
    get length() {
      return Object.keys(store).length;
    },
    key: vi.fn((index: number) => Object.keys(store)[index] ?? null),
  };
})();

Object.defineProperty(window, "localStorage", {
  value: localStorageMock,
});

Object.defineProperty(window, "sessionStorage", {
  value: localStorageMock,
});

// ============================================================================
// Clipboard Mock
// ============================================================================

const clipboardMock = {
  writeText: vi.fn().mockResolvedValue(undefined),
  readText: vi.fn().mockResolvedValue(""),
  write: vi.fn().mockResolvedValue(undefined),
  read: vi.fn().mockResolvedValue([]),
};

Object.defineProperty(navigator, "clipboard", {
  value: clipboardMock,
  configurable: true,
});

// ============================================================================
// Animation Frame Mock
// ============================================================================

let animationFrameId = 0;
const animationFrameCallbacks = new Map<number, FrameRequestCallback>();

window.requestAnimationFrame = vi.fn((callback: FrameRequestCallback) => {
  const id = ++animationFrameId;
  animationFrameCallbacks.set(id, callback);
  // Execute on next tick to simulate async behavior
  setTimeout(() => {
    const cb = animationFrameCallbacks.get(id);
    if (cb) {
      animationFrameCallbacks.delete(id);
      cb(performance.now());
    }
  }, 16); // ~60fps
  return id;
});

window.cancelAnimationFrame = vi.fn((id: number) => {
  animationFrameCallbacks.delete(id);
});

// ============================================================================
// Pointer Events Mock
// ============================================================================

// Mock pointer capture (used by Radix UI and other libraries)
Element.prototype.setPointerCapture = vi.fn();
Element.prototype.releasePointerCapture = vi.fn();
Element.prototype.hasPointerCapture = vi.fn().mockReturnValue(false);

// ============================================================================
// Focus Management
// ============================================================================

// Track focus for accessibility testing
let currentFocusedElement: Element | null = null;

document.addEventListener("focusin", (e) => {
  currentFocusedElement = e.target as Element;
});

document.addEventListener("focusout", () => {
  currentFocusedElement = null;
});

/**
 * Get the currently focused element
 */
export function getFocusedElement(): Element | null {
  return currentFocusedElement || document.activeElement;
}

// ============================================================================
// Custom Matchers
// ============================================================================

// Re-export expect matchers from @testing-library/jest-dom
// These are automatically available via the import at top of file

// ============================================================================
// Exports for Test Files
// ============================================================================

export { localStorageMock, clipboardMock };
