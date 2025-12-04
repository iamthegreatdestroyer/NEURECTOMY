/**
 * Keyboard Shortcuts Hook
 * Global keyboard shortcut handling for NEURECTOMY
 */

import { useEffect, useCallback } from 'react';
import { useAppStore } from '../stores/app.store';

interface ShortcutHandler {
  key: string;
  ctrl?: boolean;
  shift?: boolean;
  alt?: boolean;
  meta?: boolean;
  handler: () => void;
  description: string;
}

// Define all keyboard shortcuts
const shortcuts: ShortcutHandler[] = [
  {
    key: 'k',
    ctrl: true,
    handler: () => useAppStore.getState().openCommandPalette(),
    description: 'Open command palette',
  },
  {
    key: 'k',
    meta: true, // For Mac
    handler: () => useAppStore.getState().openCommandPalette(),
    description: 'Open command palette (Mac)',
  },
  {
    key: 'Escape',
    handler: () => useAppStore.getState().closeCommandPalette(),
    description: 'Close command palette',
  },
  {
    key: 'b',
    ctrl: true,
    handler: () => useAppStore.getState().toggleSidebar(),
    description: 'Toggle sidebar',
  },
  {
    key: '/',
    ctrl: true,
    handler: () => {
      // Focus search input
      const searchInput = document.querySelector('[data-search-input]');
      if (searchInput instanceof HTMLInputElement) {
        searchInput.focus();
      }
    },
    description: 'Focus search',
  },
];

// Navigation shortcuts
const navigationShortcuts: ShortcutHandler[] = [
  {
    key: '1',
    alt: true,
    handler: () => window.location.hash = '#/dashboard',
    description: 'Go to Dashboard',
  },
  {
    key: '2',
    alt: true,
    handler: () => window.location.hash = '#/forge',
    description: 'Go to Dimensional Forge',
  },
  {
    key: '3',
    alt: true,
    handler: () => window.location.hash = '#/containers',
    description: 'Go to Container Command',
  },
  {
    key: '4',
    alt: true,
    handler: () => window.location.hash = '#/foundry',
    description: 'Go to Intelligence Foundry',
  },
  {
    key: '5',
    alt: true,
    handler: () => window.location.hash = '#/discovery',
    description: 'Go to Discovery Engine',
  },
];

// Combine all shortcuts
const allShortcuts = [...shortcuts, ...navigationShortcuts];

export function useKeyboardShortcuts() {
  const handleKeyDown = useCallback((event: KeyboardEvent) => {
    // Don't trigger shortcuts when typing in inputs
    if (
      event.target instanceof HTMLInputElement ||
      event.target instanceof HTMLTextAreaElement ||
      (event.target as HTMLElement)?.isContentEditable
    ) {
      // Allow Escape to still work
      if (event.key !== 'Escape') {
        return;
      }
    }

    for (const shortcut of allShortcuts) {
      const ctrlMatch = shortcut.ctrl ? (event.ctrlKey || event.metaKey) : !event.ctrlKey;
      const shiftMatch = shortcut.shift ? event.shiftKey : !event.shiftKey;
      const altMatch = shortcut.alt ? event.altKey : !event.altKey;
      const metaMatch = shortcut.meta ? event.metaKey : true;
      const keyMatch = event.key.toLowerCase() === shortcut.key.toLowerCase();

      if (keyMatch && ctrlMatch && shiftMatch && altMatch && metaMatch) {
        event.preventDefault();
        shortcut.handler();
        return;
      }
    }
  }, []);

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);

  return { shortcuts: allShortcuts };
}

// Hook to get formatted shortcut string
export function useShortcutString(shortcut: ShortcutHandler): string {
  const isMac = navigator.platform.toUpperCase().indexOf('MAC') >= 0;
  
  const parts: string[] = [];
  
  if (shortcut.ctrl) parts.push(isMac ? '⌘' : 'Ctrl');
  if (shortcut.alt) parts.push(isMac ? '⌥' : 'Alt');
  if (shortcut.shift) parts.push(isMac ? '⇧' : 'Shift');
  if (shortcut.meta) parts.push(isMac ? '⌘' : 'Win');
  
  parts.push(shortcut.key.toUpperCase());
  
  return parts.join(isMac ? '' : '+');
}

// Export shortcuts for documentation/help menu
export { allShortcuts as keyboardShortcuts };
