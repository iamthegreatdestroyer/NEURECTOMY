/**
 * Global Application Store
 * Using Zustand for state management
 */

import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';

// Theme type
type Theme = 'dark' | 'light' | 'system';

// Sidebar state
interface SidebarState {
  isCollapsed: boolean;
  activeSection: string | null;
}

// Command palette state
interface CommandPaletteState {
  isOpen: boolean;
  searchQuery: string;
}

// Notification
interface Notification {
  id: string;
  type: 'info' | 'success' | 'warning' | 'error';
  title: string;
  message?: string;
  timestamp: Date;
  read: boolean;
}

// User preferences
interface UserPreferences {
  theme: Theme;
  sidebarCollapsed: boolean;
  showWelcome: boolean;
  enableAnimations: boolean;
  enableSounds: boolean;
}

// App store interface
interface AppStore {
  // Theme
  theme: Theme;
  setTheme: (theme: Theme) => void;
  
  // Sidebar
  sidebar: SidebarState;
  toggleSidebar: () => void;
  setSidebarSection: (section: string | null) => void;
  
  // Command Palette
  commandPalette: CommandPaletteState;
  openCommandPalette: () => void;
  closeCommandPalette: () => void;
  setCommandPaletteQuery: (query: string) => void;
  
  // Notifications
  notifications: Notification[];
  addNotification: (notification: Omit<Notification, 'id' | 'timestamp' | 'read'>) => void;
  markNotificationRead: (id: string) => void;
  clearNotifications: () => void;
  
  // User Preferences
  preferences: UserPreferences;
  updatePreferences: (updates: Partial<UserPreferences>) => void;
  
  // Connection Status
  isConnected: boolean;
  setConnectionStatus: (status: boolean) => void;
  
  // Loading States
  isLoading: boolean;
  loadingMessage: string | null;
  setLoading: (loading: boolean, message?: string) => void;
}

export const useAppStore = create<AppStore>()(
  devtools(
    persist(
      (set, get) => ({
        // Theme - defaults to dark
        theme: 'dark',
        setTheme: (theme) => {
          set({ theme });
          // Apply theme to document
          if (theme === 'system') {
            const systemTheme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
            document.documentElement.classList.toggle('dark', systemTheme === 'dark');
          } else {
            document.documentElement.classList.toggle('dark', theme === 'dark');
          }
        },
        
        // Sidebar
        sidebar: {
          isCollapsed: false,
          activeSection: 'dashboard',
        },
        toggleSidebar: () => set((state) => ({
          sidebar: { ...state.sidebar, isCollapsed: !state.sidebar.isCollapsed }
        })),
        setSidebarSection: (section) => set((state) => ({
          sidebar: { ...state.sidebar, activeSection: section }
        })),
        
        // Command Palette
        commandPalette: {
          isOpen: false,
          searchQuery: '',
        },
        openCommandPalette: () => set((state) => ({
          commandPalette: { ...state.commandPalette, isOpen: true }
        })),
        closeCommandPalette: () => set((state) => ({
          commandPalette: { ...state.commandPalette, isOpen: false, searchQuery: '' }
        })),
        setCommandPaletteQuery: (query) => set((state) => ({
          commandPalette: { ...state.commandPalette, searchQuery: query }
        })),
        
        // Notifications
        notifications: [],
        addNotification: (notification) => set((state) => ({
          notifications: [
            {
              ...notification,
              id: crypto.randomUUID(),
              timestamp: new Date(),
              read: false,
            },
            ...state.notifications,
          ].slice(0, 100), // Keep last 100 notifications
        })),
        markNotificationRead: (id) => set((state) => ({
          notifications: state.notifications.map((n) =>
            n.id === id ? { ...n, read: true } : n
          ),
        })),
        clearNotifications: () => set({ notifications: [] }),
        
        // User Preferences
        preferences: {
          theme: 'dark',
          sidebarCollapsed: false,
          showWelcome: true,
          enableAnimations: true,
          enableSounds: false,
        },
        updatePreferences: (updates) => set((state) => ({
          preferences: { ...state.preferences, ...updates },
        })),
        
        // Connection Status
        isConnected: false,
        setConnectionStatus: (status) => set({ isConnected: status }),
        
        // Loading States
        isLoading: false,
        loadingMessage: null,
        setLoading: (loading, message) => set({
          isLoading: loading,
          loadingMessage: loading ? message ?? null : null,
        }),
      }),
      {
        name: 'neurectomy-app-store',
        partialize: (state) => ({
          theme: state.theme,
          sidebar: state.sidebar,
          preferences: state.preferences,
        }),
      }
    ),
    { name: 'AppStore' }
  )
);

// Selectors for optimized re-renders
export const useTheme = () => useAppStore((state) => state.theme);
export const useSidebar = () => useAppStore((state) => state.sidebar);
export const useCommandPalette = () => useAppStore((state) => state.commandPalette);
export const useNotifications = () => useAppStore((state) => state.notifications);
export const usePreferences = () => useAppStore((state) => state.preferences);
export const useConnectionStatus = () => useAppStore((state) => state.isConnected);
export const useLoadingState = () => useAppStore((state) => ({
  isLoading: state.isLoading,
  message: state.loadingMessage,
}));
