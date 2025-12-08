/**
 * Workspace Store
 *
 * Global state management for the NEURECTOMY workspace.
 * Handles workspace layout, panels, tabs, and persistence.
 */

import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";
import { immer } from "zustand/middleware/immer";
import { devtools } from "zustand/middleware";

export type PanelPosition = "left" | "right" | "bottom";
export type PanelVisibility = Record<string, boolean>;
export type TabItem = {
  id: string;
  type:
    | "editor"
    | "forge"
    | "container"
    | "intelligence"
    | "discovery"
    | "legal";
  title: string;
  icon?: string;
  data?: unknown;
  isDirty?: boolean;
};

export interface WorkspaceLayout {
  leftSidebarWidth: number;
  rightSidebarWidth: number;
  bottomPanelHeight: number;
  leftSidebarCollapsed: boolean;
  rightSidebarCollapsed: boolean;
  bottomPanelCollapsed: boolean;
}

export interface WorkspaceState {
  // Layout
  layout: WorkspaceLayout;

  // Tabs
  tabs: TabItem[];
  activeTabId: string | null;

  // Panel visibility
  panelVisibility: PanelVisibility;

  // Recent files/projects
  recentFiles: string[];
  recentProjects: string[];

  // Current project path
  currentProjectPath: string | null;

  // Theme
  theme: "dark" | "light" | "system";

  // Actions
  setLayout: (layout: Partial<WorkspaceLayout>) => void;
  toggleSidebar: (position: "left" | "right") => void;
  toggleBottomPanel: () => void;

  addTab: (tab: TabItem) => void;
  removeTab: (tabId: string) => void;
  setActiveTab: (tabId: string) => void;
  updateTab: (tabId: string, updates: Partial<TabItem>) => void;

  togglePanel: (panelId: string) => void;

  addRecentFile: (filePath: string) => void;
  addRecentProject: (projectPath: string) => void;

  setCurrentProject: (projectPath: string | null) => void;
  setTheme: (theme: "dark" | "light" | "system") => void;

  reset: () => void;
}

const DEFAULT_LAYOUT: WorkspaceLayout = {
  leftSidebarWidth: 280,
  rightSidebarWidth: 320,
  bottomPanelHeight: 240,
  leftSidebarCollapsed: false,
  rightSidebarCollapsed: false,
  bottomPanelCollapsed: false,
};

export const useWorkspaceStore = create<WorkspaceState>()(
  devtools(
    persist(
      immer((set) => ({
        // Initial state
        layout: DEFAULT_LAYOUT,
        tabs: [],
        activeTabId: null,
        panelVisibility: {
          properties: true,
          ai_chat: true,
          terminal: true,
          logs: true,
          console: true,
        },
        recentFiles: [],
        recentProjects: [],
        currentProjectPath: null,
        theme: "dark",

        // Layout actions
        setLayout: (layoutUpdate) =>
          set((state) => {
            Object.assign(state.layout, layoutUpdate);
          }),

        toggleSidebar: (position) =>
          set((state) => {
            if (position === "left") {
              state.layout.leftSidebarCollapsed =
                !state.layout.leftSidebarCollapsed;
            } else {
              state.layout.rightSidebarCollapsed =
                !state.layout.rightSidebarCollapsed;
            }
          }),

        toggleBottomPanel: () =>
          set((state) => {
            state.layout.bottomPanelCollapsed =
              !state.layout.bottomPanelCollapsed;
          }),

        // Tab actions
        addTab: (tab) =>
          set((state) => {
            // Check if tab already exists
            const existingIndex = state.tabs.findIndex((t) => t.id === tab.id);
            if (existingIndex !== -1) {
              state.activeTabId = tab.id;
              return;
            }
            state.tabs.push(tab);
            state.activeTabId = tab.id;
          }),

        removeTab: (tabId) =>
          set((state) => {
            const index = state.tabs.findIndex((t) => t.id === tabId);
            if (index === -1) return;

            state.tabs.splice(index, 1);

            // Update active tab if needed
            if (state.activeTabId === tabId) {
              if (state.tabs.length > 0) {
                // Activate the tab to the left, or the first tab if we removed the first one
                state.activeTabId =
                  state.tabs[Math.max(0, index - 1)]?.id || null;
              } else {
                state.activeTabId = null;
              }
            }
          }),

        setActiveTab: (tabId) =>
          set((state) => {
            state.activeTabId = tabId;
          }),

        updateTab: (tabId, updates) =>
          set((state) => {
            const tab = state.tabs.find((t) => t.id === tabId);
            if (tab) {
              Object.assign(tab, updates);
            }
          }),

        // Panel actions
        togglePanel: (panelId) =>
          set((state) => {
            state.panelVisibility[panelId] = !state.panelVisibility[panelId];
          }),

        // Recent items
        addRecentFile: (filePath) =>
          set((state) => {
            // Remove if already exists
            const filtered = state.recentFiles.filter((f) => f !== filePath);
            // Add to front
            state.recentFiles = [filePath, ...filtered].slice(0, 20);
          }),

        addRecentProject: (projectPath) =>
          set((state) => {
            const filtered = state.recentProjects.filter(
              (p) => p !== projectPath
            );
            state.recentProjects = [projectPath, ...filtered].slice(0, 10);
          }),

        // Project
        setCurrentProject: (projectPath) =>
          set((state) => {
            state.currentProjectPath = projectPath;
            if (projectPath) {
              // Add to recent projects
              const filtered = state.recentProjects.filter(
                (p) => p !== projectPath
              );
              state.recentProjects = [projectPath, ...filtered].slice(0, 10);
            }
          }),

        // Theme
        setTheme: (theme) =>
          set((state) => {
            state.theme = theme;
          }),

        // Reset
        reset: () =>
          set({
            layout: DEFAULT_LAYOUT,
            tabs: [],
            activeTabId: null,
            panelVisibility: {
              properties: true,
              ai_chat: true,
              terminal: true,
              logs: true,
              console: true,
            },
            recentFiles: [],
            recentProjects: [],
            currentProjectPath: null,
            theme: "dark",
          }),
      })),
      {
        name: "neurectomy-workspace",
        storage: createJSONStorage(() => localStorage),
        partialize: (state) => ({
          layout: state.layout,
          recentFiles: state.recentFiles,
          recentProjects: state.recentProjects,
          theme: state.theme,
        }),
      }
    ),
    { name: "WorkspaceStore" }
  )
);
