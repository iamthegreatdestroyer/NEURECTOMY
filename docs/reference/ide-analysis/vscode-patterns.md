# VS Code UI/UX Patterns Analysis

## Overview

Visual Studio Code is the gold standard for modern IDEs, serving as the primary reference for NEURECTOMY's UI architecture. This analysis extracts key patterns from the VS Code codebase that should inform our implementation.

---

## 🏗️ Core Architecture

### Workbench Parts System

VS Code organizes its UI into discrete **Parts**, each responsible for a specific region of the interface:

```typescript
// Key Parts Architecture
enum Parts {
  ACTIVITYBAR_PART = "workbench.parts.activitybar",
  SIDEBAR_PART = "workbench.parts.sidebar",
  PANEL_PART = "workbench.parts.panel",
  EDITOR_PART = "workbench.parts.editor",
  STATUSBAR_PART = "workbench.parts.statusbar",
  TITLEBAR_PART = "workbench.parts.titlebar",
  BANNER_PART = "workbench.parts.banner",
  AUXILIARYBAR_PART = "workbench.parts.auxiliarybar",
}
```

**Key Insight:** Each part is a self-contained component with its own lifecycle, visibility state, and layout calculations.

### Layout Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        TITLEBAR                                  │
├──────┬────────────────────────────────────────────────────┬─────┤
│      │                                                    │     │
│  A   │                    EDITOR                          │  A  │
│  C   │              (EditorGroupsService)                 │  U  │
│  T   ├────────────────────────────────────────────────────┤  X  │
│  I   │                                                    │  I  │
│  V   │                    PANEL                           │  L  │
│  I   │             (Terminal/Output/Problems)             │  I  │
│  T   │                                                    │  A  │
│  Y   │                                                    │  R  │
│  B   │                                                    │  Y  │
│  A   │                                                    │     │
│  R   │                                                    │  B  │
│      │                                                    │  A  │
│  +   │                                                    │  R  │
│      │                                                    │     │
│  S   │                                                    │     │
│  I   │                                                    │     │
│  D   │                                                    │     │
│  E   │                                                    │     │
│  B   │                                                    │     │
│  A   │                                                    │     │
│  R   │                                                    │     │
├──────┴────────────────────────────────────────────────────┴─────┤
│                        STATUSBAR                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📑 Editor Groups & Tabs

### EditorGroupsService

The `EditorGroupsService` manages multiple editor groups with sophisticated splitting capabilities:

```typescript
interface IEditorGroupsService {
  // Group management
  readonly activeGroup: IEditorGroup;
  readonly groups: readonly IEditorGroup[];

  // Splitting
  addGroup(location: IEditorGroup, direction: GroupDirection): IEditorGroup;

  // Layout
  setGroupOrientation(orientation: GroupOrientation): void;
  applyLayout(layout: EditorGroupLayout): void;
}

// Split directions
enum GroupDirection {
  UP,
  DOWN,
  LEFT,
  RIGHT,
}
```

### Editor Tabs

Each tab tracks:

- Dirty state (unsaved changes)
- Preview mode (single-click = preview, double-click = pin)
- Pinned state
- Label (with path disambiguation)

```typescript
interface IEditorTabDto {
  id: string;
  label: string;
  isActive: boolean;
  isDirty: boolean;
  isPinned: boolean;
  isPreview: boolean;
}
```

---

## 🎯 Applicable Patterns for NEURECTOMY

### 1. Part-Based Layout System

**Implementation Priority: HIGH**

Create discrete UI parts that can be:

- Shown/hidden independently
- Resized with splitters
- Persisted to layout state

```typescript
// NEURECTOMY Implementation
interface WorkbenchPart {
  id: string;
  element: HTMLElement;
  isVisible: boolean;
  minimumWidth?: number;
  maximumWidth?: number;
  onDidVisibilityChange: Event<boolean>;
}
```

### 2. Activity Bar Pattern

**Implementation Priority: HIGH**

The activity bar provides:

- Quick access to major views (Explorer, Search, Git, etc.)
- Visual indication of active view
- Badge support for notifications
- Drag-and-drop reordering

```typescript
interface ActivityBarAction {
  id: string;
  icon: string;
  tooltip: string;
  badge?: { count: number; color: string };
  keybinding?: string;
}
```

### 3. Editor Group Split Actions

**Implementation Priority: HIGH**

Support these split operations:

- Split Right (Ctrl+\)
- Split Down
- Move to New Window
- Close All in Group

```typescript
const splitActions = [
  { id: "splitRight", icon: "split-horizontal", direction: "RIGHT" },
  { id: "splitDown", icon: "split-vertical", direction: "DOWN" },
  { id: "moveToNewWindow", icon: "multiple-windows" },
  { id: "closeAll", icon: "close-all" },
];
```

### 4. Panel Position Options

**Implementation Priority: MEDIUM**

Panels can be positioned:

- Bottom (default)
- Left
- Right

With controls for:

- Maximize/restore
- Close
- Move to opposite side

### 5. Command Palette Architecture

**Implementation Priority: HIGH**

The command palette (`Ctrl+Shift+P`) provides:

- Fuzzy search across all commands
- Recently used commands at top
- Keybinding display
- Category grouping

```typescript
interface QuickPickItem {
  label: string;
  description?: string;
  detail?: string;
  iconClasses?: string[];
  keybinding?: string;
  buttons?: QuickInputButton[];
}
```

### 6. Status Bar Segments

**Implementation Priority: MEDIUM**

Status bar organized into:

- Left side: Git branch, problems count, tasks
- Right side: Language, line/col, encoding, EOL, spaces/tabs

Each segment is:

- Clickable
- Has tooltip
- Can show progress
- Can change color (warnings, errors)

---

## 🔧 Technical Implementation Notes

### Sash (Splitter) Component

VS Code uses a `Sash` component for all resizable splits:

```typescript
class Sash {
  // Orientation
  orientation: Orientation; // VERTICAL | HORIZONTAL

  // Constraints
  minimumSize: number;
  maximumSize: number;

  // Events
  onDidStart: Event<ISashEvent>;
  onDidChange: Event<ISashEvent>;
  onDidEnd: Event<ISashEvent>;
  onDidReset: Event<void>; // Double-click to reset
}
```

### Grid Layout

For complex multi-pane layouts:

```typescript
interface SerializedGridObject<T> {
  type: "branch" | "leaf";
  data?: T;
  size?: number;
  children?: SerializedGridObject<T>[];
}
```

### Layout Persistence

Layout state is serialized and restored:

```typescript
interface WorkbenchLayoutState {
  parts: {
    [partId: string]: {
      visible: boolean;
      size: number;
    };
  };
  editorGroups: SerializedEditorGroupModel;
  panelPosition: "bottom" | "left" | "right";
}
```

---

## 📋 Checklist for NEURECTOMY Implementation

### Phase 1: Core Layout

- [ ] Implement WorkbenchPart base class
- [ ] Create Activity Bar component
- [ ] Create Sidebar container
- [ ] Create Editor area with tab support
- [ ] Create Panel area (bottom)
- [ ] Create Status Bar

### Phase 2: Split & Resize

- [ ] Implement Sash (splitter) component
- [ ] Add editor group splitting
- [ ] Add panel resize
- [ ] Add sidebar resize
- [ ] Persist layout state

### Phase 3: Polish

- [ ] Command palette
- [ ] Keyboard navigation
- [ ] Drag-and-drop tabs
- [ ] Tab pinning
- [ ] Tab preview mode

---

## 🎨 Visual Design Tokens

VS Code uses a comprehensive theming system:

```css
/* Key color tokens to implement */
--vscode-activityBar-background
--vscode-activityBar-foreground
--vscode-activityBarBadge-background
--vscode-sideBar-background
--vscode-editor-background
--vscode-editorGroupHeader-tabsBackground
--vscode-tab-activeBackground
--vscode-tab-inactiveBackground
--vscode-panel-background
--vscode-statusBar-background
--vscode-statusBar-foreground
```

---

## 📚 Reference Files

Key VS Code source files for deeper study:

- `src/vs/workbench/browser/layout.ts` - Main layout orchestration
- `src/vs/workbench/browser/parts/editor/editorGroupsControl.ts` - Editor groups
- `src/vs/base/browser/ui/sash/sash.ts` - Splitter implementation
- `src/vs/workbench/browser/parts/activitybar/` - Activity bar
- `src/vs/workbench/services/layout/browser/layoutService.ts` - Layout service

---

_Analysis completed for NEURECTOMY IDE reference_
_Source: microsoft/vscode_
