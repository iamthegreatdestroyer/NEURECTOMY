# Lapce Editor UI/UX Patterns Analysis

## Overview

Lapce is a lightning-fast, Rust-native code editor using the Floem UI framework. This analysis extracts patterns for high-performance editor UIs, modal editing, and Rust-based architecture that can inform NEURECTOMY's approach to building responsive, professional-grade interfaces.

---

## 🚀 Architecture Overview

### Floem UI Framework

Lapce uses Floem, a native Rust UI framework with reactive signals:

```rust
// Reactive signal system
use floem::reactive::{RwSignal, SignalGet, SignalUpdate, SignalWith};

// Create reactive state
let count = create_rw_signal(0);

// Read signal
let value = count.get();

// Update signal
count.set(5);
count.update(|v| *v += 1);

// React to changes
count.with(|v| println!("Value: {}", v));
```

### Window Tab Data Structure

The central data structure managing window state:

```rust
pub struct WindowTabData {
    // Scope for reactive system
    pub scope: Scope,

    // Workspace info
    pub workspace: Arc<LapceWorkspace>,

    // Main split (editor area)
    pub main_split: MainSplitData,

    // Panel management
    pub panel: PanelData,

    // Terminal
    pub terminal: TerminalPanelData,

    // Palette (command palette)
    pub palette: PaletteData,

    // Plugin system
    pub plugin: PluginData,

    // Common data (config, keypress, etc.)
    pub common: Rc<CommonData>,
}
```

---

## 📑 Main Split System

### Split Data Structure

Lapce uses a recursive split structure for editor panes:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SplitContent {
    EditorTab(EditorTabId),
    Split(SplitId),
}

pub struct SplitData {
    pub scope: Scope,
    pub parent_split: Option<SplitId>,
    pub split_id: SplitId,

    // Children with their relative sizes
    pub children: Vec<(RwSignal<f64>, SplitContent)>,

    pub direction: SplitDirection,
    pub window_origin: Point,
    pub layout_rect: Rect,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct SplitInfo {
    pub direction: SplitDirection,
    pub children: Vec<SplitContentInfo>,
}
```

### Split Move Directions

```rust
#[derive(Clone, Copy, Debug)]
pub enum SplitMoveDirection {
    Up,
    Down,
    Right,
    Left,
}
```

### Main Split Data

```rust
pub struct MainSplitData {
    pub scope: Scope,

    // Root split ID
    pub root_split: SplitId,

    // All splits indexed by ID
    pub splits: RwSignal<im::HashMap<SplitId, RwSignal<SplitData>>>,

    // All editor tabs
    pub editor_tabs: RwSignal<im::HashMap<EditorTabId, RwSignal<EditorTabData>>>,

    // Currently active editor tab
    pub active_editor_tab: RwSignal<Option<EditorTabId>>,

    // All editors
    pub editors: Editors,

    // Diff editors
    pub diff_editors: RwSignal<im::HashMap<DiffEditorId, DiffEditorData>>,

    // Width for layout
    pub width: RwSignal<f64>,
}
```

---

## 📋 Editor Tab System

### Editor Tab Data

```rust
#[derive(Clone)]
pub struct EditorTabData {
    pub scope: Scope,
    pub split: SplitId,
    pub editor_tab_id: EditorTabId,
    pub active: usize,

    // Children: (index signal, rect signal, child)
    pub children: Vec<(RwSignal<usize>, RwSignal<Rect>, EditorTabChild)>,

    pub window_origin: Point,
    pub layout_rect: Rect,

    // Navigation history
    pub locations: RwSignal<im::Vector<EditorLocation>>,
    pub current_location: RwSignal<usize>,
}
```

### Editor Tab Children Types

```rust
pub enum EditorTabChild {
    Editor(EditorId),
    DiffEditor(DiffEditorId),
    Settings(SettingsId),
    Keymap(KeymapId),
    Volt(VoltId, VoltView),  // Plugin views
}
```

### Editor Tab Header

The tab bar with navigation and split controls:

```rust
fn editor_tab_header(
    window_tab_data: Rc<WindowTabData>,
    active_editor_tab: ReadSignal<Option<EditorTabId>>,
    editor_tab: RwSignal<EditorTabData>,
    dragging: RwSignal<Option<(RwSignal<usize>, EditorTabId)>>,
) -> impl View {
    // Layout:
    // [< Previous] [Next >] | [Tabs...] | [Split ↔] [Close All]

    stack((
        // Navigation buttons
        clickable_icon(LapceIcons::TAB_PREVIOUS, ...),
        clickable_icon(LapceIcons::TAB_NEXT, ...),

        // Scrollable tab list
        scroll(tab_list(...)),

        // Split and close buttons
        clickable_icon(LapceIcons::SPLIT_HORIZONTAL, ...),
        clickable_icon(LapceIcons::CLOSE_ALL, ...),
    ))
}
```

---

## 🎛️ Panel System

### Panel Kinds

```rust
pub enum PanelKind {
    Terminal,
    FileExplorer,
    SourceControl,
    Plugin,
    Search,
    Problem,
    Debug,
    CallHierarchy,
    DocumentSymbol,
    References,
}
```

### Panel Positions

```rust
pub enum PanelPosition {
    LeftTop,
    LeftBottom,
    BottomLeft,
    BottomRight,
    RightTop,
    RightBottom,
}

pub enum PanelContainerPosition {
    Left,
    Bottom,
    Right,
}
```

### Panel Data

```rust
pub struct PanelData {
    // Panel order by position
    pub panels: RwSignal<HashMap<PanelPosition, Vec<PanelKind>>>,

    // Active panel per position
    pub styles: RwSignal<HashMap<PanelPosition, PanelStyle>>,

    // Size per container
    pub size: RwSignal<HashMap<PanelContainerPosition, f64>>,

    // Available space
    pub available_size: Memo<Size>,

    // Expanded/collapsed per section
    pub sections: RwSignal<HashMap<PanelPosition, RwSignal<bool>>>,

    pub common: Rc<CommonData>,
}
```

### Panel View Function

```rust
fn panel_view(
    window_tab_data: Rc<WindowTabData>,
    position: PanelPosition,
) -> impl View {
    let panels = move || {
        panel.panels.with(|p| p.get(&position).cloned().unwrap_or_default())
    };

    tab(
        active_fn,
        panels,
        |p| *p,
        move |kind| {
            match kind {
                PanelKind::Terminal => terminal_panel(...).into_any(),
                PanelKind::FileExplorer => file_explorer_panel(...).into_any(),
                PanelKind::SourceControl => source_control_panel(...).into_any(),
                PanelKind::Plugin => plugin_panel(...).into_any(),
                PanelKind::Search => search_panel(...).into_any(),
                PanelKind::Problem => problem_panel(...).into_any(),
                PanelKind::Debug => debug_panel(...).into_any(),
                // ... etc
            }
        },
    )
}
```

---

## 🖥️ Terminal Integration

### Terminal Panel Data

```rust
pub struct TerminalPanelData {
    pub cx: Scope,
    pub workspace: Arc<LapceWorkspace>,
    pub tab_info: RwSignal<TerminalTabInfo>,
    pub debug: RunDebugData,
    pub breakline: Memo<Option<(usize, PathBuf)>>,
    pub common: Rc<CommonData>,
    pub main_split: MainSplitData,
}

pub struct TerminalTabInfo {
    pub active: usize,
    pub tabs: im::Vector<(RwSignal<usize>, TerminalTabData)>,
}
```

### Terminal Split Support

Terminals can be split within a tab:

```rust
impl TerminalPanelData {
    pub fn split(&self, term_id: TermId) {
        if let Some((_, tab, index, _)) = self.get_terminal_in_tab(&term_id) {
            let terminal_data = TerminalData::new(...);
            tab.terminals.update(|terminals| {
                terminals.insert(index + 1, terminal_data);
            });
        }
    }

    pub fn split_next(&self, term_id: TermId) {
        // Move focus to next terminal in split
    }

    pub fn split_previous(&self, term_id: TermId) {
        // Move focus to previous terminal in split
    }

    pub fn split_exchange(&self, term_id: TermId) {
        // Swap position with next terminal
    }
}
```

### Terminal View

```rust
fn terminal_tab_split(
    terminal_panel_data: TerminalPanelData,
    terminal_tab_data: TerminalTabData,
    tab_index: usize,
) -> impl View {
    dyn_stack(
        move || terminal_tab_data.terminals.get(),
        |(_, terminal)| terminal.term_id,
        move |(index, terminal)| {
            container(
                terminal_view(
                    terminal.term_id,
                    terminal.raw.read_only(),
                    terminal.mode.read_only(),
                    terminal.run_debug.read_only(),
                    ...
                )
            )
            .style(move |s| {
                s.size_pct(100.0, 100.0)
                    .padding_horiz(10.0)
                    .apply_if(index.get() > 0, |s| {
                        s.border_left(1.0)
                         .border_color(config.get().color(LapceColor::LAPCE_BORDER))
                    })
            })
        },
    )
}
```

---

## 🎯 Applicable Patterns for NEURECTOMY

### 1. Reactive Signal System

**Implementation Priority: HIGH**

Adapt the reactive signal pattern for TypeScript:

```typescript
// NEURECTOMY reactive state (using Zustand or similar)
interface WorkspaceState {
  // Splits
  splits: Map<string, SplitData>;
  rootSplitId: string;

  // Editor tabs
  editorTabs: Map<string, EditorTabData>;
  activeEditorTabId: string | null;

  // Actions
  addSplit: (direction: SplitDirection, editorTabId: string) => void;
  removeSplit: (splitId: string) => void;
  moveFocus: (direction: SplitMoveDirection) => void;
}
```

### 2. Split with Size Signals

**Implementation Priority: HIGH**

Each child has an individual size signal:

```typescript
interface SplitChild {
  id: string;
  content: SplitContent;
  size: number; // 0-1 relative size
}

// When resizing, only update affected size signals
function resizeSplit(splitId: string, childIndex: number, delta: number) {
  const split = splits.get(splitId);
  const children = split.children;

  // Adjust sizes of adjacent children
  const currentSize = children[childIndex].size;
  const nextSize = children[childIndex + 1].size;

  children[childIndex].size = currentSize + delta;
  children[childIndex + 1].size = nextSize - delta;
}
```

### 3. Panel Container Positions

**Implementation Priority: HIGH**

Lapce's panel organization:

```typescript
// Panel containers
type PanelContainerPosition = "left" | "bottom" | "right";

// Within each container, panels can be in different sections
type PanelPosition =
  | "left-top"
  | "left-bottom"
  | "bottom-left"
  | "bottom-right"
  | "right-top"
  | "right-bottom";

// Panel configuration
interface PanelConfig {
  // Which panels are in which position
  layout: Record<PanelPosition, PanelKind[]>;

  // Active panel per position
  active: Record<PanelPosition, number>;

  // Container sizes
  sizes: Record<PanelContainerPosition, number>;

  // Expanded state per section
  expanded: Record<PanelPosition, boolean>;
}
```

### 4. Editor Tab Children Types

**Implementation Priority: HIGH**

Support multiple content types in tabs:

```typescript
type EditorTabChild =
  | { type: "editor"; editorId: string }
  | { type: "diff"; leftPath: string; rightPath: string }
  | { type: "settings" }
  | { type: "keybindings" }
  | { type: "extension"; extensionId: string; viewId: string }
  // NEURECTOMY-specific
  | { type: "container-logs"; containerId: string }
  | { type: "experiment"; experimentId: string }
  | { type: "agent-chat"; sessionId: string };
```

### 5. Workbench Command System

**Implementation Priority: HIGH**

Lapce's command dispatch pattern:

```typescript
// Command definitions
enum WorkbenchCommand {
  // Tab commands
  NextEditorTab = "next-editor-tab",
  PreviousEditorTab = "previous-editor-tab",
  CloseEditorTab = "close-editor-tab",

  // Split commands
  SplitRight = "split-right",
  SplitDown = "split-down",
  SplitMoveUp = "split-move-up",
  SplitMoveDown = "split-move-down",

  // Terminal commands
  NewTerminalTab = "new-terminal-tab",
  SplitTerminal = "split-terminal",

  // Panel commands
  TogglePanel = "toggle-panel",
  FocusPanel = "focus-panel",
}

// Command handler
function runWorkbenchCommand(cmd: WorkbenchCommand, data?: any) {
  switch (cmd) {
    case WorkbenchCommand.SplitRight:
      mainSplit.split("horizontal", activeEditorTabId);
      break;
    case WorkbenchCommand.SplitMoveUp:
      mainSplit.moveFocus("up");
      break;
    // ... etc
  }
}
```

### 6. Focus Management

**Implementation Priority: HIGH**

Lapce tracks focus state:

```typescript
type Focus =
  | { type: "workbench" } // Main editor area
  | { type: "panel"; kind: PanelKind }
  | { type: "palette" } // Command palette
  | { type: "modal"; modalId: string };

// Focus changes update keyboard shortcut context
useEffect(() => {
  updateKeyboardContext(focus);
}, [focus]);
```

---

## 🎨 Visual Design from Lapce

### Editor Tab Header Style

```css
/* Lapce-style editor tab header */
.editor-tab-header {
  display: flex;
  align-items: center;
  height: var(--header-height);
  border-bottom: 1px solid var(--border-color);
  background: var(--panel-background);
}

.tab-navigation {
  display: flex;
  margin: 6px 0;
}

.tab-list {
  flex: 1;
  overflow-x: auto;
  scrollbar-width: none;
}

.tab-actions {
  display: flex;
  margin-left: auto;
  flex-shrink: 0;
}

.tab-item {
  display: grid;
  grid-template-columns: auto 1fr auto;
  align-items: center;
  padding: 4px;
  border-right: 1px solid var(--border-color);

  &.active {
    background: var(--tab-active-background);
  }

  &:not(.pristine) .tab-modified-indicator {
    color: var(--warning-color);
  }
}
```

### Terminal Split Border

```css
/* Terminal split styling */
.terminal-split {
  display: flex;
  width: 100%;
  height: 100%;
}

.terminal-instance {
  flex: 1;
  padding: 0 10px;

  &:not(:first-child) {
    border-left: 1px solid var(--border-color);
  }
}
```

### Panel Picker

```css
/* Panel picker (activity bar equivalent) */
.panel-picker {
  display: flex;
  flex-direction: column;

  &.bottom {
    flex-direction: row;
  }
}

.panel-picker-item {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 32px;
  height: 32px;
  cursor: pointer;
  opacity: 0.6;

  &.active {
    opacity: 1;
    border-left: 2px solid var(--accent-color);
  }

  &:hover {
    opacity: 0.8;
  }
}
```

---

## 📋 Checklist for NEURECTOMY Implementation

### Phase 1: Split System

- [ ] Implement recursive SplitData structure
- [ ] Create MainSplit store with signals
- [ ] Add split operations (right, down, up, left)
- [ ] Implement focus movement between splits

### Phase 2: Editor Tabs

- [ ] EditorTabData with multiple child types
- [ ] Tab navigation (prev/next)
- [ ] Tab drag-and-drop reordering
- [ ] Navigation history (locations)

### Phase 3: Panel System

- [ ] Panel container positions (left, bottom, right)
- [ ] Panel sections within containers
- [ ] Panel picker (activity bar style)
- [ ] Panel size persistence

### Phase 4: Terminal Integration

- [ ] Terminal tabs
- [ ] Terminal splits within tabs
- [ ] Terminal navigation (next/prev)
- [ ] Terminal exchange (swap positions)

### Phase 5: Commands

- [ ] Workbench command system
- [ ] Keybinding integration
- [ ] Command palette

---

## 🔧 Technical Implementation Notes

### Immutable Data Structures

Lapce uses immutable data (im::HashMap, im::Vector):

```typescript
// Use immer for immutable updates in TypeScript
import produce from "immer";

const updateSplits = produce((draft) => {
  draft.splits.get(splitId).children.push(newChild);
});
```

### Scope-based Lifecycle

Lapce creates child scopes for cleanup:

```typescript
// React equivalent with cleanup
useEffect(() => {
  // Setup
  const cleanup = setupEditorTab(tabId);

  return () => {
    // Cleanup when tab closes
    cleanup();
  };
}, [tabId]);
```

### Layout Rect Tracking

Each component tracks its layout rectangle:

```typescript
interface LayoutRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

// Track via ResizeObserver
const layoutRect = useLayoutRect(elementRef);
```

---

_Analysis completed for NEURECTOMY IDE reference_
_Source: lapce/lapce_
