# Zed Editor UI/UX Patterns Analysis

## Overview

Zed is a high-performance, GPU-accelerated code editor built in Rust with the custom GPUI framework. This analysis focuses on Zed's unique approach to performance-first UI design, pane management, and collaborative features that could benefit NEURECTOMY's architecture.

---

## 🚀 GPUI Framework Architecture

### Element Trait System

Zed's GPUI framework uses a three-phase rendering pipeline:

```rust
// Core Element trait
pub trait Element: 'static + IntoElement {
    type RequestLayoutState: 'static;
    type PrepaintState: 'static;

    fn id(&self) -> Option<ElementId>;

    // Phase 1: Layout calculation
    fn request_layout(
        &mut self,
        id: Option<&GlobalElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState);

    // Phase 2: Pre-paint preparation
    fn prepaint(
        &mut self,
        id: Option<&GlobalElementId>,
        bounds: Bounds<Pixels>,
        request_layout: &mut Self::RequestLayoutState,
        window: &mut Window,
        cx: &mut App,
    ) -> Self::PrepaintState;

    // Phase 3: Actual rendering
    fn paint(
        &mut self,
        id: Option<&GlobalElementId>,
        bounds: Bounds<Pixels>,
        request_layout: &mut Self::RequestLayoutState,
        prepaint: &mut Self::PrepaintState,
        window: &mut Window,
        cx: &mut App,
    );
}
```

**Key Insight:** This three-phase approach separates concerns:

1. **request_layout** - Calculate sizes without knowing position
2. **prepaint** - Prepare rendering state with final bounds
3. **paint** - Actual GPU drawing operations

---

## 📑 Pane & Panel System

### Pane Architecture

Zed's `Pane` struct manages tabs within a workspace region:

```rust
pub struct Pane {
    // Tab management
    tabs: Vec<Box<dyn ItemHandle>>,
    pinned_tabs: Vec<Box<dyn ItemHandle>>,
    active_tab_index: usize,

    // Drag-drop state
    drag_state: Option<DraggedTab>,

    // Layout
    bounds: Bounds<Pixels>,

    // History
    navigation_history: VecDeque<NavigationEntry>,
}

// Tab drag-drop between panes
pub struct DraggedTab {
    pub pane: WeakView<Pane>,
    pub tab_index: usize,
    pub item: Box<dyn ItemHandle>,
}
```

### PaneGroup for Splits

```rust
// Recursive structure for split panes
pub enum PaneGroup {
    Pane(View<Pane>),
    Split {
        direction: SplitDirection,
        children: Vec<PaneGroup>,
        // Relative sizes (0.0 - 1.0)
        sizes: Vec<f32>,
    },
}

pub enum SplitDirection {
    Horizontal, // Left-Right
    Vertical,   // Top-Bottom
}
```

### Split Actions

```rust
// Available split operations
pub enum SplitAction {
    SplitRight,
    SplitLeft,
    SplitUp,
    SplitDown,
}

// Context menu for tabs
fn build_tab_context_menu(pane: &Pane, tab_index: usize) -> Menu {
    Menu::new()
        .item("Close Tab", CloseTab { tab_index })
        .item("Close Other Tabs", CloseOtherTabs { tab_index })
        .item("Close Tabs to Right", CloseTabsToRight { tab_index })
        .separator()
        .item("Split Right", SplitRight { tab_index })
        .item("Split Down", SplitDown { tab_index })
        .separator()
        .item("Pin Tab", PinTab { tab_index })
        .item("Move to New Window", MoveToNewWindow { tab_index })
}
```

---

## 🎛️ Panel System

### DockPosition Enum

```rust
pub enum DockPosition {
    Left,
    Bottom,
    Right,
}

// Panel trait
pub trait Panel: EventEmitter<PanelEvent> {
    fn position(&self, cx: &App) -> DockPosition;
    fn position_is_valid(&self, position: DockPosition) -> bool;
    fn set_position(&mut self, position: DockPosition, cx: &mut App);

    // Zoom support
    fn is_zoomed(&self, cx: &App) -> bool;
    fn set_zoomed(&mut self, zoomed: bool, cx: &mut App);

    // Visibility
    fn is_active(&self, cx: &App) -> bool;
    fn set_active(&mut self, active: bool, cx: &mut App);
}
```

### Terminal Panel Example

```rust
pub struct TerminalPanel {
    position: DockPosition,
    zoomed: bool,
    active: bool,
    terminals: Vec<TerminalView>,
    active_terminal_index: usize,
}

impl Panel for TerminalPanel {
    fn position(&self, _cx: &App) -> DockPosition {
        self.position
    }

    fn position_is_valid(&self, position: DockPosition) -> bool {
        // Terminal can be at Left, Bottom, or Right
        matches!(position, DockPosition::Left | DockPosition::Bottom | DockPosition::Right)
    }

    fn set_zoomed(&mut self, zoomed: bool, cx: &mut App) {
        self.zoomed = zoomed;
        cx.notify();
    }
}
```

---

## 🔧 Workspace Persistence

### Serialization

Zed persists workspace layout for restoration:

```rust
#[derive(Serialize, Deserialize)]
pub struct SerializedPaneGroup {
    pub direction: Option<SplitDirection>,
    pub children: Vec<SerializedPaneGroupChild>,
    pub sizes: Vec<f32>,
}

#[derive(Serialize, Deserialize)]
pub enum SerializedPaneGroupChild {
    Pane(SerializedPane),
    Group(SerializedPaneGroup),
}

#[derive(Serialize, Deserialize)]
pub struct SerializedPane {
    pub tabs: Vec<SerializedItem>,
    pub pinned_count: usize,
    pub active_tab_index: usize,
}

#[derive(Serialize, Deserialize)]
pub struct SerializedItem {
    pub kind: String,  // "editor", "terminal", "assistant", etc.
    pub serialized_data: serde_json::Value,
}
```

---

## 🎯 Applicable Patterns for NEURECTOMY

### 1. Three-Phase Rendering

**Implementation Priority: MEDIUM** (for custom components)

While React handles most rendering, custom canvas-based components could benefit:

```typescript
// Adaptation for TypeScript/React
interface CustomElement {
  // Phase 1: Calculate desired size
  measureLayout(): { width: number; height: number };

  // Phase 2: Prepare with final bounds
  prepareRender(bounds: DOMRect): PrepareState;

  // Phase 3: Actual rendering
  render(ctx: CanvasRenderingContext2D, state: PrepareState): void;
}
```

### 2. PaneGroup Recursive Structure

**Implementation Priority: HIGH**

```typescript
// NEURECTOMY PaneGroup implementation
type PaneGroup =
  | { type: 'pane'; pane: PaneData }
  | {
      type: 'split';
      direction: 'horizontal' | 'vertical';
      children: PaneGroup[];
      sizes: number[]; // Relative sizes (0-1)
    };

interface PaneData {
  id: string;
  tabs: TabItem[];
  pinnedTabs: TabItem[];
  activeTabIndex: number;
  navigationHistory: NavigationEntry[];
}

// Example workspace structure
const workspace: PaneGroup = {
  type: 'split',
  direction: 'horizontal',
  sizes: [0.3, 0.7],
  children: [
    // Left pane
    { type: 'pane', pane: { id: 'left', tabs: [...] } },
    // Right split (vertical)
    {
      type: 'split',
      direction: 'vertical',
      sizes: [0.6, 0.4],
      children: [
        { type: 'pane', pane: { id: 'main', tabs: [...] } },
        { type: 'pane', pane: { id: 'bottom', tabs: [...] } },
      ]
    }
  ]
};
```

### 3. Tab Pinning System

**Implementation Priority: MEDIUM**

```typescript
interface TabPinning {
  // Pin tab at current position
  pinTab(paneId: string, tabIndex: number): void;

  // Unpin tab (moves after last pinned)
  unpinTab(paneId: string, tabIndex: number): void;

  // Visual indicator
  isPinned(paneId: string, tabIndex: number): boolean;
}

// Tab component with pin indicator
<Tab pinned={isPinned}>
  {pinned && <PinIcon className="tab-pin-icon" />}
  <TabIcon type={tab.type} />
  <TabLabel>{tab.label}</TabLabel>
  {!pinned && <CloseButton />}
</Tab>
```

### 4. Panel Zoom Feature

**Implementation Priority: MEDIUM**

```typescript
// Panel zoom state
interface PanelState {
  position: "left" | "bottom" | "right";
  isZoomed: boolean;
  isActive: boolean;
}

// Zoom toggles panel to full size
function togglePanelZoom(panelId: string) {
  const panel = panels.get(panelId);
  if (panel.isZoomed) {
    // Restore original size
    restorePanelSize(panelId);
  } else {
    // Maximize panel
    maximizePanel(panelId);
  }
  panel.isZoomed = !panel.isZoomed;
}
```

### 5. Tab Drag-Drop Between Panes

**Implementation Priority: HIGH**

```typescript
interface DraggedTab {
  sourcePane: string;
  tabIndex: number;
  item: TabItem;
}

// Drop zones
type DropZone =
  | { type: 'tab-bar'; paneId: string; insertIndex: number }
  | { type: 'split'; paneId: string; direction: SplitDirection };

// Visual feedback during drag
<TabDropIndicator
  visible={isDraggingOver}
  position={dropZone.insertIndex}
/>

// Split drop zones at pane edges
<SplitDropZone
  position="right"
  visible={isDraggingOver}
  onDrop={() => splitAndDrop('right')}
/>
```

---

## 📊 Performance Optimizations from Zed

### 1. GPU-Accelerated Rendering

Zed uses GPU for all rendering. While NEURECTOMY uses web tech, consider:

- Use `transform` and `opacity` for animations (GPU-accelerated)
- Avoid layout thrashing
- Use `will-change` for elements that animate

### 2. Virtualized Lists

Zed only renders visible items:

```typescript
// Virtual list for large file trees
<VirtualizedList
  items={files}
  itemHeight={24}
  renderItem={(file, index) => <FileRow file={file} />}
  overscan={10} // Render 10 extra items above/below
/>
```

### 3. Debounced Layout Updates

```typescript
// Debounce layout recalculation
const debouncedLayout = useMemo(
  () =>
    debounce((sizes: number[]) => {
      updatePaneGroupSizes(sizes);
    }, 16), // ~60fps
  []
);
```

---

## 🎨 Visual Design from Zed

### Tab Bar Styling

```css
/* Zed-inspired tab bar */
.tab-bar {
  display: flex;
  align-items: center;
  height: 34px;
  background: var(--tab-bar-background);
  border-bottom: 1px solid var(--border);
}

.tab {
  display: flex;
  align-items: center;
  padding: 0 12px;
  height: 100%;
  cursor: pointer;
  border-right: 1px solid var(--border);

  &.active {
    background: var(--tab-active-background);
  }

  &.pinned {
    padding: 0 8px;

    .tab-close {
      display: none;
    }
  }
}

.tab-pin-icon {
  width: 12px;
  height: 12px;
  margin-right: 4px;
  opacity: 0.7;
}
```

### Split Resize Handle

```css
/* Zed-style resize handle */
.split-handle {
  position: absolute;
  z-index: 10;

  &.horizontal {
    width: 4px;
    height: 100%;
    cursor: col-resize;
  }

  &.vertical {
    width: 100%;
    height: 4px;
    cursor: row-resize;
  }

  &:hover {
    background: var(--accent-color);
    opacity: 0.5;
  }

  &.dragging {
    background: var(--accent-color);
    opacity: 1;
  }
}
```

---

## 📋 Checklist for NEURECTOMY Implementation

### Phase 1: Pane System

- [ ] Implement PaneGroup recursive data structure
- [ ] Create Pane component with tab management
- [ ] Add split resize handles
- [ ] Implement keyboard navigation between panes

### Phase 2: Tab Management

- [ ] Tab drag-and-drop within pane
- [ ] Tab drag-and-drop between panes
- [ ] Tab pinning (visual + behavior)
- [ ] Tab context menu with split options

### Phase 3: Advanced Features

- [ ] Panel zoom mode
- [ ] Workspace serialization
- [ ] Workspace restoration
- [ ] Navigation history (forward/back)

### Phase 4: Performance

- [ ] Virtualize large tab lists
- [ ] Debounce resize updates
- [ ] Optimize for GPU rendering
- [ ] Lazy load pane contents

---

## 🔧 Technical Notes

### React-based PaneGroup Component

```typescript
// Core PaneGroup component
function PaneGroupView({ group, onResize }: PaneGroupProps) {
  if (group.type === 'pane') {
    return <PaneView pane={group.pane} />;
  }

  const { direction, children, sizes } = group;

  return (
    <SplitView
      direction={direction}
      sizes={sizes}
      onResize={onResize}
    >
      {children.map((child, index) => (
        <PaneGroupView
          key={index}
          group={child}
          onResize={(newSizes) => handleChildResize(index, newSizes)}
        />
      ))}
    </SplitView>
  );
}
```

### Tab Drag System

```typescript
// Using dnd-kit or similar
const [activeTab, setActiveTab] = useState<DraggedTab | null>(null);

function handleDragStart(event: DragStartEvent) {
  const { paneId, tabIndex } = event.active.data;
  setActiveTab({ paneId, tabIndex, item: getTab(paneId, tabIndex) });
}

function handleDragEnd(event: DragEndEvent) {
  if (!event.over) return;

  const { type, paneId, position } = event.over.data;

  if (type === "tab-bar") {
    moveTabToPane(activeTab, paneId, position);
  } else if (type === "split-zone") {
    splitPaneAndMoveTab(activeTab, paneId, position);
  }

  setActiveTab(null);
}
```

---

_Analysis completed for NEURECTOMY IDE reference_
_Source: zed-industries/zed_
