# IntelliJ IDEA UI/UX Patterns Analysis

## Overview

IntelliJ IDEA represents the gold standard for enterprise IDE development, with decades of refinement in developer experience. This analysis extracts patterns for tool windows, docking systems, editor tabs, and professional IDE features that can elevate NEURECTOMY's interface to enterprise quality.

---

## 🏗️ Tool Window Architecture

### Tool Window System

IntelliJ's tool window system is one of the most sophisticated in any IDE:

```java
// Tool window positioning
public enum ToolWindowAnchor {
    TOP,
    LEFT,
    BOTTOM,
    RIGHT
}

// Tool window types
public interface ToolWindow {
    String getId();
    ToolWindowAnchor getAnchor();
    ToolWindowType getType();

    // Visibility
    boolean isVisible();
    void show(@Nullable Runnable runnable);
    void hide(@Nullable Runnable runnable);

    // Content management
    ContentManager getContentManager();

    // Docking
    boolean isFloating();
    boolean isDocked();
    boolean isWindowed();
}

public enum ToolWindowType {
    DOCKED,
    FLOATING,
    WINDOWED,
    SLIDING
}
```

### Tool Window Content UI

```java
// ToolWindowContentUi manages tabs within a tool window
public class ToolWindowContentUi {
    private final JBTabbedPane myTabbedPane;
    private final ContentManager myContentManager;

    // Tab configuration
    public void setTabPlacement(int tabPlacement) {
        // TOP, BOTTOM, LEFT, RIGHT
    }

    // Content management
    public void addContent(@NotNull Content content);
    public void removeContent(@NotNull Content content);
    public void selectContent(@NotNull Content content);
}
```

### Dockable Editor Tabs

```java
// DockableEditorTabbedContainer for editor window docking
public class DockableEditorTabbedContainer {
    private final EditorTabbedContainer myEditorTabbedContainer;

    // Dock/undock operations
    public void dock();
    public void undock();
    public void moveToWindow();

    // Drag-drop support
    public boolean isDragEnabled();
    public void handleDrop(DragEvent event);
}
```

---

## 📐 Splitter Components

### Basic Splitter

```java
public class Splitter extends JPanel {
    // Orientation
    private boolean myVerticalSplit;  // true = top/bottom, false = left/right

    // Proportion (0.0 - 1.0)
    private float myProportion = 0.5f;

    // Components
    private JComponent myFirstComponent;
    private JComponent mySecondComponent;

    // Constraints
    private int myMinFirstSize = 0;
    private int myMinSecondSize = 0;

    // Events
    public void setProportion(float proportion);
    public void setFirstComponent(@Nullable JComponent component);
    public void setSecondComponent(@Nullable JComponent component);
}
```

### Three Components Splitter

For complex layouts with three sections:

```java
// ThreeComponentsSplitter - used in Designer panels
public class ThreeComponentsSplitter extends JPanel {
    private JComponent myInnerComponent;  // Center
    private JComponent myFirstComponent;   // Left/Top
    private JComponent myLastComponent;    // Right/Bottom

    private int myFirstSize;
    private int myLastSize;

    private Orientation myOrientation;  // VERTICAL or HORIZONTAL

    // Configuration
    public void setOrientation(Orientation orientation);
    public void setFirstSize(int size);
    public void setLastSize(int size);
    public void setInnerComponent(@Nullable JComponent component);
}
```

### One Pixel Splitter

Ultra-thin splitter for tool windows:

```java
// OnePixelSplitter - minimal visual footprint
public class OnePixelSplitter extends Splitter {
    // Hover detection for resize
    private static final int RESIZE_ZONE = 7;  // pixels

    @Override
    protected void paintComponent(Graphics g) {
        // Paint 1px divider line
        g.setColor(UIUtil.getSplitterColor());
        if (isVertical()) {
            g.drawLine(0, dividerY, getWidth(), dividerY);
        } else {
            g.drawLine(dividerX, 0, dividerX, getHeight());
        }
    }
}
```

---

## 📑 Editor Tab System

### EditorTabbedContainer

```java
// EditorTabbedContainer manages editor tabs
public class EditorTabbedContainer {
    private final JBEditorTabs myTabs;

    // Tab operations
    public void addTab(TabInfo info);
    public void removeTab(TabInfo info);
    public void selectTab(TabInfo info);

    // Tab arrangement
    public void setTabPlacement(int placement);  // TOP, BOTTOM, LEFT, RIGHT
    public void setTabDraggingEnabled(boolean enabled);
}

// JBEditorTabs - sophisticated tab component
public class JBEditorTabs extends JBTabs {
    // Pin support
    public void setPinned(TabInfo info, boolean pinned);

    // Tab decorations
    public void setTabLabelActionsAutoHide(boolean autoHide);

    // Drag-drop
    public void setDragEnabled(boolean enabled);
    public void setDropHandler(TabsDropHandler handler);
}
```

### Tab Info Structure

```java
public class TabInfo {
    private String myText;
    private Icon myIcon;
    private JComponent myComponent;

    // State
    private boolean myPinned;
    private boolean myHidden;

    // Actions on tab
    private ActionGroup myTabActions;      // Actions shown on tab
    private ActionGroup mySideActions;     // Actions on right side

    // Tab tooltip
    private String myTooltipText;

    // Modified indicator
    private Color myTabColor;
}
```

### Tab Placement Options

```java
// IntelliJ supports 4 tab placements
public enum TabPlacement {
    TOP,      // Default - tabs above content
    BOTTOM,   // Tabs below content
    LEFT,     // Tabs on left (rotated text)
    RIGHT     // Tabs on right (rotated text)
}

// User preference
EditorSettings.setTabPlacement(TabPlacement.TOP);
```

---

## 🖥️ Workbench Layout

### Main Window Structure

```
┌─────────────────────────────────────────────────────────────────────┐
│                            MENU BAR                                  │
├─────────────────────────────────────────────────────────────────────┤
│                           TOOLBAR                                    │
├───────┬─────────────────────────────────────────────────────┬───────┤
│       │              EDITOR TABS                            │       │
│ TOOL  │  ┌─────────┬─────────┬─────────┐                   │ TOOL  │
│ WIN   │  │  Tab 1  │  Tab 2  │  Tab 3  │                   │ WIN   │
│ LEFT  ├──┴─────────┴─────────┴─────────┴───────────────────┤ RIGHT │
│       │                                                     │       │
│ Pro-  │                  EDITOR AREA                        │ Str-  │
│ ject  │               (can be split)                        │ uct-  │
│       │                                                     │ ure   │
│ Fav-  │                                                     │       │
│ orit- │                                                     │       │
│ es    │                                                     │       │
│       ├─────────────────────────────────────────────────────┤       │
│       │              TOOL WINDOW BOTTOM                      │       │
│       │         (Terminal/Version Control/etc)              │       │
├───────┴─────────────────────────────────────────────────────┴───────┤
│                          STATUS BAR                                  │
└─────────────────────────────────────────────────────────────────────┘
```

### Tool Window Buttons (Stripes)

```java
// Tool window buttons on the edges
public class StripeButton extends JToggleButton {
    private final ToolWindow myToolWindow;

    // Icon with optional badge
    public void updateIcon();

    // Drag to reposition
    public void enableDragAndDrop();

    // Click behavior
    @Override
    public void processMouseEvent(MouseEvent e) {
        if (e.getID() == MouseEvent.MOUSE_CLICKED) {
            if (myToolWindow.isVisible()) {
                myToolWindow.hide();
            } else {
                myToolWindow.show();
            }
        }
    }
}
```

---

## 🎯 Applicable Patterns for NEURECTOMY

### 1. Tool Window System

**Implementation Priority: CRITICAL**

```typescript
// NEURECTOMY Tool Window implementation
interface ToolWindow {
  id: string;
  title: string;
  icon: string;
  anchor: "left" | "right" | "bottom";
  type: "docked" | "floating" | "windowed";

  // Content tabs within tool window
  contents: ToolWindowContent[];
  activeContentIndex: number;

  // State
  isVisible: boolean;
  isPinned: boolean; // Auto-hide when unpinned
  width?: number; // For left/right
  height?: number; // For bottom
}

interface ToolWindowContent {
  id: string;
  title: string;
  icon?: string;
  component: React.ComponentType;
  closeable: boolean;
}

// Tool window manager
interface ToolWindowManager {
  windows: Map<string, ToolWindow>;

  // Operations
  show(windowId: string): void;
  hide(windowId: string): void;
  toggle(windowId: string): void;

  // Docking
  setAnchor(windowId: string, anchor: ToolWindowAnchor): void;
  setType(windowId: string, type: ToolWindowType): void;

  // Content
  addContent(windowId: string, content: ToolWindowContent): void;
  removeContent(windowId: string, contentId: string): void;
}
```

### 2. Splitter Component

**Implementation Priority: HIGH**

```typescript
// Three-way splitter for complex layouts
interface ThreeComponentsSplitterProps {
  orientation: 'horizontal' | 'vertical';

  // Components
  firstComponent?: React.ReactNode;
  innerComponent: React.ReactNode;
  lastComponent?: React.ReactNode;

  // Sizes (pixels)
  firstSize: number;
  lastSize: number;

  // Constraints
  minFirstSize?: number;
  minLastSize?: number;

  // Events
  onFirstResize?: (size: number) => void;
  onLastResize?: (size: number) => void;
}

// Usage
<ThreeComponentsSplitter
  orientation="horizontal"
  firstComponent={<LeftToolWindow />}
  innerComponent={<EditorArea />}
  lastComponent={<RightToolWindow />}
  firstSize={250}
  lastSize={300}
  minFirstSize={100}
  minLastSize={100}
/>
```

### 3. Tab Placement Options

**Implementation Priority: MEDIUM**

```typescript
// Support multiple tab positions
interface TabContainerProps {
  tabs: Tab[];
  activeTabIndex: number;
  placement: "top" | "bottom" | "left" | "right";

  // Features
  draggable?: boolean;
  pinnable?: boolean;
  closeable?: boolean;

  // Events
  onTabSelect: (index: number) => void;
  onTabClose?: (index: number) => void;
  onTabPin?: (index: number) => void;
  onTabReorder?: (fromIndex: number, toIndex: number) => void;
}

// Tab placement styling
const tabPlacementStyles = {
  top: { flexDirection: "column" },
  bottom: { flexDirection: "column-reverse" },
  left: { flexDirection: "row", writingMode: "vertical-rl" },
  right: { flexDirection: "row-reverse", writingMode: "vertical-lr" },
};
```

### 4. Tool Window Stripe (Activity Bar)

**Implementation Priority: HIGH**

```typescript
// Tool window buttons on the side
interface ToolWindowStripe {
  position: 'left' | 'right' | 'bottom';
  buttons: ToolWindowButton[];
}

interface ToolWindowButton {
  id: string;
  icon: string;
  tooltip: string;
  badge?: {
    count: number;
    type: 'info' | 'warning' | 'error';
  };
  isActive: boolean;
}

// Stripe component
<ToolWindowStripe position="left">
  {buttons.map(button => (
    <StripeButton
      key={button.id}
      icon={button.icon}
      tooltip={button.tooltip}
      badge={button.badge}
      isActive={button.isActive}
      onClick={() => toggleToolWindow(button.id)}
      onDrag={() => repositionToolWindow(button.id)}
    />
  ))}
</ToolWindowStripe>
```

### 5. Editor Splitter Service

**Implementation Priority: HIGH**

```typescript
// Service for managing editor splits
interface EditorSplitterService {
  // Current layout
  layout: EditorLayout;

  // Split operations
  splitRight(editorId: string): string; // Returns new editor ID
  splitDown(editorId: string): string;
  splitLeft(editorId: string): string;
  splitUp(editorId: string): string;

  // Unsplit
  unsplit(editorId: string): void;
  unsplitAll(): void;

  // Navigation
  focusNext(): void;
  focusPrevious(): void;
  focusLeft(): void;
  focusRight(): void;
  focusUp(): void;
  focusDown(): void;

  // Layout
  maximize(editorId: string): void;
  restore(): void;
  resetLayout(): void;
}

type EditorLayout =
  | {
      type: "leaf";
      editorId: string;
    }
  | {
      type: "split";
      direction: "horizontal" | "vertical";
      children: EditorLayout[];
      sizes: number[];
    };
```

### 6. Run/Debug Tool Window Pattern

**Implementation Priority: MEDIUM**

```typescript
// Run/Debug configuration pattern
interface RunConfiguration {
  id: string;
  name: string;
  type: 'run' | 'debug';
  icon: string;

  // Execution
  command: string;
  args: string[];
  env: Record<string, string>;
  workingDir: string;

  // Output
  outputPane: 'terminal' | 'console';
}

// Run tool window content
<RunToolWindow>
  <ConfigurationSelector
    configurations={configs}
    selected={activeConfig}
    onSelect={setActiveConfig}
  />
  <RunActions>
    <RunButton onClick={run} />
    <DebugButton onClick={debug} />
    <StopButton onClick={stop} disabled={!isRunning} />
    <RestartButton onClick={restart} />
  </RunActions>
  <OutputTabs>
    {runSessions.map(session => (
      <OutputTab
        key={session.id}
        session={session}
        onClose={() => closeSession(session.id)}
      />
    ))}
  </OutputTabs>
</RunToolWindow>
```

---

## 🎨 Visual Design from IntelliJ

### Tool Window Header

```css
/* IntelliJ-style tool window header */
.tool-window-header {
  display: flex;
  align-items: center;
  height: 28px;
  padding: 0 8px;
  background: var(--tool-window-header-bg);
  border-bottom: 1px solid var(--border);
}

.tool-window-title {
  font-size: 12px;
  font-weight: 500;
  margin-right: auto;
}

.tool-window-actions {
  display: flex;
  gap: 2px;
}

.tool-window-action-button {
  width: 20px;
  height: 20px;
  padding: 2px;
  border-radius: 3px;
  opacity: 0.7;

  &:hover {
    background: var(--hover-bg);
    opacity: 1;
  }
}
```

### Stripe Button Styling

```css
/* Tool window stripe button */
.stripe-button {
  position: relative;
  width: 20px;
  height: 42px;
  padding: 2px;
  cursor: pointer;

  /* Rotated text for side buttons */
  writing-mode: vertical-rl;
  text-orientation: mixed;

  &.active {
    background: var(--active-bg);

    &::before {
      content: "";
      position: absolute;
      left: 0;
      top: 0;
      bottom: 0;
      width: 2px;
      background: var(--accent);
    }
  }

  &:hover:not(.active) {
    background: var(--hover-bg);
  }
}

.stripe-button-badge {
  position: absolute;
  top: 4px;
  right: 2px;
  min-width: 14px;
  height: 14px;
  padding: 0 4px;
  font-size: 10px;
  border-radius: 7px;
  background: var(--badge-bg);
  color: var(--badge-text);
}
```

### One Pixel Splitter

```css
/* Ultra-thin splitter */
.one-pixel-splitter {
  position: relative;
  flex-shrink: 0;

  &.horizontal {
    width: 1px;
    height: 100%;
    cursor: col-resize;
  }

  &.vertical {
    width: 100%;
    height: 1px;
    cursor: row-resize;
  }

  /* Expand hit area */
  &::before {
    content: "";
    position: absolute;
    background: transparent;

    &.horizontal {
      left: -3px;
      right: -3px;
      top: 0;
      bottom: 0;
    }

    &.vertical {
      top: -3px;
      bottom: -3px;
      left: 0;
      right: 0;
    }
  }

  /* Visual line */
  &::after {
    content: "";
    position: absolute;
    background: var(--border);

    &.horizontal {
      left: 0;
      width: 1px;
      top: 0;
      bottom: 0;
    }

    &.vertical {
      top: 0;
      height: 1px;
      left: 0;
      right: 0;
    }
  }

  &:hover::after,
  &.dragging::after {
    background: var(--accent);
  }
}
```

---

## 📋 Checklist for NEURECTOMY Implementation

### Phase 1: Tool Window Framework

- [ ] ToolWindow component with content management
- [ ] Tool window manager service
- [ ] Tool window stripe (buttons)
- [ ] Show/hide/toggle operations
- [ ] Docked/floating/windowed modes

### Phase 2: Splitter System

- [ ] Basic two-way splitter
- [ ] Three-components splitter
- [ ] One-pixel splitter variant
- [ ] Drag resize with constraints
- [ ] Double-click to reset

### Phase 3: Editor Splits

- [ ] Editor splitter service
- [ ] Split operations (all directions)
- [ ] Focus navigation
- [ ] Maximize/restore
- [ ] Layout persistence

### Phase 4: Tab Enhancements

- [ ] Tab placement options
- [ ] Tab pinning
- [ ] Tab actions (close, split, etc.)
- [ ] Drag-and-drop reordering
- [ ] Cross-group tab dragging

### Phase 5: Polish

- [ ] Auto-hide mode for tool windows
- [ ] Recent tools popup
- [ ] Keyboard shortcuts for all operations
- [ ] State persistence

---

## 🔧 Technical Implementation Notes

### Tool Window State Persistence

```typescript
// Persist tool window layout
interface ToolWindowState {
  windows: {
    [id: string]: {
      anchor: ToolWindowAnchor;
      type: ToolWindowType;
      isVisible: boolean;
      size: number;
      activeContentId: string;
    };
  };
  stripeOrder: {
    left: string[];
    right: string[];
    bottom: string[];
  };
}

// Save/restore
function saveToolWindowState(): ToolWindowState;
function restoreToolWindowState(state: ToolWindowState): void;
```

### Keyboard Shortcuts

IntelliJ uses numbered tool windows:

```typescript
const toolWindowShortcuts = {
  "Alt+1": "Project",
  "Alt+2": "Favorites",
  "Alt+3": "Find",
  "Alt+4": "Run",
  "Alt+5": "Debug",
  "Alt+6": "Problems",
  "Alt+7": "Structure",
  "Alt+9": "Version Control",
  "Alt+F12": "Terminal",
};
```

### Recent Tool Windows

```typescript
// Track recently used tool windows
const recentToolWindows: string[] = [];

function trackToolWindowUsage(windowId: string) {
  recentToolWindows.unshift(windowId);
  recentToolWindows.splice(10); // Keep last 10
}

// Popup for quick access
<RecentToolWindowsPopup
  windows={recentToolWindows}
  onSelect={(id) => showToolWindow(id)}
/>
```

---

_Analysis completed for NEURECTOMY IDE reference_
_Source: JetBrains/intellij-community_
