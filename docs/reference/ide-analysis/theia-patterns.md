# Eclipse Theia UI/UX Patterns Analysis

## Overview

Eclipse Theia is a modern, cloud & desktop IDE framework built with TypeScript and React. It powers numerous cloud IDEs (Gitpod, Arduino IDE, etc.) and shares architectural philosophy with VS Code while offering greater extensibility. This analysis extracts patterns for widget systems, dependency injection, and modular IDE architecture applicable to NEURECTOMY.

---

## 🏗️ Core Architecture

### Widget System

Theia uses Phosphor.js (now Lumino) for its widget system:

```typescript
// Core widget interface
interface Widget {
  id: string;
  title: Title<Widget>;
  node: HTMLElement;

  // Lifecycle
  isAttached: boolean;
  isVisible: boolean;
  isDisposed: boolean;

  // Methods
  show(): void;
  hide(): void;
  close(): void;
  activate(): void;

  // Events
  disposed: Signal<Widget, void>;
}

// Title for tabs and labels
interface Title<T extends Widget> {
  label: string;
  caption: string;
  iconClass: string;
  closable: boolean;

  // State
  className: string;
  dataset: DOMStringMap;
}
```

### Application Shell

```typescript
// ApplicationShell - main layout container
class ApplicationShell extends Widget {
  // Areas
  readonly topPanel: Panel;
  readonly leftPanel: BoxPanel;
  readonly mainPanel: DockPanel;
  readonly bottomPanel: BoxPanel;
  readonly rightPanel: BoxPanel;

  // Status bar
  readonly statusBar: StatusBar;

  // Operations
  addWidget(widget: Widget, options: WidgetAddOptions): void;
  activateWidget(id: string): Widget | undefined;
  revealWidget(id: string): Widget | undefined;

  // Layout
  getWidgets(area: ApplicationShell.Area): Widget[];
  getTabBarRendererFactory(): TabBar.IRenderer;
}

namespace ApplicationShell {
  type Area = "top" | "left" | "main" | "bottom" | "right";

  interface WidgetAddOptions {
    area: Area;
    rank?: number;
    ref?: Widget;
    mode?: DockPanel.InsertMode;
  }
}
```

---

## 📑 DockPanel System

### DockPanel for Editor Area

```typescript
// DockPanel - main editor area with docking support
class DockPanel extends Widget {
  // Insert modes for new widgets
  static InsertMode = {
    "split-top": 0,
    "split-left": 1,
    "split-right": 2,
    "split-bottom": 3,
    "tab-before": 4,
    "tab-after": 5,
  };

  // Add widget with position
  addWidget(widget: Widget, options?: DockPanel.AddOptions): void;

  // Save/restore layout
  saveLayout(): DockPanel.LayoutConfig;
  restoreLayout(config: DockPanel.LayoutConfig): void;
}

// Layout configuration (serializable)
namespace DockPanel {
  interface LayoutConfig {
    main: IAreaConfig | null;
  }

  interface IAreaConfig {
    type: "tab-area" | "split-area";
    orientation?: "horizontal" | "vertical";
    children?: IAreaConfig[];
    sizes?: number[];
    widgets?: string[];
    currentIndex?: number;
  }
}
```

### Tab Bar Configuration

```typescript
// Tab bar for widget groups
class TabBar<T> extends Widget {
  // Tabs
  readonly titles: ReadonlyArray<Title<T>>;
  readonly currentTitle: Title<T> | null;

  // Configuration
  allowDeselect: boolean;
  tabsMovable: boolean;
  insertBehavior: TabBar.InsertBehavior;
  removeBehavior: TabBar.RemoveBehavior;

  // Signals
  currentChanged: Signal<TabBar<T>, TabBar.ICurrentChangedArgs<T>>;
  tabMoved: Signal<TabBar<T>, TabBar.ITabMovedArgs<T>>;
  tabDetachRequested: Signal<TabBar<T>, TabBar.ITabDetachRequestedArgs<T>>;
}
```

---

## 🔧 Frontend Application

### Frontend Application Module

```typescript
// Theia frontend application structure
@injectable()
class FrontendApplication {
  // Shell components
  @inject(ApplicationShell) protected shell: ApplicationShell;
  @inject(WidgetManager) protected widgetManager: WidgetManager;
  @inject(KeybindingRegistry) protected keybindingRegistry: KeybindingRegistry;

  // Application lifecycle
  async start(config: FrontendApplicationConfig): Promise<void> {
    await this.initializeShell();
    await this.startContributions();
    await this.revealShell();
  }

  // Shell management
  protected async initializeShell(): Promise<void> {
    this.shell.addClass("theia-ApplicationShell");
  }
}

// Widget contribution (declare widgets in DI)
interface FrontendApplicationContribution {
  onStart?(app: FrontendApplication): MaybePromise<void>;
  configure?(app: FrontendApplication): void;
  initialize?(app: FrontendApplication): MaybePromise<void>;
}
```

### Widget Manager

```typescript
// WidgetManager - creates and caches widgets
@injectable()
class WidgetManager {
  // Factory map
  protected readonly factories = new Map<string, WidgetFactory>();

  // Cache
  protected readonly widgets = new Map<string, Widget>();

  // Create or get widget
  async getOrCreateWidget<T extends Widget>(
    factoryId: string,
    options?: WidgetOptions
  ): Promise<T | undefined>;

  // Get existing widget
  getWidget<T extends Widget>(
    factoryId: string,
    options?: WidgetOptions
  ): T | undefined;

  // Register factory
  registerFactory(factory: WidgetFactory): Disposable;
}

// Widget factory pattern
interface WidgetFactory {
  id: string;
  createWidget(options?: WidgetOptions): MaybePromise<Widget>;
}
```

---

## 🎛️ View Container System

### View Container

```typescript
// ViewContainer - collapsible section container
@injectable()
class ViewContainer extends BaseWidget {
  // Parts (collapsible sections)
  readonly parts: ViewContainerPart[];

  // Add part
  addWidget(widget: Widget, options?: ViewContainer.AddOptions): Disposable;

  // Part management
  protected createPart(widget: Widget): ViewContainerPart;
}

// View container part (collapsible section)
class ViewContainerPart extends BaseWidget {
  readonly collapsed: boolean;
  readonly header: ViewContainerPartHeader;
  readonly body: Widget;

  setCollapsed(collapsed: boolean): void;
  setHidden(hidden: boolean): void;
}
```

### Side Panel

```typescript
// SidePanel - container for side panels
class SidePanel extends BoxPanel {
  readonly title: Title<SidePanel>;

  // Collapse/expand
  collapse(): void;
  expand(): void;
  toggle(): void;

  // Size
  resize(size: number): void;
}

// Side panel handler
@injectable()
class SidePanelHandler {
  // Current state
  get state(): SidePanel.State;

  // Operations
  activate(): void;
  collapse(): void;
  expand(): void;
  toggle(): void;

  // Size management
  resize(size: number): void;
  setLastPanelSize(size: number): void;
}
```

---

## 📐 Box Panel Layout

### Box Panel

```typescript
// BoxPanel - horizontal or vertical layout
class BoxPanel extends Panel {
  // Direction
  direction: BoxPanel.Direction; // 'left-to-right', 'right-to-left', 'top-to-bottom', 'bottom-to-top'

  // Alignment
  alignment: BoxPanel.Alignment; // 'start', 'center', 'end', 'justify'

  // Spacing
  spacing: number;

  // Add with sizing hints
  addWidget(widget: Widget, options?: BoxPanel.AddOptions): void;
}

namespace BoxPanel {
  interface AddOptions {
    stretch?: number; // Stretch factor (0 = fixed size)
    rank?: number; // Insertion order
  }

  type Direction =
    | "left-to-right"
    | "right-to-left"
    | "top-to-bottom"
    | "bottom-to-top";
}
```

### Split Panel

```typescript
// SplitPanel - resizable split layout
class SplitPanel extends Panel {
  // Orientation
  orientation: SplitPanel.Orientation; // 'horizontal' | 'vertical'

  // Handles
  readonly handles: HTMLDivElement[];

  // Operations
  setRelativeSizes(sizes: number[]): void;
  relativeSizes(): number[];

  // Handle events
  protected onHandleEvent(event: MouseEvent, handle: HTMLDivElement): void;
}
```

---

## 🎯 Applicable Patterns for NEURECTOMY

### 1. Widget Factory Pattern

**Implementation Priority: HIGH**

```typescript
// NEURECTOMY Widget Factory System
interface WidgetFactory<T extends Widget = Widget> {
  id: string;
  createWidget(options?: WidgetOptions): T | Promise<T>;
}

interface WidgetOptions {
  id?: string;
  [key: string]: unknown;
}

// Widget manager for centralized widget creation
class WidgetManager {
  private factories = new Map<string, WidgetFactory>();
  private widgets = new Map<string, Widget>();

  registerFactory(factory: WidgetFactory): void {
    this.factories.set(factory.id, factory);
  }

  async getOrCreateWidget<T extends Widget>(
    factoryId: string,
    options?: WidgetOptions
  ): Promise<T | undefined> {
    const cacheKey = this.getCacheKey(factoryId, options);

    if (this.widgets.has(cacheKey)) {
      return this.widgets.get(cacheKey) as T;
    }

    const factory = this.factories.get(factoryId);
    if (!factory) return undefined;

    const widget = (await factory.createWidget(options)) as T;
    this.widgets.set(cacheKey, widget);
    return widget;
  }
}
```

### 2. Application Shell Structure

**Implementation Priority: CRITICAL**

```typescript
// NEURECTOMY Application Shell
interface ApplicationShell {
  // Panel areas
  topPanel: Panel;
  leftPanel: SidePanel;
  mainPanel: DockPanel;
  bottomPanel: Panel;
  rightPanel: SidePanel;

  // Status bar
  statusBar: StatusBar;

  // Widget operations
  addWidget(widget: Widget, options: AddWidgetOptions): void;
  activateWidget(id: string): Widget | undefined;
  closeWidget(id: string): void;

  // Area queries
  getWidgets(area: ShellArea): Widget[];
  getCurrentWidget(area: ShellArea): Widget | undefined;

  // Layout
  saveLayout(): ShellLayout;
  restoreLayout(layout: ShellLayout): void;
}

type ShellArea = "top" | "left" | "main" | "bottom" | "right";

interface AddWidgetOptions {
  area: ShellArea;
  ref?: Widget;
  mode?: InsertMode;
  rank?: number;
}

type InsertMode =
  | "split-top"
  | "split-bottom"
  | "split-left"
  | "split-right"
  | "tab-before"
  | "tab-after";
```

### 3. Dock Panel Layout Serialization

**Implementation Priority: HIGH**

```typescript
// Serializable dock panel layout
interface DockLayoutConfig {
  main: AreaConfig | null;
}

interface AreaConfig {
  type: "tab-area" | "split-area";
  orientation?: "horizontal" | "vertical";
  widgets?: string[]; // Widget IDs for tab-area
  currentIndex?: number;
  children?: AreaConfig[]; // Child areas for split-area
  sizes?: number[]; // Relative sizes for split-area
}

// Example layout
const exampleLayout: DockLayoutConfig = {
  main: {
    type: "split-area",
    orientation: "horizontal",
    sizes: [0.3, 0.7],
    children: [
      {
        type: "tab-area",
        widgets: ["editor-1"],
        currentIndex: 0,
      },
      {
        type: "split-area",
        orientation: "vertical",
        sizes: [0.6, 0.4],
        children: [
          {
            type: "tab-area",
            widgets: ["editor-2", "editor-3"],
            currentIndex: 0,
          },
          {
            type: "tab-area",
            widgets: ["terminal-1"],
            currentIndex: 0,
          },
        ],
      },
    ],
  },
};
```

### 4. View Container with Collapsible Parts

**Implementation Priority: HIGH**

```typescript
// Collapsible view container (like VS Code's sidebar sections)
interface ViewContainerProps {
  id: string;
  title: string;
  parts: ViewContainerPart[];
}

interface ViewContainerPart {
  id: string;
  title: string;
  icon?: string;
  collapsed: boolean;
  component: React.ComponentType;
  toolbar?: ToolbarItem[];
}

// Component
function ViewContainer({ id, title, parts }: ViewContainerProps) {
  const [collapsedState, setCollapsedState] = useState<Record<string, boolean>>(
    Object.fromEntries(parts.map(p => [p.id, p.collapsed]))
  );

  return (
    <div className="view-container">
      {parts.map(part => (
        <ViewContainerSection
          key={part.id}
          title={part.title}
          icon={part.icon}
          collapsed={collapsedState[part.id]}
          onToggle={() => setCollapsedState(s => ({
            ...s,
            [part.id]: !s[part.id]
          }))}
          toolbar={part.toolbar}
        >
          <part.component />
        </ViewContainerSection>
      ))}
    </div>
  );
}
```

### 5. Side Panel Handler

**Implementation Priority: MEDIUM**

```typescript
// Side panel state management
interface SidePanelState {
  position: "left" | "right";
  isExpanded: boolean;
  size: number;
  lastSize: number; // Size before collapse
  activeWidget: string | null;
}

function useSidePanel(position: "left" | "right") {
  const [state, setState] = useState<SidePanelState>({
    position,
    isExpanded: true,
    size: 250,
    lastSize: 250,
    activeWidget: null,
  });

  const collapse = useCallback(() => {
    setState((s) => ({
      ...s,
      isExpanded: false,
      lastSize: s.size,
    }));
  }, []);

  const expand = useCallback(() => {
    setState((s) => ({
      ...s,
      isExpanded: true,
      size: s.lastSize,
    }));
  }, []);

  const toggle = useCallback(() => {
    state.isExpanded ? collapse() : expand();
  }, [state.isExpanded, collapse, expand]);

  const resize = useCallback((size: number) => {
    setState((s) => ({ ...s, size }));
  }, []);

  return { state, collapse, expand, toggle, resize };
}
```

### 6. Contribution Pattern (Extensibility)

**Implementation Priority: MEDIUM**

```typescript
// Contribution registry for extensible features
interface Contribution<T> {
  id: string;
  contribute(): T;
}

class ContributionRegistry<T> {
  private contributions: Contribution<T>[] = [];

  register(contribution: Contribution<T>): void {
    this.contributions.push(contribution);
  }

  getContributions(): T[] {
    return this.contributions.map((c) => c.contribute());
  }
}

// Example: Menu contribution
interface MenuContribution extends Contribution<MenuItem[]> {}

// Usage
const menuRegistry = new ContributionRegistry<MenuItem[]>();

menuRegistry.register({
  id: "file-menu",
  contribute: () => [
    { id: "new-file", label: "New File", command: "file.new" },
    { id: "open-file", label: "Open File", command: "file.open" },
  ],
});
```

---

## 🎨 Visual Design from Theia

### Application Shell Layout

```css
/* Theia-style shell layout */
.theia-ApplicationShell {
  display: grid;
  grid-template-areas:
    "top top top"
    "left main right"
    "bottom bottom bottom"
    "status status status";
  grid-template-rows: auto 1fr auto auto;
  grid-template-columns: auto 1fr auto;
  width: 100%;
  height: 100%;
}

.theia-top-panel {
  grid-area: top;
}

.theia-left-panel {
  grid-area: left;
  min-width: 50px;
  max-width: 50%;
}

.theia-main-panel {
  grid-area: main;
  overflow: hidden;
}

.theia-right-panel {
  grid-area: right;
  min-width: 50px;
  max-width: 50%;
}

.theia-bottom-panel {
  grid-area: bottom;
  min-height: 30px;
  max-height: 80%;
}

.theia-status-bar {
  grid-area: status;
  height: 22px;
}
```

### View Container Styling

```css
/* View container with collapsible sections */
.view-container {
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow: hidden;
}

.view-container-section {
  display: flex;
  flex-direction: column;

  &.collapsed {
    flex-shrink: 0;

    .section-body {
      display: none;
    }
  }

  &:not(.collapsed) {
    flex: 1;
    min-height: 100px;
  }
}

.section-header {
  display: flex;
  align-items: center;
  height: 22px;
  padding: 0 8px;
  background: var(--section-header-bg);
  border-bottom: 1px solid var(--border);
  cursor: pointer;
  user-select: none;

  &:hover {
    background: var(--section-header-hover-bg);
  }
}

.section-title {
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.section-collapse-icon {
  width: 16px;
  height: 16px;
  margin-right: 4px;
  transition: transform 0.1s;

  .collapsed & {
    transform: rotate(-90deg);
  }
}

.section-body {
  flex: 1;
  overflow: auto;
}
```

### Dock Panel Tab Styling

```css
/* Dock panel tabs */
.dock-panel-tab-bar {
  display: flex;
  height: 35px;
  background: var(--tab-bar-bg);
  border-bottom: 1px solid var(--border);
}

.dock-panel-tab {
  display: flex;
  align-items: center;
  padding: 0 12px;
  height: 100%;
  border-right: 1px solid var(--border);
  cursor: pointer;

  &.active {
    background: var(--tab-active-bg);
    border-bottom: 2px solid var(--accent);
  }

  &:not(.active):hover {
    background: var(--tab-hover-bg);
  }
}

.dock-panel-tab-icon {
  width: 16px;
  height: 16px;
  margin-right: 6px;
}

.dock-panel-tab-label {
  font-size: 13px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 150px;
}

.dock-panel-tab-close {
  width: 16px;
  height: 16px;
  margin-left: 6px;
  opacity: 0;

  .dock-panel-tab:hover &,
  .dock-panel-tab.active & {
    opacity: 0.7;
  }

  &:hover {
    opacity: 1 !important;
    background: var(--close-hover-bg);
    border-radius: 3px;
  }
}
```

---

## 📋 Checklist for NEURECTOMY Implementation

### Phase 1: Core Widget System

- [ ] Widget base class/interface
- [ ] Widget lifecycle management
- [ ] Widget factory registry
- [ ] Widget manager with caching

### Phase 2: Application Shell

- [ ] Shell layout (5 areas)
- [ ] Side panel management
- [ ] Bottom panel management
- [ ] Status bar integration

### Phase 3: Dock Panel

- [ ] DockPanel component
- [ ] Tab management
- [ ] Split operations
- [ ] Layout serialization

### Phase 4: View Container

- [ ] ViewContainer component
- [ ] Collapsible sections
- [ ] Section toolbar
- [ ] Section drag-reorder

### Phase 5: Extensibility

- [ ] Contribution registry
- [ ] Widget contribution points
- [ ] Menu contributions
- [ ] Keybinding contributions

---

## 🔧 Technical Notes

### Dependency Injection

Theia uses InversifyJS for DI:

```typescript
// In NEURECTOMY, use React context or Zustand
const WidgetManagerContext = createContext<WidgetManager | null>(null);

function useWidgetManager() {
  const manager = useContext(WidgetManagerContext);
  if (!manager) throw new Error('WidgetManager not provided');
  return manager;
}

// Provider at app root
<WidgetManagerContext.Provider value={widgetManager}>
  <ApplicationShell />
</WidgetManagerContext.Provider>
```

### Signal-based Events

Theia uses Lumino signals:

```typescript
// In React, use callbacks or event emitters
interface Widget {
  // Use callbacks
  onDisposed?: () => void;
  onActivated?: () => void;

  // Or EventEmitter pattern
  events: EventEmitter;
}

// Usage
widget.events.on("disposed", handleDisposed);
widget.events.on("activated", handleActivated);
```

---

_Analysis completed for NEURECTOMY IDE reference_
_Source: eclipse-theia/theia_
