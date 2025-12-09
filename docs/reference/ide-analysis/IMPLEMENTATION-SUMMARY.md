# NEURECTOMY IDE - Consolidated Analysis & Recommendations

## Executive Summary

This document synthesizes insights from 7 leading IDE codebases (VS Code, Continue, Zed, Lapce, IntelliJ, Theia) to provide a prioritized implementation roadmap for elevating NEURECTOMY IDE from basic to professional-grade quality.

---

## 📊 Current State Assessment

Based on the current UI state:

```
"Explorer | Search | Source Control | Agent Orchestration | Experiments | Extensions | Settings"
- Basic sidebar navigation
- Simple file explorer
- Rudimentary terminal
- Basic status bar
- No editor splitting
- No tab management
- Basic AI panel (status only)
```

### Critical Gaps Identified

| Feature            | Current State   | Target State                  | Priority    |
| ------------------ | --------------- | ----------------------------- | ----------- |
| Editor Splits      | None            | Full split support            | 🔴 CRITICAL |
| Tab Management     | Basic           | Advanced (pin, preview, drag) | 🔴 CRITICAL |
| Panel Docking      | Fixed positions | Flexible docking              | 🟡 HIGH     |
| AI Chat Interface  | Status only     | Full chat + context           | 🔴 CRITICAL |
| Tool Windows       | Hardcoded       | Configurable system           | 🟡 HIGH     |
| Command Palette    | None            | Full command access           | 🟡 HIGH     |
| Layout Persistence | None            | Full state save/restore       | 🟡 HIGH     |
| File Tree          | Basic           | Rich file operations          | 🟠 MEDIUM   |
| Terminal           | Basic           | Multi-tab + split             | 🟠 MEDIUM   |
| Status Bar         | Minimal         | Rich status segments          | 🟠 MEDIUM   |

---

## 🏗️ Recommended Architecture

### Layer 1: Core Layout System (from VS Code + Theia)

```typescript
// Application Shell - the foundation
interface ApplicationShell {
  // 5-area layout
  topPanel: TopPanel; // Menu + Toolbar
  leftPanel: SidePanel; // Activity Bar + Primary Sidebar
  mainPanel: DockPanel; // Editor Area (splittable)
  bottomPanel: BottomPanel; // Terminal + Output + Problems
  rightPanel: SidePanel; // Secondary Sidebar

  // Status
  statusBar: StatusBar;

  // Layout management
  saveLayout(): LayoutConfig;
  restoreLayout(config: LayoutConfig): void;
}
```

### Layer 2: Pane System (from Zed + Lapce)

```typescript
// Recursive pane structure for unlimited splits
type PaneGroup =
  | { type: "pane"; pane: PaneData }
  | {
      type: "split";
      direction: "horizontal" | "vertical";
      children: PaneGroup[];
      sizes: number[];
    };

interface PaneData {
  id: string;
  tabs: TabItem[];
  pinnedTabs: TabItem[];
  activeTabIndex: number;
  history: NavigationEntry[];
}
```

### Layer 3: Tool Window System (from IntelliJ)

```typescript
// Flexible tool window positioning
interface ToolWindowSystem {
  windows: Map<string, ToolWindow>;
  stripes: {
    left: ToolWindowStripe;
    right: ToolWindowStripe;
    bottom: ToolWindowStripe;
  };

  // Operations
  show(id: string): void;
  hide(id: string): void;
  float(id: string): void;
  dock(id: string, anchor: Anchor): void;
}
```

### Layer 4: AI Integration (from Continue)

```typescript
// AI chat with context providers
interface AISystem {
  chat: ChatPanel;
  contextProviders: ContextProviderRegistry;
  commands: SlashCommandRegistry;

  // Context
  addContext(item: ContextItem): void;
  removeContext(id: string): void;

  // Streaming
  sendMessage(content: string, context: ContextItem[]): AsyncGenerator<string>;
}
```

---

## 🎯 Implementation Roadmap

### Phase 1: Foundation (Week 1-2) 🔴 CRITICAL

#### 1.1 Core Layout Shell

- [ ] Create `ApplicationShell` component
- [ ] Implement 5-area grid layout
- [ ] Add basic resize handles between areas
- [ ] Wire up panel visibility toggles

```typescript
// Priority components to build
<ApplicationShell>
  <TopPanel />
  <LeftPanel>
    <ActivityBar />
    <SidebarContainer />
  </LeftPanel>
  <MainPanel>
    <EditorArea />
  </MainPanel>
  <RightPanel>
    <AIPanel />
  </RightPanel>
  <BottomPanel>
    <TerminalPanel />
  </BottomPanel>
  <StatusBar />
</ApplicationShell>
```

#### 1.2 Editor Pane System

- [ ] Implement `PaneGroup` recursive structure
- [ ] Create `Pane` component with tab bar
- [ ] Add basic split operations (split right, split down)
- [ ] Implement focus navigation between panes

#### 1.3 Tab Management

- [ ] Tab component with close button
- [ ] Tab reordering via drag-drop
- [ ] Active/inactive tab styling
- [ ] Modified indicator (dot)

---

### Phase 2: Advanced Layout (Week 3-4) 🟡 HIGH

#### 2.1 Advanced Tab Features

- [ ] Tab pinning
- [ ] Tab preview mode (single-click vs double-click)
- [ ] Tab context menu (close others, close to right, etc.)
- [ ] Cross-pane tab dragging

#### 2.2 Splitter Refinement

- [ ] One-pixel splitter style
- [ ] Minimum/maximum size constraints
- [ ] Double-click to reset
- [ ] Drag to create new splits

#### 2.3 Tool Window System

- [ ] Tool window registry
- [ ] Stripe buttons (activity bar style)
- [ ] Show/hide/toggle operations
- [ ] Auto-hide mode
- [ ] Keyboard shortcuts (Alt+1, Alt+2, etc.)

#### 2.4 Layout Persistence

- [ ] Serialize current layout to JSON
- [ ] Store in localStorage/IndexedDB
- [ ] Restore on app launch
- [ ] Named layout presets

---

### Phase 3: AI Integration (Week 5-6) 🔴 CRITICAL

#### 3.1 Chat Panel Redesign

- [ ] Message list with streaming support
- [ ] User/Assistant message components
- [ ] Code block rendering with actions
- [ ] Markdown rendering

```typescript
// Chat message structure
interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  context?: ContextItem[];
  timestamp: Date;
  isStreaming?: boolean;
}
```

#### 3.2 Context Provider System

- [ ] Context provider registry
- [ ] `@file` - Select files
- [ ] `@code` - Select code snippets
- [ ] `@terminal` - Terminal output
- [ ] `@diff` - Git diff
- [ ] `@container` - Container logs (NEURECTOMY-specific)
- [ ] `@agent` - Agent state (NEURECTOMY-specific)

#### 3.3 Input Area

- [ ] Auto-resize textarea
- [ ] Context chips display
- [ ] Model selector
- [ ] Send button with loading state
- [ ] @ mention autocomplete

#### 3.4 Slash Commands

- [ ] `/edit` - Edit selected code
- [ ] `/explain` - Explain code
- [ ] `/test` - Generate tests
- [ ] `/fix` - Fix errors
- [ ] `/deploy` - Deploy assistance (NEURECTOMY-specific)

---

### Phase 4: Editor Enhancement (Week 7-8) 🟠 MEDIUM

#### 4.1 Monaco Editor Integration

- [ ] Full Monaco setup with language support
- [ ] Custom themes matching NEURECTOMY design
- [ ] Editor actions (format, fold, etc.)
- [ ] Multi-cursor support

#### 4.2 Code Navigation

- [ ] Go to definition
- [ ] Find references
- [ ] Symbol search
- [ ] Breadcrumb navigation

#### 4.3 IntelliSense

- [ ] Autocomplete
- [ ] Parameter hints
- [ ] Quick info on hover
- [ ] Code actions (quick fixes)

---

### Phase 5: Terminal & Output (Week 9-10) 🟠 MEDIUM

#### 5.1 Terminal Enhancement

- [ ] Multiple terminal tabs
- [ ] Terminal splits within tabs
- [ ] Terminal profiles
- [ ] Copy/paste improvements

#### 5.2 Output Channels

- [ ] Output panel with multiple channels
- [ ] Problems panel with grouping
- [ ] Debug console
- [ ] Filter and search

---

### Phase 6: Command System (Week 11-12) 🟡 HIGH

#### 6.1 Command Palette

- [ ] Command registry
- [ ] Fuzzy search
- [ ] Recent commands
- [ ] Keybinding display

#### 6.2 Keyboard Shortcuts

- [ ] Keybinding registry
- [ ] Conflict detection
- [ ] User customization
- [ ] Cheat sheet

#### 6.3 Quick Open

- [ ] File picker
- [ ] Symbol picker
- [ ] Command picker
- [ ] Recent files

---

## 🎨 Design System Requirements

### Color Tokens

```css
/* Core colors from analyzed IDEs */
:root {
  /* Backgrounds */
  --background-primary: #1e1e1e;
  --background-secondary: #252526;
  --background-tertiary: #2d2d2d;

  /* Surfaces */
  --surface-editor: #1e1e1e;
  --surface-sidebar: #252526;
  --surface-panel: #1e1e1e;
  --surface-tab-bar: #252526;

  /* Tabs */
  --tab-active-bg: #1e1e1e;
  --tab-inactive-bg: transparent;
  --tab-hover-bg: #2a2d2e;

  /* Borders */
  --border-primary: #3c3c3c;
  --border-subtle: #2d2d2d;

  /* Accent */
  --accent-primary: #0078d4;
  --accent-secondary: #3794ff;

  /* Status */
  --status-error: #f14c4c;
  --status-warning: #cca700;
  --status-success: #73c991;
  --status-info: #3794ff;

  /* Text */
  --text-primary: #cccccc;
  --text-secondary: #969696;
  --text-muted: #6e6e6e;
}
```

### Typography

```css
/* IDE typography standards */
:root {
  /* UI Font */
  --font-family-ui:
    -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  --font-size-ui: 13px;
  --line-height-ui: 1.4;

  /* Editor Font */
  --font-family-mono: "JetBrains Mono", "Fira Code", Consolas, monospace;
  --font-size-mono: 14px;
  --line-height-mono: 1.5;

  /* Sizes */
  --font-size-xs: 11px;
  --font-size-sm: 12px;
  --font-size-md: 13px;
  --font-size-lg: 14px;
}
```

### Spacing

```css
/* Consistent spacing */
:root {
  --spacing-xs: 4px;
  --spacing-sm: 8px;
  --spacing-md: 12px;
  --spacing-lg: 16px;
  --spacing-xl: 24px;

  /* Component heights */
  --height-tab-bar: 35px;
  --height-status-bar: 22px;
  --height-toolbar: 38px;
  --height-activity-bar-item: 48px;

  /* Component widths */
  --width-activity-bar: 48px;
  --width-sidebar-min: 170px;
  --width-sidebar-default: 250px;
}
```

---

## 📦 Component Library Required

### Core Components (Build First)

| Component          | Source Inspiration | Priority    |
| ------------------ | ------------------ | ----------- |
| `<SplitView>`      | Zed, VS Code       | 🔴 Critical |
| `<TabBar>`         | VS Code, IntelliJ  | 🔴 Critical |
| `<Tab>`            | All                | 🔴 Critical |
| `<Pane>`           | Zed, Lapce         | 🔴 Critical |
| `<Panel>`          | Theia, VS Code     | 🔴 Critical |
| `<Sidebar>`        | VS Code            | 🔴 Critical |
| `<ActivityBar>`    | VS Code            | 🔴 Critical |
| `<StatusBar>`      | All                | 🟡 High     |
| `<CommandPalette>` | VS Code            | 🟡 High     |
| `<TreeView>`       | VS Code, Theia     | 🟡 High     |

### AI Components (From Continue)

| Component         | Purpose              | Priority    |
| ----------------- | -------------------- | ----------- |
| `<ChatPanel>`     | AI conversation      | 🔴 Critical |
| `<ChatMessage>`   | Message rendering    | 🔴 Critical |
| `<ChatInput>`     | User input area      | 🔴 Critical |
| `<ContextChip>`   | Context display      | 🔴 Critical |
| `<CodeBlock>`     | Code rendering       | 🔴 Critical |
| `<StreamingText>` | Streaming response   | 🟡 High     |
| `<ToolCallCard>`  | Agent action display | 🟡 High     |

---

## 🔧 Technical Decisions

### State Management

**Recommendation:** Zustand for global state + React Query for server state

```typescript
// Workspace store
const useWorkspaceStore = create<WorkspaceState>((set, get) => ({
  // Layout
  layout: initialLayout,
  setLayout: (layout) => set({ layout }),

  // Panes
  panes: new Map(),
  splitPane: (paneId, direction) => { ... },
  closePane: (paneId) => { ... },

  // Tool windows
  toolWindows: new Map(),
  toggleToolWindow: (id) => { ... },

  // Persistence
  saveState: () => localStorage.setItem('workspace', JSON.stringify(get())),
  loadState: () => { ... },
}));
```

### Layout Engine

**Recommendation:** Custom implementation inspired by VS Code's grid system

Key features:

- CSS Grid for main layout
- Flexbox for internal layouts
- ResizeObserver for size tracking
- Custom drag handlers for splits

### AI Integration

**Recommendation:** WebSocket for streaming + REST for state

```typescript
// AI client interface
interface AIClient {
  // Streaming chat
  chat(messages: Message[], options?: ChatOptions): AsyncGenerator<string>;

  // Cancel current stream
  cancel(): void;

  // Context operations
  getContext(provider: string, query: string): Promise<ContextItem[]>;
}
```

---

## 📚 Reference Documents Created

1. **[vscode-patterns.md](./vscode-patterns.md)** - VS Code layout, parts, editor groups
2. **[continue-patterns.md](./continue-patterns.md)** - AI chat, context providers, streaming
3. **[zed-patterns.md](./zed-patterns.md)** - Pane system, tab pinning, performance
4. **[lapce-patterns.md](./lapce-patterns.md)** - Split system, terminal tabs, panel positions
5. **[intellij-patterns.md](./intellij-patterns.md)** - Tool windows, splitters, docking
6. **[theia-patterns.md](./theia-patterns.md)** - Widget system, dock panels, extensibility

---

## 🚀 Quick Wins (Implement This Week)

### 1. Replace Basic Layout with Grid Shell

```typescript
// Before: Basic flex layout
// After: Professional grid shell
<div className="application-shell">
  <ActivityBar position="left" />
  <SidebarContainer width={250} />
  <EditorArea />
  <Panel position="bottom" />
  <StatusBar />
</div>
```

### 2. Add Editor Split Support

```typescript
// Add split buttons to tab bar
<TabBar>
  {tabs}
  <TabBarActions>
    <SplitRightButton onClick={() => splitPane('right')} />
    <SplitDownButton onClick={() => splitPane('down')} />
  </TabBarActions>
</TabBar>
```

### 3. Upgrade AI Panel to Chat Interface

```typescript
// Before: Status display
// After: Full chat interface
<AIPanel>
  <ChatHistory messages={messages} />
  <ChatInput
    onSend={sendMessage}
    onAddContext={addContext}
    contextItems={contextItems}
  />
</AIPanel>
```

### 4. Add Command Palette

```typescript
// Keyboard shortcut: Ctrl+Shift+P
<CommandPalette
  commands={allCommands}
  onSelect={executeCommand}
  isOpen={paletteOpen}
  onClose={() => setPaletteOpen(false)}
/>
```

---

## 📈 Success Metrics

| Metric                  | Current | Target           | Timeline |
| ----------------------- | ------- | ---------------- | -------- |
| Editor splits supported | 0       | Unlimited        | Week 2   |
| Tab features            | 2       | 10+              | Week 4   |
| AI context providers    | 0       | 8+               | Week 6   |
| Tool windows            | 5 fixed | 15+ configurable | Week 8   |
| Keyboard shortcuts      | ~10     | 100+             | Week 12  |
| Layout save/restore     | No      | Yes              | Week 4   |
| User satisfaction       | Basic   | Professional     | Week 12  |

---

## 🎬 Conclusion

The analyzed IDEs share common patterns that have been refined over years of development. By adopting these patterns, NEURECTOMY can achieve professional-grade quality without reinventing the wheel.

**Key Takeaways:**

1. **Layout is foundational** - Get the shell and split system right first
2. **AI is a first-class citizen** - Context and streaming are essential
3. **Everything is configurable** - Tool windows, shortcuts, layouts
4. **Performance matters** - Virtualize lists, debounce updates
5. **Persist everything** - Users expect state to survive restarts

**Next Step:** Begin with Phase 1 (Foundation) - implement the Application Shell and basic pane system.

---

_Document generated from analysis of 7 major IDE repositories_
_Version 1.0 | December 2025_
