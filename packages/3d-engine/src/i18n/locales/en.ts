/**
 * English (en) Translations - Base Language
 *
 * This is the primary translation file that all others are derived from.
 * All keys should be present in this file.
 *
 * @module @neurectomy/3d-engine/i18n/locales/en
 * @agents @LINGUA @SCRIBE
 * @phase Phase 3 - Dimensional Forge
 */

import type { NamespaceTranslations } from "../types";

export const en: NamespaceTranslations = {
  common: {
    // Application
    appName: "Neurectomy",
    loading: "Loading...",
    error: "Error",
    success: "Success",
    warning: "Warning",
    info: "Information",

    // Actions
    actions: {
      save: "Save",
      cancel: "Cancel",
      delete: "Delete",
      edit: "Edit",
      create: "Create",
      update: "Update",
      close: "Close",
      confirm: "Confirm",
      reset: "Reset",
      refresh: "Refresh",
      export: "Export",
      import: "Import",
      copy: "Copy",
      paste: "Paste",
      undo: "Undo",
      redo: "Redo",
      search: "Search",
      filter: "Filter",
      sort: "Sort",
      expand: "Expand",
      collapse: "Collapse",
      selectAll: "Select All",
      deselectAll: "Deselect All",
    },

    // Status
    status: {
      active: "Active",
      inactive: "Inactive",
      pending: "Pending",
      completed: "Completed",
      failed: "Failed",
      processing: "Processing",
      ready: "Ready",
      idle: "Idle",
    },

    // Time
    time: {
      now: "Now",
      today: "Today",
      yesterday: "Yesterday",
      tomorrow: "Tomorrow",
      thisWeek: "This Week",
      lastWeek: "Last Week",
      thisMonth: "This Month",
      lastMonth: "Last Month",
    },

    // Units
    units: {
      seconds: {
        one: "{{count}} second",
        other: "{{count}} seconds",
      },
      minutes: {
        one: "{{count}} minute",
        other: "{{count}} minutes",
      },
      hours: {
        one: "{{count}} hour",
        other: "{{count}} hours",
      },
      days: {
        one: "{{count}} day",
        other: "{{count}} days",
      },
      items: {
        one: "{{count}} item",
        other: "{{count}} items",
      },
    },
  },

  graph: {
    // Node types
    nodes: {
      title: "Nodes",
      node: "Node",
      create: "Create Node",
      delete: "Delete Node",
      edit: "Edit Node",
      count: {
        one: "{{count}} node",
        other: "{{count}} nodes",
      },
      types: {
        standard: "Standard Node",
        agent: "Agent Node",
        data: "Data Node",
        process: "Process Node",
        decision: "Decision Node",
        input: "Input Node",
        output: "Output Node",
        cluster: "Cluster",
      },
      properties: {
        id: "Node ID",
        label: "Label",
        type: "Type",
        position: "Position",
        size: "Size",
        color: "Color",
        metadata: "Metadata",
      },
    },

    // Edge types
    edges: {
      title: "Edges",
      edge: "Edge",
      create: "Create Edge",
      delete: "Delete Edge",
      edit: "Edit Edge",
      count: {
        one: "{{count}} edge",
        other: "{{count}} edges",
      },
      types: {
        default: "Default Connection",
        dataFlow: "Data Flow",
        controlFlow: "Control Flow",
        dependency: "Dependency",
        association: "Association",
        inheritance: "Inheritance",
        composition: "Composition",
        bidirectional: "Bidirectional",
      },
      properties: {
        id: "Edge ID",
        source: "Source Node",
        target: "Target Node",
        weight: "Weight",
        label: "Label",
        directed: "Directed",
      },
    },

    // Layout
    layout: {
      title: "Layout",
      forceDirect: "Force-Directed",
      hierarchical: "Hierarchical",
      circular: "Circular",
      grid: "Grid",
      radial: "Radial",
      tree: "Tree",
      dagre: "Dagre",
      apply: "Apply Layout",
      autoLayout: "Auto Layout",
    },

    // Operations
    operations: {
      addNode: "Add node",
      removeNode: "Remove node",
      connectNodes: "Connect nodes",
      disconnectNodes: "Disconnect nodes",
      groupNodes: "Group nodes",
      ungroupNodes: "Ungroup nodes",
      mergeNodes: "Merge nodes",
      splitNode: "Split node",
    },
  },

  agent: {
    // Agent general
    title: "Agent",
    name: "Agent Name",
    type: "Agent Type",
    status: "Status",
    create: "Create Agent",
    delete: "Delete Agent",
    configure: "Configure Agent",

    // Agent types
    types: {
      llm: "LLM Agent",
      tool: "Tool Agent",
      router: "Router Agent",
      orchestrator: "Orchestrator",
      worker: "Worker Agent",
      supervisor: "Supervisor",
      retriever: "Retriever Agent",
      custom: "Custom Agent",
    },

    // Agent status
    statuses: {
      idle: "Idle",
      running: "Running",
      waiting: "Waiting",
      error: "Error",
      completed: "Completed",
      terminated: "Terminated",
    },

    // Agent actions
    actions: {
      start: "Start Agent",
      stop: "Stop Agent",
      pause: "Pause Agent",
      resume: "Resume Agent",
      restart: "Restart Agent",
      clone: "Clone Agent",
    },

    // Agent metrics
    metrics: {
      title: "Metrics",
      executionTime: "Execution Time",
      tokensUsed: "Tokens Used",
      requestCount: "Request Count",
      successRate: "Success Rate",
      errorRate: "Error Rate",
      avgResponseTime: "Avg Response Time",
    },

    // Workflow
    workflow: {
      title: "Workflow",
      step: "Step",
      steps: {
        one: "{{count}} step",
        other: "{{count}} steps",
      },
      input: "Input",
      output: "Output",
      condition: "Condition",
      loop: "Loop",
      parallel: "Parallel",
      sequence: "Sequence",
    },
  },

  visualization: {
    // 3D Scene
    scene: {
      title: "Scene",
      camera: "Camera",
      lighting: "Lighting",
      background: "Background",
      grid: "Grid",
      axes: "Axes",
    },

    // Camera
    camera: {
      perspective: "Perspective",
      orthographic: "Orthographic",
      reset: "Reset Camera",
      zoomIn: "Zoom In",
      zoomOut: "Zoom Out",
      pan: "Pan",
      rotate: "Rotate",
      focus: "Focus on Selection",
      fitToView: "Fit to View",
    },

    // View modes
    views: {
      title: "View",
      top: "Top",
      bottom: "Bottom",
      front: "Front",
      back: "Back",
      left: "Left",
      right: "Right",
      isometric: "Isometric",
      custom: "Custom",
    },

    // Rendering
    rendering: {
      title: "Rendering",
      quality: "Quality",
      low: "Low",
      medium: "Medium",
      high: "High",
      ultra: "Ultra",
      wireframe: "Wireframe",
      solid: "Solid",
      textured: "Textured",
      shadows: "Shadows",
      antialiasing: "Antialiasing",
      bloom: "Bloom",
      ambientOcclusion: "Ambient Occlusion",
    },

    // Animation
    animation: {
      title: "Animation",
      play: "Play",
      pause: "Pause",
      stop: "Stop",
      speed: "Speed",
      loop: "Loop",
      reverse: "Reverse",
      frame: "Frame",
      timeline: "Timeline",
    },

    // Selection
    selection: {
      title: "Selection",
      none: "Nothing selected",
      single: "{{name}} selected",
      multiple: {
        one: "{{count}} item selected",
        other: "{{count}} items selected",
      },
      selectAll: "Select All",
      deselectAll: "Deselect All",
      invertSelection: "Invert Selection",
    },
  },

  accessibility: {
    // General
    title: "Accessibility",
    enabled: "Accessibility Enabled",
    disabled: "Accessibility Disabled",

    // Screen reader
    screenReader: {
      title: "Screen Reader",
      enabled: "Screen Reader Mode",
      announce: "Announce",
      description: "Description",
    },

    // Keyboard
    keyboard: {
      title: "Keyboard Navigation",
      shortcuts: "Keyboard Shortcuts",
      navigation: "Navigation",
      focusMode: "Focus Mode",
    },

    // Visual
    visual: {
      title: "Visual Settings",
      highContrast: "High Contrast",
      colorBlindMode: "Color Blind Mode",
      normal: "Normal Vision",
      protanopia: "Protanopia (Red-Blind)",
      deuteranopia: "Deuteranopia (Green-Blind)",
      tritanopia: "Tritanopia (Blue-Blind)",
      monochromacy: "Monochromacy (Grayscale)",
      reducedMotion: "Reduced Motion",
    },

    // Descriptions
    descriptions: {
      node: "{{type}} node labeled {{label}} at position {{x}}, {{y}}, {{z}}",
      edge: "Connection from {{source}} to {{target}}",
      cluster: "Cluster containing {{count}} nodes",
      scene: "Scene with {{nodeCount}} nodes and {{edgeCount}} edges",
    },
  },

  errors: {
    // General errors
    general: {
      unknown: "An unknown error occurred",
      network: "Network error. Please check your connection.",
      timeout: "Request timed out",
      notFound: "Resource not found",
      unauthorized: "You are not authorized to perform this action",
      forbidden: "Access forbidden",
      serverError: "Server error. Please try again later.",
    },

    // Validation errors
    validation: {
      required: "{{field}} is required",
      invalid: "{{field}} is invalid",
      tooShort: "{{field}} must be at least {{min}} characters",
      tooLong: "{{field}} must be at most {{max}} characters",
      outOfRange: "{{field}} must be between {{min}} and {{max}}",
      invalidFormat: "{{field}} format is invalid",
    },

    // Graph errors
    graph: {
      nodeNotFound: "Node not found: {{id}}",
      edgeNotFound: "Edge not found: {{id}}",
      duplicateNode: "A node with this ID already exists",
      duplicateEdge: "This connection already exists",
      selfLoop: "Self-loops are not allowed",
      cycleDetected: "This operation would create a cycle",
    },

    // Agent errors
    agent: {
      notFound: "Agent not found: {{id}}",
      alreadyRunning: "Agent is already running",
      notRunning: "Agent is not running",
      configInvalid: "Agent configuration is invalid",
      executionFailed: "Agent execution failed: {{reason}}",
    },
  },

  tooltips: {
    // Graph tooltips
    graph: {
      addNode: "Click to add a new node",
      deleteNode: "Delete this node and its connections",
      connect: "Drag to connect to another node",
      zoom: "Scroll to zoom in/out",
      pan: "Click and drag to pan the view",
      select: "Click to select, Ctrl+Click for multiple",
    },

    // Controls
    controls: {
      undo: "Undo last action (Ctrl+Z)",
      redo: "Redo last action (Ctrl+Y)",
      save: "Save changes (Ctrl+S)",
      export: "Export graph data",
      import: "Import graph data",
      settings: "Open settings",
      help: "Open help",
    },
  },

  notifications: {
    // Success messages
    success: {
      saved: "Changes saved successfully",
      created: "{{item}} created successfully",
      updated: "{{item}} updated successfully",
      deleted: "{{item}} deleted successfully",
      exported: "Export completed successfully",
      imported: "Import completed successfully",
    },

    // Info messages
    info: {
      loading: "Loading {{item}}...",
      processing: "Processing {{item}}...",
      autoSave: "Auto-saving changes...",
    },

    // Warning messages
    warning: {
      unsavedChanges: "You have unsaved changes",
      confirmDelete: "Are you sure you want to delete {{item}}?",
      irreversible: "This action cannot be undone",
    },
  },

  settings: {
    title: "Settings",

    // Categories
    categories: {
      general: "General",
      appearance: "Appearance",
      accessibility: "Accessibility",
      performance: "Performance",
      advanced: "Advanced",
    },

    // General settings
    general: {
      language: "Language",
      autoSave: "Auto-save",
      autoSaveInterval: "Auto-save interval",
      notifications: "Notifications",
    },

    // Appearance settings
    appearance: {
      theme: "Theme",
      darkMode: "Dark Mode",
      lightMode: "Light Mode",
      systemDefault: "System Default",
      fontSize: "Font Size",
      uiScale: "UI Scale",
    },

    // Performance settings
    performance: {
      hardwareAcceleration: "Hardware Acceleration",
      maxNodes: "Maximum Nodes",
      renderQuality: "Render Quality",
      animationSpeed: "Animation Speed",
    },
  },

  help: {
    title: "Help",

    // Sections
    sections: {
      gettingStarted: "Getting Started",
      tutorials: "Tutorials",
      shortcuts: "Keyboard Shortcuts",
      faq: "FAQ",
      support: "Support",
    },

    // Keyboard shortcuts
    shortcuts: {
      title: "Keyboard Shortcuts",
      navigation: {
        title: "Navigation",
        panLeft: "Pan Left",
        panRight: "Pan Right",
        panUp: "Pan Up",
        panDown: "Pan Down",
        zoomIn: "Zoom In",
        zoomOut: "Zoom Out",
        resetView: "Reset View",
      },
      editing: {
        title: "Editing",
        undo: "Undo",
        redo: "Redo",
        copy: "Copy",
        paste: "Paste",
        delete: "Delete",
        selectAll: "Select All",
      },
    },
  },
};
