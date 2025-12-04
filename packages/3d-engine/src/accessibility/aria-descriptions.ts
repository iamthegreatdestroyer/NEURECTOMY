/**
 * ARIA Description Generator
 *
 * Generates contextual ARIA descriptions for 3D scene elements
 * with support for multiple verbosity levels and spatial awareness.
 *
 * @module @neurectomy/3d-engine/accessibility/aria-descriptions
 * @agents @CANVAS @LINGUA @APEX
 * @phase Phase 3 - Dimensional Forge
 */

import type {
  AccessibleElementId,
  AccessibleElementType,
  AriaDescription,
  AriaRole,
  DescriptionLevel,
} from "./types";

// ============================================================================
// Type Mappings
// ============================================================================

/**
 * Default ARIA roles for element types
 */
const DEFAULT_ROLES: Record<AccessibleElementType, AriaRole> = {
  node: "graphics-object",
  edge: "graphics-object",
  cluster: "group",
  "timeline-point": "listitem",
  "timeline-range": "slider",
  viewport: "application",
  "camera-control": "slider",
  "selection-region": "region",
  annotation: "note" as AriaRole, // note is valid but not in our union - use 'img' as fallback
  measurement: "img",
  axis: "img",
  grid: "presentation",
  legend: "img",
  tooltip: "tooltip",
  menu: "menu",
  toolbar: "toolbar",
  panel: "region",
  dialog: "dialog",
  custom: "graphics-object",
};

// Use img as fallback for annotation since 'note' isn't in our role union
DEFAULT_ROLES["annotation"] = "img";

// ============================================================================
// Description Templates
// ============================================================================

interface DescriptionTemplate {
  minimal: (context: DescriptionContext) => string;
  standard: (context: DescriptionContext) => string;
  verbose: (context: DescriptionContext) => string;
  debug: (context: DescriptionContext) => string;
}

interface DescriptionContext {
  id: string;
  type: AccessibleElementType;
  name?: string;
  state?: string;
  value?: string | number;
  position?: { x: number; y: number; z: number };
  connections?: number;
  children?: number;
  parent?: string;
  metadata?: Record<string, unknown>;
}

/**
 * Description templates by element type
 */
const DESCRIPTION_TEMPLATES: Partial<
  Record<AccessibleElementType, DescriptionTemplate>
> = {
  node: {
    minimal: (ctx) => ctx.name || "Node",
    standard: (ctx) =>
      `${ctx.name || "Node"}${ctx.state ? ` (${ctx.state})` : ""}`,
    verbose: (ctx) =>
      `${ctx.name || "Node"} node` +
      (ctx.state ? `, status: ${ctx.state}` : "") +
      (ctx.connections !== undefined
        ? `, ${ctx.connections} connection${ctx.connections !== 1 ? "s" : ""}`
        : ""),
    debug: (ctx) =>
      `Node: ${ctx.name || ctx.id}` +
      `, Type: ${ctx.type}` +
      (ctx.state ? `, State: ${ctx.state}` : "") +
      (ctx.position
        ? `, Position: (${ctx.position.x.toFixed(2)}, ${ctx.position.y.toFixed(2)}, ${ctx.position.z.toFixed(2)})`
        : "") +
      (ctx.connections !== undefined
        ? `, Connections: ${ctx.connections}`
        : ""),
  },

  edge: {
    minimal: (ctx) => "Connection",
    standard: (ctx) => ctx.name || "Connection between nodes",
    verbose: (ctx) =>
      `Connection: ${ctx.name || "edge"}` + (ctx.state ? `, ${ctx.state}` : ""),
    debug: (ctx) =>
      `Edge: ${ctx.id}` +
      (ctx.name ? `, Label: ${ctx.name}` : "") +
      (ctx.state ? `, State: ${ctx.state}` : "") +
      (ctx.metadata ? `, Metadata: ${JSON.stringify(ctx.metadata)}` : ""),
  },

  cluster: {
    minimal: (ctx) => ctx.name || "Group",
    standard: (ctx) =>
      `${ctx.name || "Group"}${ctx.children !== undefined ? ` containing ${ctx.children} items` : ""}`,
    verbose: (ctx) =>
      `Group: ${ctx.name || "cluster"}` +
      (ctx.children !== undefined
        ? `, contains ${ctx.children} item${ctx.children !== 1 ? "s" : ""}`
        : "") +
      (ctx.state ? `, ${ctx.state}` : ""),
    debug: (ctx) =>
      `Cluster: ${ctx.id}` +
      (ctx.name ? `, Name: ${ctx.name}` : "") +
      (ctx.children !== undefined ? `, Children: ${ctx.children}` : "") +
      (ctx.state ? `, State: ${ctx.state}` : ""),
  },

  "timeline-point": {
    minimal: (ctx) => (ctx.value !== undefined ? String(ctx.value) : "Point"),
    standard: (ctx) =>
      `Timeline point${ctx.value !== undefined ? ` at ${ctx.value}` : ""}${ctx.name ? `: ${ctx.name}` : ""}`,
    verbose: (ctx) =>
      `Timeline marker` +
      (ctx.value !== undefined ? ` at position ${ctx.value}` : "") +
      (ctx.name ? `, labeled "${ctx.name}"` : "") +
      (ctx.state ? `, ${ctx.state}` : ""),
    debug: (ctx) =>
      `TimelinePoint: ${ctx.id}` +
      (ctx.value !== undefined ? `, Value: ${ctx.value}` : "") +
      (ctx.name ? `, Label: ${ctx.name}` : "") +
      (ctx.state ? `, State: ${ctx.state}` : ""),
  },

  viewport: {
    minimal: () => "3D Viewport",
    standard: () => "3D visualization viewport. Use arrow keys to navigate.",
    verbose: () =>
      "3D visualization viewport. " +
      "Use arrow keys to rotate view, " +
      "WASD to pan, " +
      "mouse wheel or +/- to zoom.",
    debug: (ctx) =>
      `Viewport: ${ctx.id}` +
      (ctx.position
        ? `, Camera: (${ctx.position.x.toFixed(2)}, ${ctx.position.y.toFixed(2)}, ${ctx.position.z.toFixed(2)})`
        : ""),
  },
};

// ============================================================================
// Spatial Description Generator
// ============================================================================

/**
 * Generate spatial relationship descriptions
 */
function generateSpatialDescription(
  position: { x: number; y: number; z: number },
  referencePoint?: { x: number; y: number; z: number }
): string {
  if (!referencePoint) {
    // Absolute position description
    const xDesc = position.x > 0 ? "right" : position.x < 0 ? "left" : "center";
    const yDesc = position.y > 0 ? "top" : position.y < 0 ? "bottom" : "middle";
    const zDesc = position.z > 0 ? "front" : position.z < 0 ? "back" : "center";

    return `Located at ${xDesc} ${yDesc} ${zDesc} of the scene`;
  }

  // Relative position description
  const dx = position.x - referencePoint.x;
  const dy = position.y - referencePoint.y;
  const dz = position.z - referencePoint.z;

  const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
  const distanceDesc =
    distance < 5 ? "nearby" : distance < 20 ? "at medium distance" : "far away";

  const directions: string[] = [];
  if (Math.abs(dx) > 1)
    directions.push(dx > 0 ? "to the right" : "to the left");
  if (Math.abs(dy) > 1) directions.push(dy > 0 ? "above" : "below");
  if (Math.abs(dz) > 1) directions.push(dz > 0 ? "in front" : "behind");

  if (directions.length === 0) {
    return `At the same position, ${distanceDesc}`;
  }

  return `${directions.join(", ")}, ${distanceDesc}`;
}

// ============================================================================
// ARIA Description Generator Class
// ============================================================================

/**
 * ARIA Description Generator
 *
 * Generates accessible descriptions for 3D scene elements
 * with configurable verbosity and spatial awareness.
 */
export class AriaDescriptionGenerator {
  private defaultLevel: DescriptionLevel;
  private customTemplates: Map<AccessibleElementType, DescriptionTemplate>;
  private descriptionCache: Map<string, AriaDescription>;
  private cacheEnabled: boolean;

  constructor(
    options: {
      defaultLevel?: DescriptionLevel;
      enableCache?: boolean;
    } = {}
  ) {
    this.defaultLevel = options.defaultLevel || "standard";
    this.customTemplates = new Map();
    this.descriptionCache = new Map();
    this.cacheEnabled = options.enableCache ?? true;
  }

  /**
   * Generate ARIA description for an element
   */
  generate(
    id: AccessibleElementId,
    type: AccessibleElementType,
    context: Omit<DescriptionContext, "id" | "type">,
    options: {
      level?: DescriptionLevel;
      role?: AriaRole;
      includePosition?: boolean;
      referencePoint?: { x: number; y: number; z: number };
    } = {}
  ): AriaDescription {
    // Check cache
    const cacheKey = this.getCacheKey(id, type, context, options);
    if (this.cacheEnabled && this.descriptionCache.has(cacheKey)) {
      return this.descriptionCache.get(cacheKey)!;
    }

    const level = options.level || this.defaultLevel;
    const role = options.role || DEFAULT_ROLES[type] || "graphics-object";

    // Get template
    const template =
      this.customTemplates.get(type) || DESCRIPTION_TEMPLATES[type];

    // Build full context
    const fullContext: DescriptionContext = {
      ...context,
      id,
      type,
    };

    // Generate level-specific descriptions
    const levelDescriptions: Record<DescriptionLevel, string> = {
      minimal: template?.minimal(fullContext) || context.name || type,
      standard: template?.standard(fullContext) || context.name || type,
      verbose: template?.verbose(fullContext) || context.name || type,
      debug: template?.debug(fullContext) || `${type}: ${id}`,
    };

    // Generate spatial description if position provided
    let spatialDescription: string | undefined;
    if (options.includePosition && context.position) {
      spatialDescription = generateSpatialDescription(
        context.position,
        options.referencePoint
      );
    }

    const description: AriaDescription = {
      id,
      type,
      role,
      label: levelDescriptions[level],
      description:
        level === "verbose" || level === "debug"
          ? levelDescriptions.verbose
          : undefined,
      levelDescriptions,
      spatialDescription,
    };

    // Add state-based attributes
    if (context.state === "selected") {
      description.selected = true;
    }
    if (context.state === "disabled") {
      description.disabled = true;
    }
    if (context.state === "expanded") {
      description.expanded = true;
    }
    if (context.state === "collapsed") {
      description.expanded = false;
    }

    // Add value for numeric elements
    if (context.value !== undefined) {
      description.value = context.value;
      description.valueText = String(context.value);
    }

    // Cache result
    if (this.cacheEnabled) {
      this.descriptionCache.set(cacheKey, description);
    }

    return description;
  }

  /**
   * Generate description for a node element
   */
  generateNodeDescription(
    id: AccessibleElementId,
    node: {
      name?: string;
      type?: string;
      state?: "active" | "inactive" | "selected" | "focused" | "error";
      position?: { x: number; y: number; z: number };
      connections?: number;
      metrics?: Record<string, number>;
    },
    level?: DescriptionLevel
  ): AriaDescription {
    return this.generate(
      id,
      "node",
      {
        name: node.name,
        state: node.state,
        position: node.position,
        connections: node.connections,
        metadata: node.metrics,
      },
      {
        level,
        includePosition: true,
      }
    );
  }

  /**
   * Generate description for an edge element
   */
  generateEdgeDescription(
    id: AccessibleElementId,
    edge: {
      sourceNode: string;
      targetNode: string;
      type?: string;
      weight?: number;
      state?: "active" | "inactive" | "highlighted";
    },
    level?: DescriptionLevel
  ): AriaDescription {
    const name = `Connection from ${edge.sourceNode} to ${edge.targetNode}`;

    return this.generate(
      id,
      "edge",
      {
        name,
        state: edge.state,
        metadata: {
          type: edge.type,
          weight: edge.weight,
        },
      },
      { level }
    );
  }

  /**
   * Generate description for a timeline point
   */
  generateTimelineDescription(
    id: AccessibleElementId,
    point: {
      timestamp: number | Date;
      label?: string;
      events?: string[];
    },
    level?: DescriptionLevel
  ): AriaDescription {
    const timeStr =
      point.timestamp instanceof Date
        ? point.timestamp.toLocaleString()
        : new Date(point.timestamp).toLocaleString();

    return this.generate(
      id,
      "timeline-point",
      {
        name: point.label,
        value: timeStr,
        children: point.events?.length,
      },
      { level }
    );
  }

  /**
   * Generate description for a cluster/group
   */
  generateClusterDescription(
    id: AccessibleElementId,
    cluster: {
      name?: string;
      nodeCount: number;
      type?: string;
      state?: "expanded" | "collapsed";
    },
    level?: DescriptionLevel
  ): AriaDescription {
    const description = this.generate(
      id,
      "cluster",
      {
        name: cluster.name,
        state: cluster.state,
        children: cluster.nodeCount,
        metadata: { type: cluster.type },
      },
      { level }
    );

    description.expanded = cluster.state === "expanded";

    return description;
  }

  /**
   * Generate description for the viewport
   */
  generateViewportDescription(
    id: AccessibleElementId,
    viewport: {
      elementCount?: number;
      zoomLevel?: number;
      cameraPosition?: { x: number; y: number; z: number };
    },
    level?: DescriptionLevel
  ): AriaDescription {
    const description = this.generate(
      id,
      "viewport",
      {
        position: viewport.cameraPosition,
        metadata: {
          elementCount: viewport.elementCount,
          zoomLevel: viewport.zoomLevel,
        },
      },
      { level }
    );

    // Viewport is always an application
    description.role = "application";

    return description;
  }

  /**
   * Register a custom description template
   */
  registerTemplate(
    type: AccessibleElementType,
    template: DescriptionTemplate
  ): void {
    this.customTemplates.set(type, template);
  }

  /**
   * Set default verbosity level
   */
  setDefaultLevel(level: DescriptionLevel): void {
    this.defaultLevel = level;
    // Clear cache when level changes
    this.clearCache();
  }

  /**
   * Get current default level
   */
  getDefaultLevel(): DescriptionLevel {
    return this.defaultLevel;
  }

  /**
   * Clear the description cache
   */
  clearCache(): void {
    this.descriptionCache.clear();
  }

  /**
   * Generate cache key
   */
  private getCacheKey(
    id: string,
    type: AccessibleElementType,
    context: Omit<DescriptionContext, "id" | "type">,
    options: Record<string, unknown>
  ): string {
    return JSON.stringify({ id, type, context, options });
  }
}

// ============================================================================
// Singleton Instance
// ============================================================================

let generatorInstance: AriaDescriptionGenerator | null = null;

/**
 * Get the global AriaDescriptionGenerator instance
 */
export function getAriaDescriptionGenerator(
  options?: ConstructorParameters<typeof AriaDescriptionGenerator>[0]
): AriaDescriptionGenerator {
  if (!generatorInstance) {
    generatorInstance = new AriaDescriptionGenerator(options);
  }
  return generatorInstance;
}

/**
 * Reset the global AriaDescriptionGenerator instance
 */
export function resetAriaDescriptionGenerator(): void {
  generatorInstance = null;
}
