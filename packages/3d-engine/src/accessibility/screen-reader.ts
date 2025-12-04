/**
 * Screen Reader Announcements
 *
 * Manages screen reader announcements with priority queuing,
 * debouncing, and ARIA live region support.
 *
 * @module @neurectomy/3d-engine/accessibility/screen-reader
 * @agents @CANVAS @APEX
 * @phase Phase 3 - Dimensional Forge
 */

import type {
  ScreenReaderAnnouncement,
  AnnouncementQueueConfig,
  AnnouncementPriority,
  AccessibleElementId,
} from "./types";

// ============================================================================
// Constants
// ============================================================================

const DEFAULT_CONFIG: AnnouncementQueueConfig = {
  maxQueueSize: 50,
  debounceMs: 150,
  coalesceSimilar: true,
  defaultPriority: "medium",
  defaultPoliteness: "polite",
};

const PRIORITY_ORDER: Record<AnnouncementPriority, number> = {
  critical: 4,
  high: 3,
  medium: 2,
  low: 1,
};

// ============================================================================
// Screen Reader Manager Class
// ============================================================================

/**
 * Screen Reader Announcement Manager
 *
 * Handles queuing and delivery of announcements to screen readers
 * via ARIA live regions.
 */
export class ScreenReaderManager {
  private config: AnnouncementQueueConfig;
  private queue: ScreenReaderAnnouncement[] = [];
  private liveRegion: HTMLElement | null = null;
  private assertiveRegion: HTMLElement | null = null;
  private processing: boolean = false;
  private debounceTimer: ReturnType<typeof setTimeout> | null = null;
  private listeners: Set<(announcement: ScreenReaderAnnouncement) => void> =
    new Set();
  private history: ScreenReaderAnnouncement[] = [];
  private historyMaxSize: number = 100;

  constructor(config: Partial<AnnouncementQueueConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.initializeLiveRegions();
  }

  // ==========================================================================
  // Initialization
  // ==========================================================================

  /**
   * Initialize ARIA live regions in the DOM
   */
  private initializeLiveRegions(): void {
    // Only create if we're in a browser environment
    if (typeof document === "undefined") return;

    // Create polite live region
    this.liveRegion = this.createLiveRegion("polite");

    // Create assertive live region
    this.assertiveRegion = this.createLiveRegion("assertive");
  }

  /**
   * Create a live region element
   */
  private createLiveRegion(politeness: "polite" | "assertive"): HTMLElement {
    const id = `sr-announcer-${politeness}`;

    // Check if already exists
    let element = document.getElementById(id);
    if (element) return element;

    element = document.createElement("div");
    element.id = id;
    element.setAttribute("role", "status");
    element.setAttribute("aria-live", politeness);
    element.setAttribute("aria-atomic", "true");
    element.setAttribute("aria-relevant", "additions text");

    // Hide visually but keep accessible
    Object.assign(element.style, {
      position: "absolute",
      width: "1px",
      height: "1px",
      padding: "0",
      margin: "-1px",
      overflow: "hidden",
      clip: "rect(0, 0, 0, 0)",
      whiteSpace: "nowrap",
      border: "0",
    });

    document.body.appendChild(element);
    return element;
  }

  // ==========================================================================
  // Announcement Methods
  // ==========================================================================

  /**
   * Queue an announcement
   */
  announce(
    message: string,
    options: {
      priority?: AnnouncementPriority;
      politeness?: "polite" | "assertive";
      clearPrevious?: boolean;
      delayMs?: number;
      durationMs?: number;
      sourceId?: AccessibleElementId;
      context?: Record<string, unknown>;
    } = {}
  ): string {
    const announcement: ScreenReaderAnnouncement = {
      id: `ann-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      message,
      priority: options.priority || this.config.defaultPriority,
      politeness: options.politeness || this.config.defaultPoliteness,
      clearPrevious: options.clearPrevious,
      delayMs: options.delayMs,
      durationMs: options.durationMs,
      sourceId: options.sourceId,
      context: options.context,
      timestamp: Date.now(),
    };

    // Check for similar recent announcements if coalescing is enabled
    if (this.config.coalesceSimilar) {
      const similar = this.findSimilarAnnouncement(message);
      if (similar) {
        // Update timestamp of existing announcement instead of adding new
        similar.timestamp = Date.now();
        return similar.id;
      }
    }

    // Clear queue if requested
    if (announcement.clearPrevious) {
      this.clearQueue();
    }

    // Add to queue with priority sorting
    this.addToQueue(announcement);

    // Schedule processing
    this.scheduleProcessing();

    return announcement.id;
  }

  /**
   * Announce immediately (bypass queue)
   */
  announceImmediate(
    message: string,
    politeness: "polite" | "assertive" = "assertive"
  ): void {
    const region =
      politeness === "assertive" ? this.assertiveRegion : this.liveRegion;
    if (region) {
      // Clear and set new content
      region.textContent = "";
      // Use setTimeout to ensure the change is registered
      setTimeout(() => {
        region.textContent = message;
      }, 50);
    }

    // Also add to history
    const announcement: ScreenReaderAnnouncement = {
      id: `imm-${Date.now()}`,
      message,
      priority: "critical",
      politeness,
      timestamp: Date.now(),
    };
    this.addToHistory(announcement);
    this.notifyListeners(announcement);
  }

  /**
   * Announce focus change
   */
  announceFocusChange(
    elementId: AccessibleElementId,
    label: string,
    description?: string
  ): void {
    let message = label;
    if (description) {
      message += `. ${description}`;
    }

    this.announce(message, {
      priority: "high",
      politeness: "polite",
      sourceId: elementId,
    });
  }

  /**
   * Announce selection change
   */
  announceSelectionChange(
    selectedCount: number,
    totalCount?: number,
    itemLabel?: string
  ): void {
    let message: string;

    if (selectedCount === 0) {
      message = "Selection cleared";
    } else if (selectedCount === 1) {
      message = itemLabel ? `${itemLabel} selected` : "1 item selected";
    } else {
      message = `${selectedCount} items selected`;
      if (totalCount !== undefined) {
        message += ` of ${totalCount}`;
      }
    }

    this.announce(message, {
      priority: "medium",
      politeness: "polite",
    });
  }

  /**
   * Announce navigation
   */
  announceNavigation(
    direction: "up" | "down" | "left" | "right" | "in" | "out",
    targetLabel: string,
    position?: { current: number; total: number }
  ): void {
    let message = `Moved ${direction} to ${targetLabel}`;

    if (position) {
      message += `, ${position.current} of ${position.total}`;
    }

    this.announce(message, {
      priority: "medium",
      politeness: "polite",
    });
  }

  /**
   * Announce loading state
   */
  announceLoading(isLoading: boolean, context?: string): void {
    const message = isLoading
      ? `Loading${context ? ` ${context}` : ""}...`
      : `${context || "Content"} loaded`;

    this.announce(message, {
      priority: isLoading ? "low" : "medium",
      politeness: "polite",
    });
  }

  /**
   * Announce error
   */
  announceError(error: string, isRecoverable: boolean = true): void {
    const prefix = isRecoverable ? "Error" : "Critical error";

    this.announce(`${prefix}: ${error}`, {
      priority: isRecoverable ? "high" : "critical",
      politeness: "assertive",
    });
  }

  /**
   * Announce success
   */
  announceSuccess(message: string): void {
    this.announce(message, {
      priority: "medium",
      politeness: "polite",
    });
  }

  // ==========================================================================
  // Queue Management
  // ==========================================================================

  /**
   * Add announcement to queue with priority sorting
   */
  private addToQueue(announcement: ScreenReaderAnnouncement): void {
    // Enforce max queue size
    while (this.queue.length >= this.config.maxQueueSize) {
      // Remove lowest priority item
      const lowestIndex = this.findLowestPriorityIndex();
      if (lowestIndex >= 0) {
        this.queue.splice(lowestIndex, 1);
      }
    }

    // Insert at correct position based on priority
    const insertIndex = this.findInsertIndex(announcement);
    this.queue.splice(insertIndex, 0, announcement);
  }

  /**
   * Find insert index based on priority
   */
  private findInsertIndex(announcement: ScreenReaderAnnouncement): number {
    const announcementOrder = PRIORITY_ORDER[announcement.priority];

    for (let i = 0; i < this.queue.length; i++) {
      const item = this.queue[i];
      if (item && PRIORITY_ORDER[item.priority] < announcementOrder) {
        return i;
      }
    }

    return this.queue.length;
  }

  /**
   * Find index of lowest priority announcement
   */
  private findLowestPriorityIndex(): number {
    let lowestIndex = -1;
    let lowestPriority = Infinity;

    for (let i = 0; i < this.queue.length; i++) {
      const item = this.queue[i];
      if (item) {
        const order = PRIORITY_ORDER[item.priority];
        if (order < lowestPriority) {
          lowestPriority = order;
          lowestIndex = i;
        }
      }
    }

    return lowestIndex;
  }

  /**
   * Find similar recent announcement
   */
  private findSimilarAnnouncement(
    message: string
  ): ScreenReaderAnnouncement | null {
    const recentThreshold = Date.now() - this.config.debounceMs;

    for (const item of this.queue) {
      if (item.message === message && item.timestamp > recentThreshold) {
        return item;
      }
    }

    return null;
  }

  /**
   * Clear the queue
   */
  clearQueue(): void {
    this.queue = [];
  }

  /**
   * Get queue size
   */
  getQueueSize(): number {
    return this.queue.length;
  }

  // ==========================================================================
  // Processing
  // ==========================================================================

  /**
   * Schedule queue processing with debouncing
   */
  private scheduleProcessing(): void {
    if (this.debounceTimer) {
      clearTimeout(this.debounceTimer);
    }

    this.debounceTimer = setTimeout(() => {
      this.processQueue();
    }, this.config.debounceMs);
  }

  /**
   * Process the announcement queue
   */
  private async processQueue(): Promise<void> {
    if (this.processing || this.queue.length === 0) return;

    this.processing = true;

    while (this.queue.length > 0) {
      const announcement = this.queue.shift();
      if (!announcement) continue;

      // Apply delay if specified
      if (announcement.delayMs) {
        await this.delay(announcement.delayMs);
      }

      // Deliver announcement
      this.deliverAnnouncement(announcement);

      // Add to history
      this.addToHistory(announcement);

      // Notify listeners
      this.notifyListeners(announcement);

      // Wait for duration if specified
      if (announcement.durationMs) {
        await this.delay(announcement.durationMs);
      } else {
        // Default pause between announcements
        await this.delay(100);
      }
    }

    this.processing = false;
  }

  /**
   * Deliver announcement to live region
   */
  private deliverAnnouncement(announcement: ScreenReaderAnnouncement): void {
    const region =
      announcement.politeness === "assertive"
        ? this.assertiveRegion
        : this.liveRegion;

    if (region) {
      // Clear first to ensure change is detected
      region.textContent = "";

      // Set new content after brief delay
      setTimeout(() => {
        region.textContent = announcement.message;
      }, 50);
    }
  }

  /**
   * Delay helper
   */
  private delay(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  // ==========================================================================
  // History
  // ==========================================================================

  /**
   * Add announcement to history
   */
  private addToHistory(announcement: ScreenReaderAnnouncement): void {
    this.history.push(announcement);

    // Trim history
    while (this.history.length > this.historyMaxSize) {
      this.history.shift();
    }
  }

  /**
   * Get announcement history
   */
  getHistory(): ScreenReaderAnnouncement[] {
    return [...this.history];
  }

  /**
   * Clear history
   */
  clearHistory(): void {
    this.history = [];
  }

  // ==========================================================================
  // Event Listeners
  // ==========================================================================

  /**
   * Subscribe to announcements
   */
  subscribe(
    listener: (announcement: ScreenReaderAnnouncement) => void
  ): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  /**
   * Notify all listeners
   */
  private notifyListeners(announcement: ScreenReaderAnnouncement): void {
    for (const listener of this.listeners) {
      try {
        listener(announcement);
      } catch (error) {
        console.error("Screen reader listener error:", error);
      }
    }
  }

  // ==========================================================================
  // Configuration
  // ==========================================================================

  /**
   * Update configuration
   */
  updateConfig(config: Partial<AnnouncementQueueConfig>): void {
    this.config = { ...this.config, ...config };
  }

  /**
   * Get current configuration
   */
  getConfig(): AnnouncementQueueConfig {
    return { ...this.config };
  }

  // ==========================================================================
  // Cleanup
  // ==========================================================================

  /**
   * Dispose of the manager
   */
  dispose(): void {
    // Clear timers
    if (this.debounceTimer) {
      clearTimeout(this.debounceTimer);
    }

    // Clear queue and history
    this.clearQueue();
    this.clearHistory();

    // Clear listeners
    this.listeners.clear();

    // Remove live regions from DOM
    this.liveRegion?.remove();
    this.assertiveRegion?.remove();
    this.liveRegion = null;
    this.assertiveRegion = null;
  }
}

// ============================================================================
// Singleton Instance
// ============================================================================

let managerInstance: ScreenReaderManager | null = null;

/**
 * Get the global ScreenReaderManager instance
 */
export function getScreenReaderManager(
  config?: Partial<AnnouncementQueueConfig>
): ScreenReaderManager {
  if (!managerInstance) {
    managerInstance = new ScreenReaderManager(config);
  }
  return managerInstance;
}

/**
 * Reset the global ScreenReaderManager instance
 */
export function resetScreenReaderManager(): void {
  if (managerInstance) {
    managerInstance.dispose();
    managerInstance = null;
  }
}
