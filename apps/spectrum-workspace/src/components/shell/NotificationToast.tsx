/**
 * Notification Toast System
 *
 * VS Code-style toast notifications with progress, actions, and stacking.
 * Non-blocking notifications that appear in the bottom-right corner.
 *
 * Features:
 * - Multiple notification types (info, success, warning, error, progress)
 * - Action buttons
 * - Progress bars
 * - Auto-dismiss with configurable duration
 * - Stacking with animations
 * - Accessible (ARIA live regions)
 *
 * @module @neurectomy/components/shell/NotificationToast
 * @author @APEX @CANVAS
 */

import React, {
  useState,
  useEffect,
  useCallback,
  createContext,
  useContext,
  useRef,
} from "react";
import { createPortal } from "react-dom";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";
import {
  Info,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  Loader2,
  X,
  type LucideIcon,
} from "lucide-react";

// ============================================================================
// Types
// ============================================================================

export type NotificationType =
  | "info"
  | "success"
  | "warning"
  | "error"
  | "progress";

export interface NotificationAction {
  label: string;
  onClick: () => void;
  primary?: boolean;
}

export interface Notification {
  id: string;
  type: NotificationType;
  title: string;
  message?: string;
  duration?: number; // ms, 0 = infinite
  progress?: number; // 0-100
  actions?: NotificationAction[];
  icon?: LucideIcon;
  onDismiss?: () => void;
}

interface NotificationState {
  notifications: Notification[];
  add: (notification: Omit<Notification, "id">) => string;
  remove: (id: string) => void;
  update: (id: string, updates: Partial<Notification>) => void;
  clear: () => void;
}

// ============================================================================
// Context
// ============================================================================

const NotificationContext = createContext<NotificationState | null>(null);

export function useNotifications() {
  const context = useContext(NotificationContext);
  if (!context) {
    throw new Error(
      "useNotifications must be used within NotificationProvider"
    );
  }
  return context;
}

// ============================================================================
// Provider
// ============================================================================

let notificationId = 0;

export function NotificationProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  const [notifications, setNotifications] = useState<Notification[]>([]);

  const add = useCallback((notification: Omit<Notification, "id">) => {
    const id = `notification-${++notificationId}`;
    const newNotification: Notification = {
      id,
      duration: getDurationForType(notification.type),
      ...notification,
    };

    setNotifications((prev) => [...prev, newNotification]);

    return id;
  }, []);

  const remove = useCallback((id: string) => {
    setNotifications((prev) => {
      const notification = prev.find((n) => n.id === id);
      if (notification?.onDismiss) {
        notification.onDismiss();
      }
      return prev.filter((n) => n.id !== id);
    });
  }, []);

  const update = useCallback((id: string, updates: Partial<Notification>) => {
    setNotifications((prev) =>
      prev.map((n) => (n.id === id ? { ...n, ...updates } : n))
    );
  }, []);

  const clear = useCallback(() => {
    setNotifications([]);
  }, []);

  return (
    <NotificationContext.Provider
      value={{ notifications, add, remove, update, clear }}
    >
      {children}
      <NotificationContainer />
    </NotificationContext.Provider>
  );
}

// ============================================================================
// Container
// ============================================================================

function NotificationContainer() {
  const { notifications, remove } = useNotifications();

  return createPortal(
    <div
      aria-live="polite"
      aria-label="Notifications"
      className="fixed bottom-4 right-4 z-50 flex flex-col gap-2 max-w-md"
    >
      <AnimatePresence mode="popLayout">
        {notifications.map((notification) => (
          <NotificationToast
            key={notification.id}
            notification={notification}
            onDismiss={() => remove(notification.id)}
          />
        ))}
      </AnimatePresence>
    </div>,
    document.body
  );
}

// ============================================================================
// Toast Component
// ============================================================================

interface NotificationToastProps {
  notification: Notification;
  onDismiss: () => void;
}

const TYPE_CONFIG: Record<
  NotificationType,
  {
    icon: LucideIcon;
    className: string;
    iconClassName: string;
  }
> = {
  info: {
    icon: Info,
    className: "border-blue-500/30 bg-blue-500/5",
    iconClassName: "text-blue-500",
  },
  success: {
    icon: CheckCircle2,
    className: "border-green-500/30 bg-green-500/5",
    iconClassName: "text-green-500",
  },
  warning: {
    icon: AlertTriangle,
    className: "border-yellow-500/30 bg-yellow-500/5",
    iconClassName: "text-yellow-500",
  },
  error: {
    icon: XCircle,
    className: "border-red-500/30 bg-red-500/5",
    iconClassName: "text-red-500",
  },
  progress: {
    icon: Loader2,
    className: "border-accent/30 bg-accent/5",
    iconClassName: "text-accent animate-spin",
  },
};

function getDurationForType(type: NotificationType): number {
  switch (type) {
    case "error":
      return 0; // Don't auto-dismiss errors
    case "warning":
      return 8000;
    case "success":
      return 3000;
    case "progress":
      return 0; // Progress notifications dismissed manually
    default:
      return 5000;
  }
}

function NotificationToast({
  notification,
  onDismiss,
}: NotificationToastProps) {
  const {
    type,
    title,
    message,
    duration,
    progress,
    actions,
    icon: CustomIcon,
  } = notification;
  const config = TYPE_CONFIG[type];
  const Icon = CustomIcon || config.icon;
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  // Auto-dismiss
  useEffect(() => {
    if (duration && duration > 0) {
      timerRef.current = setTimeout(onDismiss, duration);
    }
    return () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
      }
    };
  }, [duration, onDismiss]);

  // Pause timer on hover
  const handleMouseEnter = useCallback(() => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const handleMouseLeave = useCallback(() => {
    if (duration && duration > 0) {
      timerRef.current = setTimeout(onDismiss, duration / 2);
    }
  }, [duration, onDismiss]);

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 50, scale: 0.9 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, x: 100, scale: 0.9 }}
      transition={{ type: "spring", damping: 20, stiffness: 300 }}
      className={cn(
        "relative w-full min-w-80",
        "border rounded-lg shadow-lg",
        "bg-card",
        config.className
      )}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      role="alert"
    >
      <div className="flex items-start gap-3 p-4">
        {/* Icon */}
        <Icon
          size={20}
          className={cn("flex-shrink-0 mt-0.5", config.iconClassName)}
        />

        {/* Content */}
        <div className="flex-1 min-w-0">
          <div className="text-sm font-medium text-foreground">{title}</div>
          {message && (
            <div className="mt-1 text-sm text-muted-foreground">{message}</div>
          )}

          {/* Actions */}
          {actions && actions.length > 0 && (
            <div className="mt-3 flex gap-2">
              {actions.map((action, index) => (
                <button
                  key={index}
                  onClick={() => {
                    action.onClick();
                    if (!action.primary) {
                      onDismiss();
                    }
                  }}
                  className={cn(
                    "px-3 py-1 text-xs font-medium rounded",
                    "transition-colors",
                    action.primary
                      ? "bg-accent text-accent-foreground hover:bg-accent/90"
                      : "bg-muted text-muted-foreground hover:bg-muted/80"
                  )}
                >
                  {action.label}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Dismiss button */}
        <button
          onClick={onDismiss}
          className={cn(
            "flex-shrink-0 p-1 rounded",
            "text-muted-foreground hover:text-foreground",
            "hover:bg-muted/50 transition-colors"
          )}
          aria-label="Dismiss notification"
        >
          <X size={14} />
        </button>
      </div>

      {/* Progress bar */}
      {typeof progress === "number" && (
        <div className="absolute bottom-0 left-0 right-0 h-1 bg-muted/30 rounded-b-lg overflow-hidden">
          <motion.div
            className="h-full bg-accent"
            initial={{ width: 0 }}
            animate={{ width: `${progress}%` }}
            transition={{ duration: 0.3 }}
          />
        </div>
      )}
    </motion.div>
  );
}

// ============================================================================
// Helper Hooks
// ============================================================================

/**
 * Hook for common notification patterns
 */
export function useNotify() {
  const { add, remove, update } = useNotifications();

  const info = useCallback(
    (title: string, message?: string) => {
      return add({ type: "info", title, message });
    },
    [add]
  );

  const success = useCallback(
    (title: string, message?: string) => {
      return add({ type: "success", title, message });
    },
    [add]
  );

  const warning = useCallback(
    (title: string, message?: string) => {
      return add({ type: "warning", title, message });
    },
    [add]
  );

  const error = useCallback(
    (title: string, message?: string) => {
      return add({ type: "error", title, message });
    },
    [add]
  );

  const progress = useCallback(
    (title: string, message?: string) => {
      const id = add({ type: "progress", title, message, progress: 0 });

      return {
        id,
        update: (value: number, newMessage?: string) => {
          update(id, {
            progress: value,
            ...(newMessage && { message: newMessage }),
          });
        },
        complete: (successMessage?: string) => {
          update(id, {
            type: "success",
            title: successMessage || "Completed",
            progress: 100,
            duration: 3000,
          });
          setTimeout(() => remove(id), 3000);
        },
        fail: (errorMessage?: string) => {
          update(id, {
            type: "error",
            title: errorMessage || "Failed",
            progress: undefined,
            duration: 0,
          });
        },
        dismiss: () => remove(id),
      };
    },
    [add, update, remove]
  );

  const withProgress = useCallback(
    async <T,>(
      title: string,
      operation: (onProgress: (value: number) => void) => Promise<T>
    ): Promise<T> => {
      const tracker = progress(title);

      try {
        const result = await operation(tracker.update);
        tracker.complete();
        return result;
      } catch (err) {
        tracker.fail(err instanceof Error ? err.message : "Operation failed");
        throw err;
      }
    },
    [progress]
  );

  return {
    info,
    success,
    warning,
    error,
    progress,
    withProgress,
    add,
    remove,
    update,
  };
}

export default NotificationToast;
