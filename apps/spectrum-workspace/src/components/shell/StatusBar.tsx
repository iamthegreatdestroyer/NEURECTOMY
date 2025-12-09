/**
 * Enhanced Status Bar Component
 *
 * Professional status bar with left/center/right sections.
 * Supports dynamic items, tooltips, and click actions.
 *
 * @module @neurectomy/shell
 * @author @CANVAS
 */

import { ReactNode, forwardRef } from "react";
import { cn } from "@/lib/utils";
import { LucideIcon } from "lucide-react";

// ============================================================================
// Types
// ============================================================================

export interface StatusBarItem {
  id: string;
  content: ReactNode;
  tooltip?: string;
  onClick?: () => void;
  priority?: number;
  visible?: boolean;
  className?: string;
}

export interface StatusBarProps {
  leftItems?: StatusBarItem[];
  centerItems?: StatusBarItem[];
  rightItems?: StatusBarItem[];
  className?: string;
  backgroundColor?: string;
}

// ============================================================================
// Main Component
// ============================================================================

export function StatusBar({
  leftItems = [],
  centerItems = [],
  rightItems = [],
  className,
  backgroundColor,
}: StatusBarProps) {
  const sortByPriority = (items: StatusBarItem[]) =>
    [...items]
      .filter((i) => i.visible !== false)
      .sort((a, b) => (a.priority ?? 0) - (b.priority ?? 0));

  return (
    <div
      className={cn(
        "h-6 flex items-center justify-between px-2 text-xs",
        "border-t border-border",
        className
      )}
      style={{ backgroundColor }}
    >
      {/* Left Section */}
      <div className="flex items-center gap-3">
        {sortByPriority(leftItems).map((item) => (
          <StatusBarItemComponent key={item.id} item={item} />
        ))}
      </div>

      {/* Center Section */}
      <div className="flex items-center gap-3">
        {sortByPriority(centerItems).map((item) => (
          <StatusBarItemComponent key={item.id} item={item} />
        ))}
      </div>

      {/* Right Section */}
      <div className="flex items-center gap-3">
        {sortByPriority(rightItems).map((item) => (
          <StatusBarItemComponent key={item.id} item={item} />
        ))}
      </div>
    </div>
  );
}

// ============================================================================
// Item Component
// ============================================================================

interface StatusBarItemComponentProps {
  item: StatusBarItem;
}

function StatusBarItemComponent({ item }: StatusBarItemComponentProps) {
  const Component = item.onClick ? "button" : "div";

  return (
    <Component
      onClick={item.onClick}
      title={item.tooltip}
      className={cn(
        "flex items-center gap-1.5 text-muted-foreground",
        item.onClick &&
          "hover:text-foreground hover:bg-accent/50 px-1.5 py-0.5 rounded cursor-pointer",
        item.className
      )}
    >
      {item.content}
    </Component>
  );
}

// ============================================================================
// Preset Items (convenience components)
// ============================================================================

export interface PresetItemProps {
  icon?: LucideIcon;
  label: string;
  value?: string | number;
  tooltip?: string;
  onClick?: () => void;
  className?: string;
}

export const StatusBarBranch = forwardRef<HTMLDivElement, PresetItemProps>(
  ({ icon: Icon, label, value, tooltip, onClick, className }, ref) => (
    <div ref={ref} className={cn("flex items-center gap-1", className)}>
      {Icon && <Icon size={12} />}
      <span>{label}</span>
      {value && <span className="font-medium">{value}</span>}
    </div>
  )
);
StatusBarBranch.displayName = "StatusBarBranch";

export const StatusBarLanguage = forwardRef<HTMLDivElement, PresetItemProps>(
  ({ label, onClick, className }, ref) => (
    <div
      ref={ref}
      onClick={onClick}
      className={cn("cursor-pointer hover:text-foreground", className)}
    >
      {label}
    </div>
  )
);
StatusBarLanguage.displayName = "StatusBarLanguage";

export const StatusBarEncoding = forwardRef<HTMLDivElement, PresetItemProps>(
  ({ label, onClick, className }, ref) => (
    <div
      ref={ref}
      onClick={onClick}
      className={cn("cursor-pointer hover:text-foreground", className)}
    >
      {label}
    </div>
  )
);
StatusBarEncoding.displayName = "StatusBarEncoding";

export const StatusBarLineCol = forwardRef<
  HTMLDivElement,
  { line: number; col: number; onClick?: () => void }
>(({ line, col, onClick }, ref) => (
  <div
    ref={ref}
    onClick={onClick}
    className="cursor-pointer hover:text-foreground"
  >
    Ln {line}, Col {col}
  </div>
));
StatusBarLineCol.displayName = "StatusBarLineCol";

export const StatusBarIndent = forwardRef<
  HTMLDivElement,
  { type: "spaces" | "tabs"; size: number; onClick?: () => void }
>(({ type, size, onClick }, ref) => (
  <div
    ref={ref}
    onClick={onClick}
    className="cursor-pointer hover:text-foreground"
  >
    {type === "spaces" ? `Spaces: ${size}` : `Tab Size: ${size}`}
  </div>
));
StatusBarIndent.displayName = "StatusBarIndent";

export const StatusBarEOL = forwardRef<
  HTMLDivElement,
  { eol: "LF" | "CRLF"; onClick?: () => void }
>(({ eol, onClick }, ref) => (
  <div
    ref={ref}
    onClick={onClick}
    className="cursor-pointer hover:text-foreground"
  >
    {eol}
  </div>
));
StatusBarEOL.displayName = "StatusBarEOL";

export const StatusBarNotifications = forwardRef<
  HTMLDivElement,
  { count: number; onClick?: () => void }
>(({ count, onClick }, ref) => (
  <div
    ref={ref}
    onClick={onClick}
    className={cn(
      "flex items-center gap-1 cursor-pointer hover:text-foreground",
      count > 0 && "text-yellow-500"
    )}
  >
    <span className="relative">
      🔔
      {count > 0 && (
        <span className="absolute -top-1 -right-1 w-3 h-3 bg-yellow-500 text-[8px] text-white rounded-full flex items-center justify-center">
          {count > 9 ? "9+" : count}
        </span>
      )}
    </span>
  </div>
));
StatusBarNotifications.displayName = "StatusBarNotifications";

// ============================================================================
// Exports
// ============================================================================

export default StatusBar;
