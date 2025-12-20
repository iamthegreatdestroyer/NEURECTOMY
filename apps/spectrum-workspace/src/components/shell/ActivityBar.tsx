/**
 * Activity Bar Component
 *
 * VS Code-style vertical icon bar for workspace navigation.
 * Supports top and bottom sections with tooltips.
 *
 * @module @neurectomy/shell
 * @author @CANVAS
 */

import { ReactNode, forwardRef } from "react";
import { cn } from "@/lib/utils";
import { LucideIcon } from "lucide-react";

export interface ActivityBarItem {
  id: string;
  icon: LucideIcon;
  label: string;
  badge?: number | string;
  position?: "top" | "bottom";
}

export interface ActivityBarProps {
  items: ActivityBarItem[];
  activeId: string | null;
  onItemClick: (id: string) => void;
  className?: string;
}

export function ActivityBar({
  items,
  activeId,
  onItemClick,
  className,
}: ActivityBarProps) {
  const topItems = items.filter((item) => item.position !== "bottom");
  const bottomItems = items.filter((item) => item.position === "bottom");

  return (
    <div
      className={cn(
        "h-full w-12 flex flex-col items-center py-2 gap-1",
        className
      )}
    >
      {/* Top Items */}
      <div className="flex flex-col items-center gap-1 flex-1">
        {topItems.map((item) => (
          <ActivityBarButton
            key={item.id}
            item={item}
            active={activeId === item.id}
            onClick={() => onItemClick(item.id)}
          />
        ))}
      </div>

      {/* Bottom Items */}
      <div className="flex flex-col items-center gap-1">
        {bottomItems.map((item) => (
          <ActivityBarButton
            key={item.id}
            item={item}
            active={activeId === item.id}
            onClick={() => onItemClick(item.id)}
          />
        ))}
      </div>
    </div>
  );
}

interface ActivityBarButtonProps {
  item: ActivityBarItem;
  active: boolean;
  onClick: () => void;
}

const ActivityBarButton = forwardRef<HTMLButtonElement, ActivityBarButtonProps>(
  ({ item, active, onClick }, ref) => {
    const Icon = item.icon;

    return (
      <button
        ref={ref}
        onClick={onClick}
        title={item.label}
        aria-label={item.label}
        className={cn(
          "relative w-12 h-12 flex items-center justify-center",
          "transition-all duration-150 group",
          active
            ? "text-foreground"
            : "text-muted-foreground hover:text-foreground"
        )}
      >
        {/* Active indicator */}
        {active && (
          <div className="absolute left-0 top-1/2 -translate-y-1/2 w-0.5 h-6 bg-primary rounded-r" />
        )}

        {/* Icon */}
        <Icon size={24} strokeWidth={active ? 2 : 1.5} />

        {/* Badge */}
        {item.badge !== undefined && (
          <span
            className={cn(
              "absolute top-1.5 right-1.5 min-w-[18px] h-[18px]",
              "flex items-center justify-center",
              "text-[10px] font-medium rounded-full",
              "bg-primary text-primary-foreground",
              "px-1"
            )}
          >
            {typeof item.badge === "number" && item.badge > 99
              ? "99+"
              : item.badge}
          </span>
        )}

        {/* Tooltip */}
        <div
          className={cn(
            "absolute left-full ml-2 px-2 py-1",
            "bg-popover text-popover-foreground",
            "text-xs font-medium rounded shadow-lg",
            "opacity-0 invisible translate-x-1",
            "group-hover:opacity-100 group-hover:visible group-hover:translate-x-0",
            "transition-all duration-150 whitespace-nowrap z-50",
            "border border-border"
          )}
        >
          {item.label}
        </div>
      </button>
    );
  }
);

ActivityBarButton.displayName = "ActivityBarButton";

export default ActivityBar;
