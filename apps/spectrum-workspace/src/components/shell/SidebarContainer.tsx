/**
 * Sidebar Container Component
 *
 * Generic container for sidebar panels with header and content areas.
 * Supports collapsible behavior and loading states.
 *
 * @module @neurectomy/shell
 * @author @CANVAS @ARCHITECT
 */

import { ReactNode, useState } from "react";
import { cn } from "@/lib/utils";
import {
  ChevronDown,
  ChevronRight,
  MoreHorizontal,
  Search,
  X,
} from "lucide-react";

export interface SidebarSection {
  id: string;
  title: string;
  content: ReactNode;
  collapsible?: boolean;
  defaultExpanded?: boolean;
  actions?: SidebarAction[];
  badge?: number | string;
}

export interface SidebarAction {
  id: string;
  icon: React.ComponentType<{ size?: number; className?: string }>;
  label: string;
  onClick: () => void;
}

export interface SidebarContainerProps {
  title: string;
  sections?: SidebarSection[];
  children?: ReactNode;
  className?: string;
  showSearch?: boolean;
  searchPlaceholder?: string;
  onSearch?: (query: string) => void;
  actions?: SidebarAction[];
  loading?: boolean;
}

export function SidebarContainer({
  title,
  sections,
  children,
  className,
  showSearch,
  searchPlaceholder = "Search...",
  onSearch,
  actions,
  loading,
}: SidebarContainerProps) {
  const [searchQuery, setSearchQuery] = useState("");

  const handleSearchChange = (value: string) => {
    setSearchQuery(value);
    onSearch?.(value);
  };

  return (
    <div className={cn("h-full flex flex-col", className)}>
      {/* Header */}
      <div className="flex items-center justify-between px-4 h-9 border-b border-border shrink-0">
        <h2 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          {title}
        </h2>
        {actions && actions.length > 0 && (
          <div className="flex items-center gap-1">
            {actions.map((action) => {
              const Icon = action.icon;
              return (
                <button
                  key={action.id}
                  onClick={action.onClick}
                  title={action.label}
                  className="p-1 rounded hover:bg-accent text-muted-foreground hover:text-foreground transition-colors"
                >
                  <Icon size={14} />
                </button>
              );
            })}
          </div>
        )}
      </div>

      {/* Search */}
      {showSearch && (
        <div className="px-2 py-2 border-b border-border shrink-0">
          <div className="relative">
            <Search
              size={14}
              className="absolute left-2 top-1/2 -translate-y-1/2 text-muted-foreground"
            />
            <input
              type="text"
              placeholder={searchPlaceholder}
              value={searchQuery}
              onChange={(e) => handleSearchChange(e.target.value)}
              className={cn(
                "w-full h-7 pl-7 pr-7 text-sm",
                "bg-background border border-border rounded",
                "focus:outline-none focus:ring-1 focus:ring-ring",
                "placeholder:text-muted-foreground"
              )}
            />
            {searchQuery && (
              <button
                onClick={() => handleSearchChange("")}
                className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
              >
                <X size={14} />
              </button>
            )}
          </div>
        </div>
      )}

      {/* Content */}
      <div className="flex-1 overflow-y-auto">
        {loading ? (
          <div className="flex items-center justify-center h-32">
            <div className="animate-spin w-5 h-5 border-2 border-primary border-t-transparent rounded-full" />
          </div>
        ) : sections ? (
          sections.map((section) => (
            <SidebarSectionComponent key={section.id} section={section} />
          ))
        ) : (
          children
        )}
      </div>
    </div>
  );
}

interface SidebarSectionComponentProps {
  section: SidebarSection;
}

function SidebarSectionComponent({ section }: SidebarSectionComponentProps) {
  const [expanded, setExpanded] = useState(section.defaultExpanded ?? true);
  const isCollapsible = section.collapsible ?? true;

  return (
    <div className="border-b border-border last:border-b-0">
      {/* Section Header */}
      <button
        onClick={() => isCollapsible && setExpanded(!expanded)}
        disabled={!isCollapsible}
        className={cn(
          "w-full flex items-center justify-between px-2 h-6",
          "text-xs font-semibold uppercase tracking-wider",
          "text-muted-foreground hover:text-foreground",
          "transition-colors",
          isCollapsible && "cursor-pointer hover:bg-accent/50"
        )}
      >
        <div className="flex items-center gap-1">
          {isCollapsible &&
            (expanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />)}
          <span>{section.title}</span>
          {section.badge !== undefined && (
            <span className="ml-1 px-1.5 py-0.5 text-[10px] bg-muted rounded">
              {section.badge}
            </span>
          )}
        </div>
        {section.actions && section.actions.length > 0 && (
          <div className="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
            {section.actions.map((action) => {
              const Icon = action.icon;
              return (
                <button
                  key={action.id}
                  onClick={(e) => {
                    e.stopPropagation();
                    action.onClick();
                  }}
                  title={action.label}
                  className="p-0.5 rounded hover:bg-accent"
                >
                  <Icon size={12} />
                </button>
              );
            })}
          </div>
        )}
      </button>

      {/* Section Content */}
      {expanded && <div className="py-1">{section.content}</div>}
    </div>
  );
}

export default SidebarContainer;
