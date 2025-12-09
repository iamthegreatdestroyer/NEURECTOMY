/**
 * Context Menu Component
 *
 * VS Code-style context menus with keyboard navigation,
 * submenus, separators, and icons.
 *
 * Features:
 * - Nested submenus
 * - Keyboard navigation
 * - Keyboard shortcuts display
 * - Icons and checkmarks
 * - Separators
 * - Disabled items
 * - Click outside to close
 *
 * @module @neurectomy/components/shell/ContextMenu
 * @author @APEX @CANVAS
 */

import React, {
  useState,
  useEffect,
  useRef,
  useCallback,
  createContext,
  useContext,
} from "react";
import { createPortal } from "react-dom";
import { cn } from "@/lib/utils";
import { Check, ChevronRight, type LucideIcon } from "lucide-react";

// ============================================================================
// Types
// ============================================================================

export interface ContextMenuItem {
  id: string;
  label: string;
  icon?: LucideIcon;
  shortcut?: string;
  disabled?: boolean;
  checked?: boolean;
  danger?: boolean;
  onClick?: () => void;
  submenu?: ContextMenuItem[];
}

export interface ContextMenuSeparator {
  type: "separator";
}

export type ContextMenuEntry = ContextMenuItem | ContextMenuSeparator;

export interface ContextMenuPosition {
  x: number;
  y: number;
}

export interface ContextMenuState {
  show: (items: ContextMenuEntry[], position: ContextMenuPosition) => void;
  hide: () => void;
}

// ============================================================================
// Context
// ============================================================================

const ContextMenuContext = createContext<ContextMenuState | null>(null);

export function useContextMenu() {
  const context = useContext(ContextMenuContext);
  if (!context) {
    throw new Error("useContextMenu must be used within ContextMenuProvider");
  }
  return context;
}

// ============================================================================
// Provider
// ============================================================================

export function ContextMenuProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  const [isOpen, setIsOpen] = useState(false);
  const [items, setItems] = useState<ContextMenuEntry[]>([]);
  const [position, setPosition] = useState<ContextMenuPosition>({ x: 0, y: 0 });

  const show = useCallback(
    (menuItems: ContextMenuEntry[], pos: ContextMenuPosition) => {
      setItems(menuItems);
      setPosition(pos);
      setIsOpen(true);
    },
    []
  );

  const hide = useCallback(() => {
    setIsOpen(false);
    setItems([]);
  }, []);

  return (
    <ContextMenuContext.Provider value={{ show, hide }}>
      {children}
      {isOpen && (
        <ContextMenuPortal items={items} position={position} onClose={hide} />
      )}
    </ContextMenuContext.Provider>
  );
}

// ============================================================================
// Portal Component
// ============================================================================

interface ContextMenuPortalProps {
  items: ContextMenuEntry[];
  position: ContextMenuPosition;
  onClose: () => void;
}

function ContextMenuPortal({
  items,
  position,
  onClose,
}: ContextMenuPortalProps) {
  const menuRef = useRef<HTMLDivElement>(null);
  const [adjustedPosition, setAdjustedPosition] = useState(position);

  // Adjust position to fit within viewport
  useEffect(() => {
    if (!menuRef.current) return;

    const menu = menuRef.current;
    const rect = menu.getBoundingClientRect();
    const viewport = {
      width: window.innerWidth,
      height: window.innerHeight,
    };

    let { x, y } = position;

    // Adjust horizontal position
    if (x + rect.width > viewport.width - 8) {
      x = viewport.width - rect.width - 8;
    }
    if (x < 8) x = 8;

    // Adjust vertical position
    if (y + rect.height > viewport.height - 8) {
      y = viewport.height - rect.height - 8;
    }
    if (y < 8) y = 8;

    setAdjustedPosition({ x, y });
  }, [position]);

  // Close on outside click
  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        onClose();
      }
    };

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose();
      }
    };

    document.addEventListener("mousedown", handleClick);
    document.addEventListener("keydown", handleKeyDown);

    return () => {
      document.removeEventListener("mousedown", handleClick);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [onClose]);

  return createPortal(
    <div
      ref={menuRef}
      className="fixed z-50"
      style={{
        left: adjustedPosition.x,
        top: adjustedPosition.y,
      }}
    >
      <ContextMenuList items={items} onClose={onClose} />
    </div>,
    document.body
  );
}

// ============================================================================
// Menu List
// ============================================================================

interface ContextMenuListProps {
  items: ContextMenuEntry[];
  onClose: () => void;
  isSubmenu?: boolean;
}

function ContextMenuList({
  items,
  onClose,
  isSubmenu = false,
}: ContextMenuListProps) {
  const [selectedIndex, setSelectedIndex] = useState(-1);
  const [openSubmenu, setOpenSubmenu] = useState<string | null>(null);
  const listRef = useRef<HTMLDivElement>(null);

  // Filter out separators for keyboard navigation
  const navigableItems = items.filter(
    (item): item is ContextMenuItem => !("type" in item)
  );

  // Keyboard navigation
  useEffect(() => {
    if (isSubmenu) return; // Only handle keyboard in top-level menu

    const handleKeyDown = (e: KeyboardEvent) => {
      switch (e.key) {
        case "ArrowDown":
          e.preventDefault();
          setSelectedIndex((prev) => {
            const next = prev + 1;
            return next >= navigableItems.length ? 0 : next;
          });
          break;
        case "ArrowUp":
          e.preventDefault();
          setSelectedIndex((prev) => {
            const next = prev - 1;
            return next < 0 ? navigableItems.length - 1 : next;
          });
          break;
        case "ArrowRight":
          e.preventDefault();
          if (selectedIndex >= 0) {
            const item = navigableItems[selectedIndex];
            if (item.submenu) {
              setOpenSubmenu(item.id);
            }
          }
          break;
        case "ArrowLeft":
          e.preventDefault();
          setOpenSubmenu(null);
          break;
        case "Enter":
          e.preventDefault();
          if (selectedIndex >= 0) {
            const item = navigableItems[selectedIndex];
            if (item.submenu) {
              setOpenSubmenu(item.id);
            } else if (!item.disabled && item.onClick) {
              item.onClick();
              onClose();
            }
          }
          break;
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [isSubmenu, selectedIndex, navigableItems, onClose]);

  // Track index of navigable items
  let navigableIndex = -1;

  return (
    <div
      ref={listRef}
      className={cn(
        "min-w-48 py-1",
        "bg-card border border-border rounded-md shadow-lg",
        "backdrop-blur-sm"
      )}
      role="menu"
    >
      {items.map((item, index) => {
        if ("type" in item && item.type === "separator") {
          return <MenuSeparator key={index} />;
        }

        navigableIndex++;
        const currentIndex = navigableIndex;

        return (
          <MenuItem
            key={item.id}
            item={item}
            selected={currentIndex === selectedIndex}
            onSelect={() => setSelectedIndex(currentIndex)}
            onClick={() => {
              if (item.submenu) {
                setOpenSubmenu(openSubmenu === item.id ? null : item.id);
              } else if (!item.disabled && item.onClick) {
                item.onClick();
                onClose();
              }
            }}
            showSubmenu={openSubmenu === item.id}
            onCloseSubmenu={() => setOpenSubmenu(null)}
            onClose={onClose}
          />
        );
      })}
    </div>
  );
}

// ============================================================================
// Menu Item
// ============================================================================

interface MenuItemProps {
  item: ContextMenuItem;
  selected: boolean;
  onSelect: () => void;
  onClick: () => void;
  showSubmenu: boolean;
  onCloseSubmenu: () => void;
  onClose: () => void;
}

function MenuItem({
  item,
  selected,
  onSelect,
  onClick,
  showSubmenu,
  onCloseSubmenu,
  onClose,
}: MenuItemProps) {
  const itemRef = useRef<HTMLButtonElement>(null);
  const submenuRef = useRef<HTMLDivElement>(null);
  const [submenuPosition, setSubmenuPosition] = useState<"right" | "left">(
    "right"
  );

  // Calculate submenu position
  useEffect(() => {
    if (!showSubmenu || !itemRef.current) return;

    const rect = itemRef.current.getBoundingClientRect();
    const viewport = window.innerWidth;

    // If not enough space on right, show on left
    if (rect.right + 200 > viewport) {
      setSubmenuPosition("left");
    } else {
      setSubmenuPosition("right");
    }
  }, [showSubmenu]);

  const Icon = item.icon;

  return (
    <div className="relative" onMouseEnter={onSelect}>
      <button
        ref={itemRef}
        className={cn(
          "w-full flex items-center gap-2 px-3 py-1.5 text-left",
          "text-sm transition-colors",
          item.disabled
            ? "text-muted-foreground cursor-not-allowed opacity-50"
            : selected || showSubmenu
              ? "bg-accent text-accent-foreground"
              : "hover:bg-muted/50",
          item.danger &&
            !item.disabled &&
            "text-destructive hover:bg-destructive/10"
        )}
        onClick={onClick}
        disabled={item.disabled}
        role="menuitem"
        aria-disabled={item.disabled}
      >
        {/* Checkbox/Icon column */}
        <span className="w-4 flex justify-center flex-shrink-0">
          {item.checked !== undefined ? (
            item.checked ? (
              <Check size={14} />
            ) : null
          ) : Icon ? (
            <Icon size={14} />
          ) : null}
        </span>

        {/* Label */}
        <span className="flex-1">{item.label}</span>

        {/* Shortcut or submenu indicator */}
        {item.submenu ? (
          <ChevronRight size={14} className="text-muted-foreground" />
        ) : item.shortcut ? (
          <span className="text-xs text-muted-foreground">{item.shortcut}</span>
        ) : null}
      </button>

      {/* Submenu */}
      {showSubmenu && item.submenu && (
        <div
          ref={submenuRef}
          className={cn(
            "absolute top-0",
            submenuPosition === "right" ? "left-full -ml-1" : "right-full -mr-1"
          )}
        >
          <ContextMenuList items={item.submenu} onClose={onClose} isSubmenu />
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Separator
// ============================================================================

function MenuSeparator() {
  return <div className="my-1 h-px bg-border" role="separator" />;
}

// ============================================================================
// Utility Functions
// ============================================================================

export function isSeparator(
  item: ContextMenuEntry
): item is ContextMenuSeparator {
  return "type" in item && item.type === "separator";
}

export function separator(): ContextMenuSeparator {
  return { type: "separator" };
}

export function menuItem(
  id: string,
  label: string,
  options?: Partial<Omit<ContextMenuItem, "id" | "label">>
): ContextMenuItem {
  return { id, label, ...options };
}

// ============================================================================
// Hook for Trigger
// ============================================================================

export function useContextMenuTrigger(items: ContextMenuEntry[]) {
  const { show } = useContextMenu();

  const handleContextMenu = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      show(items, { x: e.clientX, y: e.clientY });
    },
    [show, items]
  );

  return { onContextMenu: handleContextMenu };
}

export default ContextMenuProvider;
