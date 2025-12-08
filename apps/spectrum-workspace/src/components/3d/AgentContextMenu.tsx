/**
 * AgentContextMenu Component
 *
 * Context menu for agent node operations in the Dimensional Forge.
 * Provides quick access to run, pause, delete, configure, and other actions.
 */

import { useEffect, useRef } from "react";
import {
  Play,
  Pause,
  Square,
  Trash2,
  Settings,
  Copy,
  Link,
  Info,
  Edit3,
  RefreshCw,
  Eye,
  EyeOff,
} from "lucide-react";
import type { AgentNode } from "@/stores/agent-store";

export interface AgentContextMenuProps {
  node: AgentNode;
  position: { x: number; y: number };
  onClose: () => void;
  onRun: (nodeId: string) => void;
  onPause: (nodeId: string) => void;
  onStop: (nodeId: string) => void;
  onDelete: (nodeId: string) => void;
  onConfigure: (nodeId: string) => void;
  onDuplicate: (nodeId: string) => void;
  onViewConnections: (nodeId: string) => void;
  onToggleVisibility: (nodeId: string) => void;
  onReset: (nodeId: string) => void;
  onShowInfo: (nodeId: string) => void;
  onRename: (nodeId: string) => void;
}

export function AgentContextMenu({
  node,
  position,
  onClose,
  onRun,
  onPause,
  onStop,
  onDelete,
  onConfigure,
  onDuplicate,
  onViewConnections,
  onToggleVisibility,
  onReset,
  onShowInfo,
  onRename,
}: AgentContextMenuProps) {
  const menuRef = useRef<HTMLDivElement>(null);

  // Close on outside click
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        onClose();
      }
    };

    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose();
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    document.addEventListener("keydown", handleEscape);

    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
      document.removeEventListener("keydown", handleEscape);
    };
  }, [onClose]);

  const menuItems = [
    // Execution controls
    {
      icon: Play,
      label: "Run",
      onClick: () => onRun(node.id),
      disabled: node.status === "running",
      color: "text-green-400",
      show: node.status !== "running",
    },
    {
      icon: Pause,
      label: "Pause",
      onClick: () => onPause(node.id),
      disabled: node.status !== "running",
      color: "text-yellow-400",
      show: node.status === "running",
    },
    {
      icon: Square,
      label: "Stop",
      onClick: () => onStop(node.id),
      disabled: node.status === "idle",
      color: "text-red-400",
      show: node.status === "running" || node.status === "paused",
    },
    {
      icon: RefreshCw,
      label: "Reset",
      onClick: () => onReset(node.id),
      disabled: false,
      color: "text-blue-400",
      show: node.status === "error" || node.status === "completed",
    },
    { divider: true },
    // Information
    {
      icon: Info,
      label: "Show Info",
      onClick: () => onShowInfo(node.id),
      disabled: false,
      color: "text-blue-400",
      show: true,
    },
    {
      icon: Link,
      label: "View Connections",
      onClick: () => onViewConnections(node.id),
      disabled: false,
      color: "text-purple-400",
      show: true,
    },
    { divider: true },
    // Edit operations
    {
      icon: Edit3,
      label: "Rename",
      onClick: () => onRename(node.id),
      disabled: false,
      color: "text-gray-400",
      show: true,
    },
    {
      icon: Settings,
      label: "Configure",
      onClick: () => onConfigure(node.id),
      disabled: false,
      color: "text-gray-400",
      show: true,
    },
    {
      icon: Copy,
      label: "Duplicate",
      onClick: () => onDuplicate(node.id),
      disabled: false,
      color: "text-gray-400",
      show: true,
    },
    {
      icon: node.visible ? EyeOff : Eye,
      label: node.visible ? "Hide" : "Show",
      onClick: () => onToggleVisibility(node.id),
      disabled: false,
      color: "text-gray-400",
      show: true,
    },
    { divider: true },
    // Danger zone
    {
      icon: Trash2,
      label: "Delete",
      onClick: () => onDelete(node.id),
      disabled: node.status === "running",
      color: "text-red-400",
      show: true,
    },
  ];

  return (
    <div
      ref={menuRef}
      className="fixed z-50 bg-gray-900/95 backdrop-blur-md rounded-lg shadow-2xl border border-gray-700 py-2 min-w-[200px]"
      style={{
        left: `${position.x}px`,
        top: `${position.y}px`,
      }}
    >
      {/* Node header */}
      <div className="px-4 py-2 border-b border-gray-700 mb-1">
        <div className="text-sm font-semibold text-white truncate">
          {node.codename || node.name}
        </div>
        <div className="text-xs text-gray-400 capitalize">{node.type}</div>
      </div>

      {/* Menu items */}
      {menuItems.map((item, index) => {
        if (item.divider) {
          return (
            <div key={`divider-${index}`} className="h-px bg-gray-700 my-1" />
          );
        }

        if (!item.show) {
          return null;
        }

        const Icon = item.icon;

        return (
          <button
            key={index}
            onClick={(e) => {
              e.stopPropagation();
              item.onClick();
              onClose();
            }}
            disabled={item.disabled}
            className={`
              w-full flex items-center gap-3 px-4 py-2 text-sm
              transition-colors duration-150
              ${
                item.disabled
                  ? "opacity-40 cursor-not-allowed"
                  : "hover:bg-gray-800 cursor-pointer"
              }
            `}
          >
            <Icon className={`w-4 h-4 ${item.color}`} />
            <span className="text-gray-200">{item.label}</span>
          </button>
        );
      })}
    </div>
  );
}

export default AgentContextMenu;
