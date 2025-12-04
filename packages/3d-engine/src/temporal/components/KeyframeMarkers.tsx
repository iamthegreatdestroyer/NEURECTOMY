/**
 * Keyframe Markers Component
 *
 * Visual markers on the timeline indicating keyframes and events.
 * Supports hover states, selection, and contextual actions.
 *
 * @module @neurectomy/3d-engine/temporal/components/KeyframeMarkers
 */

import React, { useCallback, useMemo, useState } from "react";
import type { TimelineMarker } from "../types";
import type { KeyframeMarkersProps, TimelineTheme } from "./types";
import { DEFAULT_TIMELINE_THEME } from "./types";

// ============================================================================
// Marker Icon
// ============================================================================

interface MarkerIconProps {
  type: "keyframe" | "event" | "note" | "error";
  size?: number;
  color?: string;
}

const MarkerIcon: React.FC<MarkerIconProps> = ({ type, size = 12, color }) => {
  switch (type) {
    case "keyframe":
      return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill={color}>
          <path d="M12 2L4.5 20.29l.71.71L12 18l6.79 3 .71-.71z" />
        </svg>
      );
    case "event":
      return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill={color}>
          <circle cx="12" cy="12" r="8" />
        </svg>
      );
    case "note":
      return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill={color}>
          <path d="M3 17.25V21h3.75L17.81 9.94l-3.75-3.75L3 17.25zM20.71 7.04c.39-.39.39-1.02 0-1.41l-2.34-2.34c-.39-.39-1.02-.39-1.41 0l-1.83 1.83 3.75 3.75 1.83-1.83z" />
        </svg>
      );
    case "error":
      return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill={color}>
          <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z" />
        </svg>
      );
  }
};

// ============================================================================
// Single Marker Component
// ============================================================================

interface MarkerProps {
  marker: TimelineMarker;
  position: number;
  isSelected: boolean;
  isHovered: boolean;
  isCurrent: boolean;
  interactive: boolean;
  theme: TimelineTheme;
  onClick?: () => void;
  onDoubleClick?: () => void;
  onMouseEnter?: () => void;
  onMouseLeave?: () => void;
  onContextMenu?: (event: React.MouseEvent) => void;
}

const Marker: React.FC<MarkerProps> = ({
  marker,
  position,
  isSelected,
  isHovered,
  isCurrent,
  interactive,
  theme,
  onClick,
  onDoubleClick,
  onMouseEnter,
  onMouseLeave,
  onContextMenu,
}) => {
  const markerColor = useMemo(() => {
    if (marker.type === "keyframe") return theme.keyframe;
    if (marker.type === "error") return "#ff4444";
    return marker.color || theme.event;
  }, [marker.type, marker.color, theme]);

  const scale = isHovered || isSelected || isCurrent ? 1.3 : 1;

  return (
    <div
      className="timeline-marker"
      onClick={interactive ? onClick : undefined}
      onDoubleClick={interactive ? onDoubleClick : undefined}
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
      onContextMenu={onContextMenu}
      style={{
        position: "absolute",
        left: `${position * 100}%`,
        top: "50%",
        transform: `translate(-50%, -50%) scale(${scale})`,
        zIndex: isHovered || isSelected ? 10 : 1,
        cursor: interactive ? "pointer" : "default",
        transition: "transform 100ms ease-out",
      }}
      title={marker.label || `${marker.type} at ${marker.time}ms`}
      role={interactive ? "button" : undefined}
      tabIndex={interactive ? 0 : undefined}
    >
      {/* Marker stem */}
      <div
        style={{
          position: "absolute",
          left: "50%",
          top: "100%",
          transform: "translateX(-50%)",
          width: "2px",
          height: "8px",
          backgroundColor: markerColor,
        }}
      />

      {/* Marker icon/shape */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          width: "16px",
          height: "16px",
          backgroundColor: theme.background,
          border: `2px solid ${markerColor}`,
          borderRadius: marker.type === "keyframe" ? "0" : "50%",
          transform: marker.type === "keyframe" ? "rotate(45deg)" : "none",
          boxShadow:
            isSelected || isCurrent ? `0 0 6px ${markerColor}` : "none",
        }}
      >
        {marker.icon && (
          <span
            style={{
              transform: marker.type === "keyframe" ? "rotate(-45deg)" : "none",
              fontSize: "10px",
            }}
          >
            {marker.icon}
          </span>
        )}
      </div>

      {/* Tooltip on hover */}
      {isHovered && marker.label && (
        <div
          style={{
            position: "absolute",
            bottom: "100%",
            left: "50%",
            transform: "translateX(-50%)",
            marginBottom: "16px",
            padding: "4px 8px",
            backgroundColor: theme.background,
            color: theme.text,
            fontSize: "11px",
            borderRadius: "4px",
            border: `1px solid ${theme.border}`,
            whiteSpace: "nowrap",
            pointerEvents: "none",
            zIndex: 100,
          }}
        >
          {marker.label}
          <div
            style={{
              position: "absolute",
              top: "100%",
              left: "50%",
              transform: "translateX(-50%)",
              width: 0,
              height: 0,
              borderLeft: "4px solid transparent",
              borderRight: "4px solid transparent",
              borderTop: `4px solid ${theme.border}`,
            }}
          />
        </div>
      )}
    </div>
  );
};

// ============================================================================
// KeyframeMarkers Component
// ============================================================================

/**
 * KeyframeMarkers - Timeline keyframe and event markers
 */
export const KeyframeMarkers: React.FC<KeyframeMarkersProps> = ({
  keyframes,
  currentTime,
  timeRange,
  selectedId,
  interactive = true,
  onClick,
  onDoubleClick,
  onHover,
  onContextMenu,
  theme: themeProp,
  className = "",
}) => {
  const theme: TimelineTheme = { ...DEFAULT_TIMELINE_THEME, ...themeProp };
  const [hoveredId, setHoveredId] = useState<string | null>(null);

  // Filter and position markers within visible range
  const visibleMarkers = useMemo(() => {
    const duration = timeRange.end - timeRange.start;
    if (duration <= 0) return [];

    return keyframes
      .filter((kf) => kf.time >= timeRange.start && kf.time <= timeRange.end)
      .map((kf) => ({
        marker: kf,
        position: (kf.time - timeRange.start) / duration,
      }));
  }, [keyframes, timeRange]);

  // Find if current time is near a keyframe
  const currentKeyframeId = useMemo(() => {
    const threshold = (timeRange.end - timeRange.start) * 0.01; // 1% of range
    const current = keyframes.find(
      (kf) => Math.abs(kf.time - currentTime) < threshold
    );
    return current?.id ?? null;
  }, [keyframes, currentTime, timeRange]);

  // Handlers
  const handleMarkerClick = useCallback(
    (marker: TimelineMarker) => {
      onClick?.(marker);
    },
    [onClick]
  );

  const handleMarkerDoubleClick = useCallback(
    (marker: TimelineMarker) => {
      onDoubleClick?.(marker);
    },
    [onDoubleClick]
  );

  const handleMarkerHover = useCallback(
    (marker: TimelineMarker | null) => {
      setHoveredId(marker?.id ?? null);
      onHover?.(marker);
    },
    [onHover]
  );

  const handleContextMenu = useCallback(
    (marker: TimelineMarker, event: React.MouseEvent) => {
      event.preventDefault();
      onContextMenu?.(marker, event);
    },
    [onContextMenu]
  );

  return (
    <div
      className={`keyframe-markers ${className}`}
      style={{
        position: "relative",
        height: "32px",
        width: "100%",
      }}
    >
      {visibleMarkers.map(({ marker, position }) => (
        <Marker
          key={marker.id}
          marker={marker}
          position={position}
          isSelected={marker.id === selectedId}
          isHovered={marker.id === hoveredId}
          isCurrent={marker.id === currentKeyframeId}
          interactive={interactive}
          theme={theme}
          onClick={() => handleMarkerClick(marker)}
          onDoubleClick={() => handleMarkerDoubleClick(marker)}
          onMouseEnter={() => handleMarkerHover(marker)}
          onMouseLeave={() => handleMarkerHover(null)}
          onContextMenu={(e) => handleContextMenu(marker, e)}
        />
      ))}

      {/* Empty state */}
      {visibleMarkers.length === 0 && (
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            height: "100%",
            color: theme.textMuted,
            fontSize: "11px",
          }}
        >
          No keyframes in view
        </div>
      )}
    </div>
  );
};

KeyframeMarkers.displayName = "KeyframeMarkers";
