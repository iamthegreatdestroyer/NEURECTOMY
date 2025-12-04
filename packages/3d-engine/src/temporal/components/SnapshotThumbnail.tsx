/**
 * Snapshot Thumbnail Component
 *
 * Preview thumbnail for a timeline state snapshot.
 * Displays captured 3D scene preview with loading states.
 *
 * @module @neurectomy/3d-engine/temporal/components/SnapshotThumbnail
 */

import React, { useState, useCallback, useMemo } from "react";
import type { SnapshotThumbnailProps, TimelineTheme } from "./types";
import { DEFAULT_TIMELINE_THEME } from "./types";

// ============================================================================
// SnapshotThumbnail Component
// ============================================================================

/**
 * SnapshotThumbnail - Preview thumbnail for timeline snapshots
 */
export const SnapshotThumbnail: React.FC<SnapshotThumbnailProps> = ({
  snapshotId,
  time,
  thumbnail,
  isCurrent = false,
  width = 120,
  height = 68,
  onClick,
  onLoad,
  theme: themeProp,
  className = "",
}) => {
  const theme: TimelineTheme = { ...DEFAULT_TIMELINE_THEME, ...themeProp };
  const [isLoading, setIsLoading] = useState(true);
  const [hasError, setHasError] = useState(false);
  const [isHovered, setIsHovered] = useState(false);

  const handleLoad = useCallback(() => {
    setIsLoading(false);
    setHasError(false);
    onLoad?.();
  }, [onLoad]);

  const handleError = useCallback(() => {
    setIsLoading(false);
    setHasError(true);
  }, []);

  const handleClick = useCallback(() => {
    onClick?.(snapshotId);
  }, [onClick, snapshotId]);

  // Format time for display
  const formattedTime = useMemo(() => {
    const seconds = Math.floor(time / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);

    if (hours > 0) {
      return `${hours}:${String(minutes % 60).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
    }
    return `${minutes}:${String(seconds % 60).padStart(2, "0")}`;
  }, [time]);

  return (
    <div
      className={`snapshot-thumbnail ${className}`}
      onClick={onClick ? handleClick : undefined}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      style={{
        position: "relative",
        width,
        height,
        borderRadius: "6px",
        overflow: "hidden",
        backgroundColor: theme.track,
        border: `2px solid ${isCurrent ? theme.playhead : isHovered ? theme.hover : theme.border}`,
        cursor: onClick ? "pointer" : "default",
        transition: "border-color 150ms, transform 150ms",
        transform: isHovered ? "scale(1.05)" : "scale(1)",
        boxShadow: isCurrent ? `0 0 8px ${theme.playhead}88` : "none",
      }}
      role={onClick ? "button" : undefined}
      tabIndex={onClick ? 0 : undefined}
      aria-label={`Snapshot at ${formattedTime}`}
    >
      {/* Thumbnail image */}
      {thumbnail && !hasError ? (
        <img
          src={thumbnail}
          alt={`Snapshot at ${formattedTime}`}
          onLoad={handleLoad}
          onError={handleError}
          style={{
            width: "100%",
            height: "100%",
            objectFit: "cover",
            opacity: isLoading ? 0 : 1,
            transition: "opacity 200ms",
          }}
        />
      ) : (
        // Placeholder when no thumbnail
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            width: "100%",
            height: "100%",
            backgroundColor: theme.background,
          }}
        >
          {hasError ? (
            <svg
              width="24"
              height="24"
              viewBox="0 0 24 24"
              fill={theme.textMuted}
            >
              <path d="M21 19V5c0-1.1-.9-2-2-2H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2zM8.5 13.5l2.5 3.01L14.5 12l4.5 6H5l3.5-4.5z" />
              <path
                d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"
                fill="#ff4444"
                opacity="0.5"
              />
            </svg>
          ) : (
            <svg
              width="24"
              height="24"
              viewBox="0 0 24 24"
              fill={theme.textMuted}
            >
              <path d="M21 19V5c0-1.1-.9-2-2-2H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2zM8.5 13.5l2.5 3.01L14.5 12l4.5 6H5l3.5-4.5z" />
            </svg>
          )}
        </div>
      )}

      {/* Loading spinner */}
      {isLoading && thumbnail && !hasError && (
        <div
          style={{
            position: "absolute",
            top: "50%",
            left: "50%",
            transform: "translate(-50%, -50%)",
          }}
        >
          <div
            style={{
              width: "24px",
              height: "24px",
              border: `2px solid ${theme.border}`,
              borderTopColor: theme.progress,
              borderRadius: "50%",
              animation: "spin 1s linear infinite",
            }}
          />
        </div>
      )}

      {/* Time label */}
      <div
        style={{
          position: "absolute",
          bottom: 0,
          left: 0,
          right: 0,
          padding: "4px 6px",
          background: "linear-gradient(transparent, rgba(0,0,0,0.8))",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
        }}
      >
        <span
          style={{
            fontSize: "10px",
            fontFamily: "monospace",
            color: "#ffffff",
          }}
        >
          {formattedTime}
        </span>

        {isCurrent && (
          <span
            style={{
              fontSize: "8px",
              padding: "1px 4px",
              backgroundColor: theme.playhead,
              color: "#ffffff",
              borderRadius: "2px",
              fontWeight: "bold",
            }}
          >
            NOW
          </span>
        )}
      </div>

      {/* Hover overlay */}
      {isHovered && onClick && (
        <div
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: "rgba(74, 144, 217, 0.2)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <svg width="32" height="32" viewBox="0 0 24 24" fill="#ffffff">
            <path d="M8 5v14l11-7z" />
          </svg>
        </div>
      )}

      {/* Inline CSS for animation */}
      <style>
        {`
          @keyframes spin {
            to { transform: rotate(360deg); }
          }
        `}
      </style>
    </div>
  );
};

SnapshotThumbnail.displayName = "SnapshotThumbnail";
