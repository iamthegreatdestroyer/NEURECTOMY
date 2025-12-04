/**
 * Timeline Scrubber Component
 *
 * Interactive timeline track with draggable playhead for time navigation.
 * Supports mouse/touch interaction, buffering indicator, and tooltips.
 *
 * @module @neurectomy/3d-engine/temporal/components/TimelineScrubber
 */

import React, { useRef, useState, useCallback, useEffect } from "react";
import type { TimelineScrubberProps, TimelineTheme } from "./types";
import { DEFAULT_TIMELINE_THEME } from "./types";

/**
 * TimelineScrubber - Interactive timeline track
 */
export const TimelineScrubber: React.FC<TimelineScrubberProps> = ({
  position,
  buffered = 0,
  isScrubbing = false,
  showTooltip = true,
  formatTime,
  currentTime,
  onScrubStart,
  onScrub,
  onScrubEnd,
  onClick,
  theme: themeProp,
  className = "",
}) => {
  const trackRef = useRef<HTMLDivElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [hoverPosition, setHoverPosition] = useState<number | null>(null);

  const theme: TimelineTheme = { ...DEFAULT_TIMELINE_THEME, ...themeProp };

  // Calculate position from event
  const getPositionFromEvent = useCallback(
    (e: MouseEvent | React.MouseEvent | TouchEvent | React.TouchEvent) => {
      if (!trackRef.current) return 0;

      const rect = trackRef.current.getBoundingClientRect();
      const clientX = "touches" in e ? (e.touches[0]?.clientX ?? 0) : e.clientX;
      const x = clientX - rect.left;

      return Math.max(0, Math.min(1, x / rect.width));
    },
    []
  );

  // Handle mouse/touch down
  const handlePointerDown = useCallback(
    (e: React.MouseEvent | React.TouchEvent) => {
      e.preventDefault();
      setIsDragging(true);

      const pos = getPositionFromEvent(e);
      onScrubStart?.();
      onScrub?.(pos);
    },
    [getPositionFromEvent, onScrubStart, onScrub]
  );

  // Handle mouse/touch move
  useEffect(() => {
    if (!isDragging) return;

    const handleMove = (e: MouseEvent | TouchEvent) => {
      e.preventDefault();
      const pos = getPositionFromEvent(e);
      onScrub?.(pos);
    };

    const handleUp = () => {
      setIsDragging(false);
      onScrubEnd?.();
    };

    window.addEventListener("mousemove", handleMove);
    window.addEventListener("mouseup", handleUp);
    window.addEventListener("touchmove", handleMove);
    window.addEventListener("touchend", handleUp);

    return () => {
      window.removeEventListener("mousemove", handleMove);
      window.removeEventListener("mouseup", handleUp);
      window.removeEventListener("touchmove", handleMove);
      window.removeEventListener("touchend", handleUp);
    };
  }, [isDragging, getPositionFromEvent, onScrub, onScrubEnd]);

  // Handle hover for tooltip
  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (!isDragging) {
        setHoverPosition(getPositionFromEvent(e));
      }
    },
    [isDragging, getPositionFromEvent]
  );

  const handleMouseLeave = useCallback(() => {
    if (!isDragging) {
      setHoverPosition(null);
    }
  }, [isDragging]);

  // Handle click
  const handleClick = useCallback(
    (e: React.MouseEvent) => {
      if (!isDragging) {
        const pos = getPositionFromEvent(e);
        onClick?.(pos);
      }
    },
    [isDragging, getPositionFromEvent, onClick]
  );

  // Format time for tooltip
  const defaultFormatTime = (ms: number) => {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);

    if (hours > 0) {
      return `${hours}:${String(minutes % 60).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
    }
    return `${minutes}:${String(seconds % 60).padStart(2, "0")}.${String(Math.floor((ms % 1000) / 10)).padStart(2, "0")}`;
  };

  const timeFormatter = formatTime || defaultFormatTime;

  return (
    <div
      ref={trackRef}
      className={`timeline-scrubber ${className}`}
      onMouseDown={handlePointerDown}
      onTouchStart={handlePointerDown}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      onClick={handleClick}
      style={{
        position: "relative",
        height: "24px",
        backgroundColor: theme.track,
        borderRadius: "4px",
        cursor: isDragging ? "grabbing" : "pointer",
        userSelect: "none",
        touchAction: "none",
      }}
      role="slider"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(position * 100)}
      aria-label="Timeline position"
      tabIndex={0}
    >
      {/* Buffered indicator */}
      <div
        className="timeline-buffered"
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          height: "100%",
          width: `${buffered * 100}%`,
          backgroundColor: `${theme.progress}33`,
          borderRadius: "4px",
          pointerEvents: "none",
        }}
      />

      {/* Progress track */}
      <div
        className="timeline-progress"
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          height: "100%",
          width: `${position * 100}%`,
          backgroundColor: theme.progress,
          borderRadius: "4px 0 0 4px",
          pointerEvents: "none",
        }}
      />

      {/* Playhead */}
      <div
        className="timeline-playhead"
        style={{
          position: "absolute",
          top: "-4px",
          left: `${position * 100}%`,
          transform: "translateX(-50%)",
          width: "12px",
          height: "32px",
          backgroundColor: theme.playhead,
          borderRadius: "6px",
          boxShadow: `0 0 8px ${theme.playhead}88`,
          pointerEvents: "none",
          transition: isDragging || isScrubbing ? "none" : "left 50ms ease-out",
        }}
      >
        {/* Playhead line */}
        <div
          style={{
            position: "absolute",
            top: "4px",
            left: "50%",
            transform: "translateX(-50%)",
            width: "2px",
            height: "24px",
            backgroundColor: "white",
            borderRadius: "1px",
          }}
        />
      </div>

      {/* Hover tooltip */}
      {showTooltip &&
        hoverPosition !== null &&
        !isDragging &&
        currentTime !== undefined && (
          <div
            className="timeline-tooltip"
            style={{
              position: "absolute",
              bottom: "100%",
              left: `${hoverPosition * 100}%`,
              transform: "translateX(-50%)",
              marginBottom: "8px",
              padding: "4px 8px",
              backgroundColor: theme.background,
              color: theme.text,
              fontSize: "12px",
              fontFamily: "monospace",
              borderRadius: "4px",
              border: `1px solid ${theme.border}`,
              whiteSpace: "nowrap",
              pointerEvents: "none",
              zIndex: 100,
            }}
          >
            {timeFormatter(currentTime)}
          </div>
        )}

      {/* Hover line */}
      {hoverPosition !== null && !isDragging && (
        <div
          className="timeline-hover-line"
          style={{
            position: "absolute",
            top: 0,
            left: `${hoverPosition * 100}%`,
            width: "1px",
            height: "100%",
            backgroundColor: `${theme.text}44`,
            pointerEvents: "none",
          }}
        />
      )}
    </div>
  );
};

TimelineScrubber.displayName = "TimelineScrubber";
