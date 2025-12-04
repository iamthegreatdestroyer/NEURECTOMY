/**
 * Time Ruler Component
 *
 * Time axis with tick marks and labels for timeline navigation.
 * Supports multiple time formats and adaptive tick density.
 *
 * @module @neurectomy/3d-engine/temporal/components/TimeRuler
 */

import React, { useMemo, useCallback } from "react";
import type { TimeRulerProps, TimelineTheme } from "./types";
import type { Timestamp } from "../types";
import { DEFAULT_TIMELINE_THEME } from "./types";

// ============================================================================
// Time Formatting Utilities
// ============================================================================

function formatTimeMs(ms: Timestamp): string {
  return `${ms.toLocaleString()}ms`;
}

function formatTimeSeconds(ms: Timestamp): string {
  const seconds = ms / 1000;
  return `${seconds.toFixed(2)}s`;
}

function formatTimeHMS(ms: Timestamp): string {
  const seconds = Math.floor(ms / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);

  if (hours > 0) {
    return `${hours}:${String(minutes % 60).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
  }
  if (minutes > 0) {
    return `${minutes}:${String(seconds % 60).padStart(2, "0")}`;
  }
  return `${seconds}s`;
}

function formatTimeRelative(ms: Timestamp, referenceMs: Timestamp = 0): string {
  const diff = ms - referenceMs;
  const sign = diff >= 0 ? "+" : "";
  return `${sign}${formatTimeHMS(Math.abs(diff))}`;
}

const TIME_FORMATTERS: Record<string, (ms: Timestamp) => string> = {
  ms: formatTimeMs,
  seconds: formatTimeSeconds,
  hms: formatTimeHMS,
  relative: (ms) => formatTimeRelative(ms, 0),
};

// ============================================================================
// Tick Calculation
// ============================================================================

interface TickMark {
  time: Timestamp;
  position: number;
  isMajor: boolean;
  label?: string;
}

function calculateTicks(
  startTime: Timestamp,
  endTime: Timestamp,
  density: "sparse" | "normal" | "dense",
  format: (ms: Timestamp) => string,
  showMinor: boolean
): TickMark[] {
  const duration = endTime - startTime;
  if (duration <= 0) return [];

  // Target number of major ticks based on density
  const targetMajorTicks = {
    sparse: 4,
    normal: 8,
    dense: 16,
  }[density];

  // Calculate appropriate interval
  const intervals = [
    10,
    20,
    50,
    100,
    200,
    500, // ms
    1000,
    2000,
    5000,
    10000,
    30000,
    60000, // seconds
    120000,
    300000,
    600000,
    1800000,
    3600000, // minutes/hours
  ];

  let majorInterval: number =
    intervals.find((i) => duration / i <= targetMajorTicks) ??
    intervals[intervals.length - 1] ??
    1000;

  // Ensure minimum reasonable interval
  if (majorInterval < 10) majorInterval = 10;

  const minorInterval = majorInterval / 5;

  // Generate ticks
  const ticks: TickMark[] = [];

  // Round start to nearest interval
  const firstMajor = Math.ceil(startTime / majorInterval) * majorInterval;
  const firstMinor = Math.ceil(startTime / minorInterval) * minorInterval;

  // Add major ticks
  for (let time = firstMajor; time <= endTime; time += majorInterval) {
    const position = (time - startTime) / duration;
    ticks.push({
      time,
      position,
      isMajor: true,
      label: format(time),
    });
  }

  // Add minor ticks
  if (showMinor) {
    for (let time = firstMinor; time <= endTime; time += minorInterval) {
      // Skip if already a major tick
      if (time % majorInterval === 0) continue;

      const position = (time - startTime) / duration;
      ticks.push({
        time,
        position,
        isMajor: false,
      });
    }
  }

  // Sort by position
  ticks.sort((a, b) => a.position - b.position);

  return ticks;
}

// ============================================================================
// TimeRuler Component
// ============================================================================

/**
 * TimeRuler - Time axis with tick marks and labels
 */
export const TimeRuler: React.FC<TimeRulerProps> = ({
  timeRange,
  currentTime,
  density = "normal",
  format = "hms",
  showMinorTicks = true,
  onClick,
  height = 32,
  theme: themeProp,
  className = "",
}) => {
  const theme: TimelineTheme = { ...DEFAULT_TIMELINE_THEME, ...themeProp };
  const formatter = TIME_FORMATTERS[format] || formatTimeHMS;

  // Calculate ticks
  const ticks = useMemo(() => {
    return calculateTicks(
      timeRange.start,
      timeRange.end,
      density,
      formatter,
      showMinorTicks
    );
  }, [timeRange, density, formatter, showMinorTicks]);

  // Current time position
  const currentPosition = useMemo(() => {
    if (currentTime === undefined) return null;
    const duration = timeRange.end - timeRange.start;
    if (duration <= 0) return null;
    const pos = (currentTime - timeRange.start) / duration;
    return pos >= 0 && pos <= 1 ? pos : null;
  }, [currentTime, timeRange]);

  // Handle click
  const handleClick = useCallback(
    (event: React.MouseEvent<HTMLDivElement>) => {
      if (!onClick) return;

      const rect = event.currentTarget.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const position = x / rect.width;
      const duration = timeRange.end - timeRange.start;
      const time = timeRange.start + position * duration;

      onClick(Math.round(time));
    },
    [onClick, timeRange]
  );

  return (
    <div
      className={`time-ruler ${className}`}
      onClick={onClick ? handleClick : undefined}
      style={{
        position: "relative",
        height,
        backgroundColor: theme.track,
        borderBottom: `1px solid ${theme.border}`,
        cursor: onClick ? "pointer" : "default",
        userSelect: "none",
      }}
    >
      {/* Ticks and labels */}
      {ticks.map((tick, i) => (
        <div
          key={`${tick.time}-${i}`}
          className={`time-ruler-tick ${tick.isMajor ? "major" : "minor"}`}
          style={{
            position: "absolute",
            left: `${tick.position * 100}%`,
            top: 0,
            transform: "translateX(-50%)",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
          }}
        >
          {/* Tick line */}
          <div
            style={{
              width: "1px",
              height: tick.isMajor ? "12px" : "6px",
              backgroundColor: tick.isMajor ? theme.text : theme.textMuted,
            }}
          />

          {/* Label (major ticks only) */}
          {tick.isMajor && tick.label && (
            <span
              style={{
                marginTop: "2px",
                fontSize: "10px",
                fontFamily: "monospace",
                color: theme.textMuted,
                whiteSpace: "nowrap",
              }}
            >
              {tick.label}
            </span>
          )}
        </div>
      ))}

      {/* Current time indicator */}
      {currentPosition !== null && (
        <div
          className="time-ruler-current"
          style={{
            position: "absolute",
            left: `${currentPosition * 100}%`,
            top: 0,
            bottom: 0,
            width: "2px",
            backgroundColor: theme.playhead,
            transform: "translateX(-50%)",
            pointerEvents: "none",
          }}
        />
      )}

      {/* Start and end labels */}
      <div
        style={{
          position: "absolute",
          left: "4px",
          bottom: "2px",
          fontSize: "9px",
          fontFamily: "monospace",
          color: theme.textMuted,
        }}
      >
        {formatter(timeRange.start)}
      </div>

      <div
        style={{
          position: "absolute",
          right: "4px",
          bottom: "2px",
          fontSize: "9px",
          fontFamily: "monospace",
          color: theme.textMuted,
        }}
      >
        {formatter(timeRange.end)}
      </div>
    </div>
  );
};

TimeRuler.displayName = "TimeRuler";
