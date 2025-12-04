/**
 * Main Timeline Component
 *
 * Complete 4D timeline interface combining scrubber, controls, markers,
 * and ruler into a cohesive temporal navigation experience.
 *
 * @module @neurectomy/3d-engine/temporal/components/Timeline
 */

import React, {
  useCallback,
  useMemo,
  useState,
  useRef,
  useEffect,
} from "react";
import type {
  TimelineProps,
  TimelineTheme,
  TimelineViewport,
  TimelineSelection,
} from "./types";
import type { TimelineMarker, Timestamp, TimeRange } from "../types";
import type { PlaybackStatus } from "../timeline-navigator";
import { DEFAULT_TIMELINE_THEME } from "./types";
import { TimelineScrubber } from "./TimelineScrubber";
import { PlaybackControls } from "./PlaybackControls";
import { KeyframeMarkers } from "./KeyframeMarkers";
import { TimeRuler } from "./TimeRuler";
import { SpeedControl } from "./SpeedControl";

// ============================================================================
// Timeline Component
// ============================================================================

/**
 * Timeline - Complete 4D temporal navigation interface
 */
export const Timeline: React.FC<TimelineProps> = ({
  currentTime,
  timeRange,
  playbackStatus,
  markers = [],
  isScrubbing: isExternalScrubbing = false,
  onSeek,
  onPlaybackChange,
  onScrubStart,
  onScrub,
  onScrubEnd,
  onMarkerClick,
  onRangeChange,
  showRuler = true,
  showKeyframes = true,
  showEvents = true,
  showThumbnails = false,
  showMinimap = false,
  height = 120,
  theme: themeProp,
  className = "",
}) => {
  const theme: TimelineTheme = { ...DEFAULT_TIMELINE_THEME, ...themeProp };
  const containerRef = useRef<HTMLDivElement>(null);

  // Local state
  const [isLocalScrubbing, setIsLocalScrubbing] = useState(false);
  const [localScrubTime, setLocalScrubTime] = useState<Timestamp | null>(null);
  const [viewport, setViewport] = useState<TimelineViewport>({
    start: timeRange.start,
    end: timeRange.end,
    zoom: 1,
    panOffset: 0,
  });
  const [selection, setSelection] = useState<TimelineSelection>({
    start: 0,
    end: 0,
    active: false,
  });
  const [selectedMarkerId, setSelectedMarkerId] = useState<string | null>(null);

  const isScrubbing = isExternalScrubbing || isLocalScrubbing;
  const displayTime =
    isScrubbing && localScrubTime !== null ? localScrubTime : currentTime;

  // Calculate position from time
  const duration = timeRange.end - timeRange.start;
  const position =
    duration > 0 ? (displayTime - timeRange.start) / duration : 0;

  // Filter markers by type
  const keyframes = useMemo(
    () => markers.filter((m) => m.type === "keyframe"),
    [markers]
  );

  const events = useMemo(
    () => markers.filter((m) => m.type !== "keyframe"),
    [markers]
  );

  // Handle scrub start
  const handleScrubStart = useCallback(() => {
    setIsLocalScrubbing(true);
    setLocalScrubTime(displayTime);
    onScrubStart?.();
  }, [displayTime, onScrubStart]);

  // Handle scrub move
  const handleScrub = useCallback(
    (pos: number) => {
      const time = Math.round(timeRange.start + pos * duration);
      setLocalScrubTime(time);
      onScrub?.(time);
    },
    [timeRange, duration, onScrub]
  );

  // Handle scrub end
  const handleScrubEnd = useCallback(() => {
    if (localScrubTime !== null) {
      onSeek?.(localScrubTime);
    }
    setIsLocalScrubbing(false);
    setLocalScrubTime(null);
    onScrubEnd?.();
  }, [localScrubTime, onSeek, onScrubEnd]);

  // Handle track click
  const handleTrackClick = useCallback(
    (pos: number) => {
      if (!isScrubbing) {
        const time = Math.round(timeRange.start + pos * duration);
        onSeek?.(time);
      }
    },
    [isScrubbing, timeRange, duration, onSeek]
  );

  // Handle playback state changes
  const handlePlay = useCallback(() => {
    onPlaybackChange?.({ state: "playing" });
  }, [onPlaybackChange]);

  const handlePause = useCallback(() => {
    onPlaybackChange?.({ state: "paused" });
  }, [onPlaybackChange]);

  const handleStop = useCallback(() => {
    onPlaybackChange?.({ state: "paused" });
    onSeek?.(timeRange.start);
  }, [onPlaybackChange, onSeek, timeRange]);

  const handleStepForward = useCallback(() => {
    const frameTime = 1000 / 30; // 30fps
    const newTime = Math.min(currentTime + frameTime, timeRange.end);
    onSeek?.(newTime);
  }, [currentTime, timeRange, onSeek]);

  const handleStepBackward = useCallback(() => {
    const frameTime = 1000 / 30;
    const newTime = Math.max(currentTime - frameTime, timeRange.start);
    onSeek?.(newTime);
  }, [currentTime, timeRange, onSeek]);

  const handleNextKeyframe = useCallback(() => {
    const nextKf = keyframes.find((kf) => kf.time > currentTime);
    if (nextKf) {
      onSeek?.(nextKf.time);
    } else {
      onSeek?.(timeRange.end);
    }
  }, [keyframes, currentTime, timeRange, onSeek]);

  const handlePrevKeyframe = useCallback(() => {
    const prevKfs = keyframes.filter((kf) => kf.time < currentTime - 100);
    const prevKf = prevKfs[prevKfs.length - 1];
    if (prevKf) {
      onSeek?.(prevKf.time);
    } else {
      onSeek?.(timeRange.start);
    }
  }, [keyframes, currentTime, timeRange, onSeek]);

  const handleSpeedChange = useCallback(
    (speed: number) => {
      onPlaybackChange?.({ speed });
    },
    [onPlaybackChange]
  );

  const handleDirectionChange = useCallback(
    (direction: "forward" | "backward") => {
      onPlaybackChange?.({ direction });
    },
    [onPlaybackChange]
  );

  const handleLoopToggle = useCallback(() => {
    onPlaybackChange?.({ loop: !playbackStatus.loop });
  }, [playbackStatus.loop, onPlaybackChange]);

  // Handle ruler click
  const handleRulerClick = useCallback(
    (time: Timestamp) => {
      onSeek?.(time);
    },
    [onSeek]
  );

  // Handle marker click
  const handleMarkerClick = useCallback(
    (marker: TimelineMarker) => {
      setSelectedMarkerId(marker.id);
      onSeek?.(marker.time);
      onMarkerClick?.(marker);
    },
    [onSeek, onMarkerClick]
  );

  const handleMarkerDoubleClick = useCallback(
    (marker: TimelineMarker) => {
      // Double-click focuses on marker range
      const padding = duration * 0.1;
      const newStart = Math.max(timeRange.start, marker.time - padding);
      const newEnd = Math.min(timeRange.end, marker.time + padding);
      onRangeChange?.({ start: newStart, end: newEnd });
    },
    [duration, timeRange, onRangeChange]
  );

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Only handle if timeline is focused or if it's a global shortcut
      const target = e.target as HTMLElement;
      if (target.tagName === "INPUT" || target.tagName === "TEXTAREA") return;

      switch (e.code) {
        case "Space":
          e.preventDefault();
          if (playbackStatus.state === "playing") {
            handlePause();
          } else {
            handlePlay();
          }
          break;
        case "ArrowLeft":
          e.preventDefault();
          if (e.shiftKey) {
            handlePrevKeyframe();
          } else {
            handleStepBackward();
          }
          break;
        case "ArrowRight":
          e.preventDefault();
          if (e.shiftKey) {
            handleNextKeyframe();
          } else {
            handleStepForward();
          }
          break;
        case "Home":
          e.preventDefault();
          onSeek?.(timeRange.start);
          break;
        case "End":
          e.preventDefault();
          onSeek?.(timeRange.end);
          break;
        case "KeyL":
          e.preventDefault();
          handleLoopToggle();
          break;
        case "Escape":
          e.preventDefault();
          handleStop();
          break;
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [
    playbackStatus.state,
    handlePlay,
    handlePause,
    handleStop,
    handleStepForward,
    handleStepBackward,
    handleNextKeyframe,
    handlePrevKeyframe,
    handleLoopToggle,
    onSeek,
    timeRange,
  ]);

  // Format current time
  const formatCurrentTime = useCallback((ms: Timestamp) => {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    const remainingMs = ms % 1000;

    if (hours > 0) {
      return `${hours}:${String(minutes % 60).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}.${String(remainingMs).padStart(3, "0")}`;
    }
    return `${minutes}:${String(seconds % 60).padStart(2, "0")}.${String(remainingMs).padStart(3, "0")}`;
  }, []);

  return (
    <div
      ref={containerRef}
      className={`timeline ${className}`}
      style={{
        display: "flex",
        flexDirection: "column",
        height,
        backgroundColor: theme.background,
        borderRadius: "8px",
        border: `1px solid ${theme.border}`,
        overflow: "hidden",
        userSelect: "none",
      }}
      tabIndex={0}
      role="group"
      aria-label="Timeline navigation"
    >
      {/* Top section: Time ruler */}
      {showRuler && (
        <TimeRuler
          timeRange={timeRange}
          currentTime={displayTime}
          density="normal"
          format="hms"
          showMinorTicks={true}
          onClick={handleRulerClick}
          height={28}
          theme={theme}
        />
      )}

      {/* Middle section: Keyframe markers */}
      {showKeyframes && keyframes.length > 0 && (
        <KeyframeMarkers
          keyframes={keyframes}
          currentTime={displayTime}
          timeRange={timeRange}
          selectedId={selectedMarkerId ?? undefined}
          interactive={true}
          onClick={handleMarkerClick}
          onDoubleClick={handleMarkerDoubleClick}
          theme={theme}
        />
      )}

      {/* Main scrubber track */}
      <div style={{ padding: "8px 12px", flex: 1 }}>
        <TimelineScrubber
          position={position}
          isScrubbing={isScrubbing}
          showTooltip={true}
          currentTime={displayTime}
          formatTime={formatCurrentTime}
          onScrubStart={handleScrubStart}
          onScrub={handleScrub}
          onScrubEnd={handleScrubEnd}
          onClick={handleTrackClick}
          theme={theme}
        />
      </div>

      {/* Bottom section: Controls */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "8px 12px",
          borderTop: `1px solid ${theme.border}`,
          gap: "12px",
        }}
      >
        {/* Left: Current time display */}
        <div
          style={{
            fontSize: "13px",
            fontFamily: "monospace",
            fontWeight: "bold",
            color: theme.text,
            minWidth: "100px",
          }}
        >
          {formatCurrentTime(displayTime)}
        </div>

        {/* Center: Playback controls */}
        <PlaybackControls
          status={playbackStatus}
          extended={false}
          showSpeedControl={false}
          showLoopControl={true}
          size="md"
          onPlay={handlePlay}
          onPause={handlePause}
          onStop={handleStop}
          onStepForward={handleStepForward}
          onStepBackward={handleStepBackward}
          onNextKeyframe={handleNextKeyframe}
          onPrevKeyframe={handlePrevKeyframe}
          onSpeedChange={handleSpeedChange}
          onDirectionChange={handleDirectionChange}
          onLoopToggle={handleLoopToggle}
          theme={theme}
        />

        {/* Right: Speed control and duration */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "12px",
          }}
        >
          <SpeedControl
            speed={playbackStatus.speed}
            presets={[0.5, 1, 2, 4]}
            variant="buttons"
            onChange={handleSpeedChange}
            theme={theme}
          />

          <div
            style={{
              fontSize: "11px",
              fontFamily: "monospace",
              color: theme.textMuted,
            }}
          >
            / {formatCurrentTime(timeRange.end)}
          </div>
        </div>
      </div>
    </div>
  );
};

Timeline.displayName = "Timeline";
