/**
 * Playback Controls Component
 *
 * VCR-style playback controls for timeline navigation.
 * Includes play/pause, step forward/back, keyframe navigation, and speed.
 *
 * @module @neurectomy/3d-engine/temporal/components/PlaybackControls
 */

import React, { useCallback, useMemo } from "react";
import type { PlaybackControlsProps, TimelineTheme } from "./types";
import { DEFAULT_TIMELINE_THEME } from "./types";

// ============================================================================
// Icon Components (inline SVG for minimal dependencies)
// ============================================================================

interface IconProps {
  size?: number;
  color?: string;
}

const PlayIcon: React.FC<IconProps> = ({
  size = 24,
  color = "currentColor",
}) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill={color}>
    <path d="M8 5v14l11-7z" />
  </svg>
);

const PauseIcon: React.FC<IconProps> = ({
  size = 24,
  color = "currentColor",
}) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill={color}>
    <path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z" />
  </svg>
);

const StopIcon: React.FC<IconProps> = ({
  size = 24,
  color = "currentColor",
}) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill={color}>
    <path d="M6 6h12v12H6z" />
  </svg>
);

const StepForwardIcon: React.FC<IconProps> = ({
  size = 24,
  color = "currentColor",
}) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill={color}>
    <path d="M6 18l8.5-6L6 6v12zM16 6v12h2V6h-2z" />
  </svg>
);

const StepBackwardIcon: React.FC<IconProps> = ({
  size = 24,
  color = "currentColor",
}) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill={color}>
    <path d="M6 6h2v12H6V6zm3.5 6l8.5 6V6l-8.5 6z" />
  </svg>
);

const SkipForwardIcon: React.FC<IconProps> = ({
  size = 24,
  color = "currentColor",
}) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill={color}>
    <path d="M4 18l8.5-6L4 6v12zm9-12v12l8.5-6L13 6z" />
  </svg>
);

const SkipBackwardIcon: React.FC<IconProps> = ({
  size = 24,
  color = "currentColor",
}) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill={color}>
    <path d="M11 18V6l-8.5 6 8.5 6zm.5-6l8.5 6V6l-8.5 6z" />
  </svg>
);

const LoopIcon: React.FC<IconProps> = ({
  size = 24,
  color = "currentColor",
}) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill={color}>
    <path d="M12 4V1L8 5l4 4V6c3.31 0 6 2.69 6 6 0 1.01-.25 1.97-.7 2.8l1.46 1.46C19.54 15.03 20 13.57 20 12c0-4.42-3.58-8-8-8zm0 14c-3.31 0-6-2.69-6-6 0-1.01.25-1.97.7-2.8L5.24 7.74C4.46 8.97 4 10.43 4 12c0 4.42 3.58 8 8 8v3l4-4-4-4v3z" />
  </svg>
);

const ReverseIcon: React.FC<IconProps> = ({
  size = 24,
  color = "currentColor",
}) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill={color}>
    <path d="M18 6l-8.5 6L18 18V6zM8 18H6V6h2v12z" />
  </svg>
);

// ============================================================================
// Control Button Component
// ============================================================================

interface ControlButtonProps {
  icon: React.FC<IconProps>;
  onClick?: () => void;
  disabled?: boolean;
  active?: boolean;
  title?: string;
  size: "sm" | "md" | "lg";
  theme: TimelineTheme;
}

const ControlButton: React.FC<ControlButtonProps> = ({
  icon: Icon,
  onClick,
  disabled = false,
  active = false,
  title,
  size,
  theme,
}) => {
  const sizes = {
    sm: { button: 28, icon: 16 },
    md: { button: 36, icon: 20 },
    lg: { button: 48, icon: 28 },
  };

  const { button: buttonSize, icon: iconSize } = sizes[size];

  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      title={title}
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        width: buttonSize,
        height: buttonSize,
        padding: 0,
        border: "none",
        borderRadius: "50%",
        backgroundColor: active ? theme.hover : "transparent",
        color: disabled ? theme.textMuted : theme.text,
        cursor: disabled ? "not-allowed" : "pointer",
        opacity: disabled ? 0.5 : 1,
        transition: "background-color 150ms, opacity 150ms",
      }}
      onMouseEnter={(e) => {
        if (!disabled) {
          e.currentTarget.style.backgroundColor = theme.hover;
        }
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.backgroundColor = active
          ? theme.hover
          : "transparent";
      }}
    >
      <Icon size={iconSize} />
    </button>
  );
};

// ============================================================================
// PlaybackControls Component
// ============================================================================

/**
 * PlaybackControls - VCR-style playback controls
 */
export const PlaybackControls: React.FC<PlaybackControlsProps> = ({
  status,
  extended = false,
  showSpeedControl = true,
  showLoopControl = true,
  size = "md",
  onPlay,
  onPause,
  onStop,
  onStepForward,
  onStepBackward,
  onNextKeyframe,
  onPrevKeyframe,
  onSpeedChange,
  onDirectionChange,
  onLoopToggle,
  theme: themeProp,
  className = "",
}) => {
  const theme: TimelineTheme = { ...DEFAULT_TIMELINE_THEME, ...themeProp };

  const isPlaying = status.state === "playing";
  const isReverse = status.direction === "backward";

  const handlePlayPause = useCallback(() => {
    if (isPlaying) {
      onPause?.();
    } else {
      onPlay?.();
    }
  }, [isPlaying, onPlay, onPause]);

  const handleSpeedPreset = useCallback(
    (preset: number) => {
      onSpeedChange?.(preset);
    },
    [onSpeedChange]
  );

  const handleDirectionToggle = useCallback(() => {
    onDirectionChange?.(isReverse ? "forward" : "backward");
  }, [isReverse, onDirectionChange]);

  // Speed presets
  const speedPresets = useMemo(() => [0.25, 0.5, 1, 2, 4, 8], []);

  return (
    <div
      className={`playback-controls ${className}`}
      style={{
        display: "flex",
        alignItems: "center",
        gap: "4px",
        padding: "8px",
        backgroundColor: theme.background,
        borderRadius: "8px",
        border: `1px solid ${theme.border}`,
      }}
    >
      {/* Extended: Skip to start */}
      {extended && (
        <ControlButton
          icon={SkipBackwardIcon}
          onClick={onPrevKeyframe}
          title="Previous Keyframe (Home)"
          size={size}
          theme={theme}
        />
      )}

      {/* Step backward */}
      <ControlButton
        icon={StepBackwardIcon}
        onClick={onStepBackward}
        title="Step Backward (←)"
        size={size}
        theme={theme}
      />

      {/* Play/Pause */}
      <ControlButton
        icon={isPlaying ? PauseIcon : PlayIcon}
        onClick={handlePlayPause}
        title={isPlaying ? "Pause (Space)" : "Play (Space)"}
        size={size}
        theme={theme}
      />

      {/* Stop */}
      {extended && (
        <ControlButton
          icon={StopIcon}
          onClick={onStop}
          title="Stop (Escape)"
          size={size}
          theme={theme}
        />
      )}

      {/* Step forward */}
      <ControlButton
        icon={StepForwardIcon}
        onClick={onStepForward}
        title="Step Forward (→)"
        size={size}
        theme={theme}
      />

      {/* Extended: Skip to end */}
      {extended && (
        <ControlButton
          icon={SkipForwardIcon}
          onClick={onNextKeyframe}
          title="Next Keyframe (End)"
          size={size}
          theme={theme}
        />
      )}

      {/* Separator */}
      <div
        style={{
          width: "1px",
          height: "24px",
          backgroundColor: theme.border,
          margin: "0 8px",
        }}
      />

      {/* Direction toggle */}
      {extended && (
        <ControlButton
          icon={ReverseIcon}
          onClick={handleDirectionToggle}
          active={isReverse}
          title={isReverse ? "Forward Playback" : "Reverse Playback"}
          size={size}
          theme={theme}
        />
      )}

      {/* Loop toggle */}
      {showLoopControl && (
        <ControlButton
          icon={LoopIcon}
          onClick={onLoopToggle}
          active={status.loop}
          title={status.loop ? "Disable Loop (L)" : "Enable Loop (L)"}
          size={size}
          theme={theme}
        />
      )}

      {/* Speed control */}
      {showSpeedControl && (
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "4px",
            marginLeft: "8px",
          }}
        >
          <span
            style={{
              fontSize: size === "sm" ? "10px" : "12px",
              color: theme.textMuted,
              fontFamily: "monospace",
              minWidth: "40px",
              textAlign: "center",
            }}
          >
            {status.speed}x
          </span>

          {extended && (
            <div style={{ display: "flex", gap: "2px" }}>
              {speedPresets.map((preset) => (
                <button
                  key={preset}
                  type="button"
                  onClick={() => handleSpeedPreset(preset)}
                  style={{
                    padding: "2px 6px",
                    fontSize: "10px",
                    fontFamily: "monospace",
                    border: "none",
                    borderRadius: "4px",
                    backgroundColor:
                      status.speed === preset ? theme.hover : "transparent",
                    color: theme.text,
                    cursor: "pointer",
                    transition: "background-color 150ms",
                  }}
                >
                  {preset}x
                </button>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

PlaybackControls.displayName = "PlaybackControls";
