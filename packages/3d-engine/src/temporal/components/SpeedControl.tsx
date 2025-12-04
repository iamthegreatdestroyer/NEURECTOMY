/**
 * Speed Control Component
 *
 * Playback speed selector with slider, buttons, or dropdown variants.
 *
 * @module @neurectomy/3d-engine/temporal/components/SpeedControl
 */

import React, { useCallback, useMemo, useState } from "react";
import type { SpeedControlProps, TimelineTheme } from "./types";
import { DEFAULT_TIMELINE_THEME } from "./types";

// ============================================================================
// SpeedControl Component
// ============================================================================

const DEFAULT_PRESETS = [0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 4, 8, 16];

/**
 * SpeedControl - Playback speed selector
 */
export const SpeedControl: React.FC<SpeedControlProps> = ({
  speed,
  presets = DEFAULT_PRESETS,
  min = 0.1,
  max = 16,
  step = 0.1,
  variant = "buttons",
  onChange,
  theme: themeProp,
  className = "",
}) => {
  const theme: TimelineTheme = { ...DEFAULT_TIMELINE_THEME, ...themeProp };
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);

  const handleSliderChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const value = parseFloat(e.target.value);
      onChange?.(value);
    },
    [onChange]
  );

  const handlePresetClick = useCallback(
    (preset: number) => {
      onChange?.(preset);
      setIsDropdownOpen(false);
    },
    [onChange]
  );

  const toggleDropdown = useCallback(() => {
    setIsDropdownOpen((prev) => !prev);
  }, []);

  // Format speed for display
  const formatSpeed = useCallback((s: number) => {
    if (s === Math.floor(s)) return `${s}x`;
    if (s * 100 === Math.floor(s * 100)) return `${s.toFixed(2)}x`;
    return `${s.toFixed(1)}x`;
  }, []);

  // Find closest preset
  const closestPreset = useMemo(() => {
    return presets.reduce((prev, curr) =>
      Math.abs(curr - speed) < Math.abs(prev - speed) ? curr : prev
    );
  }, [presets, speed]);

  // Render slider variant
  if (variant === "slider") {
    return (
      <div
        className={`speed-control speed-control-slider ${className}`}
        style={{
          display: "flex",
          alignItems: "center",
          gap: "8px",
          padding: "4px 8px",
          backgroundColor: theme.background,
          borderRadius: "6px",
          border: `1px solid ${theme.border}`,
        }}
      >
        <span
          style={{
            fontSize: "10px",
            fontFamily: "monospace",
            color: theme.textMuted,
            minWidth: "24px",
          }}
        >
          {formatSpeed(min)}
        </span>

        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={speed}
          onChange={handleSliderChange}
          style={{
            width: "80px",
            height: "4px",
            WebkitAppearance: "none",
            appearance: "none",
            background: `linear-gradient(to right, ${theme.progress} ${((speed - min) / (max - min)) * 100}%, ${theme.track} 0%)`,
            borderRadius: "2px",
            cursor: "pointer",
          }}
        />

        <span
          style={{
            fontSize: "12px",
            fontFamily: "monospace",
            color: theme.text,
            fontWeight: "bold",
            minWidth: "40px",
            textAlign: "center",
          }}
        >
          {formatSpeed(speed)}
        </span>
      </div>
    );
  }

  // Render buttons variant
  if (variant === "buttons") {
    return (
      <div
        className={`speed-control speed-control-buttons ${className}`}
        style={{
          display: "flex",
          alignItems: "center",
          gap: "2px",
          padding: "2px",
          backgroundColor: theme.background,
          borderRadius: "6px",
          border: `1px solid ${theme.border}`,
        }}
      >
        {presets.map((preset) => (
          <button
            key={preset}
            type="button"
            onClick={() => handlePresetClick(preset)}
            style={{
              padding: "4px 8px",
              fontSize: "11px",
              fontFamily: "monospace",
              fontWeight: speed === preset ? "bold" : "normal",
              border: "none",
              borderRadius: "4px",
              backgroundColor:
                speed === preset ? theme.progress : "transparent",
              color: speed === preset ? "#ffffff" : theme.text,
              cursor: "pointer",
              transition: "background-color 150ms, color 150ms",
            }}
            onMouseEnter={(e) => {
              if (speed !== preset) {
                e.currentTarget.style.backgroundColor = theme.hover;
              }
            }}
            onMouseLeave={(e) => {
              if (speed !== preset) {
                e.currentTarget.style.backgroundColor = "transparent";
              }
            }}
          >
            {formatSpeed(preset)}
          </button>
        ))}
      </div>
    );
  }

  // Render dropdown variant
  return (
    <div
      className={`speed-control speed-control-dropdown ${className}`}
      style={{
        position: "relative",
        display: "inline-block",
      }}
    >
      <button
        type="button"
        onClick={toggleDropdown}
        style={{
          display: "flex",
          alignItems: "center",
          gap: "4px",
          padding: "6px 12px",
          fontSize: "12px",
          fontFamily: "monospace",
          fontWeight: "bold",
          border: `1px solid ${theme.border}`,
          borderRadius: "6px",
          backgroundColor: theme.background,
          color: theme.text,
          cursor: "pointer",
        }}
      >
        {formatSpeed(speed)}
        <svg
          width="12"
          height="12"
          viewBox="0 0 24 24"
          fill="currentColor"
          style={{
            transform: isDropdownOpen ? "rotate(180deg)" : "none",
            transition: "transform 150ms",
          }}
        >
          <path d="M7 10l5 5 5-5z" />
        </svg>
      </button>

      {isDropdownOpen && (
        <>
          {/* Backdrop to close dropdown */}
          <div
            style={{
              position: "fixed",
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              zIndex: 99,
            }}
            onClick={() => setIsDropdownOpen(false)}
          />

          {/* Dropdown menu */}
          <div
            style={{
              position: "absolute",
              top: "100%",
              left: 0,
              marginTop: "4px",
              minWidth: "100%",
              padding: "4px",
              backgroundColor: theme.background,
              border: `1px solid ${theme.border}`,
              borderRadius: "6px",
              boxShadow: "0 4px 12px rgba(0, 0, 0, 0.3)",
              zIndex: 100,
            }}
          >
            {presets.map((preset) => (
              <button
                key={preset}
                type="button"
                onClick={() => handlePresetClick(preset)}
                style={{
                  display: "block",
                  width: "100%",
                  padding: "8px 12px",
                  fontSize: "12px",
                  fontFamily: "monospace",
                  fontWeight: speed === preset ? "bold" : "normal",
                  textAlign: "left",
                  border: "none",
                  borderRadius: "4px",
                  backgroundColor:
                    speed === preset ? theme.progress : "transparent",
                  color: speed === preset ? "#ffffff" : theme.text,
                  cursor: "pointer",
                }}
                onMouseEnter={(e) => {
                  if (speed !== preset) {
                    e.currentTarget.style.backgroundColor = theme.hover;
                  }
                }}
                onMouseLeave={(e) => {
                  if (speed !== preset) {
                    e.currentTarget.style.backgroundColor = "transparent";
                  }
                }}
              >
                {formatSpeed(preset)}
              </button>
            ))}
          </div>
        </>
      )}
    </div>
  );
};

SpeedControl.displayName = "SpeedControl";
