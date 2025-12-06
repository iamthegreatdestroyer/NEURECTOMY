/**
 * @file LoadingSkeleton Component
 * @description Loading skeleton for 3D canvas with animated placeholder
 * @module @neurectomy/3d-engine/components
 * @agents @CANVAS @APEX
 */

import React from "react";

export interface LoadingSkeletonProps {
  /** Custom message to display */
  message?: string;
  /** Show progress indicator */
  showProgress?: boolean;
  /** Progress value (0-100) */
  progress?: number;
  /** Custom className for the container */
  className?: string;
  /** Custom styles */
  style?: React.CSSProperties;
  /** Variant of the skeleton */
  variant?: "default" | "minimal" | "detailed";
}

/**
 * Loading skeleton component for 3D canvas
 * Shows animated placeholder while WebGPU/WebGL content loads
 */
export const LoadingSkeleton: React.FC<LoadingSkeletonProps> = ({
  message = "Loading 3D scene...",
  showProgress = false,
  progress = 0,
  className = "",
  style,
  variant = "default",
}) => {
  const baseStyles: React.CSSProperties = {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    width: "100%",
    height: "100%",
    minHeight: "200px",
    background:
      "linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)",
    color: "#e0e0e0",
    fontFamily: "system-ui, -apple-system, sans-serif",
    position: "relative",
    overflow: "hidden",
    ...style,
  };

  const pulseKeyframes = `
    @keyframes pulse {
      0%, 100% { opacity: 0.4; transform: scale(1); }
      50% { opacity: 1; transform: scale(1.05); }
    }
    @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }
    @keyframes shimmer {
      0% { background-position: -200% 0; }
      100% { background-position: 200% 0; }
    }
  `;

  return (
    <div
      className={`loading-skeleton ${className}`}
      style={baseStyles}
      role="status"
      aria-live="polite"
      aria-busy="true"
      aria-label={message}
    >
      <style>{pulseKeyframes}</style>

      {/* Animated background grid */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          backgroundImage: `
            linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px)
          `,
          backgroundSize: "20px 20px",
          animation: "pulse 2s ease-in-out infinite",
        }}
      />

      {/* 3D cube placeholder */}
      {variant !== "minimal" && (
        <div
          style={{
            width: "60px",
            height: "60px",
            border: "3px solid rgba(100, 149, 237, 0.5)",
            borderRadius: "8px",
            animation: "spin 3s linear infinite, pulse 2s ease-in-out infinite",
            marginBottom: "20px",
            background: "rgba(100, 149, 237, 0.1)",
          }}
          aria-hidden="true"
        />
      )}

      {/* Loading message */}
      <div
        style={{
          fontSize: "14px",
          fontWeight: 500,
          color: "#94a3b8",
          marginBottom: showProgress ? "12px" : "0",
          animation: "pulse 1.5s ease-in-out infinite",
        }}
      >
        {message}
      </div>

      {/* Progress bar */}
      {showProgress && (
        <div
          style={{
            width: "200px",
            height: "4px",
            background: "rgba(255,255,255,0.1)",
            borderRadius: "2px",
            overflow: "hidden",
            marginTop: "8px",
          }}
          role="progressbar"
          aria-valuenow={progress}
          aria-valuemin={0}
          aria-valuemax={100}
        >
          <div
            style={{
              width: `${progress}%`,
              height: "100%",
              background: "linear-gradient(90deg, #6366f1, #8b5cf6)",
              borderRadius: "2px",
              transition: "width 0.3s ease-out",
            }}
          />
        </div>
      )}

      {/* Detailed variant: additional info */}
      {variant === "detailed" && (
        <div
          style={{
            marginTop: "20px",
            fontSize: "12px",
            color: "#64748b",
            textAlign: "center",
          }}
        >
          <div>Initializing WebGPU context...</div>
          <div style={{ marginTop: "4px" }}>
            Setting up scene graph and shaders
          </div>
        </div>
      )}

      {/* Shimmer effect overlay */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          background: `linear-gradient(
            90deg,
            transparent 0%,
            rgba(255,255,255,0.02) 50%,
            transparent 100%
          )`,
          backgroundSize: "200% 100%",
          animation: "shimmer 2s ease-in-out infinite",
          pointerEvents: "none",
        }}
        aria-hidden="true"
      />
    </div>
  );
};

LoadingSkeleton.displayName = "LoadingSkeleton";

export default LoadingSkeleton;
