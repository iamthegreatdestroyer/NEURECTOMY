/**
 * @file ErrorBoundary Component
 * @description Error boundary for 3D canvas with graceful fallback UI
 * @module @neurectomy/3d-engine/components
 * @agents @CANVAS @APEX @ECLIPSE
 */

import React, { Component, ErrorInfo, ReactNode } from "react";

export interface ErrorBoundaryProps {
  /** Child components to wrap */
  children: ReactNode;
  /** Custom fallback component */
  fallback?: ReactNode | ((error: Error, reset: () => void) => ReactNode);
  /** Callback when error occurs */
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
  /** Callback when reset is triggered */
  onReset?: () => void;
  /** Show detailed error information (for development) */
  showDetails?: boolean;
  /** Custom className for the fallback container */
  className?: string;
  /** Custom styles for fallback */
  style?: React.CSSProperties;
}

export interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

/**
 * Error boundary component for 3D canvas
 * Catches rendering errors and displays graceful fallback UI
 */
export class ErrorBoundary extends Component<
  ErrorBoundaryProps,
  ErrorBoundaryState
> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
    };
  }

  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    this.setState({ errorInfo });
    this.props.onError?.(error, errorInfo);

    // Log error for debugging
    console.error(
      "3D Engine Error Boundary caught an error:",
      error,
      errorInfo
    );
  }

  handleReset = (): void => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
    });
    this.props.onReset?.();
  };

  render(): ReactNode {
    const { hasError, error, errorInfo } = this.state;
    const { children, fallback, showDetails, className, style } = this.props;

    if (hasError && error) {
      // Use custom fallback if provided
      if (fallback) {
        if (typeof fallback === "function") {
          return fallback(error, this.handleReset);
        }
        return fallback;
      }

      // Default fallback UI
      return (
        <DefaultErrorFallback
          error={error}
          errorInfo={errorInfo}
          onReset={this.handleReset}
          showDetails={showDetails}
          className={className}
          style={style}
        />
      );
    }

    return children;
  }
}

/**
 * Props for the default error fallback component
 */
interface DefaultErrorFallbackProps {
  error: Error;
  errorInfo: ErrorInfo | null;
  onReset: () => void;
  showDetails?: boolean;
  className?: string;
  style?: React.CSSProperties;
}

/**
 * Default error fallback UI component
 */
const DefaultErrorFallback: React.FC<DefaultErrorFallbackProps> = ({
  error,
  errorInfo,
  onReset,
  showDetails = false,
  className = "",
  style,
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
      "linear-gradient(135deg, #1a1a2e 0%, #2d1b3d 50%, #1a1a2e 100%)",
    color: "#e0e0e0",
    fontFamily: "system-ui, -apple-system, sans-serif",
    padding: "24px",
    boxSizing: "border-box",
    ...style,
  };

  const isWebGPUError =
    error.message.toLowerCase().includes("webgpu") ||
    error.message.toLowerCase().includes("gpu") ||
    error.message.toLowerCase().includes("webgl");

  const isContextLost =
    error.message.toLowerCase().includes("context lost") ||
    error.message.toLowerCase().includes("context");

  return (
    <div
      className={`error-boundary-fallback ${className}`}
      style={baseStyles}
      role="alert"
      aria-live="assertive"
    >
      {/* Error icon */}
      <div
        style={{
          width: "64px",
          height: "64px",
          borderRadius: "50%",
          background: "rgba(239, 68, 68, 0.1)",
          border: "2px solid rgba(239, 68, 68, 0.5)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          marginBottom: "20px",
        }}
        aria-hidden="true"
      >
        <svg
          width="32"
          height="32"
          viewBox="0 0 24 24"
          fill="none"
          stroke="#ef4444"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <circle cx="12" cy="12" r="10" />
          <line x1="12" y1="8" x2="12" y2="12" />
          <line x1="12" y1="16" x2="12.01" y2="16" />
        </svg>
      </div>

      {/* Error title */}
      <h2
        style={{
          fontSize: "18px",
          fontWeight: 600,
          color: "#f8fafc",
          margin: "0 0 8px 0",
        }}
      >
        {isWebGPUError
          ? "Graphics Error"
          : isContextLost
            ? "3D Context Lost"
            : "Something went wrong"}
      </h2>

      {/* Error message */}
      <p
        style={{
          fontSize: "14px",
          color: "#94a3b8",
          margin: "0 0 20px 0",
          textAlign: "center",
          maxWidth: "400px",
        }}
      >
        {isWebGPUError
          ? "Your browser or device may not support WebGPU/WebGL. Try updating your browser or graphics drivers."
          : isContextLost
            ? "The 3D rendering context was lost. This can happen due to GPU resource limits or driver issues."
            : "An unexpected error occurred while rendering the 3D scene."}
      </p>

      {/* Action buttons */}
      <div style={{ display: "flex", gap: "12px" }}>
        <button
          onClick={onReset}
          style={{
            padding: "10px 20px",
            fontSize: "14px",
            fontWeight: 500,
            color: "#fff",
            background: "linear-gradient(135deg, #6366f1, #8b5cf6)",
            border: "none",
            borderRadius: "8px",
            cursor: "pointer",
            transition: "transform 0.15s, box-shadow 0.15s",
          }}
          onMouseOver={(e) => {
            e.currentTarget.style.transform = "translateY(-1px)";
            e.currentTarget.style.boxShadow =
              "0 4px 12px rgba(99, 102, 241, 0.3)";
          }}
          onMouseOut={(e) => {
            e.currentTarget.style.transform = "translateY(0)";
            e.currentTarget.style.boxShadow = "none";
          }}
          aria-label="Try again to reload the 3D scene"
        >
          Try Again
        </button>

        {showDetails && (
          <button
            onClick={() => console.log(error, errorInfo)}
            style={{
              padding: "10px 20px",
              fontSize: "14px",
              fontWeight: 500,
              color: "#94a3b8",
              background: "rgba(255, 255, 255, 0.05)",
              border: "1px solid rgba(255, 255, 255, 0.1)",
              borderRadius: "8px",
              cursor: "pointer",
              transition: "background 0.15s",
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.background = "rgba(255, 255, 255, 0.1)";
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.background = "rgba(255, 255, 255, 0.05)";
            }}
            aria-label="Log error details to console"
          >
            Log Details
          </button>
        )}
      </div>

      {/* Technical details (for development) */}
      {showDetails && (
        <details
          style={{
            marginTop: "24px",
            width: "100%",
            maxWidth: "500px",
          }}
        >
          <summary
            style={{
              fontSize: "12px",
              color: "#64748b",
              cursor: "pointer",
              padding: "8px",
            }}
          >
            Technical Details
          </summary>
          <div
            style={{
              marginTop: "8px",
              padding: "12px",
              background: "rgba(0, 0, 0, 0.3)",
              borderRadius: "8px",
              fontSize: "11px",
              fontFamily: "monospace",
              color: "#ef4444",
              overflow: "auto",
              maxHeight: "200px",
            }}
          >
            <div style={{ marginBottom: "8px" }}>
              <strong>Error:</strong> {error.message}
            </div>
            {error.stack && (
              <pre
                style={{
                  margin: 0,
                  whiteSpace: "pre-wrap",
                  wordBreak: "break-all",
                  color: "#94a3b8",
                }}
              >
                {error.stack}
              </pre>
            )}
          </div>
        </details>
      )}

      {/* Helpful links */}
      <div
        style={{
          marginTop: "24px",
          fontSize: "12px",
          color: "#64748b",
        }}
      >
        <span>Need help? </span>
        <a
          href="https://get.webgl.org/"
          target="_blank"
          rel="noopener noreferrer"
          style={{ color: "#6366f1", textDecoration: "none" }}
        >
          Check WebGL support
        </a>
      </div>
    </div>
  );
};

ErrorBoundary.displayName = "ErrorBoundary";

export default ErrorBoundary;
