import React from "react";

interface ErrorFallbackProps {
  error: Error;
  resetErrorBoundary: () => void;
}

export const ErrorFallback: React.FC<ErrorFallbackProps> = ({
  error,
  resetErrorBoundary,
}) => {
  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-4">
      <div className="max-w-md w-full bg-card border border-border rounded-lg p-6 shadow-lg">
        <div className="flex items-center mb-4">
          <div className="w-8 h-8 bg-destructive/10 rounded-full flex items-center justify-center mr-3">
            <svg
              className="w-4 h-4 text-destructive"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z"
              />
            </svg>
          </div>
          <h2 className="text-lg font-semibold text-foreground">
            Something went wrong
          </h2>
        </div>

        <p className="text-sm text-muted-foreground mb-4">
          NEURECTOMY encountered an unexpected error. You can try refreshing the
          application or contact support if the problem persists.
        </p>

        <details className="mb-4">
          <summary className="text-sm font-medium text-foreground cursor-pointer hover:text-primary">
            Error Details
          </summary>
          <pre className="mt-2 p-2 bg-muted rounded text-xs text-muted-foreground overflow-auto max-h-32">
            {error.message}
            {error.stack && `\n\n${error.stack}`}
          </pre>
        </details>

        <div className="flex gap-2">
          <button
            onClick={resetErrorBoundary}
            className="flex-1 bg-primary text-primary-foreground hover:bg-primary/90 px-4 py-2 rounded-md text-sm font-medium transition-colors"
          >
            Try Again
          </button>
          <button
            onClick={() => window.location.reload()}
            className="flex-1 bg-secondary text-secondary-foreground hover:bg-secondary/90 px-4 py-2 rounded-md text-sm font-medium transition-colors"
          >
            Reload App
          </button>
        </div>
      </div>
    </div>
  );
};
