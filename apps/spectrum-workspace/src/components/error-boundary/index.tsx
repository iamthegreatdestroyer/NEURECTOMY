/**
 * Error Boundary Fallback Component
 */

import { FallbackProps } from 'react-error-boundary';
import { AlertTriangle, RefreshCw } from 'lucide-react';

export function ErrorFallback({ error, resetErrorBoundary }: FallbackProps) {
  return (
    <div className="flex flex-col items-center justify-center h-full p-8 text-center">
      <AlertTriangle className="w-16 h-16 text-destructive mb-4" />
      <h2 className="text-2xl font-bold mb-2">Something went wrong</h2>
      <p className="text-muted-foreground mb-4 max-w-md">
        {error.message || 'An unexpected error occurred'}
      </p>
      <button
        onClick={resetErrorBoundary}
        className="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors"
      >
        <RefreshCw className="w-4 h-4" />
        Try again
      </button>
      <details className="mt-4 text-left max-w-lg">
        <summary className="cursor-pointer text-sm text-muted-foreground">
          Error details
        </summary>
        <pre className="mt-2 p-4 bg-muted rounded-lg text-xs overflow-auto">
          {error.stack}
        </pre>
      </details>
    </div>
  );
}
