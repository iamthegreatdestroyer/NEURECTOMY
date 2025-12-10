import React from "react";
import ReactDOM from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ReactQueryDevtools } from "@tanstack/react-query-devtools";
import { RouterProvider, createMemoryRouter } from "react-router-dom";
import { ErrorBoundary } from "react-error-boundary";
import { Toaster } from "react-hot-toast";

import { NotificationProvider } from "./components/shell/NotificationToast";
import { ErrorFallback } from "./components/shell/ErrorFallback";
import "./styles/globals.css";

// Import routes configuration
import { routes } from "./routes";

// Log startup for debugging
console.log("[NEURECTOMY] Starting application...");
console.log("[NEURECTOMY] Environment:", import.meta.env.MODE);

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 5, // 5 minutes
      retry: 3,
      refetchOnWindowFocus: false,
    },
  },
});

// Create memory router for desktop app compatibility (like Continue IDE)
const router = createMemoryRouter(routes, {
  initialEntries: ["/"],
  initialIndex: 0,
});

const container = document.getElementById("root");

if (!container) {
  console.error("[NEURECTOMY] FATAL: Root element not found!");
  document.body.innerHTML =
    '<div style="color: red; padding: 20px;">Error: Root element not found</div>';
} else {
  console.log("[NEURECTOMY] Root element found, mounting React...");

  try {
    ReactDOM.createRoot(container).render(
      <React.StrictMode>
        <ErrorBoundary
          FallbackComponent={ErrorFallback}
          onError={(error, info) => {
            console.error("[NEURECTOMY] React Error:", error);
            console.error("[NEURECTOMY] Component Stack:", info.componentStack);
          }}
        >
          <QueryClientProvider client={queryClient}>
            <NotificationProvider>
              <RouterProvider router={router} />
              <Toaster
                position="bottom-right"
                toastOptions={{
                  className: "bg-card text-foreground border border-border",
                  duration: 4000,
                }}
              />
            </NotificationProvider>
            {import.meta.env.DEV && (
              <ReactQueryDevtools initialIsOpen={false} />
            )}
          </QueryClientProvider>
        </ErrorBoundary>
      </React.StrictMode>
    );
    console.log("[NEURECTOMY] React mounted successfully");
  } catch (error) {
    console.error("[NEURECTOMY] Failed to mount React:", error);
    container.innerHTML = `<div style="color: red; padding: 20px;">Failed to mount: ${error}</div>`;
  }
}
