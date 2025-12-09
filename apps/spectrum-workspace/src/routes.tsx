/**
 * Routes Configuration for NEURECTOMY Desktop IDE
 *
 * Uses createMemoryRouter pattern (like Continue IDE) for desktop app compatibility.
 * Memory router works correctly with file:// protocol and Tauri webview.
 */

import { lazy, Suspense } from "react";
import type { RouteObject } from "react-router-dom";
import { useRouteError } from "react-router-dom";
import { MainLayout } from "./layouts/MainLayout";
import { LoadingScreen } from "./components/loading-screen";
import { ErrorFallback } from "./components/shell/ErrorFallback";

// Lazy load major routes for code splitting
// Using V2 IDE with new professional shell components
const IDEView = lazy(() => import("./features/ide/IDEViewV2"));
const IDEViewLegacy = lazy(() => import("./features/ide/IDEView"));
const Dashboard = lazy(() => import("./features/dashboard/Dashboard"));
const DimensionalForge = lazy(
  () => import("./features/dimensional-forge/DimensionalForge")
);
const ContainerCommand = lazy(
  () => import("./features/container-command/ContainerCommand")
);
const IntelligenceFoundry = lazy(
  () => import("./features/intelligence-foundry/IntelligenceFoundry")
);
const DiscoveryEngine = lazy(
  () => import("./features/discovery-engine/DiscoveryEngine")
);
const LegalFortress = lazy(
  () => import("./features/legal-fortress/LegalFortress")
);
const AgentEditor = lazy(() => import("./features/agent-editor/AgentEditor"));
const Settings = lazy(() => import("./features/settings/Settings"));

// Wrapper component for lazy-loaded routes with proper error handling
function LazyRoute({
  component: Component,
  loadingMessage = "Loading...",
}: {
  component: React.LazyExoticComponent<React.ComponentType<any>>;
  loadingMessage?: string;
}) {
  console.log(
    "[LazyRoute] Rendering with Suspense, loadingMessage:",
    loadingMessage
  );
  return (
    <Suspense
      fallback={
        <div
          style={{
            color: "white",
            background: "red",
            padding: "20px",
            fontSize: "24px",
            height: "100vh",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          DEBUG: Loading - {loadingMessage}
        </div>
      }
    >
      <Component />
    </Suspense>
  );
}

// Error element for route errors
function RouteError() {
  // In React Router v6, errorElement components receive error via useRouteError hook
  const error = useRouteError() as Error;

  return (
    <ErrorFallback
      error={error || new Error("Failed to load this page")}
      resetErrorBoundary={() => window.location.reload()}
    />
  );
}

/**
 * Route configuration for memory router
 * Following Continue IDE pattern for desktop compatibility
 */
export const routes: RouteObject[] = [
  {
    path: "/",
    element: (
      <LazyRoute
        component={IDEViewLegacy}
        loadingMessage="Initializing IDE..."
      />
    ),
    errorElement: <RouteError />,
  },
  {
    path: "/ide",
    element: <LazyRoute component={IDEView} loadingMessage="Loading IDE..." />,
    errorElement: <RouteError />,
  },
  {
    path: "/ide-legacy",
    element: (
      <LazyRoute
        component={IDEViewLegacy}
        loadingMessage="Loading Legacy IDE..."
      />
    ),
    errorElement: <RouteError />,
  },
  {
    path: "/app",
    element: <MainLayout />,
    errorElement: <RouteError />,
    children: [
      {
        index: true,
        element: (
          <LazyRoute
            component={Dashboard}
            loadingMessage="Loading Dashboard..."
          />
        ),
      },
      {
        path: "forge",
        element: (
          <LazyRoute
            component={DimensionalForge}
            loadingMessage="Loading Dimensional Forge..."
          />
        ),
      },
      {
        path: "containers",
        element: (
          <LazyRoute
            component={ContainerCommand}
            loadingMessage="Loading Container Command..."
          />
        ),
      },
      {
        path: "intelligence",
        element: (
          <LazyRoute
            component={IntelligenceFoundry}
            loadingMessage="Loading Intelligence Foundry..."
          />
        ),
      },
      {
        path: "discovery",
        element: (
          <LazyRoute
            component={DiscoveryEngine}
            loadingMessage="Loading Discovery Engine..."
          />
        ),
      },
      {
        path: "legal",
        element: (
          <LazyRoute
            component={LegalFortress}
            loadingMessage="Loading Legal Fortress..."
          />
        ),
      },
      {
        path: "agent/:agentId",
        element: (
          <LazyRoute
            component={AgentEditor}
            loadingMessage="Loading Agent Editor..."
          />
        ),
      },
      {
        path: "settings",
        element: (
          <LazyRoute
            component={Settings}
            loadingMessage="Loading Settings..."
          />
        ),
      },
    ],
  },
];
