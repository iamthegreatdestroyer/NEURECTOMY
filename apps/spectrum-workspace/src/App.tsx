/**
 * App Component - Legacy Routes Support
 *
 * Note: Primary routing is now handled by routes.tsx with createMemoryRouter.
 * This component is kept for backwards compatibility and development mode.
 */

import { Routes, Route, Outlet } from "react-router-dom";
import { Suspense, lazy } from "react";

import { MainLayout } from "./layouts/MainLayout";
import { LoadingScreen } from "./components/loading-screen";

// Lazy load major routes for code splitting
const IDEView = lazy(() => import("./features/ide/IDEView"));
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

function App() {
  return (
    <Routes>
      {/* IDE View - Full-screen professional IDE interface */}
      <Route
        path="/"
        element={
          <Suspense fallback={<LoadingScreen />}>
            <IDEView />
          </Suspense>
        }
      />

      {/* Legacy routes with MainLayout wrapper */}
      <Route path="/app" element={<MainLayout />}>
        <Route
          index
          element={
            <Suspense fallback={<LoadingScreen />}>
              <Dashboard />
            </Suspense>
          }
        />
        <Route
          path="forge"
          element={
            <Suspense fallback={<LoadingScreen />}>
              <DimensionalForge />
            </Suspense>
          }
        />
        <Route
          path="containers"
          element={
            <Suspense fallback={<LoadingScreen />}>
              <ContainerCommand />
            </Suspense>
          }
        />
        <Route
          path="intelligence"
          element={
            <Suspense fallback={<LoadingScreen />}>
              <IntelligenceFoundry />
            </Suspense>
          }
        />
        <Route
          path="discovery"
          element={
            <Suspense fallback={<LoadingScreen />}>
              <DiscoveryEngine />
            </Suspense>
          }
        />
        <Route
          path="legal"
          element={
            <Suspense fallback={<LoadingScreen />}>
              <LegalFortress />
            </Suspense>
          }
        />
        <Route
          path="agent/:agentId"
          element={
            <Suspense fallback={<LoadingScreen />}>
              <AgentEditor />
            </Suspense>
          }
        />
        <Route
          path="settings"
          element={
            <Suspense fallback={<LoadingScreen />}>
              <Settings />
            </Suspense>
          }
        />
      </Route>
    </Routes>
  );
}

export default App;
