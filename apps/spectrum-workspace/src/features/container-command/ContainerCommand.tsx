/**
 * Container Command Feature
 *
 * Unified interface for Docker container and Kubernetes cluster management.
 * Provides Docker lifecycle control, 3D K8s visualization, and real-time resource monitoring.
 */

import { useState } from "react";
import { Container, Boxes, Activity, Settings } from "lucide-react";
import { DockerManager } from "./DockerManager";
import { K8sTopology3D } from "./K8sTopology3D";
import { ResourceMonitor } from "./ResourceMonitor";
import { useContainerStore } from "@/stores/container-store";

type TabType = "docker" | "kubernetes" | "monitor" | "settings";

/**
 * ContainerCommand - Main container orchestration interface
 *
 * Features:
 * - Docker container management (start/stop/restart/remove/logs/shell)
 * - Kubernetes 3D topology visualization
 * - Real-time resource monitoring with charts
 * - Unified tabbed interface
 *
 * @example
 * ```tsx
 * <ContainerCommand />
 * ```
 */
export function ContainerCommand() {
  const [activeTab, setActiveTab] = useState<TabType>("docker");
  const { containers, clusters, activeClusterId } = useContainerStore();

  const tabs = [
    {
      id: "docker" as TabType,
      label: "Docker",
      icon: Container,
      count: containers.length,
    },
    {
      id: "kubernetes" as TabType,
      label: "Kubernetes",
      icon: Boxes,
      count: clusters.length,
    },
    {
      id: "monitor" as TabType,
      label: "Monitor",
      icon: Activity,
      count: null,
    },
    {
      id: "settings" as TabType,
      label: "Settings",
      icon: Settings,
      count: null,
    },
  ];

  const activeCluster = clusters.find((c) => c.id === activeClusterId);
  const runningContainers = containers.filter(
    (c) => c.status === "running"
  ).length;

  return (
    <div className="h-full flex flex-col bg-background">
      {/* Header */}
      <div className="border-b border-border bg-card px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold flex items-center gap-2">
              <Container className="w-6 h-6 text-primary" />
              Container Command
            </h1>
            <p className="text-sm text-muted-foreground mt-1">
              Orchestrate Docker containers and Kubernetes clusters
            </p>
          </div>

          {/* Connection Status */}
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 text-sm">
              <div className="flex items-center gap-1.5">
                <div
                  className={`w-2 h-2 rounded-full ${runningContainers > 0 ? "bg-green-500 animate-pulse" : "bg-gray-500"}`}
                />
                <span className="text-muted-foreground">Docker:</span>
                <span className="font-semibold">
                  {runningContainers} running
                </span>
              </div>
            </div>

            {activeCluster && (
              <div className="flex items-center gap-2 text-sm">
                <div className="flex items-center gap-1.5">
                  <div className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" />
                  <span className="text-muted-foreground">Cluster:</span>
                  <span className="font-semibold">{activeCluster.name}</span>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Tabs */}
        <div className="flex items-center gap-2 mt-4">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`
                  flex items-center gap-2 px-4 py-2 rounded-lg transition-all
                  ${
                    activeTab === tab.id
                      ? "bg-primary text-primary-foreground shadow-sm"
                      : "hover:bg-muted"
                  }
                `}
              >
                <Icon className="w-4 h-4" />
                <span className="font-medium">{tab.label}</span>
                {tab.count !== null && (
                  <span
                    className={`
                      px-2 py-0.5 rounded-full text-xs font-semibold
                      ${
                        activeTab === tab.id
                          ? "bg-primary-foreground/20"
                          : "bg-muted-foreground/20"
                      }
                    `}
                  >
                    {tab.count}
                  </span>
                )}
              </button>
            );
          })}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-hidden">
        {activeTab === "docker" && <DockerManager />}
        {activeTab === "kubernetes" && <K8sTopology3D />}
        {activeTab === "monitor" && <ResourceMonitor />}
        {activeTab === "settings" && <SettingsPanel />}
      </div>
    </div>
  );
}

/**
 * SettingsPanel - Configuration and preferences
 */
function SettingsPanel() {
  return (
    <div className="h-full flex items-center justify-center p-8">
      <div className="text-center max-w-md">
        <Settings className="w-16 h-16 mx-auto text-muted-foreground mb-4" />
        <h2 className="text-xl font-semibold mb-2">Settings</h2>
        <p className="text-muted-foreground mb-4">
          Configure Docker daemon connection, Kubernetes contexts, and
          monitoring preferences.
        </p>
        <button className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors">
          Coming Soon
        </button>
      </div>
    </div>
  );
}
