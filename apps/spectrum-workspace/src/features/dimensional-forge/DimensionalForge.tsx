/**
 * Dimensional Forge Feature
 *
 * 3D/4D Agent visualization and orchestration workspace.
 * Provides interactive visualization of agent workflows and relationships.
 */

import { Suspense, useRef, useState, useCallback } from "react";
import { Canvas } from "@react-three/fiber";
import {
  OrbitControls,
  Grid,
  Environment,
  PerspectiveCamera,
  Stars,
} from "@react-three/drei";
import { Vector3 } from "three";
import {
  Play,
  Pause,
  SkipBack,
  SkipForward,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Box,
  Plus,
} from "lucide-react";

import { LoadingScreen } from "@/components/loading-screen";
import { AgentNodeMesh } from "@/components/3d/AgentNodeMesh";
import { ConnectionLine } from "@/components/3d/ConnectionLine";
import { AgentContextMenu } from "@/components/3d/AgentContextMenu";
import { useAgentStore } from "@/stores/agent-store";
import type { AgentNode } from "@/stores/agent-store";

export default function DimensionalForge() {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [totalTime] = useState(60);
  const [contextMenu, setContextMenu] = useState<{
    node: AgentNode;
    position: { x: number; y: number };
  } | null>(null);

  // Get workflows and selection state from store
  const {
    workflows,
    selectedNodeIds,
    selectNode,
    updateNode,
    addNode,
    deleteNode,
    startWorkflow,
    pauseWorkflow,
    stopWorkflow,
  } = useAgentStore();

  // Get active workflow (first running or first in list)
  const activeWorkflow =
    workflows.find((w) => w.status === "running") || workflows[0];

  // Handle node click
  const handleNodeClick = useCallback(
    (nodeId: string, event: MouseEvent) => {
      event.stopPropagation();

      // Toggle selection with Ctrl/Cmd key
      if (event.ctrlKey || event.metaKey) {
        selectNode(nodeId, !selectedNodeIds.includes(nodeId));
      } else {
        // Single selection
        selectNode(nodeId, true);
      }
    },
    [selectNode, selectedNodeIds]
  );

  // Handle node position change
  const handlePositionChange = useCallback(
    (nodeId: string, position: { x: number; y: number; z: number }) => {
      if (!activeWorkflow) return;
      updateNode(activeWorkflow.id, nodeId, { position });
    },
    [activeWorkflow, updateNode]
  );

  // Handle background click (deselect all)
  const handleBackgroundClick = useCallback(() => {
    selectedNodeIds.forEach((nodeId) => selectNode(nodeId, false));
  }, [selectedNodeIds, selectNode]);

  // Handle play/pause
  const handlePlayPause = useCallback(() => {
    if (!activeWorkflow) return;

    if (isPlaying) {
      pauseWorkflow(activeWorkflow.id);
    } else {
      startWorkflow(activeWorkflow.id);
    }
    setIsPlaying(!isPlaying);
  }, [activeWorkflow, isPlaying, startWorkflow, pauseWorkflow]);

  // Handle context menu
  const handleContextMenu = useCallback(
    (nodeId: string, event: MouseEvent) => {
      event.preventDefault();
      event.stopPropagation();

      if (!activeWorkflow) return;

      const node = activeWorkflow.nodes.find((n) => n.id === nodeId);
      if (!node) return;

      // Get screen coordinates
      const x = (event as any).clientX || event.pageX;
      const y = (event as any).clientY || event.pageY;

      setContextMenu({ node, position: { x, y } });
    },
    [activeWorkflow]
  );

  // Context menu action handlers
  const handleRunNode = useCallback(
    (nodeId: string) => {
      if (!activeWorkflow) return;
      updateNode(activeWorkflow.id, nodeId, { status: "running" });
      if (activeWorkflow.status === "idle") {
        startWorkflow(activeWorkflow.id);
      }
      setContextMenu(null);
    },
    [activeWorkflow, updateNode, startWorkflow]
  );

  const handlePauseNode = useCallback(
    (nodeId: string) => {
      if (!activeWorkflow) return;
      updateNode(activeWorkflow.id, nodeId, { status: "paused" });
      pauseWorkflow(activeWorkflow.id);
      setContextMenu(null);
    },
    [activeWorkflow, updateNode, pauseWorkflow]
  );

  const handleStopNode = useCallback(
    (nodeId: string) => {
      if (!activeWorkflow) return;
      updateNode(activeWorkflow.id, nodeId, { status: "idle" });
      stopWorkflow(activeWorkflow.id);
      setContextMenu(null);
    },
    [activeWorkflow, updateNode, stopWorkflow]
  );

  const handleDeleteNode = useCallback(
    (nodeId: string) => {
      if (!activeWorkflow) return;
      deleteNode(activeWorkflow.id, nodeId);
      setContextMenu(null);
    },
    [activeWorkflow, deleteNode]
  );

  const handleConfigureNode = useCallback((nodeId: string) => {
    // TODO: Open configuration modal/panel
    console.log("Configure node:", nodeId);
    setContextMenu(null);
  }, []);

  const handleDuplicateNode = useCallback(
    (nodeId: string) => {
      if (!activeWorkflow) return;

      const node = activeWorkflow.nodes.find((n) => n.id === nodeId);
      if (!node) return;

      // Clone node with offset position and new ID
      const clonedNode: Omit<AgentNode, "createdAt" | "updatedAt"> = {
        id: `${node.id}_copy_${Date.now()}`,
        name: `${node.name} (Copy)`,
        codename: node.codename ? `${node.codename}_copy` : undefined,
        description: node.description,
        type: node.type,
        position: {
          x: node.position.x + 2,
          y: node.position.y + 2,
          z: node.position.z,
        },
        status: "idle",
        color: node.color,
        scale: node.scale,
        visible: node.visible,
        config: { ...node.config },
        metadata: node.metadata ? { ...node.metadata } : undefined,
        metrics: { executionTime: 0, tokensUsed: 0, cost: 0, successRate: 0 },
        inputs: [],
        outputs: [],
        lastExecutedAt: undefined,
      };

      addNode(activeWorkflow.id, clonedNode);
      setContextMenu(null);
    },
    [activeWorkflow, addNode]
  );

  const handleViewConnections = useCallback(
    (nodeId: string) => {
      if (!activeWorkflow) return;

      // Find all connected nodes
      const connectedNodeIds = activeWorkflow.connections
        .filter((conn) => conn.sourceId === nodeId || conn.targetId === nodeId)
        .flatMap((conn) => [conn.sourceId, conn.targetId])
        .filter((id) => id !== nodeId);

      // Select the node and all connected nodes
      selectNode(nodeId, true);
      connectedNodeIds.forEach((id) => selectNode(id, true));

      setContextMenu(null);
    },
    [activeWorkflow, selectNode]
  );

  const handleToggleVisibility = useCallback(
    (nodeId: string) => {
      if (!activeWorkflow) return;
      const node = activeWorkflow.nodes.find((n) => n.id === nodeId);
      if (!node) return;
      updateNode(activeWorkflow.id, nodeId, { visible: !node.visible });
      setContextMenu(null);
    },
    [activeWorkflow, updateNode]
  );

  const handleResetNode = useCallback(
    (nodeId: string) => {
      if (!activeWorkflow) return;
      updateNode(activeWorkflow.id, nodeId, {
        status: "idle",
        metrics: { executionTime: 0, tokensUsed: 0, cost: 0, successRate: 0 },
      });
      setContextMenu(null);
    },
    [activeWorkflow, updateNode]
  );

  const handleShowInfo = useCallback(
    (nodeId: string) => {
      // Select node and scroll to inspector panel
      selectNode(nodeId, true);
      setContextMenu(null);
    },
    [selectNode]
  );

  const handleRenameNode = useCallback((nodeId: string) => {
    // TODO: Show inline rename input or modal
    console.log("Rename node:", nodeId);
    setContextMenu(null);
  }, []);

  return (
    <div className="h-full flex flex-col">
      {/* Toolbar */}
      <div className="h-12 px-4 flex items-center justify-between border-b border-border bg-card">
        <div className="flex items-center gap-2">
          <button className="p-2 hover:bg-muted rounded-lg transition-colors">
            <Plus className="w-4 h-4" />
          </button>
          <button className="p-2 hover:bg-muted rounded-lg transition-colors">
            <Box className="w-4 h-4" />
          </button>
          <div className="w-px h-6 bg-border mx-2" />
          <button className="p-2 hover:bg-muted rounded-lg transition-colors">
            <ZoomIn className="w-4 h-4" />
          </button>
          <button className="p-2 hover:bg-muted rounded-lg transition-colors">
            <ZoomOut className="w-4 h-4" />
          </button>
          <button className="p-2 hover:bg-muted rounded-lg transition-colors">
            <RotateCcw className="w-4 h-4" />
          </button>
        </div>
        <div className="text-sm font-medium">
          Dimensional Forge - 3D Workflow Editor
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground">View:</span>
          <select className="bg-muted px-2 py-1 rounded text-sm">
            <option>3D Perspective</option>
            <option>Top-Down</option>
            <option>Front View</option>
            <option>Side View</option>
          </select>
        </div>
      </div>

      {/* Main Canvas Area */}
      <div className="flex-1 relative">
        <Suspense fallback={<LoadingScreen />}>
          <Canvas shadows>
            <PerspectiveCamera makeDefault position={[10, 10, 10]} />
            <OrbitControls
              enableDamping
              dampingFactor={0.05}
              minDistance={5}
              maxDistance={50}
            />

            {/* Lighting */}
            <ambientLight intensity={0.4} />
            <directionalLight
              position={[10, 10, 5]}
              intensity={0.8}
              castShadow
              shadow-mapSize={[2048, 2048]}
            />

            {/* Grid */}
            <Grid
              args={[100, 100]}
              cellSize={1}
              cellThickness={0.5}
              cellColor="#1e293b"
              sectionSize={5}
              sectionThickness={1}
              sectionColor="#334155"
              fadeDistance={50}
              fadeStrength={1}
              followCamera={false}
              infiniteGrid
            />

            {/* Stars background */}
            <Stars
              radius={100}
              depth={50}
              count={5000}
              factor={4}
              saturation={0}
              fade
              speed={1}
            />

            {/* Scene content with click handler */}
            <group onClick={handleBackgroundClick}>
              {/* Render agent nodes from active workflow */}
              {activeWorkflow?.nodes.map((node) => (
                <AgentNodeMesh
                  key={node.id}
                  node={node}
                  isSelected={selectedNodeIds.includes(node.id)}
                  onClick={(event) => handleNodeClick(node.id, event)}
                  onContextMenu={(event) => handleContextMenu(node.id, event)}
                  onPositionChange={(position) =>
                    handlePositionChange(node.id, position)
                  }
                />
              ))}

              {/* Render connections between nodes */}
              {activeWorkflow?.connections.map((connection) => {
                const sourceNode = activeWorkflow.nodes.find(
                  (n) => n.id === connection.sourceId
                );
                const targetNode = activeWorkflow.nodes.find(
                  (n) => n.id === connection.targetId
                );

                if (!sourceNode || !targetNode) return null;

                const startPosition = new Vector3(
                  sourceNode.position.x,
                  sourceNode.position.y,
                  sourceNode.position.z
                );

                const endPosition = new Vector3(
                  targetNode.position.x,
                  targetNode.position.y,
                  targetNode.position.z
                );

                const isSelected =
                  selectedNodeIds.includes(connection.sourceId) ||
                  selectedNodeIds.includes(connection.targetId);

                return (
                  <ConnectionLine
                    key={`${connection.sourceId}-${connection.targetId}`}
                    connection={connection}
                    startPosition={startPosition}
                    endPosition={endPosition}
                    isSelected={isSelected}
                  />
                );
              })}
            </group>

            {/* Environment */}
            <Environment preset="night" />
          </Canvas>
        </Suspense>

        {/* Workflow Info Panel */}
        {activeWorkflow && (
          <div className="absolute top-4 left-4 bg-gray-900/80 backdrop-blur-sm rounded-lg p-4 border border-gray-700">
            <div className="text-sm text-gray-400 mb-1">Active Workflow</div>
            <div className="text-lg font-semibold text-white">
              {activeWorkflow.name}
            </div>
            <div className="flex gap-4 mt-2 text-xs text-gray-400">
              <div>
                <span className="font-medium">
                  {activeWorkflow.nodes.length}
                </span>{" "}
                Nodes
              </div>
              <div>
                <span className="font-medium">
                  {activeWorkflow.connections.length}
                </span>{" "}
                Connections
              </div>
            </div>
            <div className="mt-2">
              <span
                className={`text-xs px-2 py-1 rounded ${
                  activeWorkflow.status === "running"
                    ? "bg-green-500/20 text-green-400"
                    : activeWorkflow.status === "paused"
                      ? "bg-yellow-500/20 text-yellow-400"
                      : activeWorkflow.status === "completed"
                        ? "bg-blue-500/20 text-blue-400"
                        : activeWorkflow.status === "error"
                          ? "bg-red-500/20 text-red-400"
                          : "bg-gray-500/20 text-gray-400"
                }`}
              >
                {activeWorkflow.status.toUpperCase()}
              </span>
            </div>
          </div>
        )}

        {/* Node Inspector Panel */}
        {selectedNodeIds.length > 0 && (
          <div className="absolute top-4 right-4 w-80 panel p-4 max-h-[80vh] overflow-y-auto">
            <h3 className="font-semibold mb-3">
              {selectedNodeIds.length === 1
                ? "Node Inspector"
                : `${selectedNodeIds.length} Nodes Selected`}
            </h3>
            {selectedNodeIds.length === 1 &&
              activeWorkflow &&
              (() => {
                const node = activeWorkflow.nodes.find(
                  (n) => n.id === selectedNodeIds[0]
                );
                if (!node) return null;

                return (
                  <div className="space-y-3">
                    <div>
                      <div className="text-xs text-muted-foreground mb-1">
                        Name
                      </div>
                      <div className="font-medium">{node.name}</div>
                    </div>
                    <div>
                      <div className="text-xs text-muted-foreground mb-1">
                        Type
                      </div>
                      <div className="text-sm">{node.type}</div>
                    </div>
                    <div>
                      <div className="text-xs text-muted-foreground mb-1">
                        Status
                      </div>
                      <div
                        className={`text-sm ${
                          node.status === "running"
                            ? "text-green-400"
                            : node.status === "paused"
                              ? "text-yellow-400"
                              : node.status === "error"
                                ? "text-red-400"
                                : node.status === "completed"
                                  ? "text-blue-400"
                                  : "text-gray-400"
                        }`}
                      >
                        {node.status}
                      </div>
                    </div>
                    {node.metrics && (
                      <div>
                        <div className="text-xs text-muted-foreground mb-2">
                          Metrics
                        </div>
                        <div className="space-y-1 text-sm">
                          <div className="flex justify-between">
                            <span>Execution Time:</span>
                            <span>{node.metrics.executionTime}ms</span>
                          </div>
                          <div className="flex justify-between">
                            <span>Tokens Used:</span>
                            <span>{node.metrics.tokensUsed}</span>
                          </div>
                          <div className="flex justify-between">
                            <span>Cost:</span>
                            <span>${node.metrics.cost.toFixed(4)}</span>
                          </div>
                          <div className="flex justify-between">
                            <span>Success Rate:</span>
                            <span>
                              {(node.metrics.successRate * 100).toFixed(1)}%
                            </span>
                          </div>
                        </div>
                      </div>
                    )}
                    <div>
                      <div className="text-xs text-muted-foreground mb-1">
                        Position
                      </div>
                      <div className="text-xs font-mono">
                        ({node.position.x.toFixed(2)},{" "}
                        {node.position.y.toFixed(2)},{" "}
                        {node.position.z.toFixed(2)})
                      </div>
                    </div>
                  </div>
                );
              })()}
          </div>
        )}

        {/* Controls Guide */}
        <div className="absolute bottom-28 right-4 bg-gray-900/80 backdrop-blur-sm rounded-lg p-3 border border-gray-700">
          <div className="text-xs text-gray-400 space-y-1">
            <div>
              <span className="font-medium text-white">Left Click</span> -
              Select Node
            </div>
            <div>
              <span className="font-medium text-white">Ctrl + Click</span> -
              Multi-Select
            </div>
            <div>
              <span className="font-medium text-white">Mouse Wheel</span> - Zoom
            </div>
            <div>
              <span className="font-medium text-white">Middle Mouse</span> - Pan
            </div>
          </div>
        </div>

        {/* Empty state */}
        {workflows.length === 0 && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center">
              <div className="text-6xl mb-4">🔮</div>
              <h3 className="text-xl font-semibold text-white mb-2">
                No Workflows Yet
              </h3>
              <p className="text-gray-400 mb-4">
                Create your first AI agent workflow to see it in 3D
              </p>
              <button className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg transition-colors">
                Create Workflow
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Timeline */}
      <div className="h-24 border-t border-border bg-card p-4">
        <div className="flex items-center gap-4 mb-2">
          <div className="flex items-center gap-1">
            <button className="p-1.5 hover:bg-muted rounded transition-colors">
              <SkipBack className="w-4 h-4" />
            </button>
            <button
              className="p-1.5 hover:bg-muted rounded transition-colors"
              onClick={handlePlayPause}
              disabled={!activeWorkflow}
            >
              {isPlaying ? (
                <Pause className="w-4 h-4" />
              ) : (
                <Play className="w-4 h-4" />
              )}
            </button>
            <button className="p-1.5 hover:bg-muted rounded transition-colors">
              <SkipForward className="w-4 h-4" />
            </button>
          </div>
          <span className="text-sm font-mono">
            {formatTime(currentTime)} / {formatTime(totalTime)}
          </span>
        </div>

        {/* Timeline Track */}
        <div className="timeline-track">
          <div
            className="timeline-progress"
            style={{ width: `${(currentTime / totalTime) * 100}%` }}
          />
        </div>

        {/* Timeline Ruler */}
        <div className="flex justify-between text-xs text-muted-foreground mt-1">
          <span>0:00</span>
          <span>0:15</span>
          <span>0:30</span>
          <span>0:45</span>
          <span>1:00</span>
        </div>
      </div>

      {/* Context Menu */}
      {contextMenu && (
        <AgentContextMenu
          node={contextMenu.node}
          position={contextMenu.position}
          onClose={() => setContextMenu(null)}
          onRun={handleRunNode}
          onPause={handlePauseNode}
          onStop={handleStopNode}
          onDelete={handleDeleteNode}
          onConfigure={handleConfigureNode}
          onDuplicate={handleDuplicateNode}
          onViewConnections={handleViewConnections}
          onToggleVisibility={handleToggleVisibility}
          onReset={handleResetNode}
          onShowInfo={handleShowInfo}
          onRename={handleRenameNode}
        />
      )}
    </div>
  );
}

function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}
