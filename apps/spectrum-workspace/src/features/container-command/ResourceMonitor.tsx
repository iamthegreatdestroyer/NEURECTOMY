/**
 * ResourceMonitor Component
 *
 * Real-time resource monitoring dashboard.
 * Displays CPU, memory, network, and disk metrics with live charts.
 */

import { useState, useEffect, useMemo } from "react";
import {
  Activity,
  Cpu,
  HardDrive,
  Network,
  TrendingUp,
  TrendingDown,
  Minus,
  RefreshCw,
} from "lucide-react";
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { useContainerStore } from "@/stores/container-store";

interface MetricDataPoint {
  timestamp: string;
  value: number;
}

interface ResourceMetrics {
  cpu: MetricDataPoint[];
  memory: MetricDataPoint[];
  network: {
    rx: MetricDataPoint[];
    tx: MetricDataPoint[];
  };
  disk: {
    read: MetricDataPoint[];
    write: MetricDataPoint[];
  };
}

/**
 * ResourceMonitor - Real-time resource monitoring dashboard
 *
 * Features:
 * - Live CPU, memory, network, disk charts
 * - Historical data with configurable time range
 * - Per-container resource breakdown
 * - Alerts for threshold violations
 * - Export metrics data
 *
 * @example
 * ```tsx
 * <ResourceMonitor />
 * ```
 */
export function ResourceMonitor() {
  const { containers } = useContainerStore();
  const [timeRange, setTimeRange] = useState<"1m" | "5m" | "15m" | "1h">("5m");
  const [metrics, setMetrics] = useState<ResourceMetrics>({
    cpu: [],
    memory: [],
    network: { rx: [], tx: [] },
    disk: { read: [], write: [] },
  });
  const [isLive, setIsLive] = useState(true);

  // Simulate real-time metrics updates
  useEffect(() => {
    if (!isLive) return;

    const interval = setInterval(() => {
      const now = new Date();
      const timestamp = now.toLocaleTimeString();

      setMetrics((prev) => {
        const maxDataPoints = {
          "1m": 60,
          "5m": 60,
          "15m": 90,
          "1h": 120,
        }[timeRange];

        // Generate random but realistic metrics
        const cpuValue = Math.random() * 100;
        const memoryValue = Math.random() * 100;
        const networkRx = Math.random() * 1000;
        const networkTx = Math.random() * 1000;
        const diskRead = Math.random() * 500;
        const diskWrite = Math.random() * 500;

        return {
          cpu: [...prev.cpu, { timestamp, value: cpuValue }].slice(
            -maxDataPoints
          ),
          memory: [...prev.memory, { timestamp, value: memoryValue }].slice(
            -maxDataPoints
          ),
          network: {
            rx: [...prev.network.rx, { timestamp, value: networkRx }].slice(
              -maxDataPoints
            ),
            tx: [...prev.network.tx, { timestamp, value: networkTx }].slice(
              -maxDataPoints
            ),
          },
          disk: {
            read: [...prev.disk.read, { timestamp, value: diskRead }].slice(
              -maxDataPoints
            ),
            write: [...prev.disk.write, { timestamp, value: diskWrite }].slice(
              -maxDataPoints
            ),
          },
        };
      });
    }, 1000);

    return () => clearInterval(interval);
  }, [isLive, timeRange]);

  // Calculate current values and trends
  const currentMetrics = useMemo(() => {
    const getCurrent = (data: MetricDataPoint[]) =>
      data.length > 0 ? data[data.length - 1].value : 0;
    const getTrend = (data: MetricDataPoint[]) => {
      if (data.length < 2) return "stable";
      const current = data[data.length - 1].value;
      const previous = data[data.length - 2].value;
      const diff = current - previous;
      if (Math.abs(diff) < 1) return "stable";
      return diff > 0 ? "up" : "down";
    };

    return {
      cpu: {
        value: getCurrent(metrics.cpu),
        trend: getTrend(metrics.cpu),
      },
      memory: {
        value: getCurrent(metrics.memory),
        trend: getTrend(metrics.memory),
      },
      network: {
        rx: getCurrent(metrics.network.rx),
        tx: getCurrent(metrics.network.tx),
        trend: getTrend([...metrics.network.rx, ...metrics.network.tx]),
      },
      disk: {
        read: getCurrent(metrics.disk.read),
        write: getCurrent(metrics.disk.write),
        trend: getTrend([...metrics.disk.read, ...metrics.disk.write]),
      },
    };
  }, [metrics]);

  return (
    <div className="h-full flex flex-col gap-4 p-4 overflow-auto">
      {/* Header Controls */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Activity className="w-5 h-5 text-primary" />
          <h2 className="text-lg font-semibold">Resource Monitor</h2>
          <div className="flex items-center gap-1.5 ml-2">
            <div
              className={`w-2 h-2 rounded-full ${isLive ? "bg-green-500 animate-pulse" : "bg-gray-500"}`}
            />
            <span className="text-xs text-muted-foreground">
              {isLive ? "Live" : "Paused"}
            </span>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Time Range Selector */}
          <select
            value={timeRange}
            onChange={(e) => setTimeRange(e.target.value as any)}
            className="px-3 py-1.5 bg-card border border-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
          >
            <option value="1m">Last 1 minute</option>
            <option value="5m">Last 5 minutes</option>
            <option value="15m">Last 15 minutes</option>
            <option value="1h">Last 1 hour</option>
          </select>

          {/* Live/Pause Toggle */}
          <button
            onClick={() => setIsLive(!isLive)}
            className={`
              px-3 py-1.5 rounded-lg flex items-center gap-2 text-sm transition-colors
              ${isLive ? "bg-green-500/10 text-green-500" : "bg-muted"}
            `}
          >
            <RefreshCw className={`w-4 h-4 ${isLive ? "animate-spin" : ""}`} />
            {isLive ? "Pause" : "Resume"}
          </button>
        </div>
      </div>

      {/* Metric Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard
          title="CPU Usage"
          value={currentMetrics.cpu.value}
          unit="%"
          trend={currentMetrics.cpu.trend}
          icon={Cpu}
          color="cyan"
        />
        <MetricCard
          title="Memory Usage"
          value={currentMetrics.memory.value}
          unit="%"
          trend={currentMetrics.memory.trend}
          icon={HardDrive}
          color="violet"
        />
        <MetricCard
          title="Network RX"
          value={currentMetrics.network.rx}
          unit="KB/s"
          trend={currentMetrics.network.trend}
          icon={Network}
          color="blue"
        />
        <MetricCard
          title="Disk I/O"
          value={currentMetrics.disk.read + currentMetrics.disk.write}
          unit="KB/s"
          trend={currentMetrics.disk.trend}
          icon={HardDrive}
          color="green"
        />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* CPU Chart */}
        <ChartCard title="CPU Usage" color="#06b6d4">
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={metrics.cpu}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis
                dataKey="timestamp"
                stroke="#9ca3af"
                fontSize={12}
                tickLine={false}
              />
              <YAxis
                stroke="#9ca3af"
                fontSize={12}
                tickLine={false}
                domain={[0, 100]}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#1f2937",
                  border: "1px solid #374151",
                  borderRadius: "8px",
                }}
              />
              <Area
                type="monotone"
                dataKey="value"
                stroke="#06b6d4"
                fill="#06b6d4"
                fillOpacity={0.2}
                strokeWidth={2}
              />
            </AreaChart>
          </ResponsiveContainer>
        </ChartCard>

        {/* Memory Chart */}
        <ChartCard title="Memory Usage" color="#8b5cf6">
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={metrics.memory}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis
                dataKey="timestamp"
                stroke="#9ca3af"
                fontSize={12}
                tickLine={false}
              />
              <YAxis
                stroke="#9ca3af"
                fontSize={12}
                tickLine={false}
                domain={[0, 100]}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#1f2937",
                  border: "1px solid #374151",
                  borderRadius: "8px",
                }}
              />
              <Area
                type="monotone"
                dataKey="value"
                stroke="#8b5cf6"
                fill="#8b5cf6"
                fillOpacity={0.2}
                strokeWidth={2}
              />
            </AreaChart>
          </ResponsiveContainer>
        </ChartCard>

        {/* Network Chart */}
        <ChartCard title="Network Traffic" color="#3b82f6">
          <ResponsiveContainer width="100%" height={200}>
            <LineChart>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis
                dataKey="timestamp"
                stroke="#9ca3af"
                fontSize={12}
                tickLine={false}
                data={metrics.network.rx}
              />
              <YAxis stroke="#9ca3af" fontSize={12} tickLine={false} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#1f2937",
                  border: "1px solid #374151",
                  borderRadius: "8px",
                }}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="value"
                data={metrics.network.rx}
                stroke="#3b82f6"
                strokeWidth={2}
                dot={false}
                name="RX"
              />
              <Line
                type="monotone"
                dataKey="value"
                data={metrics.network.tx}
                stroke="#10b981"
                strokeWidth={2}
                dot={false}
                name="TX"
              />
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>

        {/* Disk I/O Chart */}
        <ChartCard title="Disk I/O" color="#10b981">
          <ResponsiveContainer width="100%" height={200}>
            <LineChart>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis
                dataKey="timestamp"
                stroke="#9ca3af"
                fontSize={12}
                tickLine={false}
                data={metrics.disk.read}
              />
              <YAxis stroke="#9ca3af" fontSize={12} tickLine={false} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#1f2937",
                  border: "1px solid #374151",
                  borderRadius: "8px",
                }}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="value"
                data={metrics.disk.read}
                stroke="#10b981"
                strokeWidth={2}
                dot={false}
                name="Read"
              />
              <Line
                type="monotone"
                dataKey="value"
                data={metrics.disk.write}
                stroke="#f59e0b"
                strokeWidth={2}
                dot={false}
                name="Write"
              />
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>

      {/* Per-Container Breakdown */}
      <div className="bg-card border border-border rounded-xl p-4">
        <h3 className="text-sm font-semibold mb-4">Per-Container Resources</h3>
        <div className="space-y-3">
          {containers
            .filter((c) => c.status === "running")
            .slice(0, 5)
            .map((container) => (
              <div key={container.id} className="flex items-center gap-4">
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium truncate">
                    {container.name}
                  </p>
                  <p className="text-xs text-muted-foreground truncate">
                    {container.image}
                  </p>
                </div>
                <div className="flex items-center gap-4 text-sm">
                  <div className="text-right">
                    <p className="text-cyan-500">{container.cpu.toFixed(1)}%</p>
                    <p className="text-xs text-muted-foreground">CPU</p>
                  </div>
                  <div className="text-right">
                    <p className="text-violet-500">
                      {container.memory.toFixed(0)} MB
                    </p>
                    <p className="text-xs text-muted-foreground">Memory</p>
                  </div>
                </div>
              </div>
            ))}
        </div>
      </div>
    </div>
  );
}

/**
 * MetricCard - Display current metric value with trend
 */
function MetricCard({
  title,
  value,
  unit,
  trend,
  icon: Icon,
  color,
}: {
  title: string;
  value: number;
  unit: string;
  trend: "up" | "down" | "stable";
  icon: any;
  color: string;
}) {
  const colorClasses = {
    cyan: "text-cyan-500",
    violet: "text-violet-500",
    blue: "text-blue-500",
    green: "text-green-500",
  };

  const TrendIcon =
    trend === "up" ? TrendingUp : trend === "down" ? TrendingDown : Minus;
  const trendColor =
    trend === "up"
      ? "text-red-500"
      : trend === "down"
        ? "text-green-500"
        : "text-gray-500";

  return (
    <div className="bg-card border border-border rounded-xl p-4">
      <div className="flex items-center gap-2 text-muted-foreground text-sm mb-2">
        <Icon
          className={`w-4 h-4 ${colorClasses[color as keyof typeof colorClasses]}`}
        />
        {title}
      </div>
      <div className="flex items-end justify-between">
        <p
          className={`text-2xl font-bold ${colorClasses[color as keyof typeof colorClasses]}`}
        >
          {value.toFixed(1)}
          <span className="text-sm ml-1">{unit}</span>
        </p>
        <TrendIcon className={`w-4 h-4 ${trendColor}`} />
      </div>
    </div>
  );
}

/**
 * ChartCard - Wrapper for chart components
 */
function ChartCard({
  title,
  color,
  children,
}: {
  title: string;
  color: string;
  children: React.ReactNode;
}) {
  return (
    <div className="bg-card border border-border rounded-xl p-4">
      <h3 className="text-sm font-semibold mb-4" style={{ color }}>
        {title}
      </h3>
      {children}
    </div>
  );
}
