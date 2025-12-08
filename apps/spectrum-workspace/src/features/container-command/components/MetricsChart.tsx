/**
 * MetricsChart Component
 *
 * Reusable chart wrapper for consistent metric visualization.
 * Supports line charts, area charts, and bar charts.
 */

import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

interface MetricDataPoint {
  timestamp: string;
  [key: string]: string | number;
}

interface MetricsChartProps {
  title: string;
  data: MetricDataPoint[];
  type?: "line" | "area" | "bar";
  dataKeys: Array<{
    key: string;
    color: string;
    label: string;
  }>;
  height?: number;
  showLegend?: boolean;
  showGrid?: boolean;
  yAxisDomain?: [number, number] | ["auto", "auto"];
  unit?: string;
}

/**
 * MetricsChart - Configurable chart component for metrics visualization
 *
 * Features:
 * - Multiple chart types (line, area, bar)
 * - Multiple data series support
 * - Customizable colors and labels
 * - Automatic responsiveness
 * - Consistent dark theme styling
 * - Optional legend and grid
 *
 * @example
 * ```tsx
 * <MetricsChart
 *   title="CPU Usage"
 *   data={cpuData}
 *   type="area"
 *   dataKeys={[
 *     { key: 'value', color: '#06b6d4', label: 'CPU %' }
 *   ]}
 *   yAxisDomain={[0, 100]}
 *   unit="%"
 * />
 * ```
 */
export function MetricsChart({
  title,
  data,
  type = "line",
  dataKeys,
  height = 200,
  showLegend = false,
  showGrid = true,
  yAxisDomain = ["auto", "auto"],
  unit,
}: MetricsChartProps) {
  const chartProps = {
    data,
    margin: { top: 5, right: 5, left: 0, bottom: 5 },
  };

  const xAxisProps = {
    dataKey: "timestamp",
    stroke: "#9ca3af",
    fontSize: 12,
    tickLine: false,
  };

  const yAxisProps = {
    stroke: "#9ca3af",
    fontSize: 12,
    tickLine: false,
    domain: yAxisDomain,
    unit,
  };

  const tooltipProps = {
    contentStyle: {
      backgroundColor: "#1f2937",
      border: "1px solid #374151",
      borderRadius: "8px",
      fontSize: "12px",
    },
    labelStyle: {
      color: "#9ca3af",
    },
  };

  const gridProps = {
    strokeDasharray: "3 3",
    stroke: "#374151",
  };

  return (
    <div className="w-full">
      {title && (
        <h3 className="text-sm font-semibold mb-3 text-foreground">{title}</h3>
      )}
      <ResponsiveContainer width="100%" height={height}>
        {type === "line" && (
          <LineChart {...chartProps}>
            {showGrid && <CartesianGrid {...gridProps} />}
            <XAxis {...xAxisProps} />
            <YAxis {...yAxisProps} />
            <Tooltip {...tooltipProps} />
            {showLegend && <Legend />}
            {dataKeys.map((dk) => (
              <Line
                key={dk.key}
                type="monotone"
                dataKey={dk.key}
                stroke={dk.color}
                strokeWidth={2}
                dot={false}
                name={dk.label}
                animationDuration={300}
              />
            ))}
          </LineChart>
        )}

        {type === "area" && (
          <AreaChart {...chartProps}>
            {showGrid && <CartesianGrid {...gridProps} />}
            <XAxis {...xAxisProps} />
            <YAxis {...yAxisProps} />
            <Tooltip {...tooltipProps} />
            {showLegend && <Legend />}
            {dataKeys.map((dk) => (
              <Area
                key={dk.key}
                type="monotone"
                dataKey={dk.key}
                stroke={dk.color}
                fill={dk.color}
                fillOpacity={0.2}
                strokeWidth={2}
                name={dk.label}
                animationDuration={300}
              />
            ))}
          </AreaChart>
        )}

        {type === "bar" && (
          <BarChart {...chartProps}>
            {showGrid && <CartesianGrid {...gridProps} />}
            <XAxis {...xAxisProps} />
            <YAxis {...yAxisProps} />
            <Tooltip {...tooltipProps} />
            {showLegend && <Legend />}
            {dataKeys.map((dk) => (
              <Bar
                key={dk.key}
                dataKey={dk.key}
                fill={dk.color}
                name={dk.label}
                radius={[4, 4, 0, 0]}
                animationDuration={300}
              />
            ))}
          </BarChart>
        )}
      </ResponsiveContainer>
    </div>
  );
}

/**
 * MetricsChartCard - Chart with card wrapper
 */
export function MetricsChartCard({
  title,
  subtitle,
  color,
  children,
  actions,
}: {
  title: string;
  subtitle?: string;
  color?: string;
  children: React.ReactNode;
  actions?: React.ReactNode;
}) {
  return (
    <div className="bg-card border border-border rounded-xl p-4">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-sm font-semibold" style={{ color }}>
            {title}
          </h3>
          {subtitle && (
            <p className="text-xs text-muted-foreground mt-0.5">{subtitle}</p>
          )}
        </div>
        {actions && <div className="flex items-center gap-2">{actions}</div>}
      </div>
      {children}
    </div>
  );
}
