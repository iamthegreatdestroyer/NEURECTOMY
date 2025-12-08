/**
 * ExperimentDashboard Component
 * MLflow experiment tracking and comparison interface
 */

import { useState } from "react";
import {
  BarChart3,
  TrendingUp,
  TrendingDown,
  Filter,
  Search,
  Calendar,
  GitBranch,
  CheckCircle,
  AlertCircle,
  Clock,
  Star,
  StarOff,
  Eye,
  Trash2,
  Download,
  RefreshCw,
} from "lucide-react";

// Experiment data structure
interface Experiment {
  id: string;
  name: string;
  runId: string;
  status: "running" | "completed" | "failed" | "stopped";
  startTime: Date;
  endTime?: Date;
  duration: string;
  modelType: string;
  dataset: string;
  starred: boolean;
  metrics: {
    finalLoss: number;
    finalAccuracy: number;
    bestEpoch: number;
    totalEpochs: number;
  };
  parameters: {
    learningRate: number;
    batchSize: number;
    optimizer: string;
  };
  artifacts: number;
  tags: string[];
}

// Mock experiment data
const mockExperiments: Experiment[] = [
  {
    id: "exp_001",
    name: "BERT Fine-tuning v1",
    runId: "run_12345",
    status: "completed",
    startTime: new Date(Date.now() - 86400000 * 2),
    endTime: new Date(Date.now() - 86400000),
    duration: "2h 34m",
    modelType: "transformer",
    dataset: "custom-dataset-001",
    starred: true,
    metrics: {
      finalLoss: 0.234,
      finalAccuracy: 0.92,
      bestEpoch: 8,
      totalEpochs: 10,
    },
    parameters: {
      learningRate: 0.0001,
      batchSize: 32,
      optimizer: "adamw",
    },
    artifacts: 5,
    tags: ["production", "bert", "nlp"],
  },
  {
    id: "exp_002",
    name: "ResNet50 Training",
    runId: "run_12346",
    status: "running",
    startTime: new Date(Date.now() - 3600000),
    duration: "1h 15m",
    modelType: "cnn",
    dataset: "imagenet-subset",
    starred: false,
    metrics: {
      finalLoss: 0.567,
      finalAccuracy: 0.78,
      bestEpoch: 5,
      totalEpochs: 20,
    },
    parameters: {
      learningRate: 0.001,
      batchSize: 64,
      optimizer: "adam",
    },
    artifacts: 3,
    tags: ["experiment", "resnet", "vision"],
  },
  {
    id: "exp_003",
    name: "LSTM Sequence Model",
    runId: "run_12347",
    status: "failed",
    startTime: new Date(Date.now() - 7200000),
    endTime: new Date(Date.now() - 5400000),
    duration: "30m",
    modelType: "rnn",
    dataset: "time-series-data",
    starred: false,
    metrics: {
      finalLoss: 1.234,
      finalAccuracy: 0.45,
      bestEpoch: 3,
      totalEpochs: 15,
    },
    parameters: {
      learningRate: 0.01,
      batchSize: 128,
      optimizer: "sgd",
    },
    artifacts: 1,
    tags: ["time-series", "lstm"],
  },
  {
    id: "exp_004",
    name: "ViT Image Classification",
    runId: "run_12348",
    status: "completed",
    startTime: new Date(Date.now() - 86400000 * 5),
    endTime: new Date(Date.now() - 86400000 * 4),
    duration: "3h 45m",
    modelType: "transformer",
    dataset: "coco-2017",
    starred: true,
    metrics: {
      finalLoss: 0.189,
      finalAccuracy: 0.94,
      bestEpoch: 12,
      totalEpochs: 15,
    },
    parameters: {
      learningRate: 0.0002,
      batchSize: 16,
      optimizer: "adamw",
    },
    artifacts: 8,
    tags: ["production", "vit", "vision"],
  },
];

export function ExperimentDashboard() {
  const [experiments] = useState<Experiment[]>(mockExperiments);
  const [searchQuery, setSearchQuery] = useState("");
  const [filterStatus, setFilterStatus] = useState<string>("all");
  const [sortBy, setSortBy] = useState<"date" | "accuracy" | "loss">("date");
  const [selectedExperiments, setSelectedExperiments] = useState<Set<string>>(
    new Set()
  );

  const statusConfig = {
    running: {
      color: "text-blue-500",
      bg: "bg-blue-500/10",
      icon: RefreshCw,
      animation: "animate-spin",
    },
    completed: {
      color: "text-green-500",
      bg: "bg-green-500/10",
      icon: CheckCircle,
      animation: "",
    },
    failed: {
      color: "text-red-500",
      bg: "bg-red-500/10",
      icon: AlertCircle,
      animation: "",
    },
    stopped: {
      color: "text-yellow-500",
      bg: "bg-yellow-500/10",
      icon: Clock,
      animation: "",
    },
  };

  const filteredExperiments = experiments
    .filter((exp) => {
      const matchesSearch =
        exp.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        exp.tags.some((tag) =>
          tag.toLowerCase().includes(searchQuery.toLowerCase())
        );
      const matchesFilter =
        filterStatus === "all" || exp.status === filterStatus;
      return matchesSearch && matchesFilter;
    })
    .sort((a, b) => {
      switch (sortBy) {
        case "accuracy":
          return b.metrics.finalAccuracy - a.metrics.finalAccuracy;
        case "loss":
          return a.metrics.finalLoss - b.metrics.finalLoss;
        case "date":
        default:
          return b.startTime.getTime() - a.startTime.getTime();
      }
    });

  const toggleExperimentSelection = (id: string) => {
    const newSelection = new Set(selectedExperiments);
    if (newSelection.has(id)) {
      newSelection.delete(id);
    } else {
      newSelection.add(id);
    }
    setSelectedExperiments(newSelection);
  };

  const stats = {
    total: experiments.length,
    running: experiments.filter((e) => e.status === "running").length,
    completed: experiments.filter((e) => e.status === "completed").length,
    failed: experiments.filter((e) => e.status === "failed").length,
    avgAccuracy:
      experiments
        .filter((e) => e.status === "completed")
        .reduce((sum, e) => sum + e.metrics.finalAccuracy, 0) /
      Math.max(1, experiments.filter((e) => e.status === "completed").length),
  };

  return (
    <div className="h-full flex flex-col bg-background">
      {/* Header */}
      <div className="border-b border-border bg-card px-6 py-4">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-xl font-bold flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-primary" />
              Experiment Dashboard
            </h2>
            <p className="text-sm text-muted-foreground mt-1">
              Track, compare, and analyze ML experiments
            </p>
          </div>
          <button className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors">
            New Experiment
          </button>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          <div className="bg-background rounded-lg p-3 border border-border">
            <p className="text-xs text-muted-foreground mb-1">
              Total Experiments
            </p>
            <p className="text-2xl font-bold">{stats.total}</p>
          </div>
          <div className="bg-background rounded-lg p-3 border border-border">
            <p className="text-xs text-muted-foreground mb-1">Running</p>
            <p className="text-2xl font-bold text-blue-500">{stats.running}</p>
          </div>
          <div className="bg-background rounded-lg p-3 border border-border">
            <p className="text-xs text-muted-foreground mb-1">Completed</p>
            <p className="text-2xl font-bold text-green-500">
              {stats.completed}
            </p>
          </div>
          <div className="bg-background rounded-lg p-3 border border-border">
            <p className="text-xs text-muted-foreground mb-1">Failed</p>
            <p className="text-2xl font-bold text-red-500">{stats.failed}</p>
          </div>
          <div className="bg-background rounded-lg p-3 border border-border">
            <p className="text-xs text-muted-foreground mb-1">Avg Accuracy</p>
            <p className="text-2xl font-bold text-primary">
              {(stats.avgAccuracy * 100).toFixed(1)}%
            </p>
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="border-b border-border bg-card px-6 py-3">
        <div className="flex flex-col sm:flex-row items-start sm:items-center gap-3">
          {/* Search */}
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <input
              type="text"
              placeholder="Search experiments, tags..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 bg-background border border-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
            />
          </div>

          {/* Filter by Status */}
          <div className="flex items-center gap-2">
            <Filter className="w-4 h-4 text-muted-foreground" />
            <select
              value={filterStatus}
              onChange={(e) => setFilterStatus(e.target.value)}
              className="px-3 py-2 bg-background border border-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
            >
              <option value="all">All Status</option>
              <option value="running">Running</option>
              <option value="completed">Completed</option>
              <option value="failed">Failed</option>
              <option value="stopped">Stopped</option>
            </select>
          </div>

          {/* Sort */}
          <div className="flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-muted-foreground" />
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as any)}
              className="px-3 py-2 bg-background border border-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
            >
              <option value="date">Latest First</option>
              <option value="accuracy">Best Accuracy</option>
              <option value="loss">Lowest Loss</option>
            </select>
          </div>
        </div>

        {/* Bulk Actions */}
        {selectedExperiments.size > 0 && (
          <div className="mt-3 flex items-center gap-2 p-2 bg-primary/10 border border-primary/20 rounded-lg">
            <span className="text-sm font-medium">
              {selectedExperiments.size} selected
            </span>
            <div className="flex-1" />
            <button className="px-3 py-1.5 text-sm bg-background border border-border rounded hover:border-primary/50 transition-colors">
              Compare
            </button>
            <button className="px-3 py-1.5 text-sm bg-background border border-border rounded hover:border-primary/50 transition-colors">
              <Download className="w-3.5 h-3.5" />
            </button>
            <button className="px-3 py-1.5 text-sm bg-red-500/10 text-red-500 border border-red-500/20 rounded hover:bg-red-500/20 transition-colors">
              <Trash2 className="w-3.5 h-3.5" />
            </button>
          </div>
        )}
      </div>

      {/* Experiments List */}
      <div className="flex-1 overflow-auto p-6">
        <div className="space-y-3">
          {filteredExperiments.map((experiment) => {
            const StatusIcon = statusConfig[experiment.status].icon;
            const isSelected = selectedExperiments.has(experiment.id);

            return (
              <div
                key={experiment.id}
                className={`bg-card border rounded-xl p-4 hover:border-primary/30 transition-colors ${
                  isSelected ? "border-primary" : "border-border"
                }`}
              >
                <div className="flex items-start gap-4">
                  {/* Checkbox */}
                  <input
                    type="checkbox"
                    checked={isSelected}
                    onChange={() => toggleExperimentSelection(experiment.id)}
                    className="mt-1 w-4 h-4 rounded border-border text-primary focus:ring-2 focus:ring-primary/50"
                  />

                  {/* Main Content */}
                  <div className="flex-1 min-w-0">
                    {/* Header */}
                    <div className="flex items-start justify-between gap-4 mb-3">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                          <h3 className="font-semibold text-base truncate">
                            {experiment.name}
                          </h3>
                          {experiment.starred && (
                            <Star className="w-4 h-4 text-yellow-500 fill-yellow-500 flex-shrink-0" />
                          )}
                        </div>
                        <div className="flex items-center gap-3 text-xs text-muted-foreground">
                          <span className="flex items-center gap-1">
                            <GitBranch className="w-3 h-3" />
                            {experiment.runId}
                          </span>
                          <span className="flex items-center gap-1">
                            <Calendar className="w-3 h-3" />
                            {experiment.startTime.toLocaleDateString()}
                          </span>
                          <span>{experiment.duration}</span>
                        </div>
                      </div>

                      {/* Status Badge */}
                      <div
                        className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full border ${statusConfig[experiment.status].bg} ${statusConfig[experiment.status].color} border-current/20`}
                      >
                        <StatusIcon
                          className={`w-3 h-3 ${statusConfig[experiment.status].animation}`}
                        />
                        <span className="text-xs font-medium capitalize">
                          {experiment.status}
                        </span>
                      </div>
                    </div>

                    {/* Metrics Grid */}
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-3">
                      <div className="bg-background rounded-lg p-2">
                        <p className="text-xs text-muted-foreground mb-0.5">
                          Final Loss
                        </p>
                        <p className="text-sm font-semibold flex items-center gap-1">
                          {experiment.metrics.finalLoss.toFixed(4)}
                          <TrendingDown className="w-3 h-3 text-green-500" />
                        </p>
                      </div>
                      <div className="bg-background rounded-lg p-2">
                        <p className="text-xs text-muted-foreground mb-0.5">
                          Accuracy
                        </p>
                        <p className="text-sm font-semibold flex items-center gap-1">
                          {(experiment.metrics.finalAccuracy * 100).toFixed(2)}%
                          <TrendingUp className="w-3 h-3 text-green-500" />
                        </p>
                      </div>
                      <div className="bg-background rounded-lg p-2">
                        <p className="text-xs text-muted-foreground mb-0.5">
                          Best Epoch
                        </p>
                        <p className="text-sm font-semibold">
                          {experiment.metrics.bestEpoch} /{" "}
                          {experiment.metrics.totalEpochs}
                        </p>
                      </div>
                      <div className="bg-background rounded-lg p-2">
                        <p className="text-xs text-muted-foreground mb-0.5">
                          Artifacts
                        </p>
                        <p className="text-sm font-semibold">
                          {experiment.artifacts} files
                        </p>
                      </div>
                    </div>

                    {/* Parameters */}
                    <div className="flex items-center gap-4 text-xs text-muted-foreground mb-3">
                      <span>LR: {experiment.parameters.learningRate}</span>
                      <span>Batch: {experiment.parameters.batchSize}</span>
                      <span>Optimizer: {experiment.parameters.optimizer}</span>
                      <span>Dataset: {experiment.dataset}</span>
                    </div>

                    {/* Tags */}
                    <div className="flex items-center gap-2 flex-wrap mb-3">
                      {experiment.tags.map((tag) => (
                        <span
                          key={tag}
                          className="px-2 py-0.5 bg-primary/10 text-primary text-xs rounded-full border border-primary/20"
                        >
                          {tag}
                        </span>
                      ))}
                    </div>

                    {/* Actions */}
                    <div className="flex items-center gap-2">
                      <button className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-background border border-border rounded hover:border-primary/50 transition-colors">
                        <Eye className="w-3 h-3" />
                        View Details
                      </button>
                      <button className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-background border border-border rounded hover:border-primary/50 transition-colors">
                        <Download className="w-3 h-3" />
                        Export
                      </button>
                      <button className="p-1.5 bg-background border border-border rounded hover:border-primary/50 transition-colors">
                        {experiment.starred ? (
                          <StarOff className="w-3 h-3" />
                        ) : (
                          <Star className="w-3 h-3" />
                        )}
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {filteredExperiments.length === 0 && (
          <div className="text-center py-12">
            <BarChart3 className="w-12 h-12 text-muted-foreground mx-auto mb-3" />
            <p className="text-muted-foreground">No experiments found</p>
            <p className="text-sm text-muted-foreground/60">
              Try adjusting your filters
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

export default ExperimentDashboard;
