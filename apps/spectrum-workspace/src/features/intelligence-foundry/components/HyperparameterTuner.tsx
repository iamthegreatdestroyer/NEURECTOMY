/**
 * HyperparameterTuner Component
 * Optuna-integrated automated hyperparameter optimization
 */

import { useState } from "react";
import {
  Sliders,
  Play,
  Square,
  TrendingUp,
  Target,
  Zap,
  Settings,
  BarChart3,
  CheckCircle,
  Loader,
  AlertCircle,
  Eye,
  Download,
  Copy,
} from "lucide-react";

// Parameter definition
interface Parameter {
  name: string;
  type: "float" | "int" | "categorical";
  min?: number;
  max?: number;
  choices?: string[];
  log?: boolean;
  current?: any;
  suggested?: any;
}

// Trial result
interface Trial {
  number: number;
  value: number; // objective value
  params: Record<string, any>;
  state: "running" | "complete" | "pruned" | "failed";
  startTime: Date;
  endTime?: Date;
  duration?: string;
}

// Study configuration
interface StudyConfig {
  name: string;
  direction: "minimize" | "maximize";
  objectiveMetric: "accuracy" | "f1" | "loss" | "custom";
  sampler: "tpe" | "random" | "grid" | "cmaes";
  pruner: "median" | "percentile" | "hyperband" | "none";
  nTrials: number;
  timeout?: number;
  parallelTrials: number;
}

export function HyperparameterTuner() {
  const [config, setConfig] = useState<StudyConfig>({
    name: "hyperparameter-study-001",
    direction: "maximize",
    objectiveMetric: "accuracy",
    sampler: "tpe",
    pruner: "median",
    nTrials: 100,
    parallelTrials: 4,
  });

  const [parameters, setParameters] = useState<Parameter[]>([
    {
      name: "learning_rate",
      type: "float",
      min: 1e-5,
      max: 1e-1,
      log: true,
      current: 0.001,
    },
    {
      name: "batch_size",
      type: "categorical",
      choices: ["16", "32", "64", "128"],
      current: 32,
    },
    {
      name: "num_layers",
      type: "int",
      min: 2,
      max: 12,
      current: 6,
    },
    {
      name: "dropout_rate",
      type: "float",
      min: 0.0,
      max: 0.5,
      current: 0.1,
    },
    {
      name: "optimizer",
      type: "categorical",
      choices: ["adam", "adamw", "sgd", "rmsprop"],
      current: "adamw",
    },
  ]);

  const [isRunning, setIsRunning] = useState(false);
  const [trials, setTrials] = useState<Trial[]>([
    {
      number: 1,
      value: 0.892,
      params: {
        learning_rate: 0.0005,
        batch_size: 32,
        num_layers: 8,
        dropout_rate: 0.15,
        optimizer: "adamw",
      },
      state: "complete",
      startTime: new Date(Date.now() - 300000),
      endTime: new Date(Date.now() - 180000),
      duration: "2m 0s",
    },
    {
      number: 2,
      value: 0.857,
      params: {
        learning_rate: 0.001,
        batch_size: 64,
        num_layers: 6,
        dropout_rate: 0.2,
        optimizer: "adam",
      },
      state: "complete",
      startTime: new Date(Date.now() - 180000),
      endTime: new Date(Date.now() - 60000),
      duration: "2m 0s",
    },
    {
      number: 3,
      value: 0.912,
      params: {
        learning_rate: 0.0003,
        batch_size: 32,
        num_layers: 10,
        dropout_rate: 0.1,
        optimizer: "adamw",
      },
      state: "complete",
      startTime: new Date(Date.now() - 60000),
      endTime: new Date(Date.now() - 30000),
      duration: "30s",
    },
    {
      number: 4,
      value: 0.0,
      params: {
        learning_rate: 0.0001,
        batch_size: 16,
        num_layers: 4,
        dropout_rate: 0.05,
        optimizer: "sgd",
      },
      state: "running",
      startTime: new Date(Date.now() - 15000),
    },
  ]);

  const bestTrial = trials
    .filter((t) => t.state === "complete")
    .sort((a, b) =>
      config.direction === "maximize" ? b.value - a.value : a.value - b.value
    )[0];

  const startOptimization = () => {
    setIsRunning(true);
    // In production, this would call the Optuna backend API
  };

  const stopOptimization = () => {
    setIsRunning(false);
  };

  const addParameter = () => {
    setParameters([
      ...parameters,
      {
        name: `param_${parameters.length + 1}`,
        type: "float",
        min: 0,
        max: 1,
        current: 0.5,
      },
    ]);
  };

  const removeParameter = (index: number) => {
    setParameters(parameters.filter((_, i) => i !== index));
  };

  const updateParameter = (index: number, updates: Partial<Parameter>) => {
    const newParams = [...parameters];
    newParams[index] = { ...newParams[index], ...updates };
    setParameters(newParams);
  };

  const completedTrials = trials.filter((t) => t.state === "complete").length;
  const progress = (completedTrials / config.nTrials) * 100;

  return (
    <div className="h-full flex flex-col bg-background">
      {/* Header */}
      <div className="border-b border-border bg-card px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-bold flex items-center gap-2">
              <Sliders className="w-5 h-5 text-primary" />
              Hyperparameter Tuner
            </h2>
            <p className="text-sm text-muted-foreground mt-1">
              Automated optimization with Optuna TPE sampler
            </p>
          </div>
          <div className="flex items-center gap-2">
            {isRunning ? (
              <button
                onClick={stopOptimization}
                className="flex items-center gap-2 px-4 py-2 bg-red-500/10 text-red-500 border border-red-500/20 rounded-lg hover:bg-red-500/20 transition-colors"
              >
                <Square className="w-4 h-4" />
                Stop Study
              </button>
            ) : (
              <button
                onClick={startOptimization}
                disabled={parameters.length === 0}
                className="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors disabled:opacity-50"
              >
                <Play className="w-4 h-4" />
                Start Optimization
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-auto p-6">
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
          {/* Configuration Panel */}
          <div className="xl:col-span-1 space-y-4">
            {/* Study Configuration */}
            <div className="bg-card border border-border rounded-xl p-4">
              <h3 className="font-semibold mb-4 flex items-center gap-2">
                <Settings className="w-4 h-4 text-primary" />
                Study Configuration
              </h3>

              <div className="space-y-4">
                <div>
                  <label className="text-sm font-medium text-muted-foreground mb-1.5 block">
                    Study Name
                  </label>
                  <input
                    type="text"
                    value={config.name}
                    onChange={(e) =>
                      setConfig({ ...config, name: e.target.value })
                    }
                    disabled={isRunning}
                    className="w-full px-3 py-2 bg-background border border-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 disabled:opacity-50"
                  />
                </div>

                <div>
                  <label className="text-sm font-medium text-muted-foreground mb-1.5 block">
                    Objective
                  </label>
                  <select
                    value={config.objectiveMetric}
                    onChange={(e) =>
                      setConfig({
                        ...config,
                        objectiveMetric: e.target.value as any,
                      })
                    }
                    disabled={isRunning}
                    className="w-full px-3 py-2 bg-background border border-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 disabled:opacity-50"
                  >
                    <option value="accuracy">Accuracy</option>
                    <option value="f1">F1 Score</option>
                    <option value="loss">Loss</option>
                    <option value="custom">Custom Metric</option>
                  </select>
                </div>

                <div>
                  <label className="text-sm font-medium text-muted-foreground mb-1.5 block">
                    Direction
                  </label>
                  <select
                    value={config.direction}
                    onChange={(e) =>
                      setConfig({ ...config, direction: e.target.value as any })
                    }
                    disabled={isRunning}
                    className="w-full px-3 py-2 bg-background border border-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 disabled:opacity-50"
                  >
                    <option value="maximize">Maximize</option>
                    <option value="minimize">Minimize</option>
                  </select>
                </div>

                <div>
                  <label className="text-sm font-medium text-muted-foreground mb-1.5 block">
                    Sampler
                  </label>
                  <select
                    value={config.sampler}
                    onChange={(e) =>
                      setConfig({ ...config, sampler: e.target.value as any })
                    }
                    disabled={isRunning}
                    className="w-full px-3 py-2 bg-background border border-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 disabled:opacity-50"
                  >
                    <option value="tpe">TPE (Recommended)</option>
                    <option value="random">Random Search</option>
                    <option value="grid">Grid Search</option>
                    <option value="cmaes">CMA-ES</option>
                  </select>
                </div>

                <div>
                  <label className="text-sm font-medium text-muted-foreground mb-1.5 block">
                    Pruner
                  </label>
                  <select
                    value={config.pruner}
                    onChange={(e) =>
                      setConfig({ ...config, pruner: e.target.value as any })
                    }
                    disabled={isRunning}
                    className="w-full px-3 py-2 bg-background border border-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 disabled:opacity-50"
                  >
                    <option value="median">Median Pruner</option>
                    <option value="percentile">Percentile Pruner</option>
                    <option value="hyperband">Hyperband</option>
                    <option value="none">No Pruning</option>
                  </select>
                </div>

                <div>
                  <label className="text-sm font-medium text-muted-foreground mb-1.5 block">
                    Number of Trials
                  </label>
                  <input
                    type="number"
                    value={config.nTrials}
                    onChange={(e) =>
                      setConfig({
                        ...config,
                        nTrials: parseInt(e.target.value),
                      })
                    }
                    disabled={isRunning}
                    min={1}
                    max={1000}
                    className="w-full px-3 py-2 bg-background border border-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 disabled:opacity-50"
                  />
                </div>

                <div>
                  <label className="text-sm font-medium text-muted-foreground mb-1.5 block">
                    Parallel Trials
                  </label>
                  <input
                    type="number"
                    value={config.parallelTrials}
                    onChange={(e) =>
                      setConfig({
                        ...config,
                        parallelTrials: parseInt(e.target.value),
                      })
                    }
                    disabled={isRunning}
                    min={1}
                    max={16}
                    className="w-full px-3 py-2 bg-background border border-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 disabled:opacity-50"
                  />
                </div>
              </div>
            </div>

            {/* Parameters */}
            <div className="bg-card border border-border rounded-xl p-4">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold flex items-center gap-2">
                  <Sliders className="w-4 h-4 text-primary" />
                  Search Space
                </h3>
                <button
                  onClick={addParameter}
                  disabled={isRunning}
                  className="text-xs px-2 py-1 bg-primary/10 text-primary border border-primary/20 rounded hover:bg-primary/20 transition-colors disabled:opacity-50"
                >
                  + Add
                </button>
              </div>

              <div className="space-y-3 max-h-[400px] overflow-auto">
                {parameters.map((param, index) => (
                  <div
                    key={index}
                    className="bg-background border border-border rounded-lg p-3"
                  >
                    <div className="flex items-center justify-between mb-2">
                      <input
                        type="text"
                        value={param.name}
                        onChange={(e) =>
                          updateParameter(index, { name: e.target.value })
                        }
                        disabled={isRunning}
                        className="font-medium text-sm bg-transparent border-none focus:outline-none flex-1 disabled:opacity-50"
                      />
                      <button
                        onClick={() => removeParameter(index)}
                        disabled={isRunning}
                        className="text-red-500 hover:text-red-600 disabled:opacity-50"
                      >
                        <AlertCircle className="w-3.5 h-3.5" />
                      </button>
                    </div>

                    <select
                      value={param.type}
                      onChange={(e) =>
                        updateParameter(index, { type: e.target.value as any })
                      }
                      disabled={isRunning}
                      className="w-full mb-2 px-2 py-1 text-xs bg-background border border-border rounded focus:outline-none focus:ring-2 focus:ring-primary/50 disabled:opacity-50"
                    >
                      <option value="float">Float</option>
                      <option value="int">Integer</option>
                      <option value="categorical">Categorical</option>
                    </select>

                    {param.type === "categorical" ? (
                      <input
                        type="text"
                        placeholder="value1, value2, value3"
                        value={param.choices?.join(", ") || ""}
                        onChange={(e) =>
                          updateParameter(index, {
                            choices: e.target.value
                              .split(",")
                              .map((s) => s.trim()),
                          })
                        }
                        disabled={isRunning}
                        className="w-full px-2 py-1 text-xs bg-background border border-border rounded focus:outline-none focus:ring-2 focus:ring-primary/50 disabled:opacity-50"
                      />
                    ) : (
                      <div className="grid grid-cols-2 gap-2">
                        <input
                          type="number"
                          placeholder="Min"
                          value={param.min || ""}
                          onChange={(e) =>
                            updateParameter(index, {
                              min: parseFloat(e.target.value),
                            })
                          }
                          disabled={isRunning}
                          step={param.type === "float" ? 0.001 : 1}
                          className="px-2 py-1 text-xs bg-background border border-border rounded focus:outline-none focus:ring-2 focus:ring-primary/50 disabled:opacity-50"
                        />
                        <input
                          type="number"
                          placeholder="Max"
                          value={param.max || ""}
                          onChange={(e) =>
                            updateParameter(index, {
                              max: parseFloat(e.target.value),
                            })
                          }
                          disabled={isRunning}
                          step={param.type === "float" ? 0.001 : 1}
                          className="px-2 py-1 text-xs bg-background border border-border rounded focus:outline-none focus:ring-2 focus:ring-primary/50 disabled:opacity-50"
                        />
                      </div>
                    )}

                    {param.type !== "categorical" && (
                      <label className="flex items-center gap-2 mt-2 text-xs text-muted-foreground">
                        <input
                          type="checkbox"
                          checked={param.log || false}
                          onChange={(e) =>
                            updateParameter(index, { log: e.target.checked })
                          }
                          disabled={isRunning}
                          className="w-3 h-3 rounded border-border text-primary disabled:opacity-50"
                        />
                        Log scale
                      </label>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Results Panel */}
          <div className="xl:col-span-2 space-y-4">
            {/* Progress */}
            {isRunning && (
              <div className="bg-card border border-border rounded-xl p-4">
                <h3 className="font-semibold mb-4">Optimization Progress</h3>
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-muted-foreground">
                      Trial {completedTrials} / {config.nTrials}
                    </span>
                    <span className="font-medium">{progress.toFixed(1)}%</span>
                  </div>
                  <div className="h-2 bg-muted rounded-full overflow-hidden">
                    <div
                      className="h-full bg-primary rounded-full transition-all duration-500"
                      style={{ width: `${progress}%` }}
                    />
                  </div>
                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
                    <Loader className="w-3 h-3 animate-spin" />
                    Estimated time remaining: ~
                    {Math.ceil(
                      ((config.nTrials - completedTrials) /
                        config.parallelTrials) *
                        2
                    )}{" "}
                    minutes
                  </div>
                </div>
              </div>
            )}

            {/* Best Trial */}
            {bestTrial && (
              <div className="bg-card border border-primary/30 rounded-xl p-4">
                <div className="flex items-center gap-2 mb-4">
                  <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-yellow-500/20 to-orange-500/20 border border-yellow-500/30 flex items-center justify-center">
                    <Target className="w-4 h-4 text-yellow-500" />
                  </div>
                  <div>
                    <h3 className="font-semibold">
                      Best Trial #{bestTrial.number}
                    </h3>
                    <p className="text-xs text-muted-foreground">
                      {config.objectiveMetric}: {bestTrial.value.toFixed(4)} (
                      {config.direction})
                    </p>
                  </div>
                  <div className="flex-1" />
                  <button className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-background border border-border rounded hover:border-primary/50 transition-colors">
                    <Copy className="w-3 h-3" />
                    Copy
                  </button>
                  <button className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-primary text-primary-foreground rounded hover:bg-primary/90 transition-colors">
                    <Play className="w-3 h-3" />
                    Train
                  </button>
                </div>

                <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                  {Object.entries(bestTrial.params).map(([key, value]) => (
                    <div key={key} className="bg-background rounded-lg p-2">
                      <p className="text-xs text-muted-foreground mb-0.5">
                        {key}
                      </p>
                      <p className="text-sm font-semibold">
                        {typeof value === "number"
                          ? value.toExponential(2)
                          : value}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Trial History */}
            <div className="bg-card border border-border rounded-xl p-4">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold flex items-center gap-2">
                  <BarChart3 className="w-4 h-4 text-primary" />
                  Trial History
                </h3>
                <button className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-background border border-border rounded hover:border-primary/50 transition-colors">
                  <Download className="w-3 h-3" />
                  Export
                </button>
              </div>

              <div className="space-y-2 max-h-[500px] overflow-auto">
                {trials
                  .slice()
                  .reverse()
                  .map((trial) => (
                    <div
                      key={trial.number}
                      className="bg-background border border-border rounded-lg p-3 hover:border-primary/30 transition-colors"
                    >
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center gap-2">
                          <span className="text-sm font-semibold">
                            Trial #{trial.number}
                          </span>
                          {trial.state === "running" && (
                            <Loader className="w-3 h-3 text-blue-500 animate-spin" />
                          )}
                          {trial.state === "complete" && (
                            <CheckCircle className="w-3 h-3 text-green-500" />
                          )}
                          {trial.state === "failed" && (
                            <AlertCircle className="w-3 h-3 text-red-500" />
                          )}
                        </div>
                        <div className="flex items-center gap-3">
                          <span className="text-sm font-semibold text-primary">
                            {trial.value.toFixed(4)}
                          </span>
                          <button className="p-1 hover:bg-muted rounded transition-colors">
                            <Eye className="w-3.5 h-3.5 text-muted-foreground" />
                          </button>
                        </div>
                      </div>

                      <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs text-muted-foreground">
                        {Object.entries(trial.params)
                          .slice(0, 4)
                          .map(([key, value]) => (
                            <div key={key}>
                              <span className="opacity-60">{key}:</span>{" "}
                              <span className="font-medium">
                                {typeof value === "number"
                                  ? value.toFixed(4)
                                  : value}
                              </span>
                            </div>
                          ))}
                      </div>

                      {trial.duration && (
                        <div className="mt-2 text-xs text-muted-foreground">
                          Duration: {trial.duration}
                        </div>
                      )}
                    </div>
                  ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default HyperparameterTuner;
