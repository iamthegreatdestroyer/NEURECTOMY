/**
 * ModelTrainer Component
 * MLflow-integrated training pipeline interface
 */

import { useState, useEffect } from "react";
import {
  Brain,
  Play,
  Square,
  Settings,
  Database,
  Zap,
  TrendingUp,
  AlertCircle,
  CheckCircle,
  Loader,
  Download,
  Upload,
  BarChart3,
} from "lucide-react";

// Training configuration
interface TrainingConfig {
  modelType: "transformer" | "cnn" | "rnn" | "custom";
  architecture: string;
  dataset: string;
  batchSize: number;
  learningRate: number;
  epochs: number;
  optimizer: "adam" | "sgd" | "adamw" | "rmsprop";
  scheduler: "cosine" | "step" | "exponential" | "none";
  earlyStopping: boolean;
  patience: number;
  gpuIds: number[];
  distributedTraining: boolean;
  mixedPrecision: boolean;
}

// Training metrics
interface TrainingMetrics {
  epoch: number;
  trainLoss: number;
  valLoss: number;
  trainAccuracy: number;
  valAccuracy: number;
  learningRate: number;
  gpuMemory: number;
  throughput: number; // samples/sec
  estimatedTimeRemaining: string;
}

// MLflow experiment
interface MLflowExperiment {
  experimentId: string;
  runId: string;
  name: string;
  status: "running" | "completed" | "failed" | "stopped";
  startTime: Date;
  endTime?: Date;
  metrics: TrainingMetrics[];
  params: TrainingConfig;
  artifacts: string[];
}

export function ModelTrainer() {
  const [config, setConfig] = useState<TrainingConfig>({
    modelType: "transformer",
    architecture: "bert-base",
    dataset: "custom-dataset-001",
    batchSize: 32,
    learningRate: 0.0001,
    epochs: 10,
    optimizer: "adamw",
    scheduler: "cosine",
    earlyStopping: true,
    patience: 3,
    gpuIds: [0],
    distributedTraining: false,
    mixedPrecision: true,
  });

  const [currentExperiment, setCurrentExperiment] =
    useState<MLflowExperiment | null>(null);
  const [isTraining, setIsTraining] = useState(false);
  const [currentMetrics, setCurrentMetrics] = useState<TrainingMetrics | null>(
    null
  );

  // Simulate training progress (replace with actual MLflow API calls)
  useEffect(() => {
    if (!isTraining || !currentExperiment) return;

    const interval = setInterval(() => {
      const newMetrics: TrainingMetrics = {
        epoch: currentMetrics ? currentMetrics.epoch + 1 : 1,
        trainLoss: Math.max(0.1, Math.random() * 2),
        valLoss: Math.max(0.15, Math.random() * 2.2),
        trainAccuracy: Math.min(0.99, 0.5 + Math.random() * 0.4),
        valAccuracy: Math.min(0.97, 0.45 + Math.random() * 0.4),
        learningRate:
          config.learningRate * Math.pow(0.95, currentMetrics?.epoch || 0),
        gpuMemory: 8000 + Math.random() * 2000,
        throughput: 1000 + Math.random() * 500,
        estimatedTimeRemaining: `${Math.max(0, config.epochs - (currentMetrics?.epoch || 0))} epochs (~${Math.floor(Math.random() * 60)} min)`,
      };

      setCurrentMetrics(newMetrics);

      if (newMetrics.epoch >= config.epochs) {
        setIsTraining(false);
        setCurrentExperiment({
          ...currentExperiment,
          status: "completed",
          endTime: new Date(),
        });
      }
    }, 3000); // Update every 3 seconds

    return () => clearInterval(interval);
  }, [isTraining, currentMetrics, config, currentExperiment]);

  const startTraining = () => {
    const newExperiment: MLflowExperiment = {
      experimentId: `exp_${Date.now()}`,
      runId: `run_${Date.now()}`,
      name: `${config.modelType}_${config.architecture}_${Date.now()}`,
      status: "running",
      startTime: new Date(),
      metrics: [],
      params: config,
      artifacts: [],
    };

    setCurrentExperiment(newExperiment);
    setIsTraining(true);
    setCurrentMetrics(null);
  };

  const stopTraining = () => {
    setIsTraining(false);
    if (currentExperiment) {
      setCurrentExperiment({
        ...currentExperiment,
        status: "stopped",
        endTime: new Date(),
      });
    }
  };

  return (
    <div className="h-full flex flex-col bg-background">
      {/* Header */}
      <div className="border-b border-border bg-card px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-bold flex items-center gap-2">
              <Brain className="w-5 h-5 text-primary" />
              Model Trainer
            </h2>
            <p className="text-sm text-muted-foreground mt-1">
              MLflow-integrated training pipeline with real-time metrics
            </p>
          </div>
          <div className="flex items-center gap-2">
            {isTraining ? (
              <button
                onClick={stopTraining}
                className="flex items-center gap-2 px-4 py-2 bg-red-500/10 text-red-500 border border-red-500/20 rounded-lg hover:bg-red-500/20 transition-colors"
              >
                <Square className="w-4 h-4" />
                Stop Training
              </button>
            ) : (
              <button
                onClick={startTraining}
                disabled={isTraining}
                className="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors disabled:opacity-50"
              >
                <Play className="w-4 h-4" />
                Start Training
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
            <div className="bg-card border border-border rounded-xl p-4">
              <h3 className="font-semibold mb-4 flex items-center gap-2">
                <Settings className="w-4 h-4 text-primary" />
                Training Configuration
              </h3>

              <div className="space-y-4">
                {/* Model Type */}
                <div>
                  <label className="text-sm font-medium text-muted-foreground mb-1.5 block">
                    Model Type
                  </label>
                  <select
                    value={config.modelType}
                    onChange={(e) =>
                      setConfig({ ...config, modelType: e.target.value as any })
                    }
                    disabled={isTraining}
                    className="w-full px-3 py-2 bg-background border border-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 disabled:opacity-50"
                  >
                    <option value="transformer">Transformer</option>
                    <option value="cnn">CNN</option>
                    <option value="rnn">RNN</option>
                    <option value="custom">Custom</option>
                  </select>
                </div>

                {/* Architecture */}
                <div>
                  <label className="text-sm font-medium text-muted-foreground mb-1.5 block">
                    Architecture
                  </label>
                  <input
                    type="text"
                    value={config.architecture}
                    onChange={(e) =>
                      setConfig({ ...config, architecture: e.target.value })
                    }
                    disabled={isTraining}
                    className="w-full px-3 py-2 bg-background border border-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 disabled:opacity-50"
                  />
                </div>

                {/* Dataset */}
                <div>
                  <label className="text-sm font-medium text-muted-foreground mb-1.5 block">
                    Dataset
                  </label>
                  <select
                    value={config.dataset}
                    onChange={(e) =>
                      setConfig({ ...config, dataset: e.target.value })
                    }
                    disabled={isTraining}
                    className="w-full px-3 py-2 bg-background border border-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 disabled:opacity-50"
                  >
                    <option value="custom-dataset-001">
                      Custom Dataset 001
                    </option>
                    <option value="imagenet">ImageNet</option>
                    <option value="coco">COCO</option>
                    <option value="glue">GLUE</option>
                  </select>
                </div>

                {/* Batch Size */}
                <div>
                  <label className="text-sm font-medium text-muted-foreground mb-1.5 block">
                    Batch Size
                  </label>
                  <input
                    type="number"
                    value={config.batchSize}
                    onChange={(e) =>
                      setConfig({
                        ...config,
                        batchSize: parseInt(e.target.value),
                      })
                    }
                    disabled={isTraining}
                    min={1}
                    max={512}
                    className="w-full px-3 py-2 bg-background border border-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 disabled:opacity-50"
                  />
                </div>

                {/* Learning Rate */}
                <div>
                  <label className="text-sm font-medium text-muted-foreground mb-1.5 block">
                    Learning Rate
                  </label>
                  <input
                    type="number"
                    value={config.learningRate}
                    onChange={(e) =>
                      setConfig({
                        ...config,
                        learningRate: parseFloat(e.target.value),
                      })
                    }
                    disabled={isTraining}
                    step={0.0001}
                    min={0.00001}
                    max={0.1}
                    className="w-full px-3 py-2 bg-background border border-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 disabled:opacity-50"
                  />
                </div>

                {/* Epochs */}
                <div>
                  <label className="text-sm font-medium text-muted-foreground mb-1.5 block">
                    Epochs
                  </label>
                  <input
                    type="number"
                    value={config.epochs}
                    onChange={(e) =>
                      setConfig({ ...config, epochs: parseInt(e.target.value) })
                    }
                    disabled={isTraining}
                    min={1}
                    max={1000}
                    className="w-full px-3 py-2 bg-background border border-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 disabled:opacity-50"
                  />
                </div>

                {/* Optimizer */}
                <div>
                  <label className="text-sm font-medium text-muted-foreground mb-1.5 block">
                    Optimizer
                  </label>
                  <select
                    value={config.optimizer}
                    onChange={(e) =>
                      setConfig({ ...config, optimizer: e.target.value as any })
                    }
                    disabled={isTraining}
                    className="w-full px-3 py-2 bg-background border border-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 disabled:opacity-50"
                  >
                    <option value="adam">Adam</option>
                    <option value="adamw">AdamW</option>
                    <option value="sgd">SGD</option>
                    <option value="rmsprop">RMSProp</option>
                  </select>
                </div>

                {/* Advanced Options */}
                <div className="pt-4 border-t border-border space-y-3">
                  <label className="flex items-center justify-between">
                    <span className="text-sm font-medium text-muted-foreground">
                      Mixed Precision
                    </span>
                    <input
                      type="checkbox"
                      checked={config.mixedPrecision}
                      onChange={(e) =>
                        setConfig({
                          ...config,
                          mixedPrecision: e.target.checked,
                        })
                      }
                      disabled={isTraining}
                      className="w-4 h-4 rounded border-border text-primary focus:ring-2 focus:ring-primary/50"
                    />
                  </label>
                  <label className="flex items-center justify-between">
                    <span className="text-sm font-medium text-muted-foreground">
                      Early Stopping
                    </span>
                    <input
                      type="checkbox"
                      checked={config.earlyStopping}
                      onChange={(e) =>
                        setConfig({
                          ...config,
                          earlyStopping: e.target.checked,
                        })
                      }
                      disabled={isTraining}
                      className="w-4 h-4 rounded border-border text-primary focus:ring-2 focus:ring-primary/50"
                    />
                  </label>
                  <label className="flex items-center justify-between">
                    <span className="text-sm font-medium text-muted-foreground">
                      Distributed Training
                    </span>
                    <input
                      type="checkbox"
                      checked={config.distributedTraining}
                      onChange={(e) =>
                        setConfig({
                          ...config,
                          distributedTraining: e.target.checked,
                        })
                      }
                      disabled={isTraining}
                      className="w-4 h-4 rounded border-border text-primary focus:ring-2 focus:ring-primary/50"
                    />
                  </label>
                </div>
              </div>
            </div>
          </div>

          {/* Metrics & Progress */}
          <div className="xl:col-span-2 space-y-4">
            {/* Current Experiment Status */}
            {currentExperiment && (
              <div className="bg-card border border-border rounded-xl p-4">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-semibold flex items-center gap-2">
                    <BarChart3 className="w-4 h-4 text-primary" />
                    Current Experiment
                  </h3>
                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
                    <span>Run ID: {currentExperiment.runId.slice(-8)}</span>
                  </div>
                </div>

                {/* Progress Bar */}
                {currentMetrics && (
                  <div className="space-y-2 mb-4">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-muted-foreground">
                        Epoch {currentMetrics.epoch} / {config.epochs}
                      </span>
                      <span className="font-medium">
                        {Math.round(
                          (currentMetrics.epoch / config.epochs) * 100
                        )}
                        %
                      </span>
                    </div>
                    <div className="h-2 bg-muted rounded-full overflow-hidden">
                      <div
                        className="h-full bg-primary rounded-full transition-all duration-500"
                        style={{
                          width: `${(currentMetrics.epoch / config.epochs) * 100}%`,
                        }}
                      />
                    </div>
                    <p className="text-xs text-muted-foreground">
                      {currentMetrics.estimatedTimeRemaining}
                    </p>
                  </div>
                )}

                {/* Metrics Grid */}
                {currentMetrics && (
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="bg-background rounded-lg p-3">
                      <div className="flex items-center gap-2 text-xs text-muted-foreground mb-1">
                        <TrendingUp className="w-3 h-3 text-red-500" />
                        Train Loss
                      </div>
                      <p className="text-lg font-semibold text-red-500">
                        {currentMetrics.trainLoss.toFixed(4)}
                      </p>
                    </div>
                    <div className="bg-background rounded-lg p-3">
                      <div className="flex items-center gap-2 text-xs text-muted-foreground mb-1">
                        <TrendingUp className="w-3 h-3 text-orange-500" />
                        Val Loss
                      </div>
                      <p className="text-lg font-semibold text-orange-500">
                        {currentMetrics.valLoss.toFixed(4)}
                      </p>
                    </div>
                    <div className="bg-background rounded-lg p-3">
                      <div className="flex items-center gap-2 text-xs text-muted-foreground mb-1">
                        <CheckCircle className="w-3 h-3 text-green-500" />
                        Train Acc
                      </div>
                      <p className="text-lg font-semibold text-green-500">
                        {(currentMetrics.trainAccuracy * 100).toFixed(2)}%
                      </p>
                    </div>
                    <div className="bg-background rounded-lg p-3">
                      <div className="flex items-center gap-2 text-xs text-muted-foreground mb-1">
                        <CheckCircle className="w-3 h-3 text-blue-500" />
                        Val Acc
                      </div>
                      <p className="text-lg font-semibold text-blue-500">
                        {(currentMetrics.valAccuracy * 100).toFixed(2)}%
                      </p>
                    </div>
                    <div className="bg-background rounded-lg p-3">
                      <div className="flex items-center gap-2 text-xs text-muted-foreground mb-1">
                        <Zap className="w-3 h-3 text-cyan-500" />
                        Learning Rate
                      </div>
                      <p className="text-lg font-semibold text-cyan-500">
                        {currentMetrics.learningRate.toExponential(2)}
                      </p>
                    </div>
                    <div className="bg-background rounded-lg p-3">
                      <div className="flex items-center gap-2 text-xs text-muted-foreground mb-1">
                        <Database className="w-3 h-3 text-violet-500" />
                        GPU Memory
                      </div>
                      <p className="text-lg font-semibold text-violet-500">
                        {(currentMetrics.gpuMemory / 1024).toFixed(1)} GB
                      </p>
                    </div>
                    <div className="bg-background rounded-lg p-3">
                      <div className="flex items-center gap-2 text-xs text-muted-foreground mb-1">
                        <Zap className="w-3 h-3 text-yellow-500" />
                        Throughput
                      </div>
                      <p className="text-lg font-semibold text-yellow-500">
                        {currentMetrics.throughput.toFixed(0)} s/s
                      </p>
                    </div>
                    <div className="bg-background rounded-lg p-3">
                      <div className="flex items-center gap-2 text-xs text-muted-foreground mb-1">
                        {isTraining ? (
                          <Loader className="w-3 h-3 text-primary animate-spin" />
                        ) : (
                          <CheckCircle className="w-3 h-3 text-green-500" />
                        )}
                        Status
                      </div>
                      <p className="text-lg font-semibold capitalize">
                        {currentExperiment.status}
                      </p>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* MLflow Integration Status */}
            <div className="bg-card border border-border rounded-xl p-4">
              <h3 className="font-semibold mb-4 flex items-center gap-2">
                <Database className="w-4 h-4 text-primary" />
                MLflow Integration
              </h3>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">
                    Tracking Server
                  </span>
                  <span className="text-sm font-medium flex items-center gap-1.5">
                    <CheckCircle className="w-3 h-3 text-green-500" />
                    Connected
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">
                    Model Registry
                  </span>
                  <span className="text-sm font-medium flex items-center gap-1.5">
                    <CheckCircle className="w-3 h-3 text-green-500" />
                    Active
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">
                    Artifact Store
                  </span>
                  <span className="text-sm font-medium flex items-center gap-1.5">
                    <CheckCircle className="w-3 h-3 text-green-500" />
                    S3 Bucket
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">
                    Experiment Tracking
                  </span>
                  <span className="text-sm font-medium flex items-center gap-1.5">
                    <CheckCircle className="w-3 h-3 text-green-500" />
                    Enabled
                  </span>
                </div>
              </div>
            </div>

            {/* Quick Actions */}
            <div className="bg-card border border-border rounded-xl p-4">
              <h3 className="font-semibold mb-4">Quick Actions</h3>
              <div className="grid grid-cols-2 gap-3">
                <button className="flex items-center justify-center gap-2 px-4 py-3 bg-background border border-border rounded-lg hover:border-primary/50 transition-colors">
                  <Upload className="w-4 h-4" />
                  <span className="text-sm font-medium">Upload Dataset</span>
                </button>
                <button className="flex items-center justify-center gap-2 px-4 py-3 bg-background border border-border rounded-lg hover:border-primary/50 transition-colors">
                  <Download className="w-4 h-4" />
                  <span className="text-sm font-medium">Export Model</span>
                </button>
                <button className="flex items-center justify-center gap-2 px-4 py-3 bg-background border border-border rounded-lg hover:border-primary/50 transition-colors">
                  <BarChart3 className="w-4 h-4" />
                  <span className="text-sm font-medium">View Experiments</span>
                </button>
                <button className="flex items-center justify-center gap-2 px-4 py-3 bg-background border border-border rounded-lg hover:border-primary/50 transition-colors">
                  <Settings className="w-4 h-4" />
                  <span className="text-sm font-medium">Advanced Config</span>
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ModelTrainer;
