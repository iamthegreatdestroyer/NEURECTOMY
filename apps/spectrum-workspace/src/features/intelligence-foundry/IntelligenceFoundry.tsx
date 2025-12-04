/**
 * Intelligence Foundry Feature
 * ML Model Training & Deployment Hub
 */

import { useState } from 'react';
import {
  Brain,
  Zap,
  Upload,
  Play,
  Pause,
  TrendingUp,
  Database,
  Layers,
  GitBranch,
  Clock,
  CheckCircle,
  AlertCircle,
  Loader,
  MoreVertical,
  Plus,
  Filter,
  Search,
  ChevronRight,
} from 'lucide-react';

// Model status type
type ModelStatus = 'training' | 'deployed' | 'idle' | 'failed' | 'queued';

// Training job type
interface TrainingJob {
  id: string;
  modelName: string;
  status: ModelStatus;
  progress: number;
  epoch: number;
  totalEpochs: number;
  loss: number;
  accuracy: number;
  startTime: Date;
  estimatedCompletion: string;
  gpuUsage: number;
  dataset: string;
}

// Model version type
interface ModelVersion {
  id: string;
  version: string;
  createdAt: Date;
  metrics: {
    accuracy: number;
    f1Score: number;
    latency: number;
  };
  status: 'deployed' | 'archived' | 'testing';
}

// Status badge component
function StatusBadge({ status }: { status: ModelStatus }) {
  const config = {
    training: { color: 'bg-blue-500/10 text-blue-500 border-blue-500/20', icon: Loader, animate: true },
    deployed: { color: 'bg-green-500/10 text-green-500 border-green-500/20', icon: CheckCircle, animate: false },
    idle: { color: 'bg-gray-500/10 text-gray-500 border-gray-500/20', icon: Pause, animate: false },
    failed: { color: 'bg-red-500/10 text-red-500 border-red-500/20', icon: AlertCircle, animate: false },
    queued: { color: 'bg-yellow-500/10 text-yellow-500 border-yellow-500/20', icon: Clock, animate: false },
  };

  const { color, icon: Icon, animate } = config[status];

  return (
    <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border ${color}`}>
      <Icon className={`w-3 h-3 ${animate ? 'animate-spin' : ''}`} />
      {status.charAt(0).toUpperCase() + status.slice(1)}
    </span>
  );
}

// Training job card
function TrainingJobCard({ job }: { job: TrainingJob }) {
  return (
    <div className="bg-card border border-border rounded-xl p-4 hover:border-primary/30 transition-colors">
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-gradient-to-br from-violet-500/20 to-fuchsia-500/20 rounded-lg flex items-center justify-center">
            <Brain className="w-5 h-5 text-violet-500" />
          </div>
          <div>
            <h3 className="font-semibold text-foreground">{job.modelName}</h3>
            <p className="text-xs text-muted-foreground">{job.dataset}</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <StatusBadge status={job.status} />
          <button className="p-1.5 hover:bg-muted rounded-lg transition-colors">
            <MoreVertical className="w-4 h-4 text-muted-foreground" />
          </button>
        </div>
      </div>

      {/* Progress */}
      {job.status === 'training' && (
        <div className="mb-4">
          <div className="flex items-center justify-between text-xs mb-2">
            <span className="text-muted-foreground">
              Epoch {job.epoch}/{job.totalEpochs}
            </span>
            <span className="text-foreground font-medium">{job.progress}%</span>
          </div>
          <div className="h-2 bg-muted rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-violet-500 to-fuchsia-500 rounded-full transition-all"
              style={{ width: `${job.progress}%` }}
            />
          </div>
        </div>
      )}

      {/* Metrics */}
      <div className="grid grid-cols-3 gap-4 mb-4">
        <div>
          <p className="text-xs text-muted-foreground mb-1">Loss</p>
          <p className="text-sm font-semibold text-foreground">{job.loss.toFixed(4)}</p>
        </div>
        <div>
          <p className="text-xs text-muted-foreground mb-1">Accuracy</p>
          <p className="text-sm font-semibold text-green-500">{(job.accuracy * 100).toFixed(1)}%</p>
        </div>
        <div>
          <p className="text-xs text-muted-foreground mb-1">GPU</p>
          <p className="text-sm font-semibold text-cyan-500">{job.gpuUsage}%</p>
        </div>
      </div>

      {/* Footer */}
      <div className="flex items-center justify-between pt-4 border-t border-border">
        <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
          <Clock className="w-3.5 h-3.5" />
          {job.estimatedCompletion}
        </div>
        <div className="flex items-center gap-2">
          {job.status === 'training' && (
            <button className="p-2 bg-destructive/10 text-destructive rounded-lg hover:bg-destructive/20 transition-colors">
              <Pause className="w-4 h-4" />
            </button>
          )}
          <button className="flex items-center gap-1.5 px-3 py-2 bg-muted rounded-lg text-sm font-medium hover:bg-muted/80 transition-colors">
            View Details
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
}

// Model card
function ModelCard({ model }: { model: { id: string; name: string; type: string; versions: number; deployed: boolean } }) {
  return (
    <div className="bg-card border border-border rounded-xl p-4 hover:border-primary/30 transition-colors cursor-pointer">
      <div className="flex items-start justify-between mb-3">
        <div className="w-10 h-10 bg-primary/10 rounded-lg flex items-center justify-center">
          <Layers className="w-5 h-5 text-primary" />
        </div>
        {model.deployed && (
          <span className="px-2 py-1 bg-green-500/10 text-green-500 text-xs font-medium rounded-full">
            Deployed
          </span>
        )}
      </div>
      <h3 className="font-semibold text-foreground mb-1">{model.name}</h3>
      <p className="text-xs text-muted-foreground mb-3">{model.type}</p>
      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        <GitBranch className="w-3.5 h-3.5" />
        {model.versions} versions
      </div>
    </div>
  );
}

// Main Intelligence Foundry component
export function IntelligenceFoundry() {
  const [activeTab, setActiveTab] = useState<'training' | 'models' | 'datasets'>('training');

  // Mock training jobs
  const trainingJobs: TrainingJob[] = [
    {
      id: '1',
      modelName: 'Agent-Transformer-v2',
      status: 'training',
      progress: 67,
      epoch: 134,
      totalEpochs: 200,
      loss: 0.0234,
      accuracy: 0.943,
      startTime: new Date(),
      estimatedCompletion: '~2h 30m remaining',
      gpuUsage: 94,
      dataset: 'agent-corpus-v3',
    },
    {
      id: '2',
      modelName: 'Code-Embeddings',
      status: 'training',
      progress: 23,
      epoch: 46,
      totalEpochs: 200,
      loss: 0.1456,
      accuracy: 0.876,
      startTime: new Date(),
      estimatedCompletion: '~8h remaining',
      gpuUsage: 87,
      dataset: 'code-snippets-1M',
    },
    {
      id: '3',
      modelName: 'Intent-Classifier',
      status: 'queued',
      progress: 0,
      epoch: 0,
      totalEpochs: 100,
      loss: 0,
      accuracy: 0,
      startTime: new Date(),
      estimatedCompletion: 'Queued',
      gpuUsage: 0,
      dataset: 'intent-dataset-v2',
    },
  ];

  // Mock models
  const models = [
    { id: '1', name: 'GPT-Agent-1B', type: 'Language Model', versions: 5, deployed: true },
    { id: '2', name: 'CodeBERT-Fine', type: 'Code Understanding', versions: 3, deployed: true },
    { id: '3', name: 'Vision-Encoder', type: 'Image Embeddings', versions: 2, deployed: false },
    { id: '4', name: 'Speech-TTS', type: 'Text-to-Speech', versions: 4, deployed: false },
    { id: '5', name: 'Sentiment-v3', type: 'Classification', versions: 7, deployed: true },
    { id: '6', name: 'RAG-Retriever', type: 'Dense Retrieval', versions: 2, deployed: true },
  ];

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground flex items-center gap-3">
            <Brain className="w-7 h-7 text-violet-500" />
            Intelligence Foundry
          </h1>
          <p className="text-muted-foreground mt-1">
            ML Model Training & Deployment Hub
          </p>
        </div>
        <div className="flex items-center gap-3">
          <button className="flex items-center gap-2 px-4 py-2 bg-muted text-foreground rounded-lg font-medium hover:bg-muted/80 transition-colors">
            <Upload className="w-4 h-4" />
            Import Model
          </button>
          <button className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-violet-500 to-fuchsia-500 text-white rounded-lg font-medium hover:opacity-90 transition-opacity">
            <Plus className="w-4 h-4" />
            New Training Job
          </button>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-4 gap-4">
        <div className="bg-card border border-border rounded-xl p-4">
          <div className="flex items-center gap-2 text-muted-foreground text-sm mb-2">
            <Zap className="w-4 h-4 text-yellow-500" />
            Active Training
          </div>
          <p className="text-2xl font-bold">2</p>
        </div>
        <div className="bg-card border border-border rounded-xl p-4">
          <div className="flex items-center gap-2 text-muted-foreground text-sm mb-2">
            <Layers className="w-4 h-4 text-violet-500" />
            Total Models
          </div>
          <p className="text-2xl font-bold">{models.length}</p>
        </div>
        <div className="bg-card border border-border rounded-xl p-4">
          <div className="flex items-center gap-2 text-muted-foreground text-sm mb-2">
            <CheckCircle className="w-4 h-4 text-green-500" />
            Deployed
          </div>
          <p className="text-2xl font-bold text-green-500">
            {models.filter((m) => m.deployed).length}
          </p>
        </div>
        <div className="bg-card border border-border rounded-xl p-4">
          <div className="flex items-center gap-2 text-muted-foreground text-sm mb-2">
            <Database className="w-4 h-4 text-cyan-500" />
            GPU Cluster
          </div>
          <p className="text-2xl font-bold text-cyan-500">91%</p>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex items-center gap-1 p-1 bg-muted rounded-lg w-fit">
        {(['training', 'models', 'datasets'] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
              activeTab === tab
                ? 'bg-background text-foreground shadow-sm'
                : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            {tab.charAt(0).toUpperCase() + tab.slice(1)}
          </button>
        ))}
      </div>

      {/* Content */}
      {activeTab === 'training' && (
        <div className="space-y-4">
          <div className="flex items-center gap-4">
            <div className="relative flex-1 max-w-md">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
              <input
                type="text"
                placeholder="Search training jobs..."
                className="w-full pl-10 pr-4 py-2 bg-card border border-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
              />
            </div>
            <button className="flex items-center gap-2 px-3 py-2 bg-card border border-border rounded-lg text-sm">
              <Filter className="w-4 h-4" />
              Filter
            </button>
          </div>

          <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
            {trainingJobs.map((job) => (
              <TrainingJobCard key={job.id} job={job} />
            ))}
          </div>
        </div>
      )}

      {activeTab === 'models' && (
        <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-4 gap-4">
          {models.map((model) => (
            <ModelCard key={model.id} model={model} />
          ))}
        </div>
      )}

      {activeTab === 'datasets' && (
        <div className="text-center py-12">
          <Database className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
          <h3 className="text-lg font-semibold mb-2">Dataset Management</h3>
          <p className="text-muted-foreground mb-4">
            Upload and manage training datasets
          </p>
          <button className="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg font-medium hover:bg-primary/90 transition-colors mx-auto">
            <Upload className="w-4 h-4" />
            Upload Dataset
          </button>
        </div>
      )}
    </div>
  );
}

export default IntelligenceFoundry;
