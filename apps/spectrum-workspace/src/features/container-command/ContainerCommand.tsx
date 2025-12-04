/**
 * Container Command Feature
 * Docker and Kubernetes management interface
 */

import { useState } from 'react';
import { 
  Box, 
  Server, 
  Play, 
  Square, 
  RefreshCw, 
  Terminal,
  Cpu,
  HardDrive,
  Network,
  Clock,
  AlertCircle,
  CheckCircle,
  MoreVertical,
  Search,
  Filter,
  Plus
} from 'lucide-react';

// Container status type
type ContainerStatus = 'running' | 'stopped' | 'paused' | 'restarting' | 'error';

// Mock container data
interface Container {
  id: string;
  name: string;
  image: string;
  status: ContainerStatus;
  cpu: number;
  memory: number;
  memoryLimit: number;
  network: string;
  ports: string[];
  created: Date;
  uptime: string;
}

// Status badge component
function StatusBadge({ status }: { status: ContainerStatus }) {
  const statusConfig = {
    running: { color: 'bg-green-500/10 text-green-500 border-green-500/20', icon: CheckCircle },
    stopped: { color: 'bg-gray-500/10 text-gray-500 border-gray-500/20', icon: Square },
    paused: { color: 'bg-yellow-500/10 text-yellow-500 border-yellow-500/20', icon: Clock },
    restarting: { color: 'bg-blue-500/10 text-blue-500 border-blue-500/20', icon: RefreshCw },
    error: { color: 'bg-red-500/10 text-red-500 border-red-500/20', icon: AlertCircle },
  };

  const config = statusConfig[status];
  const Icon = config.icon;

  return (
    <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border ${config.color}`}>
      <Icon className="w-3 h-3" />
      {status.charAt(0).toUpperCase() + status.slice(1)}
    </span>
  );
}

// Container card component
function ContainerCard({ container }: { container: Container }) {
  const memoryPercent = (container.memory / container.memoryLimit) * 100;

  return (
    <div className="bg-card border border-border rounded-xl p-4 hover:border-primary/30 transition-colors">
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-primary/10 rounded-lg flex items-center justify-center">
            <Box className="w-5 h-5 text-primary" />
          </div>
          <div>
            <h3 className="font-semibold text-foreground">{container.name}</h3>
            <p className="text-xs text-muted-foreground font-mono">{container.image}</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <StatusBadge status={container.status} />
          <button className="p-1.5 hover:bg-muted rounded-lg transition-colors">
            <MoreVertical className="w-4 h-4 text-muted-foreground" />
          </button>
        </div>
      </div>

      {/* Resource metrics */}
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="space-y-1">
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <Cpu className="w-3.5 h-3.5" />
            CPU
          </div>
          <div className="h-2 bg-muted rounded-full overflow-hidden">
            <div
              className="h-full bg-cyan-500 rounded-full transition-all"
              style={{ width: `${container.cpu}%` }}
            />
          </div>
          <p className="text-xs font-medium">{container.cpu.toFixed(1)}%</p>
        </div>
        <div className="space-y-1">
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <HardDrive className="w-3.5 h-3.5" />
            Memory
          </div>
          <div className="h-2 bg-muted rounded-full overflow-hidden">
            <div
              className="h-full bg-violet-500 rounded-full transition-all"
              style={{ width: `${memoryPercent}%` }}
            />
          </div>
          <p className="text-xs font-medium">
            {container.memory}MB / {container.memoryLimit}MB
          </p>
        </div>
      </div>

      {/* Container info */}
      <div className="flex items-center gap-4 text-xs text-muted-foreground mb-4">
        <div className="flex items-center gap-1.5">
          <Network className="w-3.5 h-3.5" />
          {container.network}
        </div>
        <div className="flex items-center gap-1.5">
          <Clock className="w-3.5 h-3.5" />
          {container.uptime}
        </div>
      </div>

      {/* Actions */}
      <div className="flex items-center gap-2 pt-4 border-t border-border">
        {container.status === 'running' ? (
          <>
            <button className="flex-1 flex items-center justify-center gap-2 py-2 px-3 bg-destructive/10 text-destructive rounded-lg text-sm font-medium hover:bg-destructive/20 transition-colors">
              <Square className="w-4 h-4" />
              Stop
            </button>
            <button className="flex-1 flex items-center justify-center gap-2 py-2 px-3 bg-muted text-foreground rounded-lg text-sm font-medium hover:bg-muted/80 transition-colors">
              <RefreshCw className="w-4 h-4" />
              Restart
            </button>
          </>
        ) : (
          <button className="flex-1 flex items-center justify-center gap-2 py-2 px-3 bg-green-500/10 text-green-500 rounded-lg text-sm font-medium hover:bg-green-500/20 transition-colors">
            <Play className="w-4 h-4" />
            Start
          </button>
        )}
        <button className="p-2 bg-muted rounded-lg hover:bg-muted/80 transition-colors">
          <Terminal className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}

// Main Container Command component
export function ContainerCommand() {
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState<ContainerStatus | 'all'>('all');

  // Mock container data
  const containers: Container[] = [
    {
      id: '1',
      name: 'neurectomy-api',
      image: 'neurectomy/api:latest',
      status: 'running',
      cpu: 45.2,
      memory: 512,
      memoryLimit: 1024,
      network: 'neurectomy-net',
      ports: ['8080:8080'],
      created: new Date(),
      uptime: '2d 14h',
    },
    {
      id: '2',
      name: 'neurectomy-ml',
      image: 'neurectomy/ml-service:latest',
      status: 'running',
      cpu: 78.5,
      memory: 2048,
      memoryLimit: 4096,
      network: 'neurectomy-net',
      ports: ['8000:8000'],
      created: new Date(),
      uptime: '2d 14h',
    },
    {
      id: '3',
      name: 'postgres',
      image: 'postgres:15-alpine',
      status: 'running',
      cpu: 12.3,
      memory: 256,
      memoryLimit: 512,
      network: 'neurectomy-net',
      ports: ['5432:5432'],
      created: new Date(),
      uptime: '5d 8h',
    },
    {
      id: '4',
      name: 'redis',
      image: 'redis:7-alpine',
      status: 'running',
      cpu: 3.1,
      memory: 64,
      memoryLimit: 128,
      network: 'neurectomy-net',
      ports: ['6379:6379'],
      created: new Date(),
      uptime: '5d 8h',
    },
    {
      id: '5',
      name: 'neo4j',
      image: 'neo4j:5.12',
      status: 'stopped',
      cpu: 0,
      memory: 0,
      memoryLimit: 2048,
      network: 'neurectomy-net',
      ports: ['7474:7474', '7687:7687'],
      created: new Date(),
      uptime: '-',
    },
    {
      id: '6',
      name: 'ollama',
      image: 'ollama/ollama:latest',
      status: 'running',
      cpu: 95.2,
      memory: 8192,
      memoryLimit: 16384,
      network: 'neurectomy-net',
      ports: ['11434:11434'],
      created: new Date(),
      uptime: '1d 6h',
    },
  ];

  // Filter containers
  const filteredContainers = containers.filter((c) => {
    if (statusFilter !== 'all' && c.status !== statusFilter) return false;
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      return (
        c.name.toLowerCase().includes(query) ||
        c.image.toLowerCase().includes(query)
      );
    }
    return true;
  });

  // Stats
  const runningCount = containers.filter((c) => c.status === 'running').length;
  const totalCpu = containers.reduce((sum, c) => sum + c.cpu, 0) / containers.length;
  const totalMemory = containers.reduce((sum, c) => sum + c.memory, 0);

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground flex items-center gap-3">
            <Server className="w-7 h-7 text-primary" />
            Container Command
          </h1>
          <p className="text-muted-foreground mt-1">
            Docker & Kubernetes management center
          </p>
        </div>
        <button className="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg font-medium hover:bg-primary/90 transition-colors">
          <Plus className="w-4 h-4" />
          New Container
        </button>
      </div>

      {/* Stats overview */}
      <div className="grid grid-cols-4 gap-4">
        <div className="bg-card border border-border rounded-xl p-4">
          <div className="flex items-center gap-2 text-muted-foreground text-sm mb-2">
            <Box className="w-4 h-4" />
            Total Containers
          </div>
          <p className="text-2xl font-bold">{containers.length}</p>
        </div>
        <div className="bg-card border border-border rounded-xl p-4">
          <div className="flex items-center gap-2 text-muted-foreground text-sm mb-2">
            <CheckCircle className="w-4 h-4 text-green-500" />
            Running
          </div>
          <p className="text-2xl font-bold text-green-500">{runningCount}</p>
        </div>
        <div className="bg-card border border-border rounded-xl p-4">
          <div className="flex items-center gap-2 text-muted-foreground text-sm mb-2">
            <Cpu className="w-4 h-4 text-cyan-500" />
            Avg CPU
          </div>
          <p className="text-2xl font-bold text-cyan-500">{totalCpu.toFixed(1)}%</p>
        </div>
        <div className="bg-card border border-border rounded-xl p-4">
          <div className="flex items-center gap-2 text-muted-foreground text-sm mb-2">
            <HardDrive className="w-4 h-4 text-violet-500" />
            Total Memory
          </div>
          <p className="text-2xl font-bold text-violet-500">
            {(totalMemory / 1024).toFixed(1)} GB
          </p>
        </div>
      </div>

      {/* Filters */}
      <div className="flex items-center gap-4">
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <input
            type="text"
            placeholder="Search containers..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-10 pr-4 py-2 bg-card border border-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
          />
        </div>
        <div className="flex items-center gap-2">
          <Filter className="w-4 h-4 text-muted-foreground" />
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value as ContainerStatus | 'all')}
            className="px-3 py-2 bg-card border border-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
          >
            <option value="all">All Status</option>
            <option value="running">Running</option>
            <option value="stopped">Stopped</option>
            <option value="paused">Paused</option>
            <option value="error">Error</option>
          </select>
        </div>
      </div>

      {/* Container grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
        {filteredContainers.map((container) => (
          <ContainerCard key={container.id} container={container} />
        ))}
      </div>

      {filteredContainers.length === 0 && (
        <div className="text-center py-12">
          <Box className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
          <h3 className="text-lg font-semibold mb-2">No containers found</h3>
          <p className="text-muted-foreground">
            {searchQuery
              ? 'Try adjusting your search query'
              : 'Create a new container to get started'}
          </p>
        </div>
      )}
    </div>
  );
}

export default ContainerCommand;
