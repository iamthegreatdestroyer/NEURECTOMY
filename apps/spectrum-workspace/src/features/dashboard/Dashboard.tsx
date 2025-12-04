/**
 * Dashboard Feature
 * 
 * Main dashboard showing:
 * - Agent overview and status
 * - System health metrics
 * - Recent activity
 * - Quick actions
 */

import { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  Activity, 
  Boxes, 
  Brain, 
  Compass, 
  Shield, 
  Zap 
} from 'lucide-react';

const moduleCards = [
  {
    id: 'dimensional-forge',
    name: 'Dimensional Forge',
    description: '3D/4D Agent visualization and orchestration',
    icon: Boxes,
    color: 'neural-blue',
    path: '/forge',
    status: 'active',
  },
  {
    id: 'container-command',
    name: 'Container Command',
    description: 'Docker & Kubernetes orchestration',
    icon: Activity,
    color: 'synapse-purple',
    path: '/containers',
    status: 'active',
  },
  {
    id: 'intelligence-foundry',
    name: 'Intelligence Foundry',
    description: 'ML/AI model training and integration',
    icon: Brain,
    color: 'forge-orange',
    path: '/intelligence',
    status: 'active',
  },
  {
    id: 'discovery-engine',
    name: 'Discovery Engine',
    description: 'Open-source discovery and integration',
    icon: Compass,
    color: 'matrix-green',
    path: '/discovery',
    status: 'coming-soon',
  },
  {
    id: 'legal-fortress',
    name: 'Legal Fortress',
    description: 'IP protection and compliance',
    icon: Shield,
    color: 'cipher-cyan',
    path: '/legal',
    status: 'coming-soon',
  },
];

export default function Dashboard() {
  const [activeAgents, setActiveAgents] = useState(12);
  const [runningContainers, setRunningContainers] = useState(8);

  return (
    <div className="h-full p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold gradient-text">NEURECTOMY</h1>
          <p className="text-muted-foreground">
            Ultimate Agent Development & Orchestration Platform
          </p>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 px-4 py-2 bg-card rounded-lg border border-border">
            <Zap className="w-4 h-4 text-matrix-green" />
            <span className="text-sm">{activeAgents} Active Agents</span>
          </div>
          <div className="flex items-center gap-2 px-4 py-2 bg-card rounded-lg border border-border">
            <Activity className="w-4 h-4 text-neural-blue" />
            <span className="text-sm">{runningContainers} Running Containers</span>
          </div>
        </div>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-4 gap-4">
        <StatCard title="Total Agents" value="40" change="+5" trend="up" />
        <StatCard title="Workflows" value="23" change="+3" trend="up" />
        <StatCard title="Models" value="12" change="0" trend="neutral" />
        <StatCard title="Containers" value="8" change="-2" trend="down" />
      </div>

      {/* Module Grid */}
      <div className="grid grid-cols-3 gap-6">
        {moduleCards.map((module, index) => (
          <motion.div
            key={module.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
          >
            <ModuleCard {...module} />
          </motion.div>
        ))}
      </div>

      {/* Recent Activity */}
      <div className="panel">
        <div className="panel-header">
          <h2 className="font-semibold">Recent Activity</h2>
        </div>
        <div className="panel-content">
          <div className="space-y-3">
            <ActivityItem
              agent="@APEX"
              action="Completed code generation"
              time="2 minutes ago"
            />
            <ActivityItem
              agent="@ARCHITECT"
              action="Designed microservices architecture"
              time="15 minutes ago"
            />
            <ActivityItem
              agent="@FLUX"
              action="Deployed container to staging"
              time="1 hour ago"
            />
            <ActivityItem
              agent="@CIPHER"
              action="Security audit completed"
              time="3 hours ago"
            />
          </div>
        </div>
      </div>
    </div>
  );
}

interface StatCardProps {
  title: string;
  value: string;
  change: string;
  trend: 'up' | 'down' | 'neutral';
}

function StatCard({ title, value, change, trend }: StatCardProps) {
  const trendColor = {
    up: 'text-matrix-green',
    down: 'text-destructive',
    neutral: 'text-muted-foreground',
  }[trend];

  return (
    <div className="panel p-4">
      <p className="text-sm text-muted-foreground">{title}</p>
      <div className="flex items-end justify-between mt-2">
        <span className="text-3xl font-bold">{value}</span>
        <span className={`text-sm ${trendColor}`}>{change}</span>
      </div>
    </div>
  );
}

interface ModuleCardProps {
  name: string;
  description: string;
  icon: React.ElementType;
  color: string;
  path: string;
  status: 'active' | 'coming-soon';
}

function ModuleCard({ name, description, icon: Icon, color, path, status }: ModuleCardProps) {
  const isActive = status === 'active';
  
  return (
    <a
      href={isActive ? path : undefined}
      className={`
        panel p-6 block transition-all duration-200
        ${isActive ? 'hover:border-primary/50 cursor-pointer' : 'opacity-60 cursor-not-allowed'}
      `}
    >
      <div className="flex items-start gap-4">
        <div className={`p-3 rounded-lg bg-${color}/10`}>
          <Icon className={`w-6 h-6 text-${color}`} />
        </div>
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <h3 className="font-semibold">{name}</h3>
            {!isActive && (
              <span className="text-xs px-2 py-0.5 rounded-full bg-muted text-muted-foreground">
                Coming Soon
              </span>
            )}
          </div>
          <p className="text-sm text-muted-foreground mt-1">{description}</p>
        </div>
      </div>
    </a>
  );
}

interface ActivityItemProps {
  agent: string;
  action: string;
  time: string;
}

function ActivityItem({ agent, action, time }: ActivityItemProps) {
  return (
    <div className="flex items-center justify-between py-2 border-b border-border last:border-0">
      <div className="flex items-center gap-3">
        <span className="font-mono text-sm text-neural-blue">{agent}</span>
        <span className="text-sm">{action}</span>
      </div>
      <span className="text-xs text-muted-foreground">{time}</span>
    </div>
  );
}
