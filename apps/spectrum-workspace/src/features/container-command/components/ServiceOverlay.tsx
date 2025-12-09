/**
 * ServiceOverlay Component
 *
 * Visual overlay for displaying Kubernetes service information and connections.
 */

import {
  Network,
  ExternalLink,
  Globe,
  Shield,
  Copy,
  CheckCircle,
} from "lucide-react";
import { useState } from "react";
import type { KubernetesService } from "@/stores/container-store";

interface ServiceOverlayProps {
  service: KubernetesService;
  connectedPods?: number;
  onClose?: () => void;
}

/**
 * ServiceOverlay - Detailed view of a Kubernetes service
 *
 * Features:
 * - Service type and metadata
 * - IP addresses (cluster and external)
 * - Port mappings
 * - Selector labels
 * - Connected pod count
 * - Copy to clipboard functionality
 *
 * @example
 * ```tsx
 * <ServiceOverlay
 *   service={service}
 *   connectedPods={3}
 *   onClose={handleClose}
 * />
 * ```
 */
export function ServiceOverlay({
  service,
  connectedPods = 0,
  onClose,
}: ServiceOverlayProps) {
  const [copiedField, setCopiedField] = useState<string | null>(null);

  const copyToClipboard = (text: string, field: string) => {
    navigator.clipboard.writeText(text);
    setCopiedField(field);
    setTimeout(() => setCopiedField(null), 2000);
  };

  const serviceTypeConfig = {
    ClusterIP: { color: "text-blue-500", icon: Shield, label: "Internal" },
    NodePort: { color: "text-purple-500", icon: Network, label: "Node Port" },
    LoadBalancer: {
      color: "text-green-500",
      icon: Globe,
      label: "Load Balancer",
    },
    ExternalName: {
      color: "text-orange-500",
      icon: ExternalLink,
      label: "External",
    },
  };

  const typeInfo =
    serviceTypeConfig[service.type] || serviceTypeConfig.ClusterIP;
  const TypeIcon = typeInfo.icon;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Content */}
      <div className="relative bg-card border border-border rounded-xl shadow-2xl max-w-2xl w-full max-h-[80vh] overflow-auto">
        {/* Header */}
        <div className="sticky top-0 bg-card border-b border-border p-6 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div
              className={`w-10 h-10 rounded-lg ${typeInfo.color.replace("text-", "bg-")}/10 flex items-center justify-center`}
            >
              <TypeIcon className={`w-5 h-5 ${typeInfo.color}`} />
            </div>
            <div>
              <h2 className="text-lg font-semibold">{service.name}</h2>
              <p className="text-sm text-muted-foreground">
                {service.namespace}
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="w-8 h-8 rounded-lg hover:bg-muted transition-colors flex items-center justify-center"
          >
            ✕
          </button>
        </div>

        {/* Body */}
        <div className="p-6 space-y-6">
          {/* Service Type */}
          <div>
            <label className="text-sm font-medium text-muted-foreground mb-2 block">
              Service Type
            </label>
            <div
              className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-lg border ${typeInfo.color.replace("text-", "border-")} ${typeInfo.color.replace("text-", "bg-")}/10`}
            >
              <TypeIcon className={`w-4 h-4 ${typeInfo.color}`} />
              <span className={`text-sm font-medium ${typeInfo.color}`}>
                {service.type}
              </span>
              <span className="text-xs text-muted-foreground">
                ({typeInfo.label})
              </span>
            </div>
          </div>

          {/* IP Addresses */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Cluster IP */}
            {service.clusterIP && (
              <div>
                <label className="text-sm font-medium text-muted-foreground mb-2 block">
                  Cluster IP
                </label>
                <div className="flex items-center gap-2 px-3 py-2 bg-muted rounded-lg font-mono text-sm">
                  <span className="flex-1">{service.clusterIP}</span>
                  <button
                    onClick={() =>
                      copyToClipboard(service.clusterIP!, "clusterIP")
                    }
                    className="p-1 hover:bg-background rounded transition-colors"
                    title="Copy to clipboard"
                  >
                    {copiedField === "clusterIP" ? (
                      <CheckCircle className="w-4 h-4 text-green-500" />
                    ) : (
                      <Copy className="w-4 h-4" />
                    )}
                  </button>
                </div>
              </div>
            )}

            {/* External IP */}
            {service.externalIP && (
              <div>
                <label className="text-sm font-medium text-muted-foreground mb-2 block">
                  External IP
                </label>
                <div className="flex items-center gap-2 px-3 py-2 bg-muted rounded-lg font-mono text-sm">
                  <span className="flex-1">{service.externalIP}</span>
                  <button
                    onClick={() =>
                      copyToClipboard(service.externalIP!, "externalIP")
                    }
                    className="p-1 hover:bg-background rounded transition-colors"
                    title="Copy to clipboard"
                  >
                    {copiedField === "externalIP" ? (
                      <CheckCircle className="w-4 h-4 text-green-500" />
                    ) : (
                      <Copy className="w-4 h-4" />
                    )}
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Ports */}
          {service.ports && service.ports.length > 0 && (
            <div>
              <label className="text-sm font-medium text-muted-foreground mb-2 block">
                Port Mappings
              </label>
              <div className="space-y-2">
                {service.ports.map((port, idx) => (
                  <div
                    key={idx}
                    className="flex items-center gap-3 px-3 py-2 bg-muted rounded-lg"
                  >
                    <span className="font-mono text-sm">
                      {port.port}
                      {port.targetPort && ` → ${port.targetPort}`}
                    </span>
                    {port.protocol && (
                      <span className="text-xs px-2 py-0.5 bg-background rounded">
                        {port.protocol}
                      </span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Selector */}
          {service.selector && Object.keys(service.selector).length > 0 && (
            <div>
              <label className="text-sm font-medium text-muted-foreground mb-2 block">
                Pod Selector
              </label>
              <div className="flex flex-wrap gap-2">
                {Object.entries(service.selector).map(([key, value]) => (
                  <div
                    key={key}
                    className="px-3 py-1.5 bg-primary/10 text-primary rounded-lg text-sm font-mono"
                  >
                    {key}={value}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Connected Pods */}
          <div>
            <label className="text-sm font-medium text-muted-foreground mb-2 block">
              Connected Pods
            </label>
            <div className="flex items-center gap-2 px-3 py-2 bg-muted rounded-lg">
              <Network className="w-4 h-4 text-primary" />
              <span className="text-sm">
                {connectedPods} pod{connectedPods !== 1 ? "s" : ""} connected
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * ServiceCard - Compact card view of a service
 */
export function ServiceCard({
  service,
  connectedPods = 0,
  onClick,
}: {
  service: KubernetesService;
  connectedPods?: number;
  onClick?: () => void;
}) {
  const serviceTypeConfig = {
    ClusterIP: { color: "text-blue-500", icon: Shield },
    NodePort: { color: "text-purple-500", icon: Network },
    LoadBalancer: { color: "text-green-500", icon: Globe },
    ExternalName: { color: "text-orange-500", icon: ExternalLink },
  };

  const typeInfo =
    serviceTypeConfig[service.type] || serviceTypeConfig.ClusterIP;
  const TypeIcon = typeInfo.icon;

  return (
    <div
      className="bg-card border border-border rounded-lg p-4 hover:border-primary/50 transition-all cursor-pointer"
      onClick={onClick}
    >
      <div className="flex items-center gap-3 mb-3">
        <TypeIcon className={`w-5 h-5 ${typeInfo.color}`} />
        <div className="flex-1 min-w-0">
          <h3 className="font-semibold truncate">{service.name}</h3>
          <p className="text-xs text-muted-foreground truncate">
            {service.namespace}
          </p>
        </div>
      </div>

      <div className="space-y-2 text-sm">
        <div className="flex items-center justify-between">
          <span className="text-muted-foreground">Type</span>
          <span className={typeInfo.color}>{service.type}</span>
        </div>
        {service.clusterIP && (
          <div className="flex items-center justify-between">
            <span className="text-muted-foreground">Cluster IP</span>
            <span className="font-mono text-xs">{service.clusterIP}</span>
          </div>
        )}
        <div className="flex items-center justify-between">
          <span className="text-muted-foreground">Pods</span>
          <span className="text-primary">{connectedPods}</span>
        </div>
      </div>
    </div>
  );
}
