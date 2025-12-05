/**
 * NEURECTOMY Service Mesh Integration
 *
 * @module @neurectomy/container-command/service-mesh
 * @agent @FLUX @PRISM
 *
 * Implements service mesh abstraction layer supporting:
 * - Istio integration with virtual services and destination rules
 * - Linkerd integration with service profiles and traffic splits
 * - Unified API for traffic management, observability, and security
 * - A/B testing and canary deployment support
 * - mTLS configuration and certificate management
 *
 * Architecture:
 * ┌─────────────────────────────────────────────────────────────┐
 * │                    Service Mesh Manager                      │
 * ├─────────────────────────────────────────────────────────────┤
 * │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐  │
 * │  │ Traffic Manager │  │ Security Manager│  │  Observ.    │  │
 * │  │  - Routing      │  │  - mTLS         │  │  - Metrics  │  │
 * │  │  - Load Balance │  │  - AuthN/AuthZ  │  │  - Tracing  │  │
 * │  │  - Retries      │  │  - Policies     │  │  - Logging  │  │
 * │  └─────────────────┘  └─────────────────┘  └─────────────┘  │
 * ├─────────────────────────────────────────────────────────────┤
 * │       ┌─────────────────┬─────────────────┐                 │
 * │       │  Istio Adapter  │ Linkerd Adapter │                 │
 * │       └─────────────────┴─────────────────┘                 │
 * └─────────────────────────────────────────────────────────────┘
 */

import { EventEmitter } from "eventemitter3";
import { z } from "zod";
import pino from "pino";
import YAML from "yaml";

// ============================================================================
// Service Mesh Type Definitions
// ============================================================================

/**
 * Supported service mesh providers
 */
export type ServiceMeshProvider = "istio" | "linkerd";

/**
 * Load balancing algorithms
 */
export type LoadBalancingAlgorithm =
  | "round_robin"
  | "least_connections"
  | "random"
  | "consistent_hash"
  | "maglev";

/**
 * Traffic split strategy
 */
export type TrafficSplitStrategy =
  | "weight"
  | "header"
  | "cookie"
  | "query_param";

/**
 * mTLS mode
 */
export type MTLSMode = "strict" | "permissive" | "disable";

/**
 * Circuit breaker state
 */
export type CircuitState = "closed" | "open" | "half_open";

// ============================================================================
// Zod Validation Schemas
// ============================================================================

export const ServiceMeshConfigSchema = z.object({
  provider: z.enum(["istio", "linkerd"]),
  namespace: z.string(),
  meshName: z.string().optional(),
  mtls: z
    .object({
      enabled: z.boolean().default(true),
      mode: z.enum(["strict", "permissive", "disable"]).default("strict"),
    })
    .optional(),
  observability: z
    .object({
      tracing: z
        .object({
          enabled: z.boolean().default(true),
          samplingRate: z.number().min(0).max(100).default(100),
          backend: z.enum(["jaeger", "zipkin", "tempo"]).default("jaeger"),
        })
        .optional(),
      metrics: z
        .object({
          enabled: z.boolean().default(true),
          scrapeInterval: z.string().default("15s"),
        })
        .optional(),
      accessLogging: z
        .object({
          enabled: z.boolean().default(true),
          format: z.enum(["json", "text"]).default("json"),
        })
        .optional(),
    })
    .optional(),
});

export const TrafficRouteSchema = z.object({
  name: z.string(),
  service: z.string(),
  namespace: z.string(),
  port: z.number().int().min(1).max(65535),
  routes: z.array(
    z.object({
      match: z
        .object({
          headers: z.record(z.string()).optional(),
          uri: z
            .object({
              exact: z.string().optional(),
              prefix: z.string().optional(),
              regex: z.string().optional(),
            })
            .optional(),
          method: z.string().optional(),
        })
        .optional(),
      destinations: z.array(
        z.object({
          host: z.string(),
          port: z.number().int().min(1).max(65535),
          weight: z.number().int().min(0).max(100),
          subset: z.string().optional(),
        })
      ),
      timeout: z.string().optional(),
      retries: z
        .object({
          attempts: z.number().int().min(0),
          perTryTimeout: z.string(),
          retryOn: z.string().optional(),
        })
        .optional(),
    })
  ),
});

export const CanaryDeploymentSchema = z.object({
  name: z.string(),
  service: z.string(),
  namespace: z.string(),
  stableVersion: z.string(),
  canaryVersion: z.string(),
  initialWeight: z.number().int().min(0).max(100).default(10),
  stepWeight: z.number().int().min(1).max(100).default(10),
  maxWeight: z.number().int().min(0).max(100).default(50),
  interval: z.string().default("1m"),
  metrics: z
    .array(
      z.object({
        name: z.string(),
        threshold: z.number(),
        operator: z.enum(["<", ">", "<=", ">=", "=="]),
      })
    )
    .optional(),
  webhooks: z
    .object({
      confirm: z.string().optional(),
      rollback: z.string().optional(),
    })
    .optional(),
});

export const CircuitBreakerSchema = z.object({
  name: z.string(),
  service: z.string(),
  namespace: z.string(),
  maxConnections: z.number().int().min(1).default(100),
  maxPendingRequests: z.number().int().min(1).default(100),
  maxRequests: z.number().int().min(1).default(1000),
  maxRetries: z.number().int().min(0).default(3),
  consecutiveErrors: z.number().int().min(1).default(5),
  interval: z.string().default("10s"),
  baseEjectionTime: z.string().default("30s"),
  maxEjectionPercent: z.number().int().min(0).max(100).default(50),
});

export const ServiceProfileSchema = z.object({
  name: z.string(),
  service: z.string(),
  namespace: z.string(),
  routes: z.array(
    z.object({
      name: z.string(),
      condition: z.object({
        method: z.string(),
        pathRegex: z.string(),
      }),
      isRetryable: z.boolean().default(false),
      timeout: z.string().optional(),
    })
  ),
  retryBudget: z
    .object({
      retryRatio: z.number().min(0).max(1).default(0.2),
      minRetriesPerSecond: z.number().int().min(0).default(10),
      ttl: z.string().default("10s"),
    })
    .optional(),
});

// ============================================================================
// Type Exports
// ============================================================================

export type ServiceMeshConfig = z.infer<typeof ServiceMeshConfigSchema>;
export type TrafficRoute = z.infer<typeof TrafficRouteSchema>;
export type CanaryDeployment = z.infer<typeof CanaryDeploymentSchema>;
export type CircuitBreaker = z.infer<typeof CircuitBreakerSchema>;
export type ServiceProfile = z.infer<typeof ServiceProfileSchema>;

// ============================================================================
// Service Mesh Events
// ============================================================================

export interface ServiceMeshEvents {
  "config:applied": { resource: string; name: string; namespace: string };
  "config:deleted": { resource: string; name: string; namespace: string };
  "canary:progress": {
    name: string;
    weight: number;
    metrics: Record<string, number>;
  };
  "canary:complete": { name: string; success: boolean; reason?: string };
  "canary:rollback": { name: string; reason: string };
  "circuit:state_change": {
    service: string;
    from: CircuitState;
    to: CircuitState;
  };
  "mtls:status_change": { namespace: string; mode: MTLSMode };
  error: { operation: string; error: Error };
}

// ============================================================================
// Service Mesh Adapter Interface
// ============================================================================

interface ServiceMeshAdapter {
  /**
   * Initialize the mesh adapter
   */
  initialize(): Promise<void>;

  /**
   * Apply traffic routing configuration
   */
  applyTrafficRoute(route: TrafficRoute): Promise<void>;

  /**
   * Delete traffic routing configuration
   */
  deleteTrafficRoute(name: string, namespace: string): Promise<void>;

  /**
   * Configure circuit breaker
   */
  applyCircuitBreaker(config: CircuitBreaker): Promise<void>;

  /**
   * Start canary deployment
   */
  startCanary(config: CanaryDeployment): Promise<string>;

  /**
   * Update canary weight
   */
  updateCanaryWeight(
    name: string,
    namespace: string,
    weight: number
  ): Promise<void>;

  /**
   * Rollback canary
   */
  rollbackCanary(name: string, namespace: string): Promise<void>;

  /**
   * Promote canary to stable
   */
  promoteCanary(name: string, namespace: string): Promise<void>;

  /**
   * Configure mTLS
   */
  configureMTLS(namespace: string, mode: MTLSMode): Promise<void>;

  /**
   * Get service metrics
   */
  getServiceMetrics(
    service: string,
    namespace: string
  ): Promise<ServiceMetrics>;

  /**
   * Apply rate limiting
   */
  applyRateLimit(config: RateLimitConfig): Promise<void>;

  /**
   * Get mesh status
   */
  getMeshStatus(): Promise<MeshStatus>;
}

// ============================================================================
// Supporting Types
// ============================================================================

export interface ServiceMetrics {
  requestRate: number;
  successRate: number;
  latencyP50: number;
  latencyP95: number;
  latencyP99: number;
  activeConnections: number;
  bytesIn: number;
  bytesOut: number;
}

export interface RateLimitConfig {
  name: string;
  service: string;
  namespace: string;
  descriptors: Array<{
    key: string;
    value: string;
    rateLimit: {
      requestsPerUnit: number;
      unit: "second" | "minute" | "hour" | "day";
    };
  }>;
}

export interface MeshStatus {
  provider: ServiceMeshProvider;
  version: string;
  healthy: boolean;
  components: Array<{
    name: string;
    status: "running" | "degraded" | "failed";
    version?: string;
  }>;
  proxies: {
    total: number;
    healthy: number;
    unhealthy: number;
  };
  mtlsStatus: {
    enabled: boolean;
    mode: MTLSMode;
    certificateExpiry?: Date;
  };
}

// ============================================================================
// Istio Adapter Implementation
// ============================================================================

class IstioAdapter implements ServiceMeshAdapter {
  private logger: pino.Logger;
  private kubeClient: any; // Would use @kubernetes/client-node
  private namespace: string;

  constructor(logger: pino.Logger, kubeClient: any, namespace: string) {
    this.logger = logger.child({ adapter: "istio" });
    this.kubeClient = kubeClient;
    this.namespace = namespace;
  }

  async initialize(): Promise<void> {
    this.logger.info("Initializing Istio adapter");
    // Verify Istio is installed and accessible
  }

  async applyTrafficRoute(route: TrafficRoute): Promise<void> {
    const virtualService = this.generateVirtualService(route);
    const destinationRule = this.generateDestinationRule(route);

    this.logger.info({ route: route.name }, "Applying traffic route");

    // Apply VirtualService
    await this.applyCustomResource(
      "networking.istio.io/v1beta1",
      "VirtualService",
      virtualService
    );

    // Apply DestinationRule
    await this.applyCustomResource(
      "networking.istio.io/v1beta1",
      "DestinationRule",
      destinationRule
    );
  }

  async deleteTrafficRoute(name: string, namespace: string): Promise<void> {
    this.logger.info({ name, namespace }, "Deleting traffic route");

    await this.deleteCustomResource(
      "networking.istio.io/v1beta1",
      "VirtualService",
      name,
      namespace
    );

    await this.deleteCustomResource(
      "networking.istio.io/v1beta1",
      "DestinationRule",
      `${name}-destination`,
      namespace
    );
  }

  async applyCircuitBreaker(config: CircuitBreaker): Promise<void> {
    this.logger.info({ config: config.name }, "Applying circuit breaker");

    const destinationRule = {
      apiVersion: "networking.istio.io/v1beta1",
      kind: "DestinationRule",
      metadata: {
        name: `${config.name}-circuit-breaker`,
        namespace: config.namespace,
      },
      spec: {
        host: config.service,
        trafficPolicy: {
          connectionPool: {
            tcp: {
              maxConnections: config.maxConnections,
            },
            http: {
              h2UpgradePolicy: "UPGRADE",
              http1MaxPendingRequests: config.maxPendingRequests,
              http2MaxRequests: config.maxRequests,
              maxRetries: config.maxRetries,
            },
          },
          outlierDetection: {
            consecutive5xxErrors: config.consecutiveErrors,
            interval: config.interval,
            baseEjectionTime: config.baseEjectionTime,
            maxEjectionPercent: config.maxEjectionPercent,
          },
        },
      },
    };

    await this.applyCustomResource(
      "networking.istio.io/v1beta1",
      "DestinationRule",
      destinationRule
    );
  }

  async startCanary(config: CanaryDeployment): Promise<string> {
    this.logger.info({ config: config.name }, "Starting canary deployment");

    const virtualService = {
      apiVersion: "networking.istio.io/v1beta1",
      kind: "VirtualService",
      metadata: {
        name: `${config.name}-canary`,
        namespace: config.namespace,
        labels: {
          "neurectomy.io/canary": "true",
          "neurectomy.io/canary-name": config.name,
        },
      },
      spec: {
        hosts: [config.service],
        http: [
          {
            route: [
              {
                destination: {
                  host: config.service,
                  subset: "stable",
                },
                weight: 100 - config.initialWeight,
              },
              {
                destination: {
                  host: config.service,
                  subset: "canary",
                },
                weight: config.initialWeight,
              },
            ],
          },
        ],
      },
    };

    const destinationRule = {
      apiVersion: "networking.istio.io/v1beta1",
      kind: "DestinationRule",
      metadata: {
        name: `${config.name}-canary-destination`,
        namespace: config.namespace,
      },
      spec: {
        host: config.service,
        subsets: [
          {
            name: "stable",
            labels: {
              version: config.stableVersion,
            },
          },
          {
            name: "canary",
            labels: {
              version: config.canaryVersion,
            },
          },
        ],
      },
    };

    await this.applyCustomResource(
      "networking.istio.io/v1beta1",
      "DestinationRule",
      destinationRule
    );

    await this.applyCustomResource(
      "networking.istio.io/v1beta1",
      "VirtualService",
      virtualService
    );

    return `${config.name}-canary`;
  }

  async updateCanaryWeight(
    name: string,
    namespace: string,
    weight: number
  ): Promise<void> {
    this.logger.info({ name, weight }, "Updating canary weight");

    // Get existing VirtualService and update weights
    const vsName = `${name}-canary`;
    const vs = await this.getCustomResource(
      "networking.istio.io/v1beta1",
      "VirtualService",
      vsName,
      namespace
    );

    if (vs?.spec?.http?.[0]?.route) {
      vs.spec.http[0].route[0].weight = 100 - weight;
      vs.spec.http[0].route[1].weight = weight;

      await this.applyCustomResource(
        "networking.istio.io/v1beta1",
        "VirtualService",
        vs
      );
    }
  }

  async rollbackCanary(name: string, namespace: string): Promise<void> {
    this.logger.info({ name }, "Rolling back canary deployment");
    await this.updateCanaryWeight(name, namespace, 0);
  }

  async promoteCanary(name: string, namespace: string): Promise<void> {
    this.logger.info({ name }, "Promoting canary to stable");
    await this.updateCanaryWeight(name, namespace, 100);

    // Clean up canary resources after promotion
    await this.deleteTrafficRoute(`${name}-canary`, namespace);
  }

  async configureMTLS(namespace: string, mode: MTLSMode): Promise<void> {
    this.logger.info({ namespace, mode }, "Configuring mTLS");

    const peerAuth = {
      apiVersion: "security.istio.io/v1beta1",
      kind: "PeerAuthentication",
      metadata: {
        name: `${namespace}-mtls`,
        namespace: namespace,
      },
      spec: {
        mtls: {
          mode: mode.toUpperCase(),
        },
      },
    };

    await this.applyCustomResource(
      "security.istio.io/v1beta1",
      "PeerAuthentication",
      peerAuth
    );
  }

  async getServiceMetrics(
    service: string,
    namespace: string
  ): Promise<ServiceMetrics> {
    this.logger.debug({ service, namespace }, "Getting service metrics");

    // In production, would query Prometheus or Istio telemetry
    // This is a placeholder implementation
    return {
      requestRate: 0,
      successRate: 100,
      latencyP50: 0,
      latencyP95: 0,
      latencyP99: 0,
      activeConnections: 0,
      bytesIn: 0,
      bytesOut: 0,
    };
  }

  async applyRateLimit(config: RateLimitConfig): Promise<void> {
    this.logger.info({ config: config.name }, "Applying rate limit");

    const envoyFilter = {
      apiVersion: "networking.istio.io/v1alpha3",
      kind: "EnvoyFilter",
      metadata: {
        name: `${config.name}-ratelimit`,
        namespace: config.namespace,
      },
      spec: {
        workloadSelector: {
          labels: {
            app: config.service,
          },
        },
        configPatches: [
          {
            applyTo: "HTTP_FILTER",
            match: {
              context: "SIDECAR_INBOUND",
              listener: {
                filterChain: {
                  filter: {
                    name: "envoy.filters.network.http_connection_manager",
                  },
                },
              },
            },
            patch: {
              operation: "INSERT_BEFORE",
              value: {
                name: "envoy.filters.http.local_ratelimit",
                typed_config: {
                  "@type": "type.googleapis.com/udpa.type.v1.TypedStruct",
                  type_url:
                    "type.googleapis.com/envoy.extensions.filters.http.local_ratelimit.v3.LocalRateLimit",
                  value: {
                    stat_prefix: "http_local_rate_limiter",
                    token_bucket: {
                      max_tokens:
                        config.descriptors[0]?.rateLimit.requestsPerUnit || 100,
                      tokens_per_fill:
                        config.descriptors[0]?.rateLimit.requestsPerUnit || 100,
                      fill_interval: this.convertRateLimitUnit(
                        config.descriptors[0]?.rateLimit.unit || "second"
                      ),
                    },
                  },
                },
              },
            },
          },
        ],
      },
    };

    await this.applyCustomResource(
      "networking.istio.io/v1alpha3",
      "EnvoyFilter",
      envoyFilter
    );
  }

  async getMeshStatus(): Promise<MeshStatus> {
    this.logger.debug("Getting mesh status");

    // In production, would query Istio control plane
    return {
      provider: "istio",
      version: "1.20.0",
      healthy: true,
      components: [
        { name: "istiod", status: "running", version: "1.20.0" },
        { name: "istio-ingressgateway", status: "running" },
        { name: "istio-egressgateway", status: "running" },
      ],
      proxies: {
        total: 0,
        healthy: 0,
        unhealthy: 0,
      },
      mtlsStatus: {
        enabled: true,
        mode: "strict",
      },
    };
  }

  // Helper methods

  private generateVirtualService(route: TrafficRoute): any {
    return {
      apiVersion: "networking.istio.io/v1beta1",
      kind: "VirtualService",
      metadata: {
        name: route.name,
        namespace: route.namespace,
      },
      spec: {
        hosts: [route.service],
        http: route.routes.map((r) => ({
          match: r.match
            ? [
                {
                  headers: r.match.headers
                    ? Object.entries(r.match.headers).reduce(
                        (acc, [k, v]) => ({
                          ...acc,
                          [k]: { exact: v },
                        }),
                        {}
                      )
                    : undefined,
                  uri: r.match.uri,
                  method: r.match.method
                    ? { exact: r.match.method }
                    : undefined,
                },
              ]
            : undefined,
          route: r.destinations.map((d) => ({
            destination: {
              host: d.host,
              port: { number: d.port },
              subset: d.subset,
            },
            weight: d.weight,
          })),
          timeout: r.timeout,
          retries: r.retries
            ? {
                attempts: r.retries.attempts,
                perTryTimeout: r.retries.perTryTimeout,
                retryOn: r.retries.retryOn,
              }
            : undefined,
        })),
      },
    };
  }

  private generateDestinationRule(route: TrafficRoute): any {
    const subsets = new Set<string>();
    route.routes.forEach((r) =>
      r.destinations.forEach((d) => {
        if (d.subset) subsets.add(d.subset);
      })
    );

    return {
      apiVersion: "networking.istio.io/v1beta1",
      kind: "DestinationRule",
      metadata: {
        name: `${route.name}-destination`,
        namespace: route.namespace,
      },
      spec: {
        host: route.service,
        subsets:
          subsets.size > 0
            ? Array.from(subsets).map((s) => ({
                name: s,
                labels: { version: s },
              }))
            : undefined,
      },
    };
  }

  private convertRateLimitUnit(unit: string): string {
    switch (unit) {
      case "second":
        return "1s";
      case "minute":
        return "60s";
      case "hour":
        return "3600s";
      case "day":
        return "86400s";
      default:
        return "1s";
    }
  }

  private async applyCustomResource(
    apiVersion: string,
    kind: string,
    resource: any
  ): Promise<void> {
    this.logger.debug(
      { apiVersion, kind, name: resource.metadata.name },
      "Applying custom resource"
    );
    // Would use Kubernetes API to apply CRD
  }

  private async deleteCustomResource(
    apiVersion: string,
    kind: string,
    name: string,
    namespace: string
  ): Promise<void> {
    this.logger.debug(
      { apiVersion, kind, name, namespace },
      "Deleting custom resource"
    );
    // Would use Kubernetes API to delete CRD
  }

  private async getCustomResource(
    apiVersion: string,
    kind: string,
    name: string,
    namespace: string
  ): Promise<any> {
    this.logger.debug(
      { apiVersion, kind, name, namespace },
      "Getting custom resource"
    );
    // Would use Kubernetes API to get CRD
    return null;
  }
}

// ============================================================================
// Linkerd Adapter Implementation
// ============================================================================

class LinkerdAdapter implements ServiceMeshAdapter {
  private logger: pino.Logger;
  private kubeClient: any;
  private namespace: string;

  constructor(logger: pino.Logger, kubeClient: any, namespace: string) {
    this.logger = logger.child({ adapter: "linkerd" });
    this.kubeClient = kubeClient;
    this.namespace = namespace;
  }

  async initialize(): Promise<void> {
    this.logger.info("Initializing Linkerd adapter");
  }

  async applyTrafficRoute(route: TrafficRoute): Promise<void> {
    this.logger.info({ route: route.name }, "Applying traffic route");

    // Create HTTPRoute for Linkerd
    const httpRoute = {
      apiVersion: "policy.linkerd.io/v1beta2",
      kind: "HTTPRoute",
      metadata: {
        name: route.name,
        namespace: route.namespace,
      },
      spec: {
        parentRefs: [
          {
            name: route.service,
            kind: "Service",
            group: "",
          },
        ],
        rules: route.routes.map((r) => ({
          matches: r.match
            ? [
                {
                  path: r.match.uri?.prefix
                    ? { type: "PathPrefix", value: r.match.uri.prefix }
                    : r.match.uri?.exact
                      ? { type: "Exact", value: r.match.uri.exact }
                      : undefined,
                  method: r.match.method,
                  headers: r.match.headers
                    ? Object.entries(r.match.headers).map(([k, v]) => ({
                        name: k,
                        value: v,
                      }))
                    : undefined,
                },
              ]
            : undefined,
          backendRefs: r.destinations.map((d) => ({
            name: d.host,
            port: d.port,
            weight: d.weight,
          })),
          timeouts: r.timeout
            ? {
                request: r.timeout,
              }
            : undefined,
        })),
      },
    };

    await this.applyCustomResource(
      "policy.linkerd.io/v1beta2",
      "HTTPRoute",
      httpRoute
    );
  }

  async deleteTrafficRoute(name: string, namespace: string): Promise<void> {
    this.logger.info({ name, namespace }, "Deleting traffic route");

    await this.deleteCustomResource(
      "policy.linkerd.io/v1beta2",
      "HTTPRoute",
      name,
      namespace
    );
  }

  async applyCircuitBreaker(config: CircuitBreaker): Promise<void> {
    this.logger.info({ config: config.name }, "Applying circuit breaker");

    // Linkerd uses ServiceProfiles for retry budgets and circuit breaking
    const serviceProfile = {
      apiVersion: "linkerd.io/v1alpha2",
      kind: "ServiceProfile",
      metadata: {
        name: `${config.service}.${config.namespace}.svc.cluster.local`,
        namespace: config.namespace,
      },
      spec: {
        routes: [],
        retryBudget: {
          retryRatio: 0.2,
          minRetriesPerSecond: 10,
          ttl: "10s",
        },
      },
    };

    await this.applyCustomResource(
      "linkerd.io/v1alpha2",
      "ServiceProfile",
      serviceProfile
    );
  }

  async startCanary(config: CanaryDeployment): Promise<string> {
    this.logger.info({ config: config.name }, "Starting canary deployment");

    // Create TrafficSplit for canary
    const trafficSplit = {
      apiVersion: "split.smi-spec.io/v1alpha2",
      kind: "TrafficSplit",
      metadata: {
        name: config.name,
        namespace: config.namespace,
      },
      spec: {
        service: config.service,
        backends: [
          {
            service: `${config.service}-stable`,
            weight: 100 - config.initialWeight,
          },
          {
            service: `${config.service}-canary`,
            weight: config.initialWeight,
          },
        ],
      },
    };

    await this.applyCustomResource(
      "split.smi-spec.io/v1alpha2",
      "TrafficSplit",
      trafficSplit
    );

    return config.name;
  }

  async updateCanaryWeight(
    name: string,
    namespace: string,
    weight: number
  ): Promise<void> {
    this.logger.info({ name, weight }, "Updating canary weight");

    const ts = await this.getCustomResource(
      "split.smi-spec.io/v1alpha2",
      "TrafficSplit",
      name,
      namespace
    );

    if (ts?.spec?.backends) {
      ts.spec.backends[0].weight = 100 - weight;
      ts.spec.backends[1].weight = weight;

      await this.applyCustomResource(
        "split.smi-spec.io/v1alpha2",
        "TrafficSplit",
        ts
      );
    }
  }

  async rollbackCanary(name: string, namespace: string): Promise<void> {
    this.logger.info({ name }, "Rolling back canary");
    await this.updateCanaryWeight(name, namespace, 0);
  }

  async promoteCanary(name: string, namespace: string): Promise<void> {
    this.logger.info({ name }, "Promoting canary");
    await this.updateCanaryWeight(name, namespace, 100);
  }

  async configureMTLS(namespace: string, mode: MTLSMode): Promise<void> {
    this.logger.info({ namespace, mode }, "Configuring mTLS");

    // Linkerd uses Server and ServerAuthorization for mTLS policies
    const server = {
      apiVersion: "policy.linkerd.io/v1beta1",
      kind: "Server",
      metadata: {
        name: `${namespace}-server`,
        namespace: namespace,
      },
      spec: {
        podSelector: {
          matchLabels: {},
        },
        port: "http",
        proxyProtocol: mode === "disable" ? "HTTP/1" : "HTTP/2",
      },
    };

    await this.applyCustomResource(
      "policy.linkerd.io/v1beta1",
      "Server",
      server
    );
  }

  async getServiceMetrics(
    service: string,
    namespace: string
  ): Promise<ServiceMetrics> {
    this.logger.debug({ service, namespace }, "Getting service metrics");

    return {
      requestRate: 0,
      successRate: 100,
      latencyP50: 0,
      latencyP95: 0,
      latencyP99: 0,
      activeConnections: 0,
      bytesIn: 0,
      bytesOut: 0,
    };
  }

  async applyRateLimit(config: RateLimitConfig): Promise<void> {
    this.logger.info({ config: config.name }, "Applying rate limit");

    // Linkerd uses HTTPLocalRateLimitPolicy
    const rateLimitPolicy = {
      apiVersion: "policy.linkerd.io/v1alpha1",
      kind: "HTTPLocalRateLimitPolicy",
      metadata: {
        name: config.name,
        namespace: config.namespace,
      },
      spec: {
        targetRef: {
          group: "",
          kind: "Service",
          name: config.service,
        },
        local: {
          defaultFill: {
            requestsPerPeriod:
              config.descriptors[0]?.rateLimit.requestsPerUnit || 100,
            period: this.convertRateLimitPeriod(
              config.descriptors[0]?.rateLimit.unit || "second"
            ),
          },
        },
      },
    };

    await this.applyCustomResource(
      "policy.linkerd.io/v1alpha1",
      "HTTPLocalRateLimitPolicy",
      rateLimitPolicy
    );
  }

  async getMeshStatus(): Promise<MeshStatus> {
    return {
      provider: "linkerd",
      version: "2.14.0",
      healthy: true,
      components: [
        { name: "linkerd-destination", status: "running" },
        { name: "linkerd-identity", status: "running" },
        { name: "linkerd-proxy-injector", status: "running" },
      ],
      proxies: {
        total: 0,
        healthy: 0,
        unhealthy: 0,
      },
      mtlsStatus: {
        enabled: true,
        mode: "strict",
      },
    };
  }

  private convertRateLimitPeriod(unit: string): string {
    switch (unit) {
      case "second":
        return "1s";
      case "minute":
        return "1m";
      case "hour":
        return "1h";
      case "day":
        return "24h";
      default:
        return "1s";
    }
  }

  private async applyCustomResource(
    apiVersion: string,
    kind: string,
    resource: any
  ): Promise<void> {
    this.logger.debug(
      { apiVersion, kind, name: resource.metadata.name },
      "Applying custom resource"
    );
  }

  private async deleteCustomResource(
    apiVersion: string,
    kind: string,
    name: string,
    namespace: string
  ): Promise<void> {
    this.logger.debug(
      { apiVersion, kind, name, namespace },
      "Deleting custom resource"
    );
  }

  private async getCustomResource(
    apiVersion: string,
    kind: string,
    name: string,
    namespace: string
  ): Promise<any> {
    this.logger.debug(
      { apiVersion, kind, name, namespace },
      "Getting custom resource"
    );
    return null;
  }
}

// ============================================================================
// Service Mesh Manager
// ============================================================================

export interface ServiceMeshManagerOptions {
  config: ServiceMeshConfig;
  kubeClient?: any;
  logger?: pino.Logger;
}

/**
 * Service Mesh Manager
 *
 * @agent @FLUX @PRISM
 *
 * Unified service mesh abstraction supporting Istio and Linkerd.
 * Provides traffic management, security, and observability capabilities.
 *
 * @example
 * ```typescript
 * const manager = new ServiceMeshManager({
 *   config: {
 *     provider: "istio",
 *     namespace: "neurectomy",
 *     mtls: { enabled: true, mode: "strict" }
 *   }
 * });
 *
 * await manager.initialize();
 *
 * // Apply traffic routing
 * await manager.applyTrafficRoute({
 *   name: "agent-routing",
 *   service: "agent-service",
 *   namespace: "neurectomy",
 *   port: 8080,
 *   routes: [...]
 * });
 *
 * // Start canary deployment
 * await manager.startCanary({
 *   name: "agent-canary",
 *   service: "agent-service",
 *   namespace: "neurectomy",
 *   stableVersion: "v1",
 *   canaryVersion: "v2",
 *   initialWeight: 10
 * });
 * ```
 */
export class ServiceMeshManager extends EventEmitter<ServiceMeshEvents> {
  private logger: pino.Logger;
  private config: ServiceMeshConfig;
  private adapter: ServiceMeshAdapter;
  private kubeClient: any;
  private canaryMonitors: Map<string, NodeJS.Timeout> = new Map();

  constructor(options: ServiceMeshManagerOptions) {
    super();

    const validatedConfig = ServiceMeshConfigSchema.parse(options.config);
    this.config = validatedConfig;
    this.kubeClient = options.kubeClient;
    this.logger = (options.logger || pino()).child({
      module: "service-mesh",
      provider: validatedConfig.provider,
    });

    // Create adapter based on provider
    this.adapter = this.createAdapter(validatedConfig.provider);
  }

  private createAdapter(provider: ServiceMeshProvider): ServiceMeshAdapter {
    switch (provider) {
      case "istio":
        return new IstioAdapter(
          this.logger,
          this.kubeClient,
          this.config.namespace
        );
      case "linkerd":
        return new LinkerdAdapter(
          this.logger,
          this.kubeClient,
          this.config.namespace
        );
      default:
        throw new Error(`Unsupported service mesh provider: ${provider}`);
    }
  }

  /**
   * Initialize the service mesh manager
   */
  async initialize(): Promise<void> {
    this.logger.info(
      { config: this.config },
      "Initializing service mesh manager"
    );

    await this.adapter.initialize();

    // Configure mTLS if enabled
    if (this.config.mtls?.enabled) {
      await this.adapter.configureMTLS(
        this.config.namespace,
        this.config.mtls.mode || "strict"
      );
    }

    this.logger.info("Service mesh manager initialized");
  }

  /**
   * Apply traffic routing configuration
   */
  async applyTrafficRoute(route: TrafficRoute): Promise<void> {
    const validatedRoute = TrafficRouteSchema.parse(route);

    this.logger.info({ route: validatedRoute.name }, "Applying traffic route");

    try {
      await this.adapter.applyTrafficRoute(validatedRoute);

      this.emit("config:applied", {
        resource: "TrafficRoute",
        name: validatedRoute.name,
        namespace: validatedRoute.namespace,
      });
    } catch (error) {
      this.emit("error", {
        operation: "applyTrafficRoute",
        error: error as Error,
      });
      throw error;
    }
  }

  /**
   * Delete traffic routing configuration
   */
  async deleteTrafficRoute(name: string, namespace: string): Promise<void> {
    this.logger.info({ name, namespace }, "Deleting traffic route");

    try {
      await this.adapter.deleteTrafficRoute(name, namespace);

      this.emit("config:deleted", {
        resource: "TrafficRoute",
        name,
        namespace,
      });
    } catch (error) {
      this.emit("error", {
        operation: "deleteTrafficRoute",
        error: error as Error,
      });
      throw error;
    }
  }

  /**
   * Apply circuit breaker configuration
   */
  async applyCircuitBreaker(config: CircuitBreaker): Promise<void> {
    const validatedConfig = CircuitBreakerSchema.parse(config);

    this.logger.info(
      { config: validatedConfig.name },
      "Applying circuit breaker"
    );

    try {
      await this.adapter.applyCircuitBreaker(validatedConfig);

      this.emit("config:applied", {
        resource: "CircuitBreaker",
        name: validatedConfig.name,
        namespace: validatedConfig.namespace,
      });
    } catch (error) {
      this.emit("error", {
        operation: "applyCircuitBreaker",
        error: error as Error,
      });
      throw error;
    }
  }

  /**
   * Start canary deployment with automatic progression
   */
  async startCanary(
    config: CanaryDeployment,
    options?: { autoProgress?: boolean }
  ): Promise<string> {
    const validatedConfig = CanaryDeploymentSchema.parse(config);

    this.logger.info(
      { config: validatedConfig.name },
      "Starting canary deployment"
    );

    try {
      const canaryId = await this.adapter.startCanary(validatedConfig);

      this.emit("canary:progress", {
        name: validatedConfig.name,
        weight: validatedConfig.initialWeight,
        metrics: {},
      });

      // Start automatic progression if enabled
      if (options?.autoProgress) {
        this.startCanaryProgression(validatedConfig);
      }

      return canaryId;
    } catch (error) {
      this.emit("error", {
        operation: "startCanary",
        error: error as Error,
      });
      throw error;
    }
  }

  /**
   * Update canary weight manually
   */
  async updateCanaryWeight(
    name: string,
    namespace: string,
    weight: number
  ): Promise<void> {
    this.logger.info({ name, weight }, "Updating canary weight");

    try {
      await this.adapter.updateCanaryWeight(name, namespace, weight);

      this.emit("canary:progress", {
        name,
        weight,
        metrics: await this.getCanaryMetrics(name, namespace),
      });
    } catch (error) {
      this.emit("error", {
        operation: "updateCanaryWeight",
        error: error as Error,
      });
      throw error;
    }
  }

  /**
   * Rollback canary deployment
   */
  async rollbackCanary(name: string, namespace: string): Promise<void> {
    this.logger.info({ name }, "Rolling back canary");

    try {
      // Stop automatic progression
      this.stopCanaryProgression(name);

      await this.adapter.rollbackCanary(name, namespace);

      this.emit("canary:rollback", {
        name,
        reason: "Manual rollback",
      });
    } catch (error) {
      this.emit("error", {
        operation: "rollbackCanary",
        error: error as Error,
      });
      throw error;
    }
  }

  /**
   * Promote canary to stable
   */
  async promoteCanary(name: string, namespace: string): Promise<void> {
    this.logger.info({ name }, "Promoting canary");

    try {
      this.stopCanaryProgression(name);

      await this.adapter.promoteCanary(name, namespace);

      this.emit("canary:complete", {
        name,
        success: true,
      });
    } catch (error) {
      this.emit("error", {
        operation: "promoteCanary",
        error: error as Error,
      });
      throw error;
    }
  }

  /**
   * Configure mTLS for a namespace
   */
  async configureMTLS(namespace: string, mode: MTLSMode): Promise<void> {
    this.logger.info({ namespace, mode }, "Configuring mTLS");

    try {
      await this.adapter.configureMTLS(namespace, mode);

      this.emit("mtls:status_change", { namespace, mode });
    } catch (error) {
      this.emit("error", {
        operation: "configureMTLS",
        error: error as Error,
      });
      throw error;
    }
  }

  /**
   * Apply rate limiting
   */
  async applyRateLimit(config: RateLimitConfig): Promise<void> {
    this.logger.info({ config: config.name }, "Applying rate limit");

    try {
      await this.adapter.applyRateLimit(config);

      this.emit("config:applied", {
        resource: "RateLimit",
        name: config.name,
        namespace: config.namespace,
      });
    } catch (error) {
      this.emit("error", {
        operation: "applyRateLimit",
        error: error as Error,
      });
      throw error;
    }
  }

  /**
   * Get service metrics
   */
  async getServiceMetrics(
    service: string,
    namespace: string
  ): Promise<ServiceMetrics> {
    return this.adapter.getServiceMetrics(service, namespace);
  }

  /**
   * Get mesh status
   */
  async getMeshStatus(): Promise<MeshStatus> {
    return this.adapter.getMeshStatus();
  }

  /**
   * Get current configuration
   */
  getConfig(): ServiceMeshConfig {
    return { ...this.config };
  }

  /**
   * Cleanup resources
   */
  async cleanup(): Promise<void> {
    this.logger.info("Cleaning up service mesh manager");

    // Stop all canary monitors
    for (const [name] of this.canaryMonitors) {
      this.stopCanaryProgression(name);
    }

    this.removeAllListeners();
  }

  // Private methods

  private startCanaryProgression(config: CanaryDeployment): void {
    const key = `${config.namespace}/${config.name}`;
    let currentWeight = config.initialWeight;

    const interval = setInterval(async () => {
      try {
        // Get metrics
        const metrics = await this.getCanaryMetrics(
          config.name,
          config.namespace
        );

        // Check if metrics pass threshold
        const metricsPass = this.checkCanaryMetrics(config, metrics);

        if (!metricsPass) {
          // Rollback on metrics failure
          this.logger.warn(
            { name: config.name },
            "Canary metrics failed, rolling back"
          );
          await this.rollbackCanary(config.name, config.namespace);
          return;
        }

        // Increase weight
        currentWeight = Math.min(
          currentWeight + config.stepWeight,
          config.maxWeight
        );

        await this.adapter.updateCanaryWeight(
          config.name,
          config.namespace,
          currentWeight
        );

        this.emit("canary:progress", {
          name: config.name,
          weight: currentWeight,
          metrics,
        });

        // Check if we've reached max weight
        if (currentWeight >= config.maxWeight) {
          this.logger.info({ name: config.name }, "Canary reached max weight");
          this.stopCanaryProgression(config.name);

          // Auto-promote if at 100%
          if (config.maxWeight === 100) {
            await this.promoteCanary(config.name, config.namespace);
          }
        }
      } catch (error) {
        this.logger.error(
          { name: config.name, error },
          "Error during canary progression"
        );
        this.emit("error", {
          operation: "canaryProgression",
          error: error as Error,
        });
      }
    }, this.parseInterval(config.interval));

    this.canaryMonitors.set(key, interval);
  }

  private stopCanaryProgression(name: string): void {
    for (const [key, interval] of this.canaryMonitors) {
      if (key.endsWith(`/${name}`)) {
        clearInterval(interval);
        this.canaryMonitors.delete(key);
      }
    }
  }

  private async getCanaryMetrics(
    name: string,
    namespace: string
  ): Promise<Record<string, number>> {
    const metrics = await this.adapter.getServiceMetrics(
      `${name}-canary`,
      namespace
    );

    return {
      successRate: metrics.successRate,
      latencyP99: metrics.latencyP99,
      requestRate: metrics.requestRate,
    };
  }

  private checkCanaryMetrics(
    config: CanaryDeployment,
    metrics: Record<string, number>
  ): boolean {
    if (!config.metrics) return true;

    for (const metric of config.metrics) {
      const value = metrics[metric.name];
      if (value === undefined) continue;

      switch (metric.operator) {
        case "<":
          if (!(value < metric.threshold)) return false;
          break;
        case ">":
          if (!(value > metric.threshold)) return false;
          break;
        case "<=":
          if (!(value <= metric.threshold)) return false;
          break;
        case ">=":
          if (!(value >= metric.threshold)) return false;
          break;
        case "==":
          if (!(value === metric.threshold)) return false;
          break;
      }
    }

    return true;
  }

  private parseInterval(interval: string): number {
    const match = interval.match(/^(\d+)(s|m|h)$/);
    if (!match) return 60000; // Default 1 minute

    const value = parseInt(match[1], 10);
    const unit = match[2];

    switch (unit) {
      case "s":
        return value * 1000;
      case "m":
        return value * 60000;
      case "h":
        return value * 3600000;
      default:
        return 60000;
    }
  }
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Create a service mesh manager with Istio
 */
export function createIstioManager(
  namespace: string,
  options?: {
    mtls?: { enabled: boolean; mode?: MTLSMode };
    kubeClient?: any;
    logger?: pino.Logger;
  }
): ServiceMeshManager {
  return new ServiceMeshManager({
    config: {
      provider: "istio",
      namespace,
      mtls: options?.mtls
        ? {
            enabled: options.mtls.enabled,
            mode: options.mtls.mode ?? "strict",
          }
        : undefined,
    },
    kubeClient: options?.kubeClient,
    logger: options?.logger,
  });
}

/**
 * Create a service mesh manager with Linkerd
 */
export function createLinkerdManager(
  namespace: string,
  options?: {
    mtls?: { enabled: boolean; mode?: MTLSMode };
    kubeClient?: any;
    logger?: pino.Logger;
  }
): ServiceMeshManager {
  return new ServiceMeshManager({
    config: {
      provider: "linkerd",
      namespace,
      mtls: options?.mtls
        ? {
            enabled: options.mtls.enabled,
            mode: options.mtls.mode ?? "strict",
          }
        : undefined,
    },
    kubeClient: options?.kubeClient,
    logger: options?.logger,
  });
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Generate a traffic route for A/B testing
 */
export function createABTestRoute(options: {
  name: string;
  service: string;
  namespace: string;
  port: number;
  versionA: { host: string; weight: number };
  versionB: { host: string; weight: number };
  headerMatch?: { name: string; value: string };
}): TrafficRoute {
  return {
    name: options.name,
    service: options.service,
    namespace: options.namespace,
    port: options.port,
    routes: [
      {
        match: options.headerMatch
          ? {
              headers: {
                [options.headerMatch.name]: options.headerMatch.value,
              },
            }
          : undefined,
        destinations: [
          {
            host: options.versionA.host,
            port: options.port,
            weight: options.versionA.weight,
          },
          {
            host: options.versionB.host,
            port: options.port,
            weight: options.versionB.weight,
          },
        ],
      },
    ],
  };
}

/**
 * Generate circuit breaker configuration from service requirements
 */
export function createCircuitBreakerFromRequirements(options: {
  name: string;
  service: string;
  namespace: string;
  maxRequestsPerSecond: number;
  errorThreshold: number;
}): CircuitBreaker {
  return {
    name: options.name,
    service: options.service,
    namespace: options.namespace,
    maxConnections: Math.ceil(options.maxRequestsPerSecond / 10),
    maxPendingRequests: Math.ceil(options.maxRequestsPerSecond / 5),
    maxRequests: options.maxRequestsPerSecond,
    maxRetries: 3,
    consecutiveErrors: options.errorThreshold,
    interval: "10s",
    baseEjectionTime: "30s",
    maxEjectionPercent: 50,
  };
}
