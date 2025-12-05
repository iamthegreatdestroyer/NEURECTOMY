/**
 * NEURECTOMY Helm Chart Generator
 *
 * @FLUX @FORGE - DevOps + Build Systems
 *
 * Generate Helm charts from agent definitions with values templating,
 * dependency management, and multi-environment support.
 */

import * as fs from "fs/promises";
import * as path from "path";
import YAML from "yaml";
import pino from "pino";
import { z } from "zod";

import type {
  HelmChart,
  HelmRelease,
  HelmValues,
  K8sDeploymentConfig,
  K8sServiceConfig,
  K8sHPAConfig,
} from "../types";

// =============================================================================
// Types
// =============================================================================

export interface AgentDefinition {
  name: string;
  version: string;
  description?: string;
  image: string;
  replicas?: number;
  resources?: {
    cpu?: string;
    memory?: string;
    cpuLimit?: string;
    memoryLimit?: string;
  };
  ports?: Array<{
    name: string;
    port: number;
    targetPort?: number;
  }>;
  environment?: Record<string, string>;
  secrets?: string[];
  configMaps?: string[];
  healthCheck?: {
    path: string;
    port?: number;
    initialDelaySeconds?: number;
    periodSeconds?: number;
  };
  autoscaling?: {
    enabled: boolean;
    minReplicas?: number;
    maxReplicas?: number;
    targetCPUUtilization?: number;
    targetMemoryUtilization?: number;
  };
  serviceAccount?: {
    create: boolean;
    name?: string;
    annotations?: Record<string, string>;
  };
  ingress?: {
    enabled: boolean;
    host?: string;
    path?: string;
    tls?: boolean;
    annotations?: Record<string, string>;
  };
  persistence?: {
    enabled: boolean;
    size?: string;
    storageClass?: string;
    accessMode?: "ReadWriteOnce" | "ReadOnlyMany" | "ReadWriteMany";
    mountPath?: string;
  };
  dependencies?: Array<{
    name: string;
    repository: string;
    version: string;
    condition?: string;
  }>;
  podAnnotations?: Record<string, string>;
  nodeSelector?: Record<string, string>;
  tolerations?: Array<{
    key: string;
    operator: string;
    value?: string;
    effect: string;
  }>;
  affinity?: {
    nodeAffinity?: object;
    podAffinity?: object;
    podAntiAffinity?: object;
  };
}

export interface ChartGeneratorOptions {
  outputDir: string;
  chartVersion?: string;
  appVersion?: string;
  namespace?: string;
  createNamespace?: boolean;
  includeTests?: boolean;
  includeNotes?: boolean;
  additionalTemplates?: Array<{
    name: string;
    content: string;
  }>;
}

export interface ValuesOverride {
  environment: string;
  values: Record<string, unknown>;
}

// =============================================================================
// Validation Schemas
// =============================================================================

const AgentDefinitionSchema = z.object({
  name: z
    .string()
    .min(1)
    .regex(/^[a-z][a-z0-9-]*$/),
  version: z.string().min(1),
  description: z.string().optional(),
  image: z.string().min(1),
  replicas: z.number().int().min(0).optional(),
  resources: z
    .object({
      cpu: z.string().optional(),
      memory: z.string().optional(),
      cpuLimit: z.string().optional(),
      memoryLimit: z.string().optional(),
    })
    .optional(),
  ports: z
    .array(
      z.object({
        name: z.string(),
        port: z.number().int().min(1).max(65535),
        targetPort: z.number().int().min(1).max(65535).optional(),
      })
    )
    .optional(),
  environment: z.record(z.string()).optional(),
  secrets: z.array(z.string()).optional(),
  configMaps: z.array(z.string()).optional(),
  healthCheck: z
    .object({
      path: z.string(),
      port: z.number().optional(),
      initialDelaySeconds: z.number().optional(),
      periodSeconds: z.number().optional(),
    })
    .optional(),
  autoscaling: z
    .object({
      enabled: z.boolean(),
      minReplicas: z.number().int().min(1).optional(),
      maxReplicas: z.number().int().min(1).optional(),
      targetCPUUtilization: z.number().int().min(1).max(100).optional(),
      targetMemoryUtilization: z.number().int().min(1).max(100).optional(),
    })
    .optional(),
  serviceAccount: z
    .object({
      create: z.boolean(),
      name: z.string().optional(),
      annotations: z.record(z.string()).optional(),
    })
    .optional(),
  ingress: z
    .object({
      enabled: z.boolean(),
      host: z.string().optional(),
      path: z.string().optional(),
      tls: z.boolean().optional(),
      annotations: z.record(z.string()).optional(),
    })
    .optional(),
  persistence: z
    .object({
      enabled: z.boolean(),
      size: z.string().optional(),
      storageClass: z.string().optional(),
      accessMode: z
        .enum(["ReadWriteOnce", "ReadOnlyMany", "ReadWriteMany"])
        .optional(),
      mountPath: z.string().optional(),
    })
    .optional(),
  dependencies: z
    .array(
      z.object({
        name: z.string(),
        repository: z.string(),
        version: z.string(),
        condition: z.string().optional(),
      })
    )
    .optional(),
  podAnnotations: z.record(z.string()).optional(),
  nodeSelector: z.record(z.string()).optional(),
  tolerations: z
    .array(
      z.object({
        key: z.string(),
        operator: z.string(),
        value: z.string().optional(),
        effect: z.string(),
      })
    )
    .optional(),
  affinity: z
    .object({
      nodeAffinity: z.object({}).passthrough().optional(),
      podAffinity: z.object({}).passthrough().optional(),
      podAntiAffinity: z.object({}).passthrough().optional(),
    })
    .optional(),
});

// =============================================================================
// Helm Chart Generator
// =============================================================================

export class HelmChartGenerator {
  private logger: pino.Logger;

  constructor() {
    this.logger = pino({
      name: "helm-generator",
      level: process.env.LOG_LEVEL || "info",
    });
  }

  // ===========================================================================
  // Main Generation Methods
  // ===========================================================================

  async generateChart(
    agent: AgentDefinition,
    options: ChartGeneratorOptions
  ): Promise<string> {
    // Validate agent definition
    const validated = AgentDefinitionSchema.parse(agent);

    this.logger.info(
      { name: agent.name, outputDir: options.outputDir },
      "Generating Helm chart"
    );

    const chartDir = path.join(options.outputDir, agent.name);

    // Create directory structure
    await this.createChartStructure(chartDir);

    // Generate files
    await Promise.all([
      this.generateChartYaml(validated, chartDir, options),
      this.generateValuesYaml(validated, chartDir),
      this.generateDeploymentTemplate(validated, chartDir),
      this.generateServiceTemplate(validated, chartDir),
      this.generateHelpersTemplate(validated, chartDir),
    ]);

    // Optional templates
    if (validated.autoscaling?.enabled) {
      await this.generateHPATemplate(validated, chartDir);
    }

    if (validated.serviceAccount?.create) {
      await this.generateServiceAccountTemplate(validated, chartDir);
    }

    if (validated.ingress?.enabled) {
      await this.generateIngressTemplate(validated, chartDir);
    }

    if (validated.persistence?.enabled) {
      await this.generatePVCTemplate(validated, chartDir);
    }

    if (validated.secrets && validated.secrets.length > 0) {
      await this.generateSecretTemplate(validated, chartDir);
    }

    if (validated.configMaps && validated.configMaps.length > 0) {
      await this.generateConfigMapTemplate(validated, chartDir);
    }

    if (options.includeTests) {
      await this.generateTestConnection(validated, chartDir);
    }

    if (options.includeNotes) {
      await this.generateNotesTemplate(validated, chartDir);
    }

    // Additional custom templates
    if (options.additionalTemplates) {
      for (const template of options.additionalTemplates) {
        await fs.writeFile(
          path.join(chartDir, "templates", template.name),
          template.content
        );
      }
    }

    this.logger.info({ chartDir }, "Helm chart generated successfully");
    return chartDir;
  }

  async generateMultiAgentChart(
    agents: AgentDefinition[],
    chartName: string,
    options: ChartGeneratorOptions
  ): Promise<string> {
    this.logger.info(
      { chartName, agentCount: agents.length },
      "Generating multi-agent umbrella chart"
    );

    const chartDir = path.join(options.outputDir, chartName);
    await this.createChartStructure(chartDir);
    await fs.mkdir(path.join(chartDir, "charts"), { recursive: true });

    // Generate sub-charts for each agent
    const subChartOptions = {
      ...options,
      outputDir: path.join(chartDir, "charts"),
      includeTests: false,
      includeNotes: false,
    };

    for (const agent of agents) {
      await this.generateChart(agent, subChartOptions);
    }

    // Generate umbrella Chart.yaml with dependencies
    const umbrellaChart = {
      apiVersion: "v2",
      name: chartName,
      description: "NEURECTOMY multi-agent deployment chart",
      type: "application",
      version: options.chartVersion || "0.1.0",
      appVersion: options.appVersion || "1.0.0",
      dependencies: agents.map((agent) => ({
        name: agent.name,
        version: agent.version,
        repository: `file://charts/${agent.name}`,
        condition: `${agent.name}.enabled`,
      })),
    };

    await fs.writeFile(
      path.join(chartDir, "Chart.yaml"),
      YAML.stringify(umbrellaChart)
    );

    // Generate umbrella values.yaml
    const umbrellaValues: Record<string, unknown> = {
      global: {
        imagePullSecrets: [],
        storageClass: "",
      },
    };

    for (const agent of agents) {
      umbrellaValues[agent.name] = {
        enabled: true,
      };
    }

    await fs.writeFile(
      path.join(chartDir, "values.yaml"),
      YAML.stringify(umbrellaValues)
    );

    // Generate README
    await this.generateUmbrellaReadme(chartName, agents, chartDir);

    this.logger.info({ chartDir }, "Multi-agent chart generated successfully");
    return chartDir;
  }

  // ===========================================================================
  // Values Override Generation
  // ===========================================================================

  async generateEnvironmentOverrides(
    agent: AgentDefinition,
    overrides: ValuesOverride[],
    outputDir: string
  ): Promise<void> {
    this.logger.info(
      { agent: agent.name, environments: overrides.map((o) => o.environment) },
      "Generating environment value overrides"
    );

    for (const override of overrides) {
      const filename = `values-${override.environment}.yaml`;
      const filePath = path.join(outputDir, agent.name, filename);

      await fs.writeFile(filePath, YAML.stringify(override.values));
      this.logger.debug({ filePath }, "Generated override file");
    }
  }

  // ===========================================================================
  // Directory Structure
  // ===========================================================================

  private async createChartStructure(chartDir: string): Promise<void> {
    await fs.mkdir(chartDir, { recursive: true });
    await fs.mkdir(path.join(chartDir, "templates"), { recursive: true });
    await fs.mkdir(path.join(chartDir, "templates", "tests"), {
      recursive: true,
    });
  }

  // ===========================================================================
  // Template Generators
  // ===========================================================================

  private async generateChartYaml(
    agent: AgentDefinition,
    chartDir: string,
    options: ChartGeneratorOptions
  ): Promise<void> {
    const chart: HelmChart = {
      apiVersion: "v2",
      name: agent.name,
      description: agent.description || `Helm chart for ${agent.name} agent`,
      type: "application",
      version: options.chartVersion || agent.version,
      appVersion: options.appVersion || agent.version,
      keywords: ["neurectomy", "agent", agent.name],
      home: "https://github.com/neurectomy/neurectomy",
      maintainers: [
        {
          name: "NEURECTOMY Team",
          email: "team@neurectomy.dev",
        },
      ],
    };

    if (agent.dependencies && agent.dependencies.length > 0) {
      chart.dependencies = agent.dependencies;
    }

    await fs.writeFile(
      path.join(chartDir, "Chart.yaml"),
      YAML.stringify(chart)
    );
  }

  private async generateValuesYaml(
    agent: AgentDefinition,
    chartDir: string
  ): Promise<void> {
    const [imageName, imageTag] = this.parseImage(agent.image);

    const values: HelmValues = {
      replicaCount: agent.replicas ?? 1,

      image: {
        repository: imageName,
        pullPolicy: "IfNotPresent",
        tag: imageTag || "latest",
      },

      imagePullSecrets: [],
      nameOverride: "",
      fullnameOverride: "",

      serviceAccount: {
        create: agent.serviceAccount?.create ?? true,
        automount: true,
        annotations: agent.serviceAccount?.annotations || {},
        name: agent.serviceAccount?.name || "",
      },

      podAnnotations: agent.podAnnotations || {},
      podLabels: {},

      podSecurityContext: {},
      securityContext: {},

      service: {
        type: "ClusterIP",
        port: agent.ports?.[0]?.port || 80,
        ports: agent.ports || [],
      },

      ingress: {
        enabled: agent.ingress?.enabled ?? false,
        className: "",
        annotations: agent.ingress?.annotations || {},
        hosts: agent.ingress?.host
          ? [
              {
                host: agent.ingress.host,
                paths: [
                  {
                    path: agent.ingress.path || "/",
                    pathType: "Prefix",
                  },
                ],
              },
            ]
          : [],
        tls: agent.ingress?.tls
          ? [
              {
                secretName: `${agent.name}-tls`,
                hosts: [agent.ingress.host!],
              },
            ]
          : [],
      },

      resources: {
        limits: {
          cpu: agent.resources?.cpuLimit || "500m",
          memory: agent.resources?.memoryLimit || "512Mi",
        },
        requests: {
          cpu: agent.resources?.cpu || "100m",
          memory: agent.resources?.memory || "128Mi",
        },
      },

      livenessProbe: agent.healthCheck
        ? {
            httpGet: {
              path: agent.healthCheck.path,
              port: agent.healthCheck.port || agent.ports?.[0]?.port || 80,
            },
            initialDelaySeconds: agent.healthCheck.initialDelaySeconds || 30,
            periodSeconds: agent.healthCheck.periodSeconds || 10,
          }
        : undefined,

      readinessProbe: agent.healthCheck
        ? {
            httpGet: {
              path: agent.healthCheck.path,
              port: agent.healthCheck.port || agent.ports?.[0]?.port || 80,
            },
            initialDelaySeconds: agent.healthCheck.initialDelaySeconds || 5,
            periodSeconds: agent.healthCheck.periodSeconds || 10,
          }
        : undefined,

      autoscaling: {
        enabled: agent.autoscaling?.enabled ?? false,
        minReplicas: agent.autoscaling?.minReplicas || 1,
        maxReplicas: agent.autoscaling?.maxReplicas || 10,
        targetCPUUtilizationPercentage:
          agent.autoscaling?.targetCPUUtilization || 80,
        targetMemoryUtilizationPercentage:
          agent.autoscaling?.targetMemoryUtilization,
      },

      volumes: [],
      volumeMounts: [],

      nodeSelector: agent.nodeSelector || {},
      tolerations: agent.tolerations || [],
      affinity: agent.affinity || {},

      env: agent.environment || {},
      envFrom: [],

      persistence: {
        enabled: agent.persistence?.enabled ?? false,
        size: agent.persistence?.size || "10Gi",
        storageClass: agent.persistence?.storageClass || "",
        accessMode: agent.persistence?.accessMode || "ReadWriteOnce",
        mountPath: agent.persistence?.mountPath || "/data",
      },
    };

    await fs.writeFile(
      path.join(chartDir, "values.yaml"),
      YAML.stringify(values)
    );
  }

  private async generateHelpersTemplate(
    agent: AgentDefinition,
    chartDir: string
  ): Promise<void> {
    const helpers = `{{/*
Expand the name of the chart.
*/}}
{{- define "${agent.name}.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "${agent.name}.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "${agent.name}.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "${agent.name}.labels" -}}
helm.sh/chart: {{ include "${agent.name}.chart" . }}
{{ include "${agent.name}.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
neurectomy.io/component: agent
{{- end }}

{{/*
Selector labels
*/}}
{{- define "${agent.name}.selectorLabels" -}}
app.kubernetes.io/name: {{ include "${agent.name}.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "${agent.name}.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "${agent.name}.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Return the proper image name
*/}}
{{- define "${agent.name}.image" -}}
{{- $registryName := .Values.image.registry | default "" -}}
{{- $repositoryName := .Values.image.repository -}}
{{- $tag := .Values.image.tag | default .Chart.AppVersion -}}
{{- if $registryName }}
{{- printf "%s/%s:%s" $registryName $repositoryName $tag -}}
{{- else }}
{{- printf "%s:%s" $repositoryName $tag -}}
{{- end }}
{{- end }}
`;

    await fs.writeFile(
      path.join(chartDir, "templates", "_helpers.tpl"),
      helpers
    );
  }

  private async generateDeploymentTemplate(
    agent: AgentDefinition,
    chartDir: string
  ): Promise<void> {
    const deployment = `apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "${agent.name}.fullname" . }}
  labels:
    {{- include "${agent.name}.labels" . | nindent 4 }}
spec:
  {{- if not .Values.autoscaling.enabled }}
  replicas: {{ .Values.replicaCount }}
  {{- end }}
  selector:
    matchLabels:
      {{- include "${agent.name}.selectorLabels" . | nindent 6 }}
  template:
    metadata:
      {{- with .Values.podAnnotations }}
      annotations:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      labels:
        {{- include "${agent.name}.labels" . | nindent 8 }}
        {{- with .Values.podLabels }}
        {{- toYaml . | nindent 8 }}
        {{- end }}
    spec:
      {{- with .Values.imagePullSecrets }}
      imagePullSecrets:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      serviceAccountName: {{ include "${agent.name}.serviceAccountName" . }}
      securityContext:
        {{- toYaml .Values.podSecurityContext | nindent 8 }}
      containers:
        - name: {{ .Chart.Name }}
          securityContext:
            {{- toYaml .Values.securityContext | nindent 12 }}
          image: {{ include "${agent.name}.image" . }}
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          {{- if .Values.service.ports }}
          ports:
            {{- range .Values.service.ports }}
            - name: {{ .name }}
              containerPort: {{ .targetPort | default .port }}
              protocol: {{ .protocol | default "TCP" }}
            {{- end }}
          {{- else }}
          ports:
            - name: http
              containerPort: {{ .Values.service.port }}
              protocol: TCP
          {{- end }}
          {{- if .Values.livenessProbe }}
          livenessProbe:
            {{- toYaml .Values.livenessProbe | nindent 12 }}
          {{- end }}
          {{- if .Values.readinessProbe }}
          readinessProbe:
            {{- toYaml .Values.readinessProbe | nindent 12 }}
          {{- end }}
          resources:
            {{- toYaml .Values.resources | nindent 12 }}
          {{- if or .Values.env .Values.envFrom }}
          {{- if .Values.env }}
          env:
            {{- range $key, $value := .Values.env }}
            - name: {{ $key }}
              value: {{ $value | quote }}
            {{- end }}
          {{- end }}
          {{- if .Values.envFrom }}
          envFrom:
            {{- toYaml .Values.envFrom | nindent 12 }}
          {{- end }}
          {{- end }}
          {{- if or .Values.volumeMounts .Values.persistence.enabled }}
          volumeMounts:
            {{- if .Values.persistence.enabled }}
            - name: data
              mountPath: {{ .Values.persistence.mountPath }}
            {{- end }}
            {{- with .Values.volumeMounts }}
            {{- toYaml . | nindent 12 }}
            {{- end }}
          {{- end }}
      {{- if or .Values.volumes .Values.persistence.enabled }}
      volumes:
        {{- if .Values.persistence.enabled }}
        - name: data
          persistentVolumeClaim:
            claimName: {{ include "${agent.name}.fullname" . }}
        {{- end }}
        {{- with .Values.volumes }}
        {{- toYaml . | nindent 8 }}
        {{- end }}
      {{- end }}
      {{- with .Values.nodeSelector }}
      nodeSelector:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.affinity }}
      affinity:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.tolerations }}
      tolerations:
        {{- toYaml . | nindent 8 }}
      {{- end }}
`;

    await fs.writeFile(
      path.join(chartDir, "templates", "deployment.yaml"),
      deployment
    );
  }

  private async generateServiceTemplate(
    agent: AgentDefinition,
    chartDir: string
  ): Promise<void> {
    const service = `apiVersion: v1
kind: Service
metadata:
  name: {{ include "${agent.name}.fullname" . }}
  labels:
    {{- include "${agent.name}.labels" . | nindent 4 }}
spec:
  type: {{ .Values.service.type }}
  ports:
    {{- if .Values.service.ports }}
    {{- range .Values.service.ports }}
    - port: {{ .port }}
      targetPort: {{ .targetPort | default .port }}
      protocol: {{ .protocol | default "TCP" }}
      name: {{ .name }}
    {{- end }}
    {{- else }}
    - port: {{ .Values.service.port }}
      targetPort: http
      protocol: TCP
      name: http
    {{- end }}
  selector:
    {{- include "${agent.name}.selectorLabels" . | nindent 4 }}
`;

    await fs.writeFile(
      path.join(chartDir, "templates", "service.yaml"),
      service
    );
  }

  private async generateHPATemplate(
    agent: AgentDefinition,
    chartDir: string
  ): Promise<void> {
    const hpa = `{{- if .Values.autoscaling.enabled }}
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {{ include "${agent.name}.fullname" . }}
  labels:
    {{- include "${agent.name}.labels" . | nindent 4 }}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {{ include "${agent.name}.fullname" . }}
  minReplicas: {{ .Values.autoscaling.minReplicas }}
  maxReplicas: {{ .Values.autoscaling.maxReplicas }}
  metrics:
    {{- if .Values.autoscaling.targetCPUUtilizationPercentage }}
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: {{ .Values.autoscaling.targetCPUUtilizationPercentage }}
    {{- end }}
    {{- if .Values.autoscaling.targetMemoryUtilizationPercentage }}
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: {{ .Values.autoscaling.targetMemoryUtilizationPercentage }}
    {{- end }}
{{- end }}
`;

    await fs.writeFile(path.join(chartDir, "templates", "hpa.yaml"), hpa);
  }

  private async generateServiceAccountTemplate(
    agent: AgentDefinition,
    chartDir: string
  ): Promise<void> {
    const sa = `{{- if .Values.serviceAccount.create -}}
apiVersion: v1
kind: ServiceAccount
metadata:
  name: {{ include "${agent.name}.serviceAccountName" . }}
  labels:
    {{- include "${agent.name}.labels" . | nindent 4 }}
  {{- with .Values.serviceAccount.annotations }}
  annotations:
    {{- toYaml . | nindent 4 }}
  {{- end }}
automountServiceAccountToken: {{ .Values.serviceAccount.automount }}
{{- end }}
`;

    await fs.writeFile(
      path.join(chartDir, "templates", "serviceaccount.yaml"),
      sa
    );
  }

  private async generateIngressTemplate(
    agent: AgentDefinition,
    chartDir: string
  ): Promise<void> {
    const ingress = `{{- if .Values.ingress.enabled -}}
{{- $fullName := include "${agent.name}.fullname" . -}}
{{- $svcPort := .Values.service.port -}}
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: {{ $fullName }}
  labels:
    {{- include "${agent.name}.labels" . | nindent 4 }}
  {{- with .Values.ingress.annotations }}
  annotations:
    {{- toYaml . | nindent 4 }}
  {{- end }}
spec:
  {{- if .Values.ingress.className }}
  ingressClassName: {{ .Values.ingress.className }}
  {{- end }}
  {{- if .Values.ingress.tls }}
  tls:
    {{- range .Values.ingress.tls }}
    - hosts:
        {{- range .hosts }}
        - {{ . | quote }}
        {{- end }}
      secretName: {{ .secretName }}
    {{- end }}
  {{- end }}
  rules:
    {{- range .Values.ingress.hosts }}
    - host: {{ .host | quote }}
      http:
        paths:
          {{- range .paths }}
          - path: {{ .path }}
            pathType: {{ .pathType }}
            backend:
              service:
                name: {{ $fullName }}
                port:
                  number: {{ $svcPort }}
          {{- end }}
    {{- end }}
{{- end }}
`;

    await fs.writeFile(
      path.join(chartDir, "templates", "ingress.yaml"),
      ingress
    );
  }

  private async generatePVCTemplate(
    agent: AgentDefinition,
    chartDir: string
  ): Promise<void> {
    const pvc = `{{- if .Values.persistence.enabled }}
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: {{ include "${agent.name}.fullname" . }}
  labels:
    {{- include "${agent.name}.labels" . | nindent 4 }}
spec:
  accessModes:
    - {{ .Values.persistence.accessMode }}
  {{- if .Values.persistence.storageClass }}
  storageClassName: {{ .Values.persistence.storageClass | quote }}
  {{- end }}
  resources:
    requests:
      storage: {{ .Values.persistence.size }}
{{- end }}
`;

    await fs.writeFile(path.join(chartDir, "templates", "pvc.yaml"), pvc);
  }

  private async generateSecretTemplate(
    agent: AgentDefinition,
    chartDir: string
  ): Promise<void> {
    const secret = `{{- if .Values.secrets }}
apiVersion: v1
kind: Secret
metadata:
  name: {{ include "${agent.name}.fullname" . }}
  labels:
    {{- include "${agent.name}.labels" . | nindent 4 }}
type: Opaque
data:
  {{- range $key, $value := .Values.secrets }}
  {{ $key }}: {{ $value | b64enc | quote }}
  {{- end }}
{{- end }}
`;

    await fs.writeFile(path.join(chartDir, "templates", "secret.yaml"), secret);
  }

  private async generateConfigMapTemplate(
    agent: AgentDefinition,
    chartDir: string
  ): Promise<void> {
    const configmap = `{{- if .Values.configMaps }}
apiVersion: v1
kind: ConfigMap
metadata:
  name: {{ include "${agent.name}.fullname" . }}
  labels:
    {{- include "${agent.name}.labels" . | nindent 4 }}
data:
  {{- range $key, $value := .Values.configMaps }}
  {{ $key }}: {{ $value | quote }}
  {{- end }}
{{- end }}
`;

    await fs.writeFile(
      path.join(chartDir, "templates", "configmap.yaml"),
      configmap
    );
  }

  private async generateTestConnection(
    agent: AgentDefinition,
    chartDir: string
  ): Promise<void> {
    const test = `apiVersion: v1
kind: Pod
metadata:
  name: "{{ include "${agent.name}.fullname" . }}-test-connection"
  labels:
    {{- include "${agent.name}.labels" . | nindent 4 }}
  annotations:
    "helm.sh/hook": test
spec:
  containers:
    - name: wget
      image: busybox
      command: ['wget']
      args: ['{{ include "${agent.name}.fullname" . }}:{{ .Values.service.port }}']
  restartPolicy: Never
`;

    await fs.writeFile(
      path.join(chartDir, "templates", "tests", "test-connection.yaml"),
      test
    );
  }

  private async generateNotesTemplate(
    agent: AgentDefinition,
    chartDir: string
  ): Promise<void> {
    const notes = `NEURECTOMY ${agent.name} Agent
=====================================

1. Get the application URL by running these commands:
{{- if .Values.ingress.enabled }}
{{- range $host := .Values.ingress.hosts }}
  {{- range .paths }}
  http{{ if $.Values.ingress.tls }}s{{ end }}://{{ $host.host }}{{ .path }}
  {{- end }}
{{- end }}
{{- else if contains "NodePort" .Values.service.type }}
  export NODE_PORT=$(kubectl get --namespace {{ .Release.Namespace }} -o jsonpath="{.spec.ports[0].nodePort}" services {{ include "${agent.name}.fullname" . }})
  export NODE_IP=$(kubectl get nodes --namespace {{ .Release.Namespace }} -o jsonpath="{.items[0].status.addresses[0].address}")
  echo http://$NODE_IP:$NODE_PORT
{{- else if contains "LoadBalancer" .Values.service.type }}
     NOTE: It may take a few minutes for the LoadBalancer IP to be available.
           You can watch its status by running 'kubectl get --namespace {{ .Release.Namespace }} svc -w {{ include "${agent.name}.fullname" . }}'
  export SERVICE_IP=$(kubectl get svc --namespace {{ .Release.Namespace }} {{ include "${agent.name}.fullname" . }} --template "{{"{{ range (index .status.loadBalancer.ingress 0) }}{{.}}{{ end }}"}}")
  echo http://$SERVICE_IP:{{ .Values.service.port }}
{{- else if contains "ClusterIP" .Values.service.type }}
  export POD_NAME=$(kubectl get pods --namespace {{ .Release.Namespace }} -l "app.kubernetes.io/name={{ include "${agent.name}.name" . }},app.kubernetes.io/instance={{ .Release.Name }}" -o jsonpath="{.items[0].metadata.name}")
  export CONTAINER_PORT=$(kubectl get pod --namespace {{ .Release.Namespace }} $POD_NAME -o jsonpath="{.spec.containers[0].ports[0].containerPort}")
  echo "Visit http://127.0.0.1:8080 to use your application"
  kubectl --namespace {{ .Release.Namespace }} port-forward $POD_NAME 8080:$CONTAINER_PORT
{{- end }}

2. Check the agent status:
   kubectl get pods -n {{ .Release.Namespace }} -l app.kubernetes.io/name={{ include "${agent.name}.name" . }}

3. View logs:
   kubectl logs -n {{ .Release.Namespace }} -l app.kubernetes.io/name={{ include "${agent.name}.name" . }} -f
`;

    await fs.writeFile(path.join(chartDir, "templates", "NOTES.txt"), notes);
  }

  private async generateUmbrellaReadme(
    chartName: string,
    agents: AgentDefinition[],
    chartDir: string
  ): Promise<void> {
    const readme = `# ${chartName}

NEURECTOMY Multi-Agent Deployment Chart

## Description

This umbrella chart deploys the following agents:

${agents.map((a) => `- **${a.name}**: ${a.description || "No description"}`).join("\n")}

## Installation

\`\`\`bash
# Add dependencies
helm dependency update

# Install chart
helm install ${chartName} ./${chartName} -n neurectomy --create-namespace
\`\`\`

## Configuration

Each agent can be configured independently under its own key in values.yaml.

### Global Values

| Parameter | Description | Default |
|-----------|-------------|---------|
| \`global.imagePullSecrets\` | Image pull secrets | \`[]\` |
| \`global.storageClass\` | Default storage class | \`""\` |

### Agent-Specific Values

${agents
  .map(
    (a) => `
#### ${a.name}

| Parameter | Description | Default |
|-----------|-------------|---------|
| \`${a.name}.enabled\` | Enable ${a.name} agent | \`true\` |
`
  )
  .join("\n")}

## Uninstallation

\`\`\`bash
helm uninstall ${chartName} -n neurectomy
\`\`\`
`;

    await fs.writeFile(path.join(chartDir, "README.md"), readme);
  }

  // ===========================================================================
  // Utility Methods
  // ===========================================================================

  private parseImage(image: string): [string, string | undefined] {
    const parts = image.split(":");
    return [parts[0], parts[1]];
  }

  /**
   * Validate a Helm chart directory
   */
  async validateChart(chartDir: string): Promise<{
    valid: boolean;
    errors: string[];
    warnings: string[];
  }> {
    const errors: string[] = [];
    const warnings: string[] = [];

    // Check required files
    const requiredFiles = ["Chart.yaml", "values.yaml"];
    for (const file of requiredFiles) {
      try {
        await fs.access(path.join(chartDir, file));
      } catch {
        errors.push(`Missing required file: ${file}`);
      }
    }

    // Check templates directory
    try {
      await fs.access(path.join(chartDir, "templates"));
    } catch {
      errors.push("Missing templates directory");
    }

    // Validate Chart.yaml
    try {
      const chartYaml = await fs.readFile(
        path.join(chartDir, "Chart.yaml"),
        "utf-8"
      );
      const chart = YAML.parse(chartYaml);

      if (!chart.apiVersion) {
        errors.push("Chart.yaml missing apiVersion");
      }
      if (!chart.name) {
        errors.push("Chart.yaml missing name");
      }
      if (!chart.version) {
        errors.push("Chart.yaml missing version");
      }
    } catch (e) {
      errors.push(`Invalid Chart.yaml: ${e}`);
    }

    // Validate values.yaml
    try {
      const valuesYaml = await fs.readFile(
        path.join(chartDir, "values.yaml"),
        "utf-8"
      );
      YAML.parse(valuesYaml);
    } catch (e) {
      errors.push(`Invalid values.yaml: ${e}`);
    }

    return {
      valid: errors.length === 0,
      errors,
      warnings,
    };
  }
}

// =============================================================================
// Convenience Export
// =============================================================================

export const helmGenerator = new HelmChartGenerator();
