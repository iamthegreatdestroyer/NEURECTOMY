/**
 * @fileoverview GitHub Webhooks Manager - Comprehensive webhook management
 * @module @neurectomy/github-universe/webhooks
 *
 * @description Provides functionality for managing GitHub webhooks including
 * creating, updating, and deleting webhooks, as well as handling webhook
 * event verification and payload processing.
 *
 * @agent @SYNAPSE - Integration Engineering & API Design
 * @agent @APEX - Elite Computer Science Engineering
 *
 * Features:
 * - Repository and organization webhook management
 * - Webhook event verification with HMAC-SHA256
 * - Event payload parsing and validation
 * - Webhook delivery status monitoring
 * - Automatic webhook registration for agents
 *
 * @example
 * ```typescript
 * import { WebhooksManager } from '@neurectomy/github-universe/webhooks';
 *
 * const webhooks = new WebhooksManager(client);
 *
 * // Create a webhook
 * const webhook = await webhooks.createWebhook(
 *   { owner: 'org', repo: 'app' },
 *   {
 *     name: 'web',
 *     events: ['push', 'pull_request'],
 *     config: {
 *       url: 'https://api.example.com/webhooks',
 *       contentType: 'json',
 *       secret: 'webhook-secret',
 *     },
 *   }
 * );
 *
 * // Verify webhook signature
 * const isValid = webhooks.verifySignature(
 *   payload,
 *   signature,
 *   'webhook-secret'
 * );
 * ```
 */

import { EventEmitter } from "eventemitter3";
import * as crypto from "crypto";
import type { GitHubClient } from "../client";
import {
  type RepoIdentifier,
  type Webhook,
  type CreateWebhookConfig,
  type WebhookEvent,
  WebhookSchema,
  CreateWebhookConfigSchema,
} from "../types";

// ============================================================================
// TYPES
// ============================================================================

/**
 * Events emitted by WebhooksManager
 */
export interface WebhooksManagerEvents {
  "webhook:created": (webhook: Webhook) => void;
  "webhook:updated": (webhook: Webhook) => void;
  "webhook:deleted": (webhookId: number, repo: RepoIdentifier) => void;
  "webhook:tested": (webhook: Webhook, success: boolean) => void;
  "webhook:verified": (eventType: string, valid: boolean) => void;
  "delivery:received": (delivery: WebhookDelivery) => void;
  "delivery:failed": (delivery: WebhookDelivery, error: Error) => void;
  "event:received": (event: WebhookPayload) => void;
}

/**
 * Webhook delivery information
 */
export interface WebhookDelivery {
  id: number;
  guid: string;
  deliveredAt: Date;
  redelivery: boolean;
  duration: number;
  status: string;
  statusCode: number;
  event: string;
  action?: string;
  installationId?: number;
  repositoryId?: number;
  request: {
    headers: Record<string, string>;
    payload: unknown;
  };
  response: {
    headers: Record<string, string>;
    payload: string | null;
  };
}

/**
 * Webhook payload received from GitHub
 */
export interface WebhookPayload {
  event: string;
  action?: string;
  sender: {
    id: number;
    login: string;
    type: string;
  };
  repository?: {
    id: number;
    name: string;
    fullName: string;
    owner: {
      id: number;
      login: string;
    };
  };
  organization?: {
    id: number;
    login: string;
  };
  installation?: {
    id: number;
  };
  payload: unknown;
  signature: string;
  deliveryId: string;
  timestamp: Date;
}

/**
 * Webhook update configuration
 */
export interface UpdateWebhookConfig {
  addEvents?: WebhookEvent[];
  removeEvents?: WebhookEvent[];
  config?: {
    url?: string;
    contentType?: "json" | "form";
    secret?: string;
    insecureSsl?: boolean;
  };
  active?: boolean;
}

/**
 * Webhook list options
 */
export interface WebhookListOptions {
  perPage?: number;
  page?: number;
}

/**
 * Organization webhook identifier
 */
export interface OrgIdentifier {
  org: string;
}

/**
 * Webhook registration result
 */
export interface WebhookRegistrationResult {
  webhook: Webhook;
  existed: boolean;
  updated: boolean;
}

// ============================================================================
// CONSTANTS
// ============================================================================

const SIGNATURE_HEADER = "x-hub-signature-256";
const EVENT_HEADER = "x-github-event";
const DELIVERY_HEADER = "x-github-delivery";

// ============================================================================
// WEBHOOKS MANAGER
// ============================================================================

/**
 * Manages GitHub webhooks for repositories and organizations
 *
 * @extends EventEmitter
 *
 * @example
 * ```typescript
 * const webhooks = new WebhooksManager(client);
 *
 * webhooks.on('webhook:created', (webhook) => {
 *   console.log(`Created webhook ${webhook.id} for ${webhook.url}`);
 * });
 *
 * webhooks.on('event:received', (event) => {
 *   console.log(`Received ${event.event} event`);
 * });
 * ```
 */
export class WebhooksManager extends EventEmitter<WebhooksManagerEvents> {
  private client: GitHubClient;

  constructor(client: GitHubClient) {
    super();
    this.client = client;
  }

  // ==========================================================================
  // REPOSITORY WEBHOOKS
  // ==========================================================================

  /**
   * Create a webhook for a repository
   *
   * @param repo - Repository identifier
   * @param config - Webhook configuration
   * @returns Created webhook
   *
   * @example
   * ```typescript
   * const webhook = await webhooks.createWebhook(
   *   { owner: 'org', repo: 'app' },
   *   {
   *     name: 'web',
   *     events: ['push', 'pull_request', 'issues'],
   *     config: {
   *       url: 'https://api.example.com/webhooks/github',
   *       contentType: 'json',
   *       secret: process.env.WEBHOOK_SECRET,
   *     },
   *   }
   * );
   * ```
   */
  async createWebhook(
    repo: RepoIdentifier,
    config: CreateWebhookConfig
  ): Promise<Webhook> {
    const validConfig = CreateWebhookConfigSchema.parse(config);

    const response = await this.client.rest<Record<string, unknown>>(
      "POST /repos/{owner}/{repo}/hooks",
      {
        owner: repo.owner,
        repo: repo.repo,
        name: validConfig.name || "web",
        active: validConfig.active ?? true,
        events: validConfig.events,
        config: {
          url: validConfig.config.url,
          content_type: validConfig.config.contentType || "json",
          secret: validConfig.config.secret,
          insecure_ssl: validConfig.config.insecureSsl ? "1" : "0",
        },
      }
    );

    const webhook = this.mapWebhook(response);
    this.emit("webhook:created", webhook);

    return webhook;
  }

  /**
   * Get a webhook by ID
   *
   * @param repo - Repository identifier
   * @param webhookId - Webhook ID
   * @returns Webhook data
   */
  async getWebhook(repo: RepoIdentifier, webhookId: number): Promise<Webhook> {
    const response = await this.client.rest<Record<string, unknown>>(
      "GET /repos/{owner}/{repo}/hooks/{hook_id}",
      {
        owner: repo.owner,
        repo: repo.repo,
        hook_id: webhookId,
      }
    );

    return this.mapWebhook(response);
  }

  /**
   * List all webhooks for a repository
   *
   * @param repo - Repository identifier
   * @param options - List options
   * @returns Array of webhooks
   *
   * @example
   * ```typescript
   * const webhooks = await webhooks.listWebhooks(
   *   { owner: 'org', repo: 'app' },
   *   { perPage: 30 }
   * );
   * ```
   */
  async listWebhooks(
    repo: RepoIdentifier,
    options: WebhookListOptions = {}
  ): Promise<Webhook[]> {
    const response = await this.client.restPaginate<Record<string, unknown>>(
      "GET /repos/{owner}/{repo}/hooks",
      {
        owner: repo.owner,
        repo: repo.repo,
        per_page: options.perPage || 30,
      }
    );

    return response.map((hook) => this.mapWebhook(hook));
  }

  /**
   * Update a webhook
   *
   * @param repo - Repository identifier
   * @param webhookId - Webhook ID
   * @param updates - Update configuration
   * @returns Updated webhook
   *
   * @example
   * ```typescript
   * const updated = await webhooks.updateWebhook(
   *   { owner: 'org', repo: 'app' },
   *   12345,
   *   {
   *     addEvents: ['issues', 'issue_comment'],
   *     config: {
   *       url: 'https://new-api.example.com/webhooks',
   *     },
   *   }
   * );
   * ```
   */
  async updateWebhook(
    repo: RepoIdentifier,
    webhookId: number,
    updates: UpdateWebhookConfig
  ): Promise<Webhook> {
    // Get current webhook to merge events
    const current = await this.getWebhook(repo, webhookId);

    let events = [...current.events];

    if (updates.addEvents) {
      events = [...new Set([...events, ...updates.addEvents])];
    }

    if (updates.removeEvents) {
      const removeSet = new Set(updates.removeEvents);
      events = events.filter((e) => !removeSet.has(e as WebhookEvent));
    }

    const response = await this.client.rest<Record<string, unknown>>(
      "PATCH /repos/{owner}/{repo}/hooks/{hook_id}",
      {
        owner: repo.owner,
        repo: repo.repo,
        hook_id: webhookId,
        active: updates.active ?? current.active,
        events,
        config: {
          url: updates.config?.url || current.config.url,
          content_type:
            updates.config?.contentType || current.config.contentType,
          secret: updates.config?.secret,
          insecure_ssl:
            updates.config?.insecureSsl !== undefined
              ? updates.config.insecureSsl
                ? "1"
                : "0"
              : current.config.insecureSsl
                ? "1"
                : "0",
        },
      }
    );

    const webhook = this.mapWebhook(response);
    this.emit("webhook:updated", webhook);

    return webhook;
  }

  /**
   * Delete a webhook
   *
   * @param repo - Repository identifier
   * @param webhookId - Webhook ID
   *
   * @example
   * ```typescript
   * await webhooks.deleteWebhook({ owner: 'org', repo: 'app' }, 12345);
   * ```
   */
  async deleteWebhook(repo: RepoIdentifier, webhookId: number): Promise<void> {
    await this.client.rest("DELETE /repos/{owner}/{repo}/hooks/{hook_id}", {
      owner: repo.owner,
      repo: repo.repo,
      hook_id: webhookId,
    });

    this.emit("webhook:deleted", webhookId, repo);
  }

  /**
   * Test (ping) a webhook
   *
   * @param repo - Repository identifier
   * @param webhookId - Webhook ID
   * @returns True if ping was successful
   *
   * @example
   * ```typescript
   * const success = await webhooks.pingWebhook(
   *   { owner: 'org', repo: 'app' },
   *   12345
   * );
   * ```
   */
  async pingWebhook(repo: RepoIdentifier, webhookId: number): Promise<boolean> {
    try {
      await this.client.rest(
        "POST /repos/{owner}/{repo}/hooks/{hook_id}/pings",
        {
          owner: repo.owner,
          repo: repo.repo,
          hook_id: webhookId,
        }
      );

      const webhook = await this.getWebhook(repo, webhookId);
      this.emit("webhook:tested", webhook, true);

      return true;
    } catch (error) {
      const webhook = await this.getWebhook(repo, webhookId);
      this.emit("webhook:tested", webhook, false);

      return false;
    }
  }

  /**
   * Get recent deliveries for a webhook
   *
   * @param repo - Repository identifier
   * @param webhookId - Webhook ID
   * @param options - List options
   * @returns Array of deliveries
   *
   * @example
   * ```typescript
   * const deliveries = await webhooks.listDeliveries(
   *   { owner: 'org', repo: 'app' },
   *   12345,
   *   { perPage: 10 }
   * );
   * ```
   */
  async listDeliveries(
    repo: RepoIdentifier,
    webhookId: number,
    options: WebhookListOptions = {}
  ): Promise<WebhookDelivery[]> {
    const response = await this.client.rest<{
      data: Record<string, unknown>[];
    }>("GET /repos/{owner}/{repo}/hooks/{hook_id}/deliveries", {
      owner: repo.owner,
      repo: repo.repo,
      hook_id: webhookId,
      per_page: options.perPage || 30,
    });

    return (
      response.data || (response as unknown as Record<string, unknown>[])
    ).map((delivery) => this.mapDelivery(delivery));
  }

  /**
   * Get a specific delivery
   *
   * @param repo - Repository identifier
   * @param webhookId - Webhook ID
   * @param deliveryId - Delivery ID
   * @returns Delivery data
   */
  async getDelivery(
    repo: RepoIdentifier,
    webhookId: number,
    deliveryId: number
  ): Promise<WebhookDelivery> {
    const response = await this.client.rest<Record<string, unknown>>(
      "GET /repos/{owner}/{repo}/hooks/{hook_id}/deliveries/{delivery_id}",
      {
        owner: repo.owner,
        repo: repo.repo,
        hook_id: webhookId,
        delivery_id: deliveryId,
      }
    );

    return this.mapDelivery(response);
  }

  /**
   * Redeliver a webhook
   *
   * @param repo - Repository identifier
   * @param webhookId - Webhook ID
   * @param deliveryId - Delivery ID to redeliver
   *
   * @example
   * ```typescript
   * await webhooks.redeliverWebhook(
   *   { owner: 'org', repo: 'app' },
   *   12345,
   *   67890
   * );
   * ```
   */
  async redeliverWebhook(
    repo: RepoIdentifier,
    webhookId: number,
    deliveryId: number
  ): Promise<void> {
    await this.client.rest(
      "POST /repos/{owner}/{repo}/hooks/{hook_id}/deliveries/{delivery_id}/attempts",
      {
        owner: repo.owner,
        repo: repo.repo,
        hook_id: webhookId,
        delivery_id: deliveryId,
      }
    );
  }

  // ==========================================================================
  // ORGANIZATION WEBHOOKS
  // ==========================================================================

  /**
   * Create a webhook for an organization
   *
   * @param org - Organization identifier
   * @param config - Webhook configuration
   * @returns Created webhook
   *
   * @example
   * ```typescript
   * const webhook = await webhooks.createOrgWebhook(
   *   { org: 'my-org' },
   *   {
   *     name: 'web',
   *     events: ['repository', 'member', 'team'],
   *     config: {
   *       url: 'https://api.example.com/webhooks/org',
   *       contentType: 'json',
   *       secret: process.env.WEBHOOK_SECRET,
   *     },
   *   }
   * );
   * ```
   */
  async createOrgWebhook(
    org: OrgIdentifier,
    config: CreateWebhookConfig
  ): Promise<Webhook> {
    const validConfig = CreateWebhookConfigSchema.parse(config);

    const response = await this.client.rest<Record<string, unknown>>(
      "POST /orgs/{org}/hooks",
      {
        org: org.org,
        name: validConfig.name || "web",
        active: validConfig.active ?? true,
        events: validConfig.events,
        config: {
          url: validConfig.config.url,
          content_type: validConfig.config.contentType || "json",
          secret: validConfig.config.secret,
          insecure_ssl: validConfig.config.insecureSsl ? "1" : "0",
        },
      }
    );

    const webhook = this.mapWebhook(response);
    this.emit("webhook:created", webhook);

    return webhook;
  }

  /**
   * List webhooks for an organization
   *
   * @param org - Organization identifier
   * @param options - List options
   * @returns Array of webhooks
   */
  async listOrgWebhooks(
    org: OrgIdentifier,
    options: WebhookListOptions = {}
  ): Promise<Webhook[]> {
    const response = await this.client.restPaginate<Record<string, unknown>>(
      "GET /orgs/{org}/hooks",
      {
        org: org.org,
        per_page: options.perPage || 30,
      }
    );

    return response.map((hook) => this.mapWebhook(hook));
  }

  /**
   * Get an organization webhook by ID
   *
   * @param org - Organization identifier
   * @param webhookId - Webhook ID
   * @returns Webhook data
   */
  async getOrgWebhook(org: OrgIdentifier, webhookId: number): Promise<Webhook> {
    const response = await this.client.rest<Record<string, unknown>>(
      "GET /orgs/{org}/hooks/{hook_id}",
      {
        org: org.org,
        hook_id: webhookId,
      }
    );

    return this.mapWebhook(response);
  }

  /**
   * Update an organization webhook
   *
   * @param org - Organization identifier
   * @param webhookId - Webhook ID
   * @param updates - Update configuration
   * @returns Updated webhook
   */
  async updateOrgWebhook(
    org: OrgIdentifier,
    webhookId: number,
    updates: UpdateWebhookConfig
  ): Promise<Webhook> {
    const current = await this.getOrgWebhook(org, webhookId);

    let events = [...current.events];

    if (updates.addEvents) {
      events = [...new Set([...events, ...updates.addEvents])];
    }

    if (updates.removeEvents) {
      const removeSet = new Set(updates.removeEvents);
      events = events.filter((e) => !removeSet.has(e as WebhookEvent));
    }

    const response = await this.client.rest<Record<string, unknown>>(
      "PATCH /orgs/{org}/hooks/{hook_id}",
      {
        org: org.org,
        hook_id: webhookId,
        active: updates.active ?? current.active,
        events,
        config: {
          url: updates.config?.url || current.config.url,
          content_type:
            updates.config?.contentType || current.config.contentType,
          secret: updates.config?.secret,
          insecure_ssl:
            updates.config?.insecureSsl !== undefined
              ? updates.config.insecureSsl
                ? "1"
                : "0"
              : current.config.insecureSsl
                ? "1"
                : "0",
        },
      }
    );

    const webhook = this.mapWebhook(response);
    this.emit("webhook:updated", webhook);

    return webhook;
  }

  /**
   * Delete an organization webhook
   *
   * @param org - Organization identifier
   * @param webhookId - Webhook ID
   */
  async deleteOrgWebhook(org: OrgIdentifier, webhookId: number): Promise<void> {
    await this.client.rest("DELETE /orgs/{org}/hooks/{hook_id}", {
      org: org.org,
      hook_id: webhookId,
    });

    this.emit("webhook:deleted", webhookId, { owner: org.org, repo: "" });
  }

  /**
   * Ping an organization webhook
   *
   * @param org - Organization identifier
   * @param webhookId - Webhook ID
   * @returns True if ping successful
   */
  async pingOrgWebhook(
    org: OrgIdentifier,
    webhookId: number
  ): Promise<boolean> {
    try {
      await this.client.rest("POST /orgs/{org}/hooks/{hook_id}/pings", {
        org: org.org,
        hook_id: webhookId,
      });

      const webhook = await this.getOrgWebhook(org, webhookId);
      this.emit("webhook:tested", webhook, true);

      return true;
    } catch (error) {
      const webhook = await this.getOrgWebhook(org, webhookId);
      this.emit("webhook:tested", webhook, false);

      return false;
    }
  }

  // ==========================================================================
  // WEBHOOK VERIFICATION & PROCESSING
  // ==========================================================================

  /**
   * Verify a webhook signature using HMAC-SHA256
   *
   * @param payload - Raw request body
   * @param signature - Signature from x-hub-signature-256 header
   * @param secret - Webhook secret
   * @returns True if signature is valid
   *
   * @example
   * ```typescript
   * // Express middleware
   * app.post('/webhooks/github', (req, res) => {
   *   const signature = req.headers['x-hub-signature-256'];
   *   const isValid = webhooks.verifySignature(
   *     req.body,
   *     signature,
   *     process.env.WEBHOOK_SECRET
   *   );
   *
   *   if (!isValid) {
   *     return res.status(401).send('Invalid signature');
   *   }
   *
   *   // Process webhook...
   * });
   * ```
   */
  verifySignature(
    payload: string | Buffer,
    signature: string,
    secret: string
  ): boolean {
    if (!signature || !signature.startsWith("sha256=")) {
      this.emit("webhook:verified", "unknown", false);
      return false;
    }

    const expected =
      "sha256=" +
      crypto.createHmac("sha256", secret).update(payload).digest("hex");

    // Use timing-safe comparison to prevent timing attacks
    const sigBuffer = Buffer.from(signature);
    const expectedBuffer = Buffer.from(expected);

    if (sigBuffer.length !== expectedBuffer.length) {
      this.emit("webhook:verified", "unknown", false);
      return false;
    }

    const isValid = crypto.timingSafeEqual(sigBuffer, expectedBuffer);
    this.emit("webhook:verified", "unknown", isValid);

    return isValid;
  }

  /**
   * Parse a webhook payload from HTTP request
   *
   * @param body - Request body (string or Buffer)
   * @param headers - Request headers
   * @param secret - Webhook secret for verification
   * @returns Parsed webhook payload
   * @throws Error if signature verification fails
   *
   * @example
   * ```typescript
   * // Express middleware
   * app.post('/webhooks/github', express.raw({ type: '*\/*' }), (req, res) => {
   *   try {
   *     const payload = webhooks.parsePayload(
   *       req.body,
   *       req.headers,
   *       process.env.WEBHOOK_SECRET
   *     );
   *
   *     console.log(`Received ${payload.event} event`);
   *     // Handle event...
   *
   *     res.status(200).send('OK');
   *   } catch (error) {
   *     res.status(401).send(error.message);
   *   }
   * });
   * ```
   */
  parsePayload(
    body: string | Buffer,
    headers: Record<string, string | string[] | undefined>,
    secret?: string
  ): WebhookPayload {
    const signature = this.getHeader(headers, SIGNATURE_HEADER);
    const event = this.getHeader(headers, EVENT_HEADER);
    const deliveryId = this.getHeader(headers, DELIVERY_HEADER);

    if (!event) {
      throw new Error("Missing X-GitHub-Event header");
    }

    if (!deliveryId) {
      throw new Error("Missing X-GitHub-Delivery header");
    }

    // Verify signature if secret provided
    if (secret) {
      if (!signature) {
        throw new Error("Missing X-Hub-Signature-256 header");
      }

      const isValid = this.verifySignature(body, signature, secret);
      if (!isValid) {
        throw new Error("Invalid webhook signature");
      }
    }

    // Parse JSON payload
    const payloadStr = typeof body === "string" ? body : body.toString("utf8");
    const parsed = JSON.parse(payloadStr);

    const webhookPayload: WebhookPayload = {
      event,
      action: parsed.action,
      sender: parsed.sender
        ? {
            id: parsed.sender.id,
            login: parsed.sender.login,
            type: parsed.sender.type,
          }
        : { id: 0, login: "unknown", type: "unknown" },
      repository: parsed.repository
        ? {
            id: parsed.repository.id,
            name: parsed.repository.name,
            fullName: parsed.repository.full_name,
            owner: {
              id: parsed.repository.owner.id,
              login: parsed.repository.owner.login,
            },
          }
        : undefined,
      organization: parsed.organization
        ? {
            id: parsed.organization.id,
            login: parsed.organization.login,
          }
        : undefined,
      installation: parsed.installation
        ? {
            id: parsed.installation.id,
          }
        : undefined,
      payload: parsed,
      signature: signature || "",
      deliveryId,
      timestamp: new Date(),
    };

    this.emit("webhook:verified", event, true);
    this.emit("event:received", webhookPayload);

    return webhookPayload;
  }

  /**
   * Create a webhook signature for testing
   *
   * @param payload - Payload to sign
   * @param secret - Webhook secret
   * @returns Signature string
   */
  createSignature(payload: string | Buffer, secret: string): string {
    return (
      "sha256=" +
      crypto.createHmac("sha256", secret).update(payload).digest("hex")
    );
  }

  // ==========================================================================
  // SMART WEBHOOK MANAGEMENT
  // ==========================================================================

  /**
   * Register or update a webhook with idempotent behavior
   *
   * @param repo - Repository identifier
   * @param config - Webhook configuration
   * @returns Registration result
   *
   * @example
   * ```typescript
   * // This will create or update the webhook as needed
   * const result = await webhooks.ensureWebhook(
   *   { owner: 'org', repo: 'app' },
   *   {
   *     name: 'web',
   *     events: ['push', 'pull_request'],
   *     config: {
   *       url: 'https://api.example.com/webhooks',
   *       contentType: 'json',
   *       secret: process.env.WEBHOOK_SECRET,
   *     },
   *   }
   * );
   *
   * console.log(`Webhook ${result.existed ? 'updated' : 'created'}`);
   * ```
   */
  async ensureWebhook(
    repo: RepoIdentifier,
    config: CreateWebhookConfig
  ): Promise<WebhookRegistrationResult> {
    const validConfig = CreateWebhookConfigSchema.parse(config);

    // Find existing webhook with same URL
    const existing = await this.listWebhooks(repo);
    const match = existing.find(
      (hook) => hook.config.url === validConfig.config.url
    );

    if (match) {
      // Check if update needed
      const eventsMatch = this.arraysEqual(
        [...match.events].sort(),
        [...validConfig.events].sort()
      );
      const activeMatch = match.active === (validConfig.active ?? true);

      if (eventsMatch && activeMatch) {
        return {
          webhook: match,
          existed: true,
          updated: false,
        };
      }

      // Update webhook
      const updated = await this.updateWebhook(repo, match.id, {
        addEvents: validConfig.events.filter(
          (e) => !match.events.includes(e as string)
        ),
        removeEvents: match.events.filter(
          (e) => !validConfig.events.includes(e as WebhookEvent)
        ) as WebhookEvent[],
        active: validConfig.active,
      });

      return {
        webhook: updated,
        existed: true,
        updated: true,
      };
    }

    // Create new webhook
    const webhook = await this.createWebhook(repo, config);

    return {
      webhook,
      existed: false,
      updated: false,
    };
  }

  /**
   * Find webhooks by URL pattern
   *
   * @param repo - Repository identifier
   * @param urlPattern - URL pattern to match (supports wildcards)
   * @returns Matching webhooks
   *
   * @example
   * ```typescript
   * const webhooks = await webhooks.findWebhooksByUrl(
   *   { owner: 'org', repo: 'app' },
   *   '*.example.com/*'
   * );
   * ```
   */
  async findWebhooksByUrl(
    repo: RepoIdentifier,
    urlPattern: string
  ): Promise<Webhook[]> {
    const webhooks = await this.listWebhooks(repo);
    const regex = this.wildcardToRegex(urlPattern);

    return webhooks.filter((hook) => regex.test(hook.config.url));
  }

  /**
   * Disable all webhooks for a repository
   *
   * @param repo - Repository identifier
   * @returns Number of webhooks disabled
   */
  async disableAllWebhooks(repo: RepoIdentifier): Promise<number> {
    const webhooks = await this.listWebhooks(repo);
    let count = 0;

    for (const webhook of webhooks) {
      if (webhook.active) {
        await this.updateWebhook(repo, webhook.id, { active: false });
        count++;
      }
    }

    return count;
  }

  /**
   * Get failed deliveries for a webhook
   *
   * @param repo - Repository identifier
   * @param webhookId - Webhook ID
   * @param options - List options
   * @returns Failed deliveries
   */
  async getFailedDeliveries(
    repo: RepoIdentifier,
    webhookId: number,
    options: WebhookListOptions = {}
  ): Promise<WebhookDelivery[]> {
    const deliveries = await this.listDeliveries(repo, webhookId, options);
    return deliveries.filter((d) => d.statusCode < 200 || d.statusCode >= 300);
  }

  /**
   * Retry all failed deliveries for a webhook
   *
   * @param repo - Repository identifier
   * @param webhookId - Webhook ID
   * @returns Number of redeliveries attempted
   */
  async retryFailedDeliveries(
    repo: RepoIdentifier,
    webhookId: number
  ): Promise<number> {
    const failed = await this.getFailedDeliveries(repo, webhookId);

    for (const delivery of failed) {
      try {
        await this.redeliverWebhook(repo, webhookId, delivery.id);
      } catch (error) {
        // Ignore redelivery errors
        this.emit("delivery:failed", delivery, error as Error);
      }
    }

    return failed.length;
  }

  // ==========================================================================
  // HELPERS
  // ==========================================================================

  /**
   * Map API response to Webhook type
   */
  private mapWebhook(data: Record<string, unknown>): Webhook {
    const config = data.config as Record<string, unknown> | undefined;

    return WebhookSchema.parse({
      id: data.id,
      type: data.type || "Repository",
      name: data.name,
      active: data.active,
      events: data.events,
      config: {
        url: config?.url || "",
        contentType: config?.content_type || "json",
        insecureSsl:
          config?.insecure_ssl === "1" || config?.insecure_ssl === true,
      },
      createdAt: new Date(data.created_at as string),
      updatedAt: new Date(data.updated_at as string),
      lastResponse: data.last_response
        ? {
            code: (data.last_response as Record<string, unknown>).code as
              | number
              | null,
            status: (data.last_response as Record<string, unknown>)
              .status as string,
            message: (data.last_response as Record<string, unknown>).message as
              | string
              | null,
          }
        : undefined,
      pingUrl: data.ping_url as string,
      testUrl: data.test_url as string | undefined,
      deliveriesUrl: data.deliveries_url as string | undefined,
    });
  }

  /**
   * Map API response to WebhookDelivery type
   */
  private mapDelivery(data: Record<string, unknown>): WebhookDelivery {
    const request = data.request as Record<string, unknown> | undefined;
    const response = data.response as Record<string, unknown> | undefined;

    return {
      id: data.id as number,
      guid: data.guid as string,
      deliveredAt: new Date(data.delivered_at as string),
      redelivery: data.redelivery as boolean,
      duration: data.duration as number,
      status: data.status as string,
      statusCode: data.status_code as number,
      event: data.event as string,
      action: data.action as string | undefined,
      installationId: data.installation_id as number | undefined,
      repositoryId: data.repository_id as number | undefined,
      request: {
        headers: (request?.headers || {}) as Record<string, string>,
        payload: request?.payload,
      },
      response: {
        headers: (response?.headers || {}) as Record<string, string>,
        payload: response?.payload as string | null,
      },
    };
  }

  /**
   * Get header value from headers object
   */
  private getHeader(
    headers: Record<string, string | string[] | undefined>,
    name: string
  ): string | undefined {
    const value = headers[name] || headers[name.toLowerCase()];
    return Array.isArray(value) ? value[0] : value;
  }

  /**
   * Compare two arrays for equality
   */
  private arraysEqual<T>(a: T[], b: T[]): boolean {
    if (a.length !== b.length) return false;
    return a.every((val, idx) => val === b[idx]);
  }

  /**
   * Convert wildcard pattern to regex
   */
  private wildcardToRegex(pattern: string): RegExp {
    const escaped = pattern
      .replace(/[.+^${}()|[\]\\]/g, "\\$&")
      .replace(/\*/g, ".*")
      .replace(/\?/g, ".");
    return new RegExp(`^${escaped}$`, "i");
  }
}
