/**
 * @fileoverview GitHub Webhooks Manager module exports
 * @module @neurectomy/github-universe/webhooks
 *
 * @description Provides comprehensive webhook management including creation,
 * verification, and event processing for GitHub webhooks.
 *
 * @example
 * ```typescript
 * import { WebhooksManager, type WebhookPayload } from '@neurectomy/github-universe/webhooks';
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
 *       secret: process.env.WEBHOOK_SECRET,
 *     },
 *   }
 * );
 *
 * // Verify and parse webhook payload
 * const payload = webhooks.parsePayload(
 *   requestBody,
 *   requestHeaders,
 *   process.env.WEBHOOK_SECRET
 * );
 * ```
 */

export { WebhooksManager } from "./manager";
export type {
  WebhooksManagerEvents,
  WebhookDelivery,
  WebhookPayload,
  UpdateWebhookConfig,
  WebhookListOptions,
  OrgIdentifier,
  WebhookRegistrationResult,
} from "./manager";
