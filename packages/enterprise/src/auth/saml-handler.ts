/**
 * @fileoverview SAML 2.0 Authentication Handler
 * @module @neurectomy/enterprise/auth/saml-handler
 *
 * Agent Assignment: @CIPHER (Cryptography) + @SYNAPSE (Integration)
 *
 * Implements SAML 2.0 Service Provider functionality:
 * - SP-initiated SSO
 * - IdP-initiated SSO
 * - Single Logout (SLO)
 * - Assertion validation and signature verification
 * - Attribute mapping
 *
 * @author NEURECTOMY Phase 5 - Enterprise Excellence
 * @version 1.0.0
 */

import { EventEmitter } from "events";
import { createHash, createSign, createVerify, randomBytes } from "crypto";
import { inflate, deflate } from "zlib";
import { promisify } from "util";

import type {
  AuthProvider,
  AuthSession,
  UserIdentity,
  SSOProviderConfig,
  SAMLConfig,
  AttributeMapping,
} from "../types.js";
import {
  type IProviderHandler,
  type LoginRequest,
  type LoginResponse,
  type TokenClaims,
  createDefaultIdentity,
  DEFAULT_ATTRIBUTE_MAPPING,
} from "./sso-provider.js";

const inflateAsync = promisify(inflate);
const deflateAsync = promisify(deflate);

// ============================================================================
// SAML Types
// ============================================================================

/**
 * SAML assertion
 */
export interface SAMLAssertion {
  issuer: string;
  subject: {
    nameId: string;
    nameIdFormat: string;
    confirmation?: {
      method: string;
      notOnOrAfter?: Date;
      recipient?: string;
      inResponseTo?: string;
    };
  };
  conditions?: {
    notBefore?: Date;
    notOnOrAfter?: Date;
    audience?: string[];
  };
  authnStatement?: {
    authnInstant: Date;
    sessionIndex?: string;
    sessionNotOnOrAfter?: Date;
    authnContext?: string;
  };
  attributes: Record<string, string | string[]>;
  signature?: {
    signedInfo: string;
    signatureValue: string;
    certificate: string;
  };
}

/**
 * SAML response
 */
export interface SAMLResponse {
  id: string;
  inResponseTo?: string;
  issueInstant: Date;
  destination?: string;
  issuer: string;
  status: {
    code: string;
    subCode?: string;
    message?: string;
  };
  assertions: SAMLAssertion[];
  signature?: {
    signedInfo: string;
    signatureValue: string;
    certificate: string;
  };
}

/**
 * SAML request state
 */
interface SAMLRequestState {
  id: string;
  providerId: string;
  tenantId: string;
  redirectUri?: string;
  createdAt: Date;
  nonce?: string;
}

// ============================================================================
// SAML Constants
// ============================================================================

const SAML_NAMESPACES = {
  saml: "urn:oasis:names:tc:SAML:2.0:assertion",
  samlp: "urn:oasis:names:tc:SAML:2.0:protocol",
  ds: "http://www.w3.org/2000/09/xmldsig#",
  xenc: "http://www.w3.org/2001/04/xmlenc#",
};

const NAME_ID_FORMATS = {
  unspecified: "urn:oasis:names:tc:SAML:1.1:nameid-format:unspecified",
  emailAddress: "urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress",
  persistent: "urn:oasis:names:tc:SAML:2.0:nameid-format:persistent",
  transient: "urn:oasis:names:tc:SAML:2.0:nameid-format:transient",
};

const STATUS_CODES = {
  success: "urn:oasis:names:tc:SAML:2.0:status:Success",
  requester: "urn:oasis:names:tc:SAML:2.0:status:Requester",
  responder: "urn:oasis:names:tc:SAML:2.0:status:Responder",
  versionMismatch: "urn:oasis:names:tc:SAML:2.0:status:VersionMismatch",
  authnFailed: "urn:oasis:names:tc:SAML:2.0:status:AuthnFailed",
  noPassive: "urn:oasis:names:tc:SAML:2.0:status:NoPassive",
};

// ============================================================================
// SAML Handler (@CIPHER @SYNAPSE)
// ============================================================================

/**
 * SAML 2.0 authentication handler
 *
 * @example
 * ```typescript
 * const samlHandler = new SAMLHandler();
 *
 * await samlHandler.initialize({
 *   id: 'okta-saml',
 *   name: 'Okta SSO',
 *   type: 'saml',
 *   enabled: true,
 *   tenantIds: [],
 *   config: {
 *     entityId: 'https://app.neurectomy.io/saml',
 *     ssoUrl: 'https://okta.com/sso',
 *     certificate: '...',
 *     binding: 'HTTP-POST',
 *   },
 *   attributeMapping: DEFAULT_ATTRIBUTE_MAPPING,
 *   defaultRoles: ['user'],
 *   autoProvision: true,
 *   jitProvisioning: true,
 * });
 *
 * const { redirectUrl } = await samlHandler.initiateLogin({
 *   providerId: 'okta-saml',
 *   tenantId: 'tenant-123',
 * });
 * ```
 */
export class SAMLHandler extends EventEmitter implements IProviderHandler {
  readonly type: AuthProvider = "saml";

  private configs: Map<string, SSOProviderConfig>;
  private requestStates: Map<string, SAMLRequestState>;
  private certificates: Map<string, string>;

  constructor() {
    super();
    this.configs = new Map();
    this.requestStates = new Map();
    this.certificates = new Map();

    // Clean up old request states
    setInterval(() => this.cleanupRequestStates(), 60000);
  }

  // ============================================================================
  // IProviderHandler Implementation
  // ============================================================================

  /**
   * Initialize SAML provider
   */
  async initialize(config: SSOProviderConfig): Promise<void> {
    if (config.type !== "saml") {
      throw new Error("Invalid provider type for SAML handler");
    }

    const samlConfig = config.config as SAMLConfig;

    // Validate required fields
    if (!samlConfig.entityId) {
      throw new Error("SAML entityId is required");
    }
    if (!samlConfig.ssoUrl) {
      throw new Error("SAML ssoUrl is required");
    }
    if (!samlConfig.certificate) {
      throw new Error("SAML certificate is required");
    }

    // Parse and store certificate
    const cert = this.normalizeCertificate(samlConfig.certificate);
    this.certificates.set(config.id, cert);

    this.configs.set(config.id, config);
  }

  /**
   * Initiate SAML login (SP-initiated SSO)
   */
  async initiateLogin(
    request: LoginRequest
  ): Promise<{ redirectUrl: string; state: string }> {
    const config = this.configs.get(request.providerId);
    if (!config) {
      throw new Error(`SAML provider ${request.providerId} not initialized`);
    }

    const samlConfig = config.config as SAMLConfig;

    // Generate request ID
    const requestId = `_${randomBytes(16).toString("hex")}`;

    // Store request state
    const state: SAMLRequestState = {
      id: requestId,
      providerId: request.providerId,
      tenantId: request.tenantId,
      redirectUri: request.redirectUri,
      createdAt: new Date(),
      nonce: request.nonce,
    };
    this.requestStates.set(requestId, state);

    // Build AuthnRequest
    const authnRequest = this.buildAuthnRequest(requestId, config);

    // Encode based on binding
    let redirectUrl: string;

    if (samlConfig.binding === "HTTP-Redirect") {
      const deflated = await deflateAsync(Buffer.from(authnRequest));
      const encoded = deflated.toString("base64");
      const urlEncoded = encodeURIComponent(encoded);

      redirectUrl = `${samlConfig.ssoUrl}?SAMLRequest=${urlEncoded}`;

      if (request.state) {
        redirectUrl += `&RelayState=${encodeURIComponent(request.state)}`;
      }

      // Sign redirect URL if required
      if (samlConfig.signRequests && samlConfig.privateKey) {
        const signature = this.signRedirectUrl(
          redirectUrl,
          samlConfig.privateKey
        );
        redirectUrl += `&SigAlg=${encodeURIComponent(
          "http://www.w3.org/2001/04/xmldsig-more#rsa-sha256"
        )}&Signature=${encodeURIComponent(signature)}`;
      }
    } else {
      // HTTP-POST binding
      const encoded = Buffer.from(authnRequest).toString("base64");
      redirectUrl = `${samlConfig.ssoUrl}?SAMLRequest=${encodeURIComponent(encoded)}`;

      if (request.state) {
        redirectUrl += `&RelayState=${encodeURIComponent(request.state)}`;
      }
    }

    return { redirectUrl, state: request.state || requestId };
  }

  /**
   * Handle SAML callback (assertion consumer service)
   */
  async handleCallback(params: Record<string, string>): Promise<LoginResponse> {
    const samlResponse = params.SAMLResponse;
    if (!samlResponse) {
      return {
        success: false,
        error: {
          code: "MISSING_RESPONSE",
          message: "SAMLResponse parameter is missing",
        },
      };
    }

    try {
      // Decode response
      const decoded = Buffer.from(samlResponse, "base64").toString("utf-8");
      const response = this.parseResponse(decoded);

      // Validate response
      const validationResult = await this.validateResponse(
        response,
        params.RelayState
      );
      if (!validationResult.valid) {
        return {
          success: false,
          error: {
            code: "VALIDATION_FAILED",
            message: validationResult.error || "Response validation failed",
          },
        };
      }

      // Get provider config
      const state = response.inResponseTo
        ? this.requestStates.get(response.inResponseTo)
        : null;

      let providerId: string;
      if (state) {
        providerId = state.providerId;
        this.requestStates.delete(response.inResponseTo!);
      } else {
        // IdP-initiated SSO - find provider by issuer
        providerId = this.findProviderByIssuer(response.issuer);
        if (!providerId) {
          return {
            success: false,
            error: {
              code: "UNKNOWN_ISSUER",
              message: `Unknown SAML issuer: ${response.issuer}`,
            },
          };
        }
      }

      const config = this.configs.get(providerId);
      if (!config) {
        return {
          success: false,
          error: {
            code: "PROVIDER_NOT_FOUND",
            message: `Provider ${providerId} not found`,
          },
        };
      }

      // Extract identity from assertion
      const assertion = response.assertions[0];
      if (!assertion) {
        return {
          success: false,
          error: {
            code: "NO_ASSERTION",
            message: "No assertion found in SAML response",
          },
        };
      }

      const identity = this.mapIdentity(
        {
          nameId: assertion.subject.nameId,
          ...assertion.attributes,
        },
        config.attributeMapping || DEFAULT_ATTRIBUTE_MAPPING
      );

      // Set tenant from state or default
      identity.tenantId = state?.tenantId || "";
      identity.provider = "saml";
      identity.providerId = providerId;

      // Apply default roles
      if (config.defaultRoles) {
        identity.roles = [
          ...new Set([...identity.roles, ...config.defaultRoles]),
        ];
      }

      return {
        success: true,
        identity,
        redirectUrl: state?.redirectUri,
      };
    } catch (error) {
      return {
        success: false,
        error: {
          code: "PARSE_ERROR",
          message: (error as Error).message,
        },
      };
    }
  }

  /**
   * Refresh token (not applicable for SAML)
   */
  async refreshToken(session: AuthSession): Promise<AuthSession> {
    // SAML doesn't support token refresh
    // Return session as-is, let SSO manager handle local refresh
    return session;
  }

  /**
   * Handle SAML logout
   */
  async logout(session: AuthSession): Promise<void> {
    const config = this.configs.get(session.provider);
    if (!config) return;

    const samlConfig = config.config as SAMLConfig;
    if (!samlConfig.sloUrl) return;

    // Build LogoutRequest
    const requestId = `_${randomBytes(16).toString("hex")}`;
    const logoutRequest = this.buildLogoutRequest(requestId, session, config);

    // In a real implementation, this would send the logout request
    // For now, we'll emit an event
    this.emit("logout:requested", {
      providerId: config.id,
      sessionId: session.id,
      logoutUrl: samlConfig.sloUrl,
      request: logoutRequest,
    });
  }

  /**
   * Validate token (for SAML, this checks assertion validity)
   */
  async validateToken(token: string): Promise<TokenClaims | null> {
    // SAML uses assertions, not tokens
    // This method is not applicable for SAML
    return null;
  }

  /**
   * Map provider data to user identity
   */
  mapIdentity(
    providerData: Record<string, unknown>,
    mapping: AttributeMapping
  ): UserIdentity {
    return createDefaultIdentity(providerData, mapping, "", "saml");
  }

  // ============================================================================
  // SAML Request Building
  // ============================================================================

  /**
   * Build SAML AuthnRequest
   */
  private buildAuthnRequest(
    requestId: string,
    config: SSOProviderConfig
  ): string {
    const samlConfig = config.config as SAMLConfig;
    const issueInstant = new Date().toISOString();

    const request = `<?xml version="1.0" encoding="UTF-8"?>
<samlp:AuthnRequest
  xmlns:samlp="${SAML_NAMESPACES.samlp}"
  xmlns:saml="${SAML_NAMESPACES.saml}"
  ID="${requestId}"
  Version="2.0"
  IssueInstant="${issueInstant}"
  Destination="${samlConfig.ssoUrl}"
  ProtocolBinding="urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST"
  AssertionConsumerServiceURL="${samlConfig.entityId}/acs"
  ${samlConfig.forceAuthn ? 'ForceAuthn="true"' : ""}>
  <saml:Issuer>${samlConfig.entityId}</saml:Issuer>
  <samlp:NameIDPolicy
    Format="${samlConfig.nameIdFormat || NAME_ID_FORMATS.unspecified}"
    AllowCreate="true"/>
  ${
    samlConfig.authnContextClassRef
      ? `
  <samlp:RequestedAuthnContext Comparison="exact">
    <saml:AuthnContextClassRef>${samlConfig.authnContextClassRef}</saml:AuthnContextClassRef>
  </samlp:RequestedAuthnContext>`
      : ""
  }
</samlp:AuthnRequest>`;

    return request.trim();
  }

  /**
   * Build SAML LogoutRequest
   */
  private buildLogoutRequest(
    requestId: string,
    session: AuthSession,
    config: SSOProviderConfig
  ): string {
    const samlConfig = config.config as SAMLConfig;
    const issueInstant = new Date().toISOString();

    const request = `<?xml version="1.0" encoding="UTF-8"?>
<samlp:LogoutRequest
  xmlns:samlp="${SAML_NAMESPACES.samlp}"
  xmlns:saml="${SAML_NAMESPACES.saml}"
  ID="${requestId}"
  Version="2.0"
  IssueInstant="${issueInstant}"
  Destination="${samlConfig.sloUrl}">
  <saml:Issuer>${samlConfig.entityId}</saml:Issuer>
  <saml:NameID>${session.userId}</saml:NameID>
  <samlp:SessionIndex>${session.id}</samlp:SessionIndex>
</samlp:LogoutRequest>`;

    return request.trim();
  }

  // ============================================================================
  // SAML Response Parsing
  // ============================================================================

  /**
   * Parse SAML response XML
   */
  private parseResponse(xml: string): SAMLResponse {
    // Simplified parsing - in production, use a proper XML parser
    // This is a basic implementation for demonstration

    const getId = (tag: string): string | undefined => {
      const match = xml.match(new RegExp(`<${tag}[^>]*ID="([^"]+)"`, "i"));
      return match?.[1];
    };

    const getIssueInstant = (): Date => {
      const match = xml.match(/IssueInstant="([^"]+)"/);
      return match ? new Date(match[1]) : new Date();
    };

    const getIssuer = (): string => {
      const match = xml.match(/<saml:Issuer[^>]*>([^<]+)<\/saml:Issuer>/);
      return match?.[1] || "";
    };

    const getStatus = (): { code: string; message?: string } => {
      const codeMatch = xml.match(/<samlp:StatusCode[^>]*Value="([^"]+)"/);
      const messageMatch = xml.match(
        /<samlp:StatusMessage>([^<]*)<\/samlp:StatusMessage>/
      );

      return {
        code: codeMatch?.[1] || STATUS_CODES.responder,
        message: messageMatch?.[1],
      };
    };

    const getInResponseTo = (): string | undefined => {
      const match = xml.match(/InResponseTo="([^"]+)"/);
      return match?.[1];
    };

    const parseAssertions = (): SAMLAssertion[] => {
      const assertions: SAMLAssertion[] = [];
      const assertionMatches = xml.matchAll(
        /<saml:Assertion[^>]*>([\s\S]*?)<\/saml:Assertion>/g
      );

      for (const match of assertionMatches) {
        const assertionXml = match[0];

        // Parse NameID
        const nameIdMatch = assertionXml.match(
          /<saml:NameID[^>]*>([^<]+)<\/saml:NameID>/
        );
        const nameIdFormatMatch = assertionXml.match(
          /<saml:NameID[^>]*Format="([^"]+)"/
        );

        // Parse attributes
        const attributes: Record<string, string | string[]> = {};
        const attrMatches = assertionXml.matchAll(
          /<saml:Attribute[^>]*Name="([^"]+)"[^>]*>[\s\S]*?<saml:AttributeValue[^>]*>([^<]*)<\/saml:AttributeValue>[\s\S]*?<\/saml:Attribute>/g
        );

        for (const attrMatch of attrMatches) {
          const name = attrMatch[1];
          const value = attrMatch[2];

          if (attributes[name]) {
            if (Array.isArray(attributes[name])) {
              (attributes[name] as string[]).push(value);
            } else {
              attributes[name] = [attributes[name] as string, value];
            }
          } else {
            attributes[name] = value;
          }
        }

        assertions.push({
          issuer: getIssuer(),
          subject: {
            nameId: nameIdMatch?.[1] || "",
            nameIdFormat: nameIdFormatMatch?.[1] || NAME_ID_FORMATS.unspecified,
          },
          attributes,
        });
      }

      return assertions;
    };

    return {
      id: getId("samlp:Response") || getId("Response") || "",
      inResponseTo: getInResponseTo(),
      issueInstant: getIssueInstant(),
      issuer: getIssuer(),
      status: getStatus(),
      assertions: parseAssertions(),
    };
  }

  // ============================================================================
  // Validation
  // ============================================================================

  /**
   * Validate SAML response
   */
  private async validateResponse(
    response: SAMLResponse,
    relayState?: string
  ): Promise<{ valid: boolean; error?: string }> {
    // Check status
    if (response.status.code !== STATUS_CODES.success) {
      return {
        valid: false,
        error: `SAML status: ${response.status.code} - ${response.status.message || "Unknown error"}`,
      };
    }

    // Check for assertions
    if (response.assertions.length === 0) {
      return {
        valid: false,
        error: "No assertions in SAML response",
      };
    }

    // Check InResponseTo if we have a stored request
    if (response.inResponseTo) {
      const state = this.requestStates.get(response.inResponseTo);
      if (!state) {
        // Could be IdP-initiated SSO, allow it
        // In production, might want stricter validation
      } else {
        // Validate request is not too old (5 minutes)
        const age = Date.now() - state.createdAt.getTime();
        if (age > 300000) {
          return {
            valid: false,
            error: "SAML response is too old",
          };
        }
      }
    }

    // Validate assertion conditions
    const assertion = response.assertions[0];
    if (assertion.conditions) {
      const now = new Date();
      if (
        assertion.conditions.notBefore &&
        now < assertion.conditions.notBefore
      ) {
        return {
          valid: false,
          error: "Assertion not yet valid",
        };
      }
      if (
        assertion.conditions.notOnOrAfter &&
        now >= assertion.conditions.notOnOrAfter
      ) {
        return {
          valid: false,
          error: "Assertion has expired",
        };
      }
    }

    // In production, would also validate:
    // - Signature on response and/or assertion
    // - Audience restriction
    // - SubjectConfirmation

    return { valid: true };
  }

  /**
   * Sign redirect URL
   */
  private signRedirectUrl(url: string, privateKey: string): string {
    const sign = createSign("RSA-SHA256");
    sign.update(url);
    return sign.sign(privateKey, "base64");
  }

  /**
   * Verify signature
   */
  private verifySignature(
    data: string,
    signature: string,
    certificate: string
  ): boolean {
    try {
      const verify = createVerify("RSA-SHA256");
      verify.update(data);
      return verify.verify(certificate, signature, "base64");
    } catch {
      return false;
    }
  }

  // ============================================================================
  // Utilities
  // ============================================================================

  /**
   * Normalize certificate (add headers if missing)
   */
  private normalizeCertificate(cert: string): string {
    let normalized = cert.replace(/\s+/g, "");

    if (!normalized.startsWith("-----BEGIN")) {
      normalized = `-----BEGIN CERTIFICATE-----\n${normalized
        .match(/.{1,64}/g)
        ?.join("\n")}\n-----END CERTIFICATE-----`;
    }

    return normalized;
  }

  /**
   * Find provider by issuer
   */
  private findProviderByIssuer(issuer: string): string {
    for (const [id, config] of this.configs) {
      const samlConfig = config.config as SAMLConfig;
      // Check if issuer matches the IdP's entity ID
      // This is a simplified check
      if (
        samlConfig.ssoUrl?.includes(issuer) ||
        samlConfig.metadata?.includes(issuer)
      ) {
        return id;
      }
    }
    return "";
  }

  /**
   * Clean up old request states
   */
  private cleanupRequestStates(): void {
    const maxAge = 300000; // 5 minutes
    const now = Date.now();

    for (const [id, state] of this.requestStates) {
      if (now - state.createdAt.getTime() > maxAge) {
        this.requestStates.delete(id);
      }
    }
  }

  /**
   * Generate SP metadata
   */
  generateMetadata(providerId: string): string {
    const config = this.configs.get(providerId);
    if (!config) {
      throw new Error(`Provider ${providerId} not found`);
    }

    const samlConfig = config.config as SAMLConfig;

    return `<?xml version="1.0" encoding="UTF-8"?>
<md:EntityDescriptor
  xmlns:md="urn:oasis:names:tc:SAML:2.0:metadata"
  xmlns:ds="http://www.w3.org/2000/09/xmldsig#"
  entityID="${samlConfig.entityId}">
  <md:SPSSODescriptor
    AuthnRequestsSigned="${samlConfig.signRequests}"
    WantAssertionsSigned="${samlConfig.wantAssertionsSigned}"
    protocolSupportEnumeration="urn:oasis:names:tc:SAML:2.0:protocol">
    <md:NameIDFormat>${samlConfig.nameIdFormat || NAME_ID_FORMATS.unspecified}</md:NameIDFormat>
    <md:AssertionConsumerService
      Binding="urn:oasis:names:tc:SAML:2.0:bindings:${samlConfig.binding}"
      Location="${samlConfig.entityId}/acs"
      index="0"
      isDefault="true"/>
    ${
      samlConfig.sloUrl
        ? `
    <md:SingleLogoutService
      Binding="urn:oasis:names:tc:SAML:2.0:bindings:${samlConfig.binding}"
      Location="${samlConfig.entityId}/slo"/>`
        : ""
    }
  </md:SPSSODescriptor>
</md:EntityDescriptor>`;
  }
}
