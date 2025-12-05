/**
 * @fileoverview SSO Provider Base Implementation
 * @module @neurectomy/enterprise/auth/sso-provider
 *
 * Agent Assignment: @CIPHER (Cryptography) + @SYNAPSE (Integration)
 *
 * Implements a unified SSO provider abstraction layer:
 * - Provider registration and management
 * - Token validation and refresh
 * - Session management
 * - Provider-agnostic identity mapping
 *
 * @author NEURECTOMY Phase 5 - Enterprise Excellence
 * @version 1.0.0
 */

import { EventEmitter } from "events";
import { createHash, randomBytes, createHmac } from "crypto";
import { SignJWT, jwtVerify, type JWTPayload } from "jose";
import { v4 as uuidv4 } from "uuid";

import type {
  AuthProvider,
  AuthSession,
  UserIdentity,
  SSOProviderConfig,
  AttributeMapping,
  SessionState,
  MFAMethod,
  IdentityMetadata,
  AuthModuleConfig,
  CookieSettings,
} from "../types.js";

// ============================================================================
// SSO Types (@CIPHER)
// ============================================================================

/**
 * SSO event types
 */
export type SSOEventType =
  | "provider:registered"
  | "provider:updated"
  | "provider:removed"
  | "session:created"
  | "session:refreshed"
  | "session:revoked"
  | "session:expired"
  | "login:success"
  | "login:failure"
  | "logout:success"
  | "mfa:required"
  | "mfa:verified";

/**
 * SSO event
 */
export interface SSOEvent<T = unknown> {
  type: SSOEventType;
  timestamp: Date;
  tenantId?: string;
  userId?: string;
  providerId?: string;
  sessionId?: string;
  data?: T;
  metadata?: Record<string, unknown>;
}

/**
 * Login request
 */
export interface LoginRequest {
  providerId: string;
  tenantId: string;
  redirectUri?: string;
  state?: string;
  nonce?: string;
  loginHint?: string;
  prompt?: "none" | "login" | "consent" | "select_account";
  scopes?: string[];
  additionalParams?: Record<string, string>;
}

/**
 * Login response
 */
export interface LoginResponse {
  success: boolean;
  session?: AuthSession;
  identity?: UserIdentity;
  redirectUrl?: string;
  error?: {
    code: string;
    message: string;
    details?: Record<string, unknown>;
  };
  mfaRequired?: boolean;
  mfaMethods?: MFAMethod[];
}

/**
 * Token claims
 */
export interface TokenClaims extends JWTPayload {
  sub: string;
  email: string;
  name?: string;
  roles: string[];
  permissions: string[];
  tenantId: string;
  sessionId: string;
  provider: AuthProvider;
  mfaVerified: boolean;
}

/**
 * Provider handler interface
 */
export interface IProviderHandler {
  readonly type: AuthProvider;
  initialize(config: SSOProviderConfig): Promise<void>;
  initiateLogin(
    request: LoginRequest
  ): Promise<{ redirectUrl: string; state: string }>;
  handleCallback(params: Record<string, string>): Promise<LoginResponse>;
  refreshToken(session: AuthSession): Promise<AuthSession>;
  logout(session: AuthSession): Promise<void>;
  validateToken(token: string): Promise<TokenClaims | null>;
  mapIdentity(
    providerData: Record<string, unknown>,
    mapping: AttributeMapping
  ): UserIdentity;
}

// ============================================================================
// Default Configuration
// ============================================================================

/**
 * Default auth module configuration
 */
export const DEFAULT_AUTH_CONFIG: AuthModuleConfig = {
  sessionTimeout: 3600000, // 1 hour
  maxConcurrentSessions: 5,
  tokenExpiration: 3600, // 1 hour in seconds
  refreshTokenExpiration: 604800, // 7 days in seconds
  mfaGracePeriod: 300000, // 5 minutes
  passwordPolicy: {
    minLength: 12,
    maxLength: 128,
    requireUppercase: true,
    requireLowercase: true,
    requireNumbers: true,
    requireSpecialChars: true,
    preventReuse: 12,
    expirationDays: 90,
    lockoutAttempts: 5,
    lockoutDuration: 900000, // 15 minutes
  },
  providers: [],
  jwtSecret: "", // Must be provided
  jwtAlgorithm: "HS256",
  cookieSettings: {
    name: "neurectomy_session",
    secure: true,
    httpOnly: true,
    sameSite: "lax",
    path: "/",
    maxAge: 3600000,
  },
};

/**
 * Default attribute mapping
 */
export const DEFAULT_ATTRIBUTE_MAPPING: AttributeMapping = {
  id: "sub",
  email: "email",
  displayName: "name",
  firstName: "given_name",
  lastName: "family_name",
  groups: "groups",
  roles: "roles",
  avatar: "picture",
  custom: {},
};

// ============================================================================
// SSO Manager (@CIPHER @SYNAPSE)
// ============================================================================

/**
 * Unified SSO manager for all authentication providers
 *
 * @example
 * ```typescript
 * const ssoManager = new SSOManager({
 *   ...DEFAULT_AUTH_CONFIG,
 *   jwtSecret: process.env.JWT_SECRET,
 * });
 *
 * // Register providers
 * await ssoManager.registerProvider(samlConfig);
 * await ssoManager.registerProvider(oauthConfig);
 *
 * // Initiate login
 * const { redirectUrl } = await ssoManager.initiateLogin({
 *   providerId: 'okta-saml',
 *   tenantId: 'tenant-123',
 * });
 *
 * // Handle callback
 * const response = await ssoManager.handleCallback('okta-saml', params);
 * ```
 */
export class SSOManager extends EventEmitter {
  private config: AuthModuleConfig;
  private providers: Map<string, SSOProviderConfig>;
  private handlers: Map<AuthProvider, IProviderHandler>;
  private sessions: Map<string, AuthSession>;
  private userSessions: Map<string, Set<string>>; // userId -> sessionIds
  private jwtSecret: Uint8Array;

  constructor(config: Partial<AuthModuleConfig> = {}) {
    super();
    this.config = { ...DEFAULT_AUTH_CONFIG, ...config };
    this.providers = new Map();
    this.handlers = new Map();
    this.sessions = new Map();
    this.userSessions = new Map();
    this.jwtSecret = new TextEncoder().encode(this.config.jwtSecret);

    // Start session cleanup interval
    this.startSessionCleanup();
  }

  // ============================================================================
  // Provider Management
  // ============================================================================

  /**
   * Register a new SSO provider
   */
  async registerProvider(config: SSOProviderConfig): Promise<void> {
    if (this.providers.has(config.id)) {
      throw new Error(`Provider ${config.id} already registered`);
    }

    // Get or create handler for provider type
    const handler = await this.getOrCreateHandler(config.type);
    await handler.initialize(config);

    this.providers.set(config.id, config);

    this.emit("sso:event", {
      type: "provider:registered",
      timestamp: new Date(),
      providerId: config.id,
      data: { name: config.name, type: config.type },
    } as SSOEvent);
  }

  /**
   * Update provider configuration
   */
  async updateProvider(
    providerId: string,
    updates: Partial<SSOProviderConfig>
  ): Promise<void> {
    const existing = this.providers.get(providerId);
    if (!existing) {
      throw new Error(`Provider ${providerId} not found`);
    }

    const updated = { ...existing, ...updates, id: providerId };
    const handler = await this.getOrCreateHandler(updated.type);
    await handler.initialize(updated);

    this.providers.set(providerId, updated);

    this.emit("sso:event", {
      type: "provider:updated",
      timestamp: new Date(),
      providerId,
    } as SSOEvent);
  }

  /**
   * Remove a provider
   */
  removeProvider(providerId: string): void {
    if (!this.providers.has(providerId)) {
      throw new Error(`Provider ${providerId} not found`);
    }

    this.providers.delete(providerId);

    this.emit("sso:event", {
      type: "provider:removed",
      timestamp: new Date(),
      providerId,
    } as SSOEvent);
  }

  /**
   * Get provider by ID
   */
  getProvider(providerId: string): SSOProviderConfig | undefined {
    return this.providers.get(providerId);
  }

  /**
   * Get all providers for a tenant
   */
  getProvidersForTenant(tenantId: string): SSOProviderConfig[] {
    return Array.from(this.providers.values()).filter(
      (p) =>
        p.enabled &&
        (p.tenantIds.length === 0 || p.tenantIds.includes(tenantId))
    );
  }

  // ============================================================================
  // Authentication Flow
  // ============================================================================

  /**
   * Initiate SSO login flow
   */
  async initiateLogin(
    request: LoginRequest
  ): Promise<{ redirectUrl: string; state: string }> {
    const provider = this.providers.get(request.providerId);
    if (!provider) {
      throw new Error(`Provider ${request.providerId} not found`);
    }

    if (!provider.enabled) {
      throw new Error(`Provider ${request.providerId} is disabled`);
    }

    // Check tenant access
    if (
      provider.tenantIds.length > 0 &&
      !provider.tenantIds.includes(request.tenantId)
    ) {
      throw new Error(
        `Provider ${request.providerId} not available for tenant`
      );
    }

    const handler = this.handlers.get(provider.type);
    if (!handler) {
      throw new Error(`Handler for ${provider.type} not found`);
    }

    // Generate secure state
    const state = this.generateState({
      providerId: request.providerId,
      tenantId: request.tenantId,
      redirectUri: request.redirectUri,
      timestamp: Date.now(),
    });

    const result = await handler.initiateLogin({
      ...request,
      state,
      nonce: request.nonce || this.generateNonce(),
    });

    return result;
  }

  /**
   * Handle SSO callback
   */
  async handleCallback(
    providerId: string,
    params: Record<string, string>
  ): Promise<LoginResponse> {
    const provider = this.providers.get(providerId);
    if (!provider) {
      return {
        success: false,
        error: {
          code: "PROVIDER_NOT_FOUND",
          message: `Provider ${providerId} not found`,
        },
      };
    }

    const handler = this.handlers.get(provider.type);
    if (!handler) {
      return {
        success: false,
        error: {
          code: "HANDLER_NOT_FOUND",
          message: `Handler for ${provider.type} not found`,
        },
      };
    }

    try {
      // Validate state
      if (params.state) {
        const stateData = this.validateState(params.state);
        if (!stateData || stateData.providerId !== providerId) {
          return {
            success: false,
            error: {
              code: "INVALID_STATE",
              message: "Invalid or expired state parameter",
            },
          };
        }
      }

      // Handle provider callback
      const response = await handler.handleCallback(params);

      if (!response.success || !response.identity) {
        this.emit("sso:event", {
          type: "login:failure",
          timestamp: new Date(),
          providerId,
          data: response.error,
        } as SSOEvent);

        return response;
      }

      // Check MFA requirement
      const tenant = response.identity.tenantId;
      if (this.isMFARequired(response.identity) && !response.mfaRequired) {
        return {
          ...response,
          mfaRequired: true,
          mfaMethods: this.getAvailableMFAMethods(response.identity),
        };
      }

      // Create session
      const session = await this.createSession(
        response.identity,
        provider,
        params
      );
      response.session = session;

      this.emit("sso:event", {
        type: "login:success",
        timestamp: new Date(),
        tenantId: response.identity.tenantId,
        userId: response.identity.id,
        providerId,
        sessionId: session.id,
      } as SSOEvent);

      return response;
    } catch (error) {
      this.emit("sso:event", {
        type: "login:failure",
        timestamp: new Date(),
        providerId,
        data: { error: (error as Error).message },
      } as SSOEvent);

      return {
        success: false,
        error: {
          code: "CALLBACK_ERROR",
          message: (error as Error).message,
        },
      };
    }
  }

  /**
   * Logout and revoke session
   */
  async logout(sessionId: string, revokeAll = false): Promise<void> {
    const session = this.sessions.get(sessionId);
    if (!session) {
      throw new Error(`Session ${sessionId} not found`);
    }

    // Revoke all sessions for user if requested
    if (revokeAll) {
      const userSessionIds = this.userSessions.get(session.userId);
      if (userSessionIds) {
        for (const sid of userSessionIds) {
          await this.revokeSession(sid);
        }
      }
    } else {
      await this.revokeSession(sessionId);
    }

    // Call provider logout if supported
    const provider = this.providers.get(session.provider);
    if (provider) {
      const handler = this.handlers.get(provider.type);
      if (handler) {
        try {
          await handler.logout(session);
        } catch (error) {
          // Log but don't fail
          console.error("Provider logout failed:", error);
        }
      }
    }

    this.emit("sso:event", {
      type: "logout:success",
      timestamp: new Date(),
      tenantId: session.tenantId,
      userId: session.userId,
      sessionId,
    } as SSOEvent);
  }

  // ============================================================================
  // Session Management
  // ============================================================================

  /**
   * Create a new session
   */
  private async createSession(
    identity: UserIdentity,
    provider: SSOProviderConfig,
    tokenData: Record<string, string>
  ): Promise<AuthSession> {
    // Check concurrent session limit
    await this.enforceSessionLimit(identity.id);

    const sessionId = uuidv4();
    const now = new Date();
    const expiresAt = new Date(now.getTime() + this.config.sessionTimeout);

    // Generate JWT access token
    const accessToken = await this.generateAccessToken({
      sub: identity.id,
      email: identity.email,
      name: identity.displayName,
      roles: identity.roles,
      permissions: identity.permissions,
      tenantId: identity.tenantId,
      sessionId,
      provider: provider.type,
      mfaVerified: identity.metadata.mfaEnabled,
    });

    // Generate refresh token
    const refreshToken = await this.generateRefreshToken(
      sessionId,
      identity.id
    );

    const session: AuthSession = {
      id: sessionId,
      userId: identity.id,
      tenantId: identity.tenantId,
      state: "active",
      provider: provider.type,
      accessToken,
      refreshToken,
      idToken: tokenData.id_token,
      tokenType: "Bearer",
      expiresAt,
      refreshExpiresAt: new Date(
        now.getTime() + this.config.refreshTokenExpiration * 1000
      ),
      scopes: (tokenData.scope || "").split(" ").filter(Boolean),
      ipAddress: "",
      userAgent: "",
      createdAt: now,
      lastAccessAt: now,
      mfaVerified: identity.metadata.mfaEnabled,
      metadata: {},
    };

    this.sessions.set(sessionId, session);

    // Track user sessions
    if (!this.userSessions.has(identity.id)) {
      this.userSessions.set(identity.id, new Set());
    }
    this.userSessions.get(identity.id)!.add(sessionId);

    this.emit("sso:event", {
      type: "session:created",
      timestamp: now,
      tenantId: identity.tenantId,
      userId: identity.id,
      sessionId,
    } as SSOEvent);

    return session;
  }

  /**
   * Refresh an existing session
   */
  async refreshSession(refreshToken: string): Promise<AuthSession> {
    // Validate refresh token
    const claims = await this.validateRefreshToken(refreshToken);
    if (!claims) {
      throw new Error("Invalid refresh token");
    }

    const session = this.sessions.get(claims.sessionId as string);
    if (!session) {
      throw new Error("Session not found");
    }

    if (session.state !== "active") {
      throw new Error(`Session is ${session.state}`);
    }

    // Refresh with provider if possible
    const provider = this.providers.get(session.provider);
    if (provider) {
      const handler = this.handlers.get(provider.type);
      if (handler && session.refreshToken) {
        try {
          const refreshed = await handler.refreshToken(session);
          Object.assign(session, refreshed);
        } catch (error) {
          // Provider refresh failed, generate new local tokens
          console.warn("Provider refresh failed, using local refresh");
        }
      }
    }

    // Generate new access token
    const newAccessToken = await this.generateAccessToken({
      sub: session.userId,
      email: "",
      roles: [],
      permissions: [],
      tenantId: session.tenantId,
      sessionId: session.id,
      provider: session.provider,
      mfaVerified: session.mfaVerified,
    });

    session.accessToken = newAccessToken;
    session.lastAccessAt = new Date();
    session.expiresAt = new Date(Date.now() + this.config.sessionTimeout);

    this.emit("sso:event", {
      type: "session:refreshed",
      timestamp: new Date(),
      tenantId: session.tenantId,
      userId: session.userId,
      sessionId: session.id,
    } as SSOEvent);

    return session;
  }

  /**
   * Get session by ID
   */
  getSession(sessionId: string): AuthSession | undefined {
    const session = this.sessions.get(sessionId);
    if (session && this.isSessionValid(session)) {
      session.lastAccessAt = new Date();
      return session;
    }
    return undefined;
  }

  /**
   * Validate access token
   */
  async validateAccessToken(token: string): Promise<TokenClaims | null> {
    try {
      const { payload } = await jwtVerify(token, this.jwtSecret);
      const claims = payload as TokenClaims;

      // Verify session is still active
      const session = this.sessions.get(claims.sessionId);
      if (!session || session.state !== "active") {
        return null;
      }

      return claims;
    } catch (error) {
      return null;
    }
  }

  /**
   * Revoke a session
   */
  private async revokeSession(sessionId: string): Promise<void> {
    const session = this.sessions.get(sessionId);
    if (session) {
      session.state = "revoked";

      // Remove from user sessions
      const userSessionIds = this.userSessions.get(session.userId);
      if (userSessionIds) {
        userSessionIds.delete(sessionId);
        if (userSessionIds.size === 0) {
          this.userSessions.delete(session.userId);
        }
      }

      this.emit("sso:event", {
        type: "session:revoked",
        timestamp: new Date(),
        tenantId: session.tenantId,
        userId: session.userId,
        sessionId,
      } as SSOEvent);
    }
  }

  /**
   * Check if session is valid
   */
  private isSessionValid(session: AuthSession): boolean {
    if (session.state !== "active") {
      return false;
    }

    if (session.expiresAt < new Date()) {
      session.state = "expired";
      return false;
    }

    return true;
  }

  /**
   * Enforce concurrent session limit
   */
  private async enforceSessionLimit(userId: string): Promise<void> {
    const userSessionIds = this.userSessions.get(userId);
    if (!userSessionIds) return;

    const activeSessions = Array.from(userSessionIds)
      .map((id) => this.sessions.get(id))
      .filter((s) => s && s.state === "active")
      .sort((a, b) => a!.createdAt.getTime() - b!.createdAt.getTime());

    // Revoke oldest sessions if over limit
    while (activeSessions.length >= this.config.maxConcurrentSessions) {
      const oldest = activeSessions.shift();
      if (oldest) {
        await this.revokeSession(oldest.id);
      }
    }
  }

  // ============================================================================
  // Token Generation
  // ============================================================================

  /**
   * Generate access token
   */
  private async generateAccessToken(claims: TokenClaims): Promise<string> {
    const jwt = new SignJWT(claims)
      .setProtectedHeader({ alg: "HS256" })
      .setIssuedAt()
      .setExpirationTime(`${this.config.tokenExpiration}s`)
      .setIssuer("neurectomy")
      .setAudience("neurectomy-api");

    return jwt.sign(this.jwtSecret);
  }

  /**
   * Generate refresh token
   */
  private async generateRefreshToken(
    sessionId: string,
    userId: string
  ): Promise<string> {
    const jwt = new SignJWT({ sessionId, userId, type: "refresh" })
      .setProtectedHeader({ alg: "HS256" })
      .setIssuedAt()
      .setExpirationTime(`${this.config.refreshTokenExpiration}s`)
      .setIssuer("neurectomy")
      .setAudience("neurectomy-refresh");

    return jwt.sign(this.jwtSecret);
  }

  /**
   * Validate refresh token
   */
  private async validateRefreshToken(
    token: string
  ): Promise<JWTPayload | null> {
    try {
      const { payload } = await jwtVerify(token, this.jwtSecret, {
        audience: "neurectomy-refresh",
      });

      if (payload.type !== "refresh") {
        return null;
      }

      return payload;
    } catch {
      return null;
    }
  }

  // ============================================================================
  // State Management
  // ============================================================================

  /**
   * Generate cryptographically secure state parameter
   */
  private generateState(data: Record<string, unknown>): string {
    const payload = JSON.stringify(data);
    const signature = createHmac("sha256", this.config.jwtSecret)
      .update(payload)
      .digest("hex");

    const encoded = Buffer.from(payload).toString("base64url");
    return `${encoded}.${signature}`;
  }

  /**
   * Validate state parameter
   */
  private validateState(state: string): Record<string, unknown> | null {
    try {
      const [encoded, signature] = state.split(".");
      if (!encoded || !signature) return null;

      const payload = Buffer.from(encoded, "base64url").toString();
      const expectedSignature = createHmac("sha256", this.config.jwtSecret)
        .update(payload)
        .digest("hex");

      if (signature !== expectedSignature) return null;

      const data = JSON.parse(payload);

      // Check timestamp (5 minute expiry)
      if (data.timestamp && Date.now() - data.timestamp > 300000) {
        return null;
      }

      return data;
    } catch {
      return null;
    }
  }

  /**
   * Generate nonce
   */
  private generateNonce(): string {
    return randomBytes(16).toString("hex");
  }

  // ============================================================================
  // MFA Helpers
  // ============================================================================

  /**
   * Check if MFA is required
   */
  private isMFARequired(identity: UserIdentity): boolean {
    // Check identity metadata
    if (identity.metadata.mfaEnabled) {
      return true;
    }

    // Could add tenant-level MFA requirements here
    return false;
  }

  /**
   * Get available MFA methods
   */
  private getAvailableMFAMethods(identity: UserIdentity): MFAMethod[] {
    return identity.metadata.mfaMethods.length > 0
      ? identity.metadata.mfaMethods
      : ["totp", "email"];
  }

  // ============================================================================
  // Handler Management
  // ============================================================================

  /**
   * Get or create handler for provider type
   */
  private async getOrCreateHandler(
    type: AuthProvider
  ): Promise<IProviderHandler> {
    if (this.handlers.has(type)) {
      return this.handlers.get(type)!;
    }

    // Dynamic handler creation based on type
    let handler: IProviderHandler;

    switch (type) {
      case "saml":
        const { SAMLHandler } = await import("./saml-handler.js");
        handler = new SAMLHandler();
        break;
      case "oauth2":
      case "oidc":
        const { OAuthHandler } = await import("./oauth-handler.js");
        handler = new OAuthHandler(type);
        break;
      default:
        throw new Error(`Unsupported provider type: ${type}`);
    }

    this.handlers.set(type, handler);
    return handler;
  }

  // ============================================================================
  // Cleanup
  // ============================================================================

  /**
   * Start periodic session cleanup
   */
  private startSessionCleanup(): void {
    setInterval(() => {
      const now = new Date();
      for (const [sessionId, session] of this.sessions) {
        if (session.expiresAt < now && session.state === "active") {
          session.state = "expired";

          this.emit("sso:event", {
            type: "session:expired",
            timestamp: now,
            tenantId: session.tenantId,
            userId: session.userId,
            sessionId,
          } as SSOEvent);
        }

        // Remove sessions older than refresh expiry + 1 day
        const maxAge = this.config.refreshTokenExpiration * 1000 + 86400000;
        if (now.getTime() - session.createdAt.getTime() > maxAge) {
          this.sessions.delete(sessionId);
          const userSessions = this.userSessions.get(session.userId);
          if (userSessions) {
            userSessions.delete(sessionId);
          }
        }
      }
    }, 60000); // Run every minute
  }

  /**
   * Get cookie settings
   */
  getCookieSettings(): CookieSettings {
    return { ...this.config.cookieSettings };
  }

  /**
   * Get all active sessions for a user
   */
  getUserSessions(userId: string): AuthSession[] {
    const sessionIds = this.userSessions.get(userId);
    if (!sessionIds) return [];

    return Array.from(sessionIds)
      .map((id) => this.sessions.get(id))
      .filter((s): s is AuthSession => s !== undefined && s.state === "active");
  }

  /**
   * Get session statistics
   */
  getStats(): {
    totalSessions: number;
    activeSessions: number;
    providers: number;
    sessionsByProvider: Record<string, number>;
  } {
    let activeSessions = 0;
    const sessionsByProvider: Record<string, number> = {};

    for (const session of this.sessions.values()) {
      if (session.state === "active") {
        activeSessions++;
        sessionsByProvider[session.provider] =
          (sessionsByProvider[session.provider] || 0) + 1;
      }
    }

    return {
      totalSessions: this.sessions.size,
      activeSessions,
      providers: this.providers.size,
      sessionsByProvider,
    };
  }
}

/**
 * Create default identity from provider data
 */
export function createDefaultIdentity(
  providerData: Record<string, unknown>,
  mapping: AttributeMapping,
  tenantId: string,
  provider: AuthProvider
): UserIdentity {
  const getValue = (key: string): string =>
    (providerData[
      mapping[key as keyof AttributeMapping] as string
    ] as string) || "";

  const getArrayValue = (key: string): string[] => {
    const value =
      providerData[mapping[key as keyof AttributeMapping] as string];
    if (Array.isArray(value)) return value as string[];
    if (typeof value === "string") return value.split(",").map((s) => s.trim());
    return [];
  };

  return {
    id: getValue("id") || uuidv4(),
    externalId: getValue("id"),
    email: getValue("email"),
    emailVerified: true,
    displayName: getValue("displayName") || getValue("email"),
    firstName: getValue("firstName"),
    lastName: getValue("lastName"),
    avatar: getValue("avatar"),
    provider,
    providerId: getValue("id"),
    tenantId,
    roles: getArrayValue("roles") || [],
    permissions: [],
    groups: getArrayValue("groups") || [],
    attributes: { ...providerData },
    metadata: {
      createdAt: new Date(),
      updatedAt: new Date(),
      loginCount: 1,
      failedLoginAttempts: 0,
      mfaEnabled: false,
      mfaMethods: [],
    },
  };
}
